import os
import json
from collections import defaultdict
from itertools import cycle
from sqlite3 import connect

import numpy as np
import pandas as pd
import faiss

import ray
from ray.experimental.metrics import Gauge


def get_db_connection():
    path = "./composition-demo.sqlite3"
    if not os.path.exists(path):
        raise Exception("""
It seems like the database file doesn't exist. Did you forget
to download it? You can try running
    make download-data
or
    wget https://ray-serve-blog.s3-us-west-2.amazonaws.com/composition-demo.sqlite3
    """)

    return connect(path)


@ray.remote(num_cpus=0)
class ImpressionStore:
    def __init__(self):
        # session_key -> {id: model}
        self.impressions = defaultdict(dict)
        # session_key -> number of impression recorded
        self.num_impressions = defaultdict(lambda: 0)

        # session_key -> {model_name: int}
        self.session_liked_model_count = defaultdict(
            lambda: defaultdict(lambda: 0))

        # model -> cliked_id_set
        self.model_clicked_ids = defaultdict(set)
        # model -> shown_id_set
        self.model_shown_ids = defaultdict(set)

        self.metric = Gauge(
            "impression_store_click_rate",
            "The click through rate of each model in impression store",
            "percent", ["model"])

    def _refresh_ctr(self):
        model_counter = defaultdict(lambda: 0)
        for liked_count in self.session_liked_model_count.values():
            for name, count in liked_count.items():
                model_counter[name] += count

        for model, clicks in model_counter.items():
            rate = clicks / self.num_impressions[model]
            self.metric.record(rate, {"model": model})

    def _record_feedback(self, session_key, liked_id):
        # Record feedback from the user
        src_model = self.impressions[session_key].get(liked_id)
        # Can't find this impression source
        if src_model is None:
            return
        self.session_liked_model_count[session_key][src_model] += 1
        self.model_clicked_ids[src_model].add(liked_id)

        self._refresh_ctr()

    def record_impressions(self, session_key, impressions):
        # Record impressions we are sending out
        for model, ids in impressions.items():
            for movie_payload in ids:
                movie_id = movie_payload["id"]
                self.impressions[session_key][movie_id] = model
                self.model_shown_ids[model].add(movie_id)
            self.num_impressions[model] += 1

        self._refresh_ctr()

    def model_distribution(self, session_key, liked_id):
        if session_key == "":
            return {}
        self._record_feedback(session_key, liked_id)
        return self.session_liked_model_count[session_key]

    def count_for_model(self, model):
        count = 0
        for model_dict in self.session_liked_model_count.values():
            if model in model_dict:
                count += model_dict[model]
        return count

    def get_model_clicks(self, model):
        positive = self.model_clicked_ids[model]
        negative = self.model_shown_ids[model] - positive
        return pd.DataFrame({
            "id": list(positive) + list(negative),
            "clicked": [1] * len(positive) + [0] * len(negative)
        })


def choose_ensemble_results(model_distribution, model_results):
    # Normalize dist
    if len(model_distribution) != 2:
        default_dist = {model: 1 for model in ["color", "plot"]}
        for name, count in model_distribution.items():
            default_dist[name] += count
    else:
        default_dist = model_distribution
    total_weights = sum(default_dist.values())
    normalized_distribution = {
        k: v / total_weights
        for k, v in default_dist.items()
    }

    # Generate num returns
    chosen = []
    impressions = defaultdict(list)
    dominant_group = max(
        list(normalized_distribution.keys()),
        key=lambda k: normalized_distribution[k])
    sorted_group = list(
        sorted(
            normalized_distribution.keys(),
            key=lambda k: -normalized_distribution[k]))
    if normalized_distribution[sorted_group[0]] > normalized_distribution[sorted_group[1]]:
        sorted_group = [dominant_group] + sorted_group

    # Rank based on weights
    groups = cycle(sorted_group)
    while len(chosen) <= 10:
        model = next(groups)
        preds = model_results[model]

        if len(preds) == 0:
            if model == dominant_group:
                break
            else:
                continue

        movie_id = preds.pop(0)

        if movie_id not in chosen:
            impressions[model].append(movie_id)
            chosen.append(movie_id)

    return normalized_distribution, impressions, chosen


class LRMovieRanker:
    def __init__(self, lr_model, features):
        self.lr_model = lr_model
        self.features = features

    def rank_movies(self, recommended_movies):
        vectors = np.array([self.features[i] for i in recommended_movies])
        ranks = self.lr_model.predict_proba(vectors)[:, 1].flatten()
        high_to_low_idx = np.argsort(ranks).tolist()[::-1]
        return [recommended_movies[i] for i in high_to_low_idx]


class KNearstNeighborIndex:
    def __init__(self, db_cursor):
        # Query all the cover image palette
        self.id_to_arr = {
            row[0]: np.array(json.loads(row[1])).flatten()
            for row in db_cursor
        }

        vector_length = len(next(iter(self.id_to_arr.values())))
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(vector_length))

        # Build the index
        arr = np.stack(list(self.id_to_arr.values())).astype('float32')
        ids = np.array(list(self.id_to_arr.keys())).astype('int')
        self.index.add_with_ids(arr, ids)

    def search(self, request):
        liked_id = request.args["liked_id"]
        num_returns = int(request.args.get("count", 6))

        # Perform nearest neighbor search
        source_color = self.id_to_arr[liked_id]
        source_color = np.expand_dims(source_color, 0).astype('float32')
        _, ids = self.index.search(source_color, num_returns)
        neighbors = ids.flatten().tolist()

        return [str(n) for n in neighbors]
