import pickle
from util import KNearstNeighborIndex, get_db_connection, LRMovieRanker
import ray
from ray import serve


class PlotRecommender:
    def __init__(self, lr_model):
        self.db = get_db_connection()

        bert_vectors = self.db.execute(
            "SELECT id, plot_vector_json FROM movies")
        self.index = KNearstNeighborIndex(bert_vectors)

        self.lr_model = LRMovieRanker(lr_model, features=self.index.id_to_arr)

    def __call__(self, request):
        # Find k nearest movies with simliar plots.
        recommended_movies = self.index.search(request)

        # Rank them using logistic regression.
        ranked_ids = self.lr_model.rank_movies(recommended_movies)

        # Let's perform some post processing.
        titles_and_ids = self.db.execute(
            f"SELECT title, id FROM movies WHERE id in ({','.join(ranked_ids)})"
        ).fetchall()

        # Wrangle the data for JSON
        return [{
            "id": movie_id,
            "title": title
        } for title, movie_id in titles_and_ids]


if __name__ == "__main__":
    # Deploy the plot model.
    ray.init(address="auto")
    client = serve.connect()

    model_weights = get_db_connection().execute(
        "SELECT weights FROM models WHERE key='ranking/lr:base'").fetchone()[0]
    base_lr_model = pickle.loads(model_weights)
    client.create_backend("plot:v0", PlotRecommender, base_lr_model)
    client.create_endpoint("plot", backend="plot:v0", route="/rec/plot")
