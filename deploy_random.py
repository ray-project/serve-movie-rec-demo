import ray
from ray import serve

from util import get_db_connection, KNearestNeighborIndex


class RandomRecommender:
    def __init__(self):
        self.db = get_db_connection()

    def __call__(self, request):
        num_returns = request.args.get("count", 5)

        # Fetch random movies from the database.
        titles_and_ids = self.db.execute(
            f"SELECT title, id FROM movies ORDER BY RANDOM() LIMIT {num_returns}"
        ).fetchall()

        return [{
            "id": movie_id,
            "title": title
        } for title, movie_id in titles_and_ids]


if __name__ == "__main__":
    # Deploy the model.
    try:
        ray.init(address="auto")
        client = serve.connect()
    except:
        raise Exception("Failed to connect to Ray Serve. Did you forget to run setup.py first?")
    client.create_backend("random:v0", RandomRecommender)
    client.create_endpoint("random", backend="random:v0", route="/rec/random")
    print("Deployed random recommender to /rec/random.")
    print("Try it out with: 'curl http://localhost:8000/rec/random'")
