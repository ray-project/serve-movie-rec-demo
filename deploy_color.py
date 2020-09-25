import ray
from ray import serve

from util import get_db_connection, KNearstNeighborIndex


class ColorRecommender:
    def __init__(self):
        self.db = get_db_connection()

        # Create index of cover image colors.
        colors = self.db.execute("SELECT id, palette_json FROM movies")
        self.color_index = KNearstNeighborIndex(colors)

    def __call__(self, request):
        # Perform KNN search for similar images.
        recommended_ids = self.color_index.search(request)

        # Let's perform some post processing.
        titles_and_ids = self.db.execute(
            f"SELECT title, id FROM movies WHERE id in ({','.join(recommended_ids)})"
        ).fetchall()

        # Wrangle the data for JSON
        return [{
            "id": movie_id,
            "title": title
        } for title, movie_id in titles_and_ids]


if __name__ == "__main__":
    # Deploy the model.
    ray.init(address="auto")
    client = serve.connect()
    client.create_backend("color:v1", ColorRecommender)
    client.create_endpoint("color", backend="color:v1", route="/rec/color")
