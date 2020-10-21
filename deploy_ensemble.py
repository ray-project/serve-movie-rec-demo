import uuid

import ray
from ray import serve

from util import ImpressionStore, choose_ensemble_results


class ComposedModel:
    def __init__(self):
        # Get handles to the two underlying models.
        client = serve.connect()
        self.random_handle = client.get_handle("random")
        self.plot_handle = client.get_handle("plot")

        # Store user click data in a detached actor.
        self.impressions = ImpressionStore.options(
            lifetime="detached", name="impressions").remote()

    async def __call__(self, request):
        # In reality, we'd want to use something like a session key to
        # differentiate between users. Here, we'll always use the same one
        # for simplicity.
        # session_key = request.args.get("session_key", str(uuid.uuid4()))
        session_key = "abc123"
        liked_id = request.args["liked_id"]

        # Call the two underlying models and get their predictions.
        results = {
            "random": await self.random_handle.remote(liked_id=liked_id),
            "plot": await self.plot_handle.remote(liked_id=liked_id),
        }

        # Get the current model distribution.
        model_distribution = await self.impressions.model_distribution.remote(
            session_key, request.args["liked_id"])

        # Select which results to send to the user based on their clicks.
        distribution, impressions, chosen = choose_ensemble_results(
            model_distribution, results)

        # Record this click and these recommendations.
        await self.impressions.record_impressions.remote(
            session_key, impressions)

        return {
            "dist": distribution,
            "recs": chosen,
        }


if __name__ == "__main__":
    # Deploy the ensemble endpoint.
    try:
        ray.init(address="auto")
        client = serve.connect()
    except:
        raise Exception("Failed to connect to Ray Serve. Did you forget to run setup.py first?")
    client.create_backend("ensemble:v0", ComposedModel)
    client.create_endpoint(
        "ensemble", backend="ensemble:v0", route="/rec/ensemble")
    print("Deployed ensemble recommender to /rec/ensemble.")
    print("Try it out with: 'curl \"http://localhost:8000/rec/ensemble?liked_id=322259\"'")
