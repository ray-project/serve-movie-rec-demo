import ray
from ray import serve

# Connecting to background Ray cluster
ray.init(address="auto")

# Create a Serve instance
serve.start(detached=True)