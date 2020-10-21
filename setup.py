import ray
from ray import serve

# Connect to the background Ray cluster.
print("Connecting to Ray.")
try:
    ray.init(address="auto")
except Exception as e:
    raise Exception("Did you forget to call 'ray start --head'?")

# Create a Serve instance.
print("Starting Serve instance.")
serve.start(detached=True)
