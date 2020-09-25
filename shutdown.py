import ray
from ray import serve

ray.init(address="auto")

# Shutdown Ray Serve
try:
    client = serve.connect()
    client.shutdown()
except:
    pass

# Shutdown impression database actor
try:
    actor = ray.get_actor("impressions")
    ray.kill(actor)
except:
    pass

# Shutdown periodic trainer
try:
    actor = ray.get_actor("periodic-trainer")
    ray.kill(actor)
except:
    pass