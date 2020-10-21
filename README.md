A simple demo of how to use Ray Serve to build a recommendation API.

First, download the database file that contains movie information:

```wget https://ray-serve-blog.s3-us-west-2.amazonaws.com/composition-demo.sqlite3```

Then, install Python dependencies:

```pip install -r requirements.txt```

Now you should be ready to run.

```
# Start ray.
> ray start --head

# Create a detached Serve instance on the Ray cluster.
> python setup.py

# Deploy a simple "model" that returns random results.
> python deploy_random.py
> curl http://localhost:8000/rec/random

# Deploy a more advanced NLP-based model that returns results based on plot similarity.
> python deploy_plot.py
> curl "http://localhost:8000/rec/plot?liked_id=322259"

# Deploy an ensemble model that dynamically selects between the two existing models.
> python deploy_ensemble.py
> curl "http://localhost:8000/rec/ensemble?liked_id=322259"

# Clean up.
> ray stop
```
