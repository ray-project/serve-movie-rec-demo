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
> curl -X GET http://localhost:8000/rec/random
```
