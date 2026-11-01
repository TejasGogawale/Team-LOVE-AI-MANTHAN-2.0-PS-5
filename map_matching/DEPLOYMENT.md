# Deploy to Render

1. Push this project to a GitHub or GitLab repository. Include `VIT_IMAGES/VIT_IMAGES/*.jpg`, `templates/`, Python source files, `requirements.txt`, and `render.yaml`. Exclude `.packages`, virtual environments, and secrets. The dataset is approximately 2.6 MB.
2. In Render, choose **New → Blueprint**, connect the repository, and deploy its `render.yaml`. It configures a free Python web service, Gunicorn, one process, two HTTP threads, and `/health` readiness checks.
3. Set `DASHBOARD_ORIGINS` under the service's **Environment** settings if your dashboard runs on a different origin. Example: `https://dashboard.example.com,http://localhost:3000`. Save and redeploy.
4. Open the generated `.onrender.com` URL. Check `/api/v1/health`, then upload a query or try the random demo. Integrate using API.md.

Manual Web Service configuration:

```text
Runtime: Python
Instance: Free
Build: pip install -r requirements.txt
Start: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --worker-class gthread --threads 2 --timeout 180 --access-logfile - --error-logfile -
Health check: /health
PYTHON_VERSION: 3.12.8
USE_DINO: 0
OMP_NUM_THREADS: 1
OPENBLAS_NUM_THREADS: 1
```

The app loads and extracts dataset features on startup. Keep the dataset in the repository so it is available after restarts. Matching uses a single request slot; a second matching request receives 503 with Retry-After rather than allocating another set of query images. Queries are limited to 4 million pixels to reduce memory pressure.

Render's free instances currently have 512 MB RAM / 0.1 CPU and sleep after 15 idle minutes. Startup and matching will be slower than local execution. DINO is disabled in this deployment. Free-tier memory, latency and Gunicorn startup still require verification on Render; this workspace is Windows, where Gunicorn does not run.

References: https://render.com/docs/deploy-flask, https://render.com/docs/blueprint-spec, https://render.com/docs/free.
