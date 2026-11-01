# Map Patch Finder

Render deployment: see [DEPLOYMENT.md](DEPLOYMENT.md) and `render.yaml`. Dashboard communication: see [API.md](API.md) for the versioned upload API, CORS setup, coordinates, polygons and matching images.

Run `./launch.ps1` from PowerShell, then open http://127.0.0.1:5000. Upload a query image or generate a random cropped, rotated, scaled patch. Results show the source image, localization polygon, a source region warped back into query orientation, and RANSAC inlier correspondence lines. Images can be downloaded.

The default mode searches all 20 images using SIFT/FLANN/RANSAC. This preserves the supplied pipeline's geometric decision stage without needing model downloads. To enable the supplied DINOv2 multi-rotation/FAISS shortlist, install requirements-dino.txt and set USE_DINO=1 before launching. DINO model weights require internet on first use. The optional DINO mode is not covered by the default evaluation.

Dataset discovery is recursive and accepts coordinate filenames. Homographies and polygons use original image coordinates. White corners in rotated demo patches are padding; the full query footprint can extend beyond the original crop. The aligned result shows matching pixels in query orientation, including surrounding source pixels under that padding.

Run evaluate.py to regenerate 60 seeded synthetic queries (three per image), compare retrieval with the known source, measure mean polygon corner error, and reject a blank query. Outputs are in results/evaluation.json and results/*.png. These tests use crops of the same source images, not independent real-world queries. Overlapping dataset tiles may both contain the queried region; exact source correctness is reported separately from visual overlap.

For another Python installation: install requirements.txt, then run `python app.py`. Local workspace dependencies are automatically loaded from .packages. App binds only to localhost.
