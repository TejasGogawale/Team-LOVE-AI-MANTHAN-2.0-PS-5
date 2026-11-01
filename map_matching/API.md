# Dashboard integration API

Base URL: `https://YOUR-SERVICE.onrender.com` (local: `http://127.0.0.1:5000`).
All responses are JSON. The API is public and has no authentication. Browser dashboards on other origins must be listed in `DASHBOARD_ORIGINS` in Render's Environment settings. Use exact origins such as `https://dashboard.example.com`, comma-separated, with no trailing slash. CORS controls browser access, not authentication. No cookies or credentials are required.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Status, API version and dataset count |
| GET | `/api/v1/dataset` | Image filenames and their latitude/longitude |
| POST | `/api/v1/localize` | Retrieve and localize an uploaded query |

Send a multipart form with image field **query**. Maximum request size: 16 MiB; maximum decoded image: 4 million pixels. Add `?include_images=false` for a smaller metadata-only response. The default includes PNG data URLs in `query`, `retrieved`, `localized`, `aligned`, and `matches_image`, which can be assigned directly to `<img src>`. Uploads and results are processed in memory, with no persistent history.

```javascript
async function locatePatch(file) {
  const form = new FormData();
  form.append('query', file);
  const response = await fetch(
    'https://YOUR-SERVICE.onrender.com/api/v1/localize',
    { method: 'POST', body: form }
  );
  const result = await response.json();
  if (!response.ok) throw new Error(result.error);
  if (!result.accepted) {
    console.log(result.reason); // Valid image, but insufficient match evidence
    return result;
  }
  console.log(result.image, result.coordinates.latitude, result.coordinates.longitude);
  console.log(result.polygon); // Four [x, y] corners in source-image pixels
  // dashboardImage.src = result.localized;
  // matchingCrop.src = result.aligned;
  return result;
}
```

Do not set Content-Type manually when sending FormData: the browser supplies the multipart boundary.

```bash
curl -X POST "https://YOUR-SERVICE.onrender.com/api/v1/localize?include_images=false" \
  -F "query=@query.png"
```

Successful matches include `accepted: true`, `image`, `coordinates`, `polygon`, `inliers`, median reprojection `error` (pixels), `ranking`, `mode`, `processing_ms`, `request_id`, and `api_version`. Rankings contain candidate filenames, inlier counts, ratios, error and coverage. Coordinates come from the **source image filename**, not a GPS estimate of the localized patch. Polygon coordinates refer to the original source image, not a resized display. No match returns HTTP 200 with `accepted: false`, `reason`, and `ranking`; image, coordinates and polygon are absent.

| Status | Meaning |
|---|---|
| 200 | Request completed; inspect `accepted` |
| 400 | Missing, invalid, oversized-dimension image or invalid query option |
| 413 | Request body exceeds 16 MiB |
| 503 | Another query is processing; retry after `Retry-After` seconds |
| 500 | Unexpected server failure |

Errors contain `error` and `code`. This service serializes matching to keep free-tier memory use predictable. Health checks remain available through a second HTTP thread. A dashboard should disable duplicate submissions, show progress, and allow for Render cold starts. This is a synchronous request/response API; there are no WebSockets, background jobs or saved result endpoints.
