# Forklift Detection — Project Memory

YOLOv8 forklift detection pipeline triggered by Rhombus Rules Engine webhooks on
the i2M org's cameras. Detections are written back to Rhombus as seekpoints +
bounding boxes.

## Deployment

- **Cloud Run service:** `forklift-detection`
  - Project: `forklift-detection-i2m`
  - Region: `us-central1`
  - URL: `https://forklift-detection-215390028467.us-central1.run.app`
  - Deploy: `gcloud run deploy forklift-detection --source . --region us-central1 --project forklift-detection-i2m`
- **Runtime:** Flask + gunicorn (`--workers 1 --threads 4 --timeout 120`)
- **mTLS:** `/run/secrets/crt/client-crt` + `/run/secrets/key/client-key` in prod; `~/.rhombus/certs/i2M/` locally
- **Secrets:** Cloud Run has `RHOMBUS_API_KEY` mounted. Locally, `.env` (gitignored) or `~/.rhombus/credentials` profile `i2M`.

## GitHub

- Repo: `Trevor-42/forklift-detection`
- Main branch protected via PR flow

## Key endpoints

- `POST /webhook` — entrypoint for Rhombus Rules Engine
- `GET /dashboard` — in-process real-time dashboard (ring buffer, polls `/stats.json` every 2s)
- `GET /stats.json` — aggregate stats + last 50 events

## Rhombus API endpoints used

| CLI command name | Actual API path |
|---|---|
| `camera/createCustomFootageSeekpoints` | same |
| `camera/createFootageBoundingBoxes` | same |
| `camera/get-camera-footage-seekpoints-v2` | **`camera/getFootageSeekpointsV2`** (no "Camera" prefix — CLI name differs from API path) |
| `video/getExactFrameUri` | same — pulls exact frame from recorded footage at a given `timestampMs` |

All calls are POST. Required headers: `X-Auth-Apikey` + `X-Auth-Scheme: api`.

## Webhook payload shape (Rules Engine)

Rules Engine fires **two pings per event**:
1. **Minimal first-ping** — no `eventUuid`, no `seekpoints`, no `tu` URLs. Nothing to fetch. Recorded as `status="deferred"` on the dashboard.
2. **Finalized payload** ~5–30s later — has `deviceEvents[0].eventUuid`, `seekpoints[]`, and `thumbnailLocation.region`. This is the one we act on.

Seekpoint short-form schema: `ts / a / l / t / r / b / c / m / objectId` where `a` is the activity type (`MOTION_CAR` for forklifts that look like cars).

## Frame-fetch strategy (in `get_frame()`)

Current order after PR #13:
1. Webhook-delivered `tu` URL — fastest when fresh (no extra API call)
2. **`video/getExactFrameUri`** — robust; reads from recorded footage, independent of thumb cache
3. `rhombus alert thumb` CLI — only works once event is promoted to an alert
4. `refresh_tu_url()` — re-query `getFootageSeekpointsV2` for a signed URL
5. Metadata URL — only if event has been promoted to an alert

### Why we needed getExactFrameUri

The webhook's `tu` URL carries a short-lived signing token (`?a=916`-style) that
expires within seconds of event generation. If the webhook lands >~10s late, the
underlying thumbnail bytes are evicted from Rhombus's in-memory cache and the URL
404s. `getFootageSeekpointsV2` re-issues the **same** expired token, so it
doesn't recover. `video/getExactFrameUri` reads from disk (recorded footage)
and works for any ts within retention.

## Known quirks / accepted limitations

- **Bounding boxes don't render on the Rhombus timeline UI.** `createFootageBoundingBoxes` stores data for search/analytics but the player doesn't overlay it. The seekpoint is what renders. Accepted as-is.
- CLI command `get-camera-footage-seekpoints-v2` maps to a different API path (see table above). Use `rhombus --verbose` to see actual paths when debugging auth failures.

## Secrets hygiene

- `.env` is gitignored; `.env.example` is committed as a template
- `~/.rhombus/credentials` (profile `i2M`) is a fallback source
- Three-tier loader in `local_test.py`: env var → `.env` → `~/.rhombus/credentials`
- **Outstanding action:** rotate leaked key `lZU-w_6JQ9eE7CaNJ9UZXA` in Rhombus console and update the Cloud Run secret. A scheduled reminder is set for 2026-04-24 09:00 PT (task `rotate-rhombus-api-key`).

## Local testing

```bash
# Full pipeline against the most recent real webhook from logs:
python local_test.py --auto

# Dry-run (fetch + detect but don't write back):
python local_test.py --auto --dry-run

# Skip fetch; run detection on a local image (useful when webhook thumb has expired):
python local_test.py --frame /tmp/frame.jpeg --camera <uuid> --ts <ms> --dry-run
```

Cert override in `local_test.py` points to `~/.rhombus/certs/i2M/`; requires
`rhombus login --profile i2M` beforehand.

## Model

- `best.pt` — YOLOv8 forklift detector, committed to the repo
- Confidence threshold: `CONFIDENCE_THRESHOLD` env var, default `0.70`
- Labels of interest: `forklift`

## PR history (relevant)

- #7  Followup: seekpoint shape + docs
- #8  Real-time dashboard
- #10 Move API key out of source into gitignored `.env`
- #11 Defer minimal Rules Engine payloads instead of erroring
- #12 `refresh_tu_url` fallback + fix `getFootageSeekpointsV2` endpoint name
- #13 `video/getExactFrameUri` as primary robust fetch path
