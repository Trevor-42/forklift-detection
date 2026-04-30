# Forklift Detection — Project Memory

YOLOv8 forklift detection pipeline triggered by Rhombus Rules Engine webhooks on
the i2M org's cameras. Detections are written back to Rhombus as seekpoints +
bounding boxes. Also detects near-miss events (forklift + human proximity) and
estimates forklift speed across a multi-frame window.

## Deployment

- **Cloud Run service:** `forklift-detection`
  - Project: `forklift-detection-i2m`
  - Region: `us-east1` (OCI Ashburn VA — close to Rhombus dash servers)
  - URL: `https://forklift-detection-215390028467.us-east1.run.app`
  - Current revision: `forklift-detection-00014-542`
  - Deploy: `gcloud run deploy forklift-detection --source . --region us-east1 --project forklift-detection-i2m --min-instances 1`
- **Runtime:** Flask + gunicorn (`--workers 2 --threads 4 --timeout 180`)
- **mTLS:** `/run/secrets/crt/client-crt` + `/run/secrets/key/client-key` in prod; `~/.rhombus/certs/i2M/` locally
- **Secrets:** `RHOMBUS_API_KEY` as Cloud Run secret env var (Secret Manager: `rhombus-api-key`). Locally, `.env` (gitignored) or `~/.rhombus/credentials` profile `i2M`.

## GitHub

- Repo: `Trevor-42/forklift-detection`
- Main branch protected via PR flow

## Key endpoints

- `POST /webhook` — entrypoint for Rhombus Rules Engine
- `GET /dashboard` — real-time dashboard, polls `/stats.json` every 2s, ring buffer of 200 events
- `GET /stats.json` — aggregate stats + last 50 events (includes speed_alerts count)
- `GET /calibrate` — camera scale calibration UI (click two points on reference object)
- `GET /calibrate/frame?camera=<uuid>` — fetches a live frame for calibration

## Rhombus Rule

- Rule UUID: `se4ooEiTRWOq5wDvgLr8pQ` (name: "Forklift Vehicle Detection")
- Trigger: `MOTION_CAR` on both cameras
- Cameras in scope:
  - R540-A25K0330: `tbp4rmdDTReKPssY2dKImQ`
  - R410-A25K0532: `1gKR-iBAQfmANqoQ9Nutjw`
- Webhook URL in rule's `webhookActions` (not org-level integration)
- To update: `rhombus --profile i2M rules update-rule --rule-update '{...}'`

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
1. **Minimal first-ping** — no `eventUuid`, no `seekpoints`, no `tu` URLs. Stashed in `_PENDING`, frames pre-fetched in background. Returns 200 `deferred`.
2. **Finalized payload** ~5–30s later — has `deviceEvents[0].eventUuid`, `seekpoints[]`, and bounding boxes. Full pipeline runs here using pre-fetched frames.

Seekpoint short-form schema: `ts / a / l / t / r / b / c / m / objectId` where `a` is the activity type (`MOTION_CAR` for forklifts that look like cars).

## Frame-fetch strategy

Primary production path: `analyze_event_frames()` — shells out to `rhombus analyze footage` CLI with `--fill --raw` over a **30-second window**, yielding ~15 frames spaced ~2s apart. Subprocess timeout: 120s.

Full fallback chain in `get_frame()`:
1. Webhook `tu` URL — fastest when fresh, but expires within seconds
2. `rhombus analyze footage` (CLI subprocess) → `video/getExactFrameUri` — robust, reads from recorded footage
3. `rhombus alert thumb` CLI — only works once event is promoted to an alert
4. `refresh_tu_url()` → `getFootageSeekpointsV2` — re-issues same expired token, rarely works
5. Metadata URL — only for promoted alerts

The webhook `tu` URL carries a short-lived token that expires within seconds. `getFootageSeekpointsV2` re-issues the same expired token. `video/getExactFrameUri` reads from disk and always works within retention.

## Speed estimation

`estimate_speed(timed_detections, camera_uuid)` — median mph across all consecutive frame pairs.

Two components, take max per pair:
- **Lateral**: centre-point displacement in permyriad → inches via scale factor
- **Radial**: bbox diagonal change in permyriad → inches via scale factor (proxy for approach/recession)

Scale factor calibrated via `/calibrate` UI using a known reference object (48" pallet):

| Camera | Model | Scale factor |
|--------|-------|-------------|
| `tbp4rmdDTReKPssY2dKImQ` | R540-A25K0330 | **16.741882** permyriad/inch |
| `1gKR-iBAQfmANqoQ9Nutjw` | R410-A25K0532 | **13.1121** permyriad/inch |

Persisted via `CAMERA_SCALES` env var: `tbp4rmdDTReKPssY2dKImQ:16.741882,1gKR-iBAQfmANqoQ9Nutjw:13.1121`

Speed limit: `SPEED_LIMIT_MPH` env var (default 10 mph) — triggers ORANGE seekpoint when exceeded.

## Near-miss detection

Queries `MOTION_HUMAN` seekpoints ±15s around a forklift event. For each human bbox:
- Skip if human centre falls inside forklift bbox (operator in cab)
- Alert if human centre within `NEAR_MISS_THRESHOLD` permyriad (default 2000) of forklift zone

Near-miss → RED seekpoint + `near_misses.csv` entry.

## Seekpoint colors

| Color | Meaning |
|-------|---------|
| PURPLE | Forklift detected |
| ORANGE | Speed > SPEED_LIMIT_MPH |
| RED | Near-miss (human within 2000 permyriad of forklift) |

## Known quirks / accepted limitations

- **Bounding boxes don't render on the Rhombus timeline UI.** Accepted as-is — seekpoint renders.
- **Speed radial component is approximate.** Scale factor is depth-calibrated at a fixed distance; radial estimation via bbox diagonal is directionally correct but not exact. True fix requires focal-length calibration.
- CLI `get-camera-footage-seekpoints-v2` maps to API path `camera/getFootageSeekpointsV2`. Use `rhombus --verbose` to see actual paths when debugging.
- **Webhook URL is embedded in the Rule's `webhookActions`**, not the org-level webhook integration. Changing it requires `rhombus rules update-rule`.

## Secrets hygiene

- `.env` is gitignored; `.env.example` is committed as a template
- `~/.rhombus/credentials` (profile `i2M`) is a fallback source for local dev
- Three-tier loader in `local_test.py`: env var → `.env` → `~/.rhombus/credentials`
- Cloud Run secrets: `rhombus-api-key`, `rhombus-i2m-client-crt`, `rhombus-i2m-client-key` in Secret Manager
- SA `215390028467-compute@developer.gserviceaccount.com` needs `roles/secretmanager.secretAccessor` on all three

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

- #7  Seekpoint shape + docs
- #8  Real-time dashboard
- #10 Move API key out of source into gitignored `.env`
- #11 Defer minimal Rules Engine payloads instead of erroring
- #12 `refresh_tu_url` fallback + fix `getFootageSeekpointsV2` endpoint name
- #13 `video/getExactFrameUri` as primary robust fetch path
- #27 Camera calibration tool (`/calibrate` UI)
- #28 Code quality: temp file cleanup, thread safety (`_CSV_LOCK`), security fixes
- #29 `/calibrate/frame` plain-text errors + 60/120/180s lookback cascade
- #30 Speed estimation + near-miss detection + dashboard columns
- #31 Speed: add radial bbox component + max(lateral,radial); frame window 20→30s; subprocess timeout 90→120s; gunicorn timeout 120→180s
