# Forklift Detection — Architecture

End-to-end AI forklift detection for Rhombus cameras. The Rhombus Rules Engine fires a webhook on `MOTION_CAR`, a Cloud Run service fetches frames from the camera's recorded footage, runs YOLOv8 across the event window, and — if a forklift is detected — writes seekpoints and bounding boxes back to the camera timeline. Also detects near-miss events (forklift + human proximity) and estimates forklift speed.

---

## System Flow

```
                    ┌─────────────────────────────────────────┐
                    │         Rhombus Rules Engine            │
                    │  Rule: se4ooEiTRWOq5wDvgLr8pQ           │
                    │  Trigger: DEVICE_ACTIVITY_EVENT         │
                    │  Activity: MOTION_CAR                   │
                    │  Cameras: R410 + R540 (i2M org)         │
                    └─────────────────┬───────────────────────┘
                                      │ POST /webhook (JSON)
                                      │ Two pings per event:
                                      │   ping 1 — minimal (no seekpoints) → deferred
                                      │   ping 2 — finalized (~5–30s later) → processed
                                      ▼
  ┌───────────────────────────────────────────────────────────────────────────┐
  │                  Cloud Run: forklift-detection                            │
  │          https://forklift-detection-215390028467.us-east1.run.app         │
  │                                                                           │
  │  webhook_server.py  (Flask + Gunicorn, 2 workers / 4 threads / 180s)      │
  │                                                                           │
  │  ┌─ Minimal ping (no eventUuid, no tu) ──────────────────────────────┐   │
  │  │  • Record in _PENDING dict, keyed by camera_uuid                  │   │
  │  │  • Kick off analyze_event_frames() in background ThreadPoolExecutor│  │
  │  │  • Return 200 "deferred"                                          │   │
  │  │  • _pending_sweeper thread: after 45s with no finalized followup, │   │
  │  │    process using pre-fetched frames (sweeper path)                │   │
  │  └───────────────────────────────────────────────────────────────────┘   │
  │                                                                           │
  │  ┌─ Finalized ping (has eventUuid + seekpoints) ─────────────────────┐   │
  │  │  1. parse_payload() → camera_uuid, event_uuid, timestamp_ms,      │   │
  │  │     region, seekpoint_tu, location_uuid                           │   │
  │  │                                                                   │   │
  │  │  2. Frame fetch — tu URL fast path + analyze_event_frames()       │   │
  │  │     • Download tu thumbnail if present (often already expired)    │   │
  │  │     • analyze_event_frames(): 30s window, --fill --raw            │   │
  │  │       → ~15 frames spaced ~2s apart from recorded footage         │   │
  │  │     • Uses pre-fetched future from minimal ping if available      │   │
  │  │                                                                   │   │
  │  │  3. run_detection() × N frames (full event window scan)           │   │
  │  │     • Self-crops full frames to MOTION_CAR bbox region + 15% pad  │   │
  │  │     • YOLOv8 (best.pt), threshold 0.70                            │   │
  │  │     • Collects timed_detections[] for speed estimation            │   │
  │  │     • Tracks best-confidence frame for annotations                │   │
  │  │                                                                   │   │
  │  │  4. If forklift detected:                                         │   │
  │  │     a. estimate_speed() → median mph across frame pairs           │   │
  │  │        • Lateral component: centre-point displacement in permyriad│   │
  │  │        • Radial component: bbox diagonal growth in permyriad      │   │
  │  │        • Takes max(lateral, radial) per frame pair                │   │
  │  │     b. create_seekpoint() → PURPLE "Forklift Detection"           │   │
  │  │        • Includes speed in description if available               │   │
  │  │     c. create_bounding_boxes() → short-key schema (permyriad)     │   │
  │  │     d. log_detection() → detections.csv                           │   │
  │  │     e. Speed alert: if speed > SPEED_LIMIT_MPH →                  │   │
  │  │        create_speed_alert_seekpoint() → ORANGE seekpoint          │   │
  │  │     f. Near-miss check: query MOTION_HUMAN seekpoints ±15s        │   │
  │  │        • check_near_miss(): human centre within 2000 permyriad    │   │
  │  │          of forklift bbox (skips operator-in-cab)                 │   │
  │  │        • If near-miss: create_near_miss_seekpoint() → RED         │   │
  │  │          + log_near_miss() → near_misses.csv                      │   │
  │  └───────────────────────────────────────────────────────────────────┘   │
  │                                                                           │
  │  Dashboard: GET /dashboard  polls /stats.json every 2s                    │
  │             ring buffer of last 200 events, thumbnails, speed column      │
  │  Calibration: GET /calibrate  — click two points on a live frame to       │
  │             compute permyriad/inch scale factor for speed estimation      │
  └───────────────────────────┬───────────────────────────────────────────────┘
                              │ mTLS + X-Auth-Apikey + X-Auth-Scheme:api
                              ▼
          ┌────────────────────────────────────────────────────┐
          │                  Rhombus APIs                      │
          │  api2.rhombussystems.com/api                       │
          │  mediaapi-v2.rhombussystems.com                    │
          │  *.dash-internal.rhombussystems.com  (CLI only)    │
          └────────────────────────────────────────────────────┘
```

---

## Frame Fetch Strategy

`analyze_event_frames()` is the primary production path. It shells out to the Rhombus CLI which can reach `.dash-internal.rhombussystems.com` — an endpoint Python `requests` cannot authenticate against directly.

| # | Method | Returns | Why it can fail |
|---|--------|---------|-----------------|
| 1 | Webhook `tu` URL (mTLS GET) | Pre-cropped thumbnail | Expires within seconds; 404 if webhook arrives late |
| 2 | `rhombus analyze footage` (CLI subprocess) | Full 1920×1080 frames × N | Primary path; reads durable recorded footage; 120s timeout |
| 3 | `rhombus alert thumb` (CLI subprocess) | Full frame | Only works once event is promoted to a Rhombus alert |
| 4 | Re-query `camera/getFootageSeekpointsV2` for fresh `tu` | Pre-cropped thumbnail | Returns same expired token if called too late |
| 5 | `mediaapi-v2/media/metadata/{region}/{eventUuid}.jpeg` | Full frame | Only works for promoted alerts |

`analyze_event_frames()` fetches a **30-second window** of footage using `--fill --raw`, yielding ~15 frames evenly spaced ~2s apart. These are used both for multi-frame YOLO scanning and speed trajectory estimation.

---

## Deferred Sweeper Pattern

The Rules Engine fires **two pings per event**:

**Ping 1 — Minimal** (fires immediately):
```json
{ "deviceEvents": [{ "deviceUuid": "tbp4rmdDTReKPssY2dKImQ.v0", "timestampMs": 1776967409560 }] }
```
No `eventUuid`, no seekpoints, no `tu`. Immediately:
1. Stashed in `_PENDING` dict keyed by `camera_uuid`
2. `analyze_event_frames()` launched in `ThreadPoolExecutor` (pre-fetch)
3. Returns `200 deferred`

**Ping 2 — Finalized** (~5–30s later):
```json
{
  "deviceEvents": [{
    "deviceUuid": "tbp4rmdDTReKPssY2dKImQ.v0",
    "eventUuid": "MvBivqkZQx2Gho_bxEwNkA",
    "boundingBoxes": [{ "activity": "MOTION_CAR", "left": 5266, "top": 6917, ... }],
    "seekpoints": [{ "activity": "MOTION_CAR", "tu": "/media/frame/…" }]
  }]
}
```
Pops the `_PENDING` entry, uses the pre-fetched frames future (usually done by now), proceeds with full detection pipeline.

**Sweeper fallback**: `_pending_sweeper` thread wakes every 10s and processes any event in `_PENDING` older than 45s — handles cases where the finalized ping never arrives.

---

## Speed Estimation

`estimate_speed(timed_detections, camera_uuid)` computes forklift speed in mph from the multi-frame detection trajectory:

```
For each consecutive frame pair (det1@ts1, det2@ts2):

  Lateral component:
    centre displacement in permyriad = √((cx2-cx1)² + (cy2-cy1)²)

  Radial component:
    bbox diagonal change in permyriad = |diag2 - diag1|
    (catches approach/recession that lateral misses when forklift drives toward camera)

  dist = max(lateral, radial)
  speed_mph = (dist / scale_factor / 12 / dt_sec) × 3600 / 5280

Return median of all valid per-pair speeds (0 < speed < 25 mph sanity bound)
```

**Scale factor** (`permyriad/inch`) is calibrated per camera via the `/calibrate` UI tool — click two points on a known reference object (e.g. 48" pallet width) to compute and save the factor. Persisted via `CAMERA_SCALES` environment variable.

Current calibrations:
| Camera | Model | Scale factor |
|--------|-------|-------------|
| `tbp4rmdDTReKPssY2dKImQ` | R540 | 16.741882 permyriad/inch |
| `1gKR-iBAQfmANqoQ9Nutjw` | R410 | 13.1121 permyriad/inch |

---

## Near-Miss Detection

`check_near_miss(forklift_detections, human_bboxes)` fires when a human is detected near an active forklift:

1. Query `camera/getFootageSeekpointsV2` for `MOTION_HUMAN` seekpoints in a ±15s window
2. Parse `tu` URL query params (`x/y/w/h`) → human bboxes in permyriad
3. For each (forklift bbox, human bbox) pair:
   - **Skip** if human centre falls inside the forklift bbox (operator in cab)
   - **Alert** if human centre falls within `NEAR_MISS_THRESHOLD` permyriad (default 2000) of the expanded forklift zone
4. On near-miss: write RED seekpoint + append to `near_misses.csv`

---

## Seekpoint Colors

| Color | Meaning | Trigger |
|-------|---------|---------|
| `PURPLE` | Forklift detected | Any confirmed detection |
| `ORANGE` | Speed alert | Speed > `SPEED_LIMIT_MPH` (default 10 mph) |
| `RED` | Near-miss alert | Human within 2000 permyriad of forklift |

---

## Rhombus API Endpoints

All calls are `POST`. Required headers: `X-Auth-Apikey` + `X-Auth-Scheme: api`.

| Endpoint | Purpose | Called from |
|----------|---------|-------------|
| `camera/createCustomFootageSeekpoints` | Write seekpoints to timeline (purple/orange/red) | `create_seekpoint()`, `create_near_miss_seekpoint()`, `create_speed_alert_seekpoint()` |
| `camera/createFootageBoundingBoxes` | Write forklift bbox overlay (short-key schema) | `create_bounding_boxes()` |
| `camera/getFootageSeekpointsV2` | Re-query for fresh `tu` URL; query MOTION_HUMAN bboxes | `refresh_tu_url()`, `get_human_bboxes()` |
| `GET mediaapi-v2/media/frame/{cam}/{ts}/thumb.jpeg?x=…` | Download webhook thumbnail (strategy 1) | `download_media()` |
| `GET mediaapi-v2/media/metadata/{region}/{uuid}.jpeg` | Fallback thumbnail by eventUuid (strategy 5) | `download_media()` |
| CLI `rhombus analyze footage` → `video/getExactFrameUri` | Pull frames from recorded footage (strategy 2) | `analyze_event_frames()`, `download_via_analyze()` |
| CLI `rhombus alert thumb` | Download alert thumbnail (strategy 3) | `download_via_cli()` |

> **Note:** The CLI command `get-camera-footage-seekpoints-v2` maps to the API path `camera/getFootageSeekpointsV2` (no "Camera" prefix). Use `rhombus --verbose` to see actual paths when debugging.

---

## Self-Crop Pipeline

When a full sensor frame is returned (strategies 2, 3, 5), YOLO would see a tiny vehicle in a 1920×1080 image. The server crops to the motion region first:

```
Full sensor frame (1920×1080)
         │
         │  extract_motion_bbox(payload)
         │  → union of all MOTION_CAR boundingBoxes in permyriad (0–10000)
         │
         ▼
Padded crop region (+15% on each side)
         │
         │  PIL crop → sub-image (e.g. 265×378)
         ▼
YOLOv8 inference on crop
         │
         │  map detected pixel coords back to full-frame space
         ▼
Detections in full-frame coordinates → normalized to permyriad for Rhombus API
```

---

## Source Files

| File | Role |
|------|------|
| [`webhook_server.py`](webhook_server.py) | Production server — Flask routes, frame fetch, YOLO, speed, near-miss, dashboard, calibration |
| [`pipeline.py`](pipeline.py) | Legacy local poller — polls `rhombus alert recent`, not used in prod |
| [`forklift_detector.py`](forklift_detector.py) | Dev tool — standalone CLI for testing YOLO on a single image |
| [`local_test.py`](local_test.py) | Replay tool — pulls the most recent real webhook from Cloud Run logs and replays it locally |
| [`best.pt`](best.pt) | YOLOv8s forklift fine-tune weights (committed to repo) |

---

## Build & Deploy

```
local webhook_server.py
  │
  ├── git commit → Trevor-42/forklift-detection (PR flow, main branch protected)
  │
  └── gcloud run deploy forklift-detection --source . --region us-east1 \
        --project forklift-detection-i2m --min-instances 1
          │
          ├── Cloud Build: Dockerfile
          │    python:3.11-slim
          │    pip: ultralytics flask requests gunicorn Pillow
          │    curl-install rhombus CLI v0.16.1
          │    COPY webhook_server.py best.pt
          │
          └── Cloud Run revision N+1 → 100% traffic
               workers: 2  threads: 4  timeout: 180s
               min-instances: 1 (no cold starts)
               memory: 2Gi
               Secrets mounted: client-crt, client-key (mTLS volumes)
               Env secret: RHOMBUS_API_KEY (Secret Manager)
               Env: CAMERA_SCALES, CONFIDENCE_THRESHOLD, SPEED_LIMIT_MPH
```

---

## Configuration

| Item | Value |
|------|-------|
| YOLO weights | `best.pt` (yolov8s forklift fine-tune) |
| Confidence threshold | `0.70` (env: `CONFIDENCE_THRESHOLD`) |
| Near-miss threshold | `2000` permyriad proximity (env: `NEAR_MISS_THRESHOLD`) |
| Speed limit | `10` mph before ORANGE alert (env: `SPEED_LIMIT_MPH`) |
| Frame scan window | `30s` (~15 frames at ~2s spacing) |
| Self-crop padding | 15% on each side of the motion bbox |
| Seekpoint colors | PURPLE = detection, ORANGE = speed alert, RED = near-miss |
| Rhombus org | `i2M` (`QwDXQrNZRzCtXGV_D5bUgQ`) |
| Cameras in scope | R540-A25K0330 (`tbp4rmdDTReKPssY2dKImQ`), R410-A25K0532 (`1gKR-iBAQfmANqoQ9Nutjw`) |
| Cloud Run region | `us-east1` (OCI Ashburn VA — close to Rhombus dash servers) |
| GCP project | `forklift-detection-i2m` |
| Current revision | `forklift-detection-00014-542` |

---

## Known Quirks

| Issue | Symptom | Status |
|-------|---------|--------|
| Bounding boxes don't render in UI | `createFootageBoundingBoxes` stores for analytics but the Rhombus player doesn't overlay | Accepted as-is — seekpoint renders correctly |
| Webhook `tu` URL consistently expires | Every finalized event falls through to `rhombus analyze` | Working — strategy 2 handles it reliably |
| `deviceUuid` has `.v0` suffix in Rules payload | API rejects UUID | `split(".")[0]` in `parse_payload()` |
| Long-form bbox keys | API returned 200 but nothing persisted | Use short keys: `a/l/t/r/b/c/m/ts/objectId` |
| Speed underestimates true radial movement | Scale factor calibrated at fixed depth; bbox diagonal proxy is approximate | Partially mitigated in PR #31 (max of lateral + radial); true fix requires focal-length calibration |
| `getFootageSeekpointsV2` re-issues expired token | Refreshed `tu` URL still 404s if called after thumbnail cache eviction | Strategy 2 (`getExactFrameUri` via CLI) bypasses this entirely |
