# Forklift Detection — Architecture

End-to-end AI forklift detection for Rhombus cameras. The Rhombus Rules Engine fires a webhook on `MOTION_CAR`, a Cloud Run service fetches a frame from the camera's recorded footage, runs YOLOv8, and — if a forklift is detected — writes a custom seekpoint and bounding box back to the camera timeline.

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
  ┌───────────────────────────────────────────────────────────────────────┐
  │                  Cloud Run: forklift-detection                        │
  │          https://forklift-detection-…run.app/webhook                  │
  │                                                                       │
  │  webhook_server.py  (Flask + Gunicorn, 1 worker / 4 threads)          │
  │                                                                       │
  │  1. parse_payload()                                                   │
  │     → (camera_uuid, event_uuid, timestamp_ms,                        │
  │        region, seekpoint_tu, location_uuid)                           │
  │     • strips .v0 suffix from deviceUuid                               │
  │     • prefers MOTION_CAR seekpoint for tu URL                         │
  │     • returns early (deferred) if no event_uuid and no tu             │
  │                                                                       │
  │  2. get_frame()  ← FRAME FETCH (5 strategies, first-wins)             │
  │     a. webhook tu URL     — mTLS GET, fastest when fresh              │
  │        ↳ returns pre_cropped=True (Rhombus thumbnail is pre-cropped)  │
  │     b. rhombus analyze footage (CLI subprocess)                       │
  │        ↳ shells out to `rhombus analyze footage <cam> --fill --raw`   │
  │        ↳ CLI calls video/getExactFrameUri + dash-internal endpoint    │
  │        ↳ returns full 1920×1080 sensor frame, pre_cropped=False       │
  │     c. rhombus alert thumb (CLI subprocess)                           │
  │        ↳ only works once event is promoted to a Rhombus alert         │
  │        ↳ returns full frame, pre_cropped=False                        │
  │     d. refresh_tu_url() → camera/getFootageSeekpointsV2               │
  │        ↳ mints a freshly-signed tu URL, re-downloads                  │
  │        ↳ returns pre_cropped=True                                     │
  │     e. metadata URL fallback                                          │
  │        ↳ mediaapi-v2/media/metadata/{region}/{eventUuid}.jpeg         │
  │        ↳ returns full frame, pre_cropped=False                        │
  │                                                                       │
  │  3. Self-crop before YOLO (only when pre_cropped=False)               │
  │     • extract_motion_bbox(): unions all MOTION_CAR boundingBoxes      │
  │       from the payload into one (l, t, r, b) in permyriad coords      │
  │     • run_detection() crops the sensor frame to that region + 15%     │
  │       padding, runs YOLO on the crop, then maps boxes back to         │
  │       full-frame pixel coords                                         │
  │                                                                       │
  │  4. run_detection() → YOLOv8 (best.pt)                                │
  │     • confidence threshold: CONFIDENCE_THRESHOLD env var (default 0.70│
  │     • returns [(label, conf, x0, y0, x1, y1, img_w, img_h), ...]      │
  │                                                                       │
  │  5. If forklifts detected (conf ≥ threshold):                         │
  │     a. create_seekpoint() → camera/createCustomFootageSeekpoints      │
  │        • color GREEN, name "Forklift Detection"                       │
  │        • includes locationUuid and confidence in description           │
  │     b. create_bounding_boxes() → camera/createFootageBoundingBoxes    │
  │        • short-key schema: ts/a/l/t/r/b/c/objectId/m                  │
  │        • coords normalized to 0–10000 permyriad scale                 │
  │     c. log_detection() → detections.csv                               │
  │                                                                       │
  │  Dashboard: GET /dashboard — polls /stats.json every 2s               │
  │             in-memory ring buffer of last 200 events                  │
  └───────────────────────────┬───────────────────────────────────────────┘
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

## Frame Fetch Strategy Detail

The webhook's `tu` URL is a pre-signed, ephemeral thumbnail that expires within seconds. The full fallback chain in `get_frame()` ([webhook_server.py:228](webhook_server.py)):

| # | Method | Returns | Why it can fail |
|---|--------|---------|-----------------|
| 1 | Webhook `tu` URL (mTLS GET) | Pre-cropped thumbnail | Expires in ~seconds; 404 if webhook arrives late |
| 2 | `rhombus analyze footage` (CLI subprocess) | Full 1920×1080 frame | Times out ~45s; requires recorded footage within retention |
| 3 | `rhombus alert thumb` (CLI subprocess) | Full frame | Only works once event is promoted to a Rhombus alert |
| 4 | Re-query `camera/getFootageSeekpointsV2` for fresh `tu` | Pre-cropped thumbnail | Returns same expired token if called too late |
| 5 | `mediaapi-v2/media/metadata/{region}/{eventUuid}.jpeg` | Full frame | Only works for promoted alerts |

**Why `rhombus analyze` (strategy 2) is the reliable fallback:** Python's `requests` cannot reach Rhombus's `.dash-internal.rhombussystems.com` endpoints directly — the CLI's embedded Go HTTP client handles that TLS trust chain. Strategy 2 reads from durable recorded footage rather than the short-lived thumbnail cache.

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

Logged as: `Self-cropped full frame 1920x1080 → (x0,y0)-(x1,y1) [WxH] before YOLO`

---

## Rhombus API Endpoints

All calls are `POST`. Required headers: `X-Auth-Apikey` + `X-Auth-Scheme: api`.

| Endpoint | Purpose | Called from |
|----------|---------|-------------|
| `camera/createCustomFootageSeekpoints` | Write "Forklift Detection" seekpoint to timeline | `create_seekpoint()` |
| `camera/createFootageBoundingBoxes` | Write forklift bbox overlay (short-key schema) | `create_bounding_boxes()` |
| `camera/getFootageSeekpointsV2` | Re-query for freshly-signed `tu` URL | `refresh_tu_url()` |
| `GET mediaapi-v2/media/frame/{cam}/{ts}/thumb.jpeg?x=…` | Download webhook thumbnail (strategy 1) | `download_media()` |
| `GET mediaapi-v2/media/metadata/{region}/{uuid}.jpeg` | Fallback thumbnail by eventUuid (strategy 5) | `download_media()` |
| CLI `rhombus analyze footage` → `video/getExactFrameUri` | Pull exact frame from recorded footage (strategy 2) | `download_via_analyze()` |
| CLI `rhombus alert thumb` | Download alert thumbnail (strategy 3) | `download_via_cli()` |

> **Note:** The CLI command `get-camera-footage-seekpoints-v2` maps to the API path `camera/getFootageSeekpointsV2` (no "Camera" prefix). Use `rhombus --verbose` to see actual paths when debugging.

---

## Webhook Payload Shapes

Rules Engine fires **two pings per event**:

**Ping 1 — Minimal** (fires immediately, no seekpoints → deferred):
```json
{
  "deviceEvents": [{
    "deviceUuid": "1gKR-iBAQfmANqoQ9Nutjw.v0",
    "timestampMs": 1776967409560,
    "activities": ["MOTION_CAR"]
  }]
}
```

**Ping 2 — Finalized** (~5–30s later, has seekpoints + boundingBoxes → processed):
```json
{
  "deviceEvents": [{
    "deviceUuid": "1gKR-iBAQfmANqoQ9Nutjw.v0",
    "eventUuid": "MvBivqkZQx2Gho_bxEwNkA",
    "timestampMs": 1776970321350,
    "boundingBoxes": [
      {"activity": "MOTION_CAR", "left": 5266, "top": 6917, "right": 6328, "bottom": 9667,
       "confidence": 0.73, "objectId": 2320}
    ],
    "seekpoints": [{
      "activity": "MOTION_CAR",
      "tu": "/media/frame/…/thumb.jpeg?x=…&y=…&w=…&h=…&d=1&a=916"
    }]
  }]
}
```

---

## Source Files

| File | Role |
|------|------|
| [`webhook_server.py`](webhook_server.py) | Production server — Flask routes, frame fetch, YOLO, Rhombus write-back, dashboard |
| [`pipeline.py`](pipeline.py) | Legacy local poller — polls `rhombus alert recent`, not used in prod |
| [`forklift_detector.py`](forklift_detector.py) | Dev tool — standalone CLI for testing YOLO or Grounding DINO on a single image |
| [`local_test.py`](local_test.py) | Replay tool — pulls the most recent real webhook from Cloud Run logs and replays it locally |
| [`best.pt`](best.pt) | YOLOv8s forklift fine-tune weights (committed to repo) |

---

## Build & Deploy

```
local webhook_server.py
  │
  ├── git commit → Trevor-42/forklift-detection (PR flow, main branch protected)
  │
  └── gcloud run deploy forklift-detection --source . --region us-central1 --project forklift-detection-i2m
        │
        ├── Cloud Build: Dockerfile
        │    python:3.11-slim
        │    pip: ultralytics flask requests gunicorn Pillow
        │    curl-install rhombus CLI
        │    COPY webhook_server.py best.pt
        │
        └── Cloud Run revision N+1 → 100% traffic
             workers: 1  threads: 4  timeout: 120s
             Secrets mounted: client-crt, client-key (mTLS)
             Env: RHOMBUS_API_KEY
```

---

## Configuration

| Item | Value |
|------|-------|
| YOLO weights | `best.pt` (yolov8s forklift fine-tune) |
| Confidence threshold | `0.70` (env: `CONFIDENCE_THRESHOLD`) |
| Seekpoint color | `GREEN` |
| Self-crop padding | 15% on each side of the motion bbox |
| Rhombus org | `i2M` (`QwDXQrNZRzCtXGV_D5bUgQ`) |
| Cameras in scope | R410-A25K0532, R540-A25K0330 |
| Cloud Run region | `us-central1` |
| GCP project | `forklift-detection-i2m` |

---

## Known Quirks

| Issue | Symptom | Status |
|-------|---------|--------|
| Bounding boxes don't render in UI | `createFootageBoundingBoxes` stores for analytics but the player doesn't overlay | Accepted as-is — seekpoint renders fine |
| Webhook `tu` URL consistently expires | Every event falls through to `rhombus analyze` | Working correctly — strategy 2 handles it |
| `deviceUuid` has `.v0` suffix in Rules payload | API rejects UUID | `split(".")[0]` in `parse_payload()` |
| Long-form bbox keys | API returned 200 but nothing persisted | Use short keys: `a/l/t/r/b/c/m/ts/objectId` |
| Missing `X-Auth-Scheme` on mediaapi-v2 | HTTP 401 despite valid cert | Header is set in `download_media()` |

---

## Future Work

- Forklift near-miss detection (forklift + human bbox proximity)
- Forklift speed estimation (bbox delta across frames)
- Multi-frame inference to reduce false positives
- Webhook timeout handling for slow YOLO cold starts
