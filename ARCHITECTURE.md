# Forklift Detection — Architecture

End-to-end AI forklift detection for Rhombus cameras. Rhombus Rules Engine fires a webhook on `MOTION_CAR`, a Cloud Run service downloads the event thumbnail, runs YOLOv8, and — if a forklift is detected — writes a custom seekpoint and bounding box back to the camera timeline.

## System Flow

```
                    ┌─────────────────────────────────────┐
                    │       Rhombus Rules Engine          │
                    │  Rule: se4ooEiTRWOq5wDvgLr8pQ       │
                    │  Trigger: DEVICE_ACTIVITY_EVENT     │
                    │  Activity: MOTION_CAR               │
                    │  Cameras: R410 + R540 (i2M org)     │
                    └──────────────┬──────────────────────┘
                                   │ POST webhook (JSON)
                                   │ deviceEvents[] + seekpoints[].tu
                                   ▼
  ┌────────────────────────────────────────────────────────────────┐
  │              Cloud Run: forklift-detection                     │
  │        https://forklift-detection-…run.app/webhook             │
  │                                                                │
  │   ┌──────────────────────────────────────────────────────┐     │
  │   │  Flask webhook_server.py (Python 3.11 + Gunicorn)    │     │
  │   │                                                      │     │
  │   │  1. parse_payload()                                  │     │
  │   │     → (camera_uuid, event_uuid, ts, region,          │     │
  │   │        seekpoint_tu, location_uuid)                  │     │
  │   │                                                      │     │
  │   │  2. get_frame()  ← THUMBNAIL FETCH                   │     │
  │   │     a. rhombus alert thumb  (CLI subprocess)         │     │
  │   │     b. download_media(tu)   (mTLS)                   │     │
  │   │     c. metadata fallback    (mTLS)                   │     │
  │   │                                                      │     │
  │   │  3. run_detection()                                  │     │
  │   │     → YOLOv8 (best.pt) + PIL  → [forklift boxes]     │     │
  │   │                                                      │     │
  │   │  4. If forklifts ≥ 0.70 confidence:                  │     │
  │   │     - create_seekpoint()                             │     │
  │   │     - create_bounding_boxes()                        │     │
  │   │     - append to detections.csv                       │     │
  │   └──────────────────────────────────────────────────────┘     │
  │                                                                │
  │   Secrets (GCP Secret Manager, mounted):                       │
  │   - /run/secrets/crt/client-crt  ← rhombus-i2m-client-crt      │
  │   - /run/secrets/key/client-key  ← rhombus-i2m-client-key      │
  │   Env: RHOMBUS_API_KEY                                         │
  └──────────────────────────────────┬─────────────────────────────┘
                                     │ mTLS + X-Auth-Apikey + X-Auth-Scheme:api
                                     ▼
            ┌──────────────────────────────────────────┐
            │             Rhombus APIs                 │
            │                                          │
            │  api2.rhombussystems.com/api             │
            │  mediaapi-v2.rhombussystems.com          │
            └──────────────────────────────────────────┘
```

## Rhombus Endpoints Used

| # | Endpoint | Purpose | Auth | Called From |
|---|----------|---------|------|-------------|
| 1 | `webhook-integrations/updateWebhookIntegrationV2` | Register Cloud Run URL as the org's webhook receiver | API key | one-time (CLI) |
| 2 | `rules/updateRule` | Attach `webhookAction` to MOTION_CAR rule | API key | one-time (CLI) |
| 3 | **Inbound** — Rules Engine → our `/webhook` | Fires on MOTION_CAR on R410/R540 | N/A (incoming) | Rhombus → us |
| 4 | CLI `rhombus alert thumb <event>` | Download thumbnail via bundled CLI binary (handles auth internally) | mTLS + key | `download_via_cli()` |
| 5 | `GET mediaapi-v2/media/frame/{cam}/{ts}/thumb.jpeg?x=…&y=…&w=…&h=…&d=…&a=…` | Fetch pre-signed cropped frame (from `seekpoints[].tu`) | **mTLS + API key + X-Auth-Scheme:api** | `download_media()` |
| 6 | `GET mediaapi-v2/media/metadata/{region}/{uuid}.jpeg` | Fallback thumbnail by eventUuid | mTLS + API key + X-Auth-Scheme:api | `download_media()` |
| 7 | `POST api/camera/createCustomFootageSeekpoints` | Write "Forklift Detection" marker to timeline | API key + X-Auth-Scheme:api | `create_seekpoint()` |
| 8 | `POST api/camera/createFootageBoundingBoxes` | Write forklift bbox overlay (short-key schema: ts/a/l/t/r/b/c/m) | API key + X-Auth-Scheme:api | `create_bounding_boxes()` |
| 9 | `POST api/eventSearch/getCameraOrDoorbellCameraSeekpoints` | Dev/test — find MOTION_CAR events for replay | mTLS + API key + X-Auth-Scheme:api | local CLI only |
| 10 | `POST api/camera/getCustomFootageSeekpointsV2` | Read-back verification | mTLS + API key + X-Auth-Scheme:api | local CLI only |
| 11 | `POST api/camera/getCameraFootageBoundingBoxes` | Read-back verification | mTLS + API key + X-Auth-Scheme:api | local CLI only |

## Key Schema Gotchas (Learned the Hard Way)

| Bug | Symptom | Fix |
|---|---|---|
| Long-form bbox keys (`activity/left/top/…`) | API returned 200 but nothing persisted | Use short keys: `a/l/t/r/b/c/m/ts/objectId` |
| Missing `X-Auth-Scheme` header on mediaapi-v2 | HTTP 401 despite valid cert | Add `X-Auth-Scheme: api` to `download_media()` |
| `deviceUuid` has `.v0` suffix in Rules payload | API rejected UUID | `split(".")[0]` in `parse_payload()` |
| Seekpoint missing `description`/`locationUuid` | Persisted but didn't render in UI | Include both fields (matches Rules Engine shape) |
| Webhook processes `tu=False` minimal payloads | Can't fetch frame, logs "Could not obtain image" | Correctly ignored — Rules Engine sends finalized payload with seekpoints ~5–30s later |

## Build / Deploy Pipeline

```
 local webhook_server.py
   │
   ├── git commit → Trevor-42/forklift-detection
   │
   └── gcloud run deploy --source .
         │
         ├── Cloud Build: Dockerfile
         │    - python:3.11-slim
         │    - pip: ultralytics, flask, requests, gunicorn
         │    - curl-install rhombus-cli v0.16.1
         │    - COPY webhook_server.py best.pt
         │
         └── Cloud Run revision N+1 → 100% traffic
              - Secrets mounted from Secret Manager
              - Service account: compute default (secretAccessor)
```

## Config / Thresholds

| Item | Value |
|---|---|
| YOLO weights | `best.pt` (yolov8s forklift fine-tune) |
| Confidence threshold | `0.70` (env: `CONFIDENCE_THRESHOLD`) |
| Seekpoint color | `GREEN` |
| Rhombus org | `i2M` (`QwDXQrNZRzCtXGV_D5bUgQ`) |
| Cameras in scope | R410-A25K0532, R540-A25K0330 |
| Cloud Run region | `us-central1` |
| Project | `forklift-detection-i2m` |

## Webhook Payload Shapes

The Rules Engine sends **two** payload variants per trigger event:

**Minimal (tu=False)** — fires immediately, no seekpoints:
```json
{
  "deviceEvents": [{
    "deviceUuid": "1gKR-iBAQfmANqoQ9Nutjw.v0",
    "timestampMs": 1776967409560,
    "activities": ["MOTION_CAR"]
  }],
  "ruleUuid": "se4ooEiTRWOq5wDvgLr8pQ"
}
```

**Finalized (tu=True)** — fires after clip finalizes (~5–30 s later) with full seekpoints/boundingBoxes:
```json
{
  "deviceEvents": [{
    "deviceUuid": "1gKR-iBAQfmANqoQ9Nutjw.v0",
    "eventUuid": "MvBivqkZQx2Gho_bxEwNkA",
    "timestampMs": 1776970321350,
    "seekpoints": [{
      "activity": "MOTION_CAR",
      "tu": "/media/frame/.../thumb.jpeg?x=...&y=...&w=...&h=...&d=2&a=916"
    }]
  }]
}
```

Only the finalized shape has a fetchable frame — the webhook processes it and skips the minimal shape.

## Future Work

- Forklift near-miss detection (forklift + human bbox proximity)
- Forklift speed estimation (bbox delta across frames)
- Multi-frame inference (reduce false positives)
- Webhook timeout handling for slow YOLO cold starts
