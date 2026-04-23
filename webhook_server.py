"""
Rhombus Forklift Detection — Webhook Server

Receives MOTION_CAR alerts from Rhombus, runs YOLO forklift detection,
and writes a blue seekpoint + bounding box back to the camera on any hit.

Deploy to Render/Railway and register the URL in Rhombus:
  rhombus webhook-integrations update-webhook-integration-v2 \
    --profile i2M \
    --cli-input-json '{"webhookUrl":"https://your-app.onrender.com/webhook","disabled":false}'

Environment variables required:
  RHOMBUS_API_KEY   — i2M org API key
  WEBHOOK_SECRET    — optional shared secret to verify requests
"""

import csv
import json
import logging
import os
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import requests
from flask import Flask, request, jsonify
from PIL import Image
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

WEIGHTS        = Path(__file__).parent / "best.pt"
LOG_FILE       = Path(__file__).parent / "detections.csv"
THRESHOLD      = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.80"))
RHOMBUS_API_KEY = os.environ.get("RHOMBUS_API_KEY", "")
RHOMBUS_API    = "https://api2.rhombussystems.com/api"

model = None


def get_model():
    global model
    if model is None:
        log.info("Loading YOLO model...")
        model = YOLO(str(WEIGHTS))
        log.info("YOLO model loaded.")
    return model


def rhombus_post(endpoint: str, payload: dict):
    resp = requests.post(
        f"{RHOMBUS_API}/{endpoint}",
        json=payload,
        headers={
            "X-Auth-Apikey": RHOMBUS_API_KEY,
            "X-Auth-Scheme": "api-token",
        },
        timeout=10,
    )
    return resp.json()


def download_thumbnail(alert_uuid: str):
    """Download alert thumbnail, return path to temp file or None."""
    try:
        data = rhombus_post("event/getPolicyAlertDetails", {"policyAlertUuid": alert_uuid})
        alert = data.get("policyAlert", {})
        region = alert.get("thumbnailLocation", {}).get("region", "us-east-2")
        url = f"https://mediaapi-v2.rhombussystems.com/media/metadata/{region}/{alert_uuid}.jpeg"
        resp = requests.get(url, headers={"X-Auth-Apikey": RHOMBUS_API_KEY}, timeout=10)
        if resp.status_code == 200 and len(resp.content) > 1000:
            tmp = tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False)
            tmp.write(resp.content)
            tmp.close()
            return Path(tmp.name)
    except Exception as e:
        log.warning(f"Thumbnail download failed: {e}")
    return None


def run_detection(image_path: Path):
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size
    results = get_model().predict(source=image, conf=THRESHOLD, verbose=False)[0]
    detections = []
    for b in results.boxes:
        label = get_model().names[int(b.cls[0])]
        conf  = float(b.conf[0])
        x0, y0, x1, y1 = b.xyxy[0].tolist()
        detections.append((label, conf, x0, y0, x1, y1, img_w, img_h))
    return detections


def create_seekpoint(camera_uuid: str, timestamp_ms: int):
    return rhombus_post("camera/createCustomFootageSeekpoints", {
        "cameraUuid": camera_uuid,
        "footageSeekPoints": [{
            "timestampMs": timestamp_ms,
            "name": "Forklift Movement",
            "color": "BLUE",
        }]
    })


def create_bounding_boxes(camera_uuid: str, timestamp_ms: int, detections: list):
    boxes = [
        {
            "timestampMs": timestamp_ms,
            "activity":    "MOTION_CAR",
            "left":        int(x0 / img_w * 10000),
            "top":         int(y0 / img_h * 10000),
            "right":       int(x1 / img_w * 10000),
            "bottom":      int(y1 / img_h * 10000),
            "confidence":  round(conf, 4),
            "objectId":    i + 1,
            "inMotion":    True,
        }
        for i, (_, conf, x0, y0, x1, y1, img_w, img_h) in enumerate(detections)
    ]
    return rhombus_post("camera/createFootageBoundingBoxes", {
        "cameraUuid": camera_uuid,
        "footageBoundingBoxes": boxes,
    })


def log_detection(camera_uuid: str, alert_uuid: str, detections: list):
    write_header = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "camera_uuid", "alert_uuid", "label", "confidence"])
        for label, conf, *_ in detections:
            writer.writerow([datetime.now().isoformat(), camera_uuid, alert_uuid, label, f"{conf:.1%}"])


def parse_payload(payload: dict):
    """Extract camera_uuid, event_uuid, timestamp_ms from either alert or rules engine payload."""
    # Rules Engine payload
    if "triggerEvent" in payload:
        event      = payload["triggerEvent"]
        camera_uuid = event.get("deviceUuid") or event.get("cameraUuid")
        event_uuid  = event.get("uuid") or event.get("eventUuid") or payload.get("ruleUuid")
        timestamp_ms = event.get("timestampMs", int(time.time() * 1000))
        return camera_uuid, event_uuid, timestamp_ms

    # Policy alert payload
    camera_uuid  = payload.get("deviceUuid") or payload.get("cameraUuid")
    event_uuid   = payload.get("policyAlertUuid") or payload.get("uuid")
    timestamp_ms = payload.get("timestampMs", int(time.time() * 1000))
    return camera_uuid, event_uuid, timestamp_ms


@app.route("/webhook", methods=["POST"])
def webhook():
    payload = request.get_json(silent=True) or {}
    log.info(f"Webhook received: {json.dumps(payload)[:400]}")

    camera_uuid, event_uuid, timestamp_ms = parse_payload(payload)

    if not camera_uuid:
        return jsonify({"status": "ignored", "reason": "no camera uuid"}), 200

    log.info(f"Vehicle event on camera {camera_uuid} — running detection...")

    thumb = download_thumbnail(event_uuid) if event_uuid else None
    if not thumb:
        # Fall back to grabbing a fresh frame at the event timestamp
        log.warning("No thumbnail — attempting frame via exact URI...")
        try:
            frame_data = rhombus_post("video/getExactFrameUri", {"cameraUuid": camera_uuid, "timestampMs": timestamp_ms})
            frame_url  = frame_data.get("frameUri")
            if frame_url:
                resp = requests.get(frame_url, timeout=10)
                if resp.status_code == 200:
                    tmp = tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False)
                    tmp.write(resp.content)
                    tmp.close()
                    thumb = Path(tmp.name)
        except Exception as e:
            log.warning(f"Frame fallback failed: {e}")

    if not thumb or not thumb.exists():
        log.warning("Could not obtain image.")
        return jsonify({"status": "error", "reason": "image unavailable"}), 200

    detections = run_detection(thumb)
    forklifts  = [d for d in detections if d[0] == "forklift"]

    if not forklifts:
        log.info("No forklift detected.")
        return jsonify({"status": "ok", "forklift": False}), 200

    log.info(f"Forklift detected! {len(forklifts)} instance(s). Creating annotations...")
    log_detection(camera_uuid, event_uuid or "", forklifts)
    create_seekpoint(camera_uuid, timestamp_ms)
    create_bounding_boxes(camera_uuid, timestamp_ms, forklifts)

    return jsonify({"status": "ok", "forklift": True, "count": len(forklifts)}), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    get_model()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
