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
import urllib3
from flask import Flask, request, jsonify

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from PIL import Image
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

WEIGHTS         = Path(__file__).parent / "best.pt"
LOG_FILE        = Path(__file__).parent / "detections.csv"
THRESHOLD       = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.70"))
RHOMBUS_API_KEY = os.environ.get("RHOMBUS_API_KEY", "")
RHOMBUS_API     = "https://api2.rhombussystems.com/api"
MEDIA_API       = "https://mediaapi-v2.rhombussystems.com"
CERT_FILE       = Path("/run/secrets/crt/client-crt")
KEY_FILE        = Path("/run/secrets/key/client-key")

model = None


def _cert():
    """Return (cert, key) tuple if available, else None."""
    if CERT_FILE.exists() and KEY_FILE.exists():
        return (str(CERT_FILE), str(KEY_FILE))
    return None


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
            "X-Auth-Scheme": "api",
        },
        cert=_cert(),
        verify=False,
        timeout=10,
    )
    return resp.json()


def download_media(url: str) -> Path | None:
    """Download an image from a Rhombus media URL using mTLS client cert."""
    try:
        full_url = url if url.startswith("http") else f"{MEDIA_API}{url}"
        resp = requests.get(
            full_url,
            headers={
                "X-Auth-Apikey": RHOMBUS_API_KEY,
                "X-Auth-Scheme": "api",
            },
            cert=_cert(),
            verify=False,
            timeout=15,
        )
        if resp.status_code == 200 and len(resp.content) > 1000:
            tmp = tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False)
            tmp.write(resp.content)
            tmp.close()
            return Path(tmp.name)
        log.warning(f"Media download HTTP {resp.status_code} for {full_url}")
    except Exception as e:
        log.warning(f"Media download failed: {e}")
    return None


def download_via_cli(event_uuid: str) -> Path | None:
    """Use the rhombus CLI to download an alert thumbnail — handles all auth internally."""
    if not event_uuid:
        return None
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False)
        tmp.close()
        # Configure CLI via environment variables pointing to our mounted secrets
        env = {
            "HOME": "/tmp",  # so CLI writes config to /tmp/.rhombus/
            "RHOMBUS_API_KEY": RHOMBUS_API_KEY,
            "PATH": "/usr/local/bin:/usr/bin:/bin",
        }
        # Write a minimal rhombus config using the mounted cert files
        config_dir = Path("/tmp/.rhombus")
        config_dir.mkdir(parents=True, exist_ok=True)
        cert_dir = config_dir / "certs" / "default"
        cert_dir.mkdir(parents=True, exist_ok=True)
        if CERT_FILE.exists():
            import shutil
            shutil.copy(CERT_FILE, cert_dir / "client.crt")
            shutil.copy(KEY_FILE, cert_dir / "client.key")
        (config_dir / "credentials").write_text(
            "[default]\nauth_type = cert\n"
            f"api_key = {RHOMBUS_API_KEY}\n"
            f"key_file = {cert_dir}/client.key\n"
            f"cert_file = {cert_dir}/client.crt\n"
        )
        import subprocess as sp
        result = sp.run(
            ["rhombus", "alert", "thumb", event_uuid, "--output", tmp.name],
            env=env, capture_output=True, text=True, timeout=30,
        )
        path = Path(tmp.name)
        if path.exists() and path.stat().st_size > 1000:
            log.info(f"CLI thumbnail download succeeded ({path.stat().st_size} bytes)")
            return path
        log.warning(f"CLI thumb failed: {result.stderr[:200]}")
    except Exception as e:
        log.warning(f"CLI download exception: {e}")
    return None


def get_frame(camera_uuid: str, timestamp_ms: int, event_uuid: str = "",
              region: str = "us-east-2", seekpoint_tu: str = "") -> Path | None:
    """Get a frame for YOLO — tries CLI auth first, then direct media download."""
    # Primary: rhombus CLI handles all Rhombus auth natively
    frame = download_via_cli(event_uuid)
    if frame:
        return frame

    # Fallback: direct media download with cert
    if seekpoint_tu:
        frame = download_media(seekpoint_tu)
        if frame:
            return frame

    if event_uuid:
        frame = download_media(f"{MEDIA_API}/media/metadata/{region}/{event_uuid}.jpeg")
        if frame:
            return frame

    log.warning("All image fetch methods failed.")
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
            "color": "GREEN",
        }]
    })


def create_bounding_boxes(camera_uuid: str, timestamp_ms: int, detections: list):
    # API uses short key names: ts, a, l, t, r, b, c, objectId, m
    boxes = [
        {
            "ts":       timestamp_ms,
            "a":        "MOTION_CAR",
            "l":        int(x0 / img_w * 10000),
            "t":        int(y0 / img_h * 10000),
            "r":        int(x1 / img_w * 10000),
            "b":        int(y1 / img_h * 10000),
            "c":        round(conf, 4),
            "objectId": i + 1,
            "m":        True,
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
    """Extract (camera_uuid, event_uuid, timestamp_ms, region, seekpoint_tu) from any Rhombus webhook format."""
    # Rules Engine deviceEvents format
    if "deviceEvents" in payload:
        events = payload["deviceEvents"]
        if not events:
            return None, None, int(time.time() * 1000), "us-east-2", ""
        event = events[0]
        camera_uuid  = event.get("deviceUuid", "").split(".")[0]  # strip .v0 suffix if present
        event_uuid   = event.get("eventUuid") or event.get("uuid")
        timestamp_ms = event.get("timestampMs", payload.get("triggeredTimestampMs", int(time.time() * 1000)))
        region       = event.get("thumbnailLocation", {}).get("region", "us-east-2")
        # Pick the best seekpoint thumbnail — prefer MOTION_CAR frames
        seekpoint_tu = ""
        for sp in event.get("seekpoints", []):
            if sp.get("tu") and sp.get("activity") == "MOTION_CAR":
                seekpoint_tu = sp["tu"]
                break
        if not seekpoint_tu:
            for sp in event.get("seekpoints", []):
                if sp.get("tu"):
                    seekpoint_tu = sp["tu"]
                    break
        return camera_uuid, event_uuid, timestamp_ms, region, seekpoint_tu

    # Legacy triggerEvent format
    if "triggerEvent" in payload:
        event        = payload["triggerEvent"]
        camera_uuid  = event.get("deviceUuid") or event.get("cameraUuid")
        event_uuid   = event.get("uuid") or event.get("eventUuid") or payload.get("ruleUuid")
        timestamp_ms = event.get("timestampMs", int(time.time() * 1000))
        return camera_uuid, event_uuid, timestamp_ms, "us-east-2", ""

    # Policy alert payload
    camera_uuid  = payload.get("deviceUuid") or payload.get("cameraUuid")
    event_uuid   = payload.get("policyAlertUuid") or payload.get("uuid")
    timestamp_ms = payload.get("timestampMs", int(time.time() * 1000))
    return camera_uuid, event_uuid, timestamp_ms, "us-east-2", ""


_last_payload: dict = {}


@app.route("/debug", methods=["GET"])
def debug():
    return jsonify(_last_payload), 200


@app.route("/debug/cert", methods=["GET"])
def debug_cert():
    """Diagnose cert mounting + mediaapi-v2 auth."""
    info = {
        "cert_path": str(CERT_FILE),
        "key_path": str(KEY_FILE),
        "cert_exists": CERT_FILE.exists(),
        "key_exists": KEY_FILE.exists(),
        "cert_size": CERT_FILE.stat().st_size if CERT_FILE.exists() else None,
        "key_size": KEY_FILE.stat().st_size if KEY_FILE.exists() else None,
        "cert_head": CERT_FILE.read_text()[:60] if CERT_FILE.exists() else None,
        "key_head": KEY_FILE.read_text()[:40] if KEY_FILE.exists() else None,
        "api_key_set": bool(RHOMBUS_API_KEY),
        "api_key_len": len(RHOMBUS_API_KEY),
    }
    # Probe a grid of mediaapi-v2 URL shapes to find which one auths cleanly
    probes = {
        "metadata_fake":  "https://mediaapi-v2.rhombussystems.com/media/metadata/us-east-2/test.jpeg",
        "frame_real":     "https://mediaapi-v2.rhombussystems.com/media/frame/1gKR-iBAQfmANqoQ9Nutjw/1776965936142/thumb.jpeg?d=1",
        "metadata_real":  "https://mediaapi-v2.rhombussystems.com/media/metadata/us-east-2/6LZ4SfTcRCuc-Df0OpuolQ.jpeg",
    }
    info["probes"] = {}
    for name, url in probes.items():
        try:
            r = requests.get(url,
                headers={"X-Auth-Apikey": RHOMBUS_API_KEY, "X-Auth-Scheme": "api"},
                cert=_cert(), verify=False, timeout=10)
            info["probes"][name] = {"status": r.status_code, "body": r.text[:150]}
        except Exception as e:
            info["probes"][name] = {"error": str(e)[:200]}
    # also probe without cert for handshake confirmation
    try:
        r = requests.get(probes["metadata_fake"],
            headers={"X-Auth-Apikey": RHOMBUS_API_KEY, "X-Auth-Scheme": "api"},
            verify=False, timeout=10)
        info["no_cert_status"] = r.status_code
    except Exception as e:
        info["no_cert_error"] = str(e)[:150]
    return jsonify(info), 200


@app.route("/webhook", methods=["POST"])
def webhook():
    global _last_payload
    payload = request.get_json(silent=True) or {}
    _last_payload = payload
    log.info(f"Webhook received: {json.dumps(payload)}")

    camera_uuid, event_uuid, timestamp_ms, region, seekpoint_tu = parse_payload(payload)

    if not camera_uuid:
        return jsonify({"status": "ignored", "reason": "no camera uuid"}), 200

    log.info(f"Vehicle event on camera {camera_uuid} at {timestamp_ms} — fetching frame (tu={bool(seekpoint_tu)})...")

    thumb = get_frame(camera_uuid, timestamp_ms, event_uuid or "", region, seekpoint_tu)

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
