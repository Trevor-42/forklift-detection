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
from __future__ import annotations

import collections
import csv
import json
import logging
import os
import subprocess
import tempfile
import threading
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
            "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
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


def refresh_tu_url(camera_uuid: str, timestamp_ms: int, window_sec: int = 10) -> str:
    """Re-query getCameraFootageSeekpointsV2 for a freshly-signed tu URL near timestamp_ms.

    The tu URLs returned directly in the webhook carry a short-lived auth token
    (?a=...) that expires within seconds. Calling this endpoint mints a new one
    that's valid at the time of the call.
    """
    try:
        start_sec = max(0, (timestamp_ms // 1000) - window_sec)
        resp = rhombus_post("camera/getFootageSeekpointsV2", {
            "cameraUuid":       f"{camera_uuid}.v0",
            "startTime":        start_sec,
            "duration":         window_sec * 2,
            "includeAnyMotion": False,
        })
        sps = resp.get("footageSeekPoints", []) if isinstance(resp, dict) else []
        # Pick the MOTION_CAR seekpoint closest to our target timestamp with a tu URL
        candidates = [sp for sp in sps if sp.get("tu") and sp.get("a") == "MOTION_CAR"]
        if not candidates:
            candidates = [sp for sp in sps if sp.get("tu")]
        if not candidates:
            return ""
        best = min(candidates, key=lambda sp: abs(sp.get("ts", 0) - timestamp_ms))
        return best.get("tu", "")
    except Exception as e:
        log.warning(f"refresh_tu_url failed: {e}")
        return ""


def get_exact_frame_uri(camera_uuid: str, timestamp_ms: int) -> str:
    """Ask Rhombus for a URI to the exact recorded frame at timestamp_ms.

    Unlike thumbnail `tu` URLs (ephemeral in-memory cache), this reads from
    recorded footage on disk, so it works even when the webhook lands long
    after the event.
    """
    try:
        resp = rhombus_post("video/getExactFrameUri", {
            "cameraUuid":  f"{camera_uuid}.v0",
            "timestampMs": timestamp_ms,
        })
        if isinstance(resp, dict) and not resp.get("error") and resp.get("frameUri"):
            return resp["frameUri"]
        log.info(f"getExactFrameUri: {resp.get('responseMessage') if isinstance(resp, dict) else resp}")
    except Exception as e:
        log.warning(f"getExactFrameUri failed: {e}")
    return ""


def get_frame(camera_uuid: str, timestamp_ms: int, event_uuid: str = "",
              region: str = "us-east-2", seekpoint_tu: str = "") -> Path | None:
    """Get a frame for YOLO — tries multiple strategies in decreasing freshness."""
    # Strategy 1: webhook-delivered tu URL — fastest path when fresh (no extra API call)
    if seekpoint_tu:
        frame = download_media(seekpoint_tu)
        if frame:
            return frame
        log.info("Webhook tu URL expired — falling back to getExactFrameUri")

    # Strategy 2: getExactFrameUri — pulls exact frame from recorded footage.
    # Robust against thumbnail-cache eviction; works for any ts in retention.
    exact_uri = get_exact_frame_uri(camera_uuid, timestamp_ms)
    if exact_uri:
        frame = download_media(exact_uri)
        if frame:
            log.info("Fetched frame via getExactFrameUri")
            return frame

    # Strategy 3: rhombus CLI (works only for promoted alert UUIDs)
    frame = download_via_cli(event_uuid)
    if frame:
        return frame

    # Strategy 4: re-query seekpoints for a freshly-signed tu URL
    fresh_tu = refresh_tu_url(camera_uuid, timestamp_ms)
    if fresh_tu and fresh_tu != seekpoint_tu:
        frame = download_media(fresh_tu)
        if frame:
            log.info("Recovered frame via refreshed tu URL")
            return frame

    # Strategy 5: metadata URL (only if event promoted to an alert)
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


def create_seekpoint(camera_uuid: str, timestamp_ms: int, confidence: float = 0.0,
                     location_uuid: str = ""):
    sp = {
        "timestampMs": timestamp_ms,
        "name":        "Forklift Detection",
        "description": f"YOLO forklift detection — {confidence:.1%} confidence",
        "color":       "GREEN",
    }
    if location_uuid:
        sp["locationUuid"] = location_uuid
    return rhombus_post("camera/createCustomFootageSeekpoints", {
        "cameraUuid": camera_uuid,
        "footageSeekPoints": [sp],
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
    """Extract (camera_uuid, event_uuid, timestamp_ms, region, seekpoint_tu, location_uuid) from any Rhombus webhook format."""
    # Rules Engine deviceEvents format
    if "deviceEvents" in payload:
        events = payload["deviceEvents"]
        if not events:
            return None, None, int(time.time() * 1000), "us-east-2", "", ""
        event = events[0]
        camera_uuid   = event.get("deviceUuid", "").split(".")[0]  # strip .v0 suffix if present
        event_uuid    = event.get("eventUuid") or event.get("uuid")
        timestamp_ms  = event.get("timestampMs", payload.get("triggeredTimestampMs", int(time.time() * 1000)))
        region        = event.get("thumbnailLocation", {}).get("region", "us-east-2")
        location_uuid = event.get("locationUuid", "")
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
        return camera_uuid, event_uuid, timestamp_ms, region, seekpoint_tu, location_uuid

    # Legacy triggerEvent format
    if "triggerEvent" in payload:
        event        = payload["triggerEvent"]
        camera_uuid  = event.get("deviceUuid") or event.get("cameraUuid")
        event_uuid   = event.get("uuid") or event.get("eventUuid") or payload.get("ruleUuid")
        timestamp_ms = event.get("timestampMs", int(time.time() * 1000))
        return camera_uuid, event_uuid, timestamp_ms, "us-east-2", "", event.get("locationUuid", "")

    # Policy alert payload
    camera_uuid  = payload.get("deviceUuid") or payload.get("cameraUuid")
    event_uuid   = payload.get("policyAlertUuid") or payload.get("uuid")
    timestamp_ms = payload.get("timestampMs", int(time.time() * 1000))
    return camera_uuid, event_uuid, timestamp_ms, "us-east-2", "", payload.get("locationUuid", "")


_last_payload: dict = {}

# In-memory ring buffer of recent webhook events for the dashboard.
# Cloud Run scales to zero so this resets on cold start — fine for a "last hour" view.
_EVENTS_LOCK = threading.Lock()
_EVENTS: collections.deque = collections.deque(maxlen=200)
_BOOT_TS = time.time()


def _record_event(event: dict) -> None:
    with _EVENTS_LOCK:
        _EVENTS.appendleft(event)


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
    t_start = time.time()
    payload = request.get_json(silent=True) or {}
    _last_payload = payload
    log.info(f"Webhook received: {json.dumps(payload)}")

    camera_uuid, event_uuid, timestamp_ms, region, seekpoint_tu, location_uuid = parse_payload(payload)

    event = {
        "received_at": t_start,
        "camera_uuid": camera_uuid or "",
        "event_uuid":  event_uuid or "",
        "timestamp_ms": timestamp_ms,
        "tu_present":  bool(seekpoint_tu),
        "status":      "pending",
        "forklift":    False,
        "count":       0,
        "best_conf":   0.0,
        "detections":  0,
        "latency_ms":  0,
        "reason":      "",
    }

    try:
        if not camera_uuid:
            event.update(status="ignored", reason="no camera uuid")
            return jsonify({"status": "ignored", "reason": "no camera uuid"}), 200

        # Rules Engine fires twice per event: a minimal "first ping" (no eventUuid,
        # no seekpoints, no tu URLs) then a finalized payload ~5-30s later with all
        # the media handles we need. Skip the minimal one — it has nothing fetchable.
        if not event_uuid and not seekpoint_tu:
            log.info(f"Minimal payload on {camera_uuid} — deferring, awaiting finalized event")
            event.update(status="deferred", reason="awaiting finalized event")
            return jsonify({"status": "deferred", "reason": "awaiting finalized event"}), 200

        log.info(f"Vehicle event on camera {camera_uuid} at {timestamp_ms} — fetching frame (tu={bool(seekpoint_tu)})...")

        thumb = get_frame(camera_uuid, timestamp_ms, event_uuid or "", region, seekpoint_tu)

        if not thumb or not thumb.exists():
            log.warning("Could not obtain image.")
            event.update(status="error", reason="media URL expired or not yet available")
            return jsonify({"status": "error", "reason": "image unavailable"}), 200

        detections = run_detection(thumb)
        forklifts  = [d for d in detections if d[0] == "forklift"]
        event["detections"] = len(detections)

        if not forklifts:
            log.info("No forklift detected.")
            event.update(status="ok", forklift=False)
            return jsonify({"status": "ok", "forklift": False}), 200

        best_conf = max(f[1] for f in forklifts)
        log.info(f"Forklift detected! {len(forklifts)} instance(s), best conf {best_conf:.1%}. Creating annotations...")
        log_detection(camera_uuid, event_uuid or "", forklifts)
        sp_resp = create_seekpoint(camera_uuid, timestamp_ms, best_conf, location_uuid)
        bb_resp = create_bounding_boxes(camera_uuid, timestamp_ms, forklifts)
        log.info(f"Seekpoint write: {sp_resp}  |  Bbox write: {bb_resp}")

        event.update(status="ok", forklift=True, count=len(forklifts), best_conf=best_conf)
        return jsonify({"status": "ok", "forklift": True, "count": len(forklifts)}), 200
    finally:
        event["latency_ms"] = int((time.time() - t_start) * 1000)
        _record_event(event)


@app.route("/stats.json", methods=["GET"])
def stats():
    with _EVENTS_LOCK:
        events = list(_EVENTS)
    total      = len(events)
    forklift_n = sum(1 for e in events if e.get("forklift"))
    errors_n   = sum(1 for e in events if e.get("status") == "error")
    ignored_n  = sum(1 for e in events if e.get("status") == "ignored")
    deferred_n = sum(1 for e in events if e.get("status") == "deferred")
    latencies  = [e["latency_ms"] for e in events if e.get("latency_ms")]
    avg_latency = int(sum(latencies) / len(latencies)) if latencies else 0
    per_camera: dict = {}
    for e in events:
        cam = e.get("camera_uuid") or "unknown"
        slot = per_camera.setdefault(cam, {"total": 0, "forklift": 0})
        slot["total"] += 1
        if e.get("forklift"):
            slot["forklift"] += 1
    return jsonify({
        "uptime_sec":   int(time.time() - _BOOT_TS),
        "total":        total,
        "forklifts":    forklift_n,
        "errors":       errors_n,
        "ignored":      ignored_n,
        "deferred":     deferred_n,
        "avg_latency":  avg_latency,
        "hit_rate":     (forklift_n / total) if total else 0,
        "per_camera":   per_camera,
        "events":       events[:50],
        "now":          time.time(),
    }), 200


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Forklift Detection — Live Dashboard</title>
<style>
  :root { color-scheme: dark; }
  body { font: 14px/1.4 -apple-system, system-ui, sans-serif; margin: 0; background: #0c0e13; color: #e6e8eb; }
  header { padding: 16px 24px; background: #14181f; border-bottom: 1px solid #222833; display: flex; justify-content: space-between; align-items: center; }
  h1 { margin: 0; font-size: 18px; font-weight: 600; }
  .pulse { display:inline-block; width: 8px; height: 8px; background: #2ecc71; border-radius: 50%; margin-right: 8px; animation: pulse 1.6s infinite; }
  @keyframes pulse { 0%,100% { opacity: 1 } 50% { opacity: .3 } }
  main { padding: 24px; max-width: 1200px; margin: 0 auto; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 24px; }
  .card { background: #14181f; border: 1px solid #222833; border-radius: 8px; padding: 14px 16px; }
  .card .label { color: #8b97a8; font-size: 11px; text-transform: uppercase; letter-spacing: .05em; }
  .card .value { font-size: 24px; font-weight: 600; margin-top: 4px; }
  .card .value.green { color: #2ecc71; }
  .card .value.red { color: #e74c3c; }
  .card .value.amber { color: #f1c40f; }
  h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .05em; color: #8b97a8; margin: 24px 0 8px; }
  table { width: 100%; border-collapse: collapse; background: #14181f; border: 1px solid #222833; border-radius: 8px; overflow: hidden; }
  th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid #222833; font-variant-numeric: tabular-nums; }
  th { background: #1a1f29; color: #8b97a8; font-weight: 500; font-size: 11px; text-transform: uppercase; }
  tr:last-child td { border-bottom: none; }
  tr.forklift { background: rgba(46, 204, 113, .08); }
  tr.error { background: rgba(231, 76, 60, .08); }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 500; }
  .badge.ok { background: #1a3a2a; color: #2ecc71; }
  .badge.error { background: #3a1a1a; color: #e74c3c; }
  .badge.ignored { background: #2a2f3a; color: #8b97a8; }
  .badge.deferred { background: #2a2a3a; color: #a29bfe; }
  .badge.pending { background: #2a2f3a; color: #8b97a8; }
  .badge.forklift { background: #1a3a2a; color: #2ecc71; }
  .muted { color: #6b7585; font-size: 11px; }
  .foot { margin-top: 20px; text-align: center; color: #6b7585; font-size: 11px; }
</style>
</head>
<body>
<header>
  <h1>Forklift Detection — Live Pipeline</h1>
  <div><span class="pulse"></span><span id="status">connecting…</span></div>
</header>
<main>
  <div class="cards">
    <div class="card"><div class="label">Events (last 200)</div><div class="value" id="total">—</div></div>
    <div class="card"><div class="label">Forklifts Detected</div><div class="value green" id="forklifts">—</div></div>
    <div class="card"><div class="label">Hit Rate</div><div class="value" id="hit_rate">—</div></div>
    <div class="card"><div class="label">Errors</div><div class="value red" id="errors">—</div></div>
    <div class="card"><div class="label">Deferred (awaiting finalized)</div><div class="value" id="deferred">—</div></div>
    <div class="card"><div class="label">Avg Latency</div><div class="value" id="avg_latency">—</div></div>
    <div class="card"><div class="label">Uptime</div><div class="value" id="uptime">—</div></div>
  </div>

  <h2>Per Camera</h2>
  <table><thead><tr><th>Camera UUID</th><th>Events</th><th>Forklifts</th><th>Hit Rate</th></tr></thead>
    <tbody id="per_camera"><tr><td colspan="4" class="muted">no data yet</td></tr></tbody>
  </table>

  <h2>Recent Events</h2>
  <table><thead><tr>
    <th>Time</th><th>Camera</th><th>Event</th><th>Status</th><th>Forklift</th><th>Conf</th><th>Dets</th><th>Latency</th><th>Reason</th>
  </tr></thead>
    <tbody id="events"><tr><td colspan="9" class="muted">waiting for webhook activity…</td></tr></tbody>
  </table>

  <div class="foot">Polls <code>/stats.json</code> every 2 s · In-memory buffer, resets on cold start</div>
</main>
<script>
function fmtTime(t) {
  const d = new Date(t * 1000);
  return d.toLocaleTimeString([], {hour12:false});
}
function fmtUptime(s) {
  const h = Math.floor(s/3600), m = Math.floor((s%3600)/60);
  return h ? `${h}h ${m}m` : `${m}m ${s%60}s`;
}
function shortId(id) { return id ? id.slice(0, 8) + '…' : '—'; }
async function tick() {
  try {
    const r = await fetch('/stats.json', {cache:'no-store'});
    const d = await r.json();
    document.getElementById('status').textContent = 'live';
    document.getElementById('total').textContent = d.total;
    document.getElementById('forklifts').textContent = d.forklifts;
    document.getElementById('hit_rate').textContent = (d.hit_rate*100).toFixed(1) + '%';
    document.getElementById('errors').textContent = d.errors;
    document.getElementById('deferred').textContent = d.deferred;
    document.getElementById('avg_latency').textContent = d.avg_latency + ' ms';
    document.getElementById('uptime').textContent = fmtUptime(d.uptime_sec);

    const pcBody = document.getElementById('per_camera');
    const cams = Object.entries(d.per_camera);
    pcBody.innerHTML = cams.length
      ? cams.map(([uuid,s]) => `<tr><td><code>${shortId(uuid)}</code></td><td>${s.total}</td><td>${s.forklift}</td><td>${s.total?((s.forklift/s.total*100).toFixed(0)+'%'):'—'}</td></tr>`).join('')
      : '<tr><td colspan="4" class="muted">no data yet</td></tr>';

    const evBody = document.getElementById('events');
    evBody.innerHTML = d.events.length
      ? d.events.map(e => {
          const statusBadge = `<span class="badge ${e.status}">${e.status}</span>`;
          const forkBadge = e.forklift ? `<span class="badge forklift">YES ×${e.count}</span>` : '—';
          const conf = e.best_conf ? (e.best_conf*100).toFixed(1)+'%' : '—';
          const rowCls = e.forklift ? 'forklift' : (e.status === 'error' ? 'error' : '');
          return `<tr class="${rowCls}"><td>${fmtTime(e.received_at)}</td><td><code>${shortId(e.camera_uuid)}</code></td><td><code>${shortId(e.event_uuid)}</code></td><td>${statusBadge}</td><td>${forkBadge}</td><td>${conf}</td><td>${e.detections}</td><td>${e.latency_ms} ms</td><td class="muted">${e.reason||''}</td></tr>`;
        }).join('')
      : '<tr><td colspan="9" class="muted">waiting for webhook activity…</td></tr>';
  } catch (e) {
    document.getElementById('status').textContent = 'reconnecting…';
  }
}
tick(); setInterval(tick, 2000);
</script>
</body></html>"""


@app.route("/dashboard", methods=["GET"])
def dashboard():
    from flask import Response
    return Response(DASHBOARD_HTML, mimetype="text/html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    get_model()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
