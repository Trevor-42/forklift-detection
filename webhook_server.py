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


_CLI_ENV_READY = False


def _cli_env() -> dict:
    """Prepare a subprocess env for the `rhombus` CLI.

    The CLI looks for certs at ~/.rhombus/certs/<profile>/client.{crt,key} and
    reads HOME for that path. We point HOME at /tmp and copy our mounted secrets
    into the expected location once per process.
    """
    global _CLI_ENV_READY
    env = {
        "HOME":            "/tmp",
        "RHOMBUS_API_KEY": RHOMBUS_API_KEY,
        "PATH":            "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
    }
    if _CLI_ENV_READY:
        return env
    try:
        config_dir = Path("/tmp/.rhombus")
        cert_dir   = config_dir / "certs" / "default"
        cert_dir.mkdir(parents=True, exist_ok=True)
        if CERT_FILE.exists() and KEY_FILE.exists():
            import shutil
            shutil.copy(CERT_FILE, cert_dir / "client.crt")
            shutil.copy(KEY_FILE,  cert_dir / "client.key")
        (config_dir / "credentials").write_text(
            "[default]\nauth_type = cert\n"
            f"api_key = {RHOMBUS_API_KEY}\n"
            f"key_file = {cert_dir}/client.key\n"
            f"cert_file = {cert_dir}/client.crt\n"
        )
        _CLI_ENV_READY = True
    except Exception as e:
        log.warning(f"CLI env setup failed: {e}")
    return env


def download_via_cli(event_uuid: str) -> Path | None:
    """Use the rhombus CLI to download an alert thumbnail — handles all auth internally."""
    if not event_uuid:
        return None
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False)
        tmp.close()
        result = subprocess.run(
            ["rhombus", "alert", "thumb", event_uuid, "--output", tmp.name],
            env=_cli_env(), capture_output=True, text=True, timeout=30,
        )
        path = Path(tmp.name)
        if path.exists() and path.stat().st_size > 1000:
            log.info(f"CLI thumbnail download succeeded ({path.stat().st_size} bytes)")
            return path
        log.warning(f"CLI thumb failed: {result.stderr[:200]}")
    except Exception as e:
        log.warning(f"CLI download exception: {e}")
    return None


def download_via_analyze(camera_uuid: str, timestamp_ms: int, window_ms: int = 3000) -> Path | None:
    """Use `rhombus analyze footage --fill --raw` to pull a frame near timestamp_ms.

    The CLI calls video/getExactFrameUri under the hood and downloads the
    resulting dash-internal URL with the right TLS client config — something
    Python requests can't reproduce (403s). Reads from recorded footage so it
    works even when the webhook's ephemeral thumbnail cache has evicted.
    """
    tmpdir = tempfile.mkdtemp(prefix="frame-")
    try:
        result = subprocess.run(
            ["rhombus", "analyze", "footage", camera_uuid,
             "--start", str(timestamp_ms),
             "--end",   str(timestamp_ms + window_ms),
             "--fill", "--raw", "--output", tmpdir],
            env=_cli_env(), capture_output=True, text=True, timeout=45,
        )
        for p in sorted(Path(tmpdir).rglob("*.jpeg")):
            if p.stat().st_size > 1000:
                log.info(f"Fetched frame via rhombus analyze: {p.stat().st_size} bytes")
                return p
        log.warning(f"rhombus analyze produced no frames. stderr: {result.stderr[:300]}")
    except subprocess.TimeoutExpired:
        log.warning("rhombus analyze timed out")
    except Exception as e:
        log.warning(f"rhombus analyze exception: {e}")
    return None


def analyze_event_frames(camera_uuid: str, timestamp_ms: int, duration_ms: int) -> list[tuple[Path, int]]:
    """Fetch all frames across the event window. Returns [(path, frame_timestamp_ms), ...].

    Uses --fill so frames are evenly spaced (~2s apart) regardless of seekpoint density.
    duration_ms is capped at 30s to bound cost.
    """
    duration_ms = min(duration_ms, 30_000)
    tmpdir = tempfile.mkdtemp(prefix="frames-")
    try:
        result = subprocess.run(
            ["rhombus", "analyze", "footage", camera_uuid,
             "--start", str(timestamp_ms),
             "--end",   str(timestamp_ms + duration_ms),
             "--fill", "--raw", "--output", tmpdir],
            env=_cli_env(), capture_output=True, text=True, timeout=60,
        )
        manifest_path = Path(tmpdir) / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            frames = []
            for cam_entry in manifest:
                for f in cam_entry.get("frames", []):
                    p = Path(f["path"])
                    if p.exists() and p.stat().st_size > 1000:
                        frames.append((p, f["timestampMs"]))
            if frames:
                log.info(f"analyze_event_frames: {len(frames)} frames over {duration_ms / 1000:.0f}s")
                return frames
        log.warning(f"analyze_event_frames: no frames. stderr: {result.stderr[:300]}")
    except subprocess.TimeoutExpired:
        log.warning("analyze_event_frames timed out")
    except Exception as e:
        log.warning(f"analyze_event_frames exception: {e}")
    return []


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


def get_frame(camera_uuid: str, timestamp_ms: int, event_uuid: str = "",
              region: str = "us-east-2", seekpoint_tu: str = "") -> tuple[Path | None, bool]:
    """Get a frame for YOLO — tries multiple strategies in decreasing freshness.

    Returns (path, was_pre_cropped). `was_pre_cropped` is True only for the
    Rhombus `tu` thumbnail path, which delivers a pre-cropped image zoomed
    in on the detected object. All other sources return the full sensor
    frame — the caller should self-crop those using the webhook's bbox
    hints before running inference.
    """
    # Strategy 1: webhook-delivered tu URL — fastest path when fresh (no extra API call)
    # Note: Rhombus returns a *cropped* thumbnail here (query params x/y/w/h).
    if seekpoint_tu:
        frame = download_media(seekpoint_tu)
        if frame:
            return frame, True
        log.info("Webhook tu URL expired — falling back to rhombus analyze")

    # Strategy 2: shell out to `rhombus analyze footage` — reads from recorded
    # footage via video/getExactFrameUri. Works even after the thumbnail cache
    # has evicted, because the CLI's Go HTTP client can talk to the
    # .dash-internal.rhombussystems.com endpoint that Python requests cannot.
    frame = download_via_analyze(camera_uuid, timestamp_ms)
    if frame:
        return frame, False

    # Strategy 3: rhombus alert thumb (works only for promoted alert UUIDs)
    frame = download_via_cli(event_uuid)
    if frame:
        return frame, False

    # Strategy 4: re-query seekpoints for a freshly-signed tu URL
    fresh_tu = refresh_tu_url(camera_uuid, timestamp_ms)
    if fresh_tu and fresh_tu != seekpoint_tu:
        frame = download_media(fresh_tu)
        if frame:
            log.info("Recovered frame via refreshed tu URL")
            return frame, True

    # Strategy 5: metadata URL (only if event promoted to an alert)
    if event_uuid:
        frame = download_media(f"{MEDIA_API}/media/metadata/{region}/{event_uuid}.jpeg")
        if frame:
            return frame, False

    log.warning("All image fetch methods failed.")
    return None, False


def extract_motion_bbox(payload: dict) -> tuple[int, int, int, int] | None:
    """Union all MOTION_CAR boundingBoxes in the payload into one (l, t, r, b) in permyriad.

    The payload contains a boundingBoxes[] array with per-frame crops as the
    object moves across the scene. Unioning them gives us a rectangle that
    covers the object's full trajectory — roughly the region YOLO should
    focus on. Returns None if no MOTION_CAR boxes are present.
    """
    events = payload.get("deviceEvents") or []
    if not events:
        return None
    motion_boxes = [b for b in events[0].get("boundingBoxes", [])
                    if b.get("activity") == "MOTION_CAR"]
    if not motion_boxes:
        return None
    l = min(b["left"]   for b in motion_boxes)
    t = min(b["top"]    for b in motion_boxes)
    r = max(b["right"]  for b in motion_boxes)
    b = max(b["bottom"] for b in motion_boxes)
    return (l, t, r, b)


def run_detection(image_path: Path, crop_permyriad: tuple[int, int, int, int] | None = None):
    """Run YOLO and return detections in FULL-FRAME coordinates.

    If `crop_permyriad` is given (tuple of l, t, r, b in 0..10000 units), crops
    the image to that region + 15% padding before inference, then maps
    detection boxes back to the original full-frame coordinate space. This is
    useful when we pulled the full sensor frame via `rhombus analyze` — YOLO
    sees the vehicle bigger if we narrow its view first.
    """
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size
    crop_x0 = crop_y0 = 0
    infer_image = image
    if crop_permyriad:
        l, t, r, b = crop_permyriad
        bw, bh = r - l, b - t
        pad_l = max(0, l - int(bw * 0.15))
        pad_t = max(0, t - int(bh * 0.15))
        pad_r = min(10000, r + int(bw * 0.15))
        pad_b = min(10000, b + int(bh * 0.15))
        x0 = int(pad_l / 10000 * img_w)
        y0 = int(pad_t / 10000 * img_h)
        x1 = int(pad_r / 10000 * img_w)
        y1 = int(pad_b / 10000 * img_h)
        if x1 - x0 > 32 and y1 - y0 > 32:
            infer_image = image.crop((x0, y0, x1, y1))
            crop_x0, crop_y0 = x0, y0
            log.info(f"Self-cropped full frame {img_w}x{img_h} → ({x0},{y0})-({x1},{y1}) "
                     f"[{x1 - x0}x{y1 - y0}] before YOLO")
    results = get_model().predict(source=infer_image, conf=THRESHOLD, verbose=False)[0]
    detections = []
    for box in results.boxes:
        label = get_model().names[int(box.cls[0])]
        conf  = float(box.conf[0])
        x0, y0, x1, y1 = box.xyxy[0].tolist()
        # Map crop-space coords back to the full original image.
        detections.append((label, conf,
                           x0 + crop_x0, y0 + crop_y0,
                           x1 + crop_x0, y1 + crop_y0,
                           img_w, img_h))
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

# Minimal pings that never received a finalized followup.
# Keyed by camera_uuid; swept every 10s and processed after a 45s timeout.
_PENDING_LOCK = threading.Lock()
_PENDING: dict = {}  # camera_uuid -> {timestamp_ms, location_uuid, deferred_at}


def _record_event(event: dict) -> None:
    with _EVENTS_LOCK:
        _EVENTS.appendleft(event)


def _process_deferred(camera_uuid: str, timestamp_ms: int, location_uuid: str) -> None:
    """Process a minimal-ping event whose finalized followup never arrived."""
    log.info(f"Sweeper: processing deferred event on {camera_uuid} at {timestamp_ms}")
    t_start = time.time()
    event = {
        "received_at":  t_start,
        "camera_uuid":  camera_uuid,
        "event_uuid":   "",
        "timestamp_ms": timestamp_ms,
        "tu_present":   False,
        "status":       "pending",
        "forklift":     False,
        "count":        0,
        "best_conf":    0.0,
        "detections":   0,
        "latency_ms":   0,
        "reason":       "deferred fallback",
    }
    try:
        # No durationSec in the minimal ping — use 20s default (~10 frames at 2s intervals).
        frames = analyze_event_frames(camera_uuid, timestamp_ms, 20_000)
        if not frames:
            log.warning(f"Sweeper: no frames for deferred event on {camera_uuid}")
            event.update(status="error", reason="deferred: frames unavailable")
            return
        # No bbox hint — minimal ping carries no boundingBoxes.
        forklifts: list = []
        det_timestamp_ms = timestamp_ms
        total_detections = 0
        for frame_path, frame_ts in frames:
            dets = run_detection(frame_path, crop_permyriad=None)
            total_detections += len(dets)
            hits = [d for d in dets if d[0] == "forklift"]
            if hits:
                forklifts = hits
                det_timestamp_ms = frame_ts
                log.info(f"Sweeper: forklift at frame ts={frame_ts} ({frames.index((frame_path, frame_ts)) + 1}/{len(frames)})")
                break
        event["detections"] = total_detections
        if not forklifts:
            log.info(f"Sweeper: no forklift on deferred event for {camera_uuid}")
            event.update(status="ok", forklift=False)
            return
        best_conf = max(f[1] for f in forklifts)
        log.info(f"Sweeper: forklift! {len(forklifts)} instance(s) at {best_conf:.1%} on {camera_uuid}")
        log_detection(camera_uuid, "", forklifts)
        sp_resp = create_seekpoint(camera_uuid, det_timestamp_ms, best_conf, location_uuid)
        bb_resp = create_bounding_boxes(camera_uuid, det_timestamp_ms, forklifts)
        log.info(f"Sweeper seekpoint: {sp_resp}  |  bbox: {bb_resp}")
        event.update(status="ok", forklift=True, count=len(forklifts), best_conf=best_conf)
    finally:
        event["latency_ms"] = int((time.time() - t_start) * 1000)
        _record_event(event)


def _pending_sweeper() -> None:
    """Background thread: retry deferred minimal pings that never got a finalized followup."""
    while True:
        time.sleep(10)
        now = time.time()
        to_process = []
        with _PENDING_LOCK:
            expired = [cam for cam, info in _PENDING.items()
                       if now - info["deferred_at"] > 45]
            for cam in expired:
                to_process.append((cam, _PENDING.pop(cam)))
        for cam, info in to_process:
            try:
                _process_deferred(cam, info["timestamp_ms"], info["location_uuid"])
            except Exception as e:
                log.warning(f"Sweeper exception for {cam}: {e}")


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
        # the media handles we need. Stash the minimal one in _PENDING; the sweeper
        # will process it via rhombus analyze if no finalized followup arrives in 45s.
        if not event_uuid and not seekpoint_tu:
            log.info(f"Minimal payload on {camera_uuid} — deferring, awaiting finalized event")
            with _PENDING_LOCK:
                _PENDING[camera_uuid] = {
                    "timestamp_ms": timestamp_ms,
                    "location_uuid": location_uuid,
                    "deferred_at": time.time(),
                }
            event.update(status="deferred", reason="awaiting finalized event")
            return jsonify({"status": "deferred", "reason": "awaiting finalized event"}), 200

        # Finalized ping — cancel any pending sweeper entry for this camera.
        with _PENDING_LOCK:
            _PENDING.pop(camera_uuid, None)

        log.info(f"Vehicle event on camera {camera_uuid} at {timestamp_ms} — fetching frame (tu={bool(seekpoint_tu)})...")

        crop_hint = extract_motion_bbox(payload)
        device_events = payload.get("deviceEvents") or [{}]
        duration_sec  = min(device_events[0].get("durationSec", 10), 30)

        # Build the frame list to scan. Start with the tu URL fast path (pre-cropped,
        # no extra API call) then fill the full event window via rhombus analyze.
        # Each entry is (path, frame_timestamp_ms, is_pre_cropped).
        frames_to_scan: list[tuple[Path, int, bool]] = []

        if seekpoint_tu:
            tu_frame = download_media(seekpoint_tu)
            if tu_frame:
                frames_to_scan.append((tu_frame, timestamp_ms, True))
            else:
                log.info("Webhook tu URL expired — scanning full event window")

        event_frames = analyze_event_frames(camera_uuid, timestamp_ms, duration_sec * 1000)
        for p, fts in event_frames:
            frames_to_scan.append((p, fts, False))

        if not frames_to_scan:
            log.warning("Could not obtain any frames.")
            event.update(status="error", reason="media unavailable")
            return jsonify({"status": "error", "reason": "image unavailable"}), 200

        forklifts: list = []
        det_timestamp_ms = timestamp_ms
        total_detections = 0

        for frame_path, frame_ts, pre_cropped in frames_to_scan:
            hint = None if pre_cropped else crop_hint
            dets = run_detection(frame_path, crop_permyriad=hint)
            total_detections += len(dets)
            hits = [d for d in dets if d[0] == "forklift"]
            if hits:
                forklifts = hits
                det_timestamp_ms = frame_ts
                log.info(f"Forklift found at frame ts={frame_ts} ({frames_to_scan.index((frame_path, frame_ts, pre_cropped)) + 1}/{len(frames_to_scan)})")
                break

        event["detections"] = total_detections

        if not forklifts:
            log.info("No forklift detected.")
            event.update(status="ok", forklift=False)
            return jsonify({"status": "ok", "forklift": False}), 200

        best_conf = max(f[1] for f in forklifts)
        log.info(f"Forklift detected! {len(forklifts)} instance(s), best conf {best_conf:.1%}. Creating annotations...")
        log_detection(camera_uuid, event_uuid or "", forklifts)
        sp_resp = create_seekpoint(camera_uuid, det_timestamp_ms, best_conf, location_uuid)
        bb_resp = create_bounding_boxes(camera_uuid, det_timestamp_ms, forklifts)
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


threading.Thread(target=_pending_sweeper, daemon=True, name="pending-sweeper").start()

if __name__ == "__main__":
    get_model()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
