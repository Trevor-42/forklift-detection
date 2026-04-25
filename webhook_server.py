"""
Rhombus Forklift Detection — Webhook Server

Receives MOTION_CAR alerts from Rhombus, runs YOLO forklift detection,
and writes a blue seekpoint + bounding box back to the camera on any hit.

Deploy to Cloud Run and register the URL in Rhombus:
  gcloud run deploy forklift-detection --source . --region us-east1 \
    --project forklift-detection-i2m --min-instances 1 --memory 2Gi

  rhombus webhook-integrations update-webhook-integration \
    --profile i2M \
    --webhook-settings '{"webhookUrl":"https://<service-url>/webhook","disabled":false}'

Environment variables required:
  RHOMBUS_API_KEY   — i2M org API key
  WEBHOOK_SECRET    — optional shared secret to verify requests
"""
from __future__ import annotations

import base64
import collections
import concurrent.futures
import csv
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from io import BytesIO
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

WEIGHTS              = Path(__file__).parent / "best.pt"
LOG_FILE             = Path(__file__).parent / "detections.csv"
NEAR_MISS_LOG_FILE   = Path(__file__).parent / "near_misses.csv"
THRESHOLD            = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.70"))
NEAR_MISS_THRESHOLD  = int(os.environ.get("NEAR_MISS_THRESHOLD", "2000"))  # permyriad proximity
SPEED_LIMIT_MPH      = float(os.environ.get("SPEED_LIMIT_MPH", "10"))      # mph — above this gets a ORANGE seekpoint
RHOMBUS_API_KEY      = os.environ.get("RHOMBUS_API_KEY", "")
RHOMBUS_API          = "https://api2.rhombussystems.com/api"
MEDIA_API            = "https://mediaapi-v2.rhombussystems.com"
CERT_FILE            = Path("/run/secrets/crt/client-crt")
KEY_FILE             = Path("/run/secrets/key/client-key")

model: "YOLO | None" = None

# Per-camera scale factors set via /calibrate — permyriad per inch.
# Persists in memory until restart; also exposed via /calibrate/scales for copy-paste into env vars.
_CAMERA_SCALES: dict[str, float] = {}
# Pre-populate from env var: CAMERA_SCALES=uuid1:0.042,uuid2:0.038
for _entry in os.environ.get("CAMERA_SCALES", "").split(","):
    if ":" in _entry:
        _uid, _val = _entry.strip().split(":", 1)
        try:
            _CAMERA_SCALES[_uid] = float(_val)
        except ValueError:
            logging.getLogger(__name__).warning(f"Skipping bad CAMERA_SCALES entry: {_entry!r}")


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


def _thumbnail_b64(path: Path, max_width: int = 320) -> str:
    """Resize image to max_width and return base64-encoded JPEG for dashboard embedding."""
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if w > max_width:
            img = img.resize((max_width, int(h * max_width / w)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=75)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        log.warning(f"Thumbnail generation failed: {e}")
        return ""


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
    if not resp.ok:
        log.warning(f"rhombus_post {endpoint} returned HTTP {resp.status_code}: {resp.text[:200]}")
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
_CLI_ENV_LOCK = threading.Lock()


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
    with _CLI_ENV_LOCK:
        if _CLI_ENV_READY:  # re-check after acquiring lock
            return env
        try:
            config_dir = Path("/tmp/.rhombus")
            cert_dir   = config_dir / "certs" / "default"
            cert_dir.mkdir(parents=True, exist_ok=True)
            if CERT_FILE.exists() and KEY_FILE.exists():
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
                # Copy to an individual temp file so tmpdir can be cleaned up.
                tf = tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False)
                tf.write(p.read_bytes())
                tf.close()
                log.info(f"Fetched frame via rhombus analyze: {p.stat().st_size} bytes")
                return Path(tf.name)
        log.warning(f"rhombus analyze produced no frames. stderr: {result.stderr[:300]}")
    except subprocess.TimeoutExpired:
        log.warning("rhombus analyze timed out")
    except Exception as e:
        log.warning(f"rhombus analyze exception: {e}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
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
            env=_cli_env(), capture_output=True, text=True, timeout=90,
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
                # Copy each frame to an individual temp file so tmpdir can be cleaned up
                # immediately — callers are responsible for unlinking the returned paths.
                output_frames = []
                for p, ts in frames:
                    tf = tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False)
                    tf.write(p.read_bytes())
                    tf.close()
                    output_frames.append((Path(tf.name), ts))
                log.info(f"analyze_event_frames: {len(output_frames)} frames over {duration_ms / 1000:.0f}s")
                return output_frames
        log.warning(f"analyze_event_frames: no frames. stderr: {result.stderr[:300]}")
    except subprocess.TimeoutExpired as e:
        stderr = (e.stderr or "").strip()[:400] if hasattr(e, "stderr") else ""
        log.warning(f"analyze_event_frames timed out for {camera_uuid}. stderr: {stderr or '(none)'}")
    except Exception as e:
        log.warning(f"analyze_event_frames exception: {e}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
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
                     location_uuid: str = "", speed_mph: float | None = None):
    desc = f"YOLO forklift detection — {confidence:.1%} confidence"
    if speed_mph is not None:
        desc += f" · {speed_mph:.1f} mph"
    sp = {
        "timestampMs": timestamp_ms,
        "name":        "Forklift Detection",
        "description": desc,
        "color":       "PURPLE",
    }
    if location_uuid:
        sp["locationUuid"] = location_uuid
    return rhombus_post("camera/createCustomFootageSeekpoints", {
        "cameraUuid": camera_uuid,
        "footageSeekPoints": [sp],
    })


def create_near_miss_seekpoint(camera_uuid: str, timestamp_ms: int,
                               forklift_conf: float, location_uuid: str = ""):
    sp = {
        "timestampMs": timestamp_ms,
        "name":        "Near-Miss Alert",
        "description": f"Forklift + human in close proximity — forklift conf {forklift_conf:.1%}",
        "color":       "RED",
    }
    if location_uuid:
        sp["locationUuid"] = location_uuid
    return rhombus_post("camera/createCustomFootageSeekpoints", {
        "cameraUuid": camera_uuid,
        "footageSeekPoints": [sp],
    })


def estimate_speed(timed_detections: list[tuple[tuple, int]], camera_uuid: str) -> float | None:
    """Estimate forklift speed in mph from a sequence of (detection, timestamp_ms) pairs.

    Each detection is (label, conf, x0, y0, x1, y1, img_w, img_h) in pixel coords.
    Returns the median speed in mph across all consecutive frame pairs, or None if
    there are fewer than 2 frames, the camera has no calibration, or all speed
    samples are implausible (> 25 mph — forklifts physically can't go faster).
    """
    scale = _CAMERA_SCALES.get(camera_uuid)
    if not scale or len(timed_detections) < 2:
        return None

    speeds = []
    for i in range(1, len(timed_detections)):
        det1, ts1 = timed_detections[i - 1]
        det2, ts2 = timed_detections[i]
        dt_sec = (ts2 - ts1) / 1000.0
        if dt_sec <= 0:
            continue
        _, _, x0_1, y0_1, x1_1, y1_1, w1, h1 = det1
        _, _, x0_2, y0_2, x1_2, y1_2, w2, h2 = det2
        # Forklift centre in permyriad for each frame
        cx1 = ((x0_1 + x1_1) / 2) / w1 * 10000
        cy1 = ((y0_1 + y1_1) / 2) / h1 * 10000
        cx2 = ((x0_2 + x1_2) / 2) / w2 * 10000
        cy2 = ((y0_2 + y1_2) / 2) / h2 * 10000
        dist_permyriad = math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)
        dist_inches    = dist_permyriad / scale          # permyriad ÷ (permyriad/inch)
        dist_feet      = dist_inches / 12
        speed_mph      = (dist_feet / dt_sec) * 3600 / 5280
        if 0 < speed_mph < 25:                           # sanity bound
            speeds.append(speed_mph)

    if not speeds:
        return None
    speeds.sort()
    mid = len(speeds) // 2
    return speeds[mid] if len(speeds) % 2 else (speeds[mid - 1] + speeds[mid]) / 2


def create_speed_alert_seekpoint(camera_uuid: str, timestamp_ms: int,
                                  speed_mph: float, forklift_conf: float,
                                  location_uuid: str = ""):
    sp = {
        "timestampMs": timestamp_ms,
        "name":        "Speed Alert",
        "description": (f"Forklift exceeded {SPEED_LIMIT_MPH:.0f} mph limit — "
                        f"estimated {speed_mph:.1f} mph · conf {forklift_conf:.1%}"),
        "color":       "ORANGE",
    }
    if location_uuid:
        sp["locationUuid"] = location_uuid
    return rhombus_post("camera/createCustomFootageSeekpoints", {
        "cameraUuid": camera_uuid,
        "footageSeekPoints": [sp],
    })


def _parse_tu_bbox(tu: str) -> tuple[int, int, int, int] | None:
    """Extract (left, top, right, bottom) in permyriad from a tu URL's x/y/w/h params."""
    try:
        from urllib.parse import urlparse, parse_qs
        qs = parse_qs(urlparse(tu).query)
        x = int(qs["x"][0])
        y = int(qs["y"][0])
        w = int(qs["w"][0])
        h = int(qs["h"][0])
        return (x, y, x + w, y + h)
    except Exception:
        return None


def get_human_bboxes(camera_uuid: str, timestamp_ms: int, window_sec: int = 15) -> list[tuple[int,int,int,int]]:
    """Query getFootageSeekpointsV2 and return MOTION_HUMAN bboxes (permyriad) near timestamp_ms."""
    try:
        start_sec = max(0, (timestamp_ms // 1000) - window_sec)
        resp = rhombus_post("camera/getFootageSeekpointsV2", {
            "cameraUuid":       f"{camera_uuid}.v0",
            "startTime":        start_sec,
            "duration":         window_sec * 2,
            "includeAnyMotion": True,
        })
        sps = resp.get("footageSeekPoints", []) if isinstance(resp, dict) else []
        bboxes = []
        for sp in sps:
            if sp.get("a") != "MOTION_HUMAN":
                continue
            tu = sp.get("tu", "")
            if not tu:
                continue
            bbox = _parse_tu_bbox(tu)
            if bbox:
                bboxes.append(bbox)
        log.info(f"get_human_bboxes: {len(bboxes)} MOTION_HUMAN boxes in ±{window_sec}s window")
        return bboxes
    except Exception as e:
        log.warning(f"get_human_bboxes failed: {e}")
        return []


def check_near_miss(forklift_detections: list, human_bboxes: list[tuple[int,int,int,int]],
                    threshold: int = NEAR_MISS_THRESHOLD) -> bool:
    """Return True if any human center falls within threshold permyriad of any forklift bbox.

    Forklift detections are (label, conf, x0, y0, x1, y1, img_w, img_h) in pixel coords.
    Human bboxes are (left, top, right, bottom) in permyriad (0–10000).
    We convert forklift pixels → permyriad for comparison.
    Also skips humans whose bbox center is fully inside the forklift bbox
    (likely the operator in the cab).
    """
    if not human_bboxes:
        return False
    for _, _, x0, y0, x1, y1, img_w, img_h in forklift_detections:
        # Convert forklift to permyriad
        fl = int(x0 / img_w * 10000); ft = int(y0 / img_h * 10000)
        fr = int(x1 / img_w * 10000); fb = int(y1 / img_h * 10000)
        # Expand by threshold on each side for proximity zone
        zl = fl - threshold; zt = ft - threshold
        zr = fr + threshold; zb = fb + threshold
        for (hl, ht, hr, hb) in human_bboxes:
            hcx = (hl + hr) // 2
            hcy = (ht + hb) // 2
            # Skip operator in cab — human center fully inside forklift bbox
            if fl <= hcx <= fr and ft <= hcy <= fb:
                log.info(f"Near-miss: skipping human at ({hcx},{hcy}) — inside forklift bbox (operator in cab)")
                continue
            if zl <= hcx <= zr and zt <= hcy <= zb:
                log.info(f"Near-miss detected! Human center ({hcx},{hcy}) within {threshold} permyriad of forklift ({fl},{ft})-({fr},{fb})")
                return True
    return False


def log_near_miss(camera_uuid: str, timestamp_ms: int, forklift_conf: float):
    with _CSV_LOCK:
        write_header = not NEAR_MISS_LOG_FILE.exists()
        with open(NEAR_MISS_LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["timestamp", "camera_uuid", "event_timestamp_ms", "forklift_confidence"])
            writer.writerow([datetime.now().isoformat(), camera_uuid, timestamp_ms, f"{forklift_conf:.1%}"])


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
    with _CSV_LOCK:
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


_CSV_LOCK = threading.Lock()  # guards both LOG_FILE and NEAR_MISS_LOG_FILE writes


def _cleanup_temp_files(paths: list) -> None:
    """Unlink a list of temp file Paths, ignoring errors."""
    for p in paths:
        try:
            if p and Path(p).exists():
                Path(p).unlink()
        except Exception:
            pass

_last_payload: dict = {}

# In-memory ring buffer of recent webhook events for the dashboard.
# Cloud Run scales to zero so this resets on cold start — fine for a "last hour" view.
_EVENTS_LOCK = threading.Lock()
_EVENTS: collections.deque = collections.deque(maxlen=200)
_BOOT_TS = time.time()

# Minimal pings that never received a finalized followup.
# Keyed by camera_uuid; swept every 10s and processed after a 45s timeout.
# Each entry also carries a frames_future — analyze_event_frames running in background
# so frames are ready (or nearly ready) when the finalized second ping arrives.
_PENDING_LOCK = threading.Lock()
_PENDING: dict = {}  # camera_uuid -> {timestamp_ms, location_uuid, deferred_at, frames_future}
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="prefetch")


def _record_event(event: dict) -> None:
    with _EVENTS_LOCK:
        _EVENTS.appendleft(event)


def _process_deferred(camera_uuid: str, timestamp_ms: int, location_uuid: str,
                      frames_future=None) -> None:
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
        "thumb_b64":    "",
        "near_miss":    False,
        "speed_mph":    None,
        "speed_alert":  False,
    }
    try:
        # After 45s the pre-fetch future (started at first-ping time) is almost certainly done.
        frames = None
        if frames_future is not None:
            try:
                frames = frames_future.result(timeout=5)
                if frames:
                    log.info(f"Sweeper: using {len(frames)} pre-fetched frames")
            except Exception as e:
                log.warning(f"Sweeper: pre-fetch future error: {e}")
        if not frames:
            frames = analyze_event_frames(camera_uuid, timestamp_ms, 20_000)
        if not frames:
            log.warning(f"Sweeper: no frames for deferred event on {camera_uuid}")
            event.update(status="error", reason="deferred: frames unavailable")
            return
        # No bbox hint — minimal ping carries no boundingBoxes.
        # Scan ALL frames to collect trajectory for speed estimation.
        best_forklifts: list = []
        best_conf_so_far = 0.0
        det_timestamp_ms = timestamp_ms
        total_detections = 0
        thumb_frame: Path | None = None
        timed_detections: list[tuple[tuple, int]] = []
        for frame_idx, (frame_path, frame_ts) in enumerate(frames):
            if thumb_frame is None:
                thumb_frame = frame_path
            dets = run_detection(frame_path, crop_permyriad=None)
            total_detections += len(dets)
            hits = [d for d in dets if d[0] == "forklift"]
            if hits:
                frame_best = max(hits, key=lambda d: d[1])
                timed_detections.append((frame_best, frame_ts))
                if frame_best[1] > best_conf_so_far:
                    best_conf_so_far = frame_best[1]
                    best_forklifts = hits
                    det_timestamp_ms = frame_ts
                    thumb_frame = frame_path
                log.info(f"Sweeper: forklift at frame ts={frame_ts} ({frame_idx + 1}/{len(frames)})")

        forklifts = best_forklifts
        event["detections"] = total_detections
        if thumb_frame:
            event["thumb_b64"] = _thumbnail_b64(thumb_frame)
        if not forklifts:
            log.info(f"Sweeper: no forklift on deferred event for {camera_uuid}")
            event.update(status="ok", forklift=False)
            return
        best_conf = max(f[1] for f in forklifts)

        speed_mph = estimate_speed(timed_detections, camera_uuid)
        if speed_mph is not None:
            log.info(f"Sweeper speed estimate: {speed_mph:.1f} mph ({len(timed_detections)} frames)")

        log.info(f"Sweeper: forklift! {len(forklifts)} instance(s) at {best_conf:.1%} on {camera_uuid}")
        log_detection(camera_uuid, "", forklifts)
        sp_resp = create_seekpoint(camera_uuid, det_timestamp_ms, best_conf, location_uuid, speed_mph)
        bb_resp = create_bounding_boxes(camera_uuid, det_timestamp_ms, forklifts)
        log.info(f"Sweeper seekpoint: {sp_resp}  |  bbox: {bb_resp}")

        # Speed alert
        speed_alert = False
        if speed_mph is not None and speed_mph > SPEED_LIMIT_MPH:
            log.info(f"Sweeper SPEED ALERT on {camera_uuid}: {speed_mph:.1f} mph")
            create_speed_alert_seekpoint(camera_uuid, det_timestamp_ms, speed_mph, best_conf, location_uuid)
            speed_alert = True

        # Near-miss check
        human_bboxes = get_human_bboxes(camera_uuid, det_timestamp_ms)
        near_miss = check_near_miss(forklifts, human_bboxes)
        if near_miss:
            log.info(f"Sweeper NEAR-MISS on {camera_uuid}! Writing RED seekpoint.")
            log_near_miss(camera_uuid, det_timestamp_ms, best_conf)
            create_near_miss_seekpoint(camera_uuid, det_timestamp_ms, best_conf, location_uuid)
        event.update(status="ok", forklift=True, count=len(forklifts), best_conf=best_conf,
                     near_miss=near_miss, speed_mph=speed_mph, speed_alert=speed_alert)
    finally:
        event["latency_ms"] = int((time.time() - t_start) * 1000)
        _record_event(event)
        _cleanup_temp_files([p for p, _ in frames] if frames else [])


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
                _process_deferred(cam, info["timestamp_ms"], info["location_uuid"],
                                  frames_future=info.get("frames_future"))
            except Exception as e:
                log.warning(f"Sweeper exception for {cam}: {e}")


_DEBUG_TOKEN = os.environ.get("DEBUG_TOKEN", "")


def _check_debug_token() -> bool:
    """Return True if the request carries a valid DEBUG_TOKEN (or none is set)."""
    if not _DEBUG_TOKEN:
        return True  # unset = unrestricted (useful in dev)
    return request.args.get("token") == _DEBUG_TOKEN or \
           request.headers.get("X-Debug-Token") == _DEBUG_TOKEN


@app.route("/debug", methods=["GET"])
def debug():
    if not _check_debug_token():
        return jsonify({"error": "unauthorized"}), 403
    return jsonify(_last_payload), 200


@app.route("/debug/cert", methods=["GET"])
def debug_cert():
    """Diagnose cert mounting + mediaapi-v2 auth."""
    if not _check_debug_token():
        return jsonify({"error": "unauthorized"}), 403
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
    # Log a condensed summary — avoid dumping the full payload which can contain
    # large base64 blobs and sensitive media URLs.
    _events = payload.get("deviceEvents") or []
    _ev0 = _events[0] if _events else {}
    log.info(
        f"Webhook received: camera={_ev0.get('deviceUuid','?')} "
        f"eventUuid={_ev0.get('eventUuid','—')} "
        f"ts={_ev0.get('timestampMs','?')} "
        f"seekpoints={len(_ev0.get('seekpoints', []))} "
        f"bboxes={len(_ev0.get('boundingBoxes', []))}"
    )

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
        "thumb_b64":   "",
        "near_miss":   False,
        "speed_mph":   None,
        "speed_alert": False,
    }
    frames_to_scan: list[tuple[Path, int, bool]] = []  # initialised here so finally can clean up

    try:
        if not camera_uuid:
            event.update(status="ignored", reason="no camera uuid")
            return jsonify({"status": "ignored", "reason": "no camera uuid"}), 200

        # Rules Engine fires twice per event: a minimal "first ping" (no eventUuid,
        # no seekpoints, no tu URLs) then a finalized payload ~5-30s later with all
        # the media handles we need. On the first ping, immediately kick off
        # analyze_event_frames in a background thread so frames are ready (or nearly
        # ready) when the finalized second ping arrives, cutting ~12s off latency.
        if not event_uuid and not seekpoint_tu:
            log.info(f"Minimal payload on {camera_uuid} — deferring, pre-fetching frames in background")
            future = _EXECUTOR.submit(analyze_event_frames, camera_uuid, timestamp_ms, 20_000)
            with _PENDING_LOCK:
                _PENDING[camera_uuid] = {
                    "timestamp_ms": timestamp_ms,
                    "location_uuid": location_uuid,
                    "deferred_at": time.time(),
                    "frames_future": future,
                }
            event.update(status="deferred", reason="awaiting finalized event")
            return jsonify({"status": "deferred", "reason": "awaiting finalized event"}), 200

        # Finalized ping — pop any pending entry (grab the pre-fetch future if present).
        frames_future = None
        with _PENDING_LOCK:
            pending_info = _PENDING.pop(camera_uuid, None)
            if pending_info:
                frames_future = pending_info.get("frames_future")

        log.info(f"Vehicle event on camera {camera_uuid} at {timestamp_ms} — fetching frame (tu={bool(seekpoint_tu)})...")

        crop_hint = extract_motion_bbox(payload)
        device_events = payload.get("deviceEvents") or [{}]
        duration_sec  = min(device_events[0].get("durationSec", 10), 30)

        # Build the frame list to scan. Start with the tu URL fast path (pre-cropped,
        # no extra API call) then fill the full event window via rhombus analyze.
        # Each entry is (path, frame_timestamp_ms, is_pre_cropped).

        if seekpoint_tu:
            tu_frame = download_media(seekpoint_tu)
            if tu_frame:
                frames_to_scan.append((tu_frame, timestamp_ms, True))
            else:
                log.info("Webhook tu URL expired — scanning full event window")

        # Use pre-fetched frames if the background thread already finished,
        # otherwise wait for it (it started at first-ping time so it's often done).
        # Fall back to a fresh narrow-window analyze if the future failed or wasn't set.
        event_frames: list[tuple[Path, int]] = []
        if frames_future is not None:
            try:
                event_frames = frames_future.result(timeout=50)
                if event_frames:
                    log.info(f"Used {len(event_frames)} pre-fetched frames from first-ping background fetch")
                else:
                    log.info("Pre-fetch returned no frames — running fresh analyze")
            except Exception as e:
                log.warning(f"Pre-fetch future failed: {e}")
        if not event_frames:
            # Narrow window first (faster CLI call); widen if no frames returned.
            event_frames = analyze_event_frames(camera_uuid, timestamp_ms, 3_000)
            if not event_frames:
                event_frames = analyze_event_frames(camera_uuid, timestamp_ms, duration_sec * 1000)
        for p, fts in event_frames:
            frames_to_scan.append((p, fts, False))

        if not frames_to_scan:
            log.warning("Could not obtain any frames.")
            event.update(status="error", reason="media unavailable")
            return jsonify({"status": "error", "reason": "image unavailable"}), 200

        # Scan ALL frames — no early break — so we can compute speed from the trajectory.
        # best_forklifts/det_timestamp_ms track the highest-confidence frame for annotations.
        best_forklifts: list = []
        best_conf_so_far = 0.0
        det_timestamp_ms = timestamp_ms
        total_detections = 0
        thumb_frame: Path | None = None
        timed_detections: list[tuple[tuple, int]] = []  # (detection, ts) for speed calc

        for frame_idx, (frame_path, frame_ts, pre_cropped) in enumerate(frames_to_scan):
            if thumb_frame is None:
                thumb_frame = frame_path
            hint = None if pre_cropped else crop_hint
            dets = run_detection(frame_path, crop_permyriad=hint)
            total_detections += len(dets)
            hits = [d for d in dets if d[0] == "forklift"]
            if hits:
                frame_best = max(hits, key=lambda d: d[1])
                timed_detections.append((frame_best, frame_ts))
                if frame_best[1] > best_conf_so_far:
                    best_conf_so_far = frame_best[1]
                    best_forklifts = hits
                    det_timestamp_ms = frame_ts
                    thumb_frame = frame_path
                log.info(f"Forklift at frame ts={frame_ts} conf={frame_best[1]:.1%} "
                         f"({frame_idx + 1}/{len(frames_to_scan)})")

        forklifts = best_forklifts
        event["detections"] = total_detections
        if thumb_frame:
            event["thumb_b64"] = _thumbnail_b64(thumb_frame)

        if not forklifts:
            log.info("No forklift detected.")
            event.update(status="ok", forklift=False)
            return jsonify({"status": "ok", "forklift": False}), 200

        best_conf = max(f[1] for f in forklifts)

        # Speed estimation from multi-frame trajectory
        speed_mph = estimate_speed(timed_detections, camera_uuid)
        if speed_mph is not None:
            log.info(f"Speed estimate: {speed_mph:.1f} mph ({len(timed_detections)} frames)")
        else:
            log.info(f"Speed: n/a (scale={'set' if camera_uuid in _CAMERA_SCALES else 'not set'}, "
                     f"frames_with_detection={len(timed_detections)})")

        log.info(f"Forklift detected! {len(forklifts)} instance(s), best conf {best_conf:.1%}. Creating annotations...")
        log_detection(camera_uuid, event_uuid or "", forklifts)
        sp_resp = create_seekpoint(camera_uuid, det_timestamp_ms, best_conf, location_uuid, speed_mph)
        bb_resp = create_bounding_boxes(camera_uuid, det_timestamp_ms, forklifts)
        log.info(f"Seekpoint write: {sp_resp}  |  Bbox write: {bb_resp}")

        # Speed alert seekpoint if above limit
        speed_alert = False
        if speed_mph is not None and speed_mph > SPEED_LIMIT_MPH:
            log.info(f"SPEED ALERT on {camera_uuid}: {speed_mph:.1f} mph > {SPEED_LIMIT_MPH} mph limit")
            create_speed_alert_seekpoint(camera_uuid, det_timestamp_ms, speed_mph, best_conf, location_uuid)
            speed_alert = True

        # Near-miss check — query MOTION_HUMAN seekpoints near detection time
        human_bboxes = get_human_bboxes(camera_uuid, det_timestamp_ms)
        near_miss = check_near_miss(forklifts, human_bboxes)
        if near_miss:
            log.info(f"NEAR-MISS on {camera_uuid}! Writing RED seekpoint.")
            log_near_miss(camera_uuid, det_timestamp_ms, best_conf)
            create_near_miss_seekpoint(camera_uuid, det_timestamp_ms, best_conf, location_uuid)
        event.update(status="ok", forklift=True, count=len(forklifts), best_conf=best_conf,
                     near_miss=near_miss, speed_mph=speed_mph, speed_alert=speed_alert)
        return jsonify({"status": "ok", "forklift": True, "count": len(forklifts),
                        "near_miss": near_miss, "speed_mph": speed_mph}), 200
    finally:
        event["latency_ms"] = int((time.time() - t_start) * 1000)
        _record_event(event)
        # Clean up all temp frame files accumulated during this request.
        _cleanup_temp_files([p for p, _, __ in frames_to_scan])


@app.route("/stats.json", methods=["GET"])
def stats():
    with _EVENTS_LOCK:
        events = list(_EVENTS)
    total       = len(events)
    forklift_n  = sum(1 for e in events if e.get("forklift"))
    near_miss_n   = sum(1 for e in events if e.get("near_miss"))
    speed_alert_n = sum(1 for e in events if e.get("speed_alert"))
    errors_n      = sum(1 for e in events if e.get("status") == "error")
    ignored_n   = sum(1 for e in events if e.get("status") == "ignored")
    deferred_n  = sum(1 for e in events if e.get("status") == "deferred")
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
        "forklifts":      forklift_n,
        "near_misses":    near_miss_n,
        "speed_alerts":   speed_alert_n,
        "errors":         errors_n,
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
  .badge.near-miss { background: #3a1a1a; color: #e74c3c; font-weight: 700; }
  .badge.speed-alert { background: #3a2a00; color: #f39c12; font-weight: 700; }
  tr.speed-alert { background: rgba(243, 156, 18, .10); }
  .muted { color: #6b7585; font-size: 11px; }
  .foot { margin-top: 20px; text-align: center; color: #6b7585; font-size: 11px; }
  .thumb { height: 54px; border-radius: 4px; cursor: pointer; display: block; border: 1px solid #222833; }
  .thumb:hover { border-color: #a29bfe; }
  #modal { display:none; position:fixed; inset:0; background:rgba(0,0,0,.85); z-index:999; align-items:center; justify-content:center; }
  #modal img { max-width:90vw; max-height:90vh; border-radius:8px; border:1px solid #333; }
  #modal-close { position:absolute; top:20px; right:28px; font-size:28px; color:#fff; cursor:pointer; line-height:1; }
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
    <div class="card"><div class="label">Near-Miss Alerts</div><div class="value red" id="near_misses">—</div></div>
    <div class="card"><div class="label">Speed Alerts</div><div class="value amber" id="speed_alerts">—</div></div>
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
    <th>Time</th><th>Camera</th><th>Event</th><th>Status</th><th>Forklift</th><th>Conf</th><th>Speed</th><th>Dets</th><th>Latency</th><th>Reason</th><th>Frame</th>
  </tr></thead>
    <tbody id="events"><tr><td colspan="11" class="muted">waiting for webhook activity…</td></tr></tbody>
  </table>

  <div id="modal" onclick="closeModal()">
    <span id="modal-close" onclick="closeModal()">×</span>
    <img id="modal-img" src="" alt="frame" onclick="event.stopPropagation()" />
  </div>

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
function showFrame(src) {
  document.getElementById('modal-img').src = src;
  document.getElementById('modal').style.display = 'flex';
}
function closeModal() {
  document.getElementById('modal').style.display = 'none';
}
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
async function tick() {
  try {
    const r = await fetch('/stats.json', {cache:'no-store'});
    const d = await r.json();
    document.getElementById('status').textContent = 'live';
    document.getElementById('total').textContent = d.total;
    document.getElementById('forklifts').textContent = d.forklifts;
    document.getElementById('hit_rate').textContent = (d.hit_rate*100).toFixed(1) + '%';
    document.getElementById('errors').textContent = d.errors;
    document.getElementById('near_misses').textContent = d.near_misses || 0;
    document.getElementById('speed_alerts').textContent = d.speed_alerts || 0;
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
          const forkBadge = e.forklift ? `<span class="badge forklift">YES ×${e.count}</span>${e.near_miss ? ' <span class="badge near-miss">⚠ NEAR-MISS</span>' : ''}${e.speed_alert ? ' <span class="badge speed-alert">⚡ SPEED</span>' : ''}` : '—';
          const conf = e.best_conf ? (e.best_conf*100).toFixed(1)+'%' : '—';
          const speedCell = e.speed_mph != null ? `${e.speed_mph.toFixed(1)} mph` : '<span class="muted">—</span>';
          const rowCls = e.speed_alert ? 'speed-alert' : (e.forklift ? 'forklift' : (e.status === 'error' ? 'error' : ''));
          const src = `data:image/jpeg;base64,${e.thumb_b64}`;
          const thumbCell = e.thumb_b64 ? `<td><img class="thumb" src="${src}" onclick="showFrame('${src}')" /></td>` : `<td class="muted">—</td>`;
          return `<tr class="${rowCls}"><td>${fmtTime(e.received_at)}</td><td><code>${shortId(e.camera_uuid)}</code></td><td><code>${shortId(e.event_uuid)}</code></td><td>${statusBadge}</td><td>${forkBadge}</td><td>${conf}</td><td>${speedCell}</td><td>${e.detections}</td><td>${e.latency_ms} ms</td><td class="muted">${e.reason||''}</td>${thumbCell}</tr>`;
        }).join('')
      : '<tr><td colspan="10" class="muted">waiting for webhook activity…</td></tr>';
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


_CALIBRATE_FRAME_CACHE: dict[str, tuple[float, bytes]] = {}  # camera_uuid -> (fetched_at, jpeg_bytes)
_CALIBRATE_FRAME_LOCK = threading.Lock()
_CALIBRATE_FRAME_TTL = 30  # seconds — re-use cached frame within this window


@app.route("/calibrate/frame")
def calibrate_frame():
    """Fetch and serve a recent frame from the requested camera as JPEG.

    Results are cached per camera for 30 s to prevent concurrent CLI subprocesses
    from spawning if the user rapidly hits the endpoint.
    Errors are returned as plain text (not HTML) so the JS can display them cleanly.
    """
    from flask import Response
    camera_uuid = request.args.get("camera", "")
    if not camera_uuid:
        return Response("camera param required", status=400, mimetype="text/plain")

    now = time.time()
    with _CALIBRATE_FRAME_LOCK:
        cached = _CALIBRATE_FRAME_CACHE.get(camera_uuid)
        if cached and now - cached[0] < _CALIBRATE_FRAME_TTL:
            return Response(cached[1], mimetype="image/jpeg")

    # Try progressively older windows — footage takes a few seconds to commit to disk.
    # Start 60s ago with a 30s window, then fall back to 2 min ago if needed.
    frame = None
    for lookback_ms in (60_000, 120_000, 180_000):
        ts = int(now * 1000) - lookback_ms
        log.info(f"calibrate/frame: trying {camera_uuid} at -{lookback_ms//1000}s")
        frame = download_via_analyze(camera_uuid, ts, window_ms=30_000)
        if frame:
            break

    if not frame:
        return Response(
            "Could not fetch frame — camera may be offline, UUID may be wrong, "
            "or footage is unavailable. Check Cloud Run logs for details.",
            status=503, mimetype="text/plain",
        )
    try:
        data = frame.read_bytes()
    finally:
        frame.unlink(missing_ok=True)

    with _CALIBRATE_FRAME_LOCK:
        _CALIBRATE_FRAME_CACHE[camera_uuid] = (now, data)
    return Response(data, mimetype="image/jpeg")


@app.route("/calibrate/save", methods=["POST"])
def calibrate_save():
    """Store a permyriad_per_inch scale factor for a camera."""
    data = request.get_json(silent=True) or {}
    camera_uuid = data.get("camera_uuid", "")
    scale = data.get("permyriad_per_inch")
    if not camera_uuid or scale is None:
        return jsonify({"error": "camera_uuid and permyriad_per_inch required"}), 400
    scale_f = float(scale)
    _CAMERA_SCALES[camera_uuid] = scale_f
    log.info(f"Calibration saved: {camera_uuid} → {scale_f:.4f} permyriad/inch")
    return jsonify({"status": "ok", "camera_uuid": camera_uuid, "permyriad_per_inch": scale})


@app.route("/calibrate/scales")
def calibrate_scales():
    """Return stored scale factors and the env var string to persist them."""
    env_str = ",".join(f"{k}:{v:.6f}" for k, v in _CAMERA_SCALES.items())
    return jsonify({"scales": _CAMERA_SCALES, "env_var": f"CAMERA_SCALES={env_str}"})


CALIBRATE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Forklift Detection — Camera Calibration</title>
<style>
  :root { color-scheme: dark; }
  body { font: 14px/1.6 -apple-system, system-ui, sans-serif; margin: 0; background: #0c0e13; color: #e6e8eb; }
  header { padding: 16px 24px; background: #14181f; border-bottom: 1px solid #222833; display: flex; justify-content: space-between; align-items: center; }
  h1 { margin: 0; font-size: 18px; font-weight: 600; }
  main { padding: 24px; max-width: 1100px; margin: 0 auto; }
  .row { display: flex; gap: 24px; align-items: flex-start; flex-wrap: wrap; }
  .controls { min-width: 260px; }
  label { display: block; color: #8b97a8; font-size: 11px; text-transform: uppercase; letter-spacing: .05em; margin: 16px 0 4px; }
  input, select { width: 100%; box-sizing: border-box; background: #1a1f29; border: 1px solid #2a3040; border-radius: 6px; padding: 8px 10px; color: #e6e8eb; font-size: 13px; }
  button { margin-top: 10px; width: 100%; padding: 10px; background: #a29bfe; color: #0c0e13; border: none; border-radius: 6px; font-size: 13px; font-weight: 600; cursor: pointer; }
  button:hover { background: #c4b5fd; }
  button.secondary { background: #2a2f3a; color: #e6e8eb; margin-top: 6px; }
  #canvas-wrap { position: relative; display: inline-block; }
  canvas { border: 1px solid #222833; border-radius: 6px; cursor: crosshair; display: block; max-width: 100%; }
  .dot { position: absolute; width: 12px; height: 12px; border-radius: 50%; margin: -6px 0 0 -6px; pointer-events: none; }
  .dot.p1 { background: #f1c40f; }
  .dot.p2 { background: #e74c3c; }
  .result { margin-top: 16px; padding: 14px; background: #14181f; border: 1px solid #222833; border-radius: 8px; font-size: 13px; }
  .result .big { font-size: 22px; font-weight: 600; color: #2ecc71; margin: 4px 0; }
  .result code { background: #1a1f29; padding: 4px 8px; border-radius: 4px; font-size: 12px; }
  .muted { color: #6b7585; font-size: 12px; }
  .step { color: #8b97a8; font-size: 12px; margin: 6px 0; }
  .step span { color: #e6e8eb; }
  #msg { margin-top: 10px; font-size: 12px; color: #f1c40f; min-height: 18px; }
</style>
</head>
<body>
<header>
  <h1>Camera Calibration — Pallet Reference</h1>
  <a href="/dashboard" style="color:#a29bfe;font-size:13px;">← Dashboard</a>
</header>
<main>
  <p class="muted">Click two points on a known object in the camera frame (e.g. two corners of a standard pallet = 48" apart). The tool calculates the permyriad-per-inch scale factor for speed estimation.</p>
  <div class="row">
    <div class="controls">
      <label>Camera UUID</label>
      <input id="camera" placeholder="paste camera UUID" />
      <button onclick="loadFrame()">Load Frame</button>

      <label>Step 1 — Click point A on image</label>
      <div class="step">A: <span id="p1txt">not set</span></div>
      <label>Step 2 — Click point B on image</label>
      <div class="step">B: <span id="p2txt">not set</span></div>

      <label>Real-world distance between A and B (inches)</label>
      <input id="dist" type="number" value="48" min="1" />
      <div class="muted" style="margin-top:4px">Standard pallet width = 48" &nbsp;|&nbsp; pallet depth = 40"</div>

      <button onclick="compute()" style="margin-top:16px;background:#2ecc71;color:#0c0e13">Calculate Scale Factor</button>
      <button class="secondary" onclick="reset()">Reset Points</button>
      <div id="msg"></div>

      <div id="result" style="display:none" class="result">
        <div class="muted">Permyriad per inch</div>
        <div class="big" id="scale_val">—</div>
        <div class="muted" style="margin-top:8px">Set this as env var:</div>
        <code id="env_hint">—</code>
        <button onclick="saveScale()" style="margin-top:12px;background:#a29bfe;color:#0c0e13">Save to Server</button>
      </div>
    </div>

    <div>
      <div id="canvas-wrap">
        <canvas id="cv" width="800" height="450"></canvas>
      </div>
    </div>
  </div>
</main>
<script>
let pts = [], scale = null, camUuid = '', imgNatW = 1, imgNatH = 1, canvW = 800, canvH = 450;

async function loadFrame() {
  camUuid = document.getElementById('camera').value.trim();
  if (!camUuid) { msg('Enter a camera UUID first'); return; }
  msg('Fetching frame…');
  const cv = document.getElementById('cv');
  const ctx = cv.getContext('2d');
  ctx.fillStyle = '#14181f'; ctx.fillRect(0,0,cv.width,cv.height);
  ctx.fillStyle = '#8b97a8'; ctx.font = '14px system-ui';
  ctx.fillText('Loading…', cv.width/2-30, cv.height/2);
  try {
    const r = await fetch('/calibrate/frame?camera=' + encodeURIComponent(camUuid));
    if (!r.ok) {
      const errText = await r.text();
      // Strip any stray HTML tags in case of unexpected error responses
      msg('Error: ' + errText.replace(/<[^>]+>/g, '').trim().slice(0, 200));
      return;
    }
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      imgNatW = img.naturalWidth; imgNatH = img.naturalHeight;
      const aspect = imgNatW / imgNatH;
      canvW = 800; canvH = Math.round(800 / aspect);
      cv.width = canvW; cv.height = canvH;
      ctx.drawImage(img, 0, 0, canvW, canvH);
      msg('Frame loaded. Click two reference points.');
      reset();
    };
    img.src = url;
  } catch(e) { msg('Error: ' + e); }
}

document.getElementById('cv').addEventListener('click', e => {
  if (pts.length >= 2) return;
  const cv = document.getElementById('cv');
  const rect = cv.getBoundingClientRect();
  const cx = (e.clientX - rect.left) * (cv.width / rect.width);
  const cy = (e.clientY - rect.top) * (cv.height / rect.height);
  // Convert to permyriad
  const px = Math.round(cx / cv.width * 10000);
  const py = Math.round(cy / cv.height * 10000);
  pts.push({cx, cy, px, py});
  drawDot(cx, cy, pts.length === 1 ? 'p1' : 'p2');
  document.getElementById(pts.length === 1 ? 'p1txt' : 'p2txt').textContent = `(${px}, ${py}) permyriad`;
  if (pts.length === 2) msg('Both points set. Enter distance and click Calculate.');
});

function drawDot(cx, cy, cls) {
  const wrap = document.getElementById('canvas-wrap');
  const cv = document.getElementById('cv');
  const rect = cv.getBoundingClientRect();
  const scaleX = rect.width / cv.width;
  const d = document.createElement('div');
  d.className = 'dot ' + cls;
  d.style.left = (cx * scaleX + rect.left - wrap.getBoundingClientRect().left) + 'px';
  d.style.top = (cy * scaleX + rect.top - wrap.getBoundingClientRect().top) + 'px';
  wrap.appendChild(d);
}

function compute() {
  if (pts.length < 2) { msg('Click two points first'); return; }
  const dist = parseFloat(document.getElementById('dist').value);
  if (!dist || dist <= 0) { msg('Enter a valid distance'); return; }
  const dx = pts[1].px - pts[0].px, dy = pts[1].py - pts[0].py;
  const permyriadDist = Math.sqrt(dx*dx + dy*dy);
  scale = permyriadDist / dist;
  document.getElementById('scale_val').textContent = scale.toFixed(4) + ' permyriad/inch';
  document.getElementById('env_hint').textContent = 'CAMERA_SCALES=' + camUuid + ':' + scale.toFixed(6);
  document.getElementById('result').style.display = 'block';
  msg('Scale factor calculated!');
}

async function saveScale() {
  if (!scale) return;
  const r = await fetch('/calibrate/save', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({camera_uuid: camUuid, permyriad_per_inch: scale})
  });
  const d = await r.json();
  msg(d.error || 'Saved to server ✓ (persists until restart — also set env var to make permanent)');
}

function reset() {
  pts = []; scale = null;
  document.getElementById('p1txt').textContent = 'not set';
  document.getElementById('p2txt').textContent = 'not set';
  document.getElementById('result').style.display = 'none';
  document.querySelectorAll('.dot').forEach(d => d.remove());
  msg('');
}

function msg(t) { document.getElementById('msg').textContent = t; }
</script>
</body></html>"""


@app.route("/calibrate")
def calibrate():
    from flask import Response
    return Response(CALIBRATE_HTML, mimetype="text/html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


threading.Thread(target=_pending_sweeper, daemon=True, name="pending-sweeper").start()
threading.Thread(target=get_model, daemon=True, name="model-preload").start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
