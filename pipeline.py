"""
Rhombus Forklift Detection Pipeline

Watches one or more Rhombus cameras for vehicle motion alerts (MOTION_CAR),
downloads the alert thumbnail, runs YOLO forklift detection, and sends a
macOS notification + logs to CSV + creates a blue seekpoint and bounding box
on any hit.

Usage:
    python pipeline.py
    python pipeline.py --cameras "R410-A25K0532,R540-A25K0330" --profile i2M
    python pipeline.py --cameras "R410-A25K0532,R540-A25K0330" --profile i2M --poll 15 --threshold 0.80
"""

import argparse
import json
import subprocess
import time
import csv
import tempfile
from datetime import datetime
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


WEIGHTS  = Path(__file__).parent / "best.pt"
LOG_FILE = Path(__file__).parent / "detections.csv"


def notify(title: str, message: str):
    script = f'display notification "{message}" with title "{title}" sound name "Funk"'
    subprocess.run(["osascript", "-e", script], check=False)


def get_recent_vehicle_alerts(camera: str, profile: str, after: str):
    """Return list of (uuid, timestampMs, cameraUuid) for MOTION_CAR alerts since `after`."""
    result = subprocess.run(
        ["rhombus", "alert", "recent",
         "--camera", camera,
         "--profile", profile,
         "--after", after,
         "--max", "50",
         "--output", "json"],
        capture_output=True, text=True
    )
    try:
        data = json.loads(result.stdout)
        alerts = data.get("policyAlerts", [])
        return [
            (a["uuid"], a.get("timestampMs", 0), a.get("deviceUuid", ""))
            for a in alerts
            if "MOTION_CAR" in a.get("policyAlertTriggers", [])
        ]
    except Exception:
        return []


def download_thumbnail(alert_uuid: str, profile: str):
    """Download alert thumbnail and return path to temp file, or None on failure."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False)
    tmp.close()
    result = subprocess.run(
        ["rhombus", "alert", "thumb", alert_uuid,
         "--profile", profile,
         "--output", tmp.name],
        capture_output=True, text=True
    )
    path = Path(tmp.name)
    if path.exists() and path.stat().st_size > 0:
        return path
    return None


def run_detection(image_path: Path, model: YOLO, threshold: float):
    """Run YOLO detection and return list of (label, confidence, box_pixels) tuples.
    box_pixels is (x0, y0, x1, y1) in image pixel coordinates.
    """
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size
    results = model.predict(source=image, conf=threshold, verbose=False)[0]
    detections = []
    for b in results.boxes:
        label = model.names[int(b.cls[0])]
        conf  = float(b.conf[0])
        x0, y0, x1, y1 = b.xyxy[0].tolist()
        detections.append((label, conf, (x0, y0, x1, y1), img_w, img_h))
    return detections


def create_bounding_boxes(camera_uuid: str, timestamp_ms: int, detections: list, profile: str):
    """Push forklift bounding boxes to Rhombus in its 0-10000 coordinate scale."""
    boxes = []
    for i, (label, conf, (x0, y0, x1, y1), img_w, img_h) in enumerate(detections):
        boxes.append({
            "timestampMs": timestamp_ms,
            "activity":    "MOTION_CAR",
            "left":        int(x0 / img_w * 10000),
            "top":         int(y0 / img_h * 10000),
            "right":       int(x1 / img_w * 10000),
            "bottom":      int(y1 / img_h * 10000),
            "confidence":  round(conf, 4),
            "objectId":    i + 1,
            "inMotion":    True,
        })
    payload = {"cameraUuid": camera_uuid, "footageBoundingBoxes": boxes}
    result = subprocess.run(
        ["rhombus", "camera", "create-footage-bounding-boxes",
         "--profile", profile,
         "--cli-input-json", json.dumps(payload)],
        capture_output=True, text=True
    )
    try:
        return not json.loads(result.stdout).get("error", True)
    except Exception:
        return False


def create_seekpoint(camera_uuid: str, timestamp_ms: int, profile: str):
    """Create a blue 'Forklift Movement' seekpoint on the camera at the given timestamp."""
    payload = {
        "cameraUuid": camera_uuid,
        "footageSeekPoints": [{
            "timestampMs": timestamp_ms,
            "name": "Forklift Movement",
            "color": "BLUE"
        }]
    }
    result = subprocess.run(
        ["rhombus", "camera", "create-custom-footage-seekpoints",
         "--profile", profile,
         "--cli-input-json", json.dumps(payload)],
        capture_output=True, text=True
    )
    try:
        return not json.loads(result.stdout).get("error", True)
    except Exception:
        return False


def log_detection(camera: str, alert_uuid: str, detections: list, image_path: Path):
    write_header = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "camera", "alert_uuid", "label", "confidence", "image"])
        for label, conf, *_ in detections:
            writer.writerow([
                datetime.now().isoformat(),
                camera,
                alert_uuid,
                label,
                f"{conf:.1%}",
                str(image_path),
            ])


def main():
    parser = argparse.ArgumentParser(description="Rhombus vehicle-triggered forklift detection.")
    parser.add_argument("--cameras",   default="R410-A25K0532,R540-A25K0330", help="Comma-separated camera names or serials.")
    parser.add_argument("--profile",   default="i2M",           help="Rhombus CLI profile.")
    parser.add_argument("--poll",      type=int, default=20,    help="How often to check for new alerts in seconds (default: 20).")
    parser.add_argument("--threshold", type=float, default=0.80, help="Detection confidence threshold (default: 0.80).")
    args = parser.parse_args()

    cameras = [c.strip() for c in args.cameras.split(",")]

    print("Loading YOLO model...")
    model = YOLO(str(WEIGHTS))
    print("Model loaded.")
    print(f"\nStarting event-driven pipeline:")
    print(f"  Cameras   : {', '.join(cameras)}")
    print(f"  Profile   : {args.profile}")
    print(f"  Poll rate : every {args.poll}s")
    print(f"  Trigger   : MOTION_CAR alerts only")
    print(f"  Threshold : {args.threshold:.0%}")
    print(f"  Log file  : {LOG_FILE}")
    print(f"\nWaiting for vehicle motion alerts... (Ctrl+C to stop)\n")

    seen_alerts = set()

    while True:
        after = f"{args.poll + 5}s ago"
        any_new = False

        for camera in cameras:
            alerts = get_recent_vehicle_alerts(camera, args.profile, after)
            new_alerts = [(u, t, c) for u, t, c in alerts if u not in seen_alerts]
            seen_alerts.update(u for u, _, _ in alerts)

            for uuid, alert_ts_ms, camera_uuid in new_alerts:
                any_new = True
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{ts}] [{camera}] Vehicle alert {uuid[:8]}... — downloading thumbnail...", end=" ", flush=True)

                thumb = download_thumbnail(uuid, args.profile)
                if not thumb:
                    print("thumbnail unavailable, skipping.")
                    continue

                detections = run_detection(thumb, model, args.threshold)
                forklifts = [d for d in detections if d[0] == "forklift"]

                if forklifts:
                    summary = ", ".join(f"{c:.0%}" for _, c, *_ in forklifts)
                    print(f"FORKLIFT DETECTED ({summary})", end=" ", flush=True)
                    log_detection(camera, uuid, forklifts, thumb)
                    notify(
                        "Forklift Detected",
                        f"{camera}: {len(forklifts)} forklift(s) detected ({forklifts[0][1]:.0%} conf)"
                    )
                    if camera_uuid and alert_ts_ms:
                        sp_ok = create_seekpoint(camera_uuid, alert_ts_ms, args.profile)
                        bb_ok = create_bounding_boxes(camera_uuid, alert_ts_ms, forklifts, args.profile)
                        status = []
                        if sp_ok: status.append("seekpoint")
                        if bb_ok: status.append("bounding box")
                        print(f"— {' + '.join(status)} created." if status else "— annotations failed.")
                    else:
                        print()
                else:
                    print("no forklift.")

        if not any_new:
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] No new vehicle alerts.", end="\r", flush=True)

        time.sleep(args.poll)


if __name__ == "__main__":
    main()
