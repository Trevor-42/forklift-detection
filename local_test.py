"""
Local integration test for webhook_server.

Runs the full webhook pipeline (parse → fetch → YOLO → write) against
real Rhombus APIs using your local ~/.rhombus/certs/i2M/ cert.

Usage:
    # Use the most recent real tu=True payload from Cloud Run logs:
    python local_test.py --auto

    # Or provide an event UUID + timestamp:
    python local_test.py --event-uuid <uuid> --ts <ms> --camera <uuid>

    # Dry-run — fetch + detect but don't write back to Rhombus:
    python local_test.py --auto --dry-run

    # Skip the fetch and run detection on a local image you already have:
    python local_test.py --frame /tmp/frame.jpeg --camera <uuid> --ts <ms> --dry-run

Environment:
    RHOMBUS_API_KEY must be set (or exported in your shell)
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

# Override cert paths before importing webhook_server
HOME_CERT = Path.home() / ".rhombus/certs/i2M/client.crt"
HOME_KEY  = Path.home() / ".rhombus/certs/i2M/client.key"

if not HOME_CERT.exists() or not HOME_KEY.exists():
    sys.exit(f"Missing local certs at {HOME_CERT.parent}. Run `rhombus login --profile i2M` first.")

os.environ.setdefault("RHOMBUS_API_KEY", "REDACTED")

# Patch the hardcoded cert paths inside webhook_server
import webhook_server as ws
ws.CERT_FILE = HOME_CERT
ws.KEY_FILE  = HOME_KEY


def fetch_recent_payload():
    """Pull the most recent tu=True webhook payload from Cloud Run logs."""
    cmd = [
        "gcloud", "logging", "read",
        'resource.type=cloud_run_revision AND resource.labels.service_name=forklift-detection '
        'AND textPayload:"Webhook received" AND textPayload:"seekpoints"',
        "--project", "forklift-detection-i2m",
        "--limit", "5", "--format", "value(textPayload)",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    for line in out.splitlines():
        if "Webhook received: " in line and '"seekpoints":' in line:
            payload_json = line.split("Webhook received: ", 1)[1]
            return json.loads(payload_json)
    return None


def run(payload: dict, dry_run: bool, frame_override: Path | None = None):
    cam, ev, ts, region, tu, loc = ws.parse_payload(payload)
    print(f"parsed  camera={cam}  event={ev}  ts={ts}  region={region}  tu_present={bool(tu)}  loc={loc}")

    if frame_override:
        thumb = frame_override
        print(f"using local frame: {thumb}")
    else:
        thumb = ws.get_frame(cam, ts, ev or "", region, tu)
    if not thumb:
        print("❌ no frame")
        return
    size = thumb.stat().st_size
    print(f"✅ got frame: {thumb}  ({size} bytes)")

    detections = ws.run_detection(thumb)
    forklifts = [d for d in detections if d[0] == "forklift"]
    print(f"detections={len(detections)}  forklifts={len(forklifts)}")
    for i, (label, conf, x0, y0, x1, y1, w, h) in enumerate(detections):
        print(f"  [{i}] {label} {conf:.3f} ({x0:.0f},{y0:.0f})-({x1:.0f},{y1:.0f}) on {w}x{h}")

    if not forklifts:
        print("→ no forklift, pipeline ends here")
        return

    best = max(f[1] for f in forklifts)
    if dry_run:
        print(f"[DRY RUN] would write seekpoint + {len(forklifts)} bbox(es) @ {best:.1%}")
        return

    sp = ws.create_seekpoint(cam, ts, best, loc)
    bb = ws.create_bounding_boxes(cam, ts, forklifts)
    print(f"seekpoint: {sp}")
    print(f"bboxes:    {bb}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", action="store_true", help="Use most recent tu=True payload from Cloud Run logs")
    ap.add_argument("--event-uuid")
    ap.add_argument("--ts", type=int)
    ap.add_argument("--camera")
    ap.add_argument("--dry-run", action="store_true", help="Skip writes to Rhombus")
    ap.add_argument("--frame", help="Path to a local JPEG to run detection on (skips fetch)")
    ap.add_argument("--location", default="", help="Location UUID (used with --frame)")
    args = ap.parse_args()

    frame_path = Path(args.frame) if args.frame else None
    if frame_path and not frame_path.exists():
        sys.exit(f"Frame not found: {frame_path}")

    if args.auto:
        p = fetch_recent_payload()
        if not p:
            sys.exit("No recent tu=True webhook found in Cloud Run logs")
        print(f"using live payload eventUuid={p['deviceEvents'][0].get('eventUuid')}")
    elif args.camera and args.ts:
        p = {"deviceEvents": [{"deviceUuid": f"{args.camera}.v0",
                               "eventUuid": args.event_uuid or "",
                               "timestampMs": args.ts,
                               "thumbnailLocation": {"region": "us-east-2"},
                               "locationUuid": args.location,
                               "seekpoints": []}]}
    else:
        ap.error("Pass --auto, or --camera+--ts (with optional --frame, --event-uuid)")

    run(p, args.dry_run, frame_path)
