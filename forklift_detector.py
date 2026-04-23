"""
Forklift detector — supports two backends:

  1. Grounding DINO (zero-shot, no weights needed, lower confidence)
  2. YOLOv8 (fine-tuned on forklifts, higher confidence, needs .pt weights)

Install dependencies:
    pip install torch torchvision transformers Pillow requests ultralytics

── Grounding DINO usage ──────────────────────────────────────────────────────
    python forklift_detector.py --image photo.jpg
    python forklift_detector.py --image photo.jpg --threshold 0.25

── YOLO usage ────────────────────────────────────────────────────────────────
    python forklift_detector.py --image photo.jpg --detector yolo --weights forklift.pt

    Get free forklift weights from Roboflow Universe:
      https://universe.roboflow.com/search?q=class:forklift+trained+model
    Open any model → Versions → Export → YOLOv8 PyTorch → download the .pt file.
"""

import argparse
from io import BytesIO
from pathlib import Path

import requests
import torch
from PIL import Image, ImageDraw, ImageFont


MODEL_ID   = "IDEA-Research/grounding-dino-base"
PROMPT     = "forklift."
DEFAULT_THRESHOLD = 0.80


# ── image loading ─────────────────────────────────────────────────────────────

def load_image(source: str) -> Image.Image:
    if source.startswith("http://") or source.startswith("https://"):
        r = requests.get(source, timeout=15)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    return Image.open(source).convert("RGB")


# ── Grounding DINO backend ────────────────────────────────────────────────────

def load_dino():
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    print("Loading Grounding DINO model (downloads once, cached after)...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on {device}.")
    return processor, model, device


def detect_dino(image: Image.Image, processor, model, device, threshold: float):
    inputs = processor(images=image, text=PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=threshold,
        text_threshold=threshold,
        target_sizes=[image.size[::-1]],
    )[0]
    boxes  = results["boxes"].cpu().tolist()
    scores = results["scores"].cpu().tolist()
    labels = [str(l) for l in results["labels"]]
    return boxes, scores, labels


# ── YOLO backend ──────────────────────────────────────────────────────────────

def load_yolo(weights: str):
    from ultralytics import YOLO
    print(f"Loading YOLO model from: {weights}")
    model = YOLO(weights)
    print("YOLO model loaded.")
    return model


def detect_yolo(image: Image.Image, model, threshold: float):
    results = model.predict(source=image, conf=threshold, verbose=False)[0]
    boxes, scores, labels = [], [], []
    names = model.names
    for box in results.boxes:
        x0, y0, x1, y1 = box.xyxy[0].tolist()
        boxes.append([x0, y0, x1, y1])
        scores.append(float(box.conf[0]))
        labels.append(names[int(box.cls[0])])
    return boxes, scores, labels


# ── box merging ───────────────────────────────────────────────────────────────

def merge_boxes(boxes, scores, labels):
    """Union overlapping boxes of the same label into a single bounding box."""
    if len(boxes) == 0:
        return boxes, scores, labels

    merged_boxes, merged_scores, merged_labels = [], [], []
    used = [False] * len(boxes)

    for i in range(len(boxes)):
        if used[i]:
            continue
        group_boxes  = [boxes[i]]
        group_scores = [scores[i]]
        used[i] = True

        for j in range(i + 1, len(boxes)):
            if used[j] or labels[j] != labels[i]:
                continue
            xi0, yi0, xi1, yi1 = boxes[i]
            xj0, yj0, xj1, yj1 = boxes[j]
            if max(0, min(xi1, xj1) - max(xi0, xj0)) > 0 and \
               max(0, min(yi1, yj1) - max(yi0, yj0)) > 0:
                group_boxes.append(boxes[j])
                group_scores.append(scores[j])
                used[j] = True

        merged_boxes.append([
            min(b[0] for b in group_boxes),
            min(b[1] for b in group_boxes),
            max(b[2] for b in group_boxes),
            max(b[3] for b in group_boxes),
        ])
        merged_scores.append(max(group_scores))
        merged_labels.append(labels[i])

    return merged_boxes, merged_scores, merged_labels


# ── drawing ───────────────────────────────────────────────────────────────────

def draw_detections(image: Image.Image, boxes, scores, labels) -> Image.Image:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=18)
    except Exception:
        font = ImageFont.load_default()

    for box, score, label in zip(boxes, scores, labels):
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
        caption = f"{label} {score:.0%}"
        draw.rectangle([x0, y0 - 22, x0 + len(caption) * 11, y0], fill="red")
        draw.text((x0 + 2, y0 - 20), caption, fill="white", font=font)

    return image


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Detect forklifts using Grounding DINO or YOLOv8.")
    parser.add_argument("--image",     required=True,  help="Path or URL to the input image.")
    parser.add_argument("--detector",  default="dino", choices=["dino", "yolo"],
                        help="Detection backend: 'dino' (default, zero-shot) or 'yolo' (fine-tuned, higher confidence).")
    parser.add_argument("--weights",   default=None,
                        help="[YOLO only] Path to .pt weights file downloaded from Roboflow Universe.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Confidence threshold (default: {DEFAULT_THRESHOLD}).")
    parser.add_argument("--output",    default=None,
                        help="Output image path (default: <input>_detected.<ext>).")
    args = parser.parse_args()

    if args.detector == "yolo" and not args.weights:
        parser.error(
            "--weights is required for YOLO.\n"
            "Download a free forklift .pt file from:\n"
            "  https://universe.roboflow.com/search?q=class:forklift+trained+model\n"
            "Then run:  python forklift_detector.py --detector yolo --weights forklift.pt --image photo.jpg"
        )

    # Output path
    if args.output:
        output_path = Path(args.output)
    elif args.image.startswith("http"):
        output_path = Path("forklift_detected.jpg")
    else:
        p = Path(args.image)
        output_path = p.parent / f"{p.stem}_detected{p.suffix}"

    print(f"Loading image: {args.image}")
    image = load_image(args.image)

    # Detect
    if args.detector == "yolo":
        model = load_yolo(args.weights)
        boxes, scores, labels = detect_yolo(image, model, args.threshold)
    else:
        processor, model, device = load_dino()
        boxes, scores, labels = detect_dino(image, processor, model, device, args.threshold)

    # Merge overlapping boxes
    if len(boxes) > 1:
        before = len(boxes)
        boxes, scores, labels = merge_boxes(boxes, scores, labels)
        if len(boxes) < before:
            print(f"Merged {before} raw detections → {len(boxes)} object(s).")

    # Report
    if len(boxes) == 0:
        print("No forklifts detected. Try lowering --threshold.")
    else:
        print(f"Detected {len(boxes)} forklift(s):")
        for i, (score, label) in enumerate(zip(scores, labels), 1):
            print(f"  [{i}] {label} — confidence: {score:.1%}")

    annotated = draw_detections(image, boxes, scores, labels)
    annotated.save(output_path)
    print(f"Annotated image saved to: {output_path}")


if __name__ == "__main__":
    main()
