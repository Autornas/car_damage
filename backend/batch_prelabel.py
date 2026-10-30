"""
batch_prelabel.py

CLI: python batch_prelabel.py --input_dir path/to/new_photos

Runs the phase 2 detector (inference_service.run_inference) over every
image in --input_dir and writes draft predictions to
data/raw/review_queue/queue.json, copying each source image into
data/raw/review_queue/images/ alongside it -- this is what the frontend's
review-queue mode (GET /queue/next, see queue_store.py) works through.

Imports inference_service directly rather than going through the running
API over HTTP, since this is a one-shot local batch job, not an API
client -- run it from the same environment the backend runs in (same
checkpoint/model requirements).

Allowed imports: inference_service, standard library.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import inference_service

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

_BACKEND_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BACKEND_DIR.parent

REVIEW_QUEUE_DIR = _PROJECT_ROOT / "data" / "raw" / "review_queue"
IMAGES_DIR = REVIEW_QUEUE_DIR / "images"
QUEUE_PATH = REVIEW_QUEUE_DIR / "queue.json"

# Matches /predict's own default threshold, so the boxes a reviewer sees
# pre-drawn are exactly what "Detect damage" would have shown them.
CONFIDENCE_THRESHOLD = 0.5

# Fixed uncertainty for a zero-box image. It's the theoretical floor of
# compute_uncertainty's formula (as if the model had produced one box at
# score exactly 0.5), so it sorts at or ahead of every box-having image:
# an empty prediction on a photo that might have damage is exactly the
# case most worth a human looking at.
EMPTY_PREDICTION_UNCERTAINTY = 0.0


def compute_uncertainty(boxes):
    """
    Mean of abs(0.5 - score) across an image's predicted boxes.

    Every box here already has score >= CONFIDENCE_THRESHOLD (0.5), so
    this value is naturally LOW when scores cluster near the 0.5 decision
    boundary (the model couldn't confidently call it either way) and HIGH
    when scores sit far from 0.5, near 1.0 (the model was confident). No
    extra sign flip is needed to get that direction -- it falls out of
    the formula as-is:

        LOWER uncertainty value == model was LESS confident on this image

    queue_store.get_next_pending() sorts ascending by this field --
    lowest first -- to surface the least-confident images first.
    """
    if not boxes:
        return EMPTY_PREDICTION_UNCERTAINTY
    return sum(abs(0.5 - b["score"]) for b in boxes) / len(boxes)


def gather_image_paths(input_dir):
    return sorted(
        p for p in Path(input_dir).iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, help="Folder of unlabeled images to pre-label")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"--input_dir not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    if not inference_service.is_model_loaded():
        print(f"Model is not loaded: {inference_service.get_load_error()}", file=sys.stderr)
        sys.exit(1)

    image_paths = gather_image_paths(input_dir)
    if not image_paths:
        print(f"No images found in {input_dir} (looked for {IMG_EXTENSIONS})", file=sys.stderr)
        sys.exit(1)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Merge into any existing queue rather than clobbering it, so re-running
    # this script (e.g. over an overlapping folder) doesn't reset review
    # progress already recorded on entries marked "reviewed".
    queue = {}
    if QUEUE_PATH.is_file():
        with open(QUEUE_PATH, "r") as f:
            queue = json.load(f)

    num_processed = 0
    num_with_boxes = 0
    uncertainties = []

    for path in image_paths:
        existing = queue.get(path.name)
        if existing is not None and existing.get("status") == "reviewed":
            continue  # already reviewed in a prior run -- leave it alone

        try:
            boxes, _width, _height = inference_service.run_inference(
                path.read_bytes(), confidence_threshold=CONFIDENCE_THRESHOLD
            )
        except Exception as exc:  # noqa: BLE001 -- one bad image shouldn't abort the whole batch
            print(f"Skipping {path.name}: inference failed ({exc})", file=sys.stderr)
            continue

        boxes = [
            {
                "x_min": round(b["x_min"]),
                "y_min": round(b["y_min"]),
                "x_max": round(b["x_max"]),
                "y_max": round(b["y_max"]),
                "score": b["score"],
            }
            for b in boxes
        ]
        uncertainty = compute_uncertainty(boxes)

        shutil.copy2(path, IMAGES_DIR / path.name)
        queue[path.name] = {
            "boxes": boxes,
            "uncertainty": uncertainty,
            "status": "pending",
        }

        num_processed += 1
        uncertainties.append(uncertainty)
        if boxes:
            num_with_boxes += 1

    with open(QUEUE_PATH, "w") as f:
        json.dump(queue, f, indent=2)

    mean_uncertainty = sum(uncertainties) / len(uncertainties) if uncertainties else 0.0
    print(f"Processed {num_processed} image(s) from {input_dir}")
    print(f"  {num_with_boxes} had at least one predicted box")
    print(f"  mean uncertainty: {mean_uncertainty:.4f}")
    print(f"Queue written to {QUEUE_PATH} ({len(queue)} total entries)")


if __name__ == "__main__":
    main()
