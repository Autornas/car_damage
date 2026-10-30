"""
validation.py

Pre-ingestion sanity checks for a data batch (images/ + annotations.json,
same layout as data/raw/phase2_pilot/), used by scripts/add_source.py
before a batch is merged into data/raw/. Catches batch-level mistakes
that would otherwise surface much later as a cryptic training crash (see
CarDamageDetectionDataset) or a silently wrong split.

Allowed imports: standard library, opencv-python (image dimension checks).
"""
import json
import os

import cv2

REQUIRED_BOX_KEYS = {"x_min", "y_min", "x_max", "y_max"}
VALID_SPLITS = {"train", "val", "test"}


def validate_batch(images_dir, annotations_path):
    """
    Returns {"valid": bool, "errors": [...], "warnings": [...]}.
    `errors` should block ingestion; `warnings` shouldn't.
    """
    images_dir = str(images_dir)
    annotations_path = str(annotations_path)

    if not os.path.isfile(annotations_path):
        return {"valid": False, "errors": [f"annotations.json not found at {annotations_path}"], "warnings": []}

    with open(annotations_path, "r") as f:
        try:
            annotations = json.load(f)
        except json.JSONDecodeError as exc:
            return {"valid": False, "errors": [f"annotations.json is not valid JSON: {exc}"], "warnings": []}

    if not os.path.isdir(images_dir):
        return {"valid": False, "errors": [f"images/ directory not found at {images_dir}"], "warnings": []}

    errors, warnings = [], []

    image_files = {f for f in os.listdir(images_dir) if not f.startswith(".")}
    annotated_files = set(annotations.keys())

    for filename in sorted(annotated_files - image_files):
        errors.append(f"{filename}: listed in annotations.json but no matching image file found")

    for filename in sorted(image_files - annotated_files):
        warnings.append(f"{filename}: image has no annotations.json entry (will be skipped)")

    for filename, entry in annotations.items():
        if filename not in image_files:
            continue  # already flagged above

        path = os.path.join(images_dir, filename)
        image = cv2.imread(path)
        if image is None:
            errors.append(f"{filename}: image file is unreadable/corrupt")
            continue
        height, width = image.shape[:2]

        split = entry.get("split")
        if split is not None and split not in VALID_SPLITS:
            errors.append(f"{filename}: invalid split {split!r} (expected one of {sorted(VALID_SPLITS)})")

        boxes = entry.get("boxes")
        if boxes is None:
            errors.append(f"{filename}: missing 'boxes' field")
            continue
        if len(boxes) == 0:
            warnings.append(f"{filename}: zero boxes (negative/no-damage example -- confirm this is intentional)")

        for i, box in enumerate(boxes):
            missing = REQUIRED_BOX_KEYS - box.keys()
            if missing:
                errors.append(f"{filename} box[{i}]: missing keys {sorted(missing)}")
                continue
            x_min, y_min, x_max, y_max = box["x_min"], box["y_min"], box["x_max"], box["y_max"]
            if x_min >= x_max or y_min >= y_max:
                errors.append(f"{filename} box[{i}]: degenerate box {box}")
            if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
                errors.append(f"{filename} box[{i}]: out of bounds {box} for a {width}x{height} image")

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}
