"""
corrections_store.py

Persists user-submitted corrections (edited/added/deleted boxes) as new
phase 2 training data, in the SAME annotations.json format read by
CarDamageDetectionDataset -- with an added "source": "user_correction"
field per image entry so a correction is distinguishable from the
original pilot batch. A correction is useless without the image it
corrected, so the uploaded image itself is copied into
data/raw/corrections/images/ alongside the annotation entry.

Strictly additive: never reads or writes data/raw/phase2_pilot/.

Allowed imports: standard library.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger("car_damage_api")

_BACKEND_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BACKEND_DIR.parent

CORRECTIONS_DIR = _PROJECT_ROOT / "data" / "raw" / "corrections"
IMAGES_DIR = CORRECTIONS_DIR / "images"
ANNOTATIONS_PATH = CORRECTIONS_DIR / "annotations.json"


def _load_annotations():
    if not ANNOTATIONS_PATH.is_file():
        return {}
    with open(ANNOTATIONS_PATH, "r") as f:
        return json.load(f)


def _write_annotations(annotations):
    # Write to a temp file and replace atomically so a crash mid-write
    # can't leave annotations.json truncated/corrupt.
    ANNOTATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = ANNOTATIONS_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(annotations, f, indent=2)
    tmp_path.replace(ANNOTATIONS_PATH)


def save_correction(image_bytes, filename, boxes):
    """
    Writes image_bytes to data/raw/corrections/images/<filename> and
    appends/updates that filename's entry in
    data/raw/corrections/annotations.json with `boxes` (a list of
    {x_min, y_min, x_max, y_max} dicts, already validated by the caller)
    and "source": "user_correction". Re-saving a correction for a
    filename that was already corrected overwrites its prior entry and
    image -- last edit wins.

    `boxes` may be an empty list: a legitimate "no damage here" negative
    example, not an error.

    Returns the new total correction count.
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    image_path = IMAGES_DIR / filename
    with open(image_path, "wb") as f:
        f.write(image_bytes)

    annotations = _load_annotations()
    annotations[filename] = {
        "split": "train",
        "boxes": boxes,
        "source": "user_correction",
    }
    _write_annotations(annotations)

    logger.info(f"Saved correction for {filename!r} ({len(boxes)} box(es))")
    return get_correction_count()


def get_correction_count():
    """Number of distinct images that have a saved correction."""
    return len(_load_annotations())
