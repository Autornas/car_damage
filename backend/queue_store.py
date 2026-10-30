"""
queue_store.py

Reads/writes data/raw/review_queue/queue.json (produced by
batch_prelabel.py) to drive the frontend's review-queue mode: which
pre-labeled image to show next (least confident first) and how far
through the queue review has gotten.

Allowed imports: standard library.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger("car_damage_api")

_BACKEND_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BACKEND_DIR.parent

REVIEW_QUEUE_DIR = _PROJECT_ROOT / "data" / "raw" / "review_queue"
IMAGES_DIR = REVIEW_QUEUE_DIR / "images"
QUEUE_PATH = REVIEW_QUEUE_DIR / "queue.json"


def queue_exists():
    return QUEUE_PATH.is_file()


def _load_queue():
    with open(QUEUE_PATH, "r") as f:
        return json.load(f)


def _write_queue(queue):
    tmp_path = QUEUE_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(queue, f, indent=2)
    tmp_path.replace(QUEUE_PATH)


def get_next_pending():
    """
    Returns (filename, entry) for the pending queue entry with the LOWEST
    "uncertainty" value -- the least-confident image, reviewed first (see
    batch_prelabel.compute_uncertainty for why lowest means least
    confident). Returns None if there's no eligible pending entry
    (including when queue.json doesn't exist yet).

    An entry whose image file is missing from IMAGES_DIR is skipped (and
    logged) rather than handed to the frontend as an unloadable image.
    """
    if not queue_exists():
        return None

    queue = _load_queue()
    candidates = []
    for filename, entry in queue.items():
        if entry.get("status") != "pending":
            continue
        if not (IMAGES_DIR / filename).is_file():
            logger.warning(f"Skipping queue entry {filename!r}: image file missing from {IMAGES_DIR}")
            continue
        candidates.append((filename, entry))

    if not candidates:
        return None

    return min(candidates, key=lambda item: item[1]["uncertainty"])


def mark_reviewed(filename):
    """
    Sets `filename`'s queue entry status to "reviewed". Returns False
    (no-op) if `filename` isn't in the queue at all -- e.g. a correction
    saved for an image that never went through the review queue -- so
    callers can tell whether a review was actually completed.
    """
    if not queue_exists():
        return False

    queue = _load_queue()
    if filename not in queue:
        return False

    queue[filename]["status"] = "reviewed"
    _write_queue(queue)
    logger.info(f"Marked {filename!r} as reviewed")
    return True


def get_queue_stats():
    if not queue_exists():
        return {"pending": 0, "reviewed": 0, "total": 0}

    queue = _load_queue()
    reviewed = sum(1 for e in queue.values() if e.get("status") == "reviewed")
    total = len(queue)
    return {"pending": total - reviewed, "reviewed": reviewed, "total": total}
