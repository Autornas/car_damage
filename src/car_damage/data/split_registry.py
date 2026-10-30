"""
split_registry.py

Single source of truth for which train/val/test split an image belongs
to, keyed by perceptual hash (not filename) so the same photo re-appearing
under a different filename from a different source is still recognized as
already-assigned and never lands in a second split -- that's the leakage
this module exists to prevent. Reuses car_damage.data.hashing's
Hamming-distance logic rather than its own image-similarity code.

Allowed imports: standard library, car_damage.data.hashing.
"""
import json
import logging
import random
from datetime import date
from pathlib import Path

from car_damage.data.hashing import hamming_distance, DEFAULT_DEDUP_THRESHOLD

logger = logging.getLogger("car_damage.data.split_registry")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "data" / "split_registry.json"

DEFAULT_TARGET_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
_NEAR_TARGET_MARGIN = 0.03  # within this many points of target -> weighted random instead of forcing


def load_registry(path=None):
    path = Path(path) if path else DEFAULT_REGISTRY_PATH
    if not path.is_file():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_registry(registry, path=None):
    path = Path(path) if path else DEFAULT_REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(registry, f, indent=2, sort_keys=True)
    tmp_path.replace(path)


def _running_counts(registry):
    counts = {"train": 0, "val": 0, "test": 0}
    for entry in registry.values():
        counts[entry["split"]] = counts.get(entry["split"], 0) + 1
    return counts


def _weighted_choice(target_ratios, rng):
    splits = list(target_ratios.keys())
    weights = [target_ratios[s] for s in splits]
    return rng.choices(splits, weights=weights, k=1)[0]


def _choose_split(registry, target_ratios, rng):
    counts = _running_counts(registry)
    total = sum(counts.values())
    current_ratios = {s: (counts.get(s, 0) / total if total else 0.0) for s in target_ratios}
    deficits = {s: target_ratios[s] - current_ratios[s] for s in target_ratios}

    worst_split = max(deficits, key=deficits.get)
    if deficits[worst_split] > _NEAR_TARGET_MARGIN:
        # clearly behind target -- assign it directly to pull the running ratio back
        return worst_split
    # already close to target overall -- weighted random using the target ratios
    return _weighted_choice(target_ratios, rng)


def _find_near_duplicate(phash, registry, threshold):
    best = None
    for existing_phash in registry:
        distance = hamming_distance(phash, existing_phash)
        if distance <= threshold and (best is None or distance < best[1]):
            best = (existing_phash, distance)
    return best


def get_or_assign_split(filename, phash, target_ratios=None, registry=None, path=None,
                         dedup_threshold=DEFAULT_DEDUP_THRESHOLD, persist=True, rng=None):
    """
    Returns (split, is_new) for `filename`/`phash`.

    - Exact phash already in the registry: returns its recorded split
      unchanged (is_new=False) -- a pure lookup, never touches
      target_ratios. This is what makes re-running registration against
      already-ingested data a no-op.
    - Near-duplicate (within `dedup_threshold` Hamming distance of some
      already-registered phash): reuses *that* entry's split, and also
      records the new phash under the same split (so an exact-hash lookup
      next time is instant) -- this is what stops the same vehicle/photo
      re-appearing from a different source (recompressed, re-cropped,
      re-uploaded, ...) from landing in a different split than its
      original. Also counted as is_new=False.
    - Genuinely new: assigned via the proportional target_ratios policy
      (see _choose_split), recorded, and returned (is_new=True). The
      image's own incoming "split" (if any, e.g. from an annotations.json
      being ingested) is intentionally NOT used here -- the registry
      decides splits from here on, that's the whole point of Part 1.

    Never reassigns a phash already present in the registry, regardless
    of how target_ratios drift -- leakage prevention beats perfect ratios.
    """
    target_ratios = target_ratios or DEFAULT_TARGET_RATIOS
    rng = rng or random
    owns_registry = registry is None
    if registry is None:
        registry = load_registry(path)

    if phash in registry:
        return registry[phash]["split"], False

    match = _find_near_duplicate(phash, registry, dedup_threshold)
    if match is not None:
        existing_phash, distance = match
        split = registry[existing_phash]["split"]
        logger.info(
            f"{filename}: phash {phash} is a near-duplicate (distance={distance}) of an "
            f"already-registered image -- reusing its split ({split!r}) instead of assigning a new one"
        )
        registry[phash] = {
            "split": split,
            "assigned_via": filename,
            "date_assigned": date.today().isoformat(),
        }
        if owns_registry and persist:
            save_registry(registry, path)
        return split, False

    split = _choose_split(registry, target_ratios, rng)
    registry[phash] = {
        "split": split,
        "assigned_via": filename,
        "date_assigned": date.today().isoformat(),
    }
    if owns_registry and persist:
        save_registry(registry, path)
    return split, True


def lookup_split(phash, registry=None, path=None):
    """Read-only: the registry's split for `phash`, or None if unregistered. Never assigns."""
    if registry is None:
        registry = load_registry(path)
    entry = registry.get(phash)
    return entry["split"] if entry else None
