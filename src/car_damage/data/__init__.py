"""Data loading, augmentation, and split utilities for phase 1 (classifier)
and phase 2 (detector), plus the data-pipeline safeguards around phase 2's
raw sources: provenance (manifest.py), dedup (hashing.py), batch sanity
checks (validation.py), leakage-safe split assignment (split_registry.py),
class balance (class_balance.py), and dataset versioning (versioning.py).

Re-exports the public API of each so callers can do
``from car_damage.data import CarDamageDataset, get_train_transform``.
"""

from car_damage.data.dataset import (
    gather_filepaths,
    make_splits,
    CarDamageDataset,
    CarDamageDetectionDataset,
    detection_collate_fn,
    DAMAGE_TYPE_TO_LABEL,
    LABEL_TO_DAMAGE_TYPE,
    PART_VALUES,
)
from car_damage.data.transforms import (
    get_train_transform,
    get_eval_transform,
    get_detection_transform,
    IMG_SIZE,
    MEAN,
    STD,
    DETECTION_IMG_SIZE,
)
from car_damage.data.manifest import load_manifest, save_manifest, register_source, list_source_dirs
from car_damage.data.hashing import (
    compute_phash,
    hamming_distance,
    get_or_compute_phash,
    find_near_duplicate,
    find_duplicate_groups,
    load_hash_cache,
    save_hash_cache,
)
from car_damage.data.validation import validate_batch
from car_damage.data.split_registry import get_or_assign_split, lookup_split, load_registry, save_registry
from car_damage.data.class_balance import summarize as summarize_class_balance, summarize_source
from car_damage.data.versioning import compute_dataset_version, build_snapshot, write_snapshot

__all__ = [
    "gather_filepaths",
    "make_splits",
    "CarDamageDataset",
    "CarDamageDetectionDataset",
    "detection_collate_fn",
    "DAMAGE_TYPE_TO_LABEL",
    "LABEL_TO_DAMAGE_TYPE",
    "PART_VALUES",
    "get_train_transform",
    "get_eval_transform",
    "get_detection_transform",
    "IMG_SIZE",
    "MEAN",
    "STD",
    "DETECTION_IMG_SIZE",
    "load_manifest",
    "save_manifest",
    "register_source",
    "list_source_dirs",
    "compute_phash",
    "hamming_distance",
    "get_or_compute_phash",
    "find_near_duplicate",
    "find_duplicate_groups",
    "load_hash_cache",
    "save_hash_cache",
    "validate_batch",
    "get_or_assign_split",
    "lookup_split",
    "load_registry",
    "save_registry",
    "summarize_class_balance",
    "summarize_source",
    "compute_dataset_version",
    "build_snapshot",
    "write_snapshot",
]
