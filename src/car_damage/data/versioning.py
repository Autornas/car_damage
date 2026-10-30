"""
versioning.py

Content-addressed dataset version hash: sha256 over manifest.json + every
data/raw/<source>/annotations.json (real sources only -- see
manifest.list_source_dirs, which excludes build artifacts like
_merged_for_retrain) + split_registry.json, in a fixed sorted order so the
hash depends only on content, never on directory iteration order. This
ties a training run's checkpoint back to the exact data that produced it
(see scripts/snapshot_dataset.py and training/train_detector.py).

Allowed imports: standard library, car_damage.data.manifest,
car_damage.data.class_balance.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from car_damage.data.class_balance import summarize
from car_damage.data.manifest import load_manifest, list_source_dirs, DEFAULT_RAW_DIR, DEFAULT_MANIFEST_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPLIT_REGISTRY_PATH = PROJECT_ROOT / "data" / "split_registry.json"
VERSIONS_DIR = PROJECT_ROOT / "data" / "dataset_versions"


def _hashed_files(raw_dir=None, manifest_path=None):
    """
    Every file that determines dataset content, in a fixed order:
    manifest.json, then each real source's annotations.json (sorted by
    source name), then split_registry.json. A missing file (e.g.
    split_registry.json before Part 1 has ever run) is skipped, not an
    error -- the hash just reflects what exists right now.
    """
    manifest_path = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST_PATH
    paths = [manifest_path]
    paths += [p / "annotations.json" for p in list_source_dirs(raw_dir)]
    paths += [SPLIT_REGISTRY_PATH]
    return [p for p in paths if p.is_file()]


def compute_dataset_version(raw_dir=None, manifest_path=None):
    """Returns (full_hash, short_hash); short_hash (first 12 hex chars) is what gets used in filenames/logs."""
    hasher = hashlib.sha256()
    for path in _hashed_files(raw_dir, manifest_path):
        hasher.update(path.read_bytes())
    full_hash = hasher.hexdigest()
    return full_hash, full_hash[:12]


def build_snapshot(raw_dir=None, manifest_path=None):
    """
    Assembles the full snapshot dict written by scripts/snapshot_dataset.py
    (and copied into each experiment folder by train_detector.py): version
    hash, timestamp, per-source image counts (from manifest.json), and
    total box counts per damage_type (from class_balance.summarize --
    not reimplemented here).
    """
    full_hash, short_hash = compute_dataset_version(raw_dir, manifest_path)
    manifest = load_manifest(manifest_path)
    balance = summarize(raw_dir=raw_dir, manifest_path=manifest_path)

    return {
        "hash": full_hash,
        "short_hash": short_hash,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            name: {"image_count": info.get("image_count"), "box_count": info.get("box_count")}
            for name, info in manifest.items()
        },
        "box_counts_by_damage_type": balance["by_damage_type"],
        "total_images": balance["total_images"],
        "total_boxes": balance["total_boxes"],
    }


def snapshot_path_for(short_hash, date_str, versions_dir=None):
    versions_dir = Path(versions_dir) if versions_dir else VERSIONS_DIR
    return versions_dir / f"{short_hash}_{date_str}.json"


def write_snapshot(raw_dir=None, manifest_path=None, versions_dir=None):
    """
    Writes the snapshot to data/dataset_versions/<short_hash>_<date>.json
    and returns (snapshot_dict, path). data/dataset_versions/ is additive
    history, never replaced: if a snapshot with this exact hash already
    exists (any date), that existing file is reused as-is instead of
    writing a new/duplicate one; only genuinely new content gets a new
    file.
    """
    snapshot = build_snapshot(raw_dir, manifest_path)
    versions_dir = Path(versions_dir) if versions_dir else VERSIONS_DIR
    versions_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(versions_dir.glob(f"{snapshot['short_hash']}_*.json"))
    if existing:
        with open(existing[0], "r") as f:
            return json.load(f), existing[0]

    date_str = datetime.now(timezone.utc).date().isoformat()
    path = snapshot_path_for(snapshot["short_hash"], date_str, versions_dir)
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)
    return snapshot, path
