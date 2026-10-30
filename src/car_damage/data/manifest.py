"""
manifest.py

Registry of every ingested data source under data/raw/ -- one entry per
source directory (phase2_pilot, corrections, and any future batch added via
scripts/add_source.py), recording where it lives, when it was ingested,
and how many images/boxes it contributed. This is the single place other
tooling (versioning.py, class_balance.py) looks up "which source did this
image come from" rather than re-deriving it from directory names.

Also owns list_source_dirs(), the one definition of "what counts as an
ingested source" (has annotations.json, isn't a generated build artifact)
-- reused by class_balance.py, versioning.py, and scripts/find_duplicates.py
so that notion exists exactly once.

Allowed imports: standard library.
"""
import json
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"


def load_manifest(path=None):
    path = Path(path) if path else DEFAULT_MANIFEST_PATH
    if not path.is_file():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_manifest(manifest, path=None):
    path = Path(path) if path else DEFAULT_MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    tmp_path.replace(path)


def register_source(name, source_path, image_count, box_count, description="", path=None):
    """
    Creates or updates `name`'s entry. Re-registering an existing source
    (e.g. re-running add_source.py --register-existing after new
    corrections land) refreshes its counts/description/last_updated but
    keeps its original ingested_date.
    """
    manifest = load_manifest(path)
    existing = manifest.get(name, {})
    manifest[name] = {
        "path": str(source_path),
        "ingested_date": existing.get("ingested_date") or date.today().isoformat(),
        "last_updated": date.today().isoformat(),
        "image_count": image_count,
        "box_count": box_count,
        "description": description or existing.get("description", ""),
    }
    save_manifest(manifest, path)
    return manifest[name]


def list_source_dirs(raw_dir=None):
    """
    Every data/raw/<name>/ directory that looks like a real ingested
    source (has annotations.json) and isn't a generated build artifact --
    directories starting with "_" (e.g. _merged_for_retrain, rebuilt from
    scratch on every retrain by backend/dataset_merge.py) are never a
    source of truth and are always excluded here.
    """
    raw_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    if not raw_dir.is_dir():
        return []
    return sorted(
        p for p in raw_dir.iterdir()
        if p.is_dir() and not p.name.startswith("_") and (p / "annotations.json").is_file()
    )
