"""
dataset_merge.py

Combines data/raw/phase2_pilot/ (the original pilot batch) and
data/raw/corrections/ (user-submitted corrections), plus every other
registered data source, into a single
images/ + annotations.json dataset that CarDamageDetectionDataset can
read directly, for the manual retrain trigger (see retrain_service.py).

The merged directory is regenerated from scratch on every call so it
always reflects the current set of corrections -- it's a build artifact,
never a source of truth, and nothing but train_detector reads it.

Allowed imports: standard library, car_damage.data.manifest.
"""

import json
import shutil
from pathlib import Path

from car_damage.data.manifest import list_source_dirs

_BACKEND_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BACKEND_DIR.parent

CORRECTIONS_DIR = _PROJECT_ROOT / "data" / "raw" / "corrections"


def merge_datasets(output_dir):
    """
    Rebuilds output_dir as images/ + annotations.json from every registered
    source under data/raw. Sources without annotations.json are ignored.

    Corrections are processed last, so when a correction's filename matches
    an existing source image its boxes replace the original ones. The
    original image's split is retained because corrections_store.py stamps
    every correction as "train"; replacing a val/test split would corrupt
    evaluation.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    images_out = output_dir / "images"
    images_out.mkdir(parents=True)

    merged = {}
    source_dirs = list_source_dirs(_PROJECT_ROOT / "data" / "raw")
    source_dirs.sort(key=lambda path: (path == CORRECTIONS_DIR, path.name))
    for source_dir in source_dirs:
        annotations_path = source_dir / "annotations.json"
        if not annotations_path.is_file():
            continue
        with open(annotations_path, "r") as f:
            entries = json.load(f)

        images_dir = source_dir / "images"
        for filename, entry in entries.items():
            if filename in merged and source_dir != CORRECTIONS_DIR:
                raise ValueError(
                    f"Image filename collision while merging {source_dir.name}: {filename!r}"
                )
            shutil.copy2(images_dir / filename, images_out / filename)
            if source_dir == CORRECTIONS_DIR and filename in merged:
                entry = {**entry, "split": merged[filename]["split"]}
            if source_dir is CORRECTIONS_DIR and filename in merged:
                entry = {**entry, "split": merged[filename]["split"]}
            merged[filename] = entry

    with open(output_dir / "annotations.json", "w") as f:
        json.dump(merged, f, indent=2)

    return output_dir
