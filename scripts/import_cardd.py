"""
Import the local CarDD COCO archives into data/raw/cardd_combined.

The full CarDD release is processed first. Photos from the Roboflow v3
export are added only when they are not a near-duplicate (dHash distance
<= 5) of an already-kept photo or an existing project source. Each kept
image appears once; CarDD release annotations take precedence for overlap.

Run from the car_damage project root:
    python scripts/import_cardd.py
"""
import json
import logging
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from car_damage.data.hashing import (
    DEFAULT_DEDUP_THRESHOLD,
    compute_phash,
    compute_phash_from_bytes,
    hamming_distance,
)
from car_damage.data.class_balance import summarize_source
from car_damage.data.manifest import DEFAULT_RAW_DIR, list_source_dirs
from car_damage.data.validation import validate_batch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_NEW_DATA_DIR = _PROJECT_ROOT.parent / "new_data"
DEFAULT_RELEASE_ZIP = _NEW_DATA_DIR / "CarDD_release.zip"
DEFAULT_ROBOFLOW_ZIP = _NEW_DATA_DIR / "Car Damage Severity Detection-CarDD.v3i.coco-mmdetection.zip"

_RELEASE_SPLITS = ("train", "val", "test")
_ROBOFLOW_SPLITS = ("train", "valid", "test")
_CATEGORY_MAPS = {
    "cardd_release": {
        "dent": ("dent", None),
        "scratch": ("scratch", None),
        "crack": ("crack", None),
        "glass shatter": ("shattered_glass", None),
        "lamp broken": ("missing_part", None),
        "tire flat": ("tire_damage", None),
    },
    "roboflow_v3": {
        "car-part-crack": ("crack", None),
        "detachment": ("missing_part", None),
        "flat-tire": ("tire_damage", None),
        "glass-crack": ("shattered_glass", None),
        "lamp-crack": ("crack", None),
        "minor-deformation": ("dent", "minor"),
        "moderate-deformation": ("dent", "moderate"),
        "paint-chips": ("paint_damage", None),
        "scratches": ("scratch", None),
        "severe-deformation": ("dent", "severe"),
        "side-mirror-crack": ("crack", None),
    },
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("import_cardd")


def _image_member(image_root, filename):
    relative = PurePosixPath(filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe COCO image path: {filename!r}")
    return f"{image_root}/{relative.as_posix()}"


def _category_lookup(coco, dataset_name):
    mapping = _CATEGORY_MAPS[dataset_name]
    result = {}
    for category in coco.get("categories", []):
        name = category["name"].strip().lower()
        if name not in mapping:
            raise ValueError(f"Unsupported {dataset_name} category: {category['name']!r}")
        result[category["id"]] = mapping[name]
    return result


def _annotations_by_image(coco, dataset_name):
    categories = _category_lookup(coco, dataset_name)
    grouped = defaultdict(list)
    for annotation in coco.get("annotations", []):
        if annotation.get("iscrowd", 0):
            continue
        category_id = annotation["category_id"]
        if category_id not in categories:
            raise ValueError(f"Unknown {dataset_name} category id: {category_id!r}")
        x_min, y_min, width, height = annotation["bbox"]
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid COCO bbox: {annotation['bbox']!r}")
        damage_type, severity = categories[category_id]
        box = {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_min + width,
            "y_max": y_min + height,
            "damage_type": damage_type,
        }
        if severity is not None:
            box["severity"] = severity
        grouped[annotation["image_id"]].append(box)
    return grouped


def _existing_hashes(raw_dir):
    hashes = []
    for source_dir in list_source_dirs(raw_dir):
        annotations_path = source_dir / "annotations.json"
        with annotations_path.open("r", encoding="utf-8") as f:
            annotations = json.load(f)
        for filename in annotations:
            image_path = source_dir / "images" / filename
            if image_path.is_file():
                hashes.append(compute_phash(image_path))
    return hashes


def _is_duplicate(phash, known_hashes, threshold):
    return any(hamming_distance(phash, known) <= threshold for known in known_hashes)


def _convert_split(
    archive,
    annotation_path,
    image_root,
    output_images,
    dataset_name,
    output_split,
    known_hashes,
    threshold,
):
    coco = json.loads(archive.read(annotation_path))
    annotations = _annotations_by_image(coco, dataset_name)
    image_count = 0
    box_count = 0
    duplicate_count = 0
    converted = {}

    for image in sorted(coco.get("images", []), key=lambda item: str(item["id"])):
        image_id = image["id"]
        member = _image_member(image_root, image["file_name"])
        image_bytes = archive.read(member)
        phash = compute_phash_from_bytes(image_bytes)
        if _is_duplicate(phash, known_hashes, threshold):
            duplicate_count += 1
            continue

        extension = PurePosixPath(image["file_name"]).suffix.lower() or ".jpg"
        filename = f"{dataset_name}_{output_split}_{image_id}{extension}"
        (output_images / filename).write_bytes(image_bytes)

        boxes = annotations.get(image_id, [])
        for box in boxes:
            x_min, y_min = box["x_min"], box["y_min"]
            x_max, y_max = box["x_max"], box["y_max"]
            if x_min < 0 or y_min < 0 or x_max > image["width"] or y_max > image["height"]:
                raise ValueError(
                    f"{member}: COCO box {box} is outside "
                    f"{image['width']}x{image['height']} image dimensions"
                )

        known_hashes.append(phash)
        image_count += 1
        box_count += len(boxes)
        converted[filename] = {"split": output_split, "boxes": boxes}

    return converted, {
        "images": image_count,
        "boxes": box_count,
        "duplicates_skipped": duplicate_count,
    }


def convert_coco_archives(release_zip, roboflow_zip, output_dir, existing_hashes=(),
                          threshold=DEFAULT_DEDUP_THRESHOLD):
    """Convert both archives to one validated images/ + annotations.json batch."""
    release_zip = Path(release_zip)
    roboflow_zip = Path(roboflow_zip)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if not release_zip.is_file() or not roboflow_zip.is_file():
        raise FileNotFoundError("Both the CarDD release and Roboflow v3 zip files are required")

    output_images = output_dir / "images"
    output_images.mkdir(parents=True)
    known_hashes = list(existing_hashes)
    merged = {}
    stats = {
        "cardd_release": {"images": 0, "boxes": 0, "duplicates_skipped": 0},
        "roboflow_v3": {"images": 0, "boxes": 0, "duplicates_skipped": 0},
    }

    with ZipFile(release_zip) as archive:
        for split in _RELEASE_SPLITS:
            annotation_path = (
                f"CarDD_release/CarDD_COCO/annotations/instances_{split}2017.json"
            )
            image_root = f"CarDD_release/CarDD_COCO/{split}2017"
            converted, split_stats = _convert_split(
                archive, annotation_path, image_root, output_images,
                "cardd_release", split, known_hashes, threshold,
            )
            merged.update(converted)
            for key, value in split_stats.items():
                stats["cardd_release"][key] += value

    with ZipFile(roboflow_zip) as archive:
        for split in _ROBOFLOW_SPLITS:
            output_split = "val" if split == "valid" else split
            annotation_path = f"{split}/_annotations.coco.json"
            converted, split_stats = _convert_split(
                archive, annotation_path, split, output_images,
                "roboflow_v3", output_split, known_hashes, threshold,
            )
            merged.update(converted)
            for key, value in split_stats.items():
                stats["roboflow_v3"][key] += value

    with (output_dir / "annotations.json").open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    return stats


def import_cardd(release_zip=DEFAULT_RELEASE_ZIP, roboflow_zip=DEFAULT_ROBOFLOW_ZIP,
                 name="cardd_combined"):
    """Convert, validate, and register both local archives as one dataset source."""
    raw_dir = DEFAULT_RAW_DIR
    destination = raw_dir / name
    if Path(name).name != name or name.startswith((".", "_")):
        raise ValueError("Source name must be a plain directory name and cannot start with '.' or '_'")
    if destination.exists():
        raise FileExistsError(f"{destination} already exists; refusing to overwrite an ingested source")
    raw_dir.mkdir(parents=True, exist_ok=True)
    known_hashes = _existing_hashes(raw_dir)

    with tempfile.TemporaryDirectory(prefix="_cardd_import_", dir=raw_dir) as temp_dir:
        batch_dir = Path(temp_dir) / "batch"
        stats = convert_coco_archives(
            release_zip, roboflow_zip, batch_dir, existing_hashes=known_hashes,
        )
        report = validate_batch(batch_dir / "images", batch_dir / "annotations.json")
        if not report["valid"]:
            details = "\n".join(report["errors"])
            raise ValueError(f"Converted CarDD batch failed validation:\n{details}")
        for warning in report["warnings"]:
            logger.warning(warning)
        batch_dir.replace(destination)

    description = (
        "CarDD COCO release plus non-duplicate images from the Roboflow v3 export; "
        "duplicates excluded at perceptual-hash distance <= "
        f"{DEFAULT_DEDUP_THRESHOLD}. Review each upstream source's license before redistribution."
    )
    subprocess.run(
        [
            sys.executable,
            str(_PROJECT_ROOT / "scripts" / "add_source.py"),
            "--register-existing",
            name,
            "--description",
            description,
        ],
        cwd=_PROJECT_ROOT,
        check=True,
    )
    summary = summarize_source(destination, name)
    return stats, summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-zip", type=Path, default=DEFAULT_RELEASE_ZIP)
    parser.add_argument("--roboflow-zip", type=Path, default=DEFAULT_ROBOFLOW_ZIP)
    parser.add_argument("--name", default="cardd_combined")
    args = parser.parse_args()

    stats, summary = import_cardd(args.release_zip, args.roboflow_zip, args.name)
    for source, values in stats.items():
        print(
            f"{source}: {values['images']} unique image(s), {values['boxes']} box(es), "
            f"{values['duplicates_skipped']} duplicate image(s) skipped"
        )
    print(
        f"Registered data/raw/{args.name}: "
        f"{summary['total_images']} images and {summary['total_boxes']} boxes."
    )


if __name__ == "__main__":
    main()
