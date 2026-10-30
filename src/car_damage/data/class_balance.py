"""
class_balance.py

Counts boxes per damage_type/severity and images/boxes per source across
data/raw/*/annotations.json, cross-referenced against manifest.json for
source attribution. Shared by scripts/class_balance_report.py (the full
report), scripts/add_source.py (a one-source summary printed right after
ingestion), and versioning.py (per-run box counts in a dataset snapshot)
-- so the counting logic itself lives here exactly once.

Every box's "damage_type", "part", and "severity" fields are optional
today (see data/raw/phase2_pilot/README.md -- the pilot batch predates
any damage taxonomy and only has generic boxes); missing values are
counted as "unknown" / "none/unset" rather than dropped, so a report
never silently under-counts a source that hasn't been re-annotated with
the taxonomy yet.

Allowed imports: standard library, car_damage.data.manifest.
"""
import json
from collections import Counter
from pathlib import Path

from car_damage.data.manifest import load_manifest, list_source_dirs, DEFAULT_RAW_DIR

MIN_BOXES_FOR_TRAINING = 20
UNKNOWN_DAMAGE_TYPE = "unknown"
UNKNOWN_PART = "unknown"
UNSET_SEVERITY = "none/unset"


def _iter_sources(raw_dir, source=None):
    raw_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    if source:
        path = raw_dir / source
        candidates = [path] if (path / "annotations.json").is_file() else []
    else:
        candidates = list_source_dirs(raw_dir)
    for path in candidates:
        yield path.name, path / "annotations.json"


def summarize(raw_dir=None, source=None, manifest_path=None):
    """
    Returns:
    {
      "by_damage_type": {type: count, ...},
      "by_part": {part: count, ...},
      "by_severity": {severity: count, ...},
      "by_source": {source_name: {"image_count": n, "box_count": n, "in_manifest": bool}, ...},
      "total_images": n,
      "total_boxes": n,
      "warnings": [...],
    }
    """
    manifest = load_manifest(manifest_path)

    by_damage_type = Counter()
    by_part = Counter()
    by_severity = Counter()
    by_source = {}
    total_images = 0

    for source_name, annotations_path in _iter_sources(raw_dir, source):
        with open(annotations_path, "r") as f:
            annotations = json.load(f)

        image_count = len(annotations)
        box_count = 0
        for entry in annotations.values():
            boxes = entry.get("boxes", [])
            box_count += len(boxes)
            for box in boxes:
                by_damage_type[box.get("damage_type", UNKNOWN_DAMAGE_TYPE)] += 1
                by_part[box.get("part", UNKNOWN_PART)] += 1
                by_severity[box.get("severity", UNSET_SEVERITY)] += 1

        by_source[source_name] = {
            "image_count": image_count,
            "box_count": box_count,
            "in_manifest": source_name in manifest,
        }
        total_images += image_count

    total_boxes = sum(by_damage_type.values())

    warnings = []
    for damage_type, count in sorted(by_damage_type.items()):
        if damage_type != UNKNOWN_DAMAGE_TYPE and count < MIN_BOXES_FOR_TRAINING:
            warnings.append(
                f"damage_type {damage_type!r}: only {count} box(es) (<{MIN_BOXES_FOR_TRAINING}) "
                f"-- insufficient for training"
            )
    unknown_damage_count = by_damage_type.get(UNKNOWN_DAMAGE_TYPE, 0)
    if unknown_damage_count:
        warnings.append(
            f"{unknown_damage_count} box(es) have no damage_type recorded (taxonomy not backfilled onto this "
            f"data yet) -- this is not a real class distribution, just missing labels; per-type counts above "
            f"only cover boxes that already have a damage_type"
        )

    for part, count in sorted(by_part.items()):
        if part != UNKNOWN_PART and count < MIN_BOXES_FOR_TRAINING:
            warnings.append(
                f"part {part!r}: only {count} box(es) (<{MIN_BOXES_FOR_TRAINING}) -- insufficient for training"
            )
    unknown_part_count = by_part.get(UNKNOWN_PART, 0)
    if unknown_part_count:
        warnings.append(
            f"{unknown_part_count} box(es) have no part recorded (part field not backfilled onto this data "
            f"yet) -- this is not a real distribution, just missing labels; per-part counts above only cover "
            f"boxes that already have a part"
        )

    return {
        "by_damage_type": dict(by_damage_type),
        "by_part": dict(by_part),
        "by_severity": dict(by_severity),
        "by_source": by_source,
        "total_images": total_images,
        "total_boxes": total_boxes,
        "warnings": warnings,
    }


def summarize_source(source_dir, source_name, manifest_path=None):
    """Convenience wrapper for add_source.py: summarize just one already-ingested source."""
    return summarize(raw_dir=Path(source_dir).parent, source=source_name, manifest_path=manifest_path)
