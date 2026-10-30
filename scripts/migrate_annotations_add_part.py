"""
migrate_annotations_add_part.py

CLI: python scripts/migrate_annotations_add_part.py

One-time (but safe to re-run) schema migration: adds an explicit
"part": "unknown" to every box across data/raw/<source>/annotations.json
(real sources only -- see car_damage.data.manifest.list_source_dirs,
which excludes build artifacts like _merged_for_retrain) that doesn't
already have a "part" field. Reuses car_damage.data.dataset's
DAMAGE_TYPE_TO_LABEL / PART_VALUES rather than duplicating the taxonomy.

Does NOT guess real part or damage_type values for any box -- "unknown"
is an explicit placeholder for "not yet reviewed," never a stand-in for
a real label. Manual review is what turns "unknown" into a real value,
not this script.

Backup + idempotency:
- Before the first write to a given annotations.json, its pre-migration
  content is copied to <name>.pre_part_migration.json. That backup is
  never overwritten by a later run -- it always reflects the file's
  state before ANY box in it had "part" added, even if this script runs
  many times as new boxes (e.g. new corrections) arrive later.
- A box that already has "part" is left untouched. Running this twice in
  a row with no new boxes in between changes nothing on the second run.

Allowed imports: standard library, car_damage.data.dataset,
car_damage.data.manifest.
"""
import json
import shutil

from car_damage.data.dataset import DAMAGE_TYPE_TO_LABEL, PART_VALUES
from car_damage.data.manifest import list_source_dirs

VALID_DAMAGE_TYPES = set(DAMAGE_TYPE_TO_LABEL.keys())
DEFAULT_PART = "unknown"


def migrate_file(annotations_path):
    """
    Returns a summary dict for one annotations.json:
    {"boxes_migrated", "boxes_already_had_part", "boxes_missing_damage_type",
     "invalid_damage_type_warnings": [...]}.
    """
    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    backup_path = annotations_path.with_name(annotations_path.stem + ".pre_part_migration.json")
    if not backup_path.is_file():
        shutil.copy2(annotations_path, backup_path)

    boxes_migrated = 0
    boxes_already_had_part = 0
    boxes_missing_damage_type = 0
    invalid_damage_type_warnings = []

    for filename, entry in annotations.items():
        for i, box in enumerate(entry.get("boxes", [])):
            if "part" not in box:
                box["part"] = DEFAULT_PART
                boxes_migrated += 1
            else:
                boxes_already_had_part += 1

            damage_type = box.get("damage_type")
            if damage_type is None:
                boxes_missing_damage_type += 1
            elif damage_type not in VALID_DAMAGE_TYPES:
                invalid_damage_type_warnings.append(
                    f"{annotations_path.parent.name}/{filename} box[{i}]: "
                    f"damage_type={damage_type!r} is not in the valid taxonomy {sorted(VALID_DAMAGE_TYPES)}"
                )

    if boxes_migrated:
        with open(annotations_path, "w") as f:
            json.dump(annotations, f, indent=2)

    return {
        "boxes_migrated": boxes_migrated,
        "boxes_already_had_part": boxes_already_had_part,
        "boxes_missing_damage_type": boxes_missing_damage_type,
        "invalid_damage_type_warnings": invalid_damage_type_warnings,
    }


def main():
    print(f"Valid part values ({len(PART_VALUES)}): {PART_VALUES}")
    print(f"Valid damage_type values ({len(VALID_DAMAGE_TYPES)}): {sorted(VALID_DAMAGE_TYPES)}\n")

    total_migrated = 0
    total_already_had_part = 0
    total_missing_damage_type = 0
    all_warnings = []

    for source_dir in list_source_dirs():
        annotations_path = source_dir / "annotations.json"
        summary = migrate_file(annotations_path)

        print(
            f"{source_dir.name}: {summary['boxes_migrated']} box(es) migrated "
            f"(part added), {summary['boxes_already_had_part']} already had part, "
            f"{summary['boxes_missing_damage_type']} still have no damage_type"
        )
        for warning in summary["invalid_damage_type_warnings"]:
            print(f"  WARNING: {warning}")

        total_migrated += summary["boxes_migrated"]
        total_already_had_part += summary["boxes_already_had_part"]
        total_missing_damage_type += summary["boxes_missing_damage_type"]
        all_warnings += summary["invalid_damage_type_warnings"]

    print(
        f"\nTotal: {total_migrated} box(es) migrated, {total_already_had_part} already had part, "
        f"{total_missing_damage_type} still have no damage_type, {len(all_warnings)} invalid-damage_type warning(s)"
    )


if __name__ == "__main__":
    main()
