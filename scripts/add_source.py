"""
add_source.py

CLI:
    python scripts/add_source.py --name <source_name> --batch-dir <path> [--description "..."]
    python scripts/add_source.py --register-existing <source_name> [--description "..."]

Ingests a batch of images+annotations.json into data/raw/<name>/, wiring
it into every other data-pipeline safeguard in one pass:

  1. validation.validate_batch()        -- structural sanity check
  2. hashing.get_or_compute_phash()     -- per-image dedup cache
  3. split_registry.get_or_assign_split() -- leakage-safe split assignment,
     reconciling against any "split" already baked into the incoming
     annotations.json (the registry always wins; a disagreement is logged,
     not silently ignored)
  4. manifest.register_source()         -- source attribution
  5. class_balance.summarize_source()   -- printed as a short summary
     (not the full report -- see scripts/class_balance_report.py for that)

--batch-dir copies a new external batch in; --register-existing wires up
a source that's already sitting under data/raw/<name>/ (e.g. phase2_pilot,
corrections -- both predate this script) without copying anything.

Allowed imports: standard library, car_damage.data.validation,
car_damage.data.hashing, car_damage.data.split_registry,
car_damage.data.manifest, car_damage.data.class_balance.
"""
import argparse
import json
import logging
import shutil
from pathlib import Path

from car_damage.data import validation, hashing, split_registry, class_balance
from car_damage.data import manifest as manifest_module
from car_damage.data.manifest import DEFAULT_RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("add_source")


def _print_validation_report(report):
    for warning in report["warnings"]:
        print(f"  WARNING: {warning}")
    for error in report["errors"]:
        print(f"  ERROR: {error}")


def _print_summary(summary):
    print(f"\nClass balance after ingest: {summary['total_images']} image(s), {summary['total_boxes']} box(es)")
    for warning in summary["warnings"]:
        print(f"  WARNING: {warning}")


def ingest_source(name, batch_dir=None, description=""):
    """
    Runs the full ingest pipeline for `name` and returns its class-balance
    summary. If batch_dir is given, copies its images/+annotations.json
    into data/raw/<name>/ first (data/raw/<name>/ must not already exist).
    If batch_dir is None, data/raw/<name>/ is assumed to already exist
    (the --register-existing path) and is validated/hashed/split-assigned/
    registered in place.
    """
    dest_dir = DEFAULT_RAW_DIR / name

    if batch_dir is not None:
        if dest_dir.exists():
            raise FileExistsError(f"{dest_dir} already exists -- pick a new source name or remove it first")
        batch_dir = Path(batch_dir)
        print(f"Validating batch at {batch_dir} ...")
        report = validation.validate_batch(batch_dir / "images", batch_dir / "annotations.json")
        _print_validation_report(report)
        if not report["valid"]:
            raise ValueError(f"Batch at {batch_dir} failed validation -- fix the errors above before ingesting")
        shutil.copytree(batch_dir / "images", dest_dir / "images")
        shutil.copy2(batch_dir / "annotations.json", dest_dir / "annotations.json")
        print(f"Copied batch into {dest_dir}")
    else:
        if not dest_dir.is_dir():
            raise FileNotFoundError(f"--register-existing {name!r} but {dest_dir} does not exist")
        print(f"Validating existing source at {dest_dir} ...")
        report = validation.validate_batch(dest_dir / "images", dest_dir / "annotations.json")
        _print_validation_report(report)
        if not report["valid"]:
            raise ValueError(f"{dest_dir} failed validation -- fix the errors above before registering")

    annotations_path = dest_dir / "annotations.json"
    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    hash_cache = hashing.load_hash_cache()
    registry = split_registry.load_registry()

    box_count = 0
    annotations_changed = False
    for filename, entry in sorted(annotations.items()):
        image_path = dest_dir / "images" / filename
        phash = hashing.get_or_compute_phash(str(image_path), filename, name, cache=hash_cache, persist=False)

        incoming_split = entry.get("split")
        resolved_split, is_new = split_registry.get_or_assign_split(
            filename, phash, registry=registry, persist=False,
        )
        if not is_new and incoming_split is not None and incoming_split != resolved_split:
            logger.warning(
                f"{filename}: annotations.json says split={incoming_split!r} but split_registry.json "
                f"(authoritative) has {resolved_split!r} for this image -- keeping {resolved_split!r}"
            )
        if entry.get("split") != resolved_split:
            entry["split"] = resolved_split
            annotations_changed = True

        box_count += len(entry.get("boxes", []))

    hashing.save_hash_cache(hash_cache)
    split_registry.save_registry(registry)
    if annotations_changed:
        with open(annotations_path, "w") as f:
            json.dump(annotations, f, indent=2)

    manifest_module.register_source(
        name, dest_dir, image_count=len(annotations), box_count=box_count, description=description,
    )

    summary = class_balance.summarize_source(dest_dir, name)
    _print_summary(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--batch-dir", help="Staging dir with images/+annotations.json to ingest as a new source")
    group.add_argument("--register-existing", metavar="NAME",
                        help="Register a source that already lives at data/raw/<NAME>/ (no copy)")
    parser.add_argument("--name", help="Source name (required with --batch-dir; becomes data/raw/<name>/)")
    parser.add_argument("--description", default="")
    args = parser.parse_args()

    if args.batch_dir:
        if not args.name:
            parser.error("--name is required with --batch-dir")
        ingest_source(args.name, batch_dir=args.batch_dir, description=args.description)
    else:
        ingest_source(args.register_existing, batch_dir=None, description=args.description)


if __name__ == "__main__":
    main()
