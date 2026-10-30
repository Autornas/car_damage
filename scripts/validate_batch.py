"""
validate_batch.py

CLI: python scripts/validate_batch.py <batch_dir>
(expects <batch_dir>/images/ + <batch_dir>/annotations.json, same layout
as data/raw/phase2_pilot/)

Presentation only -- all the actual checks live in
car_damage.data.validation.validate_batch, also used directly by
scripts/add_source.py before ingesting a batch.
"""
import argparse
import sys
from pathlib import Path

from car_damage.data.validation import validate_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", help="Directory containing images/ and annotations.json")
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir)
    report = validate_batch(batch_dir / "images", batch_dir / "annotations.json")

    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    for error in report["errors"]:
        print(f"ERROR: {error}")

    if report["valid"]:
        print(f"OK: batch at {batch_dir} is valid ({len(report['warnings'])} warning(s)).")
    else:
        print(f"INVALID: {len(report['errors'])} error(s) found in {batch_dir}.")
        sys.exit(1)


if __name__ == "__main__":
    main()
