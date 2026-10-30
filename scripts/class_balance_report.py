"""
class_balance_report.py

CLI: python scripts/class_balance_report.py [--source <name>]

Prints a class-balance table across data/raw/*/annotations.json (or one
source) and writes data/class_balance_report.json with the same data, so
later runs can be diffed against it. Counting logic lives in
car_damage.data.class_balance -- this is presentation only.
"""
import argparse
import json
from datetime import datetime, timezone

from car_damage.data.class_balance import summarize
from car_damage.data.manifest import PROJECT_ROOT

OUTPUT_PATH = PROJECT_ROOT / "data" / "class_balance_report.json"


def print_report(report, source=None):
    title = f"Class balance ({source})" if source else "Class balance (all sources)"
    print(f"\n=== {title} ===")
    print(f"\nTotal images: {report['total_images']}  |  total boxes: {report['total_boxes']}")

    print("\nBy source:")
    for name, stats in sorted(report["by_source"].items()):
        flag = "" if stats["in_manifest"] else "  (not in manifest.json)"
        print(f"  {name:<20} images={stats['image_count']:<5} boxes={stats['box_count']:<5}{flag}")

    print("\nBy damage_type:")
    for damage_type, count in sorted(report["by_damage_type"].items(), key=lambda kv: -kv[1]):
        print(f"  {damage_type:<20} {count}")

    print("\nBy part:")
    for part, count in sorted(report["by_part"].items(), key=lambda kv: -kv[1]):
        print(f"  {part:<20} {count}")

    print("\nBy severity:")
    for severity, count in sorted(report["by_severity"].items(), key=lambda kv: -kv[1]):
        print(f"  {severity:<20} {count}")

    if report["warnings"]:
        print("\nWarnings:")
        for warning in report["warnings"]:
            print(f"  - {warning}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None, help="Only report on data/raw/<source> (default: all sources)")
    args = parser.parse_args()

    report = summarize(source=args.source)
    print_report(report, source=args.source)

    output = dict(report)
    output["generated_at"] = datetime.now(timezone.utc).isoformat()
    output["source_filter"] = args.source
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
