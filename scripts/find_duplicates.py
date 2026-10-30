"""
find_duplicates.py

CLI: python scripts/find_duplicates.py [--source <name>] [--threshold N]

Computes/caches a perceptual hash for every image under data/raw/<source>
(or every real source -- see car_damage.data.manifest.list_source_dirs --
when --source is omitted) into data/image_hashes.json, then reports
near-duplicate groups. Presentation only -- hashing and grouping logic
lives in car_damage.data.hashing, also used directly by
split_registry.get_or_assign_split() and scripts/add_source.py.

Allowed imports: standard library, car_damage.data.hashing,
car_damage.data.manifest.
"""
import argparse

from car_damage.data import hashing
from car_damage.data.manifest import list_source_dirs, DEFAULT_RAW_DIR


def scan_source(source_dir, source_name, cache):
    images_dir = source_dir / "images"
    if not images_dir.is_dir():
        return
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            continue
        hashing.get_or_compute_phash(
            str(image_path), image_path.name, source_name, cache=cache, persist=False,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None, help="Only scan data/raw/<source> (default: every real source)")
    parser.add_argument("--threshold", type=int, default=hashing.DEFAULT_DEDUP_THRESHOLD,
                         help=f"Max Hamming distance to call two images near-duplicates (default: {hashing.DEFAULT_DEDUP_THRESHOLD})")
    args = parser.parse_args()

    cache = hashing.load_hash_cache()

    if args.source:
        sources = [(DEFAULT_RAW_DIR / args.source, args.source)]
    else:
        sources = [(p, p.name) for p in list_source_dirs()]

    for source_dir, name in sources:
        before = len(cache)
        scan_source(source_dir, name, cache)
        print(f"{name}: {len(cache) - before} new image(s) hashed")

    hashing.save_hash_cache(cache)

    groups = hashing.find_duplicate_groups(cache, threshold=args.threshold)
    if not groups:
        print(f"\nNo near-duplicates found (threshold={args.threshold}).")
    else:
        print(f"\nFound {len(groups)} near-duplicate group(s) (threshold={args.threshold}):")
        for group in groups:
            print("  " + ", ".join(group))


if __name__ == "__main__":
    main()
