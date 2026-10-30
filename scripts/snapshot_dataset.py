"""
snapshot_dataset.py

CLI: python scripts/snapshot_dataset.py

Computes the current dataset version hash and writes/reuses
data/dataset_versions/<short_hash>_<date>.json (see
car_damage.data.versioning -- this is presentation only, no hashing logic
here). Prints the short hash so it can be referenced elsewhere (e.g.
pasted into an experiment's dataset_version.txt -- though
training/train_detector.py does that automatically on every run).
"""
from car_damage.data.versioning import write_snapshot


def main():
    snapshot, path = write_snapshot()
    print(f"Dataset version: {snapshot['short_hash']} (full: {snapshot['hash']})")
    print(f"Sources: {snapshot['sources']}")
    print(f"Total images: {snapshot['total_images']}  |  total boxes: {snapshot['total_boxes']}")
    print(f"Snapshot: {path}")


if __name__ == "__main__":
    main()
