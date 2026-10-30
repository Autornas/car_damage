"""
dataset.py

Phase 1 (classification) — handles:
  - scanning data/damaged and data/not_damaged for images
  - splitting into train/val/test (by image, since each image is one vehicle photo)
  - a PyTorch Dataset that loads + augments images

Expected folder layout:
    <data_dir>/
        damaged/
            img001.jpg
            img002.jpg
            ...
        not_damaged/
            img101.jpg
            ...

Phase 2 (detection) — CarDamageDetectionDataset below reads a data_dir
containing images/ + annotations.json (see data/raw/phase2_pilot/README.md
for the exact format and this pilot batch's provenance/caveats) and
returns torchvision-detection-ready (image, target) pairs.

Allowed imports: standard library, cv2, sklearn, torch,
car_damage.data.hashing, car_damage.data.split_registry. Must not import
from car_damage.models or car_damage.layers.
"""

import json
import logging
import os
import cv2
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from car_damage.data.hashing import load_hash_cache
from car_damage.data.split_registry import load_registry

logger = logging.getLogger("car_damage.data.dataset")

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# Per-box damage_type taxonomy -> the detector's classification label.
# torchvision detection models reserve label 0 for background, so
# num_classes for build_detector (see
# car_damage.models.production.faster_rcnn_detector) must be
# len(DAMAGE_TYPE_TO_LABEL) + 1.
#
# 1-7 are the original taxonomy; 8-13 were appended for the part-field
# migration (scripts/migrate_annotations_add_part.py) -- never renumber an
# existing entry, only append new ones, or every checkpoint trained against
# the old numbering silently mismatches its labels.
DAMAGE_TYPE_TO_LABEL = {
    "scratch": 1,
    "dent": 2,
    "crack": 3,
    "shattered_glass": 4,
    "missing_part": 5,
    "misalignment": 6,
    "unknown": 7,
    "rust": 8,
    "paint_damage": 9,
    "hail_damage": 10,
    "tire_damage": 11,
    "graffiti": 12,
    "burn_damage": 13,
}
LABEL_TO_DAMAGE_TYPE = {label: damage_type for damage_type, label in DAMAGE_TYPE_TO_LABEL.items()}

# Which part of the car a box's damage is on. A descriptive attribute, not
# a detection class -- exposed alongside each box (see
# CarDamageDetectionDataset.__getitem__'s target["parts"]) but never fed
# into the labels tensor. "tire" (the rubber) and "wheel" (the rim) are
# kept as separate values on purpose: damage to one doesn't imply damage
# to the other, even on the same corner of the car.
PART_VALUES = [
    "front_bumper", "rear_bumper", "hood", "roof", "trunk",
    "left_fender", "right_fender",
    "left_front_door", "right_front_door", "left_rear_door", "right_rear_door",
    "windshield", "rear_windshield",
    "left_headlight", "right_headlight", "left_taillight", "right_taillight",
    "left_mirror", "right_mirror",
    "left_front_tire", "right_front_tire", "left_rear_tire", "right_rear_tire",
    "left_front_wheel", "right_front_wheel", "left_rear_wheel", "right_rear_wheel",
    "unknown",
]


def gather_filepaths(data_dir):
    """
    Walks data_dir/damaged and data_dir/not_damaged.
    Returns (filepaths, labels) where label 1 = damaged, 0 = not_damaged.
    """
    filepaths, labels = [], []

    for label_name, label_value in [("damaged", 1), ("not_damaged", 0)]:
        folder = os.path.join(data_dir, label_name)
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Expected folder not found: {folder}\n"
                f"Make sure your data is organized as data/damaged/ and data/not_damaged/"
            )
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith(IMG_EXTENSIONS):
                filepaths.append(os.path.join(folder, fname))
                labels.append(label_value)

    if len(filepaths) == 0:
        raise ValueError(f"No images found under {data_dir}. Add images before training.")

    return filepaths, labels


def make_splits(data_dir, val_size=0.15, test_size=0.15, seed=42):
    """
    Returns dicts of {split_name: (filepaths, labels)}.
    Stratified by label so damaged/not_damaged ratio stays consistent across splits.
    """
    filepaths, labels = gather_filepaths(data_dir)

    # First split off test set
    train_val_fp, test_fp, train_val_lb, test_lb = train_test_split(
        filepaths, labels, test_size=test_size, random_state=seed, stratify=labels
    )

    # Then split remaining into train/val
    val_relative_size = val_size / (1 - test_size)
    train_fp, val_fp, train_lb, val_lb = train_test_split(
        train_val_fp, train_val_lb, test_size=val_relative_size,
        random_state=seed, stratify=train_val_lb
    )

    return {
        "train": (train_fp, train_lb),
        "val": (val_fp, val_lb),
        "test": (test_fp, test_lb),
    }


class CarDamageDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        image = cv2.imread(path)
        if image is None:
            raise RuntimeError(f"Failed to read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label


class CarDamageDetectionDataset(Dataset):
    """
    Phase 2 detection dataset. Reads <data_dir>/annotations.json and
    <data_dir>/images/, filtered to entries whose "split" field matches
    `split` -- splits are respected as assigned in the annotation file,
    never re-split randomly, so train/val membership stays exactly what
    the data provider set.

    __getitem__ returns (image_tensor, target) where target is a dict with
    "boxes" (FloatTensor[N, 4], [x_min, y_min, x_max, y_max]), "labels"
    (Int64Tensor[N], each box's damage_type mapped through
    DAMAGE_TYPE_TO_LABEL -- this is exactly the format torchvision's
    detection models, e.g. fasterrcnn_resnet50_fpn, expect as a target
    dict, no further conversion needed by callers), and "parts" (a plain
    list[str] of PART_VALUES, one per box, in the same order as "boxes"/
    "labels" -- NOT a tensor and never fed into the model, since part is a
    descriptive attribute rather than a detection class). A box with no
    "damage_type"/"part" key at all (most boxes today -- see
    scripts/migrate_annotations_add_part.py, which deliberately never
    backfills real values) falls back to "unknown" for each, same as an
    explicit "unknown" value would.

    Split membership: data/split_registry.json (see
    car_damage.data.split_registry) is the authoritative source for which
    split an image belongs to, keyed by perceptual hash so the same photo
    arriving from two different sources can't land in two splits. For each
    image, this looks up its phash in data/image_hashes.json and, if that
    phash is registered, uses the registry's split -- logging a warning if
    it disagrees with this annotations.json's own "split" field. Images
    with no cache/registry entry (e.g. fixture data in tests, or data
    never run through scripts/add_source.py) fall back to their
    annotations.json "split" field untouched.
    """

    def __init__(self, data_dir, split, transform=None):
        self.images_dir = os.path.join(data_dir, "images")
        self.transform = transform

        annotations_path = os.path.join(data_dir, "annotations.json")
        with open(annotations_path, "r") as f:
            annotations = json.load(f)

        hash_cache = load_hash_cache()
        registry = load_registry()

        self.samples = []
        for filename, entry in annotations.items():
            effective_split = entry["split"]
            phash = hash_cache.get(filename, {}).get("phash")
            if phash is not None and phash in registry:
                registry_split = registry[phash]["split"]
                if entry["split"] != registry_split:
                    logger.warning(
                        f"{filename}: annotations.json says split={entry['split']!r} but "
                        f"split_registry.json (authoritative) has {registry_split!r} for this "
                        f"image -- using {registry_split!r}"
                    )
                effective_split = registry_split

            if effective_split == split:
                self.samples.append((filename, entry["boxes"]))

        if len(self.samples) == 0:
            raise ValueError(f"No samples found for split={split!r} in {annotations_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, boxes = self.samples[idx]
        path = os.path.join(self.images_dir, filename)
        image = cv2.imread(path)
        if image is None:
            raise RuntimeError(f"Failed to read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        box_list = [[b["x_min"], b["y_min"], b["x_max"], b["y_max"]] for b in boxes]
        labels = [DAMAGE_TYPE_TO_LABEL.get(b.get("damage_type"), DAMAGE_TYPE_TO_LABEL["unknown"]) for b in boxes]
        parts = [b.get("part", "unknown") for b in boxes]

        if self.transform:
            # part_labels rides along as a second Albumentations label_field
            # (see get_detection_transform) so it stays aligned with boxes/
            # labels if the transform drops or reorders a box (e.g. clipped
            # fully outside the resized image) -- a plain parallel list
            # would silently desync in that case.
            augmented = self.transform(image=image, bboxes=box_list, labels=labels, part_labels=parts)
            image = augmented["image"]
            box_list = augmented["bboxes"]
            labels = augmented["labels"]
            parts = augmented["part_labels"]

        target = {
            "boxes": torch.as_tensor(box_list, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "parts": list(parts),
        }
        return image, target


def detection_collate_fn(batch):
    """
    torchvision detection models take a list of variable-size images and a
    list of variable-length target dicts directly -- they must NOT be
    stacked into a single tensor/dict the way DataLoader's default
    collate_fn would try to (that fails since each image can have a
    different number of boxes). Pass this as DataLoader(..., collate_fn=
    detection_collate_fn) for CarDamageDetectionDataset.
    """
    return tuple(zip(*batch))
