"""
transforms.py

Phase 1 — image preprocessing / augmentation pipelines (Albumentations) for
the classifier, plus the image size and ImageNet normalization stats that
car_damage.models.production.resnet_classifier's pretrained backbone
expects.

Phase 2 — get_detection_transform() below, for the detector. Unlike the
phase 1 transforms, it must move the bounding boxes along with the image
(via Albumentations' bbox_params), since a resize that only touches pixels
would leave box coordinates pointing at the wrong place in the resized
image.

Allowed imports: albumentations. Must not import from car_damage.models or
car_damage.layers.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_SIZE = 224

# ImageNet normalization stats -- required since we're using an ImageNet-pretrained backbone
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_train_transform():
    """Augmentations applied only during training -- helps generalization."""
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.ISONoise(p=0.2),  # simulates phone camera sensor noise
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_eval_transform():
    """No augmentation for val/test/inference -- just resize + normalize."""
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


DETECTION_IMG_SIZE = 224


def get_detection_transform():
    """
    Preprocessing for the phase 2 detector. Resizes the image AND rescales
    the bounding boxes to match, via bbox_params(format="pascal_voc") --
    pascal_voc means boxes as [x_min, y_min, x_max, y_max] in pixel
    coordinates, matching annotations.json and what torchvision's
    detection models expect directly. clip=True guards against a box edge
    landing a fraction outside the resized image due to rounding.

    Only scales pixels to [0, 1] (mean=0/std=1 is a no-op except for the
    implicit /255) rather than applying the phase 1 ImageNet MEAN/STD --
    torchvision's detection models (e.g. fasterrcnn_resnet50_fpn) already
    normalize internally with their own stats, so normalizing here too
    would double-normalize. No augmentation yet: this pilot pass only
    needs to confirm images/boxes wire together correctly, not generalize.

    Two label_fields, not one: "labels" (the damage_type class per box,
    fed to the model) and "part_labels" (the descriptive part-of-car per
    box, never fed to the model -- see CarDamageDetectionDataset). Both
    ride along with "bboxes" so Albumentations keeps all three in sync if
    a box gets dropped/reordered (e.g. clip=True dropping one that lands
    fully outside the resized image); passing "part_labels" as a plain
    Python list alongside instead would silently desync from "bboxes" the
    moment that happens.
    """
    return A.Compose(
        [
            A.Resize(DETECTION_IMG_SIZE, DETECTION_IMG_SIZE),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels", "part_labels"], clip=True),
    )
