"""
faster_rcnn_detector.py

Thin wrapper around torchvision's Faster R-CNN (ResNet50 + FPN backbone,
COCO-pretrained) for phase 2 damage detection. Follows the same pattern as
phase 1's resnet_classifier.py: adapt a well-established architecture to
this project's output shape, nothing more. This code rarely changes.

Allowed imports: torch, torchvision. Must not import from
car_damage.models.research or car_damage.layers.
"""

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_detector(num_classes=14):
    """
    Faster R-CNN (ResNet50-FPN backbone) pretrained on COCO, with the box
    predictor head replaced for `num_classes` classes. torchvision
    detection models reserve class 0 for background, so num_classes=14
    (the default) means 13 real classes -- one per
    car_damage.data.dataset.DAMAGE_TYPE_TO_LABEL entry (scratch=1 through
    burn_damage=13) -- plus background.

    IMPORTANT: changing num_classes changes the shape of the final
    classification layer (roi_heads.box_predictor), so a checkpoint saved
    from a build_detector() call with a *different* num_classes cannot be
    load_state_dict()'d into this one -- torch will raise a size-mismatch
    error, not silently misalign classes. There is no automatic
    conversion between checkpoints trained at different num_classes; a
    checkpoint trained before this taxonomy was extended must be
    retrained from scratch against the new label mapping, not loaded here.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
