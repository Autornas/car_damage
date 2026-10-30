"""Production models: thin wrappers around library architectures.

Code here rarely changes. MUST NOT import from
``car_damage.models.research`` or ``car_damage.layers``.

Re-exports resnet_classifier.py (phase 1) and faster_rcnn_detector.py
(phase 2)'s public APIs so callers can do
``from car_damage.models.production import build_model, build_detector``.
"""

from car_damage.models.production.resnet_classifier import build_model, unfreeze_last_block
from car_damage.models.production.faster_rcnn_detector import build_detector

__all__ = ["build_model", "unfreeze_last_block", "build_detector"]
