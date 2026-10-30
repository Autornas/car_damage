"""
inference_service.py

Loads the trained phase 2 detector ONCE at import time (not per-request)
and wraps it behind run_inference(image_bytes). Reuses car_damage's own
model-building (build_detector) and preprocessing (get_detection_transform)
rather than duplicating either -- this file is glue between the API layer
and the existing training/inference code, not a second implementation.

If the checkpoint can't be found or loaded, that's recorded rather than
raised: the API process should still come up so GET /health can report
model_loaded: false with a clear reason, instead of the whole server
crashing at import time. Allowed imports: car_damage.models.production,
car_damage.data.transforms, torch, PIL, numpy, standard library.
"""

import io
import logging
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from car_damage.data.transforms import DETECTION_IMG_SIZE, get_detection_transform
from car_damage.models.production.faster_rcnn_detector import build_detector

logger = logging.getLogger("car_damage_api")

# Resolved relative to this file, not the process's cwd, so the default
# works whether uvicorn is started from backend/ or elsewhere -- backend/
# is always a direct child of the project root.
_BACKEND_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BACKEND_DIR.parent
_DEFAULT_CHECKPOINT_PATH = _PROJECT_ROOT / "checkpoints" / "phase2_detector" / "detector_smoke_test.pt"
CHECKPOINT_PATH = Path(os.environ.get("CAR_DAMAGE_CHECKPOINT", str(_DEFAULT_CHECKPOINT_PATH)))

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_load_error = None
_transform = get_detection_transform()


def _load_model():
    global _model, _load_error

    if not CHECKPOINT_PATH.is_file():
        _load_error = (
            f"Checkpoint not found at {CHECKPOINT_PATH}. Train phase 2 first "
            f"(see configs/phase2_detector.yaml) or set CAR_DAMAGE_CHECKPOINT "
            f"to point at an existing checkpoint."
        )
        logger.error(_load_error)
        return

    try:
        model = build_detector()
        state_dict = torch.load(CHECKPOINT_PATH, map_location=_device)
        model.load_state_dict(state_dict)
        model.to(_device)
        model.eval()
        _model = model
        logger.info(f"Loaded detector checkpoint from {CHECKPOINT_PATH} onto {_device}")
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any load failure must not crash the server
        _load_error = f"Failed to load checkpoint at {CHECKPOINT_PATH}: {exc}"
        logger.exception(_load_error)


_load_model()


def is_model_loaded():
    return _model is not None


def get_load_error():
    return _load_error


def run_inference(image_bytes, confidence_threshold=0.5):
    """
    Decodes image_bytes, runs it through the same preprocessing phase 2
    training used (get_detection_transform), runs the model in eval mode
    under no_grad, and returns a list of {x_min, y_min, x_max, y_max,
    score} dicts for predictions at or above confidence_threshold --
    plus the original image's (width, height).

    Boxes are rescaled from the model's resized input space back to the
    ORIGINAL image's pixel coordinates before being returned, so a caller
    can draw them directly over the full-size image using image_width/
    image_height, with no knowledge of DETECTION_IMG_SIZE required.
    """
    if _model is None:
        raise RuntimeError(_load_error or "Model is not loaded")

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)
    orig_h, orig_w = image_np.shape[:2]

    # The detection transform requires every configured label field even
    # at inference. Empty lists are a no-op when there are no ground-truth
    # boxes, but keep Albumentations' bbox/label validation satisfied.
    transformed = _transform(image=image_np, bboxes=[], labels=[], part_labels=[])
    tensor = transformed["image"].to(_device)

    with torch.no_grad():
        prediction = _model([tensor])[0]

    scale_x = orig_w / DETECTION_IMG_SIZE
    scale_y = orig_h / DETECTION_IMG_SIZE

    pred_boxes = prediction["boxes"].cpu().numpy()
    pred_scores = prediction["scores"].cpu().numpy()

    boxes = []
    for (x_min, y_min, x_max, y_max), score in zip(pred_boxes, pred_scores):
        if score < confidence_threshold:
            continue
        boxes.append({
            "x_min": float(x_min * scale_x),
            "y_min": float(y_min * scale_y),
            "x_max": float(x_max * scale_x),
            "y_max": float(y_max * scale_y),
            "score": float(score),
        })

    boxes.sort(key=lambda b: b["score"], reverse=True)
    return boxes, orig_w, orig_h
