"""Training package: phase 1's entrypoint (train.py) + model-agnostic epoch
loop (loop.py), and phase 2's entrypoint (train_detector.py) + detection
epoch loop (detection_loop.py). Re-exports both loops so callers can do
``from car_damage.training import run_epoch, run_detection_epoch``.
"""

from car_damage.training.loop import run_epoch
from car_damage.training.detection_loop import run_detection_epoch

__all__ = ["run_epoch", "run_detection_epoch"]
