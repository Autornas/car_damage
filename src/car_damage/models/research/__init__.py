"""Research models: custom/experimental architectures under active iteration.

May depend on ``car_damage.layers``. Re-exports scratch_cnn.py's public API
so callers can do ``from car_damage.models.research import ScratchCNN``.
"""

from car_damage.models.research.scratch_cnn import ConvBlock, ScratchCNN, count_parameters

__all__ = ["ConvBlock", "ScratchCNN", "count_parameters"]
