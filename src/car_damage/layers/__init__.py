"""Quarantined package of hand-rolled neural network building blocks.

Nothing outside ``car_damage.models.research`` should depend on this
package. Re-exports scratch_layers.py's public API so callers can do
``from car_damage.layers import Conv2dScratch``.
"""

from car_damage.layers.scratch_layers import (
    Conv2dScratch,
    BatchNorm2dScratch,
    ReLUScratch,
    MaxPool2dScratch,
    GlobalAvgPoolScratch,
    LinearScratch,
    DropoutScratch,
)

__all__ = [
    "Conv2dScratch",
    "BatchNorm2dScratch",
    "ReLUScratch",
    "MaxPool2dScratch",
    "GlobalAvgPoolScratch",
    "LinearScratch",
    "DropoutScratch",
]
