"""Model definitions, split into two subpackages with a hard boundary:

- ``production/`` — thin wrappers around well-established library
  architectures (e.g. torchvision ResNet). Code that rarely changes.
  MUST NOT import from ``research/`` or ``car_damage.layers``.

- ``research/``   — custom/experimental architectures under active
  iteration. MAY import from ``car_damage.layers`` for hand-rolled building
  blocks.

Every model in either subpackage should implement a standard ``forward()``
interface so ``training/loop.py`` can train any of them without knowing
which subpackage they came from.

TODO: as models are added, keep this boundary — do not let production/
reach into research/ or layers/ "just this once".
"""
