"""car_damage: ML pipeline for detecting and classifying car damage from images.

Top-level package. Subpackages:

- ``data``       — dataset loading, transforms, split generation.
- ``models``     — ``production/`` (thin wrappers around library architectures,
                    rarely changes) and ``research/`` (experimental architectures
                    under active iteration). See models/__init__.py for the
                    hard boundary between the two.
- ``layers``     — quarantined, hand-rolled tensor-level building blocks used
                    only by ``models/research``. Never imported by production
                    code or the training loop.
- ``training``   — config-driven CLI entrypoint (train.py) plus a model-agnostic
                    train/val loop (loop.py) and loss functions (losses.py).
- ``evaluation`` — accuracy/quality metrics and an evaluation entrypoint.
                    Model-quality checks live here, not in tests/.
- ``inference``  — loading a trained checkpoint and running it on new images.

TODO: expose a stable public API here (e.g. package version) once the
individual modules are implemented. Keep this file minimal until then.
"""
