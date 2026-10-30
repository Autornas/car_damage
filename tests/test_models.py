"""Structural tests for ``car_damage.models`` (production and research).

Checks output shapes and config loading only — not accuracy, which belongs
in ``car_damage.evaluation``.

TODO:
- For each model (``models.production.resnet_classifier.ResNetClassifier``,
  ``models.research.scratch_cnn.ScratchCNN``): build it from a small dummy
  config, run a forward pass on a random input tensor of the expected input
  shape, and assert the output shape matches ``(batch_size, num_classes)``.
- Test that both production and research models satisfy the same
  ``forward()`` interface so ``training.loop.run_training_loop`` can accept
  either one interchangeably (e.g. by running both through the same tiny
  smoke-test helper).
- Test that ``configs/phase1_classifier.yaml`` / ``phase2_detector.yaml``
  load via pyyaml and contain the expected top-level keys
  (``model``, ``data``, ``training``, ``experiment``).
"""
