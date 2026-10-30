"""Structural tests for ``car_damage.training``.

Checks that the training machinery runs and wires things together
correctly — not that the resulting model is any good.

TODO:
- Test that ``training.loop.run_training_loop`` can run for a single epoch
  on a tiny in-memory dataset + dummy model without raising, and that the
  returned history object has the expected keys (e.g. train_loss, val_loss
  per epoch).
- Test that the same ``run_training_loop`` call works with both a dummy
  "production-style" model and a dummy "research-style" model, verifying
  the loop really is model-agnostic.
- Test that ``training.losses.build_loss`` returns a callable for each
  supported ``training.loss`` value in the configs.
- Test that ``train.py``'s config loading reads all hyperparameters from
  the yaml file (no hardcoded fallbacks silently used).
"""

import torch

from car_damage.training.detection_loop import run_detection_epoch


class _DetectionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.target_keys = None

    def forward(self, images, targets):
        self.target_keys = set(targets[0])
        return {"loss": self.weight.square()}


def test_detection_epoch_does_not_pass_parts_to_model():
    model = _DetectionModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        "labels": torch.tensor([1]),
        "parts": ["hood"],
    }
    loader = [([torch.ones((3, 2, 2))], [target])]

    loss = run_detection_epoch(model, loader, optimizer, torch.device("cpu"))

    assert loss is not None
    assert model.target_keys == {"boxes", "labels"}
