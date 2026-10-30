"""Loss function definitions, selected by name from config.

Allowed imports: torch.

TODO:
- ``build_loss(cfg) -> Callable``: given ``training.loss`` from a
  ``configs/*.yaml`` file, return the corresponding loss callable (e.g.
  ``nn.CrossEntropyLoss`` for phase 1 classification; a combined
  classification + localization loss for phase 2 detection).
- Keep loss selection entirely config-driven — no hardcoded loss choice in
  ``train.py``.
"""
