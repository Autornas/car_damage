"""Structural tests for ``car_damage.data.dataset``.

Checks shapes/types/alignment only — not model or data *quality*, which is
out of scope for tests/ (see ``car_damage.evaluation`` for that).

TODO:
- Test that ``CarDamageDataset.__getitem__`` returns an image tensor of the
  expected shape/dtype (e.g. ``(C, H, W)`` float tensor) for a given
  transform/config.
- Test that the label returned for each sample matches the split file's
  label for that sample id (label alignment — no off-by-one / shuffling
  bugs between images and labels).
- Test that ``__len__`` matches the number of entries in the split file.
- Test on a small fixture split, not the real dataset in ``data/raw/``.
"""
