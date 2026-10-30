"""Train/val/test split generation and persistence.

Produces deterministic split files under ``data/splits/`` (e.g. CSV/JSON
listing image ids per split) so ``dataset.py`` always loads the same split
given the same split file, and every experiment run is reproducible.

Allowed imports: standard library, scikit-learn (for stratified splitting),
pyyaml. Must not import from ``car_damage.models``.

TODO:
- ``make_splits(processed_dir, splits_dir, val_frac, test_frac, seed,
  stratify_by=None)``: scans ``data/processed/`` labels, produces a
  stratified (by damage class) train/val/test partition, writes it to
  ``data/splits/``.
- ``load_split(splits_dir, split_name) -> list[...]``: reads a previously
  generated split file back in for use by ``dataset.py``.
- Splits should be seeded and versioned (e.g. filename includes a split
  version/date) so re-running training against an old split is still
  possible.
"""
