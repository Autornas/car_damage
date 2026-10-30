# car_damage

Car damage detection ML pipeline. Phase 1 — a binary classifier (damaged
vs. not_damaged) — is implemented and migrated in from an earlier flat
prototype. Later phases (bounding-box detection, damage type
classification, segmentation, part localization) build on this same
scaffold.

## Layout

```
car_damage/
├── configs/              # All hyperparameters live here as YAML, never hardcoded in code
├── data/
│   ├── raw/               # Untouched source images/labels
│   ├── processed/         # Cleaned/preprocessed data ready for dataset.py
│   └── splits/            # train/val/test split files (see src/car_damage/data/splits.py)
├── src/car_damage/
│   ├── data/               # Dataset, transforms, split generation
│   ├── models/
│   │   ├── production/     # Thin wrappers around library architectures (e.g. torchvision). Rarely changes.
│   │   └── research/       # Custom/experimental architectures under active iteration.
│   ├── layers/              # Hand-rolled building blocks (raw tensor ops), used only by research/
│   ├── training/            # train.py (CLI entrypoint), loop.py (model-agnostic), losses.py
│   ├── evaluation/          # metrics.py, evaluate.py — accuracy/quality checks live here, not in tests/
│   └── inference/           # predict.py — run a trained checkpoint on new images
├── experiments/            # One timestamped folder per run (config + checkpoint + curves + metrics). Gitignored.
├── notebooks/              # Exploration only. Nothing else may import from here.
├── scripts/                # One-off/utility scripts (data prep, dataset stats, etc.)
└── tests/                  # Structural tests only (shapes, label alignment, config loading)
```

## Setup

```bash
pip install -r requirements.txt
```

## 1. Add your data

Training datasets and local image collections in `data/` and `Nuotraukos/`
are intentionally excluded from Git; add your data locally after cloning.

`src/car_damage/data/dataset.py`'s `gather_filepaths()` expects:

```
<data_dir>/
  damaged/
    img001.jpg
    img002.jpg
    ...
  not_damaged/
    img101.jpg
    ...
```

`<data_dir>` is the `data_dir` value in `configs/phase1_classifier.yaml`
(default: `data`, i.e. `car_damage/data/damaged` and
`car_damage/data/not_damaged` directly — not under `data/raw/`).

You need at least a few hundred images per class to get meaningful results;
1000+ per class is much better. Good sources: the CarDD dataset, Kaggle car
damage datasets, Roboflow Universe, or your own photos.

Tips for good data:
- Include a variety of angles, lighting conditions, and car colors.
- For "not_damaged", include clean cars in varied settings, not just one type.
- Avoid duplicate/near-duplicate images across classes (leaks into eval scores).

## 2. Train

All hyperparameters live in `configs/phase1_classifier.yaml` — `train.py`
takes only a `--config` path, nothing is hardcoded:

```bash
python -m car_damage.training.train --config configs/phase1_classifier.yaml
```

What happens:
1. Data is split 70/15/15 into train/val/test (stratified by label).
2. Stage 1 trains only a new classifier head on a frozen ResNet18 backbone
   (`models/production/resnet_classifier.py`) — fast, good with limited data.
3. Stage 2 unfreezes the last conv block and fine-tunes with a lower
   learning rate for `finetune_epochs` more epochs.
4. The best checkpoint (by validation accuracy) is saved to
   `<checkpoint_dir>/best_model.pt`.
5. Training curves are saved to `<checkpoint_dir>/training_curves.png`.
6. A final evaluation on the held-out test set prints precision/recall/F1
   and a confusion matrix (`src/car_damage/evaluation/`).

Useful config keys (`configs/phase1_classifier.yaml`):
- `lr` — learning rate for stage 1 (default `0.001`)
- `finetune_lr` — learning rate for stage 2 (default `0.00001`, should be small)
- `num_workers` — DataLoader workers, increase if you have CPU cores to spare

## 3. Run inference on a single image

```bash
python -m car_damage.inference.predict --image path/to/photo.jpg --checkpoint checkpoints/best_model.pt
```

Prints the predicted class and confidence.

## What "good" looks like

- Validation/test accuracy above ~85% is a reasonable first target.
- Check the confusion matrix, not just accuracy — if your data is imbalanced
  (e.g., 80% damaged / 20% not_damaged), a model can get 80% accuracy by
  always predicting "damaged." Look at precision/recall per class.
- If train accuracy is much higher than val accuracy, you're overfitting —
  get more data, increase augmentation strength, or add dropout.
- If both train and val accuracy are low, the model is underfitting — train
  longer, unfreeze more layers, or check your labels are correct.

## Troubleshooting

- **"No images found" / "Expected folder not found"** — check your folder
  structure matches exactly `<data_dir>/damaged/` and
  `<data_dir>/not_damaged/` (these names are hardcoded in
  `src/car_damage/data/dataset.py`).
- **CUDA out of memory** — lower `batch_size` in the config.
- **Val accuracy stuck near 50%** — likely a labeling or data quality issue;
  visually spot-check a few images from each folder.

## Architectural boundaries

These are hard rules, not suggestions — see the module docstrings for the
specifics:

1. **`models/production/` vs `models/research/` is a hard boundary.**
   `production/` holds thin wrappers around library architectures (e.g.
   torchvision ResNet) — code that rarely changes. `research/` holds
   custom/experimental architectures under active iteration
   (`models/research/scratch_cnn.py`, built from the hand-rolled layers in
   `layers/scratch_layers.py`). Nothing in `production/` may import from
   `research/` or `layers/`.
2. **`layers/` is quarantined**, sitting next to `models/`, not inside it.
   It holds hand-rolled building blocks (custom conv/batchnorm/etc, built
   from raw tensor ops) for learning purposes — not production training
   speed. Nothing outside `research/` should depend on it.
3. **`training/loop.py` is model-agnostic.** `run_epoch()` accepts any
   model implementing a standard `forward()` interface and runs the same
   train/val loop regardless of whether the model comes from `production/`
   or `research/`. The loop is never duplicated per model.
4. **All hyperparameters live in `configs/*.yaml`.** `train.py` takes a
   `--config` path argument and reads everything (learning rate, batch
   size, epochs, model choice) from that file — nothing is hardcoded.
5. **`experiments/` holds every run's outputs**, one timestamped folder per
   run containing the exact config used, the checkpoint, training curves,
   and a metrics file, so any run is reproducible from its own folder
   alone. It is listed in `.gitignore`. (Phase 1's `train.py` currently
   writes to the flat `checkpoint_dir` from its config instead — also
   gitignored — rather than a timestamped `experiments/` subfolder.)
6. **`notebooks/` is exploration-only.** Nothing else in the codebase may
   import from `notebooks/`.
7. **`tests/` checks structural correctness** (output shapes, label
   alignment, config loading) — not model accuracy, which belongs in
   `evaluation/`.

## Status

- **Phase 1 (classifier)** — implemented: `data/dataset.py`,
  `data/transforms.py`, `models/production/resnet_classifier.py`,
  `training/loop.py`, `training/train.py`, `evaluation/metrics.py`,
  `evaluation/evaluate.py`, `inference/predict.py`.
- **`models/research/scratch_cnn.py`** + **`layers/scratch_layers.py`** — a
  hand-rolled CNN, implemented but not yet wired into
  `training/train.py`, which currently always trains the production
  ResNet. Swapping it in only requires building a `ScratchCNN` instead of
  calling `build_model()` and passing it through the same
  `training.loop.run_epoch` — the loop doesn't care which one it gets.
- **Phase 2 (detector)** — `configs/phase2_detector.yaml` and everything
  under it are still stubs; see the `# TODO` markers in those files.
