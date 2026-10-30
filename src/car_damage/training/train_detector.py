"""
train_detector.py

CLI entrypoint for phase 2 detection training/smoke-testing. All
hyperparameters come from a --config yaml file (see
configs/phase2_detector.yaml) -- nothing is hardcoded here. Follows the
same --config pattern as phase 1's train.py.

Usage:
    python -m car_damage.training.train_detector --config configs/phase2_detector.yaml

With the default configs/phase2_detector.yaml, this runs against the
60-image pilot batch in data/raw/phase2_pilot/ as a wiring smoke test
(2 epochs) -- NOT a real training run. See that directory's README.md for
why: 60 images / 93 boxes can produce a loss curve, not a usable detector.

Every run is versioned against the data that produced it: at startup this
snapshots the current dataset (car_damage.data.versioning.write_snapshot,
the same logic scripts/snapshot_dataset.py uses) and creates a
self-contained experiments/detector_runs/<timestamp>_<hash>/ folder
holding a copy of the config used, dataset_version.txt, a copy of the
matching data/dataset_versions/<hash>_<date>.json snapshot, the checkpoint,
and a metrics.json carrying that same dataset_version -- so a checkpoint is
never separated from exactly which data produced it. This is in addition
to (not instead of) the flat cfg["checkpoint_dir"] save, which is what the
backend (backend/inference_service.py) actually loads at inference time.

Allowed imports: standard library, car_damage.data,
car_damage.models.production.faster_rcnn_detector,
car_damage.training.detection_loop, car_damage.evaluation.evaluate,
pyyaml, argparse.
"""

import argparse
import os
import shutil
from datetime import datetime

import torch
import yaml
from torch.utils.data import DataLoader

from car_damage.data.dataset import CarDamageDetectionDataset, detection_collate_fn
from car_damage.data.transforms import get_detection_transform
from car_damage.data.versioning import write_snapshot
from car_damage.models.production.faster_rcnn_detector import build_detector
from car_damage.training.detection_loop import run_detection_epoch
from car_damage.evaluation.evaluate import save_metrics_report


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2_detector.yaml",
                         help="Path to a phase 2 config yaml (see configs/phase2_detector.yaml)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Dataset version + self-contained experiment folder ----
    snapshot, snapshot_path = write_snapshot()
    print(f"Dataset version: {snapshot['short_hash']}")

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{snapshot['short_hash']}"
    run_dir = os.path.join("experiments", "detector_runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy2(args.config, os.path.join(run_dir, os.path.basename(args.config)))
    with open(os.path.join(run_dir, "dataset_version.txt"), "w") as f:
        f.write(snapshot["short_hash"] + "\n")
    shutil.copy2(snapshot_path, os.path.join(run_dir, os.path.basename(snapshot_path)))

    # ---- Data ----
    transform = get_detection_transform()
    train_ds = CarDamageDetectionDataset(cfg["data_dir"], split="train", transform=transform)
    val_ds = CarDamageDetectionDataset(cfg["data_dir"], split="val", transform=transform)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                               collate_fn=detection_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                             collate_fn=detection_collate_fn)

    # ---- Model ----
    # Uses build_detector's default (num_classes=14: 13 damage types +
    # background, see DAMAGE_TYPE_TO_LABEL). The backend uses the same
    # default when loading this checkpoint for inference.
    model = build_detector().to(device)

    # SGD with momentum is the standard recipe for fine-tuning torchvision
    # detection models (see torchvision's own reference training scripts).
    # Adam's larger effective step size is more prone to diverging a
    # pretrained detector than it was for phase 1's classifier head.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9, weight_decay=0.0005)

    # ---- Train ----
    final_train_loss = None
    for epoch in range(cfg["epochs"]):
        final_train_loss = run_detection_epoch(model, train_loader, optimizer, device, train=True)
        print(f"Epoch {epoch+1}/{cfg['epochs']} | train_loss={final_train_loss:.4f}")
        run_detection_epoch(model, val_loader, optimizer, device, train=False)

    # ---- Save checkpoint ----
    ckpt_path = os.path.join(cfg["checkpoint_dir"], "detector_smoke_test.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")
    shutil.copy2(ckpt_path, os.path.join(run_dir, os.path.basename(ckpt_path)))

    # ---- Metrics, kept together with the dataset version that produced them ----
    metrics = {
        "final_train_loss": final_train_loss,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "epochs": cfg["epochs"],
    }
    save_metrics_report(metrics, os.path.join(run_dir, "metrics.json"), dataset_version=snapshot["short_hash"])
    print(f"Run folder: {run_dir}")


if __name__ == "__main__":
    main()
