"""
train.py

CLI entrypoint for phase 1 training (binary classifier: damaged vs
not_damaged). All hyperparameters come from a --config yaml file (see
configs/phase1_classifier.yaml) -- nothing is hardcoded here.

Usage:
    python -m car_damage.training.train --config configs/phase1_classifier.yaml

What this script does:
    1. Splits the configured data_dir into train/val/test
    2. Trains a ResNet18 head first (backbone frozen), then unfreezes
       the last block for a few epochs of fine-tuning
    3. Tracks train/val loss + accuracy each epoch
    4. Saves the best model checkpoint (by val accuracy)
    5. Runs a final evaluation on the held-out test set with a confusion matrix

Allowed imports: car_damage.data, car_damage.models.production.resnet_classifier,
car_damage.training.loop, car_damage.evaluation.evaluate, pyyaml, argparse.
"""

import argparse
import os
import copy
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from car_damage.data.dataset import make_splits, CarDamageDataset
from car_damage.data.transforms import get_train_transform, get_eval_transform
from car_damage.models.production.resnet_classifier import build_model, unfreeze_last_block
from car_damage.training.loop import run_epoch
from car_damage.evaluation.evaluate import evaluate_test_set, plot_history


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1_classifier.yaml",
                         help="Path to a phase 1 config yaml (see configs/phase1_classifier.yaml)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Data ----
    splits = make_splits(cfg["data_dir"])
    train_fp, train_lb = splits["train"]
    val_fp, val_lb = splits["val"]
    test_fp, test_lb = splits["test"]
    print(f"Train: {len(train_fp)} | Val: {len(val_fp)} | Test: {len(test_fp)}")

    train_ds = CarDamageDataset(train_fp, train_lb, transform=get_train_transform())
    val_ds = CarDamageDataset(val_fp, val_lb, transform=get_eval_transform())
    test_ds = CarDamageDataset(test_fp, test_lb, transform=get_eval_transform())

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    # ---- Model ----
    model = build_model(num_classes=2, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None

    # ---- Stage 1: train head only ----
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr"])
    print("\n--- Stage 1: training classifier head (backbone frozen) ---")
    for epoch in range(cfg["epochs"]):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{cfg['epochs']} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    # ---- Stage 2: unfreeze last block, fine-tune with lower LR ----
    print("\n--- Stage 2: fine-tuning last block ---")
    model = unfreeze_last_block(model)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["finetune_lr"])

    for epoch in range(cfg["finetune_epochs"]):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"[Fine-tune] Epoch {epoch+1}/{cfg['finetune_epochs']} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    # ---- Save best model ----
    ckpt_path = os.path.join(cfg["checkpoint_dir"], "best_model.pt")
    torch.save(best_state, ckpt_path)
    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Saved best model to {ckpt_path}")

    # ---- Plot curves ----
    plot_history(history, os.path.join(cfg["checkpoint_dir"], "training_curves.png"))

    # ---- Final test evaluation ----
    model.load_state_dict(best_state)
    evaluate_test_set(model, test_loader, device)


if __name__ == "__main__":
    main()
