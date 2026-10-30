"""
evaluate.py

Runs a trained model over the held-out test set and prints a metrics
report, and plots train/val loss + accuracy curves from a training
history dict. Metric computation itself (confusion matrix, classification
report) lives in evaluation.metrics -- this module wires that together
with the inference loop and plotting.

Also owns save_metrics_report(), the one place a metrics.json gets
written to disk -- always with the dataset version hash that produced it
merged in when given, so a metrics file is never separated from exactly
which data trained the model it's reporting on (see
training/train_detector.py, which calls this at the end of every phase 2
run).

Allowed imports: standard library, torch, tqdm, matplotlib,
car_damage.evaluation.metrics.
"""

import json
import os

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from car_damage.evaluation.metrics import compute_classification_report, compute_confusion_matrix


def evaluate_test_set(model, loader, device, class_names=("not_damaged", "damaged")):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing", leave=False):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print("\n=== Test Set Results ===")
    print(compute_classification_report(all_labels, all_preds, class_names))

    cm = compute_confusion_matrix(all_labels, all_preds)
    print("Confusion matrix:")
    print(cm)
    return cm


def plot_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved training curves to {out_path}")


def save_metrics_report(metrics, out_path, dataset_version=None):
    """
    Writes `metrics` (a plain JSON-serializable dict) to out_path, with
    "dataset_version" merged in when given -- so this metrics.json can
    always be traced back to the exact dataset snapshot
    (data/dataset_versions/<hash>_<date>.json) that produced it.
    """
    report = dict(metrics)
    if dataset_version is not None:
        report["dataset_version"] = dataset_version

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved metrics to {out_path}")
    return report
