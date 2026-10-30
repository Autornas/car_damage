"""
loop.py

Model-agnostic train/val epoch runner. Works identically whether `model`
is a production wrapper (models.production.resnet_classifier.build_model)
or a research architecture (models.research.scratch_cnn.ScratchCNN) -- it
only relies on the standard nn.Module forward() interface and never
branches on model internals. This is the one place the epoch loop is
written; it is not duplicated per model.

Allowed imports: torch, tqdm. Must not import car_damage.models or
car_damage.layers -- the model instance is passed in already constructed.
"""

import torch
from tqdm import tqdm


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()

    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total
