"""
detection_loop.py

Detection-specific train/eval epoch runner -- separate from
training/loop.py on purpose. torchvision detection models (e.g. Faster
R-CNN) return a loss dict directly from forward(images, targets) when in
train() mode; there's no external criterion to pass in, unlike the
classification loop. Forcing this into loop.py's run_epoch(model, loader,
criterion, optimizer, device, train) signature would mean threading a
criterion=None special case through code that's supposed to stay
model-agnostic, so detection gets its own loop instead.

In eval() mode, torchvision detection models return predictions (not
losses) from forward(images) -- this module does not score those
predictions (no mAP yet); that's a separate step once there's enough data
for it to be meaningful (see car_damage.evaluation).

Allowed imports: torch, tqdm. Must not import car_damage.models or
car_damage.layers -- the model instance is passed in already constructed.
"""

import torch
from tqdm import tqdm


def run_detection_epoch(model, loader, optimizer, device, train=True):
    """
    Returns the average summed loss for a training epoch, or None for an
    eval epoch (see module docstring for why eval doesn't produce a loss
    number here).
    """
    model.train() if train else model.eval()

    total_loss, num_batches = 0.0, 0

    for images, targets in tqdm(loader, leave=False):
        images = [image.to(device) for image in images]

        if train:
            targets = [
                {k: v.to(device) for k, v in t.items() if k != "parts"}
                for t in targets
            ]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
        else:
            with torch.no_grad():
                model(images)  # confirms the forward pass runs; predictions unused for now

    return (total_loss / num_batches) if train else None
