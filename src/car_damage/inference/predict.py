"""
predict.py

Run the trained phase 1 classifier on a single image.

Usage:
    python -m car_damage.inference.predict --image path/to/photo.jpg --checkpoint checkpoints/best_model.pt

Allowed imports: car_damage.models.production.resnet_classifier,
car_damage.data.transforms.
"""

import argparse
import cv2
import torch
import torch.nn.functional as F

from car_damage.models.production.resnet_classifier import build_model
from car_damage.data.transforms import get_eval_transform

CLASS_NAMES = ["not_damaged", "damaged"]


def predict(image_path, checkpoint_path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes=2, freeze_backbone=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = get_eval_transform()
    tensor = transform(image=image)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = probs.argmax()
    return {
        "prediction": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, probs)},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    args = parser.parse_args()

    result = predict(args.image, args.checkpoint)
    print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.2%})")
    print(f"Full probabilities: {result['probabilities']}")
