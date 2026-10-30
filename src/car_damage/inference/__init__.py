"""Inference package: run a trained checkpoint on new images. Re-exports
predict.py's public API so callers can do
``from car_damage.inference import predict``.
"""

from car_damage.inference.predict import predict

__all__ = ["predict"]
