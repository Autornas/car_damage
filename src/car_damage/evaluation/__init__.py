"""Evaluation package: metrics computation (metrics.py) and the evaluation
entrypoint (evaluate.py). Model-quality assessment lives here, not in
tests/. Re-exports the public API so callers can do
``from car_damage.evaluation import evaluate_test_set``.
"""

from car_damage.evaluation.metrics import compute_classification_report, compute_confusion_matrix
from car_damage.evaluation.evaluate import evaluate_test_set, plot_history, save_metrics_report

__all__ = [
    "compute_classification_report",
    "compute_confusion_matrix",
    "evaluate_test_set",
    "plot_history",
    "save_metrics_report",
]
