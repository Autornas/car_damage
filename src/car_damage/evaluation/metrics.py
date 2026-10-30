"""
metrics.py

Classification metrics for the phase 1 damaged/not_damaged classifier --
confusion matrix and precision/recall/F1 report. Model-quality assessment
lives here, not in tests/.

Allowed imports: scikit-learn.
"""

from sklearn.metrics import confusion_matrix, classification_report


def compute_classification_report(labels, preds, class_names=("not_damaged", "damaged")):
    return classification_report(labels, preds, target_names=class_names)


def compute_confusion_matrix(labels, preds):
    return confusion_matrix(labels, preds)
