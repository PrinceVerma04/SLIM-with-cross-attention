import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    det_curve,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


def compute_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thresholds[idx])
    return eer, threshold


def compute_curve_data(y_true, y_score):
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    det_fpr, det_fnr, det_thresholds = det_curve(y_true, y_score)
    return {
        "roc": {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": roc_thresholds,
        },
        "det": {
            "fpr": det_fpr,
            "fnr": det_fnr,
            "thresholds": det_thresholds,
        },
    }


def compute_classification_metrics(y_true, y_pred, y_score=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
    }

    if y_score is not None and len(np.unique(y_true)) > 1:
        eer, eer_threshold = compute_eer(y_true, y_score)
        metrics["eer"] = eer
        metrics["eer_threshold"] = eer_threshold

    return metrics
