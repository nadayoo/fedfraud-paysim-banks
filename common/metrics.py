"""Metrics shared by all experiments (PR-AUC = average precision, as in Flower Hub)."""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x.astype(np.float64), -50, 50)))


def threshold_metrics(probs: np.ndarray, labels: np.ndarray, t: float) -> dict:
    pred = probs >= t
    pos = labels == 1
    tp = int((pred & pos).sum()); fp = int((pred & ~pos).sum())
    fn = int((~pred & pos).sum()); tn = int((~pred & ~pos).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(t),
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "precision": precision, "recall": recall, "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def compute_metrics(probs: np.ndarray, labels: np.ndarray, thresholds: list) -> dict:
    both = len(np.unique(labels)) > 1
    return {
        "pr_auc": float(average_precision_score(labels, probs)) if both else float("nan"),
        "roc_auc": float(roc_auc_score(labels, probs)) if both else float("nan"),
        "thresholds": {f"{t:g}": threshold_metrics(probs, labels, t) for t in thresholds},
    }


def best_f1_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    """Pick the operating point on VALIDATION data (never on test)."""
    p, r, th = precision_recall_curve(labels, probs)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(th[int(np.argmax(f1))])
