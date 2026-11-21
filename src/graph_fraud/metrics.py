"""Metrics for the positive (illicit) class, plus threshold selection.

PR-AUC (average precision) is the headline number: with ~2% positives, ROC-AUC
hides most of the differences between models. F1 is reported at a threshold chosen
on the validation period, never on the test period.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximising F1 of the positive class on (y_true, scores)."""
    y_true = np.asarray(y_true)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.nanargmax(f1))])


def illicit_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)
    out = {
        "illicit_f1": float(f1_score(y_true, pred, zero_division=0)),
        "illicit_precision": float(precision_score(y_true, pred, zero_division=0)),
        "illicit_recall": float(recall_score(y_true, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, scores)) if y_true.any() else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, scores))
        if 0 < y_true.sum() < len(y_true)
        else float("nan"),
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "n_illicit": int(y_true.sum()),
    }
    return out


def per_timestep_f1(
    y_true: np.ndarray, scores: np.ndarray, timesteps: np.ndarray, threshold: float
) -> list[dict]:
    """F1 per time step; this is the plot every Elliptic paper shows (dark-market shutdown at step 43)."""
    rows = []
    for ts in np.unique(timesteps):
        m = timesteps == ts
        yt, pr = np.asarray(y_true)[m].astype(int), (np.asarray(scores)[m] >= threshold).astype(int)
        rows.append(
            {
                "timestep": int(ts),
                "n_illicit": int(yt.sum()),
                "illicit_f1": float(f1_score(yt, pr, zero_division=0)),
            }
        )
    return rows
