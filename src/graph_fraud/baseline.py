"""Tabular baseline: gradient boosting on node features only (no graph).

Any GNN result has to beat this before the graph earns its complexity. On Elliptic the
tabular baseline is strong (Weber et al. report RF ≈ 0.79 illicit F1 on all features).

The native ``xgboost.Booster`` API is used instead of the scikit-learn wrapper: the
wrapper's ``save_model`` depends on private scikit-learn mixin attributes that changed
between scikit-learn 1.5 and 1.9, while the Booster format is stable and SHAP reads it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xgboost as xgb

from .metrics import best_f1_threshold, illicit_metrics

log = logging.getLogger(__name__)


def predict_proba(booster: xgb.Booster, x: np.ndarray) -> np.ndarray:
    """Illicit probability using the best iteration found by early stopping."""
    best = getattr(booster, "best_iteration", None)
    rng = (0, int(best) + 1) if best is not None else (0, 0)
    return booster.predict(xgb.DMatrix(x), iteration_range=rng)


def train_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
    seed: int = 42,
    n_jobs: int = 0,
) -> tuple[xgb.Booster, dict]:
    pos = max(int(y_train.sum()), 1)
    scale_pos_weight = (len(y_train) - pos) / pos
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "max_depth": cfg["max_depth"],
        "eta": cfg["learning_rate"],
        "subsample": cfg["subsample"],
        "colsample_bytree": cfg["colsample_bytree"],
        "min_child_weight": cfg["min_child_weight"],
        "scale_pos_weight": scale_pos_weight,
        "seed": seed,
        "nthread": n_jobs,
    }
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=cfg["n_estimators"],
        evals=[(dval, "val")],
        early_stopping_rounds=cfg["early_stopping_rounds"],
        verbose_eval=False,
    )
    val_scores = predict_proba(booster, x_val)
    thr = best_f1_threshold(y_val, val_scores)
    val = illicit_metrics(y_val, val_scores, thr)
    log.info(
        "xgboost best_iteration=%s val PR-AUC %.4f val F1 %.4f",
        booster.best_iteration,
        val["pr_auc"],
        val["illicit_f1"],
    )
    summary = {
        "best_iteration": int(booster.best_iteration),
        "best_val_pr_auc": val["pr_auc"],
        "threshold": thr,
        "scale_pos_weight": scale_pos_weight,
    }
    return booster, summary


def save_xgboost(booster: xgb.Booster, path: str | Path) -> None:
    # best_iteration is persisted as a model attribute inside the UBJSON file.
    booster.set_attr(best_iteration=str(int(booster.best_iteration)))
    booster.save_model(str(path))


def load_xgboost(path: str | Path) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(str(path))
    best = booster.attr("best_iteration")
    if best is not None:
        booster.best_iteration = int(best)
    return booster
