"""Model artifact layout shared by training, evaluation, the endpoint and the explainers.

A model directory (what SageMaker tars into ``model.tar.gz``) contains::

    model_meta.json   {"model_type": "graphsage"|"xgboost", "threshold": float,
                       "feature_cols": [...], "bidirectional_edges": bool, "config": {...}}
    scaler.json       (graphsage only) train-period mean/scale
    model.pt          (graphsage) | model.ubj (xgboost)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .graph import FeatureScaler

META_FILE = "model_meta.json"
GNN_FILE = "model.pt"
XGB_FILE = "model.ubj"


def write_meta(model_dir: str | Path, **meta: Any) -> None:
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    (Path(model_dir) / META_FILE).write_text(json.dumps(meta, indent=2, default=float))


def read_meta(model_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(model_dir) / META_FILE).read_text())


class LoadedModel:
    """Uniform wrapper over the two model families."""

    def __init__(self, model_dir: str | Path, device: str = "cpu"):
        self.model_dir = Path(model_dir)
        self.meta = read_meta(model_dir)
        self.model_type: str = self.meta["model_type"]
        self.threshold: float = float(self.meta["threshold"])
        self.feature_cols: list[str] = self.meta["feature_cols"]
        self.device = device
        if self.model_type == "graphsage":
            from .gnn import load_graphsage

            self.model = load_graphsage(self.model_dir / GNN_FILE, device)
            self.scaler = FeatureScaler.load(self.model_dir / "scaler.json")
        elif self.model_type == "xgboost":
            from .baseline import load_xgboost

            self.model = load_xgboost(self.model_dir / XGB_FILE)
            self.scaler = None
        else:
            raise ValueError(f"unknown model_type {self.model_type!r}")

    @property
    def bidirectional(self) -> bool:
        return bool(self.meta.get("bidirectional_edges", True))
