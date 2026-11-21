#!/usr/bin/env python
"""Step 3 (SageMaker Processing or local): score a saved model on the test period.

Writes ``evaluation.json`` in the SageMaker Model Quality layout so the pipeline's
ConditionStep can read ``binary_classification_metrics.pr_auc.value`` and the model
registry shows the numbers on the model package.

Local:     python scripts/evaluate.py --model-dir artifacts/graphsage --data-dir data/processed --out-dir artifacts/graphsage/eval
SageMaker: model.tar.gz is mounted at /opt/ml/processing/model, data at /opt/ml/processing/input/data.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tarfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graph_fraud.artifacts import LoadedModel  # noqa: E402
from graph_fraud.data import EllipticTables, split_masks  # noqa: E402
from graph_fraud.metrics import illicit_metrics, per_timestep_f1  # noqa: E402

log = logging.getLogger("evaluate")


def resolve_model_dir(path: Path) -> Path:
    """Accept a directory or the model.tar.gz SageMaker drops in the processing container."""
    tar = path if path.is_file() else (path / "model.tar.gz")
    if tar.is_file():
        dest = path.parent / "model_extracted" if path.is_file() else path / "extracted"
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar) as tf:
            tf.extractall(dest, filter="data")
        return dest
    return path


def score_all(lm: LoadedModel, tables: EllipticTables) -> np.ndarray:
    x = tables.nodes[tables.feature_cols].to_numpy(np.float32)
    if lm.model_type == "xgboost":
        from graph_fraud.baseline import predict_proba

        return predict_proba(lm.model, x)
    from graph_fraud.graph import build_dgl_graph

    g, _ = build_dgl_graph(tables, scaler=lm.scaler, bidirectional=lm.bidirectional)
    return lm.model.predict_proba(g, g.ndata["feat"]).numpy()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", default="/opt/ml/processing/model")
    p.add_argument("--data-dir", default="/opt/ml/processing/input/data")
    p.add_argument("--out-dir", default="/opt/ml/processing/evaluation")
    p.add_argument("--split", default="test", choices=["val", "test"])
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    lm = LoadedModel(resolve_model_dir(Path(args.model_dir)))
    tables = EllipticTables.load(args.data_dir)
    mask = split_masks(tables.nodes)[args.split]
    scores = score_all(lm, tables)
    y = tables.nodes["label"].to_numpy()[mask]
    ts = tables.nodes["timestep"].to_numpy()[mask]
    m = illicit_metrics(y, scores[mask], lm.threshold)

    report = {
        "model_type": lm.model_type,
        "split": args.split,
        "binary_classification_metrics": {
            k: {"value": m[k], "standard_deviation": "NaN"}
            for k in ("pr_auc", "roc_auc", "illicit_f1", "illicit_precision", "illicit_recall")
        },
        "threshold": lm.threshold,
        "n": m["n"],
        "n_illicit": m["n_illicit"],
        "per_timestep": per_timestep_f1(y, scores[mask], ts, lm.threshold),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(report, indent=2))
    log.info("%s %s: PR-AUC %.4f F1 %.4f", lm.model_type, args.split, m["pr_auc"], m["illicit_f1"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
