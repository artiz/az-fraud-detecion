#!/usr/bin/env python
"""Step 2: train XGBoost baseline or GraphSAGE. Runs locally and as a SageMaker training job.

SageMaker contract (script mode / sagemaker-training toolkit):
  * input channel "data" is mounted at SM_CHANNEL_DATA (the processed parquet dir),
  * the model directory SM_MODEL_DIR is tarred to S3 as model.tar.gz,
  * hyperparameters arrive as ``--model graphsage --graphsage.epochs 20`` arguments.

Local:
  python scripts/train.py --model xgboost   --data-dir data/processed --model-dir artifacts/xgboost
  python scripts/train.py --model graphsage --data-dir data/processed --model-dir artifacts/graphsage
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graph_fraud.artifacts import GNN_FILE, XGB_FILE, write_meta  # noqa: E402
from graph_fraud.config import apply_overrides, load_config  # noqa: E402
from graph_fraud.data import EllipticTables, split_masks  # noqa: E402
from graph_fraud.metrics import illicit_metrics, per_timestep_f1  # noqa: E402

log = logging.getLogger("train")


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, dict]:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", choices=["xgboost", "graphsage"], required=True)
    p.add_argument("--data-dir", default=os.environ.get("SM_CHANNEL_DATA", "data/processed"))
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "artifacts/model"))
    p.add_argument(
        "--output-dir",
        default=os.environ.get("SM_OUTPUT_DATA_DIR", None),
        help="where metrics.json goes (default: model-dir)",
    )
    p.add_argument("--config", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--device", default="cuda" if os.environ.get("SM_NUM_GPUS", "0") not in ("0", "") else "cpu"
    )
    args, unknown = p.parse_known_args(argv)
    # Anything like --graphsage.epochs 5 or --xgboost.max_depth 4 becomes a config override.
    overrides: dict[str, str] = {}
    key = None
    for tok in unknown:
        if tok.startswith("--"):
            key = tok[2:].replace("-", "_") if "." not in tok else tok[2:]
            overrides[key] = "true"
        elif key is not None:
            overrides[key] = tok
            key = None
        else:
            p.error(f"unexpected argument {tok!r}")
    if args.seed is not None:
        overrides["seed"] = str(args.seed)
    return args, overrides


def evaluate_split(
    scores: np.ndarray, tables: EllipticTables, mask: np.ndarray, threshold: float
) -> dict:
    y = tables.nodes["label"].to_numpy()[mask]
    ts = tables.nodes["timestep"].to_numpy()[mask]
    out = illicit_metrics(y, scores[mask], threshold)
    out["per_timestep"] = per_timestep_f1(y, scores[mask], ts, threshold)
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args, overrides = parse_args(argv)
    cfg = apply_overrides(load_config(args.config), overrides)
    seed = int(cfg.get("seed", 42))
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = EllipticTables.load(args.data_dir)
    masks = split_masks(tables.nodes)
    log.info(
        "nodes=%d edges=%d train/val/test labelled = %d/%d/%d",
        tables.num_nodes,
        len(tables.edges),
        masks["train"].sum(),
        masks["val"].sum(),
        masks["test"].sum(),
    )
    t0 = time.time()
    extra_meta: dict = {}

    if args.model == "xgboost":
        from graph_fraud.baseline import predict_proba, save_xgboost, train_xgboost

        x = tables.nodes[tables.feature_cols].to_numpy(np.float32)
        y = tables.nodes["label"].to_numpy()
        model, summary = train_xgboost(
            x[masks["train"]],
            y[masks["train"]],
            x[masks["val"]],
            y[masks["val"]],
            cfg["xgboost"],
            seed=seed,
        )
        scores = predict_proba(model, x)
        save_xgboost(model, model_dir / XGB_FILE)
    else:
        from graph_fraud.gnn import save_graphsage, train_graphsage
        from graph_fraud.graph import build_dgl_graph

        g, scaler = build_dgl_graph(tables, bidirectional=cfg["data"]["bidirectional_edges"])
        model, summary = train_graphsage(g, cfg["graphsage"], seed=seed, device=args.device)
        scores = (
            model.predict_proba(g.to(args.device), g.ndata["feat"].to(args.device)).cpu().numpy()
        )
        save_graphsage(model, model_dir / GNN_FILE)
        scaler.save(model_dir / "scaler.json")
        extra_meta["bidirectional_edges"] = bool(cfg["data"]["bidirectional_edges"])
        summary["history"] = summary.get("history", [])

    threshold = float(summary["threshold"])
    metrics = {
        "model_type": args.model,
        "train_seconds": time.time() - t0,
        "threshold": threshold,
        "val": evaluate_split(scores, tables, masks["val"], threshold),
        "test": evaluate_split(scores, tables, masks["test"], threshold),
        "training": {k: v for k, v in summary.items() if k != "history"},
        "history": summary.get("history", []),
    }
    write_meta(
        model_dir,
        model_type=args.model,
        threshold=threshold,
        feature_cols=tables.feature_cols,
        config=cfg,
        **extra_meta,
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    # One-line summary that SageMaker metric definitions can regex from CloudWatch logs.
    t = metrics["test"]
    log.info(
        "RESULT model=%s test_pr_auc=%.4f test_f1=%.4f test_precision=%.4f test_recall=%.4f threshold=%.4f",
        args.model,
        t["pr_auc"],
        t["illicit_f1"],
        t["illicit_precision"],
        t["illicit_recall"],
        threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
