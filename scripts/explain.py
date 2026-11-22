#!/usr/bin/env python
"""Explain predictions of a trained model on the test period.

  python scripts/explain.py --model-dir artifacts/xgboost   --data-dir data/processed --out artifacts/xgboost/explanations.json
  python scripts/explain.py --model-dir artifacts/graphsage --data-dir data/processed --out artifacts/graphsage/explanations.json --num-nodes 5

For XGBoost: SHAP global importance + per-transaction SHAP for the highest-scoring test
transactions. For GraphSAGE: GNNExplainer feature and edge masks for the same kind of
transactions (top predicted illicit in the test period, plus the given ``--tx-id``s).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graph_fraud.artifacts import LoadedModel  # noqa: E402
from graph_fraud.data import EllipticTables, split_masks  # noqa: E402

log = logging.getLogger("explain")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model-dir", required=True)
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--num-nodes", type=int, default=5, help="how many top-scored test transactions to explain"
    )
    p.add_argument(
        "--tx-id",
        type=int,
        action="append",
        default=[],
        help="explicit transaction id(s) to explain",
    )
    p.add_argument("--epochs", type=int, default=200, help="GNNExplainer optimisation steps")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    lm = LoadedModel(args.model_dir)
    tables = EllipticTables.load(args.data_dir)
    test_mask = split_masks(tables.nodes)["test"]
    tx_ids = tables.nodes["tx_id"].to_numpy()
    x = tables.nodes[tables.feature_cols].to_numpy(np.float32)
    report: dict = {"model_type": lm.model_type, "threshold": lm.threshold}

    if lm.model_type == "xgboost":
        from graph_fraud.baseline import predict_proba
        from graph_fraud.explain import shap_explain_rows, shap_global_importance

        scores = predict_proba(lm.model, x)
        picked = _pick_nodes(scores, test_mask, tx_ids, args)
        report["global_importance"] = shap_global_importance(
            lm.model, x[test_mask], tables.feature_cols
        )
        rows = shap_explain_rows(lm.model, x[picked], tables.feature_cols)
        for nid, row in zip(picked, rows, strict=True):
            row.update(
                tx_id=int(tx_ids[nid]),
                label=int(tables.nodes["label"].iat[nid]),
                timestep=int(tables.nodes["timestep"].iat[nid]),
                score=float(scores[nid]),
            )
        report["transactions"] = rows
    else:
        from graph_fraud.explain import gnn_explain_node
        from graph_fraud.graph import build_dgl_graph

        g, _ = build_dgl_graph(tables, scaler=lm.scaler, bidirectional=lm.bidirectional)
        scores = lm.model.predict_proba(g, g.ndata["feat"]).numpy()
        picked = _pick_nodes(scores, test_mask, tx_ids, args)
        report["transactions"] = [
            gnn_explain_node(
                lm.model, g, int(nid), tables.feature_cols, tx_ids, num_epochs=args.epochs
            )
            for nid in picked
        ]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    log.info("wrote %d explanations to %s", len(report["transactions"]), args.out)
    return 0


def _pick_nodes(scores: np.ndarray, test_mask: np.ndarray, tx_ids: np.ndarray, args) -> list[int]:
    test_idx = np.nonzero(test_mask)[0]
    top = test_idx[np.argsort(-scores[test_idx])[: args.num_nodes]].tolist()
    explicit = [int(np.nonzero(tx_ids == t)[0][0]) for t in args.tx_id if (tx_ids == t).any()]
    return list(dict.fromkeys(explicit + top))


if __name__ == "__main__":
    raise SystemExit(main())
