"""Explainability for both model families.

* **XGBoost → SHAP TreeExplainer**: exact Shapley values in log-odds space; global
  importance (mean |SHAP|) and per-transaction top features.
* **GraphSAGE → DGL GNNExplainer** (Ying et al. 2019): learns a feature mask and an
  edge mask on the k-hop computation subgraph of one node, answering *which features*
  and *which neighbouring transactions* drove the score.

Both return plain dicts (JSON-serialisable) so the same structure can be rendered in a
notebook, logged, or handed to an analyst-facing agent.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- SHAP


def shap_global_importance(
    booster,
    x: np.ndarray,
    feature_cols: list[str],
    top_k: int = 20,
    max_rows: int = 5000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    import shap

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=min(max_rows, len(x)), replace=False)
    explainer = shap.TreeExplainer(booster)
    sv = explainer.shap_values(x[idx])
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(-mean_abs)[:top_k]
    return [{"feature": feature_cols[i], "mean_abs_shap": float(mean_abs[i])} for i in order]


def shap_explain_rows(
    booster, x_rows: np.ndarray, feature_cols: list[str], top_k: int = 10
) -> list[dict[str, Any]]:
    """Per-row explanation: top-k features by |SHAP| with sign and raw value."""
    import shap

    explainer = shap.TreeExplainer(booster)
    sv = np.atleast_2d(explainer.shap_values(x_rows))
    base = float(np.ravel(explainer.expected_value)[0])
    out = []
    for r in range(sv.shape[0]):
        order = np.argsort(-np.abs(sv[r]))[:top_k]
        logit = base + float(sv[r].sum())
        out.append(
            {
                "base_value_logit": base,
                "logit": logit,
                "probability": float(1 / (1 + np.exp(-logit))),
                "top_features": [
                    {
                        "feature": feature_cols[i],
                        "value": float(x_rows[r, i]),
                        "shap": float(sv[r, i]),
                    }
                    for i in order
                ],
            }
        )
    return out


# ------------------------------------------------------------------- GNNExplainer


def gnn_explain_node(
    model,
    g,
    node_id: int,
    feature_cols: list[str],
    tx_ids: np.ndarray,
    top_k_features: int = 10,
    top_k_edges: int = 10,
    num_epochs: int = 200,
    lr: float = 0.01,
    seed: int = 0,
) -> dict[str, Any]:
    """Explain one node's prediction with DGL's GNNExplainer.

    ``tx_ids`` maps positional node ids to original transaction ids for the report.
    """
    import dgl
    import torch
    from dgl.nn import GNNExplainer

    torch.manual_seed(seed)
    model.eval()
    feat = g.ndata["feat"]
    with torch.no_grad():
        prob = torch.softmax(model(g, feat), dim=1)[node_id, 1].item()

    explainer = GNNExplainer(
        model, num_hops=model.num_layers, lr=lr, num_epochs=num_epochs, log=False
    )
    new_center, sg, feat_mask, edge_mask = explainer.explain_node(node_id, g, feat)

    orig_nid = sg.ndata[dgl.NID].numpy()
    src, dst = (t.numpy() for t in sg.edges())
    em = edge_mask.detach().numpy()
    fm = feat_mask.detach().numpy()
    # Self-loops carry the node's own features, not neighbour influence; report real neighbours first.
    edge_rows = []
    for e in np.argsort(-em):
        s, d = int(orig_nid[src[e]]), int(orig_nid[dst[e]])
        edge_rows.append(
            {
                "src_tx_id": int(tx_ids[s]),
                "dst_tx_id": int(tx_ids[d]),
                "weight": float(em[e]),
                "self_loop": s == d,
                "src_label": int(g.ndata["label"][s]),
                "dst_label": int(g.ndata["label"][d]),
            }
        )
        if len(edge_rows) >= top_k_edges:
            break
    feat_order = np.argsort(-fm)[:top_k_features]
    return {
        "tx_id": int(tx_ids[node_id]),
        "node_id": int(node_id),
        "label": int(g.ndata["label"][node_id]),
        "timestep": int(g.ndata["timestep"][node_id]),
        "illicit_probability": prob,
        "subgraph": {
            "num_nodes": int(sg.num_nodes()),
            "num_edges": int(sg.num_edges()),
            "center_in_subgraph": int(new_center),
        },
        "top_features": [
            {"feature": feature_cols[i], "mask": float(fm[i]), "value": float(feat[node_id, i])}
            for i in feat_order
        ],
        "top_edges": edge_rows,
    }
