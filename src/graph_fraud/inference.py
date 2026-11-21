"""Framework-free scoring used by the SageMaker endpoint (``scripts/serve.py``) and tests.

Request schema (JSON)::

    {
      "nodes":   [{"id": "230425980", "features": [165 floats]}, ...],
      "edges":   [["230425980", "5530458"], ...],      # optional, directed src -> dst
      "targets": ["230425980", ...]                    # optional, default: all nodes
    }

Response::

    {"model_type": "graphsage", "threshold": 0.41,
     "scores": {"230425980": 0.93, ...}, "flags": {"230425980": 1, ...}}

For the GNN the caller ships the k-hop neighbourhood of the targets (k = number of
layers); scores for targets whose neighbourhood is missing are still returned, just
computed from fewer neighbours, exactly like an isolated transaction.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .artifacts import LoadedModel


class BadRequest(ValueError):
    pass


def parse_payload(
    payload: dict[str, Any], n_features: int
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, list[str]]:
    nodes = payload.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise BadRequest("'nodes' must be a non-empty list")
    ids: list[str] = []
    feats = np.zeros((len(nodes), n_features), dtype=np.float32)
    for i, node in enumerate(nodes):
        try:
            ids.append(str(node["id"]))
            f = np.asarray(node["features"], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as exc:
            raise BadRequest(f"node #{i} must have 'id' and numeric 'features'") from exc
        if f.shape != (n_features,):
            raise BadRequest(
                f"node {ids[-1]} has {f.shape[0] if f.ndim else 0} features, expected {n_features}"
            )
        feats[i] = f
    if len(set(ids)) != len(ids):
        raise BadRequest("duplicate node ids")
    pos = {nid: i for i, nid in enumerate(ids)}

    edges = payload.get("edges") or []
    src, dst = [], []
    for e in edges:
        if not isinstance(e, (list, tuple)) or len(e) != 2:
            raise BadRequest("each edge must be a [src, dst] pair")
        s, d = str(e[0]), str(e[1])
        if s in pos and d in pos:  # edges to nodes we were not given cannot carry messages
            src.append(pos[s])
            dst.append(pos[d])
    targets = [str(t) for t in payload.get("targets") or ids]
    missing = [t for t in targets if t not in pos]
    if missing:
        raise BadRequest(f"targets not present in nodes: {missing[:5]}")
    return ids, feats, np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64), targets


def score_payload(lm: LoadedModel, payload: dict[str, Any]) -> dict[str, Any]:
    ids, feats, src, dst, targets = parse_payload(payload, len(lm.feature_cols))
    if lm.model_type == "xgboost":
        from .baseline import predict_proba

        scores = predict_proba(lm.model, feats)
    else:
        import dgl
        import torch

        s, d = torch.from_numpy(src), torch.from_numpy(dst)
        if lm.bidirectional:
            s, d = torch.cat([s, d]), torch.cat([d, s])
        g = dgl.add_self_loop(dgl.graph((s, d), num_nodes=len(ids)))
        x = torch.from_numpy(lm.scaler.transform(feats))
        scores = lm.model.predict_proba(g, x).numpy()
    pos = {nid: i for i, nid in enumerate(ids)}
    return {
        "model_type": lm.model_type,
        "threshold": lm.threshold,
        "scores": {t: float(scores[pos[t]]) for t in targets},
        "flags": {t: int(scores[pos[t]] >= lm.threshold) for t in targets},
    }
