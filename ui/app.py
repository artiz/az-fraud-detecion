"""Exploration UI for the Elliptic graph: a timeline over the 49 steps and the transaction
graph of the selected step, coloured by ground truth (red illicit, blue licit, grey unknown)
or by a model's prediction. Click a node for its features, scores and an explanation.

    python ui/app.py --data-dir data/processed --artifacts artifacts      # http://localhost:8050

Everything is read once at start-up: the parquet tables and, for every model directory found
under --artifacts, scores for all nodes (XGBoost: 1 s; GraphSAGE full-graph forward: 2 s).
GNNExplainer / SHAP run on demand per node (a few seconds for the GNN).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graph_fraud.artifacts import LoadedModel  # noqa: E402
from graph_fraud.data import LABEL_ILLICIT, LABEL_LICIT, LABEL_UNKNOWN, EllipticTables  # noqa: E402

log = logging.getLogger("ui")
STATIC = Path(__file__).resolve().parent / "static"
LABEL_NAMES = {LABEL_ILLICIT: "illicit", LABEL_LICIT: "licit", LABEL_UNKNOWN: "unknown"}


class GraphStore:
    """In-memory tables + per-model scores; everything the API needs, indexed once."""

    def __init__(self, data_dir: str | Path, artifacts_dir: str | Path | None = None):
        self.tables = EllipticTables.load(data_dir)
        n = self.tables.nodes
        self.tx = n["tx_id"].to_numpy()
        self.ts = n["timestep"].to_numpy()
        self.label = n["label"].to_numpy()
        self.split = n["split"].to_numpy()
        self.pos = pd.Series(np.arange(len(n)), index=self.tx)
        self.src = self.tables.edges["src"].to_numpy()
        self.dst = self.tables.edges["dst"].to_numpy()
        self.edge_ts = self.ts[self.src]
        self.in_deg = np.bincount(self.dst, minlength=len(n))
        self.out_deg = np.bincount(self.src, minlength=len(n))
        self.models: dict[str, LoadedModel] = {}
        self.scores: dict[str, np.ndarray] = {}
        self._graph = None
        self._graph_cache: dict[int, dict] = {}
        if artifacts_dir and Path(artifacts_dir).is_dir():
            for d in sorted(Path(artifacts_dir).iterdir()):
                if (d / "model_meta.json").exists():
                    self._load_model(d)

    def _load_model(self, model_dir: Path) -> None:
        try:
            lm = LoadedModel(model_dir)
        except Exception as exc:  # noqa: BLE001 - a broken artifact must not take the UI down
            log.warning("skipping %s: %s", model_dir, exc)
            return
        x = self.tables.nodes[self.tables.feature_cols].to_numpy(np.float32)
        if lm.model_type == "xgboost":
            from graph_fraud.baseline import predict_proba

            scores = predict_proba(lm.model, x)
        else:
            from graph_fraud.graph import build_dgl_graph

            g, _ = build_dgl_graph(self.tables, scaler=lm.scaler, bidirectional=lm.bidirectional)
            self._graph = g
            scores = lm.model.predict_proba(g, g.ndata["feat"]).numpy()
        self.models[lm.model_type] = lm
        self.scores[lm.model_type] = scores.astype(np.float32)
        log.info("loaded %s from %s (threshold %.3f)", lm.model_type, model_dir, lm.threshold)

    # ---- API payloads -------------------------------------------------------------------

    def meta(self) -> dict:
        rows = []
        for t in np.unique(self.ts):
            m = self.ts == t
            rows.append(
                {
                    "timestep": int(t),
                    "split": str(self.split[m][0]),
                    "nodes": int(m.sum()),
                    "illicit": int((self.label[m] == LABEL_ILLICIT).sum()),
                    "licit": int((self.label[m] == LABEL_LICIT).sum()),
                    "unknown": int((self.label[m] == LABEL_UNKNOWN).sum()),
                    "edges": int((self.edge_ts == t).sum()),
                }
            )
        return {
            "timesteps": rows,
            "models": {k: {"threshold": lm.threshold} for k, lm in self.models.items()},
            "num_features": len(self.tables.feature_cols),
        }

    def graph(self, timestep: int) -> dict:
        if timestep in self._graph_cache:
            return self._graph_cache[timestep]
        idx = np.nonzero(self.ts == timestep)[0]
        if len(idx) == 0:
            raise KeyError(timestep)
        e = np.nonzero(self.edge_ts == timestep)[0]
        nodes = [
            {
                "id": int(self.tx[i]),
                "label": int(self.label[i]),
                "in": int(self.in_deg[i]),
                "out": int(self.out_deg[i]),
                **{f"s_{k}": round(float(s[i]), 4) for k, s in self.scores.items()},
            }
            for i in idx
        ]
        edges = [
            [int(self.tx[s]), int(self.tx[d])]
            for s, d in zip(self.src[e], self.dst[e], strict=True)
        ]
        return {"timestep": int(timestep), "nodes": nodes, "edges": edges}

    def node(self, tx_id: int) -> dict:
        if tx_id not in self.pos.index:
            raise KeyError(tx_id)
        i = int(self.pos[tx_id])
        x = self.tables.nodes.iloc[i][self.tables.feature_cols]
        return {
            "tx_id": int(tx_id),
            "timestep": int(self.ts[i]),
            "split": str(self.split[i]),
            "label": int(self.label[i]),
            "label_name": LABEL_NAMES[int(self.label[i])],
            "scores": {k: float(s[i]) for k, s in self.scores.items()},
            "flags": {k: int(s[i] >= self.models[k].threshold) for k, s in self.scores.items()},
            "in_neighbors": [int(self.tx[s]) for s in self.src[self.dst == i]],
            "out_neighbors": [int(self.tx[d]) for d in self.dst[self.src == i]],
            "features": {c: float(v) for c, v in x.items()},
        }

    def explain(self, tx_id: int, model: str) -> dict:
        if model not in self.models:
            raise KeyError(model)
        i = int(self.pos[tx_id])
        lm = self.models[model]
        if model == "xgboost":
            from graph_fraud.explain import shap_explain_rows

            x = self.tables.nodes[self.tables.feature_cols].to_numpy(np.float32)[i : i + 1]
            row = shap_explain_rows(lm.model, x, self.tables.feature_cols)[0]
            return {"model": model, "tx_id": int(tx_id), **row}
        from graph_fraud.explain import gnn_explain_node

        return {
            "model": model,
            **gnn_explain_node(
                lm.model, self._graph, i, self.tables.feature_cols, self.tx, num_epochs=100
            ),
        }


def create_app(data_dir: str | Path, artifacts_dir: str | Path | None = None) -> FastAPI:
    store = GraphStore(data_dir, artifacts_dir)
    app = FastAPI(title="graph-fraud-ui")
    app.state.store = store

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/meta")
    def meta():
        return JSONResponse(store.meta())

    @app.get("/api/graph/{timestep}")
    def graph(timestep: int):
        try:
            return JSONResponse(store.graph(timestep))
        except KeyError as exc:
            raise HTTPException(404, f"no time step {timestep}") from exc

    @app.get("/api/node/{tx_id}")
    def node(tx_id: int):
        try:
            return JSONResponse(store.node(tx_id))
        except KeyError as exc:
            raise HTTPException(404, f"unknown transaction {tx_id}") from exc

    @app.get("/api/explain/{tx_id}")
    def explain(tx_id: int, model: str = "graphsage"):
        if tx_id not in store.pos.index:
            raise HTTPException(404, f"unknown transaction {tx_id}")
        try:
            return JSONResponse(store.explain(tx_id, model))
        except KeyError as exc:
            raise HTTPException(404, f"model {model} not loaded") from exc

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument(
        "--artifacts", default="artifacts", help="directory with one model dir per model (optional)"
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8050")))
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    import uvicorn

    uvicorn.run(create_app(a.data_dir, a.artifacts), host=a.host, port=a.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
