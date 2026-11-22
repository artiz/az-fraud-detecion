"""GraphSAGE + XGBoost: training runs, artifacts round-trip, sampled == full-graph in eval mode."""

import json

import numpy as np
import torch

from graph_fraud.artifacts import GNN_FILE, XGB_FILE, LoadedModel, read_meta
from graph_fraud.baseline import load_xgboost, predict_proba, save_xgboost, train_xgboost
from graph_fraud.data import split_masks
from graph_fraud.gnn import GraphSAGE, load_graphsage, save_graphsage, train_graphsage
from graph_fraud.graph import build_dgl_graph

GNN_CFG = {
    "hidden_dim": 16,
    "num_layers": 2,
    "dropout": 0.1,
    "aggregator": "mean",
    "fanouts": [5, 5],
    "batch_size": 64,
    "epochs": 2,
    "lr": 0.01,
    "weight_decay": 0.0,
    "illicit_class_weight": 0.7,
    "patience": 3,
}


def test_graphsage_trains_and_roundtrips(tables, tmp_path):
    g, _ = build_dgl_graph(tables)
    model, summary = train_graphsage(g, GNN_CFG, seed=1)
    assert summary["best_epoch"] >= 1 and 0.0 <= summary["threshold"] <= 1.0
    assert len(summary["history"]) == 2
    p1 = model.predict_proba(g, g.ndata["feat"])
    save_graphsage(model, tmp_path / GNN_FILE)
    back = load_graphsage(tmp_path / GNN_FILE)
    torch.testing.assert_close(back.predict_proba(g, g.ndata["feat"]), p1)


def test_sampled_forward_matches_full_graph_with_full_fanout(tables):
    """Blocks with fanout -1 must reproduce the full-graph forward exactly (eval mode)."""
    import dgl.dataloading as dl

    g, _ = build_dgl_graph(tables)
    model = GraphSAGE(g.ndata["feat"].shape[1], 8, 2, dropout=0.0).eval()
    seeds = torch.arange(0, 40)
    loader = dl.DataLoader(g, seeds, dl.NeighborSampler([-1, -1]), batch_size=40, shuffle=False)
    _, out_nodes, blocks = next(iter(loader))
    with torch.no_grad():
        sampled = model(blocks, blocks[0].srcdata["feat"])
        full = model(g, g.ndata["feat"])[out_nodes]
    torch.testing.assert_close(sampled, full, atol=1e-5, rtol=1e-4)


def test_graphsage_rejects_bad_fanouts(tables):
    g, _ = build_dgl_graph(tables)
    try:
        train_graphsage(g, dict(GNN_CFG, fanouts=[5]), seed=0)
    except ValueError as exc:
        assert "fanouts" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_xgboost_roundtrip_keeps_best_iteration(tables, tmp_path):
    masks = split_masks(tables.nodes)
    x = tables.nodes[tables.feature_cols].to_numpy(np.float32)
    y = tables.nodes.label.to_numpy()
    cfg = {
        "n_estimators": 60,
        "max_depth": 3,
        "learning_rate": 0.2,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "min_child_weight": 1,
        "early_stopping_rounds": 10,
    }
    booster, summary = train_xgboost(
        x[masks["train"]], y[masks["train"]], x[masks["val"]], y[masks["val"]], cfg, seed=0
    )
    p1 = predict_proba(booster, x[masks["test"]])
    save_xgboost(booster, tmp_path / XGB_FILE)
    back = load_xgboost(tmp_path / XGB_FILE)
    assert back.best_iteration == summary["best_iteration"]
    np.testing.assert_allclose(predict_proba(back, x[masks["test"]]), p1, rtol=1e-6)
    assert summary["best_val_pr_auc"] > 0.5  # feature 0 carries signal


def test_train_script_artifacts(xgb_model_dir, gnn_model_dir):
    for d, kind, fname in (
        (xgb_model_dir, "xgboost", XGB_FILE),
        (gnn_model_dir, "graphsage", GNN_FILE),
    ):
        meta = read_meta(d)
        assert meta["model_type"] == kind and len(meta["feature_cols"]) == 165
        assert (d / fname).exists()
        m = json.loads((d / "metrics.json").read_text())
        assert {"val", "test", "threshold", "training"} <= m.keys()
        assert m["test"]["n_illicit"] > 0
        lm = LoadedModel(d)
        assert lm.model_type == kind and lm.threshold == meta["threshold"]
    assert (gnn_model_dir / "scaler.json").exists()
