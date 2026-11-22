"""Synthetic Elliptic-shaped data so the whole pipeline runs in seconds without the 700 MB download."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts", ROOT / "aws", ROOT / "ui"):
    sys.path.insert(0, str(p))

from graph_fraud.data import NUM_FEATURES, EllipticTables, build_tables  # noqa: E402

DATA_CFG = {
    "train_timesteps": [1, 4],
    "val_timesteps": [5, 6],
    "test_timesteps": [7, 9],
    "feature_set": "all",
    "bidirectional_edges": True,
}


def make_raw(num_timesteps: int = 9, nodes_per_step: int = 120, seed: int = 0):
    """Raw frames in the exact shape of the Kaggle CSVs, with a learnable signal.

    Illicit nodes have a shifted feature 0 *and* illicit neighbours, so both a tabular model
    and a GNN can find something in a few epochs.
    """
    rng = np.random.default_rng(seed)
    rows, cls_rows, edge_rows = [], [], []
    tx = 1000
    for ts in range(1, num_timesteps + 1):
        ids = np.arange(tx, tx + nodes_per_step)
        tx += nodes_per_step
        illicit = rng.random(nodes_per_step) < 0.15
        feats = rng.normal(size=(nodes_per_step, NUM_FEATURES)).astype(np.float32)
        feats[illicit, 0] += 1.5
        for i, nid in enumerate(ids):
            rows.append([nid, ts, *feats[i]])
            r = rng.random()
            cls_rows.append([nid, "unknown" if r < 0.3 else ("1" if illicit[i] else "2")])
        # homophilous edges inside the time step
        for i in range(nodes_per_step):
            for _ in range(2):
                pool = ids[illicit] if (illicit[i] and rng.random() < 0.7) else ids
                j = int(rng.choice(pool))
                if j != ids[i]:
                    edge_rows.append([ids[i], j])
    features = pd.DataFrame(rows)
    classes = pd.DataFrame(cls_rows, columns=["txId", "class"])
    edges = pd.DataFrame(edge_rows, columns=["txId1", "txId2"]).drop_duplicates()
    return features, classes, edges


def make_tables(**kw) -> EllipticTables:
    features, classes, edges = make_raw(**kw)
    features.columns = ["tx_id", "timestep", *[f"f{i}" for i in range(NUM_FEATURES)]]
    classes.columns = ["tx_id", "class"]
    edges.columns = ["tx_id_src", "tx_id_dst"]
    return build_tables(features, classes, edges, DATA_CFG)


@pytest.fixture(scope="session")
def tables() -> EllipticTables:
    return make_tables()


@pytest.fixture(scope="session")
def processed_dir(tables, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("processed")
    tables.save(out)
    return out


@pytest.fixture(scope="session")
def xgb_model_dir(processed_dir, tmp_path_factory) -> Path:
    import train as train_script

    out = tmp_path_factory.mktemp("xgb")
    train_script.main(
        [
            "--model",
            "xgboost",
            "--data-dir",
            str(processed_dir),
            "--model-dir",
            str(out),
            "--xgboost.n_estimators",
            "40",
            "--xgboost.early_stopping_rounds",
            "10",
        ]
    )
    return out


@pytest.fixture(scope="session")
def gnn_model_dir(processed_dir, tmp_path_factory) -> Path:
    import train as train_script

    out = tmp_path_factory.mktemp("gnn")
    train_script.main(
        [
            "--model",
            "graphsage",
            "--data-dir",
            str(processed_dir),
            "--model-dir",
            str(out),
            "--graphsage.epochs",
            "3",
            "--graphsage.hidden_dim",
            "16",
            "--graphsage.batch_size",
            "64",
        ]
    )
    return out
