import numpy as np
import pandas as pd
import pytest

from conftest import DATA_CFG, make_raw
from graph_fraud.data import (
    LABEL_ILLICIT,
    LABEL_LICIT,
    LABEL_UNKNOWN,
    NUM_FEATURES,
    EllipticTables,
    assign_split,
    build_tables,
    split_masks,
)


def test_labels_and_splits(tables):
    n = tables.nodes
    assert set(n.label.unique()) == {LABEL_UNKNOWN, LABEL_LICIT, LABEL_ILLICIT}
    assert set(n.split.unique()) == {"train", "val", "test"}
    assert (n.loc[n.split == "train", "timestep"].between(1, 4)).all()
    assert (n.loc[n.split == "test", "timestep"].between(7, 9)).all()
    assert len(tables.feature_cols) == NUM_FEATURES
    assert n[tables.feature_cols].dtypes.eq(np.float32).all()


def test_node_order_is_time_sorted_and_edges_positional(tables):
    ts = tables.nodes.timestep.to_numpy()
    assert (np.diff(ts) >= 0).all()
    e = tables.edges
    assert (
        e.src.between(0, tables.num_nodes - 1).all()
        and e.dst.between(0, tables.num_nodes - 1).all()
    )
    assert (ts[e.src] == ts[e.dst]).all(), "edges must stay inside a time step"
    assert not e.duplicated().any()


def test_split_masks_exclude_unknown(tables):
    masks = split_masks(tables.nodes)
    unknown = tables.nodes.label.to_numpy() == LABEL_UNKNOWN
    for m in masks.values():
        assert not (m & unknown).any()
    assert sum(m.sum() for m in masks.values()) == (~unknown).sum()


def test_assign_split_gap_is_none():
    cfg = dict(DATA_CFG, train_timesteps=[1, 2], val_timesteps=[4, 4], test_timesteps=[6, 9])
    s = assign_split(pd.Series([1, 2, 3, 4, 5, 6]), cfg)
    assert s.tolist() == ["train", "train", "none", "val", "none", "test"]


def test_cross_timestep_edge_is_rejected():
    features, classes, edges = make_raw(num_timesteps=2, nodes_per_step=10)
    features.columns = ["tx_id", "timestep", *[f"f{i}" for i in range(NUM_FEATURES)]]
    classes.columns = ["tx_id", "class"]
    edges.columns = ["tx_id_src", "tx_id_dst"]
    bad = pd.concat([edges, pd.DataFrame({"tx_id_src": [1000], "tx_id_dst": [1010]})])
    with pytest.raises(ValueError, match="different time steps"):
        build_tables(features, classes, bad, DATA_CFG)


def test_local_feature_set():
    features, classes, edges = make_raw(num_timesteps=3, nodes_per_step=20)
    features.columns = ["tx_id", "timestep", *[f"f{i}" for i in range(NUM_FEATURES)]]
    classes.columns = ["tx_id", "class"]
    edges.columns = ["tx_id_src", "tx_id_dst"]
    tl = build_tables(features, classes, edges, dict(DATA_CFG, feature_set="local"))
    assert len(tl.feature_cols) == 93


def test_parquet_roundtrip(tables, tmp_path):
    tables.save(tmp_path)
    back = EllipticTables.load(tmp_path)
    pd.testing.assert_frame_equal(back.nodes, tables.nodes)
    pd.testing.assert_frame_equal(back.edges, tables.edges)
    assert back.feature_cols == tables.feature_cols
