import numpy as np
import torch

from graph_fraud.graph import FeatureScaler, build_dgl_graph


def test_graph_shapes_and_masks(tables):
    g, scaler = build_dgl_graph(tables, bidirectional=True)
    assert g.num_nodes() == tables.num_nodes
    assert g.num_edges() == 2 * len(tables.edges) + tables.num_nodes  # reverse edges + self loops
    assert g.ndata["feat"].shape == (tables.num_nodes, len(tables.feature_cols))
    assert g.ndata["train_mask"].sum() > 0 and g.ndata["test_mask"].sum() > 0
    assert not (g.ndata["train_mask"] & g.ndata["test_mask"]).any()
    # train-period features are standardised with train statistics
    x_train = g.ndata["feat"][g.ndata["train_mask"]]
    assert abs(float(x_train.mean())) < 0.2


def test_no_message_path_from_train_to_test(tables):
    """The invariant that makes full-graph training equivalent to inductive training."""
    g, _ = build_dgl_graph(tables, bidirectional=True)
    src, dst = g.edges()
    ts = g.ndata["timestep"]
    assert torch.equal(ts[src], ts[dst])


def test_unidirectional_option(tables):
    g, _ = build_dgl_graph(tables, bidirectional=False, add_self_loops=False)
    assert g.num_edges() == len(tables.edges)


def test_scaler_roundtrip(tmp_path):
    x = np.random.default_rng(0).normal(3, 2, size=(50, 5)).astype(np.float32)
    x[:, 1] = 7.0  # constant column must not divide by zero
    s = FeatureScaler.fit(x)
    s.save(tmp_path / "s.json")
    s2 = FeatureScaler.load(tmp_path / "s.json")
    np.testing.assert_allclose(s.transform(x), s2.transform(x))
    assert np.isfinite(s2.transform(x)).all()
