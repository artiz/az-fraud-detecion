"""Endpoint contract: payload parsing, scoring for both model types, FastAPI routes."""

import numpy as np
import pytest

from graph_fraud.artifacts import LoadedModel
from graph_fraud.data import EllipticTables
from graph_fraud.inference import BadRequest, parse_payload, score_payload
from graph_fraud.payload import build_request, k_hop_nodes


def test_parse_payload_validation():
    with pytest.raises(BadRequest, match="nodes"):
        parse_payload({}, 3)
    with pytest.raises(BadRequest, match="expected 3"):
        parse_payload({"nodes": [{"id": "a", "features": [1, 2]}]}, 3)
    with pytest.raises(BadRequest, match="duplicate"):
        parse_payload(
            {"nodes": [{"id": "a", "features": [1, 2, 3]}, {"id": "a", "features": [1, 2, 3]}]}, 3
        )
    with pytest.raises(BadRequest, match="targets"):
        parse_payload({"nodes": [{"id": "a", "features": [1, 2, 3]}], "targets": ["zzz"]}, 3)
    ids, feats, src, dst, targets = parse_payload(
        {
            "nodes": [{"id": 1, "features": [1, 2, 3]}, {"id": "2", "features": [0, 0, 0]}],
            "edges": [[1, "2"], ["2", "nope"]],
        },
        3,
    )
    assert (
        ids == ["1", "2"]
        and feats.shape == (2, 3)
        and src.tolist() == [0]
        and dst.tolist() == [1]
        and targets == ["1", "2"]
    )


def test_k_hop_nodes_direction():
    src = np.array([0, 1, 2])
    dst = np.array([1, 2, 3])
    assert k_hop_nodes(src, dst, [3], k=1, bidirectional=False) == {2, 3}
    assert k_hop_nodes(src, dst, [3], k=2, bidirectional=False) == {1, 2, 3}
    assert k_hop_nodes(src, dst, [0], k=1, bidirectional=False) == {0}
    assert k_hop_nodes(src, dst, [0], k=1, bidirectional=True) == {0, 1}


def test_gnn_payload_score_matches_full_graph(processed_dir, gnn_model_dir):
    """Scoring a 2-hop neighbourhood through the endpoint path == scoring inside the full graph."""
    from graph_fraud.graph import build_dgl_graph

    tables = EllipticTables.load(processed_dir)
    lm = LoadedModel(gnn_model_dir)
    g, _ = build_dgl_graph(tables, scaler=lm.scaler, bidirectional=lm.bidirectional)
    full = lm.model.predict_proba(g, g.ndata["feat"]).numpy()
    tx_ids = tables.nodes.tx_id.to_numpy()
    targets = [int(tx_ids[i]) for i in (5, 77, 150)]
    resp = score_payload(
        lm,
        build_request(
            tables, targets, num_hops=lm.model.num_layers, bidirectional=lm.bidirectional
        ),
    )
    assert resp["model_type"] == "graphsage" and set(resp["scores"]) == {str(t) for t in targets}
    for t in targets:
        pos = int(np.nonzero(tx_ids == t)[0][0])
        assert abs(resp["scores"][str(t)] - full[pos]) < 1e-4
        assert resp["flags"][str(t)] == int(full[pos] >= lm.threshold)


def test_xgb_payload_ignores_edges(processed_dir, xgb_model_dir):
    tables = EllipticTables.load(processed_dir)
    lm = LoadedModel(xgb_model_dir)
    req = build_request(tables, [int(tables.nodes.tx_id.iat[3])], num_hops=1)
    a = score_payload(lm, req)
    req["edges"] = []
    b = score_payload(lm, req)
    assert a["scores"] == b["scores"] and a["model_type"] == "xgboost"


def test_serve_routes(gnn_model_dir, processed_dir):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import serve

    client = TestClient(serve.create_app(str(gnn_model_dir)))
    assert client.get("/ping").status_code == 200
    tables = EllipticTables.load(processed_dir)
    req = build_request(tables, [int(tables.nodes.tx_id.iat[0])], num_hops=2)
    r = client.post("/invocations", json=req)
    assert r.status_code == 200 and r.json()["model_type"] == "graphsage"
    assert client.post("/invocations", json={"nodes": []}).status_code == 400
    assert (
        client.post(
            "/invocations", content=b"not json", headers={"content-type": "application/json"}
        ).status_code
        == 415
    )
