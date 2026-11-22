"""The exploration UI's JSON API against the synthetic tables and the tiny trained models."""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client(processed_dir, xgb_model_dir, gnn_model_dir, tmp_path_factory):
    import app as ui_app

    artifacts = tmp_path_factory.mktemp("artifacts")
    (artifacts / "xgboost").symlink_to(xgb_model_dir, target_is_directory=True)
    (artifacts / "graphsage").symlink_to(gnn_model_dir, target_is_directory=True)
    return TestClient(ui_app.create_app(processed_dir, artifacts))


def test_index_and_meta(client):
    assert client.get("/").status_code == 200
    meta = client.get("/api/meta").json()
    assert [t["timestep"] for t in meta["timesteps"]] == list(range(1, 10))
    assert {"xgboost", "graphsage"} == set(meta["models"])
    row = meta["timesteps"][0]
    assert (
        row["nodes"] == row["illicit"] + row["licit"] + row["unknown"] and row["split"] == "train"
    )


def test_graph_payload(client):
    g = client.get("/api/graph/7").json()
    ids = {n["id"] for n in g["nodes"]}
    assert len(ids) == len(g["nodes"]) > 0
    assert all(s in ids and d in ids for s, d in g["edges"]), "edges stay inside the time step"
    assert all({"label", "in", "out", "s_xgboost", "s_graphsage"} <= n.keys() for n in g["nodes"])
    assert client.get("/api/graph/99").status_code == 404


def test_node_and_explanations(client):
    g = client.get("/api/graph/8").json()
    node = next(n for n in g["nodes"] if n["out"] > 0)
    d = client.get(f"/api/node/{node['id']}").json()
    assert (
        d["timestep"] == 8
        and len(d["features"]) == 165
        and set(d["scores"]) == {"xgboost", "graphsage"}
    )
    assert len(d["out_neighbors"]) == node["out"]
    x = client.get(f"/api/explain/{node['id']}?model=xgboost").json()
    assert abs(x["probability"] - d["scores"]["xgboost"]) < 1e-4
    gx = client.get(f"/api/explain/{node['id']}?model=graphsage").json()
    assert gx["tx_id"] == node["id"] and gx["top_edges"]
    assert client.get("/api/explain/1?model=graphsage").status_code == 404
    assert client.get(f"/api/explain/{node['id']}?model=nope").status_code == 404
