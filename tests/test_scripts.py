"""evaluate.py and explain.py run end to end on the synthetic artifacts."""

import json
import tarfile


def test_evaluate_from_dir_and_tarball(xgb_model_dir, gnn_model_dir, processed_dir, tmp_path):
    import evaluate

    evaluate.main(
        [
            "--model-dir",
            str(gnn_model_dir),
            "--data-dir",
            str(processed_dir),
            "--out-dir",
            str(tmp_path / "gnn"),
        ]
    )
    rep = json.loads((tmp_path / "gnn" / "evaluation.json").read_text())
    assert rep["model_type"] == "graphsage"
    assert 0.0 <= rep["binary_classification_metrics"]["pr_auc"]["value"] <= 1.0
    # SageMaker mounts model.tar.gz, not a directory
    tar_dir = tmp_path / "model"
    tar_dir.mkdir()
    with tarfile.open(tar_dir / "model.tar.gz", "w:gz") as tf:
        for f in xgb_model_dir.iterdir():
            tf.add(f, arcname=f.name)
    evaluate.main(
        [
            "--model-dir",
            str(tar_dir),
            "--data-dir",
            str(processed_dir),
            "--out-dir",
            str(tmp_path / "xgb"),
        ]
    )
    rep = json.loads((tmp_path / "xgb" / "evaluation.json").read_text())
    assert rep["model_type"] == "xgboost" and rep["n_illicit"] > 0


def test_explain_both_models(xgb_model_dir, gnn_model_dir, processed_dir, tmp_path):
    import explain

    explain.main(
        [
            "--model-dir",
            str(xgb_model_dir),
            "--data-dir",
            str(processed_dir),
            "--out",
            str(tmp_path / "x.json"),
            "--num-nodes",
            "2",
        ]
    )
    x = json.loads((tmp_path / "x.json").read_text())
    assert (
        len(x["transactions"]) == 2 and x["global_importance"][0]["feature"] == "f0"
    )  # f0 is the planted signal
    t = x["transactions"][0]
    assert abs(t["probability"] - t["score"]) < 1e-4  # SHAP values sum back to the model output

    explain.main(
        [
            "--model-dir",
            str(gnn_model_dir),
            "--data-dir",
            str(processed_dir),
            "--out",
            str(tmp_path / "g.json"),
            "--num-nodes",
            "1",
            "--epochs",
            "20",
        ]
    )
    gx = json.loads((tmp_path / "g.json").read_text())
    tr = gx["transactions"][0]
    assert tr["subgraph"]["num_nodes"] >= 1 and len(tr["top_features"]) == 10 and tr["top_edges"]
    assert all(0.0 <= e["weight"] <= 1.0 for e in tr["top_edges"])
