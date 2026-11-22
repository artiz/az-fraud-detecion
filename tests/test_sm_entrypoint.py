"""The container's `train` entrypoint honours the SageMaker /opt/ml contract (run against a temp root)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import sm_train_entrypoint as ep

ENTRYPOINT = Path(__file__).resolve().parents[1] / "scripts" / "sm_train_entrypoint.py"


def test_hyperparameters_are_json_decoded_and_sdk_keys_dropped():
    hps = {
        "model": '"graphsage"',  # SageMaker SDK JSON-encodes values
        "graphsage.epochs": "3",
        "graphsage.fanouts": "[5, 5]",
        "sagemaker_program": '"train.py"',
        "plain": "literal-string",
    }
    args = ep.hyperparameters_to_args(hps)
    assert args == [
        "--graphsage.epochs",
        "3",
        "--graphsage.fanouts",
        "[5, 5]",
        "--model",
        "graphsage",
        "--plain",
        "literal-string",
    ]


def test_entrypoint_trains_from_opt_ml_layout(processed_dir, tmp_path):
    ml = tmp_path / "opt_ml"
    (ml / "input" / "config").mkdir(parents=True)
    (ml / "input" / "data").mkdir()
    (ml / "input" / "data" / "data").symlink_to(processed_dir, target_is_directory=True)
    (ml / "input" / "config" / "hyperparameters.json").write_text(
        json.dumps(
            {
                "model": '"xgboost"',
                "xgboost.n_estimators": "20",
                "xgboost.early_stopping_rounds": "5",
            }
        )
    )
    env = dict(os.environ, SM_ML_ROOT=str(ml))
    proc = subprocess.run(
        [sys.executable, str(ENTRYPOINT)], env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert (ml / "model" / "model.ubj").exists() and (ml / "model" / "model_meta.json").exists()
    assert (ml / "output" / "data" / "metrics.json").exists()
    assert not (ml / "output" / "failure").exists()


def test_entrypoint_forwards_cli_args_without_hyperparameters(processed_dir, tmp_path):
    """`docker run <image> train --model xgboost`: no hyperparameters.json, flags come from argv."""
    ml = tmp_path / "opt_ml"
    (ml / "input" / "data").mkdir(parents=True)
    (ml / "input" / "data" / "data").symlink_to(processed_dir, target_is_directory=True)
    env = dict(os.environ, SM_ML_ROOT=str(ml))
    cmd = [
        sys.executable,
        str(ENTRYPOINT),
        "--model",
        "xgboost",
        "--xgboost.n_estimators",
        "10",
        "--xgboost.early_stopping_rounds",
        "5",
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert (ml / "model" / "model.ubj").exists()


def test_entrypoint_reports_failure_file(tmp_path):
    ml = tmp_path / "opt_ml"
    (ml / "input" / "config").mkdir(parents=True)
    (ml / "input" / "config" / "hyperparameters.json").write_text(
        json.dumps({"model": '"xgboost"'})
    )
    env = dict(os.environ, SM_ML_ROOT=str(ml), SM_CHANNEL_DATA=str(tmp_path / "missing"))
    proc = subprocess.run(
        [sys.executable, str(ENTRYPOINT)], env=env, capture_output=True, text=True
    )
    assert proc.returncode != 0
    assert "exited with code" in (ml / "output" / "failure").read_text()
