#!/usr/bin/env python
"""`train` entrypoint of the SageMaker container (``docker run <image> train``).

A ~40-line replacement for the sagemaker-training toolkit: it reads the SageMaker training
contract from /opt/ml, turns hyperparameters into CLI flags for ``scripts/train.py`` and
reports failures the way SageMaker expects (``/opt/ml/output/failure``). The SDK JSON-encodes
hyperparameter values (``"graphsage"`` arrives as ``"\\"graphsage\\""``), so they are decoded.
Any extra command-line arguments are appended, so ``docker run <image> train --model xgboost``
works locally without a hyperparameters.json.

Contract: https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ML = Path(os.environ.get("SM_ML_ROOT", "/opt/ml"))
CODE_DIR = Path(__file__).resolve().parent


def hyperparameters_to_args(hps: dict[str, str]) -> list[str]:
    args: list[str] = []
    for key, raw in sorted(hps.items()):
        if key.startswith("sagemaker_"):  # SDK bookkeeping (program, submit dir, region ...)
            continue
        try:
            value = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            value = raw
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
            continue
        args += [f"--{key}", json.dumps(value) if isinstance(value, (list, dict)) else str(value)]
    return args


def main() -> int:
    hp_file = ML / "input" / "config" / "hyperparameters.json"
    hps = json.loads(hp_file.read_text()) if hp_file.exists() else {}
    env = dict(os.environ)
    env.setdefault("SM_MODEL_DIR", str(ML / "model"))
    env.setdefault("SM_OUTPUT_DATA_DIR", str(ML / "output" / "data"))
    channels = ML / "input" / "data"
    if channels.is_dir():
        for ch in channels.iterdir():
            env.setdefault(f"SM_CHANNEL_{ch.name.upper()}", str(ch))
    env.setdefault("SM_NUM_GPUS", env.get("SM_NUM_GPUS", "0"))
    # Extra command-line arguments (``docker run <image> train --model xgboost``) win over the JSON.
    cmd = [sys.executable, str(CODE_DIR / "train.py"), *hyperparameters_to_args(hps), *sys.argv[1:]]
    print("running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        out = ML / "output"
        out.mkdir(parents=True, exist_ok=True)
        (out / "failure").write_text(
            f"train.py exited with code {proc.returncode}; see CloudWatch logs"
        )
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
