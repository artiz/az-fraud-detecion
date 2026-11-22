"""Experiment configuration: YAML defaults + dotted CLI overrides.

SageMaker passes hyperparameters as ``--key value`` arguments to the training
script, so the same override mechanism (``--graphsage.epochs 5``) serves both
local runs and training jobs.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def _coerce(value: str) -> Any:
    """Parse a CLI string into YAML/JSON-ish Python value (``"[15, 10]"`` -> list)."""
    if not isinstance(value, str):
        return value
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``config`` with dotted keys (``graphsage.lr``) overridden."""
    out = copy.deepcopy(config)
    for dotted, raw in overrides.items():
        if raw is None:
            continue
        node = out
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(raw)
    return out


def dump_config(config: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(config, indent=2, sort_keys=True))
