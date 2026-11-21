"""Build the DGL graph from the cleaned tables.

The graph holds *all* nodes (labelled and unknown) and all edges. Because every edge
stays inside one time step, training on the full graph with a loss mask is exactly
equivalent to inductive training on the train-period subgraph; the test period is
never reachable by message passing from a train seed. ``tests/test_graph.py`` pins
that invariant.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import LABEL_UNKNOWN, EllipticTables, split_masks


@dataclass
class FeatureScaler:
    """Train-period standardisation; stored next to the model so inference matches."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> FeatureScaler:
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-8] = 1.0
        return cls(mean=mean.astype(np.float32), scale=scale.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float32) - self.mean) / self.scale).astype(np.float32)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"mean": self.mean.tolist(), "scale": self.scale.tolist()})
        )

    @classmethod
    def load(cls, path: str | Path) -> FeatureScaler:
        d = json.loads(Path(path).read_text())
        return cls(mean=np.asarray(d["mean"], np.float32), scale=np.asarray(d["scale"], np.float32))


def make_edge_index(edges: pd.DataFrame, bidirectional: bool) -> tuple[torch.Tensor, torch.Tensor]:
    src = torch.as_tensor(edges["src"].to_numpy().copy(), dtype=torch.int64)
    dst = torch.as_tensor(edges["dst"].to_numpy().copy(), dtype=torch.int64)
    if bidirectional:
        src, dst = torch.cat([src, dst]), torch.cat([dst, src])
    return src, dst


def build_dgl_graph(
    tables: EllipticTables,
    scaler: FeatureScaler | None = None,
    bidirectional: bool = True,
    add_self_loops: bool = True,
):
    """Return a homogeneous DGLGraph with ``feat``, ``label``, ``timestep`` and split masks."""
    import dgl  # imported lazily so tabular code paths work without DGL

    nodes = tables.nodes
    x = nodes[tables.feature_cols].to_numpy(dtype=np.float32)
    masks = split_masks(nodes)
    if scaler is None:
        scaler = FeatureScaler.fit(x[masks["train"] | (nodes["split"].to_numpy() == "train")])
    x = scaler.transform(x)

    src, dst = make_edge_index(tables.edges, bidirectional)
    g = dgl.graph((src, dst), num_nodes=tables.num_nodes)
    if add_self_loops:
        # Isolated transactions (no neighbours in the snapshot) still get their own features.
        g = dgl.add_self_loop(g)
    g.ndata["feat"] = torch.from_numpy(x)
    g.ndata["label"] = torch.as_tensor(nodes["label"].to_numpy().copy(), dtype=torch.int64)
    g.ndata["timestep"] = torch.as_tensor(nodes["timestep"].to_numpy().copy(), dtype=torch.int64)
    for name, m in masks.items():
        g.ndata[f"{name}_mask"] = torch.from_numpy(m)
    g.ndata["labelled"] = g.ndata["label"] != LABEL_UNKNOWN
    return g, scaler
