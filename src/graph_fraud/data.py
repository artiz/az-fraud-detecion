"""Elliptic dataset: download, load, clean and split *by time*.

Facts about the raw data that the code below relies on (checked in tests):

* ``elliptic_txs_features.csv`` has no header: column 0 is ``txId``, column 1 is the
  time step (1..49, roughly two weeks apart), then 93 local + 72 aggregated features.
* ``elliptic_txs_classes.csv`` maps ``txId`` to ``1`` (illicit), ``2`` (licit) or
  ``unknown``. Only ~23% of nodes are labelled.
* ``elliptic_txs_edgelist.csv`` lists directed payment flows ``txId1 -> txId2``.
  Every edge connects two nodes of the **same time step**, so the graph is a union of
  49 disconnected snapshots. A time-based split therefore has no edge leakage.
"""

from __future__ import annotations

import json
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

RAW_FILES = (
    "elliptic_txs_features.csv",
    "elliptic_txs_edgelist.csv",
    "elliptic_txs_classes.csv",
)
# PyTorch Geometric mirrors the Kaggle files (ellipticco/elliptic-data-set) as zips.
MIRROR_URL = "https://data.pyg.org/datasets/elliptic/"

NUM_LOCAL_FEATURES = 93
NUM_FEATURES = 165
LABEL_UNKNOWN, LABEL_LICIT, LABEL_ILLICIT = -1, 0, 1
SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST, SPLIT_NONE = "train", "val", "test", "none"


def feature_names(feature_set: str = "all") -> list[str]:
    n = NUM_LOCAL_FEATURES if feature_set == "local" else NUM_FEATURES
    return [f"f{i}" for i in range(n)]


def download_raw(raw_dir: str | Path, url: str = MIRROR_URL) -> Path:
    """Fetch the three CSVs (≈150 MB zipped, ≈700 MB unzipped) if missing."""
    import urllib.request

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in RAW_FILES:
        target = raw_dir / name
        if target.exists():
            continue
        zip_path = raw_dir / f"{name}.zip"
        log.info("downloading %s", url + zip_path.name)
        urllib.request.urlretrieve(url + zip_path.name, zip_path)  # noqa: S310 - fixed https URL
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract(name, raw_dir)
        zip_path.unlink()
    return raw_dir


@dataclass
class EllipticTables:
    """Cleaned tabular form of the dataset, ready for a graph or a tabular model."""

    nodes: pd.DataFrame  # columns: tx_id, timestep, label, split, f0..f164 ; row order = node id
    edges: pd.DataFrame  # columns: src, dst  (0-based positional node ids)
    feature_cols: list[str]

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    def save(self, out_dir: str | Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.nodes.to_parquet(out_dir / "nodes.parquet", index=False)
        self.edges.to_parquet(out_dir / "edges.parquet", index=False)
        meta = {
            "feature_cols": self.feature_cols,
            "num_nodes": int(self.num_nodes),
            "num_edges": int(len(self.edges)),
            "split_counts": {
                s: {
                    "nodes": int((self.nodes.split == s).sum()),
                    "illicit": int(((self.nodes.split == s) & (self.nodes.label == 1)).sum()),
                    "licit": int(((self.nodes.split == s) & (self.nodes.label == 0)).sum()),
                }
                for s in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)
            },
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        return out_dir

    @classmethod
    def load(cls, in_dir: str | Path) -> EllipticTables:
        in_dir = Path(in_dir)
        meta = json.loads((in_dir / "meta.json").read_text())
        nodes = pd.read_parquet(in_dir / "nodes.parquet")
        # pandas 3 reads strings back as StringDtype; keep plain object so the frame equals the built one.
        nodes["split"] = nodes["split"].astype(object)
        return cls(
            nodes=nodes,
            edges=pd.read_parquet(in_dir / "edges.parquet"),
            feature_cols=meta["feature_cols"],
        )


def load_raw(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    features = pd.read_csv(raw_dir / "elliptic_txs_features.csv", header=None)
    features.columns = ["tx_id", "timestep", *[f"f{i}" for i in range(features.shape[1] - 2)]]
    classes = pd.read_csv(raw_dir / "elliptic_txs_classes.csv")
    classes.columns = ["tx_id", "class"]
    edges = pd.read_csv(raw_dir / "elliptic_txs_edgelist.csv")
    edges.columns = ["tx_id_src", "tx_id_dst"]
    return features, classes, edges


def assign_split(timestep: pd.Series, data_cfg: dict) -> pd.Series:
    """Map each time step to train / val / test according to inclusive ranges."""
    split = pd.Series(SPLIT_NONE, index=timestep.index, dtype=object)
    for name, key in (
        (SPLIT_TRAIN, "train_timesteps"),
        (SPLIT_VAL, "val_timesteps"),
        (SPLIT_TEST, "test_timesteps"),
    ):
        lo, hi = data_cfg[key]
        split[(timestep >= lo) & (timestep <= hi)] = name
    return split


def build_tables(
    features: pd.DataFrame,
    classes: pd.DataFrame,
    edges: pd.DataFrame,
    data_cfg: dict,
) -> EllipticTables:
    """Join raw frames into a positional node table and a positional edge list."""
    feature_set = data_cfg.get("feature_set", "all")
    cols = feature_names(feature_set)

    nodes = features.merge(classes, on="tx_id", how="left", validate="one_to_one")
    label_map = {"1": LABEL_ILLICIT, "2": LABEL_LICIT, "unknown": LABEL_UNKNOWN}
    timestep = nodes["timestep"].astype(np.int16)
    nodes = nodes.assign(
        label=nodes["class"].astype(str).map(label_map).fillna(LABEL_UNKNOWN).astype(np.int8),
        timestep=timestep,
        split=assign_split(timestep, data_cfg),
    )
    # Deterministic node ids: sort by time then tx_id so that a time range is a contiguous block.
    nodes = nodes.sort_values(["timestep", "tx_id"], kind="mergesort").reset_index(drop=True)
    nodes = nodes[["tx_id", "timestep", "label", "split", *cols]]
    nodes[cols] = nodes[cols].astype(np.float32)

    id_to_pos = pd.Series(np.arange(len(nodes), dtype=np.int64), index=nodes["tx_id"].to_numpy())
    src = id_to_pos.reindex(edges["tx_id_src"].to_numpy()).to_numpy()
    dst = id_to_pos.reindex(edges["tx_id_dst"].to_numpy()).to_numpy()
    keep = ~(np.isnan(src) | np.isnan(dst))
    if (~keep).any():
        log.warning("dropping %d edges whose endpoints have no features", int((~keep).sum()))
    edge_df = pd.DataFrame({"src": src[keep].astype(np.int64), "dst": dst[keep].astype(np.int64)})
    edge_df = edge_df.drop_duplicates().reset_index(drop=True)

    ts = nodes["timestep"].to_numpy()
    cross_time = int((ts[edge_df.src.to_numpy()] != ts[edge_df.dst.to_numpy()]).sum())
    if cross_time:
        # Would break the "time split has no edge leakage" guarantee; fail loudly.
        raise ValueError(f"{cross_time} edges connect different time steps; split is not leak-free")

    return EllipticTables(nodes=nodes, edges=edge_df, feature_cols=cols)


def prepare(
    raw_dir: str | Path, out_dir: str | Path, data_cfg: dict, download: bool = True
) -> EllipticTables:
    """End-to-end: (download) -> load -> clean -> split -> parquet. Used by the SageMaker processing step."""
    if download:
        download_raw(raw_dir)
    tables = build_tables(*load_raw(raw_dir), data_cfg=data_cfg)
    tables.save(out_dir)
    return tables


def split_masks(nodes: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean masks of *labelled* nodes per split (unknown nodes never enter a loss or a metric)."""
    labelled = nodes["label"].to_numpy() != LABEL_UNKNOWN
    split = nodes["split"].to_numpy()
    return {s: labelled & (split == s) for s in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)}
