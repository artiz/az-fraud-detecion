"""Build an endpoint request for one or more transactions from the cleaned tables.

The GNN needs the k-hop neighbourhood of each target (k = number of GraphSAGE layers).
In production a Kafka consumer would assemble this from a graph store; here it is
assembled from the parquet tables so the endpoint can be smoke-tested end to end.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .data import EllipticTables


def k_hop_nodes(
    edges_src: np.ndarray,
    edges_dst: np.ndarray,
    seeds: list[int],
    k: int,
    bidirectional: bool = True,
) -> set[int]:
    adj: dict[int, list[int]] = {}
    for s, d in zip(edges_src.tolist(), edges_dst.tolist(), strict=True):
        adj.setdefault(d, []).append(s)  # messages flow src -> dst, so predecessors matter
        if bidirectional:
            adj.setdefault(s, []).append(d)
    seen = set(seeds)
    frontier = deque((s, 0) for s in seeds)
    while frontier:
        node, depth = frontier.popleft()
        if depth == k:
            continue
        for nb in adj.get(node, ()):
            if nb not in seen:
                seen.add(nb)
                frontier.append((nb, depth + 1))
    return seen


def build_request(
    tables: EllipticTables, tx_ids: list[int], num_hops: int = 2, bidirectional: bool = True
) -> dict:
    tx_arr = tables.nodes["tx_id"].to_numpy()
    pos = {int(t): i for i, t in enumerate(tx_arr)}
    seeds = [pos[t] for t in tx_ids]
    src, dst = tables.edges["src"].to_numpy(), tables.edges["dst"].to_numpy()
    keep = k_hop_nodes(src, dst, seeds, num_hops, bidirectional)
    keep_arr = np.fromiter(sorted(keep), dtype=np.int64)
    feats = tables.nodes[tables.feature_cols].to_numpy(np.float32)
    mask = np.isin(src, keep_arr) & np.isin(dst, keep_arr)
    return {
        "nodes": [{"id": str(int(tx_arr[i])), "features": feats[i].tolist()} for i in keep_arr],
        "edges": [
            [str(int(tx_arr[s])), str(int(tx_arr[d]))]
            for s, d in zip(src[mask], dst[mask], strict=True)
        ],
        "targets": [str(t) for t in tx_ids],
    }
