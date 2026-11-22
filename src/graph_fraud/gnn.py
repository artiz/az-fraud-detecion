"""GraphSAGE on DGL with mini-batch neighbour sampling.

The model exposes one ``forward(graph_or_blocks, feat, eweight=None)``:

* training/inference with ``dgl.dataloading.NeighborSampler`` passes a list of
  message-flow-graph *blocks*, one per layer;
* full-graph inference and ``dgl.nn.GNNExplainer`` pass a single ``DGLGraph`` (the
  explainer additionally passes learned edge weights through ``eweight``).
"""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv

from .metrics import best_f1_threshold, illicit_metrics

log = logging.getLogger(__name__)


class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        aggregator: str = "mean",
        num_classes: int = 2,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [num_classes]
        self.layers = nn.ModuleList(
            SAGEConv(dims[i], dims[i + 1], aggregator) for i in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.hparams = {
            "in_dim": in_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "aggregator": aggregator,
            "num_classes": num_classes,
        }

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def forward(self, graph, feat, eweight=None):
        blocks = graph if isinstance(graph, (list, tuple)) else [graph] * self.num_layers
        h = feat
        for i, (layer, block) in enumerate(zip(self.layers, blocks, strict=True)):
            h = layer(block, h, edge_weight=eweight)
            if i < self.num_layers - 1:
                h = self.dropout(F.relu(h))
        return h

    @torch.no_grad()
    def predict_proba(self, graph, feat) -> torch.Tensor:
        """Full-graph illicit probability for every node (exact, no sampling)."""
        self.eval()
        return torch.softmax(self(graph, feat), dim=1)[:, 1]


def _make_loader(
    g, seeds: torch.Tensor, fanouts: list[int], batch_size: int, shuffle: bool, seed: int
):
    import dgl.dataloading as dl

    sampler = dl.NeighborSampler(fanouts, prefetch_node_feats=["feat"], prefetch_labels=["label"])
    gen = torch.Generator()
    gen.manual_seed(seed)
    return dl.DataLoader(
        g,
        seeds,
        sampler,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        generator=gen,
        use_uva=False,
    )


def train_graphsage(g, cfg: dict, seed: int = 42, device: str = "cpu") -> tuple[GraphSAGE, dict]:
    """Mini-batch training with early stopping on validation PR-AUC.

    Returns the best model (by val PR-AUC) and a history/summary dict that includes the
    F1-optimal threshold selected on the validation period.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = g.to(device)
    fanouts = list(cfg["fanouts"])
    if len(fanouts) != cfg["num_layers"]:
        raise ValueError("graphsage.fanouts must have one entry per layer")

    model = GraphSAGE(
        g.ndata["feat"].shape[1],
        cfg["hidden_dim"],
        cfg["num_layers"],
        cfg["dropout"],
        cfg["aggregator"],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    w1 = float(cfg["illicit_class_weight"])
    class_weight = torch.tensor([1.0 - w1, w1], device=device)

    train_seeds = torch.nonzero(g.ndata["train_mask"], as_tuple=True)[0]
    val_mask = g.ndata["val_mask"]
    y_val = g.ndata["label"][val_mask].cpu().numpy()
    loader = _make_loader(g, train_seeds, fanouts, cfg["batch_size"], shuffle=True, seed=seed)

    best = {"pr_auc": -1.0, "epoch": -1, "state": None}
    history = []
    bad_epochs = 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0, total_loss, n_batches = time.time(), 0.0, 0
        for _input_nodes, _output_nodes, blocks in loader:
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            logits = model(blocks, x)
            loss = F.cross_entropy(logits, y, weight=class_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss)
            n_batches += 1

        scores = model.predict_proba(g, g.ndata["feat"]).cpu().numpy()
        val_scores = scores[val_mask.cpu().numpy()]
        thr = best_f1_threshold(y_val, val_scores)
        val = illicit_metrics(y_val, val_scores, thr)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(n_batches, 1),
            "val_pr_auc": val["pr_auc"],
            "val_f1": val["illicit_f1"],
            "seconds": time.time() - t0,
        }
        history.append(row)
        log.info(
            "epoch %02d loss %.4f val PR-AUC %.4f val F1 %.4f (%.1fs)",
            epoch,
            row["train_loss"],
            val["pr_auc"],
            val["illicit_f1"],
            row["seconds"],
        )

        if val["pr_auc"] > best["pr_auc"]:
            best = {
                "pr_auc": val["pr_auc"],
                "epoch": epoch,
                "state": copy.deepcopy(model.state_dict()),
                "threshold": thr,
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                log.info("early stopping at epoch %d (best epoch %d)", epoch, best["epoch"])
                break

    model.load_state_dict(best["state"])
    model.eval()
    summary = {
        "best_epoch": best["epoch"],
        "best_val_pr_auc": best["pr_auc"],
        "threshold": best["threshold"],
        "history": history,
    }
    return model, summary


def save_graphsage(model: GraphSAGE, path: str | Path) -> None:
    torch.save({"state_dict": model.state_dict(), "hparams": model.hparams}, path)


def load_graphsage(path: str | Path, device: str = "cpu") -> GraphSAGE:
    ckpt = torch.load(path, map_location=device)
    model = GraphSAGE(**ckpt["hparams"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
