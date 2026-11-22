#!/usr/bin/env python
"""Plot illicit-class F1 per test time step for every model in artifacts/ (README figure)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--out", default="docs/f1_per_timestep.png")
    a = p.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    counts = None
    for mdir in sorted(Path(a.artifacts).iterdir()):
        mfile = mdir / "metrics.json"
        if not mfile.exists():
            continue
        m = json.loads(mfile.read_text())
        rows = m["test"]["per_timestep"]
        ts = [r["timestep"] for r in rows]
        ax.plot(
            ts,
            [r["illicit_f1"] for r in rows],
            marker="o",
            label=f"{m['model_type']} (test F1 {m['test']['illicit_f1']:.2f}, PR-AUC {m['test']['pr_auc']:.2f})",
        )
        counts = counts or (ts, [r["n_illicit"] for r in rows])
    if counts:
        ax2.bar(*counts, color="#999999")
    ax.axvline(42.5, color="k", ls="--", lw=1)
    ax.text(42.7, 0.95, "dark market shutdown", fontsize=8, va="top")
    ax.set_ylabel("illicit F1")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    ax2.set_ylabel("# illicit")
    ax2.set_xlabel("time step (test period 35-49)")
    fig.suptitle(
        "Elliptic: illicit-class F1 per time step, threshold chosen on validation steps 30-34"
    )
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
