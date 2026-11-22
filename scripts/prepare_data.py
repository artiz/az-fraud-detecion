#!/usr/bin/env python
"""Step 1 (SageMaker Processing or local): raw Elliptic CSVs -> cleaned parquet tables.

Local:      python scripts/prepare_data.py --raw-dir data/raw --out-dir data/processed
SageMaker:  the pipeline mounts the raw CSVs at /opt/ml/processing/input/raw and
            collects /opt/ml/processing/output/processed as the step output.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graph_fraud.config import apply_overrides, load_config  # noqa: E402
from graph_fraud.data import prepare  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", default="/opt/ml/processing/input/raw")
    p.add_argument("--out-dir", default="/opt/ml/processing/output/processed")
    p.add_argument("--config", default=None)
    p.add_argument(
        "--no-download", action="store_true", help="fail instead of downloading missing CSVs"
    )
    p.add_argument("--feature-set", dest="data.feature_set", default=None)
    args = vars(p.parse_args(argv))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = apply_overrides(
        load_config(args.pop("config")), {k: v for k, v in args.items() if "." in k}
    )
    tables = prepare(
        args["raw_dir"], args["out_dir"], cfg["data"], download=not args["no_download"]
    )
    logging.info(
        "wrote %d nodes / %d edges to %s", tables.num_nodes, len(tables.edges), args["out_dir"]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
