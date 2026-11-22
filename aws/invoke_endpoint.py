#!/usr/bin/env python
"""Score real Elliptic transactions through a deployed endpoint (evening 2, step 4).

    python aws/invoke_endpoint.py --endpoint-name graph-fraud-endpoint --tx-id 70335036 --tx-id 230425980

Builds the 2-hop neighbourhood of each transaction from data/processed and posts it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import DEFAULT_REGION  # noqa: E402
from graph_fraud.data import EllipticTables  # noqa: E402
from graph_fraud.payload import build_request  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint-name", required=True)
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--tx-id", type=int, action="append", required=True)
    p.add_argument("--num-hops", type=int, default=2)
    p.add_argument("--region", default=DEFAULT_REGION)
    a = p.parse_args()

    tables = EllipticTables.load(a.data_dir)
    payload = build_request(tables, a.tx_id, num_hops=a.num_hops)
    print(
        f"request: {len(payload['nodes'])} nodes, {len(payload['edges'])} edges, targets {payload['targets']}"
    )
    rt = boto3.client("sagemaker-runtime", region_name=a.region)
    resp = rt.invoke_endpoint(
        EndpointName=a.endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    print(json.dumps(json.loads(resp["Body"].read()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
