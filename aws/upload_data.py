#!/usr/bin/env python
"""Upload the raw Elliptic CSVs to the project bucket (input of the pipeline's first step).

python aws/upload_data.py --raw-dir data/raw [--stack graph-fraud-infra]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import boto3

from common import DEFAULT_REGION, DEFAULT_STACK, stack_outputs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--stack", default=DEFAULT_STACK)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--prefix", default="elliptic/raw")
    a = p.parse_args()
    bucket = stack_outputs(a.stack, a.region)["BucketName"]
    s3 = boto3.client("s3", region_name=a.region)
    for f in sorted(Path(a.raw_dir).glob("elliptic_txs_*.csv")):
        key = f"{a.prefix}/{f.name}"
        print(f"uploading {f} -> s3://{bucket}/{key}")
        s3.upload_file(str(f.resolve()), bucket, key)
    print(f"RAW_DATA_S3_URI=s3://{bucket}/{a.prefix}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
