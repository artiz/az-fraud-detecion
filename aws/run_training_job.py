#!/usr/bin/env python
"""Launch a single SageMaker training job with the custom DGL container (evening 2, step 2).

    python aws/run_training_job.py --model graphsage --data-s3-uri s3://<bucket>/elliptic/processed/
    python aws/run_training_job.py --model xgboost   --data-s3-uri s3://<bucket>/elliptic/processed/ --instance-type ml.m5.large

The processed data can be produced locally (scripts/prepare_data.py) and synced with
`aws s3 sync data/processed s3://<bucket>/elliptic/processed/`, or by the pipeline.
"""

from __future__ import annotations

import argparse
import time

import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput

from common import DEFAULT_REGION, DEFAULT_STACK, image_uri, stack_outputs

METRIC_DEFINITIONS = [
    {"Name": "test:pr_auc", "Regex": r"test_pr_auc=([0-9\.]+)"},
    {"Name": "test:f1", "Regex": r"test_f1=([0-9\.]+)"},
    {"Name": "val:pr_auc", "Regex": r"val PR-AUC ([0-9\.]+)"},
    {"Name": "train:loss", "Regex": r"loss ([0-9\.]+) val"},
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["xgboost", "graphsage"], default="graphsage")
    p.add_argument("--data-s3-uri", required=True)
    p.add_argument("--stack", default=DEFAULT_STACK)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--image-tag", default="latest")
    p.add_argument("--instance-type", default="ml.m5.2xlarge")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--wait", action="store_true")
    a = p.parse_args()

    out = stack_outputs(a.stack, a.region)
    session = sagemaker.Session(boto_session=sagemaker.Session().boto_session)
    hps = {"model": a.model}
    if a.epochs is not None:
        hps["graphsage.epochs"] = a.epochs
    est = Estimator(
        image_uri=image_uri(out, a.image_tag),
        role=out["SageMakerRoleArn"],
        instance_count=1,
        instance_type=a.instance_type,
        volume_size=30,
        max_run=2 * 3600,
        output_path=f"s3://{out['BucketName']}/models/{a.model}",
        base_job_name=f"graph-fraud-{a.model}",
        hyperparameters=hps,
        metric_definitions=METRIC_DEFINITIONS,
        sagemaker_session=session,
        environment={"DGLBACKEND": "pytorch"},
    )
    est.fit(
        {"data": TrainingInput(a.data_s3_uri, content_type="application/x-parquet")},
        wait=a.wait,
        logs=a.wait,
    )
    job = est.latest_training_job.name
    print(f"TRAINING_JOB={job}")
    if a.wait:
        print(f"MODEL_DATA_URL={est.model_data}")
    else:
        print("poll with: aws sagemaker describe-training-job --training-job-name", job)
        time.sleep(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
