"""The SageMaker pipeline definition is built offline against mocked AWS (moto).

Catches SDK API drift (parameter names, step wiring) without touching an account.
"""

import json
import os

import pytest

sagemaker = pytest.importorskip("sagemaker")
moto = pytest.importorskip("moto")


@pytest.fixture
def aws_env(monkeypatch):
    for k, v in {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_SECURITY_TOKEN": "testing",
        "AWS_SESSION_TOKEN": "testing",
        "AWS_DEFAULT_REGION": "eu-central-1",
        "SAGEMAKER_SUPPRESS_V2_WARNING": "1",
    }.items():
        monkeypatch.setenv(k, v)
    with moto.mock_aws():
        yield


def test_definition_has_four_steps_and_gate(aws_env):
    import boto3
    from sagemaker.workflow.pipeline_context import PipelineSession

    import pipeline as pl

    bucket = "graph-fraud-123456789012-eu-central-1"
    boto3.client("s3", region_name="eu-central-1").create_bucket(
        Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": "eu-central-1"}
    )
    session = PipelineSession(
        boto_session=boto3.Session(region_name="eu-central-1"), default_bucket=bucket
    )
    p = pl.build_pipeline(
        name="graph-fraud-pipeline",
        role="arn:aws:iam::123456789012:role/graph-fraud-sagemaker-eu-central-1",
        bucket=bucket,
        image="123456789012.dkr.ecr.eu-central-1.amazonaws.com/graph-fraud:latest",
        model_package_group="graph-fraud-models",
        session=session,
    )
    d = json.loads(p.definition())
    steps = {s["Name"]: s for s in d["Steps"]}
    assert list(steps) == ["PrepareData", "TrainModel", "EvaluateModel", "CheckPrAuc"]
    assert (
        steps["PrepareData"]["Type"] == "Processing"
        and steps["TrainModel"]["Type"] == "Training"
        and steps["EvaluateModel"]["Type"] == "Processing"
    )
    gate = steps["CheckPrAuc"]
    assert gate["Type"] == "Condition"
    cond = gate["Arguments"]["Conditions"][0]
    assert cond["Type"] == "GreaterThanOrEqualTo"
    assert cond["LeftValue"]["Std:JsonGet"]["Path"] == "binary_classification_metrics.pr_auc.value"
    if_names = [s["Name"] for s in gate["Arguments"]["IfSteps"]]
    assert any(n.startswith("RegisterModel") for n in if_names)
    assert gate["Arguments"]["ElseSteps"][0]["Type"] == "Fail"
    params = {prm["Name"] for prm in d["Parameters"]}
    assert {"RawDataS3Uri", "ModelType", "Epochs", "MinPrAuc", "TrainingInstanceType"} <= params
    # training step consumes the processing output and passes the model type through
    train_args = steps["TrainModel"]["Arguments"]
    assert "PrepareData" in json.dumps(train_args["InputDataConfig"])
    assert train_args["HyperParameters"]["model"] == {"Get": "Parameters.ModelType"}
    assert "ProjectName" not in os.environ  # sanity: no real stack lookup happened
