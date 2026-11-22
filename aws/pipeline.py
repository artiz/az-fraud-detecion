#!/usr/bin/env python
"""SageMaker Pipeline: prepare -> train -> evaluate -> (PR-AUC >= threshold) register.

    python aws/pipeline.py --raw-data-s3-uri s3://<bucket>/elliptic/raw/ --upsert --start
    python aws/pipeline.py --definition-only > pipeline.json   # inspect without AWS calls to SageMaker

Every step runs the same container (aws/docker/Dockerfile). Hyperparameters, the
registration threshold and instance types are pipeline parameters so a re-run with
other settings does not need a code change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

from common import DEFAULT_REGION, DEFAULT_STACK, image_uri, stack_outputs

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_pipeline(
    *,
    name: str,
    role: str,
    bucket: str,
    image: str,
    model_package_group: str,
    session: PipelineSession,
) -> Pipeline:
    # ---- parameters --------------------------------------------------------------------
    raw_data_uri = ParameterString("RawDataS3Uri", default_value=f"s3://{bucket}/elliptic/raw/")
    model_type = ParameterString("ModelType", default_value="graphsage")
    epochs = ParameterInteger("Epochs", default_value=30)
    min_pr_auc = ParameterFloat("MinPrAuc", default_value=0.60)
    processing_instance = ParameterString("ProcessingInstanceType", default_value="ml.m5.xlarge")
    training_instance = ParameterString("TrainingInstanceType", default_value="ml.m5.2xlarge")
    approval_status = ParameterString("ModelApprovalStatus", default_value="PendingManualApproval")

    # ---- step 1: prepare data ------------------------------------------------------------
    prep = ScriptProcessor(
        image_uri=image,
        command=["python3"],
        role=role,
        instance_count=1,
        instance_type=processing_instance,
        base_job_name="graph-fraud-prepare",
        sagemaker_session=session,
    )
    step_prepare = ProcessingStep(
        name="PrepareData",
        step_args=prep.run(
            code=str(REPO_ROOT / "scripts" / "prepare_data.py"),
            inputs=[
                ProcessingInput(source=raw_data_uri, destination="/opt/ml/processing/input/raw")
            ],
            outputs=[
                ProcessingOutput(
                    output_name="processed",
                    source="/opt/ml/processing/output/processed",
                    destination=f"s3://{bucket}/elliptic/processed",
                )
            ],
            arguments=["--no-download"],
        ),
    )
    processed_uri = step_prepare.properties.ProcessingOutputConfig.Outputs[
        "processed"
    ].S3Output.S3Uri

    # ---- step 2: train -----------------------------------------------------------------
    estimator = Estimator(
        image_uri=image,
        role=role,
        instance_count=1,
        instance_type=training_instance,
        volume_size=30,
        max_run=2 * 3600,
        output_path=f"s3://{bucket}/models",
        base_job_name="graph-fraud-train",
        hyperparameters={"model": model_type, "graphsage.epochs": epochs},
        metric_definitions=[
            {"Name": "test:pr_auc", "Regex": r"test_pr_auc=([0-9\.]+)"},
            {"Name": "test:f1", "Regex": r"test_f1=([0-9\.]+)"},
            {"Name": "val:pr_auc", "Regex": r"val PR-AUC ([0-9\.]+)"},
        ],
        environment={"DGLBACKEND": "pytorch"},
        sagemaker_session=session,
    )
    step_train = TrainingStep(
        name="TrainModel",
        step_args=estimator.fit(
            {"data": TrainingInput(processed_uri, content_type="application/x-parquet")}
        ),
    )
    model_data = step_train.properties.ModelArtifacts.S3ModelArtifacts

    # ---- step 3: evaluate on the test period --------------------------------------------
    evaluator = ScriptProcessor(
        image_uri=image,
        command=["python3"],
        role=role,
        instance_count=1,
        instance_type=processing_instance,
        base_job_name="graph-fraud-evaluate",
        sagemaker_session=session,
    )
    evaluation_report = PropertyFile(
        name="EvaluationReport", output_name="evaluation", path="evaluation.json"
    )
    step_evaluate = ProcessingStep(
        name="EvaluateModel",
        step_args=evaluator.run(
            code=str(REPO_ROOT / "scripts" / "evaluate.py"),
            inputs=[
                ProcessingInput(source=model_data, destination="/opt/ml/processing/model"),
                ProcessingInput(source=processed_uri, destination="/opt/ml/processing/input/data"),
            ],
            outputs=[
                ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")
            ],
        ),
        property_files=[evaluation_report],
    )
    evaluation_uri = Join(
        on="/",
        values=[
            step_evaluate.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
            "evaluation.json",
        ],
    )

    # ---- step 4: register if the gate passes ---------------------------------------------
    model = Model(
        image_uri=image,
        model_data=model_data,
        role=role,
        sagemaker_session=session,
        env={"MODEL_DIR": "/opt/ml/model"},
    )
    step_register = ModelStep(
        name="RegisterModel",
        step_args=model.register(
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=["ml.m5.large", "ml.m5.xlarge"],
            transform_instances=["ml.m5.xlarge"],
            model_package_group_name=model_package_group,
            approval_status=approval_status,
            model_metrics=ModelMetrics(
                model_statistics=MetricsSource(
                    s3_uri=evaluation_uri, content_type="application/json"
                )
            ),
            description="GraphSAGE/XGBoost illicit-transaction scorer trained on Elliptic",
        ),
    )
    step_fail = FailStep(
        name="PrAucBelowThreshold",
        error_message=Join(on=" ", values=["Test PR-AUC below", min_pr_auc]),
    )
    step_gate = ConditionStep(
        name="CheckPrAuc",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=step_evaluate.name,
                    property_file=evaluation_report,
                    json_path="binary_classification_metrics.pr_auc.value",
                ),
                right=min_pr_auc,
            )
        ],
        if_steps=[step_register],
        else_steps=[step_fail],
    )

    return Pipeline(
        name=name,
        parameters=[
            raw_data_uri,
            model_type,
            epochs,
            min_pr_auc,
            processing_instance,
            training_instance,
            approval_status,
        ],
        steps=[step_prepare, step_train, step_evaluate, step_gate],
        sagemaker_session=session,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="graph-fraud-pipeline")
    p.add_argument("--stack", default=DEFAULT_STACK)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--image-tag", default="latest")
    p.add_argument("--raw-data-s3-uri", default=None)
    p.add_argument("--model-type", default=None, choices=[None, "graphsage", "xgboost"])
    p.add_argument(
        "--definition-only", action="store_true", help="print the pipeline JSON and exit"
    )
    p.add_argument("--upsert", action="store_true")
    p.add_argument("--start", action="store_true")
    a = p.parse_args()

    out = stack_outputs(a.stack, a.region)
    session = PipelineSession(
        boto_session=sagemaker.Session(default_bucket=out["BucketName"]).boto_session,
        default_bucket=out["BucketName"],
    )
    pipeline = build_pipeline(
        name=a.name,
        role=out["SageMakerRoleArn"],
        bucket=out["BucketName"],
        image=image_uri(out, a.image_tag),
        model_package_group=out["ModelPackageGroupName"],
        session=session,
    )
    if a.definition_only:
        print(json.dumps(json.loads(pipeline.definition()), indent=2))
        return 0
    if a.upsert:
        pipeline.upsert(role_arn=out["SageMakerRoleArn"])
        print(f"upserted pipeline {a.name}")
    if a.start:
        params = {}
        if a.raw_data_s3_uri:
            params["RawDataS3Uri"] = a.raw_data_s3_uri
        if a.model_type:
            params["ModelType"] = a.model_type
        execution = pipeline.start(parameters=params)
        print(f"EXECUTION_ARN={execution.arn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
