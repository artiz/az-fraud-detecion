"""Shared helpers for the SageMaker scripts: read stack outputs, resolve image URI."""

from __future__ import annotations

import os

import boto3

DEFAULT_REGION = os.environ.get("AWS_REGION", "eu-central-1")
DEFAULT_STACK = os.environ.get("GF_STACK", "graph-fraud-infra")


def stack_outputs(stack_name: str = DEFAULT_STACK, region: str = DEFAULT_REGION) -> dict[str, str]:
    cfn = boto3.client("cloudformation", region_name=region)
    stacks = cfn.describe_stacks(StackName=stack_name)["Stacks"]
    return {o["OutputKey"]: o["OutputValue"] for o in stacks[0].get("Outputs", [])}


def image_uri(outputs: dict[str, str], tag: str = "latest") -> str:
    return f"{outputs['EcrRepositoryUri']}:{tag}"
