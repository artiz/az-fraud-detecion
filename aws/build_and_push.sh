#!/usr/bin/env bash
# Build the container and push it to the ECR repository created by cloudformation/infra.yaml.
#   ./aws/build_and_push.sh <stack-name> [region]
set -euo pipefail
STACK="${1:?stack name}"
REGION="${2:-${AWS_REGION:-eu-central-1}}"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REPO_URI=$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='EcrRepositoryUri'].OutputValue" --output text)
TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD)}"

aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
docker build --platform linux/amd64 -t "$REPO_URI:$TAG" -t "$REPO_URI:latest" -f "$(dirname "$0")/docker/Dockerfile" "$(dirname "$0")/.."
docker push "$REPO_URI:$TAG"
docker push "$REPO_URI:latest"
echo "IMAGE_URI=$REPO_URI:$TAG"
