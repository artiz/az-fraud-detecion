# One image for SageMaker Processing, Training and Hosting.
#
# Why a custom image: the AWS PyTorch Deep Learning Containers install extra packages
# with pip, but DGL no longer has a pip source (data.dgl.ai is gone), so DGL 2.3.0 is
# taken from conda-forge via micromamba, together with the PyTorch 2.3.1 it was built
# against. Build on x86_64 (ml.m5/ml.c5 instances); conda-forge has no CUDA build for
# aarch64 PyTorch 2.3.
#
#   docker build -t graph-fraud -f aws/docker/Dockerfile .
#   docker run --rm -e SM_CHANNEL_DATA=/data -v $PWD/data/processed:/data graph-fraud train --model xgboost
#   docker run --rm -p 8080:8080 -v $PWD/artifacts/graphsage:/opt/ml/model graph-fraud serve
FROM mambaorg/micromamba:2.0.5-ubuntu22.04

USER root
WORKDIR /opt/ml/code

COPY environment.yml pyproject.toml ./
COPY src ./src
RUN sed -i '/^  - pip:/,$d' environment.yml \
 && micromamba install -y -n base -f environment.yml \
 && micromamba run -n base pip install --no-cache-dir --no-deps -e . \
 && micromamba run -n base pip install --no-cache-dir "sagemaker-training>=4.7" "fastapi>=0.110" "uvicorn>=0.29" \
 && micromamba clean -a -y

COPY scripts ./scripts
COPY configs ./configs
# libtorch from conda-forge carries an executable-stack flag that newer glibc rejects; clear it.
RUN micromamba run -n base bash scripts/fix_execstack.sh /opt/conda

# sagemaker-training provides the `train` entrypoint and runs $SAGEMAKER_PROGRAM with the
# job's hyperparameters as CLI flags; `serve` is our FastAPI server.
ENV PATH=/opt/conda/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    DGLBACKEND=pytorch \
    SAGEMAKER_PROGRAM=scripts/train.py \
    SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code \
    MODEL_DIR=/opt/ml/model
RUN printf '#!/bin/sh\nexec python /opt/ml/code/scripts/serve.py "$@"\n' > /usr/local/bin/serve && chmod +x /usr/local/bin/serve

EXPOSE 8080
