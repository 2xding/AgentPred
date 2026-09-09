# Local stand-in for Kaggle's sandbox image.
#
# The real harness uses gcr.io/kaggle-images/python, which is amd64-only and
# tens of GB — it neither fits in the free space here nor runs natively on
# arm64. This image carries the libraries the agent actually reaches for, so
# agent behaviour can be tested locally. It is NOT byte-identical to the
# competition sandbox; anything version-sensitive should still be treated as
# unverified until it runs on Kaggle.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        pandas numpy scipy scikit-learn lightgbm xgboost catboost

WORKDIR /workspace
