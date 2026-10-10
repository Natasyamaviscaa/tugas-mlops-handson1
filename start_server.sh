#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"
exec mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts \
  --host 127.0.0.1 \
  --port "${MLFLOW_PORT:-5000}"

