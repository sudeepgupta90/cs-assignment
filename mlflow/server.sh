#!/bin/sh
set -euo pipefail
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri ${BACKEND_URI} \
  --default-artifact-root ${ARTIFACT_ROOT}