#!/bin/bash
set -euo pipefail

# Only run in remote (Claude Code on the web) environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

echo "Installing Python dependencies for machine_learning_sample..."

pip install --quiet \
  numpy \
  pandas \
  matplotlib \
  seaborn \
  scikit-learn \
  xgboost \
  gdown \
  torch \
  torchvision \
  jupyter \
  nbformat \
  nbconvert \
  ipykernel \
  flake8 \
  nbqa \
  nbmake

echo "All dependencies installed successfully."
