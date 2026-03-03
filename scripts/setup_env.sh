#!/bin/bash
# Setup conda environment for CS208 Final Project
#
# Usage:
#   bash scripts/setup_env.sh
#
# On SLURM cluster, run this once before submitting jobs.

set -euo pipefail

ENV_NAME="cs208-env"
PYTHON_VERSION="3.10"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== CS208 Environment Setup ==="
echo "Project: $PROJECT_DIR"

# Initialize conda
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Please install Miniconda/Anaconda first."
    exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create env if it doesn't exist
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '$ENV_NAME' already exists, updating..."
    conda activate "$ENV_NAME"
else
    echo "Creating environment '$ENV_NAME' (Python $PYTHON_VERSION)..."
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
    conda activate "$ENV_NAME"
fi

echo "Python: $(python --version) @ $(which python)"

# Install Redis via conda
echo "=== Installing Redis ==="
conda install -y redis

# Install PyTorch with CUDA via pip (avoids conda/pip MKL conflicts)
echo "=== Installing PyTorch with CUDA ==="
pip install 'torch>=2.6' torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install pip dependencies
echo "=== Installing pip dependencies ==="
pip install -r "$PROJECT_DIR/requirements.txt"

# Install matplotlib for plots
pip install matplotlib

# Download models
echo "=== Downloading models ==="
python "$PROJECT_DIR/scripts/download_models.py"

# Verify installation
echo ""
echo "=== Verification ==="
python -c "
import torch
print(f'PyTorch:      {torch.__version__}')
print(f'CUDA avail:   {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:          {torch.cuda.get_device_name(0)}')
import transformers
print(f'Transformers: {transformers.__version__}')
import grpc
print(f'gRPC:         {grpc.__version__}')
import redis
print(f'Redis-py:     {redis.__version__}')
import numpy
print(f'NumPy:        {numpy.__version__}')
import matplotlib
print(f'Matplotlib:   {matplotlib.__version__}')
print('All dependencies OK')
"

echo ""
echo "=== Setup complete ==="
echo "Activate with: conda activate $ENV_NAME"
