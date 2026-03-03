#!/bin/bash
#SBATCH --job-name="multimodel"
#SBATCH --time=06:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rmose009@ucr.edu
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Project root
PROJECT_DIR="/data/SalmanAsif/RobbyMoseley/elastic/ElasticInference"
cd "$PROJECT_DIR"

echo "=== Multi-Model Autotune Experiment ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Date:      $(date)"
echo "Directory: $PROJECT_DIR"
echo "GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Create directories upfront
mkdir -p logs results .tmp_configs

# --- Cleanup trap ---
cleanup() {
    echo "=== Cleaning up ==="
    # Kill any leftover python processes from our experiment
    pkill -f "workers.prefill" 2>/dev/null || true
    pkill -f "workers.decode" 2>/dev/null || true
    pkill -f "gateway.server" 2>/dev/null || true
    redis-cli shutdown 2>/dev/null || true
    # Clean up temp configs
    rm -rf .tmp_configs
    echo "Cleanup complete"
}
trap cleanup EXIT

# Now enable strict mode (after trap is set)
set -euo pipefail

# --- 1. Activate conda environment ---
echo "=== Activating conda env cs208-env ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cs208-env

# --- 2. Install dependencies ---
echo "=== Installing dependencies ==="
pip install -r requirements.txt

# --- 3. Start Redis ---
echo "=== Setting up Redis ==="
redis-server --daemonize yes
sleep 2

if redis-cli ping | grep -q PONG; then
    echo "Redis is running"
else
    echo "ERROR: Redis failed to start"
    exit 1
fi

# --- 4. Run multi-model experiment ---
# The orchestrator manages all worker/gateway lifecycle internally
echo "=== Running multi-model experiment ==="
python scripts/run_multimodel_experiment.py
EXPERIMENT_EXIT=$?

# --- 5. Generate figures (in case experiment didn't complete all) ---
echo "=== Generating figures ==="
python scripts/generate_multimodel_figures.py || echo "Figure generation had issues (non-fatal)"

# --- 6. Results ---
echo "=== Results ==="
echo "Results directory:"
find results/ -type f | head -50
echo ""
echo "=== Experiment finished (exit code: $EXPERIMENT_EXIT) ==="
echo "Date: $(date)"
exit $EXPERIMENT_EXIT
