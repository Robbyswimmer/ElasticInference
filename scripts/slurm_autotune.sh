#!/bin/bash
#SBATCH --job-name="autotune"
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rmose009@ucr.edu
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

echo "=== CS208 Autotune Experiment ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Project root (where this script lives is scripts/, go up one level)
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
echo "Working directory: $PROJECT_DIR"

# --- 1. Activate conda environment ---
echo "=== Activating conda env cs208-env ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cs208-env

# --- 2. Install dependencies ---
echo "=== Installing dependencies ==="
pip install -r requirements.txt

# --- 3. Install & start Redis ---
echo "=== Setting up Redis ==="
conda install -y redis
redis-server --daemonize yes
sleep 2

# Verify Redis
if redis-cli ping | grep -q PONG; then
    echo "Redis is running"
else
    echo "ERROR: Redis failed to start"
    exit 1
fi

# --- 4. Set config path ---
export CONFIG_PATH="$PROJECT_DIR/config.yaml"
echo "CONFIG_PATH=$CONFIG_PATH"

# --- 5. Create logs directory ---
mkdir -p logs results

# --- 6. Launch prefill worker ---
echo "=== Launching prefill worker ==="
python -m workers.prefill > logs/prefill.log 2>&1 &
PREFILL_PID=$!
echo "Prefill worker PID: $PREFILL_PID"

# --- 7. Launch decode worker ---
echo "=== Launching decode worker ==="
python -m workers.decode > logs/decode.log 2>&1 &
DECODE_PID=$!
echo "Decode worker PID: $DECODE_PID"

# --- 8. Wait for model loading ---
echo "=== Waiting 60s for model loading ==="
sleep 60

# Check workers are still running
if ! kill -0 $PREFILL_PID 2>/dev/null; then
    echo "ERROR: Prefill worker died. Check logs/prefill.log"
    cat logs/prefill.log
    exit 1
fi
if ! kill -0 $DECODE_PID 2>/dev/null; then
    echo "ERROR: Decode worker died. Check logs/decode.log"
    cat logs/decode.log
    exit 1
fi

# --- 9. Launch gateway ---
echo "=== Launching gateway ==="
python -m gateway.server > logs/gateway.log 2>&1 &
GATEWAY_PID=$!
echo "Gateway PID: $GATEWAY_PID"

# --- 10. Wait for gateway to connect ---
echo "=== Waiting 15s for gateway startup ==="
sleep 15

if ! kill -0 $GATEWAY_PID 2>/dev/null; then
    echo "ERROR: Gateway died. Check logs/gateway.log"
    cat logs/gateway.log
    exit 1
fi

# --- 11. Run autotune experiment ---
echo "=== Running autotune experiment ==="
python scripts/run_autotune_experiment.py
EXPERIMENT_EXIT=$?

# --- 12. Cleanup ---
echo "=== Cleaning up ==="
cleanup() {
    echo "Killing background processes..."
    kill $GATEWAY_PID 2>/dev/null || true
    kill $PREFILL_PID 2>/dev/null || true
    kill $DECODE_PID 2>/dev/null || true
    redis-cli shutdown 2>/dev/null || true
    echo "Cleanup complete"
}
trap cleanup EXIT
cleanup

# --- 13. Copy results ---
OUTPUT_DIR="$PROJECT_DIR/results"
echo "=== Results saved to $OUTPUT_DIR ==="
ls -la "$OUTPUT_DIR"/

echo ""
echo "=== Experiment finished (exit code: $EXPERIMENT_EXIT) ==="
echo "Date: $(date)"
exit $EXPERIMENT_EXIT
