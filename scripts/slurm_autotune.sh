#!/bin/bash
#SBATCH --job-name="autotune"
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rmose009@ucr.edu
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Project root
PROJECT_DIR="/data/SalmanAsif/RobbyMoseley/elastic/ElasticInference"
cd "$PROJECT_DIR"

echo "=== Autotune Experiment ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Date:      $(date)"
echo "Directory: $PROJECT_DIR"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Create directories upfront
mkdir -p logs results

# --- Cleanup trap (registered early so it always runs) ---
PREFILL_PID=""
DECODE_PID=""
GATEWAY_PID=""
cleanup() {
    echo "=== Cleaning up ==="
    [ -n "$GATEWAY_PID" ] && kill $GATEWAY_PID 2>/dev/null || true
    [ -n "$PREFILL_PID" ] && kill $PREFILL_PID 2>/dev/null || true
    [ -n "$DECODE_PID" ]  && kill $DECODE_PID 2>/dev/null || true
    redis-cli shutdown 2>/dev/null || true
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

# --- 4. Set config path ---
export CONFIG_PATH="$PROJECT_DIR/config.yaml"
echo "CONFIG_PATH=$CONFIG_PATH"

# --- 5. Launch prefill worker ---
echo "=== Launching prefill worker ==="
python -m workers.prefill > logs/prefill.log 2>&1 &
PREFILL_PID=$!
echo "Prefill worker PID: $PREFILL_PID"

# --- 6. Launch decode worker ---
echo "=== Launching decode worker ==="
python -m workers.decode > logs/decode.log 2>&1 &
DECODE_PID=$!
echo "Decode worker PID: $DECODE_PID"

# --- 7. Wait for model loading ---
echo "=== Waiting 60s for model loading ==="
sleep 60

if ! kill -0 $PREFILL_PID 2>/dev/null; then
    echo "ERROR: Prefill worker died. Log:"
    cat logs/prefill.log
    exit 1
fi
if ! kill -0 $DECODE_PID 2>/dev/null; then
    echo "ERROR: Decode worker died. Log:"
    cat logs/decode.log
    exit 1
fi

# --- 8. Launch gateway ---
echo "=== Launching gateway ==="
python -m gateway.server > logs/gateway.log 2>&1 &
GATEWAY_PID=$!
echo "Gateway PID: $GATEWAY_PID"

echo "=== Waiting 15s for gateway startup ==="
sleep 15

if ! kill -0 $GATEWAY_PID 2>/dev/null; then
    echo "ERROR: Gateway died. Log:"
    cat logs/gateway.log
    exit 1
fi

# --- 9. Run autotune experiment ---
echo "=== Running autotune experiment ==="
python scripts/run_autotune_experiment.py
EXPERIMENT_EXIT=$?

# --- 10. Results ---
echo "=== Results saved to $PROJECT_DIR/results ==="
ls -la results/

echo ""
echo "=== Experiment finished (exit code: $EXPERIMENT_EXIT) ==="
echo "Date: $(date)"
exit $EXPERIMENT_EXIT
