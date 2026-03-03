#!/usr/bin/env bash
set -euo pipefail

# End-to-end observability demo for decode workload:
# 1) run decode scaling workload
# 2) query Prometheus
# 3) generate offline analysis artifacts
#
# Usage:
#   bash tests/run_decode_prometheus_observability.sh
#
# Optional env vars:
#   NS=llm-inference
#   PROM_LOCAL_PORT=19090
#   LOOKBACK_MINUTES=35
#   ANALYSIS_OUT_DIR=eval_results/prometheus_decode

NS="${NS:-llm-inference}"
PROM_LOCAL_PORT="${PROM_LOCAL_PORT:-19090}"
LOOKBACK_MINUTES="${LOOKBACK_MINUTES:-35}"
ANALYSIS_OUT_DIR="${ANALYSIS_OUT_DIR:-eval_results/prometheus_decode}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PF_PID=""
cleanup() {
  if [[ -n "$PF_PID" ]]; then
    kill "$PF_PID" >/dev/null 2>&1 || true
    wait "$PF_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "=== Ensure Prometheus and controller are ready ==="
kubectl -n "$NS" rollout status deploy/prometheus --timeout=300s
kubectl -n "$NS" rollout status deploy/scaling-controller --timeout=300s

echo "=== Start port-forward for Prometheus (localhost:${PROM_LOCAL_PORT}) ==="
kubectl -n "$NS" port-forward svc/prometheus "${PROM_LOCAL_PORT}:9090" >/tmp/prom_port_forward.log 2>&1 &
PF_PID=$!
sleep 3

if ! curl -fsS "http://127.0.0.1:${PROM_LOCAL_PORT}/-/ready" >/dev/null; then
  echo "ERROR: Prometheus is not reachable at localhost:${PROM_LOCAL_PORT}"
  exit 1
fi

echo "=== Run decode workload (1->5->1) ==="
set +e
PREFILL_PULSE_ENABLE=0 \
SPIKE_N="${SPIKE_N:-540}" \
SPIKE_RPS="${SPIKE_RPS:-6.0}" \
SPIKE_WORKERS="${SPIKE_WORKERS:-20}" \
SPIKE_MAX_TOKENS="${SPIKE_MAX_TOKENS:-48}" \
IDLE_SECONDS="${IDLE_SECONDS:-420}" \
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-10}" \
bash "$PROJECT_DIR/tests/run_decode_1to5to1.sh"
WORKLOAD_RC=$?
set -e

if [[ "$WORKLOAD_RC" -ne 0 ]]; then
  echo "WARN: decode workload script exited with code $WORKLOAD_RC (continuing observability analysis)."
fi

echo "=== Run Prometheus offline analysis ==="
python3 "$PROJECT_DIR/monitoring/analyze_decode_from_prometheus.py" \
  --prom-url "http://127.0.0.1:${PROM_LOCAL_PORT}" \
  --lookback-minutes "$LOOKBACK_MINUTES" \
  --step-seconds 10 \
  --out-dir "$PROJECT_DIR/${ANALYSIS_OUT_DIR}"

echo "=== Check active alerts ==="
python3 - <<PY
import json, urllib.request
url = "http://127.0.0.1:${PROM_LOCAL_PORT}/api/v1/alerts"
with urllib.request.urlopen(url, timeout=20) as resp:
    payload = json.loads(resp.read().decode("utf-8"))
alerts = payload.get("data", {}).get("alerts", [])
print(f"alerts_total={len(alerts)}")
for a in alerts:
    state = a.get("state")
    name = a.get("labels", {}).get("alertname")
    sev = a.get("labels", {}).get("severity", "")
    if state == "firing":
        print(f"firing name={name} severity={sev}")
PY

echo "=== Done ==="
echo "Artifacts: ${PROJECT_DIR}/${ANALYSIS_OUT_DIR}"
echo "decode_workload_exit_code=${WORKLOAD_RC}"
