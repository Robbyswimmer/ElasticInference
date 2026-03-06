#!/usr/bin/env bash
set -euo pipefail

NS="${NS:-llm-inference}"
PROM_URL="${PROM_URL:-http://127.0.0.1:30090}"
PROM_JOB="${PROM_JOB:-llm-inference-static}"

# Workload (kept conservative for stability)
SPIKE_N="${SPIKE_N:-120}"
SPIKE_RPS="${SPIKE_RPS:-2.0}"
SPIKE_WORKERS="${SPIKE_WORKERS:-6}"
SPIKE_MAX_TOKENS="${SPIKE_MAX_TOKENS:-24}"
POST_IDLE_SECONDS="${POST_IDLE_SECONDS:-45}"

# Experimental knobs
METRICS_RPC_DELAY_MS="${METRICS_RPC_DELAY_MS:-120}"
SCALING_METRICS_INTERVAL="${SCALING_METRICS_INTERVAL:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="${PROJECT_DIR}/eval_results/piggyback_controlplane_$(date +%Y%m%d_%H%M%S)"
CSV_PATH="${OUT_DIR}/results.csv"
PLOT_PATH="${OUT_DIR}/plot.png"

mkdir -p "$OUT_DIR"
echo "mode,ok,err,duration_s,total_rpc_calls,avg_fetch_rpc_ms,avg_fetch_piggy_ms,avg_step_ms,step_overrun_pct,decode_p95_ms,e2e_p95_ms" > "$CSV_PATH"

prom_query() {
  local query="$1"
  python3 - "$PROM_URL" "$query" <<'PY'
import json, sys, urllib.parse, urllib.request
base = sys.argv[1].rstrip('/')
query = sys.argv[2]
url = base + '/api/v1/query?' + urllib.parse.urlencode({'query': query})
with urllib.request.urlopen(url, timeout=20) as resp:
    payload = json.loads(resp.read().decode('utf-8'))
res = payload.get('data', {}).get('result', [])
if not res:
    print('0')
else:
    try:
        print(res[0]['value'][1])
    except Exception:
        print('0')
PY
}

run_load() {
  local gateway_pod
  gateway_pod="$(kubectl -n "$NS" get pod -l app=gateway --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
  kubectl -v=0 -n "$NS" exec -i "$gateway_pod" -- python3 - <<PY 2>/dev/null
import time, uuid, grpc, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from common.proto import inference_pb2, inference_pb2_grpc

N = int(${SPIKE_N})
RPS = float(${SPIKE_RPS})
W = int(${SPIKE_WORKERS})
MAX_TOKENS = int(${SPIKE_MAX_TOKENS})
TIMEOUT = 60

prompts = [
    "Hello, how are you?",
    "Explain distributed systems in one paragraph.",
    "Introduce yourself briefly.",
    "What is autoscaling in Kubernetes?",
]

ch = grpc.insecure_channel("localhost:50051")
stub = inference_pb2_grpc.GatewayServiceStub(ch)

ok = 0
err = 0

def send(i):
    global ok, err
    req = inference_pb2.InferRequest(
        request_id=f"piggy-ctrl-{i}-{uuid.uuid4().hex[:8]}",
        prompt=random.choice(prompts),
        max_tokens=MAX_TOKENS,
    )
    try:
        stub.Infer(req, timeout=TIMEOUT)
        ok += 1
    except Exception:
        err += 1

start = time.time()
next_t = start
futs = []
with ThreadPoolExecutor(max_workers=W) as ex:
    for i in range(N):
        now = time.time()
        if next_t > now:
            time.sleep(next_t - now)
        futs.append(ex.submit(send, i))
        next_t += 1.0 / RPS
    for f in as_completed(futs):
        f.result()

print(f"RESULT ok={ok} err={err} duration_s={time.time()-start:.1f}", flush=True)
PY
}

set_common_env() {
  echo "=== Configure common experiment env ==="
  kubectl -n "$NS" set env deploy/prefill METRICS_RPC_DELAY_MS="${METRICS_RPC_DELAY_MS}" >/dev/null
  kubectl -n "$NS" set env deploy/decode METRICS_RPC_DELAY_MS="${METRICS_RPC_DELAY_MS}" >/dev/null
  kubectl -n "$NS" set env deploy/scaling-controller SCALING_METRICS_INTERVAL="${SCALING_METRICS_INTERVAL}" >/dev/null
  kubectl -n "$NS" rollout status deploy/prefill --timeout=900s >/dev/null
  kubectl -n "$NS" rollout status deploy/decode --timeout=900s >/dev/null
  kubectl -n "$NS" rollout status deploy/scaling-controller --timeout=900s >/dev/null
}

set_mode() {
  local use_piggy="$1"
  local fallback_rpc="$2"
  echo "=== Set mode USE_PIGGYBACK_METRICS=${use_piggy} fallback=${fallback_rpc} ==="
  kubectl -n "$NS" set env deploy/gateway USE_PIGGYBACK_METRICS="${use_piggy}" >/dev/null
  kubectl -n "$NS" set env deploy/scaling-controller \
    USE_PIGGYBACK_METRICS="${use_piggy}" \
    PIGGYBACK_FALLBACK_TO_RPC="${fallback_rpc}" >/dev/null
  kubectl -n "$NS" rollout status deploy/gateway --timeout=600s >/dev/null
  kubectl -n "$NS" rollout status deploy/scaling-controller --timeout=600s >/dev/null
}

run_mode_once() {
  local mode_name="$1"
  local use_piggy="$2"
  local fallback_rpc="$3"

  set_mode "$use_piggy" "$fallback_rpc"

  local start_epoch end_epoch window_s
  start_epoch="$(date +%s)"

  echo "=== Run workload (${mode_name}) ==="
  local load_out
  load_out="$(run_load)"
  echo "$load_out"

  local ok err duration_s
  ok="$(echo "$load_out" | sed -n 's/.*RESULT ok=\([0-9]\+\).*/\1/p' | tail -1)"
  err="$(echo "$load_out" | sed -n 's/.*RESULT ok=[0-9]\+ err=\([0-9]\+\).*/\1/p' | tail -1)"
  duration_s="$(echo "$load_out" | sed -n 's/.*duration_s=\([0-9.]*\).*/\1/p' | tail -1)"
  ok="${ok:-0}"
  err="${err:-0}"
  duration_s="${duration_s:-0}"

  echo "=== Post idle ${POST_IDLE_SECONDS}s (${mode_name}) ==="
  sleep "$POST_IDLE_SECONDS"

  end_epoch="$(date +%s)"
  window_s="$(( end_epoch - start_epoch + 10 ))"

  local total_rpc avg_fetch_rpc avg_fetch_piggy avg_step_ms overrun_pct decode_p95 e2e_p95
  total_rpc="$(prom_query "sum(increase(scaler_worker_metrics_rpc_total{job=\"${PROM_JOB}\",stage=~\"decode|prefill\"}[${window_s}s]))")"
  avg_fetch_rpc="$(prom_query "1000 * (sum(increase(scaler_metrics_fetch_duration_seconds_total{job=\"${PROM_JOB}\",source=\"rpc\",stage=~\"decode|prefill\"}[${window_s}s])) / clamp_min(sum(increase(scaler_metrics_fetch_samples_total{job=\"${PROM_JOB}\",source=\"rpc\",stage=~\"decode|prefill\"}[${window_s}s])),1))")"
  avg_fetch_piggy="$(prom_query "1000 * (sum(increase(scaler_metrics_fetch_duration_seconds_total{job=\"${PROM_JOB}\",source=\"piggyback\",stage=~\"decode|prefill\"}[${window_s}s])) / clamp_min(sum(increase(scaler_metrics_fetch_samples_total{job=\"${PROM_JOB}\",source=\"piggyback\",stage=~\"decode|prefill\"}[${window_s}s])),1))")"
  avg_step_ms="$(prom_query "1000 * (increase(scaler_control_step_duration_sum_seconds_total{job=\"${PROM_JOB}\"}[${window_s}s]) / clamp_min(increase(scaler_control_step_total{job=\"${PROM_JOB}\"}[${window_s}s]),1))")"
  overrun_pct="$(prom_query "100 * (increase(scaler_control_step_overrun_total{job=\"${PROM_JOB}\"}[${window_s}s]) / clamp_min(increase(scaler_control_step_total{job=\"${PROM_JOB}\"}[${window_s}s]),1))")"
  decode_p95="$(prom_query "histogram_quantile(0.95, sum(increase(gateway_decode_latency_ms_bucket{job=\"${PROM_JOB}\"}[${window_s}s])) by (le))")"
  e2e_p95="$(prom_query "histogram_quantile(0.95, sum(increase(gateway_e2e_latency_ms_bucket{job=\"${PROM_JOB}\"}[${window_s}s])) by (le))")"

  echo "${mode_name},${ok},${err},${duration_s},${total_rpc},${avg_fetch_rpc},${avg_fetch_piggy},${avg_step_ms},${overrun_pct},${decode_p95},${e2e_p95}" >> "$CSV_PATH"
}

echo "=== Check Prometheus endpoint ==="
curl -fsS "${PROM_URL}/-/ready" >/dev/null

set_common_env

echo "=== Start control-plane A/B ==="
run_mode_once polling false true
run_mode_once piggyback true false

python3 "$PROJECT_DIR/monitoring/plot_piggyback_controlplane.py" --csv "$CSV_PATH" --out "$PLOT_PATH" >/dev/null

echo "=== Done ==="
echo "CSV:  $CSV_PATH"
echo "PLOT: $PLOT_PATH"
cat "$CSV_PATH"
