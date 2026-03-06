#!/usr/bin/env bash
set -euo pipefail

NS="${NS:-llm-inference}"
PROM_URL="${PROM_URL:-http://127.0.0.1:30090}"
SPIKE_N="${SPIKE_N:-240}"
SPIKE_RPS="${SPIKE_RPS:-4.0}"
SPIKE_WORKERS="${SPIKE_WORKERS:-12}"
SPIKE_MAX_TOKENS="${SPIKE_MAX_TOKENS:-32}"
POST_IDLE_SECONDS="${POST_IDLE_SECONDS:-60}"
PROM_JOB="${PROM_JOB:-llm-inference-static}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="${PROJECT_DIR}/eval_results/piggyback_ab_$(date +%Y%m%d_%H%M%S)"
CSV_PATH="${OUT_DIR}/ab_results.csv"
PLOT_PATH="${OUT_DIR}/ab_plot.png"

mkdir -p "$OUT_DIR"
echo "mode,ok,err,duration_s,rpc_decode,rpc_prefill,max_decode_replicas,avg_piggyback_age_s,decode_p95_ms,e2e_p95_ms,scaleup_reaction_s" > "$CSV_PATH"

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

prom_query_range() {
  local query="$1"
  local start="$2"
  local end="$3"
  local step="${4:-5}"
  python3 - "$PROM_URL" "$query" "$start" "$end" "$step" <<'PY'
import json, sys, urllib.parse, urllib.request
base = sys.argv[1].rstrip('/')
query = sys.argv[2]
start = sys.argv[3]
end = sys.argv[4]
step = sys.argv[5]
url = base + '/api/v1/query_range?' + urllib.parse.urlencode(
    {'query': query, 'start': start, 'end': end, 'step': step}
)
with urllib.request.urlopen(url, timeout=20) as resp:
    payload = json.loads(resp.read().decode('utf-8'))
res = payload.get('data', {}).get('result', [])
if not res:
    print("")
else:
    # Output lines: "<ts>,<val>"
    for point in res[0].get('values', []):
        print(f"{point[0]},{point[1]}")
PY
}

set_mode() {
  local mode_flag="$1"
  echo "=== Set mode USE_PIGGYBACK_METRICS=${mode_flag} ==="
  kubectl -n "$NS" set env deploy/gateway USE_PIGGYBACK_METRICS="$mode_flag" >/dev/null
  kubectl -n "$NS" set env deploy/scaling-controller USE_PIGGYBACK_METRICS="$mode_flag" >/dev/null
  kubectl -n "$NS" rollout status deploy/gateway --timeout=600s >/dev/null
  kubectl -n "$NS" rollout status deploy/scaling-controller --timeout=600s >/dev/null
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
    "Explain the theory of relativity in simple terms.",
    "Introduce yourself.",
    "Write a short paragraph about distributed systems.",
]

ch = grpc.insecure_channel("localhost:50051")
stub = inference_pb2_grpc.GatewayServiceStub(ch)

ok = 0
err = 0

def send(i):
    global ok, err
    req = inference_pb2.InferRequest(
        request_id=f"piggyback-ab-{i}-{uuid.uuid4().hex[:8]}",
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

run_mode_once() {
  local mode_name="$1"
  local mode_flag="$2"

  set_mode "$mode_flag"

  echo "=== Reset replicas to baseline ==="
  kubectl -n "$NS" scale deploy/prefill --replicas=1 >/dev/null
  kubectl -n "$NS" scale deploy/decode --replicas=1 >/dev/null
  kubectl -n "$NS" rollout status deploy/prefill --timeout=900s >/dev/null
  kubectl -n "$NS" rollout status deploy/decode --timeout=900s >/dev/null

  local start_epoch
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

  local end_epoch window_s
  end_epoch="$(date +%s)"
  window_s="$(( end_epoch - start_epoch + 15 ))"

  local rpc_decode rpc_prefill max_rep avg_age decode_p95 e2e_p95 reaction_s
  rpc_decode="$(prom_query "sum(increase(scaler_worker_metrics_rpc_total{job=\"${PROM_JOB}\",stage=\"decode\"}[${window_s}s]))")"
  rpc_prefill="$(prom_query "sum(increase(scaler_worker_metrics_rpc_total{job=\"${PROM_JOB}\",stage=\"prefill\"}[${window_s}s]))")"
  max_rep="$(prom_query "max_over_time(scaler_target_replicas{job=\"${PROM_JOB}\",stage=\"decode\"}[${window_s}s])")"
  avg_age="$(prom_query "avg_over_time(scaler_piggyback_age_seconds{job=\"${PROM_JOB}\",stage=\"decode\"}[${window_s}s])")"
  decode_p95="$(prom_query "histogram_quantile(0.95, sum(increase(gateway_decode_latency_ms_bucket{job=\"${PROM_JOB}\"}[${window_s}s])) by (le))")"
  e2e_p95="$(prom_query "histogram_quantile(0.95, sum(increase(gateway_e2e_latency_ms_bucket{job=\"${PROM_JOB}\"}[${window_s}s])) by (le))")"
  reaction_s="$(python3 - <<PY
import math
start = int("${start_epoch}")
end = int("${end_epoch}")
vals = """$(prom_query_range "scaler_target_replicas{job=\"${PROM_JOB}\",stage=\"decode\"}" "${start_epoch}" "${end_epoch}" 5)""".strip().splitlines()
first = None
for line in vals:
    if not line.strip():
        continue
    ts_s, val_s = line.split(",", 1)
    try:
        ts = float(ts_s)
        val = float(val_s)
    except Exception:
        continue
    if val > 1.0:
        first = ts
        break
if first is None:
    print(-1)
else:
    print(max(0.0, first - start))
PY
)"

  echo "${mode_name},${ok},${err},${duration_s},${rpc_decode},${rpc_prefill},${max_rep},${avg_age},${decode_p95},${e2e_p95},${reaction_s}" >> "$CSV_PATH"
}

echo "=== Check Prometheus endpoint ==="
curl -fsS "${PROM_URL}/-/ready" >/dev/null

echo "=== A/B start ==="
run_mode_once polling false
run_mode_once piggyback true

python3 "$PROJECT_DIR/monitoring/plot_piggyback_ab.py" --csv "$CSV_PATH" --out "$PLOT_PATH" >/dev/null

echo "=== A/B done ==="
echo "CSV:  $CSV_PATH"
echo "PLOT: $PLOT_PATH"
cat "$CSV_PATH"
