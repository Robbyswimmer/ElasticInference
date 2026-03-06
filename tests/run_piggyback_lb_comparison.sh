#!/usr/bin/env bash
set -euo pipefail

NS="${NS:-llm-inference}"
PROM_URL="${PROM_URL:-http://127.0.0.1:30090}"
PROM_JOB="${PROM_JOB:-llm-inference-static}"
OUT_DIR="${OUT_DIR:-eval_results/piggyback_lb_compare_$(date +%Y%m%d_%H%M%S)}"

PREFILL_REPLICAS="${PREFILL_REPLICAS:-2}"
DECODE_REPLICAS="${DECODE_REPLICAS:-3}"

N="${N:-180}"
RPS="${RPS:-3.0}"
WORKERS="${WORKERS:-10}"
MAX_TOKENS="${MAX_TOKENS:-32}"
TIMEOUT="${TIMEOUT:-90}"
QUEUE_STEP_S="${QUEUE_STEP_S:-2}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR_ABS="${PROJECT_DIR}/${OUT_DIR}"
mkdir -p "$OUT_DIR_ABS"

ORIG_CONTROLLER_REPLICAS="$(kubectl -n "$NS" get deploy scaling-controller -o jsonpath='{.spec.replicas}' 2>/dev/null || echo 1)"

cleanup() {
  kubectl -n "$NS" scale deploy/scaling-controller --replicas="${ORIG_CONTROLLER_REPLICAS}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

fetch_queue_series() {
  local expr="$1"
  local start_ts="$2"
  local end_ts="$3"
  local out_csv="$4"
  python3 - "$PROM_URL" "$expr" "$start_ts" "$end_ts" "$QUEUE_STEP_S" "$out_csv" <<'PY'
import csv
import json
import sys
import urllib.parse
import urllib.request

prom_url, expr, start_ts, end_ts, step_s, out_csv = sys.argv[1:]
url = prom_url.rstrip("/") + "/api/v1/query_range?" + urllib.parse.urlencode(
    {"query": expr, "start": start_ts, "end": end_ts, "step": step_s}
)
with urllib.request.urlopen(url, timeout=20) as resp:
    payload = json.loads(resp.read().decode("utf-8"))
series = payload.get("data", {}).get("result", [])
rows = []
if series:
    for ts, val in series[0].get("values", []):
        try:
            rows.append((float(ts), float(val)))
        except Exception:
            continue

with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["ts", "value"])
    w.writerows(rows)
PY
}

set_mode() {
  local flag="$1"
  echo "=== Set mode USE_PIGGYBACK_METRICS=${flag} ==="
  kubectl -n "$NS" set env deploy/gateway USE_PIGGYBACK_METRICS="$flag" >/dev/null
  kubectl -n "$NS" set env deploy/scaling-controller USE_PIGGYBACK_METRICS="$flag" >/dev/null || true
  kubectl -n "$NS" rollout status deploy/gateway --timeout=600s >/dev/null
}

prepare_cluster() {
  echo "=== Apply headless services ==="
  kubectl apply -f "$PROJECT_DIR/k8s/prefill-headless-service.yaml" >/dev/null
  kubectl apply -f "$PROJECT_DIR/k8s/decode-headless-service.yaml" >/dev/null

  echo "=== Freeze controller and pin replicas ==="
  kubectl -n "$NS" scale deploy/scaling-controller --replicas=0 >/dev/null || true
  kubectl -n "$NS" scale deploy/prefill --replicas="$PREFILL_REPLICAS" >/dev/null
  kubectl -n "$NS" scale deploy/decode --replicas="$DECODE_REPLICAS" >/dev/null
  kubectl -n "$NS" rollout status deploy/prefill --timeout=900s >/dev/null
  kubectl -n "$NS" rollout status deploy/decode --timeout=900s >/dev/null
}

run_mode() {
  local mode="$1"
  local flag="$2"

  set_mode "$flag"

  local pod
  pod="$(kubectl -n "$NS" get pod -l app=gateway --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
  local raw_file="${OUT_DIR_ABS}/${mode}_raw.log"
  local req_csv="${OUT_DIR_ABS}/${mode}_requests.csv"
  local q_prefill_csv="${OUT_DIR_ABS}/${mode}_queue_prefill.csv"
  local q_decode_csv="${OUT_DIR_ABS}/${mode}_queue_decode.csv"

  echo "=== Run workload mode=${mode} ==="
  kubectl -v=0 -n "$NS" exec -i "$pod" -- python3 - <<PY > "$raw_file" 2>/dev/null
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import grpc
from transformers import AutoTokenizer

from common import load_config
from common.proto import inference_pb2, inference_pb2_grpc

N = int(${N})
RPS = float(${RPS})
WORKERS = int(${WORKERS})
MAX_TOKENS = int(${MAX_TOKENS})
TIMEOUT = int(${TIMEOUT})

cfg = load_config()
tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
stub = inference_pb2_grpc.GatewayServiceStub(grpc.insecure_channel("localhost:50051"))

prompts = [
    "Explain autoscaling and load balancing in cloud systems.",
    "Write a concise summary of transformer decoding.",
    "How does Kubernetes service routing work?",
    "Give an example of queueing delay in online inference.",
    "Describe prefill and decode separation in one paragraph.",
]

results = []

def send(i):
    prompt = random.choice(prompts)
    in_tokens = len(tokenizer.encode(prompt))
    req = inference_pb2.InferRequest(
        request_id=f"lb-cmp-{i}-{uuid.uuid4().hex[:8]}",
        prompt=prompt,
        max_tokens=MAX_TOKENS,
    )
    t0 = time.time()
    try:
        resp = stub.Infer(req, timeout=TIMEOUT)
        t1 = time.time()
        out_tokens = max(1, int(resp.tokens_generated))
        decode_ms = float(resp.latency.decode_ms)
        tpot_ms = decode_ms / max(out_tokens, 1)
        return ("ok", i, t1, in_tokens, out_tokens, decode_ms, float(resp.latency.total_ms), tpot_ms)
    except Exception:
        t1 = time.time()
        return ("err", i, t1, in_tokens, 0, 0.0, 0.0, 0.0)

start = time.time()
next_t = start
futs = []
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    for i in range(N):
        now = time.time()
        if next_t > now:
            time.sleep(next_t - now)
        futs.append(ex.submit(send, i))
        next_t += 1.0 / RPS
    for f in as_completed(futs):
        results.append(f.result())
end = time.time()

ok = sum(1 for r in results if r[0] == "ok")
err = sum(1 for r in results if r[0] != "ok")
print(f"RESULT,start={start:.3f},end={end:.3f},duration_s={end-start:.3f},ok={ok},err={err}")
for r in sorted(results, key=lambda x: x[1]):
    print(
        "REQ,status=%s,idx=%d,completion_ts=%.3f,input_tokens=%d,output_tokens=%d,decode_ms=%.3f,total_ms=%.3f,tpot_ms=%.3f"
        % r
    )
PY

  python3 - "$raw_file" "$req_csv" <<'PY'
import csv
import re
import sys

raw_path, out_csv = sys.argv[1], sys.argv[2]
pat = re.compile(
    r"^REQ,status=(?P<status>[^,]+),idx=(?P<idx>\d+),completion_ts=(?P<completion_ts>[0-9.]+),"
    r"input_tokens=(?P<input_tokens>\d+),output_tokens=(?P<output_tokens>\d+),"
    r"decode_ms=(?P<decode_ms>[0-9.]+),total_ms=(?P<total_ms>[0-9.]+),tpot_ms=(?P<tpot_ms>[0-9.]+)$"
)
rows = []
with open(raw_path) as f:
    for line in f:
        line = line.strip()
        m = pat.match(line)
        if not m:
            continue
        d = m.groupdict()
        rows.append(d)
rows.sort(key=lambda x: int(x["idx"]))
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            "idx",
            "status",
            "completion_ts",
            "input_tokens",
            "output_tokens",
            "decode_ms",
            "total_ms",
            "tpot_ms",
        ],
    )
    w.writeheader()
    w.writerows(rows)
PY

  local start_ts end_ts
  start_ts="$(awk -F',' '/^RESULT/{for(i=1;i<=NF;i++){if($i~"^start="){sub("start=","",$i);print $i;exit}}}' "$raw_file")"
  end_ts="$(awk -F',' '/^RESULT/{for(i=1;i<=NF;i++){if($i~"^end="){sub("end=","",$i);print $i;exit}}}' "$raw_file")"

  if [[ -z "$start_ts" || -z "$end_ts" ]]; then
    echo "ERROR: failed to parse RESULT line for mode=$mode"
    exit 1
  fi

  fetch_queue_series "sum(gateway_prefill_queue_length{job=\"${PROM_JOB}\"})" "$start_ts" "$end_ts" "$q_prefill_csv"
  fetch_queue_series "sum(gateway_decode_queue_length{job=\"${PROM_JOB}\"})" "$start_ts" "$end_ts" "$q_decode_csv"

  echo "mode=${mode} raw=${raw_file} req=${req_csv}"
}

echo "=== Check Prometheus endpoint ==="
curl -fsS "${PROM_URL}/-/ready" >/dev/null

prepare_cluster
run_mode baseline 0
run_mode piggyback_lb 1

python3 "$PROJECT_DIR/monitoring/plot_piggyback_lb_comparison.py" --out-dir "$OUT_DIR_ABS" > "${OUT_DIR_ABS}/summary_print.json"

echo "=== Done ==="
echo "out_dir=${OUT_DIR_ABS}"
ls -1 "$OUT_DIR_ABS"
