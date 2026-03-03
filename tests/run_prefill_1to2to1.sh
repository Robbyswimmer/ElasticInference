#!/usr/bin/env bash
set -euo pipefail

# Repro script for prefill scaling trajectory: 1 -> 2 -> 1
#
# Usage:
#   bash tests/run_prefill_1to2to1.sh
#
# Optional env vars:
#   NS=llm-inference
#   BASE_REPLICAS=1
#   TARGET_MAX=2
#   PREFILL_N=220
#   PREFILL_RPS=2.8
#   PREFILL_WORKERS=8
#   PREFILL_TOKEN_LEN=192
#   PREFILL_TIMEOUT=90
#   IDLE_SECONDS=240
#   SAMPLE_INTERVAL=8

NS="${NS:-llm-inference}"
BASE_REPLICAS="${BASE_REPLICAS:-1}"
TARGET_MAX="${TARGET_MAX:-2}"

PREFILL_N="${PREFILL_N:-220}"
PREFILL_RPS="${PREFILL_RPS:-2.8}"
PREFILL_WORKERS="${PREFILL_WORKERS:-8}"
PREFILL_TOKEN_LEN="${PREFILL_TOKEN_LEN:-192}"
PREFILL_TIMEOUT="${PREFILL_TIMEOUT:-90}"

IDLE_SECONDS="${IDLE_SECONDS:-240}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-8}"

TOTAL_SECONDS=$((IDLE_SECONDS + 900))
SAMPLES=$((TOTAL_SECONDS / SAMPLE_INTERVAL))

trace_file="/tmp/prefill_1to2to1_$(date +%Y%m%d_%H%M%S).log"

echo "=== Prefill 1->2->1 Repro ==="
echo "namespace:      $NS"
echo "trace_file:     $trace_file"
echo "pulse:          N=$PREFILL_N RPS=$PREFILL_RPS workers=$PREFILL_WORKERS token_len=$PREFILL_TOKEN_LEN"
echo "idle_seconds:   $IDLE_SECONDS"
echo "sample_interval:$SAMPLE_INTERVAL"
echo ""

cleanup() {
  if [[ -n "${MON_PID:-}" ]]; then
    kill "$MON_PID" >/dev/null 2>&1 || true
    wait "$MON_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "=== Reset prefill baseline and restart controller ==="
kubectl -n "$NS" scale deploy/prefill --replicas="$BASE_REPLICAS" >/dev/null
kubectl -n "$NS" rollout status deploy/prefill --timeout=600s >/dev/null
kubectl -n "$NS" rollout restart deploy/scaling-controller >/dev/null
kubectl -n "$NS" rollout status deploy/scaling-controller --timeout=300s >/dev/null

GATEWAY_POD="$(kubectl -n "$NS" get pod -l app=gateway --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
echo "gateway_pod:    $GATEWAY_POD"
echo ""

echo "=== Start prefill replica monitor ==="
(
  for _ in $(seq 1 "$SAMPLES"); do
    ts="$(date +%H:%M:%S)"
    spec="$(kubectl -n "$NS" get deploy prefill -o jsonpath='{.spec.replicas}')"
    ready="$(kubectl -n "$NS" get deploy prefill -o jsonpath='{.status.readyReplicas}')"
    avail="$(kubectl -n "$NS" get deploy prefill -o jsonpath='{.status.availableReplicas}')"
    echo "$ts prefill=$spec/$ready/$avail"
    sleep "$SAMPLE_INTERVAL"
  done
) > "$trace_file" &
MON_PID=$!

echo "=== Prefill pulse ==="
kubectl -n "$NS" exec -i "$GATEWAY_POD" -- python3 - <<PY
import time, uuid, grpc
from concurrent.futures import ThreadPoolExecutor, as_completed
from common.proto import inference_pb2, inference_pb2_grpc

N = int(${PREFILL_N})
RPS = float(${PREFILL_RPS})
W = int(${PREFILL_WORKERS})
TOKEN_LEN = int(${PREFILL_TOKEN_LEN})
TIMEOUT = int(${PREFILL_TIMEOUT})

ch = grpc.insecure_channel("prefill:50052")
stub = inference_pb2_grpc.PrefillServiceStub(ch)
base_tokens = [((i % 1000) + 10) for i in range(TOKEN_LEN)]
ok = 0
err = 0

def send(i):
    global ok, err
    req = inference_pb2.PrefillRequest(
        request_id=f"prefill-repro-{i}-{uuid.uuid4().hex[:8]}",
        token_ids=base_tokens,
        model_name="prefill-repro",
    )
    try:
        _ = stub.RunPrefill(req, timeout=TIMEOUT)
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

print(f"prefill_pulse done total={N} ok={ok} err={err} duration_s={time.time()-start:.1f}", flush=True)
PY

echo "=== Idle phase (${IDLE_SECONDS}s) ==="
sleep "$IDLE_SECONDS"

cleanup
trap - EXIT

echo "=== Prefill trace ==="
cat "$trace_file"

echo "=== Controller prefill events ==="
kubectl -n "$NS" logs deploy/scaling-controller --since=45m | egrep 'Scaling prefill|Scaling weights|Oscillation' || true

echo "=== Prefill transitions ==="
awk -F'[=/]' '
  /prefill=/ {
    ts=$1
    spec=$2+0
    if (first || spec != prev) {
      printf("%s %d\n", ts, spec)
      prev=spec
      first=0
    }
  }
' "$trace_file"

analysis="$(awk -F'[=/]' '
  /prefill=/ {
    spec=$2+0
    arr[n]=spec
    n++
    if (spec > max) max=spec
  }
  END {
    if (n == 0) { print "NO_DATA"; exit 0 }
    hit=-1
    min_after=999999
    for (i=0; i<n; i++) {
      if (hit < 0 && arr[i] >= '"$TARGET_MAX"') hit=i
      if (hit >= 0 && arr[i] < min_after) min_after=arr[i]
    }
    if (hit < 0) {
      printf("NO_SCALE_TO_TARGET max=%d target=%d\n", max, '"$TARGET_MAX"')
    } else {
      printf("OK max=%d min_after_target=%d target=%d\n", max, min_after, '"$TARGET_MAX"')
    }
  }
' "$trace_file")"
echo "$analysis"

if echo "$analysis" | grep -q '^OK '; then
  min_after="$(echo "$analysis" | sed -n 's/.*min_after_target=\([0-9]\+\).*/\1/p')"
  if [[ -n "$min_after" && "$min_after" -le 1 ]]; then
    echo "RESULT: PASS (observed prefill scale up to >=${TARGET_MAX} and down to <=1)"
    exit 0
  fi
  echo "RESULT: FAIL (prefill scaled up, but did not fall to 1; min_after_target=$min_after)"
  exit 2
fi

echo "RESULT: FAIL ($analysis)"
exit 2
