#!/usr/bin/env bash
set -euo pipefail

# Repro script for decode scaling trajectory: 1 -> 5 -> 1
#
# Usage:
#   bash tests/run_decode_1to5to1.sh
#
# Optional env vars:
#   NS=llm-inference
#   TARGET_MAX=5
#   BASE_REPLICAS=1
#   PIN_CONTROLLER_IMAGE=docker.io/<user>/llm-inference:<tag>
#   PIN_WORKER_IMAGE=docker.io/<user>/llm-inference:<tag>
#   EXPECT_CONFIG_SHA256=<sha256_of_config_yaml>
#   PREFILL_PULSE_ENABLE=1
#   PREFILL_N=240
#   PREFILL_RPS=10.0
#   PREFILL_WORKERS=24
#   PREFILL_TOKEN_LEN=384
#   PREFILL_TIMEOUT=60
#   PREFILL_POST_IDLE=180
#   SPIKE_N=900
#   SPIKE_RPS=6.0
#   SPIKE_WORKERS=18
#   SPIKE_MAX_TOKENS=32
#   SPIKE_TIMEOUT=60
#   IDLE_SECONDS=600
#   SAMPLE_INTERVAL=10

# PIN_WORKER_IMAGE=docker.io/lxzhong/llm-inference:worker-activefix-20260228013835 
# PIN_CONTROLLER_IMAGE=docker.io/lxzhong/llm-inference:controller-demo-1to5to1-20260228015747 


NS="${NS:-llm-inference}"
TARGET_MAX="${TARGET_MAX:-5}"
BASE_REPLICAS="${BASE_REPLICAS:-1}"
PIN_CONTROLLER_IMAGE="${PIN_CONTROLLER_IMAGE:-}"
PIN_WORKER_IMAGE="${PIN_WORKER_IMAGE:-}"
EXPECT_CONFIG_SHA256="${EXPECT_CONFIG_SHA256:-}"

PREFILL_PULSE_ENABLE="${PREFILL_PULSE_ENABLE:-1}"
PREFILL_N="${PREFILL_N:-240}"
PREFILL_RPS="${PREFILL_RPS:-10.0}"
PREFILL_WORKERS="${PREFILL_WORKERS:-24}"
PREFILL_TOKEN_LEN="${PREFILL_TOKEN_LEN:-384}"
PREFILL_TIMEOUT="${PREFILL_TIMEOUT:-60}"
PREFILL_POST_IDLE="${PREFILL_POST_IDLE:-180}"

SPIKE_N="${SPIKE_N:-900}"
SPIKE_RPS="${SPIKE_RPS:-6.0}"
SPIKE_WORKERS="${SPIKE_WORKERS:-18}"
SPIKE_MAX_TOKENS="${SPIKE_MAX_TOKENS:-32}"
SPIKE_TIMEOUT="${SPIKE_TIMEOUT:-60}"

IDLE_SECONDS="${IDLE_SECONDS:-600}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-10}"

TOTAL_SECONDS=$((IDLE_SECONDS + 1200))
SAMPLES=$((TOTAL_SECONDS / SAMPLE_INTERVAL))

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_PATH="$PROJECT_DIR/config.yaml"

start_epoch="$(date +%s)"
trace_file="/tmp/scaling_1to5to1_$(date +%Y%m%d_%H%M%S).log"
manifest_file="${trace_file%.log}.manifest.txt"

echo "=== Scaling 1->5->1 Repro ==="
echo "namespace:      $NS"
echo "trace_file:     $trace_file"
echo "manifest_file:  $manifest_file"
echo "prefill_pulse:  enable=$PREFILL_PULSE_ENABLE N=$PREFILL_N RPS=$PREFILL_RPS workers=$PREFILL_WORKERS token_len=$PREFILL_TOKEN_LEN"
echo "spike:          N=$SPIKE_N RPS=$SPIKE_RPS workers=$SPIKE_WORKERS max_tokens=$SPIKE_MAX_TOKENS"
echo "idle_seconds:   $IDLE_SECONDS"
echo "sample_interval:$SAMPLE_INTERVAL"
echo ""

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: config not found at $CONFIG_PATH"
  exit 1
fi
CONFIG_SHA256="$(sha256sum "$CONFIG_PATH" | awk '{print $1}')"
if [[ -n "$EXPECT_CONFIG_SHA256" && "$CONFIG_SHA256" != "$EXPECT_CONFIG_SHA256" ]]; then
  echo "ERROR: config sha256 mismatch"
  echo "expected: $EXPECT_CONFIG_SHA256"
  echo "actual:   $CONFIG_SHA256"
  exit 1
fi

cleanup() {
  if [[ -n "${MON_PID:-}" ]]; then
    kill "$MON_PID" >/dev/null 2>&1 || true
    wait "$MON_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "=== Optional image pinning ==="
if [[ -n "$PIN_WORKER_IMAGE" ]]; then
  echo "pin worker image:      $PIN_WORKER_IMAGE"
  kubectl -n "$NS" set image deployment/prefill prefill="$PIN_WORKER_IMAGE" >/dev/null
  kubectl -n "$NS" set image deployment/decode decode="$PIN_WORKER_IMAGE" >/dev/null
fi
if [[ -n "$PIN_CONTROLLER_IMAGE" ]]; then
  echo "pin controller image:  $PIN_CONTROLLER_IMAGE"
  kubectl -n "$NS" set image deployment/scaling-controller controller="$PIN_CONTROLLER_IMAGE" >/dev/null
fi
echo ""

echo "=== Reset baseline and restart controller ==="
kubectl -n "$NS" scale deploy/prefill --replicas="$BASE_REPLICAS" >/dev/null
kubectl -n "$NS" scale deploy/decode --replicas="$BASE_REPLICAS" >/dev/null
kubectl -n "$NS" rollout status deploy/prefill --timeout=600s >/dev/null
kubectl -n "$NS" rollout status deploy/decode --timeout=600s >/dev/null
kubectl -n "$NS" rollout restart deploy/scaling-controller >/dev/null
kubectl -n "$NS" rollout status deploy/scaling-controller --timeout=300s >/dev/null

GATEWAY_POD="$(kubectl -n "$NS" get pod -l app=gateway --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
PREFILL_IMAGE="$(kubectl -n "$NS" get deploy prefill -o jsonpath='{.spec.template.spec.containers[0].image}')"
DECODE_IMAGE="$(kubectl -n "$NS" get deploy decode -o jsonpath='{.spec.template.spec.containers[0].image}')"
CONTROLLER_IMAGE="$(kubectl -n "$NS" get deploy scaling-controller -o jsonpath='{.spec.template.spec.containers[0].image}')"
GIT_HEAD="$(cd "$PROJECT_DIR" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"

cat > "$manifest_file" <<EOF
run_started_epoch=${start_epoch}
namespace=${NS}
project_dir=${PROJECT_DIR}
git_head=${GIT_HEAD}
config_path=${CONFIG_PATH}
config_sha256=${CONFIG_SHA256}
pin_worker_image=${PIN_WORKER_IMAGE}
pin_controller_image=${PIN_CONTROLLER_IMAGE}
prefill_image=${PREFILL_IMAGE}
decode_image=${DECODE_IMAGE}
controller_image=${CONTROLLER_IMAGE}
prefill_pulse_enable=${PREFILL_PULSE_ENABLE}
prefill_n=${PREFILL_N}
prefill_rps=${PREFILL_RPS}
prefill_workers=${PREFILL_WORKERS}
prefill_token_len=${PREFILL_TOKEN_LEN}
prefill_timeout=${PREFILL_TIMEOUT}
prefill_post_idle=${PREFILL_POST_IDLE}
target_max=${TARGET_MAX}
base_replicas=${BASE_REPLICAS}
spike_n=${SPIKE_N}
spike_rps=${SPIKE_RPS}
spike_workers=${SPIKE_WORKERS}
spike_max_tokens=${SPIKE_MAX_TOKENS}
spike_timeout=${SPIKE_TIMEOUT}
idle_seconds=${IDLE_SECONDS}
sample_interval=${SAMPLE_INTERVAL}
trace_file=${trace_file}
EOF

echo "gateway_pod:    $GATEWAY_POD"
echo "prefill_image:  $PREFILL_IMAGE"
echo "decode_image:   $DECODE_IMAGE"
echo "controller_img: $CONTROLLER_IMAGE"
echo "config_sha256:  $CONFIG_SHA256"
echo ""

echo "=== Start replica monitor ==="
(
  for _ in $(seq 1 "$SAMPLES"); do
    ts="$(date +%H:%M:%S)"
    p_spec="$(kubectl -n "$NS" get deploy prefill -o jsonpath='{.spec.replicas}')"
    p_ready="$(kubectl -n "$NS" get deploy prefill -o jsonpath='{.status.readyReplicas}')"
    p_avail="$(kubectl -n "$NS" get deploy prefill -o jsonpath='{.status.availableReplicas}')"
    d_spec="$(kubectl -n "$NS" get deploy decode -o jsonpath='{.spec.replicas}')"
    d_ready="$(kubectl -n "$NS" get deploy decode -o jsonpath='{.status.readyReplicas}')"
    d_avail="$(kubectl -n "$NS" get deploy decode -o jsonpath='{.status.availableReplicas}')"
    echo "$ts prefill=$p_spec/$p_ready/$p_avail decode=$d_spec/$d_ready/$d_avail"
    sleep "$SAMPLE_INTERVAL"
  done
) > "$trace_file" &
MON_PID=$!

echo "=== Spike phase ==="
if [[ "$PREFILL_PULSE_ENABLE" == "1" ]]; then
  echo "=== Prefill-only pulse phase ==="
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
ok = 0
err = 0
base_tokens = [((i % 1000) + 10) for i in range(TOKEN_LEN)]

def send(i):
    global ok, err
    req = inference_pb2.PrefillRequest(
        request_id=f"prefill-pulse-{i}-{uuid.uuid4().hex[:8]}",
        token_ids=base_tokens,
        model_name="pulse",
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

print(f"phase_prefill_pulse done total={N} ok={ok} err={err} duration_s={time.time()-start:.1f}", flush=True)
PY
  echo "=== Prefill pulse post-idle (${PREFILL_POST_IDLE}s) ==="
  sleep "$PREFILL_POST_IDLE"
fi

echo "=== Decode spike phase ==="
kubectl -n "$NS" exec -i "$GATEWAY_POD" -- python3 - <<PY
import time, uuid, grpc, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from common.proto import inference_pb2, inference_pb2_grpc

N = int(${SPIKE_N})
RPS = float(${SPIKE_RPS})
W = int(${SPIKE_WORKERS})
MAX_TOKENS = int(${SPIKE_MAX_TOKENS})
TIMEOUT = int(${SPIKE_TIMEOUT})
prompts = [
    "Hello, how are you?",
    "Explain the theory of relativity in simple terms.",
    "Introduce yourself.",
]

ch = grpc.insecure_channel("localhost:50051")
stub = inference_pb2_grpc.GatewayServiceStub(ch)
ok = 0
err = 0

def send(i):
    global ok, err
    req = inference_pb2.InferRequest(
        request_id=f"repro-{i}-{uuid.uuid4().hex[:8]}",
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

print(f"phase_decode_spike done total={N} ok={ok} err={err} duration_s={time.time()-start:.1f}", flush=True)
PY

echo "=== Idle phase (${IDLE_SECONDS}s) ==="
sleep "$IDLE_SECONDS"

cleanup
trap - EXIT

echo "=== Trace (tail) ==="
tail -n 160 "$trace_file"

echo "=== Controller decode events ==="
elapsed="$(( $(date +%s) - start_epoch + 60 ))"
kubectl -n "$NS" logs deploy/scaling-controller --since="${elapsed}s" | egrep 'Scaling prefill|Scaling decode|Oscillation|Scaling weights' || true

echo "=== Analyze decode trajectory ==="
analysis="$(awk '
  /decode=/ {
    if (match($0, /decode=([0-9]+)/, m) == 0) next
    spec=m[1]+0
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

echo "=== Decode replica transitions ==="
awk '
  /decode=/ {
    if (match($0, /decode=([0-9]+)/, m) == 0) next
    ts=$1
    spec=m[1]+0
    if (first || spec != prev) {
      printf("%s %d\n", ts, spec)
      prev=spec
      first=0
    }
  }
' "$trace_file"

echo "=== Prefill replica transitions ==="
awk '
  /prefill=/ {
    if (match($0, /prefill=([0-9]+)/, m) == 0) next
    ts=$1
    spec=m[1]+0
    if (first || spec != prev) {
      printf("%s %d\n", ts, spec)
      prev=spec
      first=0
    }
  }
' "$trace_file"

echo "=== Analyze prefill trajectory ==="
awk '
  /prefill=/ {
    if (match($0, /prefill=([0-9]+)/, m) == 0) next
    spec=m[1]+0
    if (n == 0 || spec < min) min=spec
    if (spec > max) max=spec
    n++
  }
  END {
    if (n == 0) {
      print "prefill: NO_DATA"
    } else {
      printf("prefill: min=%d max=%d samples=%d\n", min, max, n)
    }
  }
' "$trace_file"

echo "=== Manifest ==="
cat "$manifest_file"

if echo "$analysis" | grep -q '^OK '; then
  min_after="$(echo "$analysis" | sed -n 's/.*min_after_target=\([0-9]\+\).*/\1/p')"
  if [[ -n "$min_after" && "$min_after" -le 1 ]]; then
    echo "RESULT: PASS (observed scale up to >=${TARGET_MAX} and scale down to <=1)"
    exit 0
  fi
  echo "RESULT: FAIL (scaled up, but did not scale down to 1; min_after_target=$min_after)"
  exit 2
fi

echo "RESULT: FAIL ($analysis)"
exit 2
