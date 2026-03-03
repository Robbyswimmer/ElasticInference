#!/usr/bin/env python3
"""Multi-model autotune experiment orchestrator.

Manages the full service lifecycle: start workers -> run experiment -> kill workers
-> change config -> restart -> repeat. Loops over all 4 models.

Phases per model:
  A) Profile GPU contention levels via concurrent load
  B) Build autotune lookup table from profiles (pure math, no services)
  C) Autotune comparison with REAL config changes (restart between OFF/ON)
  D) Ablation sweep with REAL config changes (restart per variant)
  E) Per-model figures

Then: cross-model comparison figures + summary.
"""

import copy
import json
import logging
import os
import signal
import subprocess
import sys
import time
from concurrent import futures

import grpc
import numpy as np
import yaml

# Add project root to path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from common import load_config
from common.proto import inference_pb2, inference_pb2_grpc
from common.gpu_utils import log_gpu_info
from autotune.profiler import KneeProfiler
from autotune.selector import ConfigSelector
from eval.runner import run_single_experiment
from eval.workloads import SAMPLE_PROMPTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

GATEWAY_TARGET = "localhost:50051"
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
TMP_CONFIG_DIR = os.path.join(PROJECT_DIR, ".tmp_configs")

# Models to test (smallest to largest)
MODELS = [
    "facebook/opt-1.3b",
    "Qwen/Qwen2-1.5B",
    "microsoft/phi-2",
    "facebook/opt-6.7b",
]

# Concurrency-to-GPU% mapping
CONCURRENCY_GPU_MAP = {1: 100, 2: 50, 4: 25}

# Experiment parameters
NUM_WARMUP = 3
NUM_MEASURE = 15
NUM_EVAL_REQUESTS = 30
ARRIVAL_RATE = 1.0
MAX_TOKENS = 128

# Default batch sizes (from config.yaml)
DEFAULT_PREFILL_BATCH = 8
DEFAULT_DECODE_BATCH = 32

# Default MoE config
DEFAULT_MOE_EXPERTS = 8
DEFAULT_MOE_TOP_K = 2

# Ablation variant definitions
ABLATION_VARIANTS = {
    "full_system": {
        "description": "Optimal batches + MoE ON",
        "use_optimal_batches": True,
        "moe_num_experts": DEFAULT_MOE_EXPERTS,
        "moe_top_k": DEFAULT_MOE_TOP_K,
    },
    "no_autotune": {
        "description": "Default batches + MoE ON",
        "use_optimal_batches": False,
        "moe_num_experts": DEFAULT_MOE_EXPERTS,
        "moe_top_k": DEFAULT_MOE_TOP_K,
    },
    "no_moe": {
        "description": "Optimal batches + MoE OFF",
        "use_optimal_batches": True,
        "moe_num_experts": 1,
        "moe_top_k": 1,
    },
    "static_baseline": {
        "description": "Default batches + MoE OFF",
        "use_optimal_batches": False,
        "moe_num_experts": 1,
        "moe_top_k": 1,
    },
}


# ---------------------------------------------------------------------------
# Service Manager
# ---------------------------------------------------------------------------

class ServiceManager:
    """Manages prefill/decode/gateway subprocess lifecycle."""

    def __init__(self):
        self._processes = []  # list of (name, Popen)

    def start_services(self, config_path, label):
        """Start prefill, decode, and gateway with given config.

        Sets CONFIG_PATH env var so workers load the right config file.
        Polls gRPC endpoints until responsive.
        """
        logger.info("=== Starting services for: %s ===", label)
        logger.info("Config: %s", config_path)

        env = os.environ.copy()
        env["CONFIG_PATH"] = config_path
        env["REDIS_HOST"] = "localhost"
        env["PREFILL_HOST"] = "localhost"
        env["DECODE_HOST"] = "localhost"

        log_dir = os.path.join(PROJECT_DIR, "logs")
        os.makedirs(log_dir, exist_ok=True)

        safe_label = label.replace("/", "_").replace(" ", "_")

        # Start prefill worker
        prefill_log = open(os.path.join(log_dir, f"prefill_{safe_label}.log"), "w")
        prefill_proc = subprocess.Popen(
            [sys.executable, "-m", "workers.prefill"],
            cwd=PROJECT_DIR,
            env=env,
            stdout=prefill_log,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        self._processes.append(("prefill", prefill_proc, prefill_log))
        logger.info("Prefill PID: %d", prefill_proc.pid)

        # Start decode worker
        decode_log = open(os.path.join(log_dir, f"decode_{safe_label}.log"), "w")
        decode_proc = subprocess.Popen(
            [sys.executable, "-m", "workers.decode"],
            cwd=PROJECT_DIR,
            env=env,
            stdout=decode_log,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        self._processes.append(("decode", decode_proc, decode_log))
        logger.info("Decode PID: %d", decode_proc.pid)

        # Wait for model loading — poll prefill/decode gRPC ports
        logger.info("Waiting for workers to load model...")
        self._wait_for_grpc("localhost:50052", "prefill", timeout=180)
        self._wait_for_grpc("localhost:50053", "decode", timeout=180)

        # Start gateway
        gateway_log = open(os.path.join(log_dir, f"gateway_{safe_label}.log"), "w")
        gateway_proc = subprocess.Popen(
            [sys.executable, "-m", "gateway.server"],
            cwd=PROJECT_DIR,
            env=env,
            stdout=gateway_log,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        self._processes.append(("gateway", gateway_proc, gateway_log))
        logger.info("Gateway PID: %d", gateway_proc.pid)

        # Wait for gateway to be ready
        if not self._wait_for_gateway(GATEWAY_TARGET, timeout=60):
            logger.error("Gateway failed to become ready for %s", label)
            self.stop_all()
            raise RuntimeError(f"Gateway not ready for {label}")

        logger.info("All services ready for: %s", label)

    def stop_all(self):
        """Stop all managed services. SIGTERM -> wait -> SIGKILL if needed."""
        if not self._processes:
            return

        logger.info("=== Stopping all services ===")

        # Send SIGTERM to process groups
        for name, proc, logfile in self._processes:
            try:
                if proc.poll() is None:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    logger.info("Sent SIGTERM to %s (PID %d)", name, proc.pid)
            except (ProcessLookupError, OSError):
                pass

        # Wait up to 10s for graceful shutdown
        deadline = time.time() + 10
        for name, proc, logfile in self._processes:
            remaining = max(0, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                pass

        # SIGKILL any survivors
        for name, proc, logfile in self._processes:
            try:
                if proc.poll() is None:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    logger.warning("Sent SIGKILL to %s (PID %d)", name, proc.pid)
                    proc.wait(timeout=5)
            except (ProcessLookupError, OSError, subprocess.TimeoutExpired):
                pass
            finally:
                logfile.close()

        self._processes.clear()

        # Wait for GPU memory to be released
        self._wait_for_gpu_release()

        # Brief delay for port release
        time.sleep(3)
        logger.info("All services stopped")

    def _wait_for_grpc(self, target, name, timeout=180):
        """Poll a gRPC endpoint until it responds."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            # Check if process died
            for pname, proc, _ in self._processes:
                if pname == name and proc.poll() is not None:
                    logger.error("%s process died (exit code %d)", name, proc.returncode)
                    raise RuntimeError(f"{name} process died during startup")
            try:
                channel = grpc.insecure_channel(target)
                grpc.channel_ready_future(channel).result(timeout=5)
                channel.close()
                logger.info("%s ready at %s", name, target)
                return
            except Exception:
                time.sleep(5)
        raise RuntimeError(f"{name} not ready at {target} after {timeout}s")

    def _wait_for_gateway(self, target, timeout=60):
        """Wait until gateway responds to inference request."""
        channel = grpc.insecure_channel(target)
        stub = inference_pb2_grpc.GatewayServiceStub(channel)
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                req = inference_pb2.InferRequest(
                    request_id="health-check",
                    prompt="Hello",
                    max_tokens=1,
                )
                stub.Infer(req, timeout=15)
                channel.close()
                return True
            except grpc.RpcError:
                time.sleep(3)
        channel.close()
        return False

    def _wait_for_gpu_release(self, timeout=30):
        """Wait until GPU memory usage drops below threshold."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    mem_used = int(result.stdout.strip().split("\n")[0])
                    if mem_used < 500:
                        logger.info("GPU memory released (%d MB used)", mem_used)
                        return
                    logger.info("Waiting for GPU memory release (%d MB)...", mem_used)
            except Exception:
                pass
            time.sleep(3)
        logger.warning("GPU memory release timed out after %ds", timeout)


# ---------------------------------------------------------------------------
# Config Generation
# ---------------------------------------------------------------------------

def load_base_config():
    """Load the base config.yaml."""
    config_path = os.path.join(PROJECT_DIR, "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_config(model_name, prefill_batch=None, decode_batch=None,
                    moe_num_experts=None, moe_top_k=None, label="default"):
    """Generate a temporary config file with overrides.

    Returns the path to the written config file.
    """
    os.makedirs(TMP_CONFIG_DIR, exist_ok=True)

    config = load_base_config()

    # Model
    config["model"]["name"] = model_name

    # Batch sizes
    if prefill_batch is not None:
        config["prefill"]["batch_size"] = prefill_batch
    if decode_batch is not None:
        config["decode"]["max_batch_size"] = decode_batch

    # MoE
    if moe_num_experts is not None:
        config["decode"]["moe_num_experts"] = moe_num_experts
    if moe_top_k is not None:
        config["decode"]["moe_top_k"] = moe_top_k

    # Localhost overrides
    config["prefill"]["connect_host"] = "localhost"
    config["decode"]["connect_host"] = "localhost"
    config["redis"]["host"] = "localhost"

    # Disable prometheus to avoid port conflicts between restarts
    config["prometheus"] = {"enabled": False}
    config["scaling"]["prometheus_enabled"] = False

    safe_name = model_name.replace("/", "_")
    safe_label = label.replace("/", "_").replace(" ", "_")
    config_path = os.path.join(TMP_CONFIG_DIR, f"{safe_name}_{safe_label}.yaml")

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info("Generated config: %s", config_path)
    return config_path


# ---------------------------------------------------------------------------
# GPU Memory Check
# ---------------------------------------------------------------------------

def check_gpu_memory(model_name):
    """Check if there's enough GPU memory for the model. Returns True if OK."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            logger.warning("nvidia-smi failed, proceeding anyway")
            return True

        line = result.stdout.strip().split("\n")[0]
        total, used = [int(x.strip()) for x in line.split(",")]
        available = total - used

        # Rough memory estimates (fp16, both workers load model)
        model_sizes = {
            "facebook/opt-1.3b": 3000,    # ~2.6GB x 2
            "Qwen/Qwen2-1.5B": 3500,     # ~3GB x 2
            "microsoft/phi-2": 6000,      # ~5.5GB x 2
            "facebook/opt-6.7b": 26000,   # ~13GB x 2
        }
        needed = model_sizes.get(model_name, 10000)

        if available < needed:
            logger.warning("Insufficient GPU memory for %s: %dMB available, ~%dMB needed",
                          model_name, available, needed)
            return False

        logger.info("GPU memory check OK for %s: %dMB available, ~%dMB needed",
                    model_name, available, needed)
        return True

    except Exception as e:
        logger.warning("GPU memory check failed: %s, proceeding anyway", e)
        return True


# ---------------------------------------------------------------------------
# Profiling (Phase A)
# ---------------------------------------------------------------------------

def send_single_request(stub, prompt, request_id, max_tokens):
    """Send one inference request and return latency breakdown."""
    req = inference_pb2.InferRequest(
        request_id=request_id,
        prompt=prompt,
        max_tokens=max_tokens,
    )
    try:
        resp = stub.Infer(req, timeout=120)
        return {
            "request_id": resp.request_id,
            "prefill_ms": resp.latency.prefill_ms,
            "decode_ms": resp.latency.decode_ms,
            "total_ms": resp.latency.total_ms,
            "tokens_generated": resp.tokens_generated,
        }
    except grpc.RpcError as e:
        logger.warning("Request %s failed: %s", request_id, e)
        return None


def profile_at_concurrency(concurrency, num_warmup, num_measure):
    """Profile the system at a given concurrency level."""
    gpu_pct = CONCURRENCY_GPU_MAP[concurrency]
    logger.info("=== Profiling at concurrency=%d (simulated GPU%%=%d) ===",
                concurrency, gpu_pct)

    channel = grpc.insecure_channel(GATEWAY_TARGET)
    stub = inference_pb2_grpc.GatewayServiceStub(channel)

    # Warmup
    logger.info("Sending %d warmup requests...", num_warmup)
    for i in range(num_warmup):
        prompt = SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]
        send_single_request(stub, prompt, f"warmup-{concurrency}-{i}", MAX_TOKENS)

    # Measurement
    logger.info("Sending %d measurement requests at concurrency %d...",
                num_measure, concurrency)
    all_results = []
    pool = futures.ThreadPoolExecutor(max_workers=concurrency)

    batch_idx = 0
    while len(all_results) < num_measure:
        batch_futures = []
        for c in range(concurrency):
            idx = batch_idx * concurrency + c
            prompt = SAMPLE_PROMPTS[idx % len(SAMPLE_PROMPTS)]
            rid = f"profile-c{concurrency}-{idx}"
            fut = pool.submit(send_single_request, stub, prompt, rid, MAX_TOKENS)
            batch_futures.append(fut)

        for fut in futures.as_completed(batch_futures):
            result = fut.result()
            if result is not None:
                all_results.append(result)

        batch_idx += 1

    pool.shutdown()
    channel.close()

    all_results = all_results[:num_measure]

    if not all_results:
        logger.error("No successful measurements at concurrency %d", concurrency)
        return None

    prefill_lats = [r["prefill_ms"] for r in all_results]
    decode_lats = [r["decode_ms"] for r in all_results]
    total_lats = [r["total_ms"] for r in all_results]
    tokens = [r["tokens_generated"] for r in all_results]
    throughputs = [t / (lat / 1000.0) for t, lat in zip(tokens, total_lats) if lat > 0]

    return {
        "gpu_pct": gpu_pct,
        "concurrency": concurrency,
        "num_measurements": len(all_results),
        "prefill": {
            "avg_latency_ms": float(np.mean(prefill_lats)),
            "p95_latency_ms": float(np.percentile(prefill_lats, 95)),
            "p99_latency_ms": float(np.percentile(prefill_lats, 99)),
        },
        "decode": {
            "avg_latency_ms": float(np.mean(decode_lats)),
            "p95_latency_ms": float(np.percentile(decode_lats, 95)),
            "p99_latency_ms": float(np.percentile(decode_lats, 99)),
        },
        "e2e": {
            "avg_latency_ms": float(np.mean(total_lats)),
            "p95_latency_ms": float(np.percentile(total_lats, 95)),
            "p99_latency_ms": float(np.percentile(total_lats, 99)),
        },
        "avg_throughput_tps": float(np.mean(throughputs)) if throughputs else 0.0,
    }


def run_phase_a(svc, model_name, model_results_dir):
    """Phase A: Profile at all concurrency levels."""
    logger.info("=" * 60)
    logger.info("PHASE A: GPU%% Simulation via Concurrent Load Profiling")
    logger.info("=" * 60)

    # Start services with default config
    config_path = generate_config(model_name, label="profile")
    svc.start_services(config_path, f"{model_name}_profile")

    profile_data = []
    for concurrency in sorted(CONCURRENCY_GPU_MAP.keys()):
        result = profile_at_concurrency(concurrency, NUM_WARMUP, NUM_MEASURE)
        if result:
            profile_data.append(result)
        time.sleep(5)

    svc.stop_all()

    # Save profile data
    profile_path = os.path.join(model_results_dir, "profile_data.json")
    with open(profile_path, "w") as f:
        json.dump(profile_data, f, indent=2)
    logger.info("Saved profile data to %s (%d levels)", profile_path, len(profile_data))

    return profile_data


# ---------------------------------------------------------------------------
# Knee Analysis (Phase B) — pure math, no services needed
# ---------------------------------------------------------------------------

def run_phase_b(profile_data, model_results_dir):
    """Phase B: Build autotune lookup table and find knee points."""
    logger.info("=" * 60)
    logger.info("PHASE B: Build Autotune Lookup Table")
    logger.info("=" * 60)

    config = load_base_config()
    selector = ConfigSelector(config)
    profiler = KneeProfiler(config)

    prefill_results = []
    decode_results = []
    for entry in profile_data:
        prefill_results.append({
            "gpu_pct": entry["gpu_pct"],
            "stage": "prefill",
            "avg_latency_ms": entry["prefill"]["avg_latency_ms"],
            "p95_latency_ms": entry["prefill"]["p95_latency_ms"],
            "p99_latency_ms": entry["prefill"]["p99_latency_ms"],
            "avg_throughput_tps": entry["avg_throughput_tps"],
            "best_batch_size": config["prefill"].get("batch_size", 8),
        })
        decode_results.append({
            "gpu_pct": entry["gpu_pct"],
            "stage": "decode",
            "avg_latency_ms": entry["decode"]["avg_latency_ms"],
            "p95_latency_ms": entry["decode"]["p95_latency_ms"],
            "p99_latency_ms": entry["decode"]["p99_latency_ms"],
            "avg_throughput_tps": entry["avg_throughput_tps"],
            "best_batch_size": config["decode"].get("max_batch_size", 32),
        })

    selector.update_from_profile("prefill", prefill_results)
    selector.update_from_profile("decode", decode_results)

    knee_analysis = {}
    prefill_knee = profiler.find_knee(prefill_results)
    decode_knee = profiler.find_knee(decode_results)
    knee_analysis["prefill"] = prefill_knee
    knee_analysis["decode"] = decode_knee

    logger.info("Prefill knee: GPU%%=%s, avg_latency=%.1fms",
                prefill_knee.get("gpu_pct") if prefill_knee else "N/A",
                prefill_knee.get("avg_latency_ms", 0) if prefill_knee else 0)
    logger.info("Decode knee: GPU%%=%s, avg_latency=%.1fms",
                decode_knee.get("gpu_pct") if decode_knee else "N/A",
                decode_knee.get("avg_latency_ms", 0) if decode_knee else 0)

    # Determine optimal batch sizes from knee analysis
    # Use the batch size at the knee point's GPU% level, or pick from config space
    optimal_batches = _compute_optimal_batches(profile_data, config)
    knee_analysis["optimal_batches"] = optimal_batches

    knee_path = os.path.join(model_results_dir, "knee_analysis.json")
    with open(knee_path, "w") as f:
        json.dump(knee_analysis, f, indent=2)
    logger.info("Saved knee analysis to %s", knee_path)

    return knee_analysis, optimal_batches


def _compute_optimal_batches(profile_data, config):
    """Determine optimal batch sizes based on profiling data.

    Heuristic: at high concurrency (low GPU%), smaller batches reduce
    contention. At low concurrency (high GPU%), larger batches improve
    throughput. Pick batch sizes that minimize p95 latency at 50% GPU.
    """
    batch_options_prefill = config.get("autotune", {}).get("config_space", {}).get(
        "prefill_batch_size", [1, 2, 4, 8, 16])
    batch_options_decode = config.get("autotune", {}).get("config_space", {}).get(
        "decode_batch_size", [4, 8, 16, 32, 64])

    # Find knee point data (50% GPU is typical knee)
    knee_entry = None
    for entry in profile_data:
        if entry["gpu_pct"] == 50:
            knee_entry = entry
            break
    if knee_entry is None and profile_data:
        knee_entry = profile_data[0]

    # If latency at 50% is significantly higher than at 100%, reduce batch sizes
    if len(profile_data) >= 2:
        lat_100 = next((e["e2e"]["p95_latency_ms"] for e in profile_data
                       if e["gpu_pct"] == 100), None)
        lat_50 = next((e["e2e"]["p95_latency_ms"] for e in profile_data
                      if e["gpu_pct"] == 50), None)

        if lat_100 and lat_50 and lat_50 > lat_100 * 1.5:
            # Significant contention — reduce batch sizes
            optimal_prefill = max(1, DEFAULT_PREFILL_BATCH // 2)
            optimal_decode = max(4, DEFAULT_DECODE_BATCH // 2)
        else:
            # Moderate contention — slight reduction
            optimal_prefill = DEFAULT_PREFILL_BATCH
            optimal_decode = max(8, DEFAULT_DECODE_BATCH - 8)
    else:
        optimal_prefill = DEFAULT_PREFILL_BATCH
        optimal_decode = DEFAULT_DECODE_BATCH

    # Snap to nearest valid option
    optimal_prefill = min(batch_options_prefill,
                         key=lambda x: abs(x - optimal_prefill))
    optimal_decode = min(batch_options_decode,
                        key=lambda x: abs(x - optimal_decode))

    result = {
        "prefill_batch_size": optimal_prefill,
        "decode_batch_size": optimal_decode,
    }
    logger.info("Optimal batch sizes: prefill=%d, decode=%d",
                optimal_prefill, optimal_decode)
    return result


# ---------------------------------------------------------------------------
# Autotune Comparison (Phase C)
# ---------------------------------------------------------------------------

def run_phase_c(svc, model_name, optimal_batches, model_results_dir):
    """Phase C: Autotune OFF vs ON with real config changes."""
    logger.info("=" * 60)
    logger.info("PHASE C: Autotune OFF vs ON (Real Config Changes)")
    logger.info("=" * 60)

    results = {}

    # --- Autotune OFF: default batch sizes ---
    logger.info("--- Running: autotune OFF (default batch sizes) ---")
    config_path = generate_config(model_name, label="autotune_off")
    svc.start_services(config_path, f"{model_name}_autotune_off")

    off_summary = run_single_experiment(
        gateway_target=GATEWAY_TARGET,
        name="autotune_off",
        num_requests=NUM_EVAL_REQUESTS,
        arrival_rate=ARRIVAL_RATE,
        max_tokens=MAX_TOKENS,
        gpu_count=1,
        gpu_pct=100,
        output_dir=model_results_dir,
    )
    results["autotune_off"] = off_summary
    svc.stop_all()

    # --- Autotune ON: optimal batch sizes ---
    logger.info("--- Running: autotune ON (optimal batch sizes) ---")
    config_path = generate_config(
        model_name,
        prefill_batch=optimal_batches["prefill_batch_size"],
        decode_batch=optimal_batches["decode_batch_size"],
        label="autotune_on",
    )
    svc.start_services(config_path, f"{model_name}_autotune_on")

    on_summary = run_single_experiment(
        gateway_target=GATEWAY_TARGET,
        name="autotune_on",
        num_requests=NUM_EVAL_REQUESTS,
        arrival_rate=ARRIVAL_RATE,
        max_tokens=MAX_TOKENS,
        gpu_count=1,
        gpu_pct=100,
        output_dir=model_results_dir,
    )
    results["autotune_on"] = on_summary
    svc.stop_all()

    # Compute improvements
    comparison = {
        "autotune_off": off_summary,
        "autotune_on": on_summary,
        "improvement": {},
        "config": {
            "off": {"prefill_batch": DEFAULT_PREFILL_BATCH,
                    "decode_batch": DEFAULT_DECODE_BATCH},
            "on": optimal_batches,
        },
    }

    for metric in ["e2e_p50_ms", "e2e_p95_ms", "e2e_p99_ms",
                   "throughput_tps", "gpu_seconds_per_request"]:
        off_val = off_summary.get(metric, 0)
        on_val = on_summary.get(metric, 0)
        if off_val > 0:
            if "throughput" in metric:
                pct_change = ((on_val - off_val) / off_val) * 100
            else:
                pct_change = ((off_val - on_val) / off_val) * 100
            comparison["improvement"][metric] = {
                "off": off_val,
                "on": on_val,
                "change_pct": round(pct_change, 2),
            }

    comparison_path = os.path.join(model_results_dir, "comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info("Saved comparison to %s", comparison_path)

    return results


# ---------------------------------------------------------------------------
# Ablation Sweep (Phase D)
# ---------------------------------------------------------------------------

def run_phase_d(svc, model_name, optimal_batches, model_results_dir):
    """Phase D: Ablation study with real config changes per variant."""
    logger.info("=" * 60)
    logger.info("PHASE D: Ablation Sweep (Real Config Changes)")
    logger.info("=" * 60)

    ablation_results = {}

    for variant_name, variant_cfg in ABLATION_VARIANTS.items():
        logger.info("--- Ablation: %s — %s ---", variant_name, variant_cfg["description"])

        # Determine batch sizes
        if variant_cfg["use_optimal_batches"]:
            pb = optimal_batches["prefill_batch_size"]
            db = optimal_batches["decode_batch_size"]
        else:
            pb = DEFAULT_PREFILL_BATCH
            db = DEFAULT_DECODE_BATCH

        # Generate config with this variant's settings
        config_path = generate_config(
            model_name,
            prefill_batch=pb,
            decode_batch=db,
            moe_num_experts=variant_cfg["moe_num_experts"],
            moe_top_k=variant_cfg["moe_top_k"],
            label=f"ablation_{variant_name}",
        )

        svc.start_services(config_path, f"{model_name}_{variant_name}")

        summary = run_single_experiment(
            gateway_target=GATEWAY_TARGET,
            name=f"ablation_{variant_name}",
            num_requests=NUM_EVAL_REQUESTS,
            arrival_rate=ARRIVAL_RATE,
            max_tokens=MAX_TOKENS,
            gpu_count=1,
            gpu_pct=100,
            output_dir=model_results_dir,
        )
        ablation_results[variant_name] = summary

        svc.stop_all()

    # Save ablation results
    ablation_path = os.path.join(model_results_dir, "ablation.json")
    with open(ablation_path, "w") as f:
        json.dump(ablation_results, f, indent=2)
    logger.info("Saved ablation results to %s", ablation_path)

    # Print summary table
    logger.info("=== Ablation Summary ===")
    logger.info("%-20s %10s %10s %10s %12s %8s %6s",
                "Config", "p95(ms)", "p99(ms)", "TPS", "GPU-s/req",
                "MoE-exp", "Batch")
    for name, s in ablation_results.items():
        v = ABLATION_VARIANTS[name]
        batch_label = "opt" if v["use_optimal_batches"] else "def"
        logger.info("%-20s %10.1f %10.1f %10.1f %12.3f %8d %6s",
                    name,
                    s.get("e2e_p95_ms", 0),
                    s.get("e2e_p99_ms", 0),
                    s.get("throughput_tps", 0),
                    s.get("gpu_seconds_per_request", 0),
                    v["moe_num_experts"],
                    batch_label)

    return ablation_results


# ---------------------------------------------------------------------------
# Per-Model Figures (Phase E)
# ---------------------------------------------------------------------------

def run_phase_e(model_name, model_results_dir):
    """Phase E: Generate per-model figures."""
    logger.info("=" * 60)
    logger.info("PHASE E: Per-Model Figures for %s", model_name)
    logger.info("=" * 60)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping figures")
        return

    figures_dir = os.path.join(model_results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    safe_name = model_name.split("/")[-1]

    # Load results
    try:
        with open(os.path.join(model_results_dir, "profile_data.json")) as f:
            profile_data = json.load(f)
        with open(os.path.join(model_results_dir, "comparison.json")) as f:
            comparison = json.load(f)
        with open(os.path.join(model_results_dir, "ablation.json")) as f:
            ablation = json.load(f)
    except FileNotFoundError as e:
        logger.warning("Missing results file for figures: %s", e)
        return

    # Fig 1: Profiling latency
    _plot_profiling_latency(profile_data, safe_name, figures_dir)

    # Fig 2: Autotune comparison
    _plot_autotune_comparison(comparison, safe_name, figures_dir)

    # Fig 3: Ablation
    _plot_ablation(ablation, safe_name, figures_dir)

    logger.info("Per-model figures saved to %s", figures_dir)


def _plot_profiling_latency(profile_data, model_label, figures_dir):
    """Plot profiling latency vs GPU%."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gpu_pcts = [d["gpu_pct"] for d in profile_data]
    prefill_avg = [d["prefill"]["avg_latency_ms"] for d in profile_data]
    decode_avg = [d["decode"]["avg_latency_ms"] for d in profile_data]
    throughputs = [d["avg_throughput_tps"] for d in profile_data]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    ax1.plot(gpu_pcts, prefill_avg, "o-", linewidth=2.5, markersize=10, color="#2E86AB")
    ax1.set_xlabel("Simulated GPU %")
    ax1.set_ylabel("Prefill Latency (ms)")
    ax1.set_title("Prefill Latency")
    ax1.set_xticks(gpu_pcts)
    ax1.invert_xaxis()
    ax1.grid(True, alpha=0.3)

    ax2.plot(gpu_pcts, decode_avg, "s-", linewidth=2.5, markersize=10, color="#F5A623")
    ax2.set_xlabel("Simulated GPU %")
    ax2.set_ylabel("Decode Latency (ms)")
    ax2.set_title("Decode Latency")
    ax2.set_xticks(gpu_pcts)
    ax2.invert_xaxis()
    ax2.grid(True, alpha=0.3)

    ax3.bar(gpu_pcts, throughputs, width=15, color=["#44BBA4", "#F5A623", "#E84855"])
    ax3.set_xlabel("Simulated GPU %")
    ax3.set_ylabel("Throughput (tokens/sec)")
    ax3.set_title("Throughput")
    ax3.set_xticks(gpu_pcts)
    ax3.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"GPU Contention Profiling — {model_label}",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "profiling.png"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(figures_dir, "profiling.pdf"), bbox_inches="tight")
    plt.close()


def _plot_autotune_comparison(comparison, model_label, figures_dir):
    """Plot autotune OFF vs ON comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    off = comparison["autotune_off"]
    on = comparison["autotune_on"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Latency comparison
    labels = ["p50", "p95", "p99"]
    off_vals = [off.get("e2e_p50_ms", 0), off.get("e2e_p95_ms", 0), off.get("e2e_p99_ms", 0)]
    on_vals = [on.get("e2e_p50_ms", 0), on.get("e2e_p95_ms", 0), on.get("e2e_p99_ms", 0)]

    x = np.arange(len(labels))
    width = 0.35
    ax1.bar(x - width / 2, off_vals, width, label="Autotune OFF", color="#E84855")
    ax1.bar(x + width / 2, on_vals, width, label="Autotune ON", color="#44BBA4")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("End-to-End Latency")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Throughput + GPU efficiency
    categories = ["Autotune OFF", "Autotune ON"]
    tps = [off.get("throughput_tps", 0), on.get("throughput_tps", 0)]
    ax2.bar(categories, tps, color=["#E84855", "#44BBA4"])
    ax2.set_ylabel("Tokens/sec")
    ax2.set_title("Throughput")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Autotune Comparison — {model_label}",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "autotune_comparison.png"), dpi=150,
                bbox_inches="tight")
    plt.savefig(os.path.join(figures_dir, "autotune_comparison.pdf"), bbox_inches="tight")
    plt.close()


def _plot_ablation(ablation, model_label, figures_dir):
    """Plot ablation study results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = list(ablation.keys())
    p95s = [ablation[c].get("e2e_p95_ms", 0) for c in configs]
    tps = [ablation[c].get("throughput_tps", 0) for c in configs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # p95 latency
    colors = []
    for val in p95s:
        if val < 3500:
            colors.append("#44BBA4")
        elif val < 6000:
            colors.append("#F5A623")
        else:
            colors.append("#E84855")

    bars = ax1.barh(range(len(configs)), p95s, color=colors, edgecolor="white",
                    linewidth=1.5, height=0.6)
    ax1.set_yticks(range(len(configs)))
    ax1.set_yticklabels([c.replace("_", "\n") for c in configs])
    ax1.set_xlabel("p95 Latency (ms)")
    ax1.set_title("Tail Latency by Config")
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis="x")

    # Throughput
    ax2.barh(range(len(configs)), tps, color="#2E86AB", edgecolor="white",
             linewidth=1.5, height=0.6)
    ax2.set_yticks(range(len(configs)))
    ax2.set_yticklabels([c.replace("_", "\n") for c in configs])
    ax2.set_xlabel("Throughput (tokens/sec)")
    ax2.set_title("Throughput by Config")
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis="x")

    fig.suptitle(f"Ablation Study — {model_label}",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "ablation.png"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(figures_dir, "ablation.pdf"), bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Cross-Model Figures
# ---------------------------------------------------------------------------

def generate_cross_model_figures(all_model_results):
    """Generate cross-model comparison figures."""
    logger.info("=" * 60)
    logger.info("Generating Cross-Model Comparison Figures")
    logger.info("=" * 60)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping cross-model figures")
        return

    cross_dir = os.path.join(RESULTS_DIR, "cross_model")
    os.makedirs(cross_dir, exist_ok=True)

    model_labels = []
    for model_name in all_model_results:
        model_labels.append(model_name.split("/")[-1])

    # Color palette for models
    model_colors = ["#2E86AB", "#E84855", "#44BBA4", "#F5A623"]

    # --- Figure 1: Profiling comparison ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for i, (model_name, data) in enumerate(all_model_results.items()):
        profile = data.get("profile_data", [])
        if not profile:
            continue
        label = model_labels[i]
        color = model_colors[i % len(model_colors)]
        gpu_pcts = [d["gpu_pct"] for d in profile]
        decode_lats = [d["decode"]["avg_latency_ms"] for d in profile]
        throughputs = [d["avg_throughput_tps"] for d in profile]

        ax1.plot(gpu_pcts, decode_lats, "o-", label=label, color=color,
                linewidth=2, markersize=8)
        ax2.plot(gpu_pcts, throughputs, "s-", label=label, color=color,
                linewidth=2, markersize=8)

    ax1.set_xlabel("Simulated GPU %")
    ax1.set_ylabel("Decode Latency (ms)")
    ax1.set_title("Decode Latency vs GPU Contention")
    ax1.legend()
    ax1.invert_xaxis()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Simulated GPU %")
    ax2.set_ylabel("Throughput (tokens/sec)")
    ax2.set_title("Throughput vs GPU Contention")
    ax2.legend()
    ax2.invert_xaxis()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Cross-Model Profiling Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(cross_dir, "fig1_profiling_comparison.png"), dpi=200,
                bbox_inches="tight")
    plt.savefig(os.path.join(cross_dir, "fig1_profiling_comparison.pdf"), bbox_inches="tight")
    plt.close()
    logger.info("  Saved fig1_profiling_comparison")

    # --- Figure 2: Autotune improvement per model ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    improvements_p95 = []
    improvements_tps = []
    improvements_gpu = []

    for model_name, data in all_model_results.items():
        comp = data.get("comparison", {})
        imp = comp.get("improvement", {})
        improvements_p95.append(imp.get("e2e_p95_ms", {}).get("change_pct", 0))
        improvements_tps.append(imp.get("throughput_tps", {}).get("change_pct", 0))
        improvements_gpu.append(imp.get("gpu_seconds_per_request", {}).get("change_pct", 0))

    x = np.arange(len(model_labels))
    colors = [c for c in model_colors[:len(model_labels)]]

    ax1.bar(x, improvements_p95, color=colors, edgecolor="white", linewidth=1.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, rotation=15)
    ax1.set_ylabel("Improvement (%)")
    ax1.set_title("p95 Latency Reduction")
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(x, improvements_tps, color=colors, edgecolor="white", linewidth=1.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_labels, rotation=15)
    ax2.set_ylabel("Improvement (%)")
    ax2.set_title("Throughput Improvement")
    ax2.grid(True, alpha=0.3, axis="y")

    ax3.bar(x, improvements_gpu, color=colors, edgecolor="white", linewidth=1.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_labels, rotation=15)
    ax3.set_ylabel("Improvement (%)")
    ax3.set_title("GPU Cost Reduction")
    ax3.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Autotune Improvement Across Models", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(cross_dir, "fig2_autotune_improvement.png"), dpi=200,
                bbox_inches="tight")
    plt.savefig(os.path.join(cross_dir, "fig2_autotune_improvement.pdf"), bbox_inches="tight")
    plt.close()
    logger.info("  Saved fig2_autotune_improvement")

    # --- Figure 3: Ablation comparison across models ---
    fig, ax = plt.subplots(figsize=(14, 7))

    ablation_keys = list(ABLATION_VARIANTS.keys())
    bar_width = 0.18
    x = np.arange(len(ablation_keys))

    for i, (model_name, data) in enumerate(all_model_results.items()):
        ablation = data.get("ablation", {})
        vals = [ablation.get(k, {}).get("e2e_p95_ms", 0) for k in ablation_keys]
        ax.bar(x + i * bar_width, vals, bar_width, label=model_labels[i],
               color=model_colors[i % len(model_colors)], edgecolor="white", linewidth=1)

    ax.set_xticks(x + bar_width * (len(all_model_results) - 1) / 2)
    ax.set_xticklabels([k.replace("_", "\n") for k in ablation_keys])
    ax.set_ylabel("p95 Latency (ms)")
    ax.set_title("Ablation: p95 Latency Across Models")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(cross_dir, "fig3_ablation_comparison.png"), dpi=200,
                bbox_inches="tight")
    plt.savefig(os.path.join(cross_dir, "fig3_ablation_comparison.pdf"), bbox_inches="tight")
    plt.close()
    logger.info("  Saved fig3_ablation_comparison")

    # --- Figure 4: Model scaling (latency/throughput vs model size) ---
    model_sizes = {
        "opt-1.3b": 1.3,
        "Qwen2-1.5B": 1.5,
        "phi-2": 2.7,
        "opt-6.7b": 6.7,
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    sizes = []
    lats_full = []
    tps_full = []
    labels_plot = []

    for i, (model_name, data) in enumerate(all_model_results.items()):
        label = model_labels[i]
        size = model_sizes.get(label, 1.0)
        ablation = data.get("ablation", {})
        full = ablation.get("full_system", {})
        if full:
            sizes.append(size)
            lats_full.append(full.get("e2e_p95_ms", 0))
            tps_full.append(full.get("throughput_tps", 0))
            labels_plot.append(label)

    if sizes:
        ax1.plot(sizes, lats_full, "o-", color="#E84855", linewidth=2.5, markersize=12)
        for i, label in enumerate(labels_plot):
            ax1.annotate(f"  {label}", (sizes[i], lats_full[i]),
                        fontsize=10, fontweight="bold")
        ax1.set_xlabel("Model Size (B parameters)")
        ax1.set_ylabel("p95 Latency (ms)")
        ax1.set_title("Latency Scaling")
        ax1.grid(True, alpha=0.3)

        ax2.plot(sizes, tps_full, "s-", color="#44BBA4", linewidth=2.5, markersize=12)
        for i, label in enumerate(labels_plot):
            ax2.annotate(f"  {label}", (sizes[i], tps_full[i]),
                        fontsize=10, fontweight="bold")
        ax2.set_xlabel("Model Size (B parameters)")
        ax2.set_ylabel("Throughput (tokens/sec)")
        ax2.set_title("Throughput Scaling")
        ax2.grid(True, alpha=0.3)

    fig.suptitle("Model Size Scaling — Latency & Throughput vs Parameters",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(cross_dir, "fig4_model_scaling.png"), dpi=200,
                bbox_inches="tight")
    plt.savefig(os.path.join(cross_dir, "fig4_model_scaling.pdf"), bbox_inches="tight")
    plt.close()
    logger.info("  Saved fig4_model_scaling")


# ---------------------------------------------------------------------------
# Experiment Summary
# ---------------------------------------------------------------------------

def generate_summary(all_model_results):
    """Generate human-readable experiment summary."""
    lines = []
    lines.append("=" * 70)
    lines.append("CS208 Multi-Model Autotune Experiment — Summary")
    lines.append("=" * 70)
    lines.append("")

    for model_name, data in all_model_results.items():
        short = model_name.split("/")[-1]
        lines.append(f"{'=' * 60}")
        lines.append(f"Model: {model_name}")
        lines.append(f"{'=' * 60}")
        lines.append("")

        # Profile summary
        profile = data.get("profile_data", [])
        if profile:
            lines.append("--- Phase A: GPU% Profiling ---")
            for entry in profile:
                lines.append(
                    f"  Concurrency {entry['concurrency']} "
                    f"(GPU% = {entry['gpu_pct']}%):"
                )
                lines.append(f"    E2E avg:     {entry['e2e']['avg_latency_ms']:.1f} ms")
                lines.append(f"    E2E p95:     {entry['e2e']['p95_latency_ms']:.1f} ms")
                lines.append(f"    Throughput:  {entry['avg_throughput_tps']:.1f} tps")
            lines.append("")

        # Knee analysis
        knee = data.get("knee_analysis", {})
        if knee:
            lines.append("--- Phase B: Knee Analysis ---")
            for stage in ["prefill", "decode"]:
                k = knee.get(stage)
                if k:
                    lines.append(
                        f"  {stage}: knee at GPU% = {k.get('gpu_pct', 'N/A')}, "
                        f"avg_latency = {k.get('avg_latency_ms', 0):.1f} ms"
                    )
            opt = knee.get("optimal_batches", {})
            if opt:
                lines.append(
                    f"  Optimal batches: prefill={opt.get('prefill_batch_size')}, "
                    f"decode={opt.get('decode_batch_size')}"
                )
            lines.append("")

        # Autotune comparison
        comp = data.get("comparison", {})
        if comp:
            lines.append("--- Phase C: Autotune OFF vs ON ---")
            off = comp.get("autotune_off", {})
            on = comp.get("autotune_on", {})
            for lbl, d in [("OFF", off), ("ON", on)]:
                lines.append(f"  Autotune {lbl}:")
                lines.append(f"    p95 latency: {d.get('e2e_p95_ms', 0):.1f} ms")
                lines.append(f"    Throughput:  {d.get('throughput_tps', 0):.1f} tps")
                lines.append(f"    GPU-s/req:   {d.get('gpu_seconds_per_request', 0):.3f}")
            imp = comp.get("improvement", {})
            for metric, label in [("e2e_p95_ms", "p95 latency"),
                                  ("throughput_tps", "throughput")]:
                m = imp.get(metric, {})
                if m:
                    lines.append(f"  {label}: {abs(m['change_pct']):.1f}% "
                                f"{'improvement' if m['change_pct'] > 0 else 'regression'}")
            lines.append("")

        # Ablation summary
        ablation = data.get("ablation", {})
        if ablation:
            lines.append("--- Phase D: Ablation ---")
            lines.append(
                f"  {'Config':<20} {'p95(ms)':>10} {'TPS':>10} {'GPU-s/req':>12}"
            )
            for name, s in ablation.items():
                lines.append(
                    f"  {name:<20} "
                    f"{s.get('e2e_p95_ms', 0):>10.1f} "
                    f"{s.get('throughput_tps', 0):>10.1f} "
                    f"{s.get('gpu_seconds_per_request', 0):>12.3f}"
                )
            lines.append("")

    lines.append("=" * 70)
    lines.append("Models tested: %d" % len(all_model_results))
    lines.append("=" * 70)

    summary_text = "\n".join(lines)

    summary_path = os.path.join(RESULTS_DIR, "experiment_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)

    print(summary_text)
    logger.info("Saved experiment summary to %s", summary_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TMP_CONFIG_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("CS208 Multi-Model Autotune Experiment — Starting")
    logger.info("Models: %s", ", ".join(MODELS))
    logger.info("=" * 60)

    log_gpu_info()

    t0 = time.time()
    svc = ServiceManager()

    # Ensure cleanup on unexpected exit
    import atexit
    atexit.register(svc.stop_all)

    all_model_results = {}

    for model_idx, model_name in enumerate(MODELS):
        logger.info("")
        logger.info("#" * 60)
        logger.info("# MODEL %d/%d: %s", model_idx + 1, len(MODELS), model_name)
        logger.info("#" * 60)
        logger.info("")

        # Check GPU memory
        if not check_gpu_memory(model_name):
            logger.warning("SKIPPING %s due to insufficient GPU memory", model_name)
            continue

        safe_name = model_name.replace("/", "_")
        model_results_dir = os.path.join(RESULTS_DIR, safe_name)
        os.makedirs(model_results_dir, exist_ok=True)

        model_t0 = time.time()
        model_data = {}

        try:
            # Phase A: Profile
            profile_data = run_phase_a(svc, model_name, model_results_dir)
            if not profile_data:
                logger.error("Phase A produced no data for %s, skipping", model_name)
                continue
            model_data["profile_data"] = profile_data

            # Phase B: Knee analysis (no services needed)
            knee_analysis, optimal_batches = run_phase_b(
                profile_data, model_results_dir
            )
            model_data["knee_analysis"] = knee_analysis

            # Phase C: Autotune comparison
            eval_results = run_phase_c(
                svc, model_name, optimal_batches, model_results_dir
            )
            # Load the saved comparison.json for cross-model analysis
            comp_path = os.path.join(model_results_dir, "comparison.json")
            with open(comp_path) as f:
                model_data["comparison"] = json.load(f)

            # Phase D: Ablation sweep
            ablation_results = run_phase_d(
                svc, model_name, optimal_batches, model_results_dir
            )
            model_data["ablation"] = ablation_results

            # Phase E: Per-model figures
            run_phase_e(model_name, model_results_dir)

            model_elapsed = time.time() - model_t0
            logger.info("Model %s complete in %.1f minutes",
                       model_name, model_elapsed / 60)

        except Exception as e:
            logger.exception("Error processing model %s: %s", model_name, e)
            svc.stop_all()  # Ensure cleanup
            continue

        all_model_results[model_name] = model_data

    # Cross-model figures
    if len(all_model_results) > 1:
        generate_cross_model_figures(all_model_results)

    # Summary
    generate_summary(all_model_results)

    elapsed = time.time() - t0
    logger.info("Total experiment time: %.1f minutes", elapsed / 60)
    logger.info("Results saved to %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
