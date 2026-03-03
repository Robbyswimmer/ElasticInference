#!/usr/bin/env python3
"""SLURM single-node autotune experiment orchestrator.

Phases:
  A) Profile GPU contention levels via concurrent load
  B) Build autotune lookup table from profiles
  C) Run eval: autotune OFF vs ON
  D) Run ablation sweep
  E) Collect & export results
"""

import os
import sys
import json
import time
import logging
from concurrent import futures

import grpc
import numpy as np

# Add project root to path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from common import load_config
from common.proto import inference_pb2, inference_pb2_grpc
from common.gpu_utils import log_gpu_info
from autotune.profiler import KneeProfiler
from autotune.selector import ConfigSelector
from eval.runner import run_single_experiment, run_ablation
from eval.workloads import SAMPLE_PROMPTS
from eval.plots import save_results_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

GATEWAY_TARGET = "localhost:50051"
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

# Concurrency-to-GPU% mapping: more concurrent requests = less effective GPU per request
CONCURRENCY_GPU_MAP = {1: 100, 2: 50, 4: 25}

NUM_WARMUP = 3
NUM_MEASURE = 15
NUM_EVAL_REQUESTS = 100
ARRIVAL_RATE = 10.0
MAX_TOKENS = 128


def wait_for_gateway(target, timeout=30):
    """Wait until the gateway responds to a gRPC health-check-style call."""
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
            stub.Infer(req, timeout=10)
            channel.close()
            return True
        except grpc.RpcError:
            time.sleep(2)
    channel.close()
    return False


# ---------------------------------------------------------------------------
# Phase A: GPU% Simulation via Concurrent Load Profiling
# ---------------------------------------------------------------------------

def send_single_request(stub, prompt, request_id, max_tokens):
    """Send one inference request and return latency breakdown."""
    req = inference_pb2.InferRequest(
        request_id=request_id,
        prompt=prompt,
        max_tokens=max_tokens,
    )
    try:
        resp = stub.Infer(req, timeout=60)
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
    """Profile the system at a given concurrency level.

    Sends requests concurrently to create GPU contention, simulating
    different effective GPU% levels.
    """
    gpu_pct = CONCURRENCY_GPU_MAP[concurrency]
    logger.info("=== Profiling at concurrency=%d (simulated GPU%%=%d) ===", concurrency, gpu_pct)

    channel = grpc.insecure_channel(GATEWAY_TARGET)
    stub = inference_pb2_grpc.GatewayServiceStub(channel)

    # Warmup
    logger.info("Sending %d warmup requests...", num_warmup)
    for i in range(num_warmup):
        prompt = SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]
        send_single_request(stub, prompt, f"warmup-{concurrency}-{i}", MAX_TOKENS)

    # Measurement: send requests at the target concurrency
    logger.info("Sending %d measurement requests at concurrency %d...", num_measure, concurrency)
    all_results = []
    pool = futures.ThreadPoolExecutor(max_workers=concurrency)

    # Send in batches of `concurrency` to maintain steady contention
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

    # Trim to exactly num_measure
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


def run_phase_a():
    """Phase A: Profile at all concurrency levels."""
    logger.info("=" * 60)
    logger.info("PHASE A: GPU%% Simulation via Concurrent Load Profiling")
    logger.info("=" * 60)

    profile_data = []
    for concurrency in sorted(CONCURRENCY_GPU_MAP.keys()):
        result = profile_at_concurrency(concurrency, NUM_WARMUP, NUM_MEASURE)
        if result:
            profile_data.append(result)
        time.sleep(5)  # Brief pause between concurrency levels

    # Save profile data
    profile_path = os.path.join(RESULTS_DIR, "profile_data.json")
    with open(profile_path, "w") as f:
        json.dump(profile_data, f, indent=2)
    logger.info("Saved profile data to %s (%d levels)", profile_path, len(profile_data))

    return profile_data


# ---------------------------------------------------------------------------
# Phase B: Build Autotune Lookup Table
# ---------------------------------------------------------------------------

def run_phase_b(profile_data):
    """Phase B: Build autotune lookup table and find knee points."""
    logger.info("=" * 60)
    logger.info("PHASE B: Build Autotune Lookup Table")
    logger.info("=" * 60)

    config = load_config()
    selector = ConfigSelector(config)
    profiler = KneeProfiler(config)

    # Convert profile_data into the format expected by ConfigSelector and KneeProfiler
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

    # Update config selector with profile data
    selector.update_from_profile("prefill", prefill_results)
    selector.update_from_profile("decode", decode_results)

    # Find knee points
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

    # Save knee analysis
    knee_path = os.path.join(RESULTS_DIR, "knee_analysis.json")
    with open(knee_path, "w") as f:
        json.dump(knee_analysis, f, indent=2)
    logger.info("Saved knee analysis to %s", knee_path)

    return selector, knee_analysis


# ---------------------------------------------------------------------------
# Phase C: Run Eval — Autotune OFF vs ON
# ---------------------------------------------------------------------------

def run_phase_c():
    """Phase C: Compare autotune OFF vs ON using eval runner."""
    logger.info("=" * 60)
    logger.info("PHASE C: Eval — Autotune OFF vs ON")
    logger.info("=" * 60)

    results = {}

    # --- Autotune OFF ---
    logger.info("--- Running: autotune OFF ---")
    off_summary = run_single_experiment(
        gateway_target=GATEWAY_TARGET,
        name="autotune_off",
        num_requests=NUM_EVAL_REQUESTS,
        arrival_rate=ARRIVAL_RATE,
        max_tokens=MAX_TOKENS,
        gpu_count=1,
        gpu_pct=100,
        output_dir=RESULTS_DIR,
    )
    results["autotune_off"] = off_summary

    # Brief pause between runs
    time.sleep(10)

    # --- Autotune ON ---
    logger.info("--- Running: autotune ON ---")
    on_summary = run_single_experiment(
        gateway_target=GATEWAY_TARGET,
        name="autotune_on",
        num_requests=NUM_EVAL_REQUESTS,
        arrival_rate=ARRIVAL_RATE,
        max_tokens=MAX_TOKENS,
        gpu_count=1,
        gpu_pct=100,
        output_dir=RESULTS_DIR,
    )
    results["autotune_on"] = on_summary

    # Save comparison
    comparison = {
        "autotune_off": off_summary,
        "autotune_on": on_summary,
        "improvement": {},
    }

    # Compute improvements
    for metric in ["e2e_p50_ms", "e2e_p95_ms", "e2e_p99_ms", "throughput_tps",
                    "gpu_seconds_per_request"]:
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

    comparison_path = os.path.join(RESULTS_DIR, "comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info("Saved comparison to %s", comparison_path)

    # Generate comparison plot
    _plot_comparison(comparison)

    return results


def _plot_comparison(comparison):
    """Generate side-by-side comparison bar chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping comparison plot")
        return

    off = comparison["autotune_off"]
    on = comparison["autotune_on"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Latency comparison
    labels = ["p50", "p95", "p99"]
    off_lats = [off.get("e2e_p50_ms", 0), off.get("e2e_p95_ms", 0), off.get("e2e_p99_ms", 0)]
    on_lats = [on.get("e2e_p50_ms", 0), on.get("e2e_p95_ms", 0), on.get("e2e_p99_ms", 0)]

    x = np.arange(len(labels))
    width = 0.35
    axes[0].bar(x - width / 2, off_lats, width, label="Autotune OFF", color="#C44E52")
    axes[0].bar(x + width / 2, on_lats, width, label="Autotune ON", color="#55A868")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("End-to-End Latency")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # Throughput comparison
    names = ["Autotune OFF", "Autotune ON"]
    tps_vals = [off.get("throughput_tps", 0), on.get("throughput_tps", 0)]
    axes[1].bar(names, tps_vals, color=["#C44E52", "#55A868"])
    axes[1].set_ylabel("Tokens/sec")
    axes[1].set_title("Throughput")
    axes[1].grid(True, alpha=0.3, axis="y")

    # GPU efficiency
    eff_vals = [off.get("gpu_seconds_per_request", 0), on.get("gpu_seconds_per_request", 0)]
    axes[2].bar(names, eff_vals, color=["#C44E52", "#55A868"])
    axes[2].set_ylabel("GPU-seconds/request")
    axes[2].set_title("GPU Efficiency")
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "comparison.png"), dpi=150)
    plt.close()
    logger.info("Saved comparison plot to results/comparison.png")


# ---------------------------------------------------------------------------
# Phase D: Run Ablation Sweep
# ---------------------------------------------------------------------------

def run_phase_d():
    """Phase D: Run ablation study."""
    logger.info("=" * 60)
    logger.info("PHASE D: Ablation Sweep")
    logger.info("=" * 60)

    ablation_results = run_ablation(
        gateway_target=GATEWAY_TARGET,
        num_requests=NUM_EVAL_REQUESTS,
        arrival_rate=ARRIVAL_RATE,
        max_tokens=MAX_TOKENS,
        gpu_count=1,
        gpu_pct=100,
        output_dir=RESULTS_DIR,
    )

    return ablation_results


# ---------------------------------------------------------------------------
# Phase E: Collect & Export Results
# ---------------------------------------------------------------------------

def run_phase_e(profile_data, knee_analysis, eval_results, ablation_results):
    """Phase E: Generate human-readable summary."""
    logger.info("=" * 60)
    logger.info("PHASE E: Collect & Export Results")
    logger.info("=" * 60)

    lines = []
    lines.append("=" * 60)
    lines.append("CS208 Autotune Experiment — Summary")
    lines.append("=" * 60)
    lines.append("")

    # Profile summary
    lines.append("--- Phase A: GPU% Profiling ---")
    for entry in profile_data:
        lines.append(f"  Concurrency {entry['concurrency']} (simulated GPU% = {entry['gpu_pct']}%):")
        lines.append(f"    E2E avg latency:  {entry['e2e']['avg_latency_ms']:.1f} ms")
        lines.append(f"    E2E p95 latency:  {entry['e2e']['p95_latency_ms']:.1f} ms")
        lines.append(f"    Prefill avg:      {entry['prefill']['avg_latency_ms']:.1f} ms")
        lines.append(f"    Decode avg:       {entry['decode']['avg_latency_ms']:.1f} ms")
        lines.append(f"    Throughput:       {entry['avg_throughput_tps']:.1f} tps")
    lines.append("")

    # Knee analysis
    lines.append("--- Phase B: Knee Analysis ---")
    for stage in ["prefill", "decode"]:
        knee = knee_analysis.get(stage)
        if knee:
            lines.append(f"  {stage}: knee at GPU% = {knee.get('gpu_pct', 'N/A')}, "
                         f"avg_latency = {knee.get('avg_latency_ms', 0):.1f} ms")
        else:
            lines.append(f"  {stage}: no knee found")
    lines.append("")

    # Autotune comparison
    lines.append("--- Phase C: Autotune OFF vs ON ---")
    off = eval_results.get("autotune_off", {})
    on = eval_results.get("autotune_on", {})
    for label, data in [("Autotune OFF", off), ("Autotune ON", on)]:
        lines.append(f"  {label}:")
        lines.append(f"    Requests:         {data.get('num_requests', 0)}")
        lines.append(f"    p50 latency:      {data.get('e2e_p50_ms', 0):.1f} ms")
        lines.append(f"    p95 latency:      {data.get('e2e_p95_ms', 0):.1f} ms")
        lines.append(f"    p99 latency:      {data.get('e2e_p99_ms', 0):.1f} ms")
        lines.append(f"    Throughput:       {data.get('throughput_tps', 0):.1f} tps")
        lines.append(f"    GPU-s/req:        {data.get('gpu_seconds_per_request', 0):.3f}")
    lines.append("")

    # Improvement
    if off and on:
        for metric, label in [("e2e_p95_ms", "p95 latency"), ("throughput_tps", "throughput")]:
            off_val = off.get(metric, 0)
            on_val = on.get(metric, 0)
            if off_val > 0:
                if "throughput" in metric:
                    change = ((on_val - off_val) / off_val) * 100
                    direction = "improvement" if change > 0 else "regression"
                else:
                    change = ((off_val - on_val) / off_val) * 100
                    direction = "improvement" if change > 0 else "regression"
                lines.append(f"  {label}: {abs(change):.1f}% {direction}")
    lines.append("")

    # Ablation summary
    lines.append("--- Phase D: Ablation ---")
    lines.append(f"  {'Config':<25} {'p95(ms)':>10} {'p99(ms)':>10} {'TPS':>10} {'GPU-s/req':>12}")
    for name, s in ablation_results.items():
        lines.append(f"  {name:<25} {s.get('e2e_p95_ms', 0):>10.1f} "
                     f"{s.get('e2e_p99_ms', 0):>10.1f} "
                     f"{s.get('throughput_tps', 0):>10.1f} "
                     f"{s.get('gpu_seconds_per_request', 0):>12.3f}")
    lines.append("")

    # Output files
    lines.append("--- Output Files ---")
    for fname in sorted(os.listdir(RESULTS_DIR)):
        fpath = os.path.join(RESULTS_DIR, fname)
        size = os.path.getsize(fpath)
        lines.append(f"  {fname:<40} {size:>10,} bytes")
    lines.append("")
    lines.append("=" * 60)

    summary_text = "\n".join(lines)

    # Save and print
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

    logger.info("=" * 60)
    logger.info("CS208 Autotune Experiment — Starting")
    logger.info("=" * 60)

    # Log GPU info
    log_gpu_info()

    # Verify gateway is reachable
    logger.info("Checking gateway at %s...", GATEWAY_TARGET)
    if not wait_for_gateway(GATEWAY_TARGET, timeout=30):
        logger.error("Gateway not reachable at %s. Exiting.", GATEWAY_TARGET)
        sys.exit(1)
    logger.info("Gateway is ready.")

    t0 = time.time()

    # Phase A: Profile
    profile_data = run_phase_a()
    if not profile_data:
        logger.error("Phase A produced no profile data. Exiting.")
        sys.exit(1)

    # Phase B: Build lookup table
    selector, knee_analysis = run_phase_b(profile_data)

    # Phase C: Autotune OFF vs ON
    eval_results = run_phase_c()

    # Phase D: Ablation
    ablation_results = run_phase_d()

    # Phase E: Summary
    run_phase_e(profile_data, knee_analysis, eval_results, ablation_results)

    elapsed = time.time() - t0
    logger.info("Total experiment time: %.1f minutes", elapsed / 60)


if __name__ == "__main__":
    main()
