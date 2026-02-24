"""Evaluation runner with baseline comparisons and ablation studies.

Experiment modes:
  single:     Run one workload experiment
  baselines:  Run static / elastic / elastic+autotune and compare
  burst:      Sweep burstiness levels (1x, 2x, 5x, 10x)
  ablation:   Toggle individual features to measure contribution

Produces comparison plots and JSON results for the paper.
"""
import os
import json
import logging
import time

from eval.metrics import LatencyCollector, ScalingStabilityAnalyzer
from eval.workloads import WorkloadGenerator, SAMPLE_PROMPTS
from eval.plots import (
    plot_latency_cdf, plot_latency_breakdown,
    plot_throughput_over_time, plot_scaling_stability,
    plot_ablation, save_results_json,
)

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


BASELINE_CONFIGS = {
    "static": {
        "description": "Fixed replicas (1 prefill, 1 decode), no autotune",
        "scaling_enabled": False,
        "autotune_enabled": False,
    },
    "elastic": {
        "description": "Elastic scaling enabled, no autotune",
        "scaling_enabled": True,
        "autotune_enabled": False,
    },
    "elastic_autotune": {
        "description": "Elastic scaling + serving-time autotune",
        "scaling_enabled": True,
        "autotune_enabled": True,
    },
}

# Ablation configurations: each disables one feature from the full system
ABLATION_CONFIGS = {
    "full_system": {
        "description": "All features enabled (elastic + autotune + MoE + anti-oscillation)",
        "scaling_enabled": True,
        "autotune_enabled": True,
        "moe_enabled": True,
        "anti_oscillation": True,
    },
    "no_autotune": {
        "description": "Ablation: disable autotune, keep elastic + MoE + anti-oscillation",
        "scaling_enabled": True,
        "autotune_enabled": False,
        "moe_enabled": True,
        "anti_oscillation": True,
    },
    "no_anti_oscillation": {
        "description": "Ablation: disable anti-oscillation, keep elastic + autotune + MoE",
        "scaling_enabled": True,
        "autotune_enabled": True,
        "moe_enabled": True,
        "anti_oscillation": False,
    },
    "no_moe": {
        "description": "Ablation: disable MoE simulation, keep elastic + autotune",
        "scaling_enabled": True,
        "autotune_enabled": True,
        "moe_enabled": False,
        "anti_oscillation": True,
    },
    "static_baseline": {
        "description": "Ablation: everything disabled (static baseline)",
        "scaling_enabled": False,
        "autotune_enabled": False,
        "moe_enabled": False,
        "anti_oscillation": False,
    },
}


def run_single_experiment(gateway_target, name, num_requests=100,
                          arrival_rate=10.0, max_tokens=128, burstiness=1.0,
                          gpu_count=1, gpu_pct=100,
                          output_dir="eval_results",
                          scaling_events=None):
    """Run a single workload experiment and save results."""
    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.join(output_dir, name)

    logger.info("=== Running experiment: %s ===", name)
    gen = WorkloadGenerator(
        gateway_target=gateway_target,
        num_requests=num_requests,
        arrival_rate=arrival_rate,
        max_tokens=max_tokens,
        burstiness=burstiness,
    )
    # Set GPU info for efficiency metrics
    gen.collector._gpu_count = gpu_count
    gen.collector._gpu_pct = gpu_pct

    summary = gen.run()
    records = gen.collector.records

    # Analyze scaling stability if events provided
    if scaling_events:
        stability = ScalingStabilityAnalyzer(scaling_events).analyze()
        summary["scaling_stability"] = stability
        plot_scaling_stability(stability, f"{prefix}_scaling_stability.png")
        logger.info("Scaling stability: %d events, osc_freq=%.2f/min, "
                     "avg_convergence=%.1fs, dcr=%.3f",
                     stability["total_scaling_events"],
                     stability["oscillation_frequency_per_min"],
                     stability["avg_convergence_time_s"],
                     stability["direction_change_ratio"])

    # Save raw results
    save_results_json(summary, records, f"{prefix}_results.json")

    # Generate plots
    plot_latency_cdf(records, f"{prefix}_cdf.png")
    plot_latency_breakdown(summary, f"{prefix}_breakdown.png")
    plot_throughput_over_time(records, output_path=f"{prefix}_throughput.png")

    logger.info("Experiment %s complete: %d reqs, p50=%.1fms p95=%.1fms p99=%.1fms, "
                "tps=%.1f, gpu_s/req=%.3f, gpu_util=%.1f%%",
                name, summary.get("num_requests", 0),
                summary.get("e2e_p50_ms", 0),
                summary.get("e2e_p95_ms", 0),
                summary.get("e2e_p99_ms", 0),
                summary.get("throughput_tps", 0),
                summary.get("gpu_seconds_per_request", 0),
                summary.get("gpu_utilization_pct", 0))
    return summary


def run_all_baselines(gateway_target, num_requests=100, arrival_rate=10.0,
                      max_tokens=128, gpu_count=1, gpu_pct=100,
                      output_dir="eval_results"):
    """Run all baseline configurations and produce comparison results."""
    all_results = {}
    for name, cfg in BASELINE_CONFIGS.items():
        logger.info("--- Baseline: %s - %s ---", name, cfg["description"])
        summary = run_single_experiment(
            gateway_target=gateway_target,
            name=name,
            num_requests=num_requests,
            arrival_rate=arrival_rate,
            max_tokens=max_tokens,
            gpu_count=gpu_count,
            gpu_pct=gpu_pct,
            output_dir=output_dir,
        )
        all_results[name] = summary

    # Save combined comparison
    comparison_path = os.path.join(output_dir, "comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved comparison results to %s", comparison_path)

    # Generate comparison plots
    if HAS_MPL and all_results:
        _plot_comparison(all_results, output_dir)

    return all_results


def run_burstiness_sweep(gateway_target, num_requests=100, arrival_rate=10.0,
                         max_tokens=128, gpu_count=1, gpu_pct=100,
                         output_dir="eval_results"):
    """Run experiments with varying burstiness to test scaling responsiveness."""
    results = {}
    for burst in [1.0, 2.0, 5.0, 10.0]:
        name = f"burst_{burst:.0f}x"
        summary = run_single_experiment(
            gateway_target=gateway_target,
            name=name,
            num_requests=num_requests,
            arrival_rate=arrival_rate,
            max_tokens=max_tokens,
            burstiness=burst,
            gpu_count=gpu_count,
            gpu_pct=gpu_pct,
            output_dir=output_dir,
        )
        results[name] = summary

    # Save sweep comparison
    sweep_path = os.path.join(output_dir, "burst_sweep.json")
    with open(sweep_path, "w") as f:
        json.dump(results, f, indent=2)

    if HAS_MPL and results:
        _plot_burst_sweep(results, output_dir)

    return results


def run_ablation(gateway_target, num_requests=100, arrival_rate=10.0,
                 max_tokens=128, gpu_count=1, gpu_pct=100,
                 output_dir="eval_results"):
    """Run ablation study: toggle features one-by-one to measure contribution.

    Each experiment disables one feature from the full system to show
    the marginal contribution of that feature.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for name, cfg in ABLATION_CONFIGS.items():
        logger.info("--- Ablation: %s - %s ---", name, cfg["description"])
        summary = run_single_experiment(
            gateway_target=gateway_target,
            name=f"ablation_{name}",
            num_requests=num_requests,
            arrival_rate=arrival_rate,
            max_tokens=max_tokens,
            gpu_count=gpu_count,
            gpu_pct=gpu_pct,
            output_dir=output_dir,
        )
        results[name] = summary

    # Save ablation results
    ablation_path = os.path.join(output_dir, "ablation.json")
    with open(ablation_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved ablation results to %s", ablation_path)

    # Generate ablation plot
    plot_ablation(results, os.path.join(output_dir, "ablation.png"))

    # Print ablation summary table
    logger.info("=== Ablation Summary ===")
    logger.info("%-25s %10s %10s %10s %12s", "Config", "p95(ms)", "p99(ms)", "TPS", "GPU-s/req")
    for name, s in results.items():
        logger.info("%-25s %10.1f %10.1f %10.1f %12.3f",
                     name,
                     s.get("e2e_p95_ms", 0),
                     s.get("e2e_p99_ms", 0),
                     s.get("throughput_tps", 0),
                     s.get("gpu_seconds_per_request", 0))

    return results


def _plot_comparison(all_results, output_dir):
    """Generate comparison bar charts across baselines."""
    names = list(all_results.keys())
    p50s = [all_results[n].get("e2e_p50_ms", 0) for n in names]
    p95s = [all_results[n].get("e2e_p95_ms", 0) for n in names]
    p99s = [all_results[n].get("e2e_p99_ms", 0) for n in names]
    tps = [all_results[n].get("throughput_tps", 0) for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Latency comparison
    x = range(len(names))
    width = 0.25
    axes[0].bar([i - width for i in x], p50s, width, label="p50", color="#4C72B0")
    axes[0].bar(list(x), p95s, width, label="p95", color="#DD8452")
    axes[0].bar([i + width for i in x], p99s, width, label="p99", color="#C44E52")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(names, rotation=15)
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("End-to-End Latency")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # Throughput comparison
    axes[1].bar(names, tps, color="#55A868")
    axes[1].set_ylabel("Tokens/sec")
    axes[1].set_title("Throughput")
    axes[1].set_xticklabels(names, rotation=15)
    axes[1].grid(True, alpha=0.3, axis="y")

    # GPU efficiency
    gpu_eff = [all_results[n].get("gpu_seconds_per_request", 0) for n in names]
    axes[2].bar(names, gpu_eff, color="#8172B2")
    axes[2].set_ylabel("GPU-seconds/request")
    axes[2].set_title("GPU Efficiency")
    axes[2].set_xticklabels(names, rotation=15)
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison.png"), dpi=150)
    plt.close()
    logger.info("Saved comparison plot to %s", os.path.join(output_dir, "comparison.png"))

    # TTFT comparison if available
    ttft_p50 = [all_results[n].get("ttft_p50_ms", 0) for n in names]
    ttft_p95 = [all_results[n].get("ttft_p95_ms", 0) for n in names]
    if any(t > 0 for t in ttft_p50):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar([i - 0.15 for i in x], ttft_p50, 0.3, label="p50", color="#4C72B0")
        ax.bar([i + 0.15 for i in x], ttft_p95, 0.3, label="p95", color="#DD8452")
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=15)
        ax.set_ylabel("TTFT (ms)")
        ax.set_title("Time to First Token Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ttft_comparison.png"), dpi=150)
        plt.close()


def _plot_burst_sweep(results, output_dir):
    """Plot how metrics change with burstiness level."""
    names = list(results.keys())
    bursts = [float(n.replace("burst_", "").replace("x", "")) for n in names]
    p95s = [results[n].get("e2e_p95_ms", 0) for n in names]
    tps = [results[n].get("throughput_tps", 0) for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(bursts, p95s, "o-", linewidth=2, color="#C44E52")
    ax1.set_xlabel("Burstiness Factor")
    ax1.set_ylabel("p95 Latency (ms)")
    ax1.set_title("Latency vs Burstiness")
    ax1.grid(True, alpha=0.3)

    ax2.plot(bursts, tps, "s-", linewidth=2, color="#55A868")
    ax2.set_xlabel("Burstiness Factor")
    ax2.set_ylabel("Tokens/sec")
    ax2.set_title("Throughput vs Burstiness")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "burst_sweep.png"), dpi=150)
    plt.close()
    logger.info("Saved burst sweep plot to %s", os.path.join(output_dir, "burst_sweep.png"))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run evaluation experiments")
    parser.add_argument("--gateway", default="localhost:50051", help="Gateway gRPC target")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests")
    parser.add_argument("--rate", type=float, default=10.0, help="Arrival rate (rps)")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens per request")
    parser.add_argument("--output-dir", default="eval_results", help="Output directory")
    parser.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs in cluster")
    parser.add_argument("--gpu-pct", type=int, default=100, help="GPU% allocation per pod")
    parser.add_argument("--mode", choices=["single", "baselines", "burst", "ablation"],
                        default="single", help="Experiment mode")
    parser.add_argument("--name", default="experiment", help="Experiment name (single mode)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.mode == "baselines":
        run_all_baselines(args.gateway, args.requests, args.rate,
                          args.max_tokens, args.gpu_count, args.gpu_pct,
                          args.output_dir)
    elif args.mode == "burst":
        run_burstiness_sweep(args.gateway, args.requests, args.rate,
                             args.max_tokens, args.gpu_count, args.gpu_pct,
                             args.output_dir)
    elif args.mode == "ablation":
        run_ablation(args.gateway, args.requests, args.rate,
                     args.max_tokens, args.gpu_count, args.gpu_pct,
                     args.output_dir)
    else:
        run_single_experiment(args.gateway, args.name, args.requests,
                              args.rate, args.max_tokens,
                              gpu_count=args.gpu_count, gpu_pct=args.gpu_pct,
                              output_dir=args.output_dir)


if __name__ == "__main__":
    main()
