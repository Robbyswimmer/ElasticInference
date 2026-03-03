#!/usr/bin/env python3
"""Generate publication-quality figures from multi-model experiment results.

Produces per-model figures (profiling, autotune comparison, ablation)
and 4 cross-model comparison figures for the paper:
  1. Profiling comparison — latency at each GPU% across all models
  2. Autotune improvement — % improvement (p95, throughput, GPU cost) per model
  3. Ablation comparison — full_system vs static_baseline across models
  4. Model scaling — latency/throughput vs model size (1.3B -> 6.7B)

Usage:
    python scripts/generate_multimodel_figures.py [--results-dir results/]
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

# Style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Color palette
C_BLUE = "#2E86AB"
C_RED = "#E84855"
C_GREEN = "#44BBA4"
C_ORANGE = "#F5A623"
C_PURPLE = "#7B68EE"
C_GRAY = "#8B8B8B"

# Model colors (consistent across figures)
MODEL_COLORS = [C_BLUE, C_RED, C_GREEN, C_ORANGE]

# Model sizes in billions of parameters
MODEL_SIZES = {
    "opt-1.3b": 1.3,
    "Qwen2-1.5B": 1.5,
    "phi-2": 2.7,
    "opt-6.7b": 6.7,
}

# Expected model directory names (as produced by run_multimodel_experiment.py)
MODEL_DIRS = [
    "facebook_opt-1.3b",
    "Qwen_Qwen2-1.5B",
    "microsoft_phi-2",
    "facebook_opt-6.7b",
]

MODEL_DISPLAY_NAMES = {
    "facebook_opt-1.3b": "OPT-1.3B",
    "Qwen_Qwen2-1.5B": "Qwen2-1.5B",
    "microsoft_phi-2": "Phi-2 (2.7B)",
    "facebook_opt-6.7b": "OPT-6.7B",
}


def load_json(path):
    """Load JSON file, return None if missing."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"  Warning: {path} not found, skipping")
        return None


def discover_models(results_dir):
    """Find which model directories exist and load their data."""
    models = {}
    for model_dir_name in MODEL_DIRS:
        model_path = os.path.join(results_dir, model_dir_name)
        if not os.path.isdir(model_path):
            continue

        data = {}
        data["profile"] = load_json(os.path.join(model_path, "profile_data.json"))
        data["comparison"] = load_json(os.path.join(model_path, "comparison.json"))
        data["ablation"] = load_json(os.path.join(model_path, "ablation.json"))
        data["knee"] = load_json(os.path.join(model_path, "knee_analysis.json"))

        # Only include if we have at least some data
        if any(v is not None for v in data.values()):
            data["display_name"] = MODEL_DISPLAY_NAMES.get(model_dir_name, model_dir_name)
            data["dir_name"] = model_dir_name
            models[model_dir_name] = data

    return models


# =========================================================================
# Per-Model Figures
# =========================================================================

def gen_per_model_figures(model_data, model_dir, figures_dir):
    """Generate per-model figures (3 figures per model)."""
    os.makedirs(figures_dir, exist_ok=True)
    display = model_data["display_name"]

    # --- Profiling ---
    profile = model_data.get("profile")
    if profile:
        gpu_pcts = [d["gpu_pct"] for d in profile]
        prefill_avg = [d["prefill"]["avg_latency_ms"] for d in profile]
        decode_avg = [d["decode"]["avg_latency_ms"] for d in profile]
        e2e_p95 = [d["e2e"]["p95_latency_ms"] for d in profile]
        throughputs = [d["avg_throughput_tps"] for d in profile]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Prefill latency
        axes[0].plot(gpu_pcts, prefill_avg, "o-", color=C_BLUE, linewidth=2.5,
                     markersize=10)
        axes[0].fill_between(gpu_pcts, 0, prefill_avg, alpha=0.1, color=C_BLUE)
        axes[0].set_xlabel("Simulated GPU %")
        axes[0].set_ylabel("Prefill Latency (ms)")
        axes[0].set_title("Prefill Stage")
        axes[0].set_xticks(gpu_pcts)
        axes[0].invert_xaxis()

        # Decode latency
        axes[1].plot(gpu_pcts, decode_avg, "s-", color=C_ORANGE, linewidth=2.5,
                     markersize=10, label="Avg")
        axes[1].plot(gpu_pcts, e2e_p95, "^--", color=C_RED, linewidth=1.5,
                     markersize=8, label="E2E p95", alpha=0.8)
        axes[1].set_xlabel("Simulated GPU %")
        axes[1].set_ylabel("Latency (ms)")
        axes[1].set_title("Decode Stage")
        axes[1].set_xticks(gpu_pcts)
        axes[1].invert_xaxis()
        axes[1].legend()

        # Throughput
        bar_colors = [C_GREEN if pct >= 75 else C_ORANGE if pct >= 50 else C_RED
                      for pct in gpu_pcts]
        bars = axes[2].bar(gpu_pcts, throughputs, width=15, color=bar_colors,
                           edgecolor="white", linewidth=1.5)
        for bar, val in zip(bars, throughputs):
            axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         f"{val:.0f}", ha="center", fontweight="bold", fontsize=11)
        axes[2].set_xlabel("Simulated GPU %")
        axes[2].set_ylabel("Throughput (tokens/sec)")
        axes[2].set_title("Throughput")
        axes[2].set_xticks(gpu_pcts)

        fig.suptitle(f"GPU Contention Profiling — {display}",
                     fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        for ext in ["png", "pdf"]:
            plt.savefig(os.path.join(figures_dir, f"profiling.{ext}"))
        plt.close()
        print(f"  Saved profiling figure for {display}")

    # --- Autotune comparison ---
    comparison = model_data.get("comparison")
    if comparison:
        off = comparison.get("autotune_off", {})
        on = comparison.get("autotune_on", {})

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

        # Latency percentiles
        labels = ["p50", "p95", "p99"]
        off_vals = [off.get("e2e_p50_ms", 0), off.get("e2e_p95_ms", 0),
                    off.get("e2e_p99_ms", 0)]
        on_vals = [on.get("e2e_p50_ms", 0), on.get("e2e_p95_ms", 0),
                   on.get("e2e_p99_ms", 0)]

        x = np.arange(len(labels))
        width = 0.35
        ax1.bar(x - width / 2, off_vals, width, label="Autotune OFF",
                color=C_RED, edgecolor="white")
        ax1.bar(x + width / 2, on_vals, width, label="Autotune ON",
                color=C_GREEN, edgecolor="white")
        for i in range(len(labels)):
            if off_vals[i] > 0:
                pct = (off_vals[i] - on_vals[i]) / off_vals[i] * 100
                ax1.annotate(f"-{pct:.1f}%",
                            xy=(x[i], max(off_vals[i], on_vals[i]) + 200),
                            ha="center", fontsize=10, fontweight="bold", color=C_BLUE)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.set_ylabel("Latency (ms)")
        ax1.set_title("Latency Percentiles")
        ax1.legend()

        # Throughput
        cats = ["OFF", "ON"]
        tps = [off.get("throughput_tps", 0), on.get("throughput_tps", 0)]
        ax2.bar(cats, tps, color=[C_RED, C_GREEN], edgecolor="white", width=0.6)
        for i, val in enumerate(tps):
            ax2.text(i, val + 1, f"{val:.1f}", ha="center", fontweight="bold")
        ax2.set_ylabel("Tokens/sec")
        ax2.set_title("Throughput")

        # GPU efficiency
        gpu_s = [off.get("gpu_seconds_per_request", 0),
                 on.get("gpu_seconds_per_request", 0)]
        ax3.bar(cats, gpu_s, color=[C_RED, C_GREEN], edgecolor="white", width=0.6)
        for i, val in enumerate(gpu_s):
            ax3.text(i, val + 0.05, f"{val:.2f}", ha="center", fontweight="bold")
        ax3.set_ylabel("GPU-seconds / request")
        ax3.set_title("GPU Cost per Request")

        imp = comparison.get("improvement", {})
        p95_imp = imp.get("e2e_p95_ms", {}).get("change_pct", 0)
        fig.suptitle(f"Autotune Comparison — {display} ({p95_imp:.1f}% p95 reduction)",
                     fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        for ext in ["png", "pdf"]:
            plt.savefig(os.path.join(figures_dir, f"autotune_comparison.{ext}"))
        plt.close()
        print(f"  Saved autotune comparison for {display}")

    # --- Ablation ---
    ablation = model_data.get("ablation")
    if ablation:
        configs = list(ablation.keys())
        p95s = [ablation[c].get("e2e_p95_ms", 0) for c in configs]
        tps = [ablation[c].get("throughput_tps", 0) for c in configs]
        gpu_eff = [ablation[c].get("gpu_seconds_per_request", 0) for c in configs]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        short_labels = [c.replace("_", "\n") for c in configs]
        x = np.arange(len(configs))

        # p95 Latency
        colors_lat = [C_GREEN if v == min(p95s) else C_RED if v == max(p95s)
                      else C_ORANGE for v in p95s]
        bars = axes[0].bar(x, p95s, color=colors_lat, edgecolor="white", linewidth=1.5)
        for bar, val in zip(bars, p95s):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                         f"{val:.0f}", ha="center", fontsize=9, fontweight="bold")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(short_labels, fontsize=8)
        axes[0].set_ylabel("p95 Latency (ms)")
        axes[0].set_title("Tail Latency (lower = better)")

        # Throughput
        bars = axes[1].bar(x, tps, color=C_BLUE, edgecolor="white", linewidth=1.5)
        for bar, val in zip(bars, tps):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         f"{val:.0f}", ha="center", fontsize=9, fontweight="bold")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(short_labels, fontsize=8)
        axes[1].set_ylabel("Tokens/sec")
        axes[1].set_title("Throughput")

        # GPU Cost
        colors_gpu = [C_GREEN if v == min(gpu_eff) else C_RED if v == max(gpu_eff)
                      else C_ORANGE for v in gpu_eff]
        bars = axes[2].bar(x, gpu_eff, color=colors_gpu, edgecolor="white", linewidth=1.5)
        for bar, val in zip(bars, gpu_eff):
            axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                         f"{val:.1f}", ha="center", fontsize=9, fontweight="bold")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(short_labels, fontsize=8)
        axes[2].set_ylabel("GPU-seconds / request")
        axes[2].set_title("GPU Cost (lower = better)")

        fig.suptitle(f"Ablation Study — {display}",
                     fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        for ext in ["png", "pdf"]:
            plt.savefig(os.path.join(figures_dir, f"ablation.{ext}"))
        plt.close()
        print(f"  Saved ablation figure for {display}")


# =========================================================================
# Cross-Model Figures (4)
# =========================================================================

def fig1_profiling_comparison(models, cross_dir):
    """Figure 1: Latency at each GPU% across all models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for i, (dir_name, data) in enumerate(models.items()):
        profile = data.get("profile")
        if not profile:
            continue
        label = data["display_name"]
        color = MODEL_COLORS[i % len(MODEL_COLORS)]

        gpu_pcts = [d["gpu_pct"] for d in profile]
        decode_lats = [d["decode"]["avg_latency_ms"] for d in profile]
        throughputs = [d["avg_throughput_tps"] for d in profile]

        ax1.plot(gpu_pcts, decode_lats, "o-", label=label, color=color,
                linewidth=2.5, markersize=10)
        ax2.plot(gpu_pcts, throughputs, "s-", label=label, color=color,
                linewidth=2.5, markersize=10)

    ax1.set_xlabel("Simulated GPU %")
    ax1.set_ylabel("Avg Decode Latency (ms)")
    ax1.set_title("Decode Latency Under GPU Contention")
    ax1.legend(loc="upper right")
    ax1.invert_xaxis()

    ax2.set_xlabel("Simulated GPU %")
    ax2.set_ylabel("Throughput (tokens/sec)")
    ax2.set_title("Throughput Under GPU Contention")
    ax2.legend(loc="lower right")
    ax2.invert_xaxis()

    fig.suptitle("Cross-Model Profiling — Contention Impact by Model Size",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(os.path.join(cross_dir, f"fig1_profiling_comparison.{ext}"))
    plt.close()
    print("  Saved fig1_profiling_comparison")


def fig2_autotune_improvement(models, cross_dir):
    """Figure 2: % improvement (p95, throughput, GPU cost) per model."""
    labels = []
    imp_p95 = []
    imp_tps = []
    imp_gpu = []

    for dir_name, data in models.items():
        comp = data.get("comparison")
        if not comp:
            continue
        labels.append(data["display_name"])
        imp = comp.get("improvement", {})
        imp_p95.append(imp.get("e2e_p95_ms", {}).get("change_pct", 0))
        imp_tps.append(imp.get("throughput_tps", {}).get("change_pct", 0))
        imp_gpu.append(imp.get("gpu_seconds_per_request", {}).get("change_pct", 0))

    if not labels:
        print("  Skipping fig2: no comparison data")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    x = np.arange(len(labels))
    colors = MODEL_COLORS[:len(labels)]

    # p95 latency reduction
    bars = ax1.bar(x, imp_p95, color=colors, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, imp_p95):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f"{val:.1f}%", ha="center", fontweight="bold", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15)
    ax1.set_ylabel("Improvement (%)")
    ax1.set_title("p95 Latency Reduction")
    ax1.axhline(y=0, color="black", linewidth=0.5)

    # Throughput improvement
    bars = ax2.bar(x, imp_tps, color=colors, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, imp_tps):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f"{val:.1f}%", ha="center", fontweight="bold", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15)
    ax2.set_ylabel("Improvement (%)")
    ax2.set_title("Throughput Improvement")
    ax2.axhline(y=0, color="black", linewidth=0.5)

    # GPU cost reduction
    bars = ax3.bar(x, imp_gpu, color=colors, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, imp_gpu):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f"{val:.1f}%", ha="center", fontweight="bold", fontsize=11)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=15)
    ax3.set_ylabel("Improvement (%)")
    ax3.set_title("GPU Cost Reduction")
    ax3.axhline(y=0, color="black", linewidth=0.5)

    fig.suptitle("Autotune Improvement Across Models",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(os.path.join(cross_dir, f"fig2_autotune_improvement.{ext}"))
    plt.close()
    print("  Saved fig2_autotune_improvement")


def fig3_ablation_comparison(models, cross_dir):
    """Figure 3: full_system vs static_baseline across models."""
    ablation_keys = ["full_system", "no_autotune", "no_moe", "static_baseline"]
    ablation_labels = ["Full\nSystem", "No\nAutotune", "No\nMoE", "Static\nBaseline"]

    # Collect model names that have ablation data
    valid_models = []
    for dir_name, data in models.items():
        if data.get("ablation"):
            valid_models.append((dir_name, data))

    if not valid_models:
        print("  Skipping fig3: no ablation data")
        return

    fig, ax = plt.subplots(figsize=(16, 7))

    bar_width = 0.8 / len(valid_models)
    x = np.arange(len(ablation_keys))

    for i, (dir_name, data) in enumerate(valid_models):
        ablation = data["ablation"]
        vals = [ablation.get(k, {}).get("e2e_p95_ms", 0) for k in ablation_keys]
        offset = (i - len(valid_models) / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, vals, bar_width,
                      label=data["display_name"],
                      color=MODEL_COLORS[i % len(MODEL_COLORS)],
                      edgecolor="white", linewidth=1)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 50,
                        f"{val:.0f}", ha="center", fontsize=8, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(ablation_labels)
    ax.set_ylabel("p95 Latency (ms)")
    ax.set_title("Ablation Study — p95 Latency Across Models and Configurations",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(os.path.join(cross_dir, f"fig3_ablation_comparison.{ext}"))
    plt.close()
    print("  Saved fig3_ablation_comparison")


def fig4_model_scaling(models, cross_dir):
    """Figure 4: Latency/throughput vs model size."""
    sizes = []
    lats = []
    tps_vals = []
    labels = []

    for dir_name, data in models.items():
        ablation = data.get("ablation", {})
        full = ablation.get("full_system", {})
        if not full:
            # Fall back to comparison ON data
            comp = data.get("comparison", {})
            full = comp.get("autotune_on", {})
        if not full:
            continue

        display = data["display_name"]
        # Extract model short name for size lookup
        for size_key, size_val in MODEL_SIZES.items():
            if size_key.lower() in display.lower() or size_key.lower() in dir_name.lower():
                sizes.append(size_val)
                lats.append(full.get("e2e_p95_ms", 0))
                tps_vals.append(full.get("throughput_tps", 0))
                labels.append(display)
                break

    if len(sizes) < 2:
        print("  Skipping fig4: need at least 2 models with data")
        return

    # Sort by model size
    order = np.argsort(sizes)
    sizes = [sizes[i] for i in order]
    lats = [lats[i] for i in order]
    tps_vals = [tps_vals[i] for i in order]
    labels = [labels[i] for i in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Latency vs model size
    ax1.plot(sizes, lats, "o-", color=C_RED, linewidth=2.5, markersize=14, zorder=5)
    for i, label in enumerate(labels):
        ax1.annotate(f"  {label}", (sizes[i], lats[i]),
                    fontsize=11, fontweight="bold",
                    xytext=(10, 5), textcoords="offset points")
    ax1.set_xlabel("Model Size (B parameters)")
    ax1.set_ylabel("p95 Latency (ms)")
    ax1.set_title("Tail Latency vs Model Size")

    # Throughput vs model size
    ax2.plot(sizes, tps_vals, "s-", color=C_GREEN, linewidth=2.5, markersize=14, zorder=5)
    for i, label in enumerate(labels):
        ax2.annotate(f"  {label}", (sizes[i], tps_vals[i]),
                    fontsize=11, fontweight="bold",
                    xytext=(10, 5), textcoords="offset points")
    ax2.set_xlabel("Model Size (B parameters)")
    ax2.set_ylabel("Throughput (tokens/sec)")
    ax2.set_title("Throughput vs Model Size")

    fig.suptitle("Model Size Scaling — Impact on Serving Performance",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(os.path.join(cross_dir, f"fig4_model_scaling.{ext}"))
    plt.close()
    print("  Saved fig4_model_scaling")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-model experiment figures")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                       help="Results directory (default: results/)")
    args = parser.parse_args()

    results_dir = args.results_dir
    print(f"Loading results from {results_dir}...")

    models = discover_models(results_dir)
    if not models:
        print("ERROR: No model results found. Expected directories like:")
        for d in MODEL_DIRS:
            print(f"  {results_dir}/{d}/")
        sys.exit(1)

    print(f"Found {len(models)} model(s): "
          f"{', '.join(d['display_name'] for d in models.values())}")
    print()

    # Per-model figures
    print("=== Per-Model Figures ===")
    for dir_name, data in models.items():
        model_dir = os.path.join(results_dir, dir_name)
        figures_dir = os.path.join(model_dir, "figures")
        gen_per_model_figures(data, model_dir, figures_dir)
    print()

    # Cross-model figures (only if multiple models)
    if len(models) >= 2:
        print("=== Cross-Model Figures ===")
        cross_dir = os.path.join(results_dir, "cross_model")
        os.makedirs(cross_dir, exist_ok=True)

        fig1_profiling_comparison(models, cross_dir)
        fig2_autotune_improvement(models, cross_dir)
        fig3_ablation_comparison(models, cross_dir)
        fig4_model_scaling(models, cross_dir)
        print()

    # Summary
    print("=== Output ===")
    for dir_name, data in models.items():
        fig_dir = os.path.join(results_dir, dir_name, "figures")
        if os.path.isdir(fig_dir):
            files = sorted(os.listdir(fig_dir))
            print(f"  {dir_name}/figures/ ({len(files)} files)")
            for f in files:
                size = os.path.getsize(os.path.join(fig_dir, f))
                print(f"    {f:<45} {size:>8,} bytes")

    if len(models) >= 2:
        cross_dir = os.path.join(results_dir, "cross_model")
        if os.path.isdir(cross_dir):
            files = sorted(os.listdir(cross_dir))
            print(f"  cross_model/ ({len(files)} files)")
            for f in files:
                size = os.path.getsize(os.path.join(cross_dir, f))
                print(f"    {f:<45} {size:>8,} bytes")


if __name__ == "__main__":
    main()
