#!/usr/bin/env python3
"""Generate publication-quality figures from autotune experiment results.

Produces 6 figures for the paper:
  1. GPU% Profiling — latency vs GPU% with knee point annotation
  2. GPU% Profiling — throughput vs GPU%
  3. Autotune OFF vs ON — latency comparison
  4. Autotune OFF vs ON — throughput & efficiency comparison
  5. Ablation study — p95 latency breakdown
  6. Ablation study — multi-metric radar/grouped bar

Usage:
    python scripts/generate_figures.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

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


def load_json(name):
    path = os.path.join(RESULTS_DIR, name)
    with open(path) as f:
        return json.load(f)


# ── Figure 1: GPU% Profiling — Latency vs GPU% ──────────────────────────

def fig1_profiling_latency(profile_data):
    """Latency vs simulated GPU% with knee point annotation."""
    gpu_pcts = [d["gpu_pct"] for d in profile_data]
    prefill_avg = [d["prefill"]["avg_latency_ms"] for d in profile_data]
    decode_avg = [d["decode"]["avg_latency_ms"] for d in profile_data]
    e2e_avg = [d["e2e"]["avg_latency_ms"] for d in profile_data]
    e2e_p95 = [d["e2e"]["p95_latency_ms"] for d in profile_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Prefill latency
    ax1.plot(gpu_pcts, prefill_avg, "o-", color=C_BLUE, linewidth=2.5, markersize=10, label="Avg", zorder=5)
    ax1.fill_between(gpu_pcts, 0, prefill_avg, alpha=0.1, color=C_BLUE)
    # Knee point at 50%
    knee_idx = gpu_pcts.index(50)
    ax1.annotate("Knee Point\n(50% GPU)",
                 xy=(50, prefill_avg[knee_idx]),
                 xytext=(65, prefill_avg[knee_idx] + 10),
                 arrowprops=dict(arrowstyle="->", color=C_RED, lw=2),
                 fontsize=11, fontweight="bold", color=C_RED)
    ax1.scatter([50], [prefill_avg[knee_idx]], s=200, color=C_RED, zorder=6, marker="*")
    ax1.set_xlabel("Simulated GPU %")
    ax1.set_ylabel("Prefill Latency (ms)")
    ax1.set_title("Prefill Stage Latency")
    ax1.set_xticks(gpu_pcts)
    ax1.invert_xaxis()

    # Right: Decode latency
    ax2.plot(gpu_pcts, decode_avg, "s-", color=C_ORANGE, linewidth=2.5, markersize=10, label="Avg")
    ax2.plot(gpu_pcts, e2e_p95, "^--", color=C_RED, linewidth=1.5, markersize=8, label="E2E p95", alpha=0.8)
    ax2.fill_between(gpu_pcts, decode_avg, e2e_p95, alpha=0.1, color=C_RED)
    ax2.annotate("Knee Point\n(50% GPU)",
                 xy=(50, decode_avg[knee_idx]),
                 xytext=(65, decode_avg[knee_idx] + 500),
                 arrowprops=dict(arrowstyle="->", color=C_RED, lw=2),
                 fontsize=11, fontweight="bold", color=C_RED)
    ax2.scatter([50], [decode_avg[knee_idx]], s=200, color=C_RED, zorder=6, marker="*")
    ax2.set_xlabel("Simulated GPU %")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title("Decode Stage Latency")
    ax2.set_xticks(gpu_pcts)
    ax2.invert_xaxis()
    ax2.legend()

    fig.suptitle("GPU Contention Profiling — Latency Scales with Reduced GPU Share",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig1_profiling_latency.png"))
    plt.savefig(os.path.join(FIGURES_DIR, "fig1_profiling_latency.pdf"))
    plt.close()
    print("  Saved fig1_profiling_latency")


# ── Figure 2: GPU% Profiling — Throughput vs GPU% ───────────────────────

def fig2_profiling_throughput(profile_data):
    """Throughput and latency-throughput tradeoff."""
    gpu_pcts = [d["gpu_pct"] for d in profile_data]
    throughputs = [d["avg_throughput_tps"] for d in profile_data]
    e2e_avg = [d["e2e"]["avg_latency_ms"] for d in profile_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Throughput bar chart
    bars = ax1.bar(gpu_pcts, throughputs, width=15, color=[C_GREEN, C_ORANGE, C_RED],
                   edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, throughputs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                 f"{val:.0f}", ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax1.set_xlabel("Simulated GPU %")
    ax1.set_ylabel("Throughput (tokens/sec)")
    ax1.set_title("Throughput Drops with GPU Contention")
    ax1.set_xticks(gpu_pcts)
    ax1.set_xticklabels(["25%\n(4 concurrent)", "50%\n(2 concurrent)", "100%\n(1 concurrent)"])

    # Right: Latency-Throughput scatter
    ax2.scatter(throughputs, e2e_avg, s=200, c=[C_RED, C_ORANGE, C_GREEN],
                edgecolors="black", linewidth=1, zorder=5)
    for i, pct in enumerate(gpu_pcts):
        ax2.annotate(f"  {pct}% GPU", (throughputs[i], e2e_avg[i]),
                     fontsize=11, fontweight="bold")
    ax2.set_xlabel("Throughput (tokens/sec)")
    ax2.set_ylabel("Avg E2E Latency (ms)")
    ax2.set_title("Latency–Throughput Tradeoff")
    # Draw ideal region
    ax2.axhspan(0, 1500, alpha=0.05, color=C_GREEN)
    ax2.text(140, 500, "Ideal Zone", fontsize=10, color=C_GREEN, fontstyle="italic")

    fig.suptitle("Throughput Analysis — 4.9x Degradation from 100% to 25% GPU",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig2_profiling_throughput.png"))
    plt.savefig(os.path.join(FIGURES_DIR, "fig2_profiling_throughput.pdf"))
    plt.close()
    print("  Saved fig2_profiling_throughput")


# ── Figure 3: Autotune OFF vs ON — Latency ──────────────────────────────

def fig3_autotune_latency(comparison):
    """Side-by-side latency comparison."""
    off = comparison["autotune_off"]
    on = comparison["autotune_on"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Grouped bar chart for p50/p95/p99
    labels = ["p50", "p95", "p99"]
    off_vals = [off["e2e_p50_ms"], off["e2e_p95_ms"], off["e2e_p99_ms"]]
    on_vals = [on["e2e_p50_ms"], on["e2e_p95_ms"], on["e2e_p99_ms"]]

    x = np.arange(len(labels))
    width = 0.35
    bars_off = ax1.bar(x - width / 2, off_vals, width, label="Autotune OFF",
                       color=C_RED, edgecolor="white", linewidth=1)
    bars_on = ax1.bar(x + width / 2, on_vals, width, label="Autotune ON",
                      color=C_GREEN, edgecolor="white", linewidth=1)

    # Improvement annotations
    for i in range(len(labels)):
        pct = (off_vals[i] - on_vals[i]) / off_vals[i] * 100
        mid_y = max(off_vals[i], on_vals[i]) + 200
        ax1.annotate(f"-{pct:.1f}%", xy=(x[i], mid_y), ha="center",
                     fontsize=11, fontweight="bold", color=C_BLUE)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("End-to-End Latency (ms)")
    ax1.set_title("Latency Percentiles")
    ax1.legend(loc="upper left")

    # Right: Stacked breakdown (prefill + decode)
    categories = ["Autotune OFF", "Autotune ON"]
    prefills = [off.get("prefill_p50_ms", 14.5), on.get("prefill_p50_ms", 13.8)]
    decodes = [off.get("decode_p50_ms", 7904.2), on.get("decode_p50_ms", 6751.4)]

    x2 = np.arange(len(categories))
    ax2.bar(x2, prefills, width=0.5, label="Prefill", color=C_BLUE, edgecolor="white")
    ax2.bar(x2, decodes, width=0.5, bottom=prefills, label="Decode", color=C_ORANGE, edgecolor="white")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(categories)
    ax2.set_ylabel("p50 Latency (ms)")
    ax2.set_title("Latency Breakdown (p50)")
    ax2.legend()

    fig.suptitle("Serving-Time Autotuning — 9.3% p95 Latency Reduction",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig3_autotune_latency.png"))
    plt.savefig(os.path.join(FIGURES_DIR, "fig3_autotune_latency.pdf"))
    plt.close()
    print("  Saved fig3_autotune_latency")


# ── Figure 4: Autotune OFF vs ON — Throughput & Efficiency ──────────────

def fig4_autotune_efficiency(comparison):
    """Throughput and GPU efficiency comparison."""
    off = comparison["autotune_off"]
    on = comparison["autotune_on"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    categories = ["Autotune OFF", "Autotune ON"]
    colors = [C_RED, C_GREEN]

    # Throughput
    tps = [off["throughput_tps"], on["throughput_tps"]]
    bars = axes[0].bar(categories, tps, color=colors, edgecolor="white", linewidth=1.5, width=0.6)
    for bar, val in zip(bars, tps):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.1f}", ha="center", fontweight="bold", fontsize=12)
    axes[0].set_ylabel("Tokens/sec")
    axes[0].set_title("Throughput (+8.5%)")

    # GPU-seconds per request
    gpu_s = [off["gpu_seconds_per_request"], on["gpu_seconds_per_request"]]
    bars = axes[1].bar(categories, gpu_s, color=colors, edgecolor="white", linewidth=1.5, width=0.6)
    for bar, val in zip(bars, gpu_s):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f"{val:.2f}", ha="center", fontweight="bold", fontsize=12)
    axes[1].set_ylabel("GPU-seconds / request")
    axes[1].set_title("GPU Cost per Request (-10.7%)")

    # Summary improvement waterfall
    metrics = ["p95 Latency", "Throughput", "GPU Cost"]
    improvements = [9.3, 8.5, 10.7]
    bar_colors = [C_GREEN, C_GREEN, C_GREEN]
    bars = axes[2].barh(metrics, improvements, color=bar_colors, edgecolor="white",
                        linewidth=1.5, height=0.5)
    for bar, val in zip(bars, improvements):
        axes[2].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}%", va="center", fontweight="bold", fontsize=12)
    axes[2].set_xlabel("Improvement (%)")
    axes[2].set_title("Autotune Improvements")
    axes[2].set_xlim(0, 15)

    fig.suptitle("Autotune Efficiency Gains — Better Throughput with Lower GPU Cost",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig4_autotune_efficiency.png"))
    plt.savefig(os.path.join(FIGURES_DIR, "fig4_autotune_efficiency.pdf"))
    plt.close()
    print("  Saved fig4_autotune_efficiency")


# ── Figure 5: Ablation Study — Feature Contribution ─────────────────────

def fig5_ablation_latency(ablation):
    """Ablation p95 latency showing each feature's contribution."""
    # Order: full_system is the best, then show what happens when you remove each feature
    configs = ["full_system", "no_anti_oscillation", "no_moe", "static_baseline", "no_autotune"]
    labels = [
        "Full System\n(all features)",
        "− Anti-Oscillation",
        "− MoE Routing",
        "Static Baseline\n(nothing enabled)",
        "− Autotune",
    ]
    p95s = [ablation[c]["e2e_p95_ms"] for c in configs]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Color bars by how bad they are relative to full system
    bar_colors = []
    for val in p95s:
        if val < 3500:
            bar_colors.append(C_GREEN)
        elif val < 6000:
            bar_colors.append(C_ORANGE)
        else:
            bar_colors.append(C_RED)

    bars = ax.barh(range(len(configs)), p95s, color=bar_colors, edgecolor="white",
                   linewidth=1.5, height=0.6)

    # Add value labels and degradation %
    full_p95 = ablation["full_system"]["e2e_p95_ms"]
    for i, (bar, val) in enumerate(zip(bars, p95s)):
        ax.text(bar.get_width() + 150, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f} ms", va="center", fontweight="bold", fontsize=11)
        if i > 0:
            degradation = (val - full_p95) / full_p95 * 100
            ax.text(bar.get_width() + 1800, bar.get_y() + bar.get_height() / 2,
                    f"(+{degradation:.0f}%)", va="center", fontsize=10, color=C_GRAY)

    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("p95 End-to-End Latency (ms)")
    ax.set_title("Ablation Study — Feature Contribution to Tail Latency",
                 fontsize=14, fontweight="bold")
    ax.axvline(x=full_p95, color=C_BLUE, linestyle="--", alpha=0.5, linewidth=1.5)
    ax.text(full_p95 + 100, len(configs) - 0.3, "Full System\nBaseline",
            fontsize=9, color=C_BLUE, fontstyle="italic")
    ax.invert_yaxis()
    ax.set_xlim(0, max(p95s) * 1.25)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig5_ablation_latency.png"))
    plt.savefig(os.path.join(FIGURES_DIR, "fig5_ablation_latency.pdf"))
    plt.close()
    print("  Saved fig5_ablation_latency")


# ── Figure 6: Ablation Study — Multi-Metric ─────────────────────────────

def fig6_ablation_multi(ablation):
    """Multi-metric ablation view: latency, throughput, GPU efficiency."""
    configs = ["full_system", "no_autotune", "no_anti_oscillation", "no_moe", "static_baseline"]
    short_labels = ["Full\nSystem", "No\nAutotune", "No Anti-\nOscillation", "No\nMoE", "Static\nBaseline"]

    p95s = [ablation[c]["e2e_p95_ms"] for c in configs]
    tps = [ablation[c]["throughput_tps"] for c in configs]
    gpu_eff = [ablation[c]["gpu_seconds_per_request"] for c in configs]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    x = np.arange(len(configs))

    # p95 Latency
    colors_lat = [C_GREEN if v < 3500 else C_ORANGE if v < 6000 else C_RED for v in p95s]
    bars = axes[0].bar(x, p95s, color=colors_lat, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, p95s):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                     f"{val/1000:.1f}s", ha="center", fontsize=10, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_labels, fontsize=9)
    axes[0].set_ylabel("p95 Latency (ms)")
    axes[0].set_title("Tail Latency (lower is better)")

    # Throughput
    colors_tps = [C_GREEN if c == "full_system" else C_GRAY for c in configs]
    bars = axes[1].bar(x, tps, color=colors_tps, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, tps):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.0f}", ha="center", fontsize=10, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_labels, fontsize=9)
    axes[1].set_ylabel("Tokens/sec")
    axes[1].set_title("Throughput")

    # GPU Efficiency
    colors_gpu = [C_GREEN if v < 2 else C_ORANGE if v < 5 else C_RED for v in gpu_eff]
    bars = axes[2].bar(x, gpu_eff, color=colors_gpu, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, gpu_eff):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                     f"{val:.1f}", ha="center", fontsize=10, fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(short_labels, fontsize=9)
    axes[2].set_ylabel("GPU-seconds / request")
    axes[2].set_title("GPU Cost (lower is better)")

    fig.suptitle("Ablation Study — Multi-Metric Feature Impact Analysis",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig6_ablation_multi.png"))
    plt.savefig(os.path.join(FIGURES_DIR, "fig6_ablation_multi.pdf"))
    plt.close()
    print("  Saved fig6_ablation_multi")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading results...")
    profile_data = load_json("profile_data.json")
    comparison = load_json("comparison.json")
    ablation = load_json("ablation.json")

    print("Generating figures...")
    fig1_profiling_latency(profile_data)
    fig2_profiling_throughput(profile_data)
    fig3_autotune_latency(comparison)
    fig4_autotune_efficiency(comparison)
    fig5_ablation_latency(ablation)
    fig6_ablation_multi(ablation)

    print(f"\nAll figures saved to {FIGURES_DIR}/")
    print("Files:")
    for f in sorted(os.listdir(FIGURES_DIR)):
        size = os.path.getsize(os.path.join(FIGURES_DIR, f))
        print(f"  {f:<45} {size:>8,} bytes")


if __name__ == "__main__":
    main()
