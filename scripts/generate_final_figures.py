#!/usr/bin/env python3
"""Generate comprehensive publication-quality figures from multi-model experiment.

Produces 8 figures for the paper:
  1. Cross-model GPU contention profiling (latency + throughput)
  2. Autotune OFF vs ON latency comparison (all 4 models)
  3. Autotune improvement summary (% change per model)
  4. Ablation study — p95 latency (all 4 models grouped)
  5. Ablation study — throughput (all 4 models grouped)
  6. Ablation study — GPU efficiency (all 4 models grouped)
  7. Model scaling — latency & throughput vs model size
  8. Summary dashboard — key findings
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
# Handle the nested results/results from scp
DATA_DIR = os.path.join(PROJECT_DIR, "results", "results")
if not os.path.isdir(DATA_DIR):
    DATA_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "results", "final_figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Style ───────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Colors
C_BLUE = "#2E86AB"
C_RED = "#E84855"
C_GREEN = "#44BBA4"
C_ORANGE = "#F5A623"
C_PURPLE = "#7B68EE"
C_GRAY = "#8B8B8B"
C_DARK = "#2C3E50"

# Model info
MODELS = [
    ("facebook_opt-1.3b", "OPT-1.3B", 1.3, C_BLUE),
    ("Qwen_Qwen2-1.5B", "Qwen2-1.5B", 1.5, C_RED),
    ("microsoft_phi-2", "Phi-2 (2.7B)", 2.7, C_GREEN),
    ("facebook_opt-6.7b", "OPT-6.7B", 6.7, C_ORANGE),
]

ABLATION_KEYS = ["full_system", "no_autotune", "no_moe", "static_baseline"]
ABLATION_LABELS = ["Full System\n(opt batch + MoE)",
                   "No Autotune\n(def batch + MoE)",
                   "No MoE\n(opt batch only)",
                   "Static Baseline\n(def batch, no MoE)"]
ABLATION_SHORT = ["Full\nSystem", "No\nAutotune", "No\nMoE", "Static\nBaseline"]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_all():
    """Load all model data."""
    data = {}
    for dir_name, display, size, color in MODELS:
        model_dir = os.path.join(DATA_DIR, dir_name)
        if not os.path.isdir(model_dir):
            print(f"  Warning: {model_dir} not found")
            continue
        data[dir_name] = {
            "display": display,
            "size": size,
            "color": color,
            "profile": load_json(os.path.join(model_dir, "profile_data.json")),
            "comparison": load_json(os.path.join(model_dir, "comparison.json")),
            "ablation": load_json(os.path.join(model_dir, "ablation.json")),
        }
    return data


def savefig(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(FIGURES_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  Saved {name}")


# ════════════════════════════════════════════════════════════════════════
# Figure 1: Cross-Model GPU Contention Profiling
# ════════════════════════════════════════════════════════════════════════

def fig1_profiling(data):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    for dir_name, info in data.items():
        profile = info["profile"]
        gpu_pcts = [d["gpu_pct"] for d in profile]
        label = info["display"]
        color = info["color"]

        prefill_avg = [d["prefill"]["avg_latency_ms"] for d in profile]
        decode_avg = [d["decode"]["avg_latency_ms"] for d in profile]
        throughputs = [d["avg_throughput_tps"] for d in profile]

        ax1.plot(gpu_pcts, prefill_avg, "o-", label=label, color=color,
                linewidth=2.5, markersize=10)
        ax2.plot(gpu_pcts, decode_avg, "s-", label=label, color=color,
                linewidth=2.5, markersize=10)
        ax3.plot(gpu_pcts, throughputs, "^-", label=label, color=color,
                linewidth=2.5, markersize=10)

    for ax, title, ylabel in [
        (ax1, "Prefill Latency", "Latency (ms)"),
        (ax2, "Decode Latency", "Latency (ms)"),
        (ax3, "Throughput", "Tokens/sec"),
    ]:
        ax.set_xlabel("Simulated GPU %")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.set_xticks([25, 50, 100])
        ax.invert_xaxis()

    fig.suptitle("Figure 1: GPU Contention Profiling Across Model Sizes",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    savefig(fig, "fig1_profiling")


# ════════════════════════════════════════════════════════════════════════
# Figure 2: Autotune OFF vs ON Latency (all models)
# ════════════════════════════════════════════════════════════════════════

def fig2_autotune_latency(data):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (dir_name, info) in enumerate(data.items()):
        ax = axes[idx]
        comp = info["comparison"]
        off = comp["autotune_off"]
        on = comp["autotune_on"]

        labels = ["p50", "p95", "p99"]
        off_vals = [off["e2e_p50_ms"], off["e2e_p95_ms"], off["e2e_p99_ms"]]
        on_vals = [on["e2e_p50_ms"], on["e2e_p95_ms"], on["e2e_p99_ms"]]

        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width/2, off_vals, width, label="Autotune OFF",
               color=C_RED, edgecolor="white", linewidth=1.5, alpha=0.9)
        ax.bar(x + width/2, on_vals, width, label="Autotune ON",
               color=C_GREEN, edgecolor="white", linewidth=1.5, alpha=0.9)

        # Improvement annotations
        for i in range(len(labels)):
            if off_vals[i] > 0:
                pct = (off_vals[i] - on_vals[i]) / off_vals[i] * 100
                mid_y = max(off_vals[i], on_vals[i]) * 1.05
                sign = "+" if pct < 0 else "-"
                color = C_GREEN if pct > 0 else C_RED
                ax.annotate(f"{sign}{abs(pct):.1f}%", xy=(x[i], mid_y), ha="center",
                           fontsize=10, fontweight="bold", color=color)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("E2E Latency (ms)")
        ax.set_title(info["display"], fontweight="bold")
        ax.legend(loc="upper left", fontsize=9)

    fig.suptitle("Figure 2: Autotune OFF vs ON — End-to-End Latency by Model",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    savefig(fig, "fig2_autotune_latency")


# ════════════════════════════════════════════════════════════════════════
# Figure 3: Autotune Improvement Summary
# ════════════════════════════════════════════════════════════════════════

def fig3_autotune_summary(data):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    model_labels = []
    p50_imp = []
    p95_imp = []
    tps_imp = []
    gpu_imp = []
    colors = []

    for dir_name, info in data.items():
        imp = info["comparison"]["improvement"]
        model_labels.append(info["display"])
        colors.append(info["color"])
        p50_imp.append(imp.get("e2e_p50_ms", {}).get("change_pct", 0))
        p95_imp.append(imp.get("e2e_p95_ms", {}).get("change_pct", 0))
        tps_imp.append(imp.get("throughput_tps", {}).get("change_pct", 0))
        gpu_imp.append(imp.get("gpu_seconds_per_request", {}).get("change_pct", 0))

    x = np.arange(len(model_labels))

    # p50 and p95 latency change
    width = 0.35
    bars1 = ax1.bar(x - width/2, p50_imp, width, label="p50", color=C_BLUE,
                    edgecolor="white", linewidth=1.5)
    bars2 = ax1.bar(x + width/2, p95_imp, width, label="p95", color=C_PURPLE,
                    edgecolor="white", linewidth=1.5)
    for bars in [bars1, bars2]:
        for bar in bars:
            val = bar.get_height()
            color = C_GREEN if val > 0 else C_RED
            y_pos = val + 1 if val >= 0 else val - 3
            ax1.text(bar.get_x() + bar.get_width()/2, y_pos,
                    f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold", color=color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, rotation=15)
    ax1.set_ylabel("Latency Change (%)")
    ax1.set_title("Latency (positive = improvement)")
    ax1.axhline(y=0, color="black", linewidth=0.8)
    ax1.legend()

    # Throughput change
    bar_colors = [C_GREEN if v > 0 else C_RED for v in tps_imp]
    bars = ax2.bar(x, tps_imp, color=bar_colors, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, tps_imp):
        y_pos = val + 1 if val >= 0 else val - 3
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                f"{val:+.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_labels, rotation=15)
    ax2.set_ylabel("Throughput Change (%)")
    ax2.set_title("Throughput (positive = improvement)")
    ax2.axhline(y=0, color="black", linewidth=0.8)

    # GPU cost change
    bar_colors = [C_GREEN if v > 0 else C_RED for v in gpu_imp]
    bars = ax3.bar(x, gpu_imp, color=bar_colors, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, gpu_imp):
        y_pos = val + 1 if val >= 0 else val - 3
        ax3.text(bar.get_x() + bar.get_width()/2, y_pos,
                f"{val:+.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_labels, rotation=15)
    ax3.set_ylabel("GPU Cost Change (%)")
    ax3.set_title("GPU Cost/Request (positive = cheaper)")
    ax3.axhline(y=0, color="black", linewidth=0.8)

    fig.suptitle("Figure 3: Autotune Impact — Latency vs Throughput Tradeoff",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    savefig(fig, "fig3_autotune_summary")


# ════════════════════════════════════════════════════════════════════════
# Figure 4: Ablation — p95 Latency (grouped bars)
# ════════════════════════════════════════════════════════════════════════

def fig4_ablation_latency(data):
    fig, ax = plt.subplots(figsize=(16, 8))

    n_models = len(data)
    n_configs = len(ABLATION_KEYS)
    bar_width = 0.8 / n_models
    x = np.arange(n_configs)

    for i, (dir_name, info) in enumerate(data.items()):
        ablation = info["ablation"]
        vals = [ablation.get(k, {}).get("e2e_p95_ms", 0) for k in ABLATION_KEYS]
        offset = (i - n_models/2 + 0.5) * bar_width
        bars = ax.bar(x + offset, vals, bar_width, label=info["display"],
                      color=info["color"], edgecolor="white", linewidth=1,
                      alpha=0.9)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                       f"{val:.0f}", ha="center", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(ABLATION_LABELS, fontsize=10)
    ax.set_ylabel("p95 End-to-End Latency (ms)")
    ax.set_title("Figure 4: Ablation Study — Tail Latency Across Models & Configurations",
                 fontsize=16, fontweight="bold")
    ax.legend(loc="upper left", fontsize=11)

    plt.tight_layout()
    savefig(fig, "fig4_ablation_latency")


# ════════════════════════════════════════════════════════════════════════
# Figure 5: Ablation — Throughput (grouped bars)
# ════════════════════════════════════════════════════════════════════════

def fig5_ablation_throughput(data):
    fig, ax = plt.subplots(figsize=(16, 8))

    n_models = len(data)
    bar_width = 0.8 / n_models
    x = np.arange(len(ABLATION_KEYS))

    for i, (dir_name, info) in enumerate(data.items()):
        ablation = info["ablation"]
        vals = [ablation.get(k, {}).get("throughput_tps", 0) for k in ABLATION_KEYS]
        offset = (i - n_models/2 + 0.5) * bar_width
        bars = ax.bar(x + offset, vals, bar_width, label=info["display"],
                      color=info["color"], edgecolor="white", linewidth=1,
                      alpha=0.9)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f"{val:.0f}", ha="center", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(ABLATION_LABELS, fontsize=10)
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title("Figure 5: Ablation Study — Throughput Across Models & Configurations",
                 fontsize=16, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)

    plt.tight_layout()
    savefig(fig, "fig5_ablation_throughput")


# ════════════════════════════════════════════════════════════════════════
# Figure 6: Ablation — GPU Efficiency (grouped bars)
# ════════════════════════════════════════════════════════════════════════

def fig6_ablation_gpu(data):
    fig, ax = plt.subplots(figsize=(16, 8))

    n_models = len(data)
    bar_width = 0.8 / n_models
    x = np.arange(len(ABLATION_KEYS))

    for i, (dir_name, info) in enumerate(data.items()):
        ablation = info["ablation"]
        vals = [ablation.get(k, {}).get("gpu_seconds_per_request", 0) for k in ABLATION_KEYS]
        offset = (i - n_models/2 + 0.5) * bar_width
        bars = ax.bar(x + offset, vals, bar_width, label=info["display"],
                      color=info["color"], edgecolor="white", linewidth=1,
                      alpha=0.9)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f"{val:.2f}", ha="center", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(ABLATION_LABELS, fontsize=10)
    ax.set_ylabel("GPU-seconds / request")
    ax.set_title("Figure 6: Ablation Study — GPU Cost per Request",
                 fontsize=16, fontweight="bold")
    ax.legend(loc="upper left", fontsize=11)

    plt.tight_layout()
    savefig(fig, "fig6_ablation_gpu")


# ════════════════════════════════════════════════════════════════════════
# Figure 7: Model Scaling — Latency & Throughput vs Model Size
# ════════════════════════════════════════════════════════════════════════

def fig7_model_scaling(data):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    sizes = []
    labels = []
    colors_list = []

    # Collect per-model data
    profile_100_lat = []
    profile_100_tps = []
    full_sys_p95 = []
    full_sys_tps = []
    baseline_p95 = []
    baseline_tps = []

    for dir_name, info in data.items():
        sizes.append(info["size"])
        labels.append(info["display"])
        colors_list.append(info["color"])

        # From profiling at 100% GPU
        p100 = next((d for d in info["profile"] if d["gpu_pct"] == 100), info["profile"][0])
        profile_100_lat.append(p100["e2e"]["avg_latency_ms"])
        profile_100_tps.append(p100["avg_throughput_tps"])

        # From ablation
        full_sys_p95.append(info["ablation"].get("full_system", {}).get("e2e_p95_ms", 0))
        full_sys_tps.append(info["ablation"].get("full_system", {}).get("throughput_tps", 0))
        baseline_p95.append(info["ablation"].get("static_baseline", {}).get("e2e_p95_ms", 0))
        baseline_tps.append(info["ablation"].get("static_baseline", {}).get("throughput_tps", 0))

    # Top-left: Profiling latency vs size
    ax = axes[0, 0]
    ax.plot(sizes, profile_100_lat, "o-", color=C_DARK, linewidth=2.5, markersize=14, zorder=5)
    for i, label in enumerate(labels):
        ax.scatter([sizes[i]], [profile_100_lat[i]], s=200, color=colors_list[i], zorder=6,
                  edgecolors="white", linewidth=2)
        ax.annotate(f"  {label}", (sizes[i], profile_100_lat[i]),
                   fontsize=10, fontweight="bold")
    ax.set_xlabel("Model Size (B parameters)")
    ax.set_ylabel("Avg E2E Latency (ms)")
    ax.set_title("Baseline Latency (100% GPU, no contention)")

    # Top-right: Profiling throughput vs size
    ax = axes[0, 1]
    ax.plot(sizes, profile_100_tps, "s-", color=C_DARK, linewidth=2.5, markersize=14, zorder=5)
    for i, label in enumerate(labels):
        ax.scatter([sizes[i]], [profile_100_tps[i]], s=200, color=colors_list[i], zorder=6,
                  edgecolors="white", linewidth=2)
        ax.annotate(f"  {label}", (sizes[i], profile_100_tps[i]),
                   fontsize=10, fontweight="bold")
    ax.set_xlabel("Model Size (B parameters)")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title("Baseline Throughput (100% GPU)")

    # Bottom-left: Full system vs baseline p95
    ax = axes[1, 0]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, full_sys_p95, width, label="Full System",
           color=C_GREEN, edgecolor="white", linewidth=1.5)
    ax.bar(x + width/2, baseline_p95, width, label="Static Baseline",
           color=C_RED, edgecolor="white", linewidth=1.5)
    for i in range(len(labels)):
        if baseline_p95[i] > 0 and full_sys_p95[i] > 0:
            diff = (baseline_p95[i] - full_sys_p95[i]) / baseline_p95[i] * 100
            y = max(full_sys_p95[i], baseline_p95[i]) * 1.05
            color = C_GREEN if diff > 0 else C_RED
            ax.annotate(f"{diff:+.0f}%", xy=(x[i], y), ha="center",
                       fontsize=10, fontweight="bold", color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("p95 Latency (ms)")
    ax.set_title("Full System vs Static Baseline — Tail Latency")
    ax.legend()

    # Bottom-right: Full system vs baseline throughput
    ax = axes[1, 1]
    ax.bar(x - width/2, full_sys_tps, width, label="Full System",
           color=C_GREEN, edgecolor="white", linewidth=1.5)
    ax.bar(x + width/2, baseline_tps, width, label="Static Baseline",
           color=C_RED, edgecolor="white", linewidth=1.5)
    for i in range(len(labels)):
        if baseline_tps[i] > 0:
            diff = (full_sys_tps[i] - baseline_tps[i]) / baseline_tps[i] * 100
            y = max(full_sys_tps[i], baseline_tps[i]) * 1.05
            color = C_GREEN if diff > 0 else C_RED
            ax.annotate(f"{diff:+.0f}%", xy=(x[i], y), ha="center",
                       fontsize=10, fontweight="bold", color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title("Full System vs Static Baseline — Throughput")
    ax.legend()

    fig.suptitle("Figure 7: Model Size Scaling — Performance vs Parameters",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    savefig(fig, "fig7_model_scaling")


# ════════════════════════════════════════════════════════════════════════
# Figure 8: Summary Dashboard
# ════════════════════════════════════════════════════════════════════════

def fig8_dashboard(data):
    fig = plt.figure(figsize=(20, 14))

    # Create a 3x3 grid
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    # --- Top row: Per-model ablation heatmap ---
    ax = fig.add_subplot(gs[0, :])

    model_names = [info["display"] for info in data.values()]
    metrics = ["p95 Latency\n(ms)", "Throughput\n(tps)", "GPU Cost\n(s/req)"]
    configs = ABLATION_KEYS

    # Build matrix for full_system config across models
    table_data = []
    cell_colors = []
    for dir_name, info in data.items():
        row = []
        row_colors = []
        abl = info["ablation"]
        for cfg in configs:
            d = abl.get(cfg, {})
            p95 = d.get("e2e_p95_ms", 0)
            tps = d.get("throughput_tps", 0)
            gpu = d.get("gpu_seconds_per_request", 0)
            row.append(f"p95={p95:.0f}\ntps={tps:.1f}\ngpu={gpu:.2f}")
            row_colors.append("white")
        table_data.append(row)
        cell_colors.append(row_colors)

    table = ax.table(cellText=table_data, rowLabels=model_names,
                     colLabels=[k.replace("_", "\n") for k in configs],
                     cellColours=cell_colors, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.5)
    ax.set_title("Results Summary — All Models x All Ablation Configs",
                 fontsize=14, fontweight="bold", pad=20)
    ax.axis("off")

    # --- Middle-left: Contention degradation ---
    ax = fig.add_subplot(gs[1, 0])
    for dir_name, info in data.items():
        profile = info["profile"]
        tps_100 = next(d["avg_throughput_tps"] for d in profile if d["gpu_pct"] == 100)
        degradation = []
        gpu_pcts = []
        for d in sorted(profile, key=lambda x: -x["gpu_pct"]):
            gpu_pcts.append(d["gpu_pct"])
            degradation.append(d["avg_throughput_tps"] / tps_100 * 100)
        ax.plot(gpu_pcts, degradation, "o-", label=info["display"],
               color=info["color"], linewidth=2, markersize=8)
    ax.set_xlabel("Simulated GPU %")
    ax.set_ylabel("% of Peak Throughput")
    ax.set_title("Throughput Degradation")
    ax.set_xticks([25, 50, 100])
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    ax.set_ylim(0, 110)

    # --- Middle-center: MoE impact ---
    ax = fig.add_subplot(gs[1, 1])
    model_labels = [info["display"] for info in data.values()]
    moe_on = []
    moe_off = []
    for dir_name, info in data.items():
        abl = info["ablation"]
        moe_on.append(abl.get("full_system", {}).get("e2e_p95_ms", 0))
        moe_off.append(abl.get("no_moe", {}).get("e2e_p95_ms", 0))

    x = np.arange(len(model_labels))
    width = 0.35
    ax.bar(x - width/2, moe_on, width, label="MoE ON", color=C_PURPLE,
           edgecolor="white", linewidth=1.5)
    ax.bar(x + width/2, moe_off, width, label="MoE OFF", color=C_GRAY,
           edgecolor="white", linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([l.split()[0] for l in model_labels], rotation=15, fontsize=9)
    ax.set_ylabel("p95 Latency (ms)")
    ax.set_title("MoE Impact on Tail Latency")
    ax.legend(fontsize=9)

    # --- Middle-right: Batch size impact ---
    ax = fig.add_subplot(gs[1, 2])
    opt_batch = []
    def_batch = []
    for dir_name, info in data.items():
        abl = info["ablation"]
        opt_batch.append(abl.get("full_system", {}).get("throughput_tps", 0))
        def_batch.append(abl.get("no_autotune", {}).get("throughput_tps", 0))

    ax.bar(x - width/2, opt_batch, width, label="Optimal Batch", color=C_GREEN,
           edgecolor="white", linewidth=1.5)
    ax.bar(x + width/2, def_batch, width, label="Default Batch", color=C_ORANGE,
           edgecolor="white", linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([l.split()[0] for l in model_labels], rotation=15, fontsize=9)
    ax.set_ylabel("Throughput (tps)")
    ax.set_title("Batch Size Impact on Throughput")
    ax.legend(fontsize=9)

    # --- Bottom row: Key findings text ---
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")

    findings = []
    for dir_name, info in data.items():
        abl = info["ablation"]
        full = abl.get("full_system", {})
        baseline = abl.get("static_baseline", {})
        if full and baseline:
            lat_diff = (baseline.get("e2e_p95_ms", 0) - full.get("e2e_p95_ms", 0)) / baseline.get("e2e_p95_ms", 1) * 100
            tps_diff = (full.get("throughput_tps", 0) - baseline.get("throughput_tps", 0)) / baseline.get("throughput_tps", 1) * 100
            findings.append(f"{info['display']:>12s}: Full vs Baseline  "
                          f"p95={lat_diff:+.1f}%  throughput={tps_diff:+.1f}%  "
                          f"(full={full.get('e2e_p95_ms',0):.0f}ms / "
                          f"baseline={baseline.get('e2e_p95_ms',0):.0f}ms)")

    text = "Key Findings — Full System vs Static Baseline:\n\n" + "\n".join(findings)
    text += "\n\nNote: Positive p95% = lower latency (improvement). "
    text += "Positive throughput% = higher throughput (improvement)."

    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa",
                     edgecolor="#dee2e6"))

    fig.suptitle("Figure 8: Experiment Dashboard — Multi-Model Elastic Inference",
                 fontsize=16, fontweight="bold", y=1.01)
    savefig(fig, "fig8_dashboard")


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════

def main():
    print(f"Loading data from {DATA_DIR}...")
    data = load_all()
    print(f"Found {len(data)} models: {', '.join(info['display'] for info in data.values())}")
    print()

    print("Generating figures...")
    fig1_profiling(data)
    fig2_autotune_latency(data)
    fig3_autotune_summary(data)
    fig4_ablation_latency(data)
    fig5_ablation_throughput(data)
    fig6_ablation_gpu(data)
    fig7_model_scaling(data)
    fig8_dashboard(data)

    print(f"\nAll figures saved to {FIGURES_DIR}/")
    for f in sorted(os.listdir(FIGURES_DIR)):
        size = os.path.getsize(os.path.join(FIGURES_DIR, f))
        print(f"  {f:<45} {size:>10,} bytes")


if __name__ == "__main__":
    main()
