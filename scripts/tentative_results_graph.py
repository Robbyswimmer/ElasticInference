#!/usr/bin/env python3
"""Generate tentative results graph from in-progress experiment logs.

Uses data extracted from the SLURM log output for models that have completed.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Data from logs ──────────────────────────────────────────────────────

# OPT-1.3B (Phase C)
# autotune_off: p50=6583.8ms p95=9846.1ms p99=9954.5ms, tps=120.5, gpu_s/req=6.019
# autotune_on:  p50=5615.3ms p95=9513.4ms p99=9631.6ms, tps=108.7, gpu_s/req=5.171

# Phi-2 ablation summary:
# full_system:     p95=1898.3  p99=1983.8  tps=28.6  gpu_s/req=1.143  MoE=8  batch=opt
# no_autotune:     p95=1346.8  p99=1402.4  tps=32.5  gpu_s/req=0.976  MoE=8  batch=def
# no_moe:          p95=2428.2  p99=2527.5  tps=37.0  gpu_s/req=1.126  MoE=1  batch=opt
# static_baseline: p95=2320.0  p99=2363.4  tps=32.6  gpu_s/req=1.046  MoE=1  batch=def

models_completed = ["OPT-1.3B", "Phi-2 (2.7B)"]

# ── Autotune OFF vs ON (OPT-1.3B) ──────────────────────────────────────

opt_off = {"p50": 6583.8, "p95": 9846.1, "p99": 9954.5, "tps": 120.5, "gpu_s": 6.019}
opt_on  = {"p50": 5615.3, "p95": 9513.4, "p99": 9631.6, "tps": 108.7, "gpu_s": 5.171}

# ── Phi-2 Ablation ─────────────────────────────────────────────────────

phi2_ablation = {
    "full_system":     {"p95": 1898.3, "p99": 1983.8, "tps": 28.6, "gpu_s": 1.143},
    "no_autotune":     {"p95": 1346.8, "p99": 1402.4, "tps": 32.5, "gpu_s": 0.976},
    "no_moe":          {"p95": 2428.2, "p99": 2527.5, "tps": 37.0, "gpu_s": 1.126},
    "static_baseline": {"p95": 2320.0, "p99": 2363.4, "tps": 32.6, "gpu_s": 1.046},
}

# ── Style ───────────────────────────────────────────────────────────────

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

C_BLUE = "#2E86AB"
C_RED = "#E84855"
C_GREEN = "#44BBA4"
C_ORANGE = "#F5A623"
C_PURPLE = "#7B68EE"

# ════════════════════════════════════════════════════════════════════════
# Figure: 2x2 dashboard
# ════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ── Top-Left: OPT-1.3B Autotune OFF vs ON (Latency) ───────────────────
ax = axes[0, 0]
labels = ["p50", "p95", "p99"]
off_vals = [opt_off["p50"], opt_off["p95"], opt_off["p99"]]
on_vals = [opt_on["p50"], opt_on["p95"], opt_on["p99"]]

x = np.arange(len(labels))
width = 0.35
bars_off = ax.bar(x - width/2, off_vals, width, label="Autotune OFF",
                  color=C_RED, edgecolor="white", linewidth=1.5)
bars_on = ax.bar(x + width/2, on_vals, width, label="Autotune ON",
                 color=C_GREEN, edgecolor="white", linewidth=1.5)

# Improvement annotations
for i in range(len(labels)):
    pct = (off_vals[i] - on_vals[i]) / off_vals[i] * 100
    mid_y = max(off_vals[i], on_vals[i]) + 200
    ax.annotate(f"-{pct:.1f}%", xy=(x[i], mid_y), ha="center",
                fontsize=11, fontweight="bold", color=C_BLUE)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("End-to-End Latency (ms)")
ax.set_title("OPT-1.3B — Autotune Latency Comparison")
ax.legend(loc="upper left")

# ── Top-Right: OPT-1.3B Throughput & GPU Cost ─────────────────────────
ax = axes[0, 1]

# Throughput subplot
cats = ["OFF", "ON"]
tps = [opt_off["tps"], opt_on["tps"]]
gpu = [opt_off["gpu_s"], opt_on["gpu_s"]]

x2 = np.arange(2)
bar_w = 0.3

ax_twin = ax.twinx()
b1 = ax.bar(x2 - bar_w/2, tps, bar_w, label="Throughput (tps)",
            color=C_BLUE, edgecolor="white", linewidth=1.5, alpha=0.85)
b2 = ax_twin.bar(x2 + bar_w/2, gpu, bar_w, label="GPU-s/req",
                 color=C_ORANGE, edgecolor="white", linewidth=1.5, alpha=0.85)

for bar, val in zip(b1, tps):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{val:.1f}", ha="center", fontweight="bold", fontsize=11)
for bar, val in zip(b2, gpu):
    ax_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{val:.2f}", ha="center", fontweight="bold", fontsize=11)

ax.set_xticks(x2)
ax.set_xticklabels(["Autotune OFF", "Autotune ON"])
ax.set_ylabel("Throughput (tokens/sec)")
ax_twin.set_ylabel("GPU-seconds / request")
ax.set_title("OPT-1.3B — Throughput & GPU Efficiency")

# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax_twin.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

# GPU cost improvement annotation
gpu_imp = (opt_off["gpu_s"] - opt_on["gpu_s"]) / opt_off["gpu_s"] * 100
ax.annotate(f"GPU cost: -{gpu_imp:.1f}%",
            xy=(0.5, 0.02), xycoords="axes fraction",
            fontsize=12, fontweight="bold", color=C_GREEN,
            ha="center", bbox=dict(boxstyle="round,pad=0.3",
                                   facecolor="white", edgecolor=C_GREEN, alpha=0.8))

# ── Bottom-Left: Phi-2 Ablation — p95 Latency ────────────────────────
ax = axes[1, 0]

configs = list(phi2_ablation.keys())
display_labels = ["Full\nSystem", "No\nAutotune", "No\nMoE", "Static\nBaseline"]
p95s = [phi2_ablation[c]["p95"] for c in configs]

colors = []
best = min(p95s)
worst = max(p95s)
for v in p95s:
    if v == best:
        colors.append(C_GREEN)
    elif v == worst:
        colors.append(C_RED)
    else:
        colors.append(C_ORANGE)

bars = ax.barh(range(len(configs)), p95s, color=colors, edgecolor="white",
               linewidth=1.5, height=0.6)

for i, (bar, val) in enumerate(zip(bars, p95s)):
    ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
            f"{val:.0f} ms", va="center", fontweight="bold", fontsize=11)
    if i > 0:
        degradation = (val - p95s[0]) / p95s[0] * 100
        sign = "+" if degradation > 0 else ""
        ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2,
                f"({sign}{degradation:.0f}%)", va="center", fontsize=10,
                color="#8B8B8B")

ax.set_yticks(range(len(configs)))
ax.set_yticklabels(display_labels)
ax.set_xlabel("p95 End-to-End Latency (ms)")
ax.set_title("Phi-2 (2.7B) — Ablation: Tail Latency")
ax.invert_yaxis()
ax.set_xlim(0, max(p95s) * 1.35)
ax.axvline(x=p95s[0], color=C_BLUE, linestyle="--", alpha=0.4, linewidth=1.5)

# ── Bottom-Right: Phi-2 Ablation — Throughput & GPU Cost ──────────────
ax = axes[1, 1]

tps_vals = [phi2_ablation[c]["tps"] for c in configs]
gpu_vals = [phi2_ablation[c]["gpu_s"] for c in configs]

x3 = np.arange(len(configs))
bar_w2 = 0.35

ax_twin2 = ax.twinx()
b1 = ax.bar(x3 - bar_w2/2, tps_vals, bar_w2, label="Throughput (tps)",
            color=C_BLUE, edgecolor="white", linewidth=1.5, alpha=0.85)
b2 = ax_twin2.bar(x3 + bar_w2/2, gpu_vals, bar_w2, label="GPU-s/req",
                  color=C_PURPLE, edgecolor="white", linewidth=1.5, alpha=0.85)

for bar, val in zip(b1, tps_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}", ha="center", fontweight="bold", fontsize=10)
for bar, val in zip(b2, gpu_vals):
    ax_twin2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                  f"{val:.2f}", ha="center", fontweight="bold", fontsize=10)

ax.set_xticks(x3)
ax.set_xticklabels(display_labels, fontsize=9)
ax.set_ylabel("Throughput (tokens/sec)")
ax_twin2.set_ylabel("GPU-seconds / request")
ax.set_title("Phi-2 (2.7B) — Ablation: Throughput & Efficiency")

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax_twin2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

# ── Title & Save ──────────────────────────────────────────────────────

fig.suptitle("Elastic Inference — Tentative Multi-Model Results\n"
             "OPT-1.3B (autotune comparison) + Phi-2 (ablation study)",
             fontsize=15, fontweight="bold", y=1.02)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "tentative_results.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved to {out_path}")

# Also save PDF
pdf_path = os.path.join(OUTPUT_DIR, "tentative_results.pdf")
# Re-create for PDF
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# (Duplicate plot code for PDF — simpler to just save both formats)
# For brevity, just copy PNG to PDF equivalent
import shutil
print(f"Saved PNG: {out_path}")
print("\nTentative Results Summary:")
print("=" * 60)
print("\nOPT-1.3B Autotune:")
print(f"  p95 latency:  {opt_off['p95']:.1f} -> {opt_on['p95']:.1f} ms "
      f"(-{(opt_off['p95']-opt_on['p95'])/opt_off['p95']*100:.1f}%)")
print(f"  GPU cost:     {opt_off['gpu_s']:.2f} -> {opt_on['gpu_s']:.2f} s/req "
      f"(-{(opt_off['gpu_s']-opt_on['gpu_s'])/opt_off['gpu_s']*100:.1f}%)")
print(f"  Throughput:   {opt_off['tps']:.1f} -> {opt_on['tps']:.1f} tps "
      f"({(opt_on['tps']-opt_off['tps'])/opt_off['tps']*100:+.1f}%)")
print()
print("Phi-2 (2.7B) Ablation:")
for name, d in phi2_ablation.items():
    print(f"  {name:<20} p95={d['p95']:>7.1f}ms  tps={d['tps']:>5.1f}  "
          f"gpu_s={d['gpu_s']:.3f}")
plt.close()
