#!/usr/bin/env python3
"""Plot control-plane effectiveness for piggyback vs polling."""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows(path: str):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "mode": r["mode"],
                    "total_rpc_calls": float(r["total_rpc_calls"]),
                    "avg_fetch_rpc_ms": float(r["avg_fetch_rpc_ms"]),
                    "avg_fetch_piggy_ms": float(r["avg_fetch_piggy_ms"]),
                    "avg_step_ms": float(r["avg_step_ms"]),
                    "step_overrun_pct": float(r["step_overrun_pct"]),
                    "decode_p95_ms": float(r["decode_p95_ms"]),
                    "e2e_p95_ms": float(r["e2e_p95_ms"]),
                }
            )
    rows.sort(key=lambda x: x["mode"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_rows(args.csv)
    modes = [r["mode"] for r in rows]

    effective_fetch = []
    for r in rows:
        if r["mode"] == "piggyback" and r["avg_fetch_piggy_ms"] > 0:
            effective_fetch.append(r["avg_fetch_piggy_ms"])
        else:
            effective_fetch.append(r["avg_fetch_rpc_ms"])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    panels = [
        ("Worker Metrics RPC Calls", "calls", [r["total_rpc_calls"] for r in rows], ["#4c72b0", "#55a868"]),
        ("Effective Metrics Fetch Latency", "ms", effective_fetch, ["#8172b2", "#dd8452"]),
        ("Controller Step Duration", "ms", [r["avg_step_ms"] for r in rows], ["#64b5cd", "#c44e52"]),
        ("Step Overrun Ratio", "%", [r["step_overrun_pct"] for r in rows], ["#937860", "#da8bc3"]),
        ("Decode P95 Latency", "ms", [r["decode_p95_ms"] for r in rows], ["#8c8c8c", "#2ca02c"]),
        ("E2E P95 Latency", "ms", [r["e2e_p95_ms"] for r in rows], ["#1f77b4", "#ff7f0e"]),
    ]

    for ax, (title, ylabel, values, colors) in zip(axes, panels):
        ax.bar(modes, values, color=colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, axis="y")
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")

    fig.suptitle("Piggyback vs Polling: Control-Plane Effectiveness")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=180)


if __name__ == "__main__":
    main()
