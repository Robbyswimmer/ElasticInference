#!/usr/bin/env python3
"""Plot piggyback A/B experiment results."""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "mode": r["mode"],
                "ok": int(float(r["ok"])),
                "err": int(float(r["err"])),
                "duration_s": float(r["duration_s"]),
                "rpc_decode": float(r["rpc_decode"]),
                "rpc_prefill": float(r["rpc_prefill"]),
                "max_decode_replicas": float(r["max_decode_replicas"]),
                "avg_piggyback_age_s": float(r["avg_piggyback_age_s"]),
                "decode_p95_ms": float(r.get("decode_p95_ms", 0) or 0),
                "e2e_p95_ms": float(r.get("e2e_p95_ms", 0) or 0),
                "scaleup_reaction_s": float(r.get("scaleup_reaction_s", -1) or -1),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_rows(args.csv)
    rows = sorted(rows, key=lambda x: x["mode"])
    modes = [r["mode"] for r in rows]

    total_rpc = [r["rpc_decode"] + r["rpc_prefill"] for r in rows]
    max_rep = [r["max_decode_replicas"] for r in rows]
    decode_p95 = [r["decode_p95_ms"] for r in rows]
    e2e_p95 = [r["e2e_p95_ms"] for r in rows]
    reaction = [r["scaleup_reaction_s"] for r in rows]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    axes = axes.flatten()

    axes[0].bar(modes, total_rpc, color=["#4c72b0", "#55a868"])
    axes[0].set_title("Worker Metrics RPC Calls")
    axes[0].set_ylabel("calls (lower is better)")
    axes[0].grid(alpha=0.25, axis="y")
    for i, v in enumerate(total_rpc):
        axes[0].text(i, v, f"{v:.0f}", ha="center", va="bottom")

    axes[1].bar(modes, max_rep, color=["#8172b2", "#dd8452"])
    axes[1].set_title("Max Decode Replicas")
    axes[1].set_ylabel("replicas")
    axes[1].grid(alpha=0.25, axis="y")
    for i, v in enumerate(max_rep):
        axes[1].text(i, v, f"{v:.1f}", ha="center", va="bottom")

    axes[2].bar(modes, decode_p95, color=["#64b5cd", "#c44e52"])
    axes[2].set_title("Decode Token Latency P95")
    axes[2].set_ylabel("ms")
    axes[2].grid(alpha=0.25, axis="y")
    for i, v in enumerate(decode_p95):
        axes[2].text(i, v, f"{v:.1f}", ha="center", va="bottom")

    axes[3].bar(modes, e2e_p95, color=["#937860", "#da8bc3"])
    axes[3].set_title("E2E Latency P95")
    axes[3].set_ylabel("ms")
    axes[3].grid(alpha=0.25, axis="y")
    for i, v in enumerate(e2e_p95):
        axes[3].text(i, v, f"{v:.1f}", ha="center", va="bottom")

    axes[4].bar(modes, reaction, color=["#8c8c8c", "#2ca02c"])
    axes[4].set_title("Scale-up Reaction Time")
    axes[4].set_ylabel("s")
    axes[4].grid(alpha=0.25, axis="y")
    for i, v in enumerate(reaction):
        if v < 0:
            axes[4].text(i, 0.1, "N/A", ha="center", va="bottom")
        else:
            axes[4].text(i, v, f"{v:.1f}", ha="center", va="bottom")

    axes[5].axis("off")

    fig.suptitle("Piggyback vs Polling (Decode A/B)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=180)
    print(args.out)


if __name__ == "__main__":
    main()
