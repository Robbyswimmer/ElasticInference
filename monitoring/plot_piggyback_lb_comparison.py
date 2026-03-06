#!/usr/bin/env python3
"""Analyze and plot baseline vs piggyback+LB experiment results."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ModeStats:
    mode: str
    count: int
    duration_s: float
    tpot_p50_ms: float
    tpot_p95_ms: float
    tpot_p99_ms: float
    prefill_tokens_per_s: float
    decode_tokens_per_s: float
    resp_mean_ms: float
    queue_prefill_mean: float
    queue_prefill_p50: float
    queue_prefill_p95: float
    queue_decode_mean: float
    queue_decode_p50: float
    queue_decode_p95: float


def load_requests(path: str) -> List[dict]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["status"] != "ok":
                continue
            rows.append(
                {
                    "completion_ts": float(r["completion_ts"]),
                    "total_ms": float(r["total_ms"]),
                    "decode_ms": float(r["decode_ms"]),
                    "input_tokens": int(r["input_tokens"]),
                    "output_tokens": int(r["output_tokens"]),
                    "tpot_ms": float(r["tpot_ms"]),
                }
            )
    return rows


def load_series(path: str) -> np.ndarray:
    vals = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            vals.append(float(r["value"]))
    return np.array(vals, dtype=float) if vals else np.array([], dtype=float)


def quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, q))


def compute_mode_stats(mode: str, out_dir: str) -> ModeStats:
    req = load_requests(os.path.join(out_dir, f"{mode}_requests.csv"))
    q_prefill = load_series(os.path.join(out_dir, f"{mode}_queue_prefill.csv"))
    q_decode = load_series(os.path.join(out_dir, f"{mode}_queue_decode.csv"))

    if not req:
        raise RuntimeError(f"no successful requests for mode={mode}")

    completion = np.array([r["completion_ts"] for r in req], dtype=float)
    total_ms = np.array([r["total_ms"] for r in req], dtype=float)
    tpot_ms = np.array([r["tpot_ms"] for r in req], dtype=float)
    input_tokens = np.array([r["input_tokens"] for r in req], dtype=float)
    output_tokens = np.array([r["output_tokens"] for r in req], dtype=float)
    duration_s = max(1e-6, float(np.max(completion) - np.min(completion)))

    return ModeStats(
        mode=mode,
        count=len(req),
        duration_s=duration_s,
        tpot_p50_ms=quantile(tpot_ms, 0.50),
        tpot_p95_ms=quantile(tpot_ms, 0.95),
        tpot_p99_ms=quantile(tpot_ms, 0.99),
        prefill_tokens_per_s=float(np.sum(input_tokens) / duration_s),
        decode_tokens_per_s=float(np.sum(output_tokens) / duration_s),
        resp_mean_ms=float(np.mean(total_ms)),
        queue_prefill_mean=float(np.mean(q_prefill)) if q_prefill.size else 0.0,
        queue_prefill_p50=quantile(q_prefill, 0.50),
        queue_prefill_p95=quantile(q_prefill, 0.95),
        queue_decode_mean=float(np.mean(q_decode)) if q_decode.size else 0.0,
        queue_decode_p50=quantile(q_decode, 0.50),
        queue_decode_p95=quantile(q_decode, 0.95),
    )


def write_summary(stats: List[ModeStats], out_dir: str) -> None:
    rows = [s.__dict__ for s in stats]
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)

    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_tpot(stats: List[ModeStats], out_dir: str) -> None:
    modes = [s.mode for s in stats]
    x = np.arange(len(modes))
    p50 = [s.tpot_p50_ms for s in stats]
    p95 = [s.tpot_p95_ms for s in stats]
    p99 = [s.tpot_p99_ms for s in stats]

    w = 0.24
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - w, p50, width=w, label="p50", color="#4c72b0")
    ax.bar(x, p95, width=w, label="p95", color="#55a868")
    ax.bar(x + w, p99, width=w, label="p99", color="#c44e52")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_title("TPOT (decode_ms / output_tokens)")
    ax.set_ylabel("ms/token")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tpot_quantiles.png"), dpi=180)


def plot_tokens_per_sec(stats: List[ModeStats], out_dir: str) -> None:
    modes = [s.mode for s in stats]
    x = np.arange(len(modes))
    prefill = [s.prefill_tokens_per_s for s in stats]
    decode = [s.decode_tokens_per_s for s in stats]
    w = 0.34

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - w / 2, prefill, width=w, label="prefill tokens/s", color="#8172b2")
    ax.bar(x + w / 2, decode, width=w, label="decode tokens/s", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_title("Token Throughput")
    ax.set_ylabel("tokens/s")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tokens_per_sec.png"), dpi=180)


def plot_queue(stats: List[ModeStats], out_dir: str) -> None:
    modes = [s.mode for s in stats]
    x = np.arange(len(modes))
    w = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    pref_mean = [s.queue_prefill_mean for s in stats]
    pref_p50 = [s.queue_prefill_p50 for s in stats]
    pref_p95 = [s.queue_prefill_p95 for s in stats]
    axes[0].bar(x - w, pref_mean, width=w, label="mean", color="#64b5cd")
    axes[0].bar(x, pref_p50, width=w, label="p50", color="#4c72b0")
    axes[0].bar(x + w, pref_p95, width=w, label="p95", color="#c44e52")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(modes)
    axes[0].set_title("Prefill Queue Distribution")
    axes[0].set_ylabel("queue length")
    axes[0].grid(alpha=0.25, axis="y")
    axes[0].legend()

    dec_mean = [s.queue_decode_mean for s in stats]
    dec_p50 = [s.queue_decode_p50 for s in stats]
    dec_p95 = [s.queue_decode_p95 for s in stats]
    axes[1].bar(x - w, dec_mean, width=w, label="mean", color="#64b5cd")
    axes[1].bar(x, dec_p50, width=w, label="p50", color="#4c72b0")
    axes[1].bar(x + w, dec_p95, width=w, label="p95", color="#c44e52")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(modes)
    axes[1].set_title("Decode Queue Distribution")
    axes[1].set_ylabel("queue length")
    axes[1].grid(alpha=0.25, axis="y")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "queue_distribution.png"), dpi=180)


def plot_response_scatter(out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for mode, color, mean_color in [
        ("baseline", "#4c72b0", "#1f3a63"),
        ("piggyback_lb", "#55a868", "#2f6b38"),
    ]:
        req = load_requests(os.path.join(out_dir, f"{mode}_requests.csv"))
        total = np.array([r["total_ms"] for r in req], dtype=float)
        x = np.arange(len(total))
        ax.scatter(x, total, s=14, alpha=0.65, color=color, label=f"{mode} points")
        mean_v = float(np.mean(total)) if total.size else 0.0
        ax.axhline(
            mean_v,
            linestyle="--",
            color=mean_color,
            linewidth=1.4,
            label=f"{mode} mean={mean_v:.1f}ms",
        )
    ax.set_title("Response Time Scatter (Combined)")
    ax.set_xlabel("request index")
    ax.set_ylabel("total response time (ms)")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "response_scatter.png"), dpi=180)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    stats = [
        compute_mode_stats("baseline", out_dir),
        compute_mode_stats("piggyback_lb", out_dir),
    ]
    write_summary(stats, out_dir)
    plot_tpot(stats, out_dir)
    plot_tokens_per_sec(stats, out_dir)
    plot_queue(stats, out_dir)
    plot_response_scatter(out_dir)
    print(json.dumps([s.__dict__ for s in stats], indent=2))


if __name__ == "__main__":
    main()
