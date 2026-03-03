#!/usr/bin/env python3
"""Generate dashboard/alert/offline-analysis figures from Prometheus exports."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows(csv_path: str) -> List[Dict[str, float]]:
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if k in {"datetime_utc"}:
                    parsed[k] = v
                else:
                    try:
                        parsed[k] = float(v)
                    except Exception:
                        parsed[k] = 0.0
            rows.append(parsed)
    # Keep only range where controller metrics are valid.
    filtered = [r for r in rows if r.get("decode_target_replicas", 0.0) > 0]
    return filtered if filtered else rows


def load_decode_latency_samples(csv_path: str) -> List[float]:
    vals = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                vals.append(float(row["decode_ms"]))
            except Exception:
                continue
    return vals


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = int(round((len(vals) - 1) * p))
    idx = max(0, min(idx, len(vals) - 1))
    return vals[idx]


def _time_axis(rows: List[Dict[str, float]]) -> List[float]:
    if not rows:
        return []
    t0 = rows[0]["timestamp"]
    return [(r["timestamp"] - t0) / 60.0 for r in rows]


def plot_dashboard(rows: List[Dict[str, float]], out_path: str) -> None:
    x = _time_axis(rows)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    ax = axes.ravel()

    ax[0].plot(x, [r["qps"] for r in rows], color="#1f77b4", linewidth=2, label="QPS")
    ax[0].plot(x, [r["tokens_per_s"] for r in rows], color="#ff7f0e", linewidth=2, label="Tokens/s")
    ax[0].set_title("Business Health")
    ax[0].set_ylabel("rate")
    ax[0].grid(alpha=0.3)
    ax[0].legend()

    ax[1].plot(x, [r["e2e_p95_ms"] for r in rows], color="#d62728", linewidth=2, label="E2E p95")
    ax[1].plot(x, [r["decode_p95_ms"] for r in rows], color="#9467bd", linewidth=2, label="Decode p95")
    ax[1].set_title("Latency")
    ax[1].set_ylabel("ms")
    ax[1].grid(alpha=0.3)
    ax[1].legend()

    ax[2].plot(x, [r["decode_queue"] for r in rows], color="#2ca02c", linewidth=2, label="Decode queue")
    ax[2].plot(x, [r["active_requests"] for r in rows], color="#8c564b", linewidth=2, label="Active req")
    ax[2].set_title("Queue and Inflight")
    ax[2].set_xlabel("time (min)")
    ax[2].set_ylabel("count")
    ax[2].grid(alpha=0.3)
    ax[2].legend()

    ax[3].step(
        x,
        [r["decode_target_replicas"] for r in rows],
        where="post",
        color="#17becf",
        linewidth=2.4,
        label="Decode target replicas",
    )
    ax[3].plot(
        x,
        [r["decode_ema_load"] for r in rows],
        color="#bcbd22",
        linewidth=1.8,
        label="Decode EMA load",
    )
    ax[3].set_title("Scaling Behavior")
    ax[3].set_xlabel("time (min)")
    ax[3].set_ylabel("replicas / load")
    ax[3].grid(alpha=0.3)
    ax[3].legend()

    fig.suptitle("Prometheus Dashboard View (Decode Workload)", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_alerts(rows: List[Dict[str, float]], out_path: str) -> None:
    x = _time_axis(rows)
    e2e_thr = 3000.0
    decode_q_thr = 8.0
    err_thr = 0.02

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    e2e = [r["e2e_p95_ms"] for r in rows]
    axes[0].plot(x, e2e, color="#d62728", linewidth=2, label="E2E p95")
    axes[0].axhline(e2e_thr, color="#111111", linestyle="--", linewidth=1.2, label="alert threshold")
    axes[0].fill_between(
        x,
        [e2e_thr] * len(x),
        e2e,
        where=[v > e2e_thr for v in e2e],
        color="#ff9896",
        alpha=0.35,
        label="firing zone",
    )
    axes[0].set_ylabel("ms")
    axes[0].set_title("Alert: E2ELatencyP95High")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper left")

    dq = [r["decode_queue"] for r in rows]
    axes[1].plot(x, dq, color="#2ca02c", linewidth=2, label="Decode queue")
    axes[1].axhline(decode_q_thr, color="#111111", linestyle="--", linewidth=1.2, label="alert threshold")
    axes[1].fill_between(
        x,
        [decode_q_thr] * len(x),
        dq,
        where=[v > decode_q_thr for v in dq],
        color="#98df8a",
        alpha=0.35,
        label="firing zone",
    )
    axes[1].set_ylabel("queue")
    axes[1].set_title("Alert: DecodeQueueBacklog")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper left")

    er = [r["error_rate"] for r in rows]
    axes[2].plot(x, er, color="#1f77b4", linewidth=2, label="Error rate")
    axes[2].axhline(err_thr, color="#111111", linestyle="--", linewidth=1.2, label="alert threshold")
    axes[2].fill_between(
        x,
        [err_thr] * len(x),
        er,
        where=[v > err_thr for v in er],
        color="#aec7e8",
        alpha=0.35,
        label="firing zone",
    )
    axes[2].set_ylabel("ratio")
    axes[2].set_xlabel("time (min)")
    axes[2].set_title("Alert: HighErrorRate")
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="upper left")

    fig.suptitle("Prometheus Alert View (Threshold vs Signal)", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _replica_transitions(values: List[float]) -> int:
    if not values:
        return 0
    changes = 0
    prev = values[0]
    for v in values[1:]:
        if v != prev:
            changes += 1
            prev = v
    return changes


def plot_offline(
    rows: List[Dict[str, float]],
    summary: Dict,
    out_path: str,
    decode_latency_samples: Optional[List[float]] = None,
) -> None:
    reps = [r["decode_target_replicas"] for r in rows]
    decode_latency = [r["decode_p95_ms"] for r in rows if r["decode_p95_ms"] > 0]
    qps = [r["qps"] for r in rows]

    transition_count = _replica_transitions(reps)
    max_rep = max(reps) if reps else 0.0
    min_rep = min(reps) if reps else 0.0
    peak_qps = max(qps) if qps else 0.0
    p50_decode = summary.get("decode_latency_ms", {}).get("p50", 0.0)
    p95_decode = summary.get("decode_latency_ms", {}).get("p95", 0.0)
    right_title = "Offline Summary (Latency from Prom Histogram)"
    if decode_latency_samples:
        p50_decode = percentile(decode_latency_samples, 0.50)
        p95_decode = percentile(decode_latency_samples, 0.95)
        right_title = "Offline Summary (Latency from Request-level Samples)"

    left_names = ["min_replica", "max_replica", "transitions", "peak_qps"]
    left_vals = [min_rep, max_rep, float(transition_count), peak_qps]
    left_colors = ["#9edae5", "#17becf", "#bcbd22", "#1f77b4"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2.6, 1.2]})

    bars1 = ax1.bar(left_names, left_vals, color=left_colors)
    ax1.set_title("Offline Summary (Replica/QPS)")
    ax1.grid(alpha=0.25, axis="y")
    for b, v in zip(bars1, left_vals):
        ax1.text(b.get_x() + b.get_width() / 2.0, b.get_height(), f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    bars2 = ax2.bar(["decode_p50_ms", "decode_p95_ms"], [p50_decode, p95_decode], color=["#ff9896", "#d62728"])
    ax2.set_title(right_title)
    ax2.grid(alpha=0.25, axis="y")
    for b, v in zip(bars2, [p50_decode, p95_decode]):
        ax2.text(b.get_x() + b.get_width() / 2.0, b.get_height(), f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Offline Analysis Summary (Decode)", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visualization figures for dashboard/alerts/offline analysis.")
    parser.add_argument("--csv", required=True, help="Path to decode_observability_timeseries.csv")
    parser.add_argument("--summary", required=True, help="Path to decode_observability_summary.json")
    parser.add_argument("--out-dir", required=True, help="Output directory for figures")
    parser.add_argument(
        "--decode-latency-samples-csv",
        default="",
        help="Optional request-level decode latency samples CSV (columns: timestamp,request_id,decode_ms)",
    )
    args = parser.parse_args()

    with open(args.summary) as f:
        summary = json.load(f)
    rows = load_rows(args.csv)
    decode_samples = None
    if args.decode_latency_samples_csv and os.path.isfile(args.decode_latency_samples_csv):
        decode_samples = load_decode_latency_samples(args.decode_latency_samples_csv)

    os.makedirs(args.out_dir, exist_ok=True)
    dashboard_path = os.path.join(args.out_dir, "01_dashboard_view.png")
    alerts_path = os.path.join(args.out_dir, "02_alerts_view.png")
    offline_path = os.path.join(args.out_dir, "03_offline_summary.png")

    plot_dashboard(rows, dashboard_path)
    plot_alerts(rows, alerts_path)
    plot_offline(rows, summary, offline_path, decode_samples)

    print(dashboard_path)
    print(alerts_path)
    print(offline_path)


if __name__ == "__main__":
    main()
