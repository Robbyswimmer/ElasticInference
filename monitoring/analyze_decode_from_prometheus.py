#!/usr/bin/env python3
"""Offline analysis for decode workload using Prometheus data.

Outputs:
  - decode_observability_timeseries.csv
  - decode_observability_summary.json
  - decode_observability_plot.png (if matplotlib is installed)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass


try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


@dataclass
class QuerySpec:
    key: str
    expr: str


RANGE_QUERIES = [
    QuerySpec("qps", "sum(rate(gateway_requests_total[1m]))"),
    QuerySpec("error_rate", "sum(rate(gateway_requests_total{status!=\"ok\"}[5m])) / clamp_min(sum(rate(gateway_requests_total[5m])), 1e-9)"),
    QuerySpec("tokens_per_s", "sum(rate(gateway_tokens_generated_total[1m]))"),
    QuerySpec("decode_p95_ms", "histogram_quantile(0.95, sum(rate(gateway_decode_latency_ms_bucket[1m])) by (le))"),
    QuerySpec("e2e_p95_ms", "histogram_quantile(0.95, sum(rate(gateway_e2e_latency_ms_bucket[1m])) by (le))"),
    QuerySpec("decode_queue", "avg(gateway_decode_queue_length)"),
    QuerySpec("active_requests", "avg(gateway_active_requests)"),
    QuerySpec("decode_target_replicas", "max(scaler_target_replicas{stage=\"decode\"})"),
    QuerySpec("decode_current_replicas", "max(scaler_current_replicas{stage=\"decode\"})"),
    QuerySpec("decode_ema_load", "max(scaler_ema_load{stage=\"decode\"})"),
    QuerySpec("decode_instant_load", "max(scaler_instant_load{stage=\"decode\"})"),
    QuerySpec("decode_cooldown_s", "max(scaler_cooldown_seconds{stage=\"decode\"})"),
]

INSTANT_QUERIES = [
    QuerySpec("decode_scale_up_events", "increase(scaler_scale_events_total{stage=\"decode\",direction=\"up\"}[30m])"),
    QuerySpec("decode_scale_down_events", "increase(scaler_scale_events_total{stage=\"decode\",direction=\"down\"}[30m])"),
    QuerySpec("decode_oscillation_events", "increase(scaler_oscillation_events_total{stage=\"decode\"}[30m])"),
]


def prom_get_json(base_url: str, path: str, params: dict) -> dict:
    query = urllib.parse.urlencode(params)
    url = f"{base_url.rstrip('/')}{path}?{query}"
    with urllib.request.urlopen(url, timeout=20) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if payload.get("status") != "success":
        raise RuntimeError(f"Prometheus query failed: {payload}")
    return payload


def query_range(base_url: str, expr: str, start_ts: int, end_ts: int, step_s: int) -> list[tuple[int, float]]:
    payload = prom_get_json(
        base_url,
        "/api/v1/query_range",
        {
            "query": expr,
            "start": start_ts,
            "end": end_ts,
            "step": step_s,
        },
    )
    result = payload.get("data", {}).get("result", [])
    if not result:
        return []
    values = result[0].get("values", [])
    out = []
    for ts_raw, val_raw in values:
        ts = int(float(ts_raw))
        try:
            val = float(val_raw)
            if math.isinf(val) or math.isnan(val):
                val = 0.0
        except Exception:
            val = 0.0
        out.append((ts, val))
    return out


def query_instant(base_url: str, expr: str) -> float:
    payload = prom_get_json(base_url, "/api/v1/query", {"query": expr})
    result = payload.get("data", {}).get("result", [])
    if not result:
        return 0.0
    try:
        return float(result[0]["value"][1])
    except Exception:
        return 0.0


def to_iso(ts: int) -> str:
    return dt.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = int(round((len(vals) - 1) * p))
    return vals[max(0, min(idx, len(vals) - 1))]


def plot(rows: list[dict], out_path: str) -> None:
    if not HAS_MPL or not rows:
        return
    x = [r["timestamp"] for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(x, [r["qps"] for r in rows], label="QPS", color="#1f77b4", linewidth=2)
    axes[0].plot(x, [r["tokens_per_s"] for r in rows], label="Tokens/s", color="#ff7f0e", linewidth=2)
    axes[0].set_ylabel("Throughput")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")

    axes[1].plot(x, [r["decode_p95_ms"] for r in rows], label="Decode P95 (ms)", color="#d62728", linewidth=2)
    axes[1].plot(x, [r["e2e_p95_ms"] for r in rows], label="E2E P95 (ms)", color="#9467bd", linewidth=2)
    axes[1].set_ylabel("Latency (ms)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left")

    axes[2].step(x, [r["decode_target_replicas"] for r in rows], where="post", label="Decode target replicas", color="#2ca02c", linewidth=2)
    axes[2].plot(x, [r["decode_ema_load"] for r in rows], label="Decode EMA load", color="#8c564b", linewidth=1.5)
    axes[2].set_ylabel("Replicas / Load")
    axes[2].set_xlabel("Epoch seconds")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper left")

    fig.suptitle("Decode Observability from Prometheus", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def build_summary(rows: list[dict], instant: dict) -> dict:
    decode_replicas = [r["decode_target_replicas"] for r in rows if r["decode_target_replicas"] > 0]
    decode_p95 = [r["decode_p95_ms"] for r in rows if r["decode_p95_ms"] > 0]
    e2e_p95 = [r["e2e_p95_ms"] for r in rows if r["e2e_p95_ms"] > 0]
    qps = [r["qps"] for r in rows]
    rounded_instant = {k: round(v, 3) for k, v in instant.items()}
    return {
        "samples": len(rows),
        "decode_replicas": {
            "min": min(decode_replicas) if decode_replicas else 0,
            "max": max(decode_replicas) if decode_replicas else 0,
            "p95": percentile(decode_replicas, 0.95),
        },
        "decode_latency_ms": {
            "p50": percentile(decode_p95, 0.5),
            "p95": percentile(decode_p95, 0.95),
            "max": max(decode_p95) if decode_p95 else 0,
        },
        "e2e_latency_ms": {
            "p50": percentile(e2e_p95, 0.5),
            "p95": percentile(e2e_p95, 0.95),
            "max": max(e2e_p95) if e2e_p95 else 0,
        },
        "qps": {
            "avg": sum(qps) / len(qps) if qps else 0,
            "max": max(qps) if qps else 0,
        },
        "scale_events": rounded_instant,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze decode workload observability from Prometheus")
    p.add_argument("--prom-url", default="http://127.0.0.1:19090", help="Prometheus base URL")
    p.add_argument("--lookback-minutes", type=int, default=30, help="How many minutes to analyze")
    p.add_argument("--step-seconds", type=int, default=10, help="Query range step")
    p.add_argument("--out-dir", default="eval_results/prometheus_decode", help="Output directory")
    args = p.parse_args()

    now_ts = int(time.time())
    start_ts = now_ts - args.lookback_minutes * 60
    end_ts = now_ts

    ts_set: set[int] = set()
    series_by_key: dict[str, dict[int, float]] = {}

    for q in RANGE_QUERIES:
        values = query_range(args.prom_url, q.expr, start_ts, end_ts, args.step_seconds)
        store = {ts: val for ts, val in values}
        series_by_key[q.key] = store
        ts_set.update(store.keys())

    rows = []
    for ts in sorted(ts_set):
        row = {
            "timestamp": ts,
            "datetime_utc": to_iso(ts),
        }
        for q in RANGE_QUERIES:
            row[q.key] = float(series_by_key.get(q.key, {}).get(ts, 0.0))
        rows.append(row)

    instant_summary = {}
    for q in INSTANT_QUERIES:
        instant_summary[q.key] = query_instant(args.prom_url, q.expr)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "decode_observability_timeseries.csv")
    summary_path = os.path.join(args.out_dir, "decode_observability_summary.json")
    plot_path = os.path.join(args.out_dir, "decode_observability_plot.png")

    write_csv(rows, csv_path)
    summary = build_summary(rows, instant_summary)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    plot(rows, plot_path)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved summary: {summary_path}")
    if HAS_MPL:
        print(f"Saved plot: {plot_path}")
    else:
        print("matplotlib not available; skipped plot")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
