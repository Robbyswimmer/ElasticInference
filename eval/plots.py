import json
import logging

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    logger.warning("matplotlib not installed, plotting disabled")


def plot_latency_cdf(records, output_path="latency_cdf.png"):
    """Plot CDF of end-to-end latency."""
    if not HAS_MPL:
        return
    totals = sorted(r["total_ms"] for r in records)
    cdf = [(i + 1) / len(totals) for i in range(len(totals))]
    plt.figure(figsize=(8, 5))
    plt.plot(totals, cdf, linewidth=2)
    plt.xlabel("End-to-End Latency (ms)")
    plt.ylabel("CDF")
    plt.title("Request Latency CDF")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.95, color="r", linestyle="--", alpha=0.5, label="p95")
    plt.axhline(y=0.99, color="orange", linestyle="--", alpha=0.5, label="p99")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved latency CDF to %s", output_path)


def plot_latency_breakdown(summary, output_path="latency_breakdown.png"):
    """Plot stacked bar of prefill vs decode latency at p50/p95."""
    if not HAS_MPL:
        return
    percentiles = ["p50", "p95"]
    prefill = [summary.get(f"prefill_{p}_ms", 0) for p in percentiles]
    decode = [summary.get(f"decode_{p}_ms", 0) for p in percentiles]

    x = range(len(percentiles))
    plt.figure(figsize=(6, 5))
    plt.bar(x, prefill, label="Prefill", color="#4C72B0")
    plt.bar(x, decode, bottom=prefill, label="Decode", color="#DD8452")
    plt.xticks(list(x), percentiles)
    plt.ylabel("Latency (ms)")
    plt.title("Latency Breakdown: Prefill vs Decode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved latency breakdown to %s", output_path)


def plot_throughput_over_time(records, window_s=5, output_path="throughput.png"):
    """Plot requests/sec and tokens/sec over time."""
    if not HAS_MPL or not records:
        return
    t0 = records[0]["timestamp"]
    times = [r["timestamp"] - t0 for r in records]
    max_t = max(times)

    bins = int(max_t / window_s) + 1
    rps = [0] * bins
    tps = [0] * bins
    for r in records:
        b = min(int((r["timestamp"] - t0) / window_s), bins - 1)
        rps[b] += 1.0 / window_s
        tps[b] += r["tokens_generated"] / window_s

    x = [i * window_s for i in range(bins)]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.plot(x, rps, linewidth=2)
    ax1.set_ylabel("Requests/sec")
    ax1.set_title("Throughput Over Time")
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, tps, linewidth=2, color="#DD8452")
    ax2.set_ylabel("Tokens/sec")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved throughput plot to %s", output_path)


def save_results_json(summary, records, output_path="results.json"):
    """Save summary + raw records to JSON."""
    with open(output_path, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    logger.info("Saved results to %s", output_path)
