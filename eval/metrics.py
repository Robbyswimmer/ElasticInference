import time
import logging
import numpy as np

logger = logging.getLogger(__name__)


class LatencyCollector:
    """Collects per-request latency data and computes summary statistics."""

    def __init__(self, gpu_count=1, gpu_pct=100):
        self._records = []
        self._gpu_count = gpu_count
        self._gpu_pct = gpu_pct

    def record(self, request_id, queue_wait_ms, prefill_ms, decode_ms,
               total_ms, tokens_generated, ttft_ms=None):
        self._records.append({
            "request_id": request_id,
            "queue_wait_ms": queue_wait_ms,
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "total_ms": total_ms,
            "tokens_generated": tokens_generated,
            "ttft_ms": ttft_ms,
            "timestamp": time.time(),
        })

    def summary(self):
        """Compute p50/p95/p99 latency, throughput, TTFT, and efficiency stats."""
        if not self._records:
            return {}

        totals = np.array([r["total_ms"] for r in self._records])
        prefills = np.array([r["prefill_ms"] for r in self._records])
        decodes = np.array([r["decode_ms"] for r in self._records])
        tokens = sum(r["tokens_generated"] for r in self._records)
        ttfts = np.array([r["ttft_ms"] for r in self._records if r["ttft_ms"] is not None])

        elapsed_s = (self._records[-1]["timestamp"] - self._records[0]["timestamp"])
        elapsed_s = max(elapsed_s, 0.001)

        result = {
            "num_requests": len(self._records),
            "total_tokens": int(tokens),
            "throughput_rps": len(self._records) / elapsed_s,
            "throughput_tps": tokens / elapsed_s,
            "e2e_p50_ms": float(np.percentile(totals, 50)),
            "e2e_p95_ms": float(np.percentile(totals, 95)),
            "e2e_p99_ms": float(np.percentile(totals, 99)),
            "prefill_p50_ms": float(np.percentile(prefills, 50)),
            "prefill_p95_ms": float(np.percentile(prefills, 95)),
            "decode_p50_ms": float(np.percentile(decodes, 50)),
            "decode_p95_ms": float(np.percentile(decodes, 95)),
        }

        if len(ttfts) > 0:
            result["ttft_p50_ms"] = float(np.percentile(ttfts, 50))
            result["ttft_p95_ms"] = float(np.percentile(ttfts, 95))
            result["ttft_p99_ms"] = float(np.percentile(ttfts, 99))

        # GPU efficiency metrics
        # GPU-seconds per request: (total_ms / 1000) * gpu_fraction * num_gpus
        gpu_fraction = self._gpu_pct / 100.0
        gpu_seconds = [
            (r["total_ms"] / 1000.0) * gpu_fraction * self._gpu_count
            for r in self._records
        ]
        result["gpu_seconds_per_request"] = float(np.mean(gpu_seconds))
        result["gpu_seconds_total"] = float(np.sum(gpu_seconds))
        # GPU utilization estimate: active GPU-time / wall-clock GPU-time
        wall_gpu_seconds = elapsed_s * self._gpu_count * gpu_fraction
        result["gpu_utilization_pct"] = float(
            min(result["gpu_seconds_total"] / max(wall_gpu_seconds, 0.001) * 100, 100)
        )

        return result

    @property
    def records(self):
        return list(self._records)

    def reset(self):
        self._records.clear()


class ScalingStabilityAnalyzer:
    """Analyze scaling event logs for stability metrics.

    Computes: event count, oscillation frequency, convergence time,
    and direction-change ratio.
    """

    def __init__(self, scaling_events):
        self._events = scaling_events

    def analyze(self):
        """Return scaling stability summary dict."""
        if not self._events:
            return {
                "total_scaling_events": 0,
                "scale_up_events": 0,
                "scale_down_events": 0,
                "oscillation_frequency_per_min": 0.0,
                "avg_convergence_time_s": 0.0,
                "max_convergence_time_s": 0.0,
                "direction_change_ratio": 0.0,
                "duration_s": 0.0,
            }

        ups = [e for e in self._events if e["direction"] == "up"]
        downs = [e for e in self._events if e["direction"] == "down"]

        # Oscillation frequency: direction changes per minute
        directions = [e["direction"] for e in self._events]
        changes = sum(1 for i in range(1, len(directions)) if directions[i] != directions[i - 1])

        if len(self._events) >= 2:
            duration_s = self._events[-1]["timestamp"] - self._events[0]["timestamp"]
            duration_s = max(duration_s, 0.001)
            osc_per_min = (changes / duration_s) * 60.0
        else:
            duration_s = 0.0
            osc_per_min = 0.0

        # Convergence time: time from first event to last event in each burst
        # A "burst" ends when no event happens for > 60s
        convergence_times = []
        burst_start = self._events[0]["timestamp"]
        for i in range(1, len(self._events)):
            gap = self._events[i]["timestamp"] - self._events[i - 1]["timestamp"]
            if gap > 60:
                convergence_times.append(self._events[i - 1]["timestamp"] - burst_start)
                burst_start = self._events[i]["timestamp"]
        convergence_times.append(self._events[-1]["timestamp"] - burst_start)

        # Direction change ratio: changes / total events (lower = more stable)
        dcr = changes / max(len(self._events) - 1, 1)

        return {
            "total_scaling_events": len(self._events),
            "scale_up_events": len(ups),
            "scale_down_events": len(downs),
            "oscillation_frequency_per_min": round(osc_per_min, 2),
            "avg_convergence_time_s": round(float(np.mean(convergence_times)), 2) if convergence_times else 0.0,
            "max_convergence_time_s": round(float(np.max(convergence_times)), 2) if convergence_times else 0.0,
            "direction_change_ratio": round(dcr, 3),
            "duration_s": round(duration_s, 2),
        }
