import time
import logging
import numpy as np

logger = logging.getLogger(__name__)


class LatencyCollector:
    """Collects per-request latency data and computes summary statistics."""

    def __init__(self):
        self._records = []

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
        """Compute p50/p95/p99 latency, throughput, and TTFT stats."""
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

        return result

    @property
    def records(self):
        return list(self._records)

    def reset(self):
        self._records.clear()
