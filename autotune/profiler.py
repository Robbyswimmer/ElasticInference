import time
import logging

import grpc

from common import load_config
from common.proto import inference_pb2, inference_pb2_grpc
from common.gpu_utils import set_cuda_mps_percentage

logger = logging.getLogger(__name__)


class KneeProfiler:
    """Profile inference latency across GPU% allocations to find the knee point.

    The knee point is where adding more GPU% yields diminishing latency returns.
    """

    def __init__(self, config=None):
        self._config = config or load_config()
        autotune_cfg = self._config.get("autotune", {})
        self._gpu_pcts = autotune_cfg.get("gpu_percentages", [25, 50, 75])
        self._config_space = autotune_cfg.get("config_space", {})

    def profile_prefill(self, gateway_target, sample_prompts, num_warmup=3, num_measure=10):
        """Measure prefill latency at each GPU% level.

        Returns list of {gpu_pct, avg_latency_ms, p95_latency_ms}.
        """
        channel = grpc.insecure_channel(gateway_target)
        stub = inference_pb2_grpc.GatewayServiceStub(channel)
        results = []

        for pct in self._gpu_pcts:
            logger.info("Profiling prefill at GPU %d%%", pct)
            set_cuda_mps_percentage(pct)
            time.sleep(2)  # Let MPS setting take effect

            latencies = []
            for i, prompt in enumerate(sample_prompts * ((num_warmup + num_measure) // len(sample_prompts) + 1)):
                req = inference_pb2.InferRequest(
                    request_id=f"profile-prefill-{pct}-{i}",
                    prompt=prompt,
                    max_tokens=1,  # Only care about prefill time
                )
                try:
                    resp = stub.Infer(req, timeout=30)
                    if i >= num_warmup:
                        latencies.append(resp.latency.prefill_ms)
                except grpc.RpcError as e:
                    logger.warning("Profile request failed at %d%%: %s", pct, e)

                if len(latencies) >= num_measure:
                    break

            if latencies:
                import numpy as np
                results.append({
                    "gpu_pct": pct,
                    "avg_latency_ms": float(np.mean(latencies)),
                    "p95_latency_ms": float(np.percentile(latencies, 95)),
                })

        channel.close()
        return results

    def find_knee(self, profile_results):
        """Find the knee point — the GPU% with the best latency/cost tradeoff.

        Uses the maximum curvature method on the latency vs GPU% curve.
        """
        if len(profile_results) < 2:
            return profile_results[0] if profile_results else None

        # Normalize axes
        pcts = [r["gpu_pct"] for r in profile_results]
        lats = [r["avg_latency_ms"] for r in profile_results]
        pct_range = max(pcts) - min(pcts) or 1
        lat_range = max(lats) - min(lats) or 1

        norm_pcts = [(p - min(pcts)) / pct_range for p in pcts]
        norm_lats = [(l - min(lats)) / lat_range for l in lats]

        # Maximum distance from line between first and last point
        x0, y0 = norm_pcts[0], norm_lats[0]
        x1, y1 = norm_pcts[-1], norm_lats[-1]

        best_idx = 0
        best_dist = 0
        for i in range(1, len(norm_pcts) - 1):
            px, py = norm_pcts[i], norm_lats[i]
            dist = abs((y1 - y0) * px - (x1 - x0) * py + x1 * y0 - y1 * x0)
            dist /= ((y1 - y0) ** 2 + (x1 - x0) ** 2) ** 0.5 + 1e-9
            if dist > best_dist:
                best_dist = dist
                best_idx = i

        return profile_results[best_idx]
