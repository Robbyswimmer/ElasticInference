import time
import logging

import grpc
import numpy as np

from common import load_config
from common.proto import inference_pb2, inference_pb2_grpc
from common.gpu_utils import set_cuda_mps_percentage

logger = logging.getLogger(__name__)


class KneeProfiler:
    """Profile inference latency across GPU% allocations to find the knee point.

    Covers both prefill and decode stages at each GPU% level.
    The knee point is where adding more GPU% yields diminishing latency returns.
    """

    def __init__(self, config=None):
        self._config = config or load_config()
        autotune_cfg = self._config.get("autotune", {})
        self._gpu_pcts = autotune_cfg.get("gpu_percentages", [25, 50, 75, 100])
        self._config_space = autotune_cfg.get("config_space", {})

    def _profile_stage(self, gateway_target, sample_prompts, stage,
                       max_tokens, num_warmup=3, num_measure=10):
        """Profile a single stage at all GPU% levels."""
        channel = grpc.insecure_channel(gateway_target)
        stub = inference_pb2_grpc.GatewayServiceStub(channel)
        results = []

        for pct in self._gpu_pcts:
            logger.info("Profiling %s at GPU %d%%", stage, pct)
            set_cuda_mps_percentage(pct)
            time.sleep(2)

            latencies = []
            throughputs = []
            total_runs = num_warmup + num_measure
            prompts_cycle = sample_prompts * (total_runs // len(sample_prompts) + 1)

            for i in range(total_runs):
                req = inference_pb2.InferRequest(
                    request_id=f"profile-{stage}-{pct}-{i}",
                    prompt=prompts_cycle[i],
                    max_tokens=max_tokens,
                )
                try:
                    resp = stub.Infer(req, timeout=60)
                    if i >= num_warmup:
                        if stage == "prefill":
                            latencies.append(resp.latency.prefill_ms)
                        else:
                            latencies.append(resp.latency.decode_ms)
                        if resp.latency.total_ms > 0:
                            throughputs.append(
                                resp.tokens_generated / (resp.latency.total_ms / 1000.0)
                            )
                except grpc.RpcError as e:
                    logger.warning("Profile request failed at %d%%: %s", pct, e)

                if len(latencies) >= num_measure:
                    break

            if latencies:
                results.append({
                    "gpu_pct": pct,
                    "stage": stage,
                    "avg_latency_ms": float(np.mean(latencies)),
                    "p95_latency_ms": float(np.percentile(latencies, 95)),
                    "p99_latency_ms": float(np.percentile(latencies, 99)),
                    "avg_throughput_tps": float(np.mean(throughputs)) if throughputs else 0.0,
                })

        channel.close()
        return results

    def profile_prefill(self, gateway_target, sample_prompts, num_warmup=3, num_measure=10):
        """Profile prefill latency at each GPU% level."""
        return self._profile_stage(
            gateway_target, sample_prompts, "prefill",
            max_tokens=1, num_warmup=num_warmup, num_measure=num_measure,
        )

    def profile_decode(self, gateway_target, sample_prompts, num_warmup=3, num_measure=10):
        """Profile decode latency at each GPU% level."""
        return self._profile_stage(
            gateway_target, sample_prompts, "decode",
            max_tokens=64, num_warmup=num_warmup, num_measure=num_measure,
        )

    def profile_all(self, gateway_target, sample_prompts, num_warmup=3, num_measure=10):
        """Profile both stages. Returns {prefill: [...], decode: [...]}."""
        return {
            "prefill": self.profile_prefill(gateway_target, sample_prompts, num_warmup, num_measure),
            "decode": self.profile_decode(gateway_target, sample_prompts, num_warmup, num_measure),
        }

    def find_knee(self, profile_results):
        """Find the knee point — the GPU% with the best latency/cost tradeoff.

        Uses the maximum distance from line method (Kneedle algorithm).
        """
        if len(profile_results) < 2:
            return profile_results[0] if profile_results else None

        pcts = [r["gpu_pct"] for r in profile_results]
        lats = [r["avg_latency_ms"] for r in profile_results]
        pct_range = max(pcts) - min(pcts) or 1
        lat_range = max(lats) - min(lats) or 1

        norm_pcts = [(p - min(pcts)) / pct_range for p in pcts]
        norm_lats = [(l - min(lats)) / lat_range for l in lats]

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
