import time
import uuid
import logging
from concurrent import futures

import grpc
import numpy as np

from common.proto import inference_pb2, inference_pb2_grpc

from eval.metrics import LatencyCollector

logger = logging.getLogger(__name__)

# Sample prompts of varying lengths
SAMPLE_PROMPTS = [
    "Hello, how are you?",
    "Explain the theory of relativity in simple terms.",
    "Write a short story about a robot learning to paint. The robot was built in a factory "
    "in Detroit and shipped to an art school in Paris where it began its journey.",
    "What is the meaning of life? Philosophers have debated this question for centuries, "
    "from Socrates to Nietzsche, and yet no consensus has emerged. Consider the perspectives "
    "of existentialism, nihilism, and absurdism in your response.",
    "Summarize the key events of World War II.",
]


class WorkloadGenerator:
    """Generate synthetic inference workloads against the gateway."""

    def __init__(self, gateway_target, num_requests=100, arrival_rate=10.0,
                 max_tokens=128, burstiness=1.0):
        self._target = gateway_target
        self._num_requests = num_requests
        self._arrival_rate = arrival_rate
        self._max_tokens = max_tokens
        self._burstiness = burstiness
        self._collector = LatencyCollector()

    @property
    def collector(self):
        return self._collector

    def _generate_interarrival(self):
        """Generate inter-arrival time. burstiness > 1 = more bursty."""
        mean = 1.0 / self._arrival_rate
        if self._burstiness <= 1.0:
            return np.random.exponential(mean)
        # Use Pareto-like distribution for bursty traffic
        return np.random.exponential(mean / self._burstiness)

    def _send_request(self, stub, prompt, request_id):
        """Send a single inference request and record metrics."""
        req = inference_pb2.InferRequest(
            request_id=request_id,
            prompt=prompt,
            max_tokens=self._max_tokens,
        )
        try:
            t0 = time.perf_counter()
            resp = stub.Infer(req, timeout=30)
            t1 = time.perf_counter()

            latency = resp.latency
            ttft_ms = latency.queue_wait_ms + latency.prefill_ms
            self._collector.record(
                request_id=resp.request_id,
                queue_wait_ms=latency.queue_wait_ms,
                prefill_ms=latency.prefill_ms,
                decode_ms=latency.decode_ms,
                total_ms=latency.total_ms,
                tokens_generated=resp.tokens_generated,
                ttft_ms=ttft_ms,
            )
            logger.debug("Request %s: %d tokens in %.1fms",
                         request_id, resp.tokens_generated, latency.total_ms)
        except grpc.RpcError as e:
            logger.error("Request %s failed: %s", request_id, e)

    def run(self):
        """Run the workload and return summary metrics."""
        channel = grpc.insecure_channel(self._target)
        stub = inference_pb2_grpc.GatewayServiceStub(channel)

        logger.info("Starting workload: %d requests @ %.1f rps to %s",
                     self._num_requests, self._arrival_rate, self._target)

        pool = futures.ThreadPoolExecutor(max_workers=min(self._num_requests, 64))
        futs = []

        for i in range(self._num_requests):
            prompt = SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]
            request_id = str(uuid.uuid4())
            fut = pool.submit(self._send_request, stub, prompt, request_id)
            futs.append(fut)
            time.sleep(self._generate_interarrival())

        # Wait for all to complete
        for fut in futures.as_completed(futs):
            fut.result()

        pool.shutdown()
        channel.close()

        summary = self._collector.summary()
        logger.info("Workload complete: %s", summary)
        return summary
