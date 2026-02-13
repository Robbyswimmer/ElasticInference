import logging
from prometheus_client import Counter, Histogram, Gauge, start_http_server

logger = logging.getLogger(__name__)

# Request counters
REQUESTS_TOTAL = Counter(
    "gateway_requests_total",
    "Total inference requests received",
    ["status"],
)

# Latency histograms
PREFILL_LATENCY = Histogram(
    "gateway_prefill_latency_ms",
    "Prefill stage latency in milliseconds",
    buckets=[5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
)

DECODE_LATENCY = Histogram(
    "gateway_decode_latency_ms",
    "Decode stage latency in milliseconds",
    buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
)

E2E_LATENCY = Histogram(
    "gateway_e2e_latency_ms",
    "End-to-end request latency in milliseconds",
    buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000],
)

TTFT_LATENCY = Histogram(
    "gateway_time_to_first_token_ms",
    "Time to first token in milliseconds",
    buckets=[5, 10, 25, 50, 100, 250, 500, 1000, 2500],
)

# Queue gauges
PREFILL_QUEUE = Gauge("gateway_prefill_queue_length", "Prefill queue depth")
DECODE_QUEUE = Gauge("gateway_decode_queue_length", "Decode queue depth")
ACTIVE_REQUESTS = Gauge("gateway_active_requests", "Currently active requests")

# Throughput
TOKENS_GENERATED = Counter("gateway_tokens_generated_total", "Total tokens generated")


def start_metrics_server(port=9090):
    """Start Prometheus metrics HTTP server."""
    start_http_server(port)
    logger.info("Prometheus metrics server on :%d", port)
