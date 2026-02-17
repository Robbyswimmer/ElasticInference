import time
import uuid
import logging
import threading
from concurrent import futures

import grpc
from transformers import AutoTokenizer

from common import load_config
from common.proto import inference_pb2, inference_pb2_grpc

logger = logging.getLogger(__name__)

# gRPC channel options for resilient connections
_CHANNEL_OPTIONS = [
    ("grpc.keepalive_time_ms", 10000),
    ("grpc.keepalive_timeout_ms", 5000),
    ("grpc.keepalive_permit_without_calls", True),
    ("grpc.http2.max_pings_without_data", 0),
    ("grpc.initial_reconnect_backoff_ms", 500),
    ("grpc.max_reconnect_backoff_ms", 5000),
]


def _wait_for_channel(channel, target, timeout=30):
    """Block until a gRPC channel is ready, with logging."""
    logger.info("Waiting for %s to be ready (timeout=%ds)...", target, timeout)
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout)
        logger.info("Connected to %s", target)
    except grpc.FutureTimeoutError:
        logger.warning("Timeout waiting for %s — will retry on first request", target)


class GatewayServicer(inference_pb2_grpc.GatewayServiceServicer):

    def __init__(self, config):
        self._config = config
        model_cfg = config["model"]
        self._tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
        logger.info("Tokenizer loaded for %s", model_cfg["name"])

        # Use connect_host (not bind host) for client connections
        prefill_cfg = config["prefill"]
        decode_cfg = config["decode"]
        prefill_host = prefill_cfg.get("connect_host", "localhost")
        decode_host = decode_cfg.get("connect_host", "localhost")
        prefill_target = f"{prefill_host}:{prefill_cfg['port']}"
        decode_target = f"{decode_host}:{decode_cfg['port']}"

        self._prefill_channel = grpc.insecure_channel(prefill_target, options=_CHANNEL_OPTIONS)
        self._decode_channel = grpc.insecure_channel(decode_target, options=_CHANNEL_OPTIONS)

        # Wait for workers to be available (non-blocking fallback)
        _wait_for_channel(self._prefill_channel, prefill_target, timeout=30)
        _wait_for_channel(self._decode_channel, decode_target, timeout=30)

        self._prefill_stub = inference_pb2_grpc.PrefillServiceStub(self._prefill_channel)
        self._decode_stub = inference_pb2_grpc.DecodeServiceStub(self._decode_channel)
        logger.info("Connected to prefill=%s decode=%s", prefill_target, decode_target)

        self._timeout_ms = config["gateway"].get("request_timeout_ms", 30000)

        # Metrics tracking
        self._lock = threading.Lock()
        self._active = 0
        self._total = 0
        self._window_start = time.monotonic()
        self._window_count = 0
        self._prefill_queue = 0
        self._decode_queue = 0

    def Infer(self, request, context):
        request_id = request.request_id or str(uuid.uuid4())
        t_start = time.perf_counter()

        with self._lock:
            self._active += 1
            self._window_count += 1
            self._prefill_queue += 1

        try:
            # Tokenize
            token_ids = self._tokenizer.encode(request.prompt)
            max_tokens = request.max_tokens if request.max_tokens > 0 else 128
            timeout_s = (request.deadline_ms / 1000.0) if request.deadline_ms > 0 else (self._timeout_ms / 1000.0)

            # Phase 1: Prefill (wait_for_ready retries if channel is transiently down)
            t_queue = time.perf_counter()
            prefill_req = inference_pb2.PrefillRequest(
                request_id=request_id,
                token_ids=token_ids,
                model_name=self._config["model"]["name"],
                sampling_params=request.sampling_params,
            )
            prefill_resp = self._prefill_stub.RunPrefill(
                prefill_req, timeout=timeout_s, wait_for_ready=True,
            )
            t_prefill_done = time.perf_counter()

            with self._lock:
                self._prefill_queue -= 1
                self._decode_queue += 1

            # Phase 2: Decode (streaming)
            decode_req = inference_pb2.DecodeRequest(
                request_id=request_id,
                kv_cache_key=prefill_resp.kv_cache_key,
                first_token_id=prefill_resp.first_token_id,
                max_tokens=max_tokens,
                sampling_params=request.sampling_params,
            )

            remaining_s = timeout_s - (t_prefill_done - t_queue)
            if remaining_s <= 0:
                context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
                context.set_details("Deadline exceeded after prefill")
                return inference_pb2.InferResponse()

            tokens = [self._tokenizer.decode([prefill_resp.first_token_id], skip_special_tokens=False)]
            tokens_generated = 1

            for decode_resp in self._decode_stub.RunDecode(
                decode_req, timeout=remaining_s, wait_for_ready=True,
            ):
                tokens.append(decode_resp.token_text)
                tokens_generated = decode_resp.tokens_generated + 1  # +1 for first token from prefill
                if decode_resp.is_finished:
                    break

            t_done = time.perf_counter()

            with self._lock:
                self._decode_queue -= 1

            generated_text = "".join(tokens)
            queue_wait_ms = (t_queue - t_start) * 1000
            prefill_ms = prefill_resp.prefill_time_ms
            decode_ms = (t_done - t_prefill_done) * 1000
            total_ms = (t_done - t_start) * 1000

            logger.info(
                "Infer %s — %d tokens, total=%.1fms (prefill=%.1fms decode=%.1fms)",
                request_id, tokens_generated, total_ms, prefill_ms, decode_ms,
            )

            return inference_pb2.InferResponse(
                request_id=request_id,
                generated_text=generated_text,
                tokens_generated=tokens_generated,
                latency=inference_pb2.LatencyBreakdown(
                    queue_wait_ms=queue_wait_ms,
                    prefill_ms=prefill_ms,
                    decode_ms=decode_ms,
                    total_ms=total_ms,
                ),
            )

        except grpc.RpcError as e:
            logger.error("Infer %s failed: %s", request_id, e)
            context.set_code(e.code())
            context.set_details(e.details())
            return inference_pb2.InferResponse()
        except Exception as e:
            logger.exception("Infer %s unexpected error: %s", request_id, e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return inference_pb2.InferResponse()
        finally:
            with self._lock:
                self._active -= 1
                self._total += 1

    def GetMetrics(self, request, context):
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._window_start
            arrival_rate = self._window_count / elapsed if elapsed > 0 else 0.0
            self._window_start = now
            self._window_count = 0
            return inference_pb2.GatewayMetrics(
                prefill_queue_length=self._prefill_queue,
                decode_queue_length=self._decode_queue,
                arrival_rate=arrival_rate,
                active_requests=self._active,
            )


def serve(config=None):
    config = config or load_config()
    log_cfg = config.get("logging", {})
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=log_cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s"),
    )

    gw_cfg = config["gateway"]
    host = gw_cfg.get("host", "0.0.0.0")
    port = gw_cfg.get("port", 50051)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=gw_cfg.get("max_queue_size", 256)))
    inference_pb2_grpc.add_GatewayServiceServicer_to_server(GatewayServicer(config), server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logger.info("Gateway serving on %s:%d", host, port)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down gateway...")
        server.stop(grace=5)


if __name__ == "__main__":
    serve()
