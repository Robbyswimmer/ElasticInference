import time
import uuid
import socket
import logging
import threading
from concurrent import futures

import grpc
from transformers import AutoTokenizer

from common import load_config
from common.proto import inference_pb2, inference_pb2_grpc
from common.piggyback_metrics import PiggybackMetricsStore
from gateway.metrics import (
    REQUESTS_TOTAL,
    PREFILL_LATENCY,
    DECODE_LATENCY,
    E2E_LATENCY,
    TTFT_LATENCY,
    PREFILL_QUEUE,
    DECODE_QUEUE,
    ACTIVE_REQUESTS,
    TOKENS_GENERATED,
    start_metrics_server,
)

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


class StagePodRouter:
    """Resolve per-pod endpoints from headless DNS and choose the best pod."""

    def __init__(
        self,
        stage,
        host,
        port,
        metrics_timeout_s=1.0,
        refresh_interval_s=5,
        metrics_poll_interval_s=0.2,
        metrics_stale_after_s=1.0,
    ):
        self._stage = stage
        self._host = host
        self._port = int(port)
        self._metrics_timeout_s = float(metrics_timeout_s)
        self._refresh_interval_s = float(refresh_interval_s)
        self._metrics_poll_interval_s = float(metrics_poll_interval_s)
        self._metrics_stale_after_s = float(metrics_stale_after_s)
        self._lock = threading.Lock()
        self._last_refresh = 0.0
        self._targets = {}  # ip -> {"channel": ch, "stub": stub, "metrics": WorkerMetrics|None, "last_metrics_ts": float}
        self._pick_rr = 0
        self._poll_stop = threading.Event()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

    def _resolve_ips(self):
        infos = socket.getaddrinfo(self._host, self._port, family=socket.AF_INET, type=socket.SOCK_STREAM)
        return sorted({info[4][0] for info in infos if info and info[4] and info[4][0]})

    def _make_stub(self, ip):
        target = f"{ip}:{self._port}"
        channel = grpc.insecure_channel(target, options=_CHANNEL_OPTIONS)
        if self._stage == "prefill":
            stub = inference_pb2_grpc.PrefillServiceStub(channel)
        else:
            stub = inference_pb2_grpc.DecodeServiceStub(channel)
        return channel, stub

    def _refresh(self):
        now = time.monotonic()
        if now - self._last_refresh < self._refresh_interval_s:
            return
        ips = []
        try:
            ips = self._resolve_ips()
        except Exception as e:
            logger.debug("Failed to resolve %s host=%s: %s", self._stage, self._host, e)
            self._last_refresh = now
            return

        with self._lock:
            old_ips = set(self._targets.keys())
            new_ips = set(ips)
            for ip in (new_ips - old_ips):
                ch, stub = self._make_stub(ip)
                self._targets[ip] = {"channel": ch, "stub": stub, "metrics": None, "last_metrics_ts": 0.0}
            for ip in (old_ips - new_ips):
                entry = self._targets.pop(ip, None)
                if entry:
                    try:
                        entry["channel"].close()
                    except Exception:
                        pass
            self._last_refresh = now

    def _poll_loop(self):
        while not self._poll_stop.is_set():
            t0 = time.monotonic()
            self._refresh()
            with self._lock:
                items = [(ip, entry["stub"]) for ip, entry in self._targets.items()]
            now_ts = time.time()
            for ip, stub in items:
                try:
                    m = stub.GetMetrics(
                        inference_pb2.MetricsRequest(),
                        timeout=self._metrics_timeout_s,
                        wait_for_ready=True,
                    )
                except grpc.RpcError:
                    continue
                with self._lock:
                    entry = self._targets.get(ip)
                    if entry is not None:
                        entry["metrics"] = m
                        entry["last_metrics_ts"] = now_ts
            dt = max(0.0, time.monotonic() - t0)
            sleep_s = max(0.0, self._metrics_poll_interval_s - dt)
            self._poll_stop.wait(sleep_s)

    def pick(self, new_request_tokens):
        """Pick the pod with min estimated response time.

        R = (queued_tokens + new_request_tokens) / capacity_tokens_per_sec
        """
        with self._lock:
            items = list(self._targets.items())
        if not items:
            return None, None, None

        best_stub = None
        best_ip = None
        best_r = None
        new_tokens = max(0, int(new_request_tokens))
        now_ts = time.time()
        for ip, entry in items:
            m = entry.get("metrics")
            last_ts = float(entry.get("last_metrics_ts", 0.0))
            if m is None:
                continue
            if now_ts - last_ts > self._metrics_stale_after_s:
                continue
            cap = max(float(m.capacity_tokens_per_sec), 1e-6)
            queued_tokens = max(0, int(m.queued_tokens))
            r = (queued_tokens + new_tokens) / cap
            if best_r is None or r < best_r:
                best_r = r
                best_ip = ip
                best_stub = entry["stub"]

        if best_stub is not None:
            return best_stub, best_ip, best_r

        # Cold start / stale cache fallback: pick from existing stubs without extra RPC.
        with self._lock:
            fallback_items = [(ip, entry["stub"]) for ip, entry in self._targets.items()]
        if not fallback_items:
            return None, None, None
        idx = self._pick_rr % len(fallback_items)
        self._pick_rr += 1
        ip, stub = fallback_items[idx]
        return stub, ip, None


def _wait_for_channel(channel, target, timeout=30):
    """Block until a gRPC channel is ready, with logging."""
    logger.info("Waiting for %s to be ready (timeout=%ds)...", target, timeout)
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout)
        logger.info("Connected to %s", target)
    except grpc.FutureTimeoutError:
        logger.warning("Timeout waiting for %s — will retry on first request", target)


class GatewayServicer(inference_pb2_grpc.GatewayServiceServicer):

    @staticmethod
    def _make_stage_stub(stage, channel):
        if stage == "prefill":
            return inference_pb2_grpc.PrefillServiceStub(channel)
        return inference_pb2_grpc.DecodeServiceStub(channel)

    def _build_service_pool(self, stage, target, pool_size):
        entries = []
        for i in range(pool_size):
            channel = grpc.insecure_channel(target, options=_CHANNEL_OPTIONS)
            # Keep startup behavior deterministic while avoiding long waits for extra pool members.
            if i == 0:
                _wait_for_channel(channel, target, timeout=30)
            entries.append({
                "channel": channel,
                "stub": self._make_stage_stub(stage, channel),
            })
        return entries

    def _pick_service_stub(self, stage):
        pool = self._prefill_service_pool if stage == "prefill" else self._decode_service_pool
        if len(pool) == 1:
            return pool[0]["stub"], "service"
        with self._service_pool_lock:
            idx = self._service_pool_rr[stage]
            self._service_pool_rr[stage] = (idx + 1) % len(pool)
        return pool[idx]["stub"], f"service-pool-{idx}"

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

        gw_cfg = config["gateway"]
        self._service_channel_pool_size = max(1, int(gw_cfg.get("service_channel_pool_size", 1)))
        self._service_pool_lock = threading.Lock()
        self._service_pool_rr = {"prefill": 0, "decode": 0}

        self._prefill_service_pool = self._build_service_pool(
            "prefill", prefill_target, self._service_channel_pool_size,
        )
        self._decode_service_pool = self._build_service_pool(
            "decode", decode_target, self._service_channel_pool_size,
        )
        self._prefill_channel = self._prefill_service_pool[0]["channel"]
        self._decode_channel = self._decode_service_pool[0]["channel"]
        self._prefill_stub = self._prefill_service_pool[0]["stub"]
        self._decode_stub = self._decode_service_pool[0]["stub"]
        logger.info(
            "Connected to prefill=%s decode=%s (service_channel_pool_size=%d)",
            prefill_target, decode_target, self._service_channel_pool_size,
        )

        self._timeout_ms = gw_cfg.get("request_timeout_ms", 30000)
        scaling_cfg = config.get("scaling", {})
        self._use_piggyback_metrics = bool(scaling_cfg.get("use_piggyback_metrics", False))
        self._piggyback_ttl_seconds = int(scaling_cfg.get("piggyback_ttl_seconds", 30))
        self._piggyback_store = None
        if self._use_piggyback_metrics:
            self._piggyback_store = PiggybackMetricsStore(config["redis"])
            logger.info(
                "Piggyback metrics enabled (ttl=%ss)",
                self._piggyback_ttl_seconds,
            )

        # Shared flag: when piggyback is enabled, enable pod-aware load balance too.
        self._lb_enabled = self._use_piggyback_metrics
        self._prefill_router = None
        self._decode_router = None
        if self._lb_enabled:
            lb_cfg = config.get("gateway", {}).get("load_balancer", {})
            refresh_s = float(lb_cfg.get("refresh_interval_s", 5))
            metrics_timeout_s = float(lb_cfg.get("metrics_timeout_s", 1))
            metrics_poll_interval_s = float(lb_cfg.get("metrics_poll_interval_s", 0.2))
            metrics_stale_after_s = float(lb_cfg.get("metrics_stale_after_s", 1.0))
            prefill_lb_host = prefill_cfg.get("lb_host", "prefill-headless")
            decode_lb_host = decode_cfg.get("lb_host", "decode-headless")
            self._prefill_router = StagePodRouter(
                stage="prefill",
                host=prefill_lb_host,
                port=prefill_cfg["port"],
                metrics_timeout_s=metrics_timeout_s,
                refresh_interval_s=refresh_s,
                metrics_poll_interval_s=metrics_poll_interval_s,
                metrics_stale_after_s=metrics_stale_after_s,
            )
            self._decode_router = StagePodRouter(
                stage="decode",
                host=decode_lb_host,
                port=decode_cfg["port"],
                metrics_timeout_s=metrics_timeout_s,
                refresh_interval_s=refresh_s,
                metrics_poll_interval_s=metrics_poll_interval_s,
                metrics_stale_after_s=metrics_stale_after_s,
            )
            logger.info(
                "Pod-aware LB enabled prefill_host=%s decode_host=%s refresh=%.1fs poll=%.2fs stale=%.2fs",
                prefill_lb_host, decode_lb_host, refresh_s, metrics_poll_interval_s, metrics_stale_after_s,
            )

        # Metrics tracking
        self._lock = threading.Lock()
        self._active = 0
        self._total = 0
        self._window_start = time.monotonic()
        self._window_count = 0
        self._prefill_queue = 0
        self._decode_queue = 0
        PREFILL_QUEUE.set(0)
        DECODE_QUEUE.set(0)
        ACTIVE_REQUESTS.set(0)

    def _publish_piggyback(self, stage, worker_metrics):
        """Write piggybacked worker metrics to Redis for controller consumption."""
        if not self._use_piggyback_metrics or self._piggyback_store is None:
            return
        if worker_metrics is None:
            return
        if (
            worker_metrics.active_requests == 0
            and worker_metrics.arrival_rate <= 0
            and worker_metrics.avg_service_time_ms <= 0
            and worker_metrics.queue_length == 0
        ):
            return
        try:
            self._piggyback_store.put(
                stage=stage,
                queue_length=worker_metrics.queue_length,
                arrival_rate=worker_metrics.arrival_rate,
                avg_service_time_ms=worker_metrics.avg_service_time_ms,
                active_requests=worker_metrics.active_requests,
                ttl_seconds=self._piggyback_ttl_seconds,
            )
        except Exception as e:
            logger.debug("Failed to publish piggyback metrics for %s: %s", stage, e)

    def Infer(self, request, context):
        request_id = request.request_id or str(uuid.uuid4())
        t_start = time.perf_counter()
        in_prefill_queue = False
        in_decode_queue = False

        with self._lock:
            self._active += 1
            self._window_count += 1
            self._prefill_queue += 1
            in_prefill_queue = True
            ACTIVE_REQUESTS.set(self._active)
            PREFILL_QUEUE.set(self._prefill_queue)

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
            prefill_stub = self._prefill_stub
            prefill_target_used = "service"
            if self._lb_enabled and self._prefill_router is not None:
                picked_stub, picked_ip, picked_r = self._prefill_router.pick(len(token_ids))
                if picked_stub is not None:
                    prefill_stub = picked_stub
                    prefill_target_used = picked_ip
                    logger.debug(
                        "LB prefill picked pod=%s est_r=%.4fs tokens=%d",
                        picked_ip, picked_r if picked_r is not None else -1.0, len(token_ids),
                    )
            else:
                prefill_stub, prefill_target_used = self._pick_service_stub("prefill")
            try:
                prefill_resp = prefill_stub.RunPrefill(
                    prefill_req, timeout=timeout_s, wait_for_ready=True,
                )
            except grpc.RpcError:
                if prefill_stub is self._prefill_stub:
                    raise
                logger.warning("Prefill target %s failed; fallback to primary service channel", prefill_target_used)
                prefill_resp = self._prefill_stub.RunPrefill(
                    prefill_req, timeout=timeout_s, wait_for_ready=True,
                )
            self._publish_piggyback("prefill", prefill_resp.worker_metrics)
            t_prefill_done = time.perf_counter()

            with self._lock:
                if in_prefill_queue:
                    self._prefill_queue = max(0, self._prefill_queue - 1)
                    in_prefill_queue = False
                self._decode_queue += 1
                in_decode_queue = True
                PREFILL_QUEUE.set(self._prefill_queue)
                DECODE_QUEUE.set(self._decode_queue)

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
                REQUESTS_TOTAL.labels(status="deadline_exceeded").inc()
                context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
                context.set_details("Deadline exceeded after prefill")
                return inference_pb2.InferResponse()

            tokens = [self._tokenizer.decode([prefill_resp.first_token_id], skip_special_tokens=False)]
            tokens_generated = 1

            decode_stub = self._decode_stub
            decode_target_used = "service"
            if self._lb_enabled and self._decode_router is not None:
                picked_stub, picked_ip, picked_r = self._decode_router.pick(max_tokens)
                if picked_stub is not None:
                    decode_stub = picked_stub
                    decode_target_used = picked_ip
                    logger.debug(
                        "LB decode picked pod=%s est_r=%.4fs request_tokens=%d",
                        picked_ip, picked_r if picked_r is not None else -1.0, max_tokens,
                    )
            else:
                decode_stub, decode_target_used = self._pick_service_stub("decode")
            try:
                decode_iter = decode_stub.RunDecode(
                    decode_req, timeout=remaining_s, wait_for_ready=True,
                )
            except grpc.RpcError:
                if decode_stub is self._decode_stub:
                    raise
                logger.warning("Decode target %s failed; fallback to primary service channel", decode_target_used)
                decode_iter = self._decode_stub.RunDecode(
                    decode_req, timeout=remaining_s, wait_for_ready=True,
                )

            for decode_resp in decode_iter:
                tokens.append(decode_resp.token_text)
                tokens_generated = decode_resp.tokens_generated + 1  # +1 for first token from prefill
                if decode_resp.is_finished:
                    self._publish_piggyback("decode", decode_resp.worker_metrics)
                if decode_resp.is_finished:
                    break

            t_done = time.perf_counter()

            with self._lock:
                if in_decode_queue:
                    self._decode_queue = max(0, self._decode_queue - 1)
                    in_decode_queue = False
                    DECODE_QUEUE.set(self._decode_queue)

            generated_text = "".join(tokens)
            queue_wait_ms = (t_queue - t_start) * 1000
            prefill_ms = prefill_resp.prefill_time_ms
            decode_ms = (t_done - t_prefill_done) * 1000
            total_ms = (t_done - t_start) * 1000
            ttft_ms = queue_wait_ms + prefill_ms

            PREFILL_LATENCY.observe(prefill_ms)
            DECODE_LATENCY.observe(decode_ms)
            E2E_LATENCY.observe(total_ms)
            TTFT_LATENCY.observe(ttft_ms)
            TOKENS_GENERATED.inc(tokens_generated)
            REQUESTS_TOTAL.labels(status="ok").inc()

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
            status = e.code().name.lower() if e.code() else "grpc_error"
            REQUESTS_TOTAL.labels(status=status).inc()
            logger.error("Infer %s failed: %s", request_id, e)
            context.set_code(e.code())
            context.set_details(e.details() or str(e))
            return inference_pb2.InferResponse()
        except Exception as e:
            REQUESTS_TOTAL.labels(status="internal_error").inc()
            logger.exception("Infer %s unexpected error: %s", request_id, e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return inference_pb2.InferResponse()
        finally:
            with self._lock:
                if in_prefill_queue:
                    self._prefill_queue = max(0, self._prefill_queue - 1)
                if in_decode_queue:
                    self._decode_queue = max(0, self._decode_queue - 1)
                self._active = max(0, self._active - 1)
                self._total += 1
                PREFILL_QUEUE.set(self._prefill_queue)
                DECODE_QUEUE.set(self._decode_queue)
                ACTIVE_REQUESTS.set(self._active)

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
    prom_cfg = config.get("prometheus", {})
    if prom_cfg.get("enabled", True):
        start_metrics_server(prom_cfg.get("gateway_port", prom_cfg.get("port", 9090)))

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
