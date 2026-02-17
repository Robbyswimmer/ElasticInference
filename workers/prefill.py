import time
import logging
import threading
from concurrent import futures

import grpc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import load_config
from common.proto import inference_pb2, inference_pb2_grpc
from common.kv_store import RedisKVStore
from common.sampling import apply_sampling
from common.metrics import MetricsTracker

logger = logging.getLogger(__name__)


class PrefillServicer(inference_pb2_grpc.PrefillServiceServicer):

    def __init__(self, config):
        self._config = config
        model_cfg = config["model"]
        device = model_cfg.get("device", "cuda")
        self._device = device if torch.cuda.is_available() else "cpu"
        dtype_str = model_cfg.get("dtype", "float16")
        self._dtype = getattr(torch, dtype_str, torch.float32)

        logger.info("Loading model %s on %s (%s)...", model_cfg["name"], self._device, dtype_str)
        self._tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
        self._model = AutoModelForCausalLM.from_pretrained(
            model_cfg["name"],
            torch_dtype=self._dtype,
        ).to(self._device).eval()
        logger.info("Model loaded.")

        self._kv_store = RedisKVStore(config["redis"])
        self._metrics = MetricsTracker()
        self._max_concurrent = config["prefill"].get("max_concurrent", 4)
        self._semaphore = threading.Semaphore(self._max_concurrent)

    @torch.inference_mode()
    def RunPrefill(self, request, context):
        if not self._semaphore.acquire(timeout=0):
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            context.set_details("Prefill worker at max concurrency")
            return inference_pb2.PrefillResponse()

        self._metrics.start_request()
        t_start = time.perf_counter()

        try:
            token_ids = list(request.token_ids)
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=self._device)

            # Forward pass — produces KV cache + logits
            outputs = self._model(input_ids=input_ids, use_cache=True)
            logits = outputs.logits  # (1, seq_len, vocab_size)
            past_kv = outputs.past_key_values  # tuple of (key, value) per layer

            # Sample first token from last position
            last_logits = logits[0, -1, :].float()
            sp = request.sampling_params
            if sp.temperature > 0 or sp.top_k > 0 or sp.top_p > 0:
                first_token_id = apply_sampling(last_logits, sp, token_ids)
            else:
                first_token_id = torch.argmax(last_logits).item()

            # Convert HF past_key_values to our KVCache format and store
            kv_cache = [(layer_kv[0], layer_kv[1]) for layer_kv in past_kv]
            kv_cache_key = self._kv_store.put(request.request_id, kv_cache)

            elapsed_ms = (time.perf_counter() - t_start) * 1000
            self._metrics.end_request(elapsed_ms)

            logger.info(
                "Prefill %s — %d tokens, first_token=%d, %.1fms",
                request.request_id, len(token_ids), first_token_id, elapsed_ms,
            )
            return inference_pb2.PrefillResponse(
                request_id=request.request_id,
                kv_cache_key=kv_cache_key,
                first_token_id=first_token_id,
                prefill_time_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            self._metrics.end_request(elapsed_ms)
            logger.exception("Prefill failed for %s: %s", request.request_id, e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return inference_pb2.PrefillResponse()
        finally:
            self._semaphore.release()

    def GetMetrics(self, request, context):
        active, arrival_rate, avg_time = self._metrics.snapshot()
        return inference_pb2.WorkerMetrics(
            queue_length=0,
            arrival_rate=arrival_rate,
            avg_service_time_ms=avg_time,
            active_requests=active,
        )


def serve(config=None):
    config = config or load_config()
    log_cfg = config.get("logging", {})
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=log_cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s"),
    )

    prefill_cfg = config["prefill"]
    host = prefill_cfg.get("host", "0.0.0.0")
    port = prefill_cfg.get("port", 50052)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=prefill_cfg.get("max_concurrent", 4)))
    inference_pb2_grpc.add_PrefillServiceServicer_to_server(PrefillServicer(config), server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logger.info("Prefill worker serving on %s:%d", host, port)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down prefill worker...")
        server.stop(grace=5)


if __name__ == "__main__":
    serve()
