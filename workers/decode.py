import os
import sys
import time
import logging
import threading
from concurrent import futures

import grpc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow importing generated proto stubs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common", "proto"))

import inference_pb2
import inference_pb2_grpc

from common import load_config
from common.kv_store import RedisKVStore

logger = logging.getLogger(__name__)


def _apply_sampling(logits, sampling_params, generated_ids=None):
    """Apply sampling parameters to logits and return a sampled token id."""
    # Repetition penalty
    if generated_ids and sampling_params.repetition_penalty > 1.0:
        for token_id in set(generated_ids):
            if logits[token_id] > 0:
                logits[token_id] /= sampling_params.repetition_penalty
            else:
                logits[token_id] *= sampling_params.repetition_penalty

    # Temperature
    temp = sampling_params.temperature if sampling_params.temperature > 0 else 1.0
    logits = logits / temp

    # Top-k filtering
    if sampling_params.top_k > 0:
        top_k = min(sampling_params.top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k).values[-1]
        logits[logits < threshold] = float("-inf")

    # Top-p (nucleus) filtering
    if 0.0 < sampling_params.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        cutoff_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= sampling_params.top_p
        sorted_logits[cutoff_mask] = float("-inf")
        logits = sorted_logits.scatter(0, sorted_indices, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


class _MetricsTracker:
    """Thread-safe request metrics for GetMetrics RPC."""

    def __init__(self):
        self._lock = threading.Lock()
        self._active = 0
        self._total = 0
        self._total_time_ms = 0.0
        self._window_start = time.monotonic()
        self._window_count = 0

    def start_request(self):
        with self._lock:
            self._active += 1
            self._window_count += 1

    def end_request(self, duration_ms):
        with self._lock:
            self._active -= 1
            self._total += 1
            self._total_time_ms += duration_ms

    def snapshot(self):
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._window_start
            arrival_rate = self._window_count / elapsed if elapsed > 0 else 0.0
            avg_time = self._total_time_ms / self._total if self._total > 0 else 0.0
            # Reset window
            self._window_start = now
            self._window_count = 0
            return self._active, arrival_rate, avg_time


class DecodeServicer(inference_pb2_grpc.DecodeServiceServicer):

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

        self._eos_token_id = self._tokenizer.eos_token_id
        self._kv_store = RedisKVStore(config["redis"])
        self._metrics = _MetricsTracker()
        self._max_tokens_default = config["decode"].get("max_new_tokens", 512)

    @torch.inference_mode()
    def RunDecode(self, request, context):
        self._metrics.start_request()
        t_start = time.perf_counter()
        tokens_generated = 0

        try:
            # Retrieve KV cache from Redis
            kv_cache = self._kv_store.get(request.request_id, device=self._device)
            if kv_cache is None:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"KV cache not found for {request.request_id}")
                self._metrics.end_request((time.perf_counter() - t_start) * 1000)
                return

            # Convert list of (K, V) tuples back to HuggingFace DynamicCache-compatible format
            past_key_values = tuple(kv_cache)

            max_tokens = request.max_tokens if request.max_tokens > 0 else self._max_tokens_default
            current_token_id = request.first_token_id
            generated_ids = [current_token_id]

            for step in range(max_tokens):
                t_step = time.perf_counter()

                input_ids = torch.tensor([[current_token_id]], dtype=torch.long, device=self._device)
                outputs = self._model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values

                # Sample next token
                last_logits = outputs.logits[0, -1, :].float()
                sp = request.sampling_params
                if sp.temperature > 0 or sp.top_k > 0 or sp.top_p > 0:
                    next_token_id = _apply_sampling(last_logits, sp, generated_ids)
                else:
                    next_token_id = torch.argmax(last_logits).item()

                tokens_generated = step + 1
                is_finished = (next_token_id == self._eos_token_id) or (tokens_generated >= max_tokens)
                token_text = self._tokenizer.decode([next_token_id], skip_special_tokens=False)
                step_ms = (time.perf_counter() - t_step) * 1000

                yield inference_pb2.DecodeResponse(
                    request_id=request.request_id,
                    token_id=next_token_id,
                    token_text=token_text,
                    is_finished=is_finished,
                    tokens_generated=tokens_generated,
                    decode_time_ms=step_ms,
                )

                if is_finished:
                    break

                current_token_id = next_token_id
                generated_ids.append(next_token_id)

            elapsed_ms = (time.perf_counter() - t_start) * 1000
            self._metrics.end_request(elapsed_ms)

            # Clean up KV cache from Redis
            self._kv_store.delete(request.request_id)
            logger.info(
                "Decode %s — %d tokens, %.1fms total",
                request.request_id, tokens_generated, elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            self._metrics.end_request(elapsed_ms)
            self._kv_store.delete(request.request_id)
            logger.exception("Decode failed for %s: %s", request.request_id, e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))

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

    decode_cfg = config["decode"]
    host = decode_cfg.get("host", "0.0.0.0")
    port = decode_cfg.get("port", 50053)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=decode_cfg.get("max_batch_size", 32)))
    inference_pb2_grpc.add_DecodeServiceServicer_to_server(DecodeServicer(config), server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logger.info("Decode worker serving on %s:%d", host, port)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down decode worker...")
        server.stop(grace=5)


if __name__ == "__main__":
    serve()
