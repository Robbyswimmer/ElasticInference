"""Tests for protobuf message contracts."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common", "proto"))

import inference_pb2
import inference_pb2_grpc


class TestSamplingParams:

    def test_defaults_are_zero(self):
        sp = inference_pb2.SamplingParams()
        assert sp.temperature == 0.0
        assert sp.top_p == 0.0
        assert sp.top_k == 0
        assert sp.repetition_penalty == 0.0

    def test_set_values(self):
        sp = inference_pb2.SamplingParams(
            temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.2
        )
        assert abs(sp.temperature - 0.7) < 1e-5
        assert abs(sp.top_p - 0.9) < 1e-5
        assert sp.top_k == 50
        assert abs(sp.repetition_penalty - 1.2) < 1e-5


class TestInferRequest:

    def test_fields(self):
        req = inference_pb2.InferRequest(
            request_id="r-001", prompt="Hello world", max_tokens=128, deadline_ms=5000
        )
        assert req.request_id == "r-001"
        assert req.prompt == "Hello world"
        assert req.max_tokens == 128
        assert req.deadline_ms == 5000

    def test_nested_sampling_params(self):
        sp = inference_pb2.SamplingParams(temperature=0.5)
        req = inference_pb2.InferRequest(
            request_id="r-002", prompt="test", sampling_params=sp
        )
        assert abs(req.sampling_params.temperature - 0.5) < 1e-5

    def test_serialization_roundtrip(self):
        req = inference_pb2.InferRequest(
            request_id="r-003", prompt="roundtrip", max_tokens=64
        )
        data = req.SerializeToString()
        req2 = inference_pb2.InferRequest()
        req2.ParseFromString(data)
        assert req2.request_id == "r-003"
        assert req2.prompt == "roundtrip"
        assert req2.max_tokens == 64


class TestPrefillMessages:

    def test_request(self):
        req = inference_pb2.PrefillRequest(
            request_id="p-001", token_ids=[1, 2, 3, 4], model_name="opt-1.3b"
        )
        assert list(req.token_ids) == [1, 2, 3, 4]
        assert req.model_name == "opt-1.3b"

    def test_response(self):
        resp = inference_pb2.PrefillResponse(
            request_id="p-001", kv_cache_key="kv_cache:p-001",
            first_token_id=42, prefill_time_ms=15.5
        )
        assert resp.kv_cache_key == "kv_cache:p-001"
        assert resp.first_token_id == 42
        assert abs(resp.prefill_time_ms - 15.5) < 0.1


class TestDecodeMessages:

    def test_request(self):
        req = inference_pb2.DecodeRequest(
            request_id="d-001", kv_cache_key="kv_cache:d-001",
            first_token_id=42, max_tokens=256
        )
        assert req.kv_cache_key == "kv_cache:d-001"
        assert req.max_tokens == 256

    def test_response_streaming_fields(self):
        resp = inference_pb2.DecodeResponse(
            request_id="d-001", token_id=100, token_text="hello",
            is_finished=False, tokens_generated=5, decode_time_ms=3.2
        )
        assert resp.token_id == 100
        assert resp.token_text == "hello"
        assert resp.is_finished is False
        assert resp.tokens_generated == 5

    def test_finished_response(self):
        resp = inference_pb2.DecodeResponse(is_finished=True, tokens_generated=50)
        assert resp.is_finished is True


class TestLatencyBreakdown:

    def test_all_fields(self):
        lb = inference_pb2.LatencyBreakdown(
            queue_wait_ms=2.5, prefill_ms=15.0, decode_ms=80.0, total_ms=97.5
        )
        assert abs(lb.total_ms - 97.5) < 0.1


class TestWorkerMetrics:

    def test_all_fields(self):
        wm = inference_pb2.WorkerMetrics(
            queue_length=5, arrival_rate=10.0, avg_service_time_ms=25.0, active_requests=3
        )
        assert wm.queue_length == 5
        assert wm.active_requests == 3


class TestServiceStubs:

    def test_prefill_servicer_has_methods(self):
        servicer = inference_pb2_grpc.PrefillServiceServicer()
        assert hasattr(servicer, "RunPrefill")
        assert hasattr(servicer, "GetMetrics")

    def test_decode_servicer_has_methods(self):
        servicer = inference_pb2_grpc.DecodeServiceServicer()
        assert hasattr(servicer, "RunDecode")
        assert hasattr(servicer, "GetMetrics")

    def test_gateway_servicer_has_methods(self):
        servicer = inference_pb2_grpc.GatewayServiceServicer()
        assert hasattr(servicer, "Infer")
        assert hasattr(servicer, "GetMetrics")
