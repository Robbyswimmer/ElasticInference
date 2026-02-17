"""Smoke test for all project modules."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

def test_config():
    from common import load_config
    config = load_config()
    assert "model" in config
    assert "redis" in config
    assert "gateway" in config
    assert "prefill" in config
    assert "decode" in config
    assert "scaling" in config
    assert config["model"]["name"] == "facebook/opt-1.3b"
    print("  config: OK")

def test_proto():
    from common.proto import inference_pb2
    req = inference_pb2.InferRequest(request_id="test-001", prompt="Hello", max_tokens=64)
    assert req.request_id == "test-001"
    sp = inference_pb2.SamplingParams(temperature=0.7, top_p=0.9, top_k=50)
    assert sp.temperature - 0.7 < 0.01
    print("  proto: OK")

def test_kv_store_import():
    from common.kv_store import KVStoreBase
    assert KVStoreBase is not None
    print("  kv_store: OK")

def test_workers_import():
    from workers.prefill import PrefillServicer
    from workers.decode import DecodeServicer
    from common.proto import inference_pb2_grpc
    assert issubclass(PrefillServicer, inference_pb2_grpc.PrefillServiceServicer)
    assert issubclass(DecodeServicer, inference_pb2_grpc.DecodeServiceServicer)
    print("  workers: OK")

def test_gateway_import():
    from gateway.server import GatewayServicer
    from common.proto import inference_pb2_grpc
    assert issubclass(GatewayServicer, inference_pb2_grpc.GatewayServiceServicer)
    print("  gateway: OK")

def test_scaling_import():
    from scaling.controller import ScalingController
    assert ScalingController is not None
    print("  scaling: OK")

def test_eval_import():
    from eval.metrics import LatencyCollector
    c = LatencyCollector()
    c.record("r1", 1.0, 10.0, 20.0, 31.0, 50, ttft_ms=11.0)
    c.record("r2", 2.0, 12.0, 22.0, 36.0, 60, ttft_ms=14.0)
    s = c.summary()
    assert s["num_requests"] == 2
    assert s["total_tokens"] == 110
    print("  eval: OK")

def test_sampling():
    import torch
    from common.sampling import apply_sampling
    from common.proto import inference_pb2
    logits = torch.randn(50257)
    sp = inference_pb2.SamplingParams(temperature=0.8, top_p=0.9, top_k=50, repetition_penalty=1.1)
    token = apply_sampling(logits.clone(), sp, [1, 2, 3])
    assert isinstance(token, int) and 0 <= token < 50257
    print("  sampling: OK")

def test_metrics_tracker():
    from common.metrics import MetricsTracker
    m = MetricsTracker()
    m.start_request()
    m.end_request(42.5)
    active, rate, avg = m.snapshot()
    assert active == 0
    assert avg == 42.5
    print("  metrics_tracker: OK")

def test_gpu_utils():
    from common.gpu_utils import get_gpu_count
    count = get_gpu_count()
    assert isinstance(count, int)
    print(f"  gpu_utils: OK (GPUs={count})")

def test_autotune_import():
    from autotune.selector import ConfigSelector
    sel = ConfigSelector()
    result = sel.select("prefill", 50)
    assert "batch_size" in result
    print("  autotune: OK")


if __name__ == "__main__":
    print("Running smoke tests...")
    tests = [
        test_config,
        test_proto,
        test_kv_store_import,
        test_workers_import,
        test_gateway_import,
        test_scaling_import,
        test_eval_import,
        test_sampling,
        test_metrics_tracker,
        test_gpu_utils,
        test_autotune_import,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} tests passed")
    sys.exit(1 if failed else 0)
