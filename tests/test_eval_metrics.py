"""Tests for eval.metrics.LatencyCollector and ScalingStabilityAnalyzer."""
import time
import pytest

from eval.metrics import LatencyCollector, ScalingStabilityAnalyzer


class TestLatencyCollector:

    def test_empty_summary(self):
        c = LatencyCollector()
        assert c.summary() == {}

    def test_single_record(self):
        c = LatencyCollector()
        c.record("r1", queue_wait_ms=1.0, prefill_ms=10.0, decode_ms=20.0,
                 total_ms=31.0, tokens_generated=50, ttft_ms=11.0)
        s = c.summary()
        assert s["num_requests"] == 1
        assert s["total_tokens"] == 50
        assert s["e2e_p50_ms"] == 31.0
        assert s["e2e_p95_ms"] == 31.0
        assert s["ttft_p50_ms"] == 11.0

    def test_multiple_records_percentiles(self):
        c = LatencyCollector()
        # Insert 100 records with increasing latency
        for i in range(100):
            c.record(f"r{i}", queue_wait_ms=1.0, prefill_ms=10.0,
                     decode_ms=float(i), total_ms=float(i + 11),
                     tokens_generated=10)
        s = c.summary()
        assert s["num_requests"] == 100
        assert s["total_tokens"] == 1000
        # p50 ≈ 61, p95 ≈ 106, p99 ≈ 110
        assert 55 <= s["e2e_p50_ms"] <= 65
        assert 100 <= s["e2e_p95_ms"] <= 110

    def test_ttft_only_from_records_with_ttft(self):
        c = LatencyCollector()
        c.record("r1", 1, 10, 20, 31, 50, ttft_ms=11.0)
        c.record("r2", 1, 10, 20, 31, 50, ttft_ms=None)
        c.record("r3", 1, 10, 20, 31, 50, ttft_ms=15.0)
        s = c.summary()
        assert "ttft_p50_ms" in s
        # Only 2 records have TTFT

    def test_throughput(self):
        c = LatencyCollector()
        c.record("r1", 1, 10, 20, 31, 50)
        time.sleep(0.1)
        c.record("r2", 1, 10, 20, 31, 50)
        s = c.summary()
        assert s["throughput_rps"] > 0
        assert s["throughput_tps"] > 0

    def test_records_property(self):
        c = LatencyCollector()
        c.record("r1", 1, 10, 20, 31, 50)
        c.record("r2", 2, 12, 22, 36, 60)
        records = c.records
        assert len(records) == 2
        assert records[0]["request_id"] == "r1"
        assert records[1]["request_id"] == "r2"

    def test_reset(self):
        c = LatencyCollector()
        c.record("r1", 1, 10, 20, 31, 50)
        assert len(c.records) == 1
        c.reset()
        assert len(c.records) == 0
        assert c.summary() == {}

    def test_decode_percentiles(self):
        c = LatencyCollector()
        for i in range(50):
            c.record(f"r{i}", 1, 10, decode_ms=float(i * 2),
                     total_ms=float(i * 2 + 11), tokens_generated=10)
        s = c.summary()
        assert s["decode_p50_ms"] > 0
        assert s["decode_p95_ms"] > s["decode_p50_ms"]

    def test_gpu_efficiency_metrics(self):
        c = LatencyCollector(gpu_count=2, gpu_pct=50)
        c.record("r1", 1, 10, 20, 100, 50)
        time.sleep(0.05)
        c.record("r2", 1, 10, 20, 200, 50)
        s = c.summary()
        assert "gpu_seconds_per_request" in s
        assert "gpu_seconds_total" in s
        assert "gpu_utilization_pct" in s
        # 100ms * 0.5 * 2 gpus = 0.1 gpu-seconds for r1
        # 200ms * 0.5 * 2 gpus = 0.2 gpu-seconds for r2
        # avg = 0.15
        assert abs(s["gpu_seconds_per_request"] - 0.15) < 0.01
        assert s["gpu_utilization_pct"] > 0

    def test_gpu_efficiency_defaults(self):
        c = LatencyCollector()  # defaults: 1 GPU, 100%
        c.record("r1", 1, 10, 20, 1000, 50)
        s = c.summary()
        # 1000ms * 1.0 * 1 = 1.0 gpu-seconds
        assert abs(s["gpu_seconds_per_request"] - 1.0) < 0.01


class TestScalingStabilityAnalyzer:

    def test_empty_events(self):
        a = ScalingStabilityAnalyzer([])
        s = a.analyze()
        assert s["total_scaling_events"] == 0
        assert s["oscillation_frequency_per_min"] == 0.0

    def test_single_event(self):
        events = [{"timestamp": 100.0, "direction": "up", "stage": "prefill",
                    "from_replicas": 1, "to_replicas": 2, "ema_load": 0.9}]
        s = ScalingStabilityAnalyzer(events).analyze()
        assert s["total_scaling_events"] == 1
        assert s["scale_up_events"] == 1
        assert s["scale_down_events"] == 0

    def test_oscillation_detected(self):
        t = 1000.0
        events = []
        for i, d in enumerate(["up", "down", "up", "down", "up", "down"]):
            events.append({"timestamp": t + i * 10, "direction": d, "stage": "decode",
                           "from_replicas": 1, "to_replicas": 2, "ema_load": 0.5})
        s = ScalingStabilityAnalyzer(events).analyze()
        assert s["total_scaling_events"] == 6
        assert s["direction_change_ratio"] > 0.8  # almost every event is a change
        assert s["oscillation_frequency_per_min"] > 0

    def test_stable_scaling(self):
        t = 1000.0
        events = []
        for i in range(5):
            events.append({"timestamp": t + i * 30, "direction": "up", "stage": "prefill",
                           "from_replicas": i + 1, "to_replicas": i + 2, "ema_load": 0.9})
        s = ScalingStabilityAnalyzer(events).analyze()
        assert s["direction_change_ratio"] == 0.0  # no direction changes
        assert s["oscillation_frequency_per_min"] == 0.0

    def test_convergence_time_with_burst_gap(self):
        events = [
            {"timestamp": 100.0, "direction": "up", "stage": "prefill",
             "from_replicas": 1, "to_replicas": 2, "ema_load": 0.9},
            {"timestamp": 110.0, "direction": "up", "stage": "prefill",
             "from_replicas": 2, "to_replicas": 3, "ema_load": 0.85},
            # 120s gap -> new burst
            {"timestamp": 230.0, "direction": "down", "stage": "prefill",
             "from_replicas": 3, "to_replicas": 2, "ema_load": 0.2},
        ]
        s = ScalingStabilityAnalyzer(events).analyze()
        # First burst: 110-100=10s, Second burst: single event = 0s
        assert s["avg_convergence_time_s"] == 5.0  # (10+0)/2
