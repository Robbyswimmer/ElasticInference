"""Tests for eval.metrics.LatencyCollector."""
import time
import pytest

from eval.metrics import LatencyCollector


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
