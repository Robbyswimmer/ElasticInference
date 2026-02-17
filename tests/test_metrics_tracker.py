"""Tests for the shared MetricsTracker."""
import time
import threading
import pytest

from common.metrics import MetricsTracker as _MetricsTracker


class TestMetricsTracker:

    def test_initial_state(self):
        m = _MetricsTracker()
        active, rate, avg = m.snapshot()
        assert active == 0
        assert avg == 0.0

    def test_single_request(self):
        m = _MetricsTracker()
        m.start_request()
        # While active
        active, _, _ = m.snapshot()
        assert active == 1
        m.end_request(100.0)
        active, _, avg = m.snapshot()
        assert active == 0
        assert avg == 100.0

    def test_multiple_requests(self):
        m = _MetricsTracker()
        m.start_request()
        m.end_request(10.0)
        m.start_request()
        m.end_request(20.0)
        m.start_request()
        m.end_request(30.0)
        active, _, avg = m.snapshot()
        assert active == 0
        assert abs(avg - 20.0) < 0.01  # (10+20+30)/3

    def test_concurrent_requests(self):
        m = _MetricsTracker()
        m.start_request()
        m.start_request()
        m.start_request()
        active, _, _ = m.snapshot()
        assert active == 3
        m.end_request(10.0)
        m.end_request(20.0)
        active, _, _ = m.snapshot()
        assert active == 1
        m.end_request(30.0)
        active, _, _ = m.snapshot()
        assert active == 0

    def test_arrival_rate(self):
        m = _MetricsTracker()
        # Reset window
        m.snapshot()
        time.sleep(0.1)
        m.start_request()
        m.end_request(5.0)
        m.start_request()
        m.end_request(5.0)
        _, rate, _ = m.snapshot()
        # 2 requests in ~0.1s ≈ 20 rps (allow wide tolerance)
        assert rate > 1.0

    def test_window_resets_on_snapshot(self):
        m = _MetricsTracker()
        m.start_request()
        m.end_request(10.0)
        _, rate1, _ = m.snapshot()
        assert rate1 > 0

        # After snapshot, window resets — no new requests
        time.sleep(0.05)
        _, rate2, _ = m.snapshot()
        assert rate2 == 0.0

    def test_thread_safety(self):
        m = _MetricsTracker()
        errors = []

        def worker():
            try:
                for _ in range(100):
                    m.start_request()
                    m.end_request(1.0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        active, _, avg = m.snapshot()
        assert active == 0
        assert abs(avg - 1.0) < 0.01
