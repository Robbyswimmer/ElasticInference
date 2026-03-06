"""Shared worker metrics tracker used by prefill and decode services."""
import time
import threading


class MetricsTracker:
    """Thread-safe request metrics for GetMetrics RPC."""

    def __init__(self):
        self._lock = threading.Lock()
        self._active = 0
        self._total = 0
        self._total_time_ms = 0.0
        self._window_start = time.monotonic()
        self._window_count = 0
        self._window_total = 0
        self._window_time_ms = 0.0

    def start_request(self):
        with self._lock:
            self._active += 1
            self._window_count += 1

    def end_request(self, duration_ms):
        with self._lock:
            self._active -= 1
            self._total += 1
            self._total_time_ms += duration_ms
            self._window_total += 1
            self._window_time_ms += duration_ms

    def snapshot(self, reset=True):
        """Return (active_requests, arrival_rate, avg_service_time_ms).

        Args:
            reset: when True, reset the rolling window after reading.
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._window_start
            arrival_rate = self._window_count / elapsed if elapsed > 0 else 0.0
            avg_time = self._window_time_ms / self._window_total if self._window_total > 0 else 0.0
            if reset:
                # Reset window
                self._window_start = now
                self._window_count = 0
                self._window_total = 0
                self._window_time_ms = 0.0
            return self._active, arrival_rate, avg_time

    def peek(self):
        """Read metrics without resetting the rolling window."""
        return self.snapshot(reset=False)
