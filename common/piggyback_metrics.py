"""Redis-backed store for piggybacked worker metrics."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional

import redis


@dataclass
class WorkerMetricsSnapshot:
    """Normalized metrics payload consumed by scaling controller."""

    queue_length: int
    arrival_rate: float
    avg_service_time_ms: float
    active_requests: int
    timestamp: float


class PiggybackMetricsStore:
    """Read/write latest worker metrics snapshots by stage."""

    KEY_PREFIX = "piggyback_metrics:"

    def __init__(self, redis_config: dict):
        pool = redis.ConnectionPool(
            host=redis_config.get("host", "localhost"),
            port=redis_config.get("port", 6379),
            db=redis_config.get("db", 0),
            max_connections=redis_config.get("max_connections", 20),
            decode_responses=True,
        )
        self._client = redis.Redis(connection_pool=pool)

    def _key(self, stage: str) -> str:
        return f"{self.KEY_PREFIX}{stage}"

    def put(self, stage: str, queue_length: int, arrival_rate: float, avg_service_time_ms: float,
            active_requests: int, ttl_seconds: int = 30) -> None:
        payload = {
            "queue_length": int(queue_length),
            "arrival_rate": float(arrival_rate),
            "avg_service_time_ms": float(avg_service_time_ms),
            "active_requests": int(active_requests),
            "timestamp": float(time.time()),
        }
        self._client.setex(self._key(stage), int(ttl_seconds), json.dumps(payload))

    def get(self, stage: str) -> Optional[WorkerMetricsSnapshot]:
        raw = self._client.get(self._key(stage))
        if not raw:
            return None
        try:
            data = json.loads(raw)
            return WorkerMetricsSnapshot(
                queue_length=int(data.get("queue_length", 0)),
                arrival_rate=float(data.get("arrival_rate", 0.0)),
                avg_service_time_ms=float(data.get("avg_service_time_ms", 0.0)),
                active_requests=int(data.get("active_requests", 0)),
                timestamp=float(data.get("timestamp", 0.0)),
            )
        except Exception:
            return None
