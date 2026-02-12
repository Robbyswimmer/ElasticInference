import io
import time
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
import redis

logger = logging.getLogger(__name__)

# One (K, V) tensor pair per transformer layer
KVCache = List[Tuple[torch.Tensor, torch.Tensor]]


class KVStoreBase(ABC):
    """Abstract interface for KV cache storage."""

    @abstractmethod
    def put(self, request_id: str, kv_cache: KVCache, ttl: Optional[int] = None) -> str:
        """Store a KV cache. Returns the storage key."""

    @abstractmethod
    def get(self, request_id: str, device: str = "cpu") -> Optional[KVCache]:
        """Retrieve a KV cache, moving tensors to `device`."""

    @abstractmethod
    def delete(self, request_id: str) -> bool:
        """Delete a KV cache. Returns True if the key existed."""

    @abstractmethod
    def exists(self, request_id: str) -> bool:
        """Check whether a KV cache exists for the given request."""


class RedisKVStore(KVStoreBase):
    """Redis-backed KV cache store using torch serialization."""

    KEY_PREFIX = "kv_cache:"

    def __init__(self, redis_config: dict):
        self._ttl = redis_config.get("ttl_seconds", 300)
        pool = redis.ConnectionPool(
            host=redis_config.get("host", "localhost"),
            port=redis_config.get("port", 6379),
            db=redis_config.get("db", 0),
            max_connections=redis_config.get("max_connections", 20),
        )
        self._client = redis.Redis(connection_pool=pool)
        logger.info(
            "RedisKVStore connected to %s:%s db=%s",
            redis_config.get("host", "localhost"),
            redis_config.get("port", 6379),
            redis_config.get("db", 0),
        )

    def _key(self, request_id: str) -> str:
        return f"{self.KEY_PREFIX}{request_id}"

    def put(self, request_id: str, kv_cache: KVCache, ttl: Optional[int] = None) -> str:
        key = self._key(request_id)
        ttl = ttl if ttl is not None else self._ttl

        # Move tensors to CPU for serialization
        t0 = time.perf_counter()
        cpu_cache = [(k.cpu(), v.cpu()) for k, v in kv_cache]
        t_cpu = time.perf_counter()

        buf = io.BytesIO()
        torch.save(cpu_cache, buf)
        data = buf.getvalue()
        t_ser = time.perf_counter()

        self._client.setex(key, ttl, data)
        t_write = time.perf_counter()

        logger.debug(
            "PUT %s — cpu_move=%.1fms serialize=%.1fms write=%.1fms size=%dB",
            key,
            (t_cpu - t0) * 1000,
            (t_ser - t_cpu) * 1000,
            (t_write - t_ser) * 1000,
            len(data),
        )
        return key

    def get(self, request_id: str, device: str = "cpu") -> Optional[KVCache]:
        key = self._key(request_id)

        t0 = time.perf_counter()
        data = self._client.get(key)
        t_read = time.perf_counter()

        if data is None:
            logger.debug("GET %s — miss", key)
            return None

        buf = io.BytesIO(data)
        cpu_cache: KVCache = torch.load(buf, weights_only=True)
        t_deser = time.perf_counter()

        result = [(k.to(device), v.to(device)) for k, v in cpu_cache]
        t_move = time.perf_counter()

        logger.debug(
            "GET %s — read=%.1fms deserialize=%.1fms device_move=%.1fms size=%dB",
            key,
            (t_read - t0) * 1000,
            (t_deser - t_read) * 1000,
            (t_move - t_deser) * 1000,
            len(data),
        )
        return result

    def delete(self, request_id: str) -> bool:
        key = self._key(request_id)
        deleted = self._client.delete(key)
        logger.debug("DELETE %s — existed=%s", key, bool(deleted))
        return bool(deleted)

    def exists(self, request_id: str) -> bool:
        return bool(self._client.exists(self._key(request_id)))
