"""Tests for common.kv_store — uses mocked Redis."""
import io
import pytest
from unittest.mock import MagicMock, patch

import torch
from common.kv_store import RedisKVStore, KVStoreBase


@pytest.fixture
def mock_redis():
    """Create a RedisKVStore with a mocked Redis client."""
    with patch("common.kv_store.redis.ConnectionPool"):
        with patch("common.kv_store.redis.Redis") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            store = RedisKVStore({"host": "localhost", "port": 6379, "ttl_seconds": 60})
            yield store, mock_client


def _make_kv_cache(num_layers=2, batch=1, heads=4, seq_len=8, head_dim=16):
    """Create a dummy KV cache."""
    return [
        (torch.randn(batch, heads, seq_len, head_dim),
         torch.randn(batch, heads, seq_len, head_dim))
        for _ in range(num_layers)
    ]


class TestRedisKVStore:

    def test_put_returns_key(self, mock_redis):
        store, client = mock_redis
        kv = _make_kv_cache()
        key = store.put("req-1", kv)
        assert key == "kv_cache:req-1"
        client.setex.assert_called_once()
        call_args = client.setex.call_args
        assert call_args[0][0] == "kv_cache:req-1"
        assert call_args[0][1] == 60  # TTL

    def test_put_custom_ttl(self, mock_redis):
        store, client = mock_redis
        kv = _make_kv_cache()
        store.put("req-2", kv, ttl=120)
        call_args = client.setex.call_args
        assert call_args[0][1] == 120

    def test_put_serializes_valid_data(self, mock_redis):
        store, client = mock_redis
        kv = _make_kv_cache(num_layers=3)
        store.put("req-3", kv)
        data = client.setex.call_args[0][2]
        # Verify the serialized data can be deserialized
        buf = io.BytesIO(data)
        loaded = torch.load(buf, weights_only=True)
        assert len(loaded) == 3
        assert all(len(pair) == 2 for pair in loaded)

    def test_get_returns_none_on_miss(self, mock_redis):
        store, client = mock_redis
        client.get.return_value = None
        result = store.get("nonexistent")
        assert result is None

    def test_get_deserializes_correctly(self, mock_redis):
        store, client = mock_redis
        original = _make_kv_cache(num_layers=2, seq_len=4)
        # Serialize manually
        cpu_cache = [(k.cpu(), v.cpu()) for k, v in original]
        buf = io.BytesIO()
        torch.save(cpu_cache, buf)
        client.get.return_value = buf.getvalue()

        result = store.get("req-4", device="cpu")
        assert result is not None
        assert len(result) == 2
        for (ok, ov), (rk, rv) in zip(original, result):
            assert torch.allclose(ok.cpu(), rk, atol=1e-6)
            assert torch.allclose(ov.cpu(), rv, atol=1e-6)

    def test_delete_returns_true_when_existed(self, mock_redis):
        store, client = mock_redis
        client.delete.return_value = 1
        assert store.delete("req-5") is True

    def test_delete_returns_false_when_missing(self, mock_redis):
        store, client = mock_redis
        client.delete.return_value = 0
        assert store.delete("req-6") is False

    def test_exists(self, mock_redis):
        store, client = mock_redis
        client.exists.return_value = 1
        assert store.exists("req-7") is True
        client.exists.return_value = 0
        assert store.exists("req-8") is False

    def test_put_moves_tensors_to_cpu(self, mock_redis):
        store, client = mock_redis
        kv = _make_kv_cache()
        store.put("req-9", kv)
        data = client.setex.call_args[0][2]
        buf = io.BytesIO(data)
        loaded = torch.load(buf, weights_only=True)
        for k, v in loaded:
            assert k.device.type == "cpu"
            assert v.device.type == "cpu"

    def test_key_format(self, mock_redis):
        store, _ = mock_redis
        assert store._key("abc-123") == "kv_cache:abc-123"
        assert store._key("") == "kv_cache:"


class TestKVStoreBase:

    def test_abstract_methods(self):
        with pytest.raises(TypeError):
            KVStoreBase()
