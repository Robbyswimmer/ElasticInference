"""Tests for common.load_config."""
import os
import tempfile
import pytest
import yaml

from common import load_config


def test_load_default_config():
    config = load_config()
    assert config["model"]["name"] == "facebook/opt-1.3b"
    assert config["model"]["dtype"] == "float16"
    assert config["model"]["max_seq_len"] == 2048
    assert config["gateway"]["port"] == 50051
    assert config["prefill"]["port"] == 50052
    assert config["decode"]["port"] == 50053
    assert config["redis"]["host"] == "localhost"
    assert config["redis"]["ttl_seconds"] == 300


def test_load_explicit_path():
    data = {"model": {"name": "test-model"}, "redis": {"host": "testhost"}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        path = f.name
    try:
        config = load_config(path)
        assert config["model"]["name"] == "test-model"
        assert config["redis"]["host"] == "testhost"
    finally:
        os.unlink(path)


def test_env_var_config_path():
    data = {"model": {"name": "env-model"}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        path = f.name
    try:
        os.environ["CONFIG_PATH"] = path
        config = load_config()
        assert config["model"]["name"] == "env-model"
    finally:
        del os.environ["CONFIG_PATH"]
        os.unlink(path)


def test_env_overrides():
    os.environ["REDIS_HOST"] = "redis-cluster"
    os.environ["REDIS_PORT"] = "6380"
    os.environ["MODEL_NAME"] = "facebook/opt-6.7b"
    os.environ["GATEWAY_PORT"] = "9999"
    try:
        config = load_config()
        assert config["redis"]["host"] == "redis-cluster"
        assert config["redis"]["port"] == 6380
        assert config["model"]["name"] == "facebook/opt-6.7b"
        assert config["gateway"]["port"] == 9999
    finally:
        del os.environ["REDIS_HOST"]
        del os.environ["REDIS_PORT"]
        del os.environ["MODEL_NAME"]
        del os.environ["GATEWAY_PORT"]


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")


def test_config_has_all_sections():
    config = load_config()
    required = ["model", "gateway", "prefill", "decode", "redis", "scaling", "autotune", "logging", "prometheus"]
    for section in required:
        assert section in config, f"Missing config section: {section}"


def test_scaling_config_structure():
    config = load_config()
    scaling = config["scaling"]
    assert scaling["metrics_interval"] > 0
    assert scaling["cooldown"] > 0
    assert 0 < scaling["ema_alpha"] < 1
    for stage in ("prefill", "decode"):
        assert scaling[stage]["min_replicas"] >= 1
        assert scaling[stage]["max_replicas"] >= scaling[stage]["min_replicas"]
        assert 0 < scaling[stage]["scale_up_threshold"] <= 1
        assert 0 < scaling[stage]["scale_down_threshold"] <= 1
