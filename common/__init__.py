import os
import yaml
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cast_int_env(value):
    """Cast numeric env vars and tolerate K8s service-link style tcp URLs."""
    try:
        return int(value)
    except ValueError:
        parsed = urlparse(value)
        if parsed.scheme and parsed.port is not None:
            return parsed.port
        raise


def _cast_bool_env(value):
    value = str(value).strip().lower()
    return value in {"1", "true", "yes", "on"}

def _cast_float_env(value):
    return float(value)


def load_config(path=None):
    """Load configuration from YAML with environment variable overrides.

    Resolution order for config file path:
        1. Explicit `path` argument
        2. CONFIG_PATH environment variable
        3. <project_root>/config.yaml
    """
    if path is None:
        path = os.environ.get("CONFIG_PATH", os.path.join(_PROJECT_ROOT, "config.yaml"))

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Environment variable overrides for K8s deployments
    _env_overrides = {
        "REDIS_HOST":   ("redis", "host"),
        "REDIS_PORT":   ("redis", "port", _cast_int_env),
        "MODEL_NAME":   ("model", "name"),
        "LOG_LEVEL":    ("logging", "level"),
        "GATEWAY_PORT": ("gateway", "port", _cast_int_env),
        "PREFILL_HOST": ("prefill", "connect_host"),
        "PREFILL_PORT": ("prefill", "port", _cast_int_env),
        "DECODE_HOST":  ("decode", "connect_host"),
        "DECODE_PORT":  ("decode", "port", _cast_int_env),
        "USE_PIGGYBACK_METRICS": ("scaling", "use_piggyback_metrics", _cast_bool_env),
        "PIGGYBACK_FALLBACK_TO_RPC": ("scaling", "piggyback_fallback_to_rpc", _cast_bool_env),
        "PIGGYBACK_TTL_SECONDS": ("scaling", "piggyback_ttl_seconds", _cast_int_env),
        "SCALING_METRICS_INTERVAL": ("scaling", "metrics_interval", _cast_float_env),
    }

    for env_var, spec in _env_overrides.items():
        value = os.environ.get(env_var)
        if value is not None:
            section, key = spec[0], spec[1]
            cast = spec[2] if len(spec) > 2 else str
            config.setdefault(section, {})[key] = cast(value)
            logger.debug("Override %s.%s = %s (from %s)", section, key, value, env_var)

    return config
