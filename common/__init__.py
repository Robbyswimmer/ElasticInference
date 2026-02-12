import os
import yaml
import logging

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
        "REDIS_PORT":   ("redis", "port", int),
        "MODEL_NAME":   ("model", "name"),
        "LOG_LEVEL":    ("logging", "level"),
        "GATEWAY_PORT": ("gateway", "port", int),
        "PREFILL_PORT": ("prefill", "port", int),
        "DECODE_PORT":  ("decode", "port", int),
    }

    for env_var, spec in _env_overrides.items():
        value = os.environ.get(env_var)
        if value is not None:
            section, key = spec[0], spec[1]
            cast = spec[2] if len(spec) > 2 else str
            config.setdefault(section, {})[key] = cast(value)
            logger.debug("Override %s.%s = %s (from %s)", section, key, value, env_var)

    return config
