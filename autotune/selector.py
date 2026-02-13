import logging

from common import load_config

logger = logging.getLogger(__name__)


class ConfigSelector:
    """Select runtime configuration based on current GPU% and workload.

    Maps (gpu_pct, workload_intensity) -> (batch_size, concurrency) for each stage.
    Uses profile data from KneeProfiler to build the mapping.
    """

    def __init__(self, config=None):
        self._config = config or load_config()
        autotune = self._config.get("autotune", {})
        self._enabled = autotune.get("enabled", False)
        self._config_space = autotune.get("config_space", {})
        # Lookup table: (stage, gpu_pct) -> best config
        self._table = {}

    @property
    def enabled(self):
        return self._enabled

    def update_from_profile(self, stage, profile_results):
        """Build config lookup from profiling results.

        For each GPU%, pick the config that achieved best latency.
        """
        for result in profile_results:
            pct = result["gpu_pct"]
            self._table[(stage, pct)] = {
                "gpu_pct": pct,
                "avg_latency_ms": result.get("avg_latency_ms"),
                "batch_size": result.get("best_batch_size"),
            }
            logger.info("ConfigSelector: %s @ %d%% -> %s", stage, pct, self._table[(stage, pct)])

    def select(self, stage, gpu_pct):
        """Select best config for a stage at a given GPU%.

        Falls back to the nearest profiled GPU% if exact match not found.
        """
        if not self._enabled:
            # Return defaults from config
            if stage == "prefill":
                return {"batch_size": self._config["prefill"].get("batch_size", 8)}
            else:
                return {"batch_size": self._config["decode"].get("max_batch_size", 32)}

        # Exact match
        if (stage, gpu_pct) in self._table:
            return self._table[(stage, gpu_pct)]

        # Nearest GPU%
        candidates = [(k, v) for k, v in self._table.items() if k[0] == stage]
        if not candidates:
            logger.warning("No profile data for %s, using defaults", stage)
            return self.select.__wrapped__(self, stage, gpu_pct) if hasattr(self.select, '__wrapped__') else {}

        nearest = min(candidates, key=lambda x: abs(x[0][1] - gpu_pct))
        logger.debug("ConfigSelector: %s @ %d%% -> nearest match %d%%", stage, gpu_pct, nearest[0][1])
        return nearest[1]

    def get_config_space(self, stage):
        """Return the tunable parameter space for a stage."""
        if stage == "prefill":
            return self._config_space.get("prefill_batch_size", [1, 2, 4, 8, 16])
        else:
            return self._config_space.get("decode_batch_size", [4, 8, 16, 32, 64])
