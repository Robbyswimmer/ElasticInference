import os
import sys
import time
import logging
import subprocess

import grpc

# Allow importing generated proto stubs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common", "proto"))

import inference_pb2
import inference_pb2_grpc

from common import load_config

logger = logging.getLogger(__name__)


class ScalingController:
    """Polls worker metrics and scales deployments via kubectl."""

    def __init__(self, config):
        self._config = config
        scaling = config["scaling"]
        self._interval = scaling.get("metrics_interval", 5)
        self._cooldown = scaling.get("cooldown", 30)
        self._ema_alpha = scaling.get("ema_alpha", 0.3)

        self._stages = {}
        for stage in ("prefill", "decode"):
            s_cfg = scaling[stage]
            port = config[stage]["port"]
            host = config[stage].get("host", "localhost")
            self._stages[stage] = {
                "target": f"{host}:{port}",
                "min_replicas": s_cfg["min_replicas"],
                "max_replicas": s_cfg["max_replicas"],
                "scale_up_threshold": s_cfg["scale_up_threshold"],
                "scale_down_threshold": s_cfg["scale_down_threshold"],
                "ema_load": 0.0,
                "last_scale_time": 0.0,
                "current_replicas": 1,
            }

    def _get_metrics(self, stage):
        """Poll GetMetrics RPC for a worker stage."""
        info = self._stages[stage]
        try:
            channel = grpc.insecure_channel(info["target"])
            if stage == "prefill":
                stub = inference_pb2_grpc.PrefillServiceStub(channel)
            else:
                stub = inference_pb2_grpc.DecodeServiceStub(channel)
            resp = stub.GetMetrics(inference_pb2.MetricsRequest(), timeout=2)
            channel.close()
            return resp
        except grpc.RpcError as e:
            logger.warning("Failed to get metrics for %s: %s", stage, e)
            return None

    def _compute_load(self, metrics, stage):
        """Compute normalized load signal (0-1) from worker metrics."""
        if stage == "prefill":
            capacity = self._config["prefill"].get("max_concurrent", 4)
        else:
            capacity = self._config["decode"].get("max_batch_size", 32)
        return metrics.active_requests / max(capacity, 1)

    def _scale(self, stage, replicas):
        """Scale a deployment via kubectl."""
        replicas = int(replicas)
        info = self._stages[stage]
        replicas = max(info["min_replicas"], min(info["max_replicas"], replicas))
        if replicas == info["current_replicas"]:
            return
        logger.info("Scaling %s: %d -> %d replicas", stage, info["current_replicas"], replicas)
        try:
            subprocess.run(
                ["kubectl", "-n", "llm-inference", "scale", "deployment",
                 stage, f"--replicas={replicas}"],
                check=True, capture_output=True, text=True,
            )
            info["current_replicas"] = replicas
            info["last_scale_time"] = time.monotonic()
        except subprocess.CalledProcessError as e:
            logger.error("kubectl scale failed for %s: %s", stage, e.stderr)

    def _step(self):
        """One control loop iteration."""
        now = time.monotonic()
        for stage, info in self._stages.items():
            metrics = self._get_metrics(stage)
            if metrics is None:
                continue

            load = self._compute_load(metrics, stage)
            alpha = self._ema_alpha
            info["ema_load"] = alpha * load + (1 - alpha) * info["ema_load"]
            ema = info["ema_load"]

            # Cooldown check
            if now - info["last_scale_time"] < self._cooldown:
                continue

            current = info["current_replicas"]
            if ema > info["scale_up_threshold"]:
                self._scale(stage, current + 1)
            elif ema < info["scale_down_threshold"] and current > info["min_replicas"]:
                self._scale(stage, current - 1)

            logger.debug(
                "%s: load=%.2f ema=%.2f replicas=%d",
                stage, load, ema, info["current_replicas"],
            )

    def run(self):
        """Main control loop."""
        logger.info("Scaling controller started (interval=%ds cooldown=%ds)",
                     self._interval, self._cooldown)
        while True:
            try:
                self._step()
            except Exception:
                logger.exception("Controller step failed")
            time.sleep(self._interval)


def main():
    config = load_config()
    log_cfg = config.get("logging", {})
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=log_cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s"),
    )
    ScalingController(config).run()


if __name__ == "__main__":
    main()
