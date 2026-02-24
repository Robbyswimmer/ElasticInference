import time
import json
import logging
import subprocess
from collections import deque

import grpc

from common import load_config
from common.proto import inference_pb2, inference_pb2_grpc
from autotune.selector import ConfigSelector

logger = logging.getLogger(__name__)


class ScalingController:
    """Multi-metric scaling controller with oscillation detection.

    Two-level control loop:
      Outer: determines replica count per worker stage
      Inner: determines GPU% and runtime params (via autotune selector)

    Scaling signals (weighted combination):
      - active_requests / capacity  (utilization)
      - arrival_rate / throughput   (demand pressure)
      - avg_service_time_ms         (latency signal)

    Anti-oscillation:
      - EMA smoothing on composite load signal
      - Cooldown period after each scaling event
      - Hysteresis: separate up/down thresholds (gap prevents flapping)
      - Fast scale-up (+2), slow scale-down (-1)
      - Oscillation frequency detection with adaptive cooldown
    """

    def __init__(self, config):
        self._config = config
        scaling = config["scaling"]
        self._interval = scaling.get("metrics_interval", 5)
        self._cooldown = scaling.get("cooldown", 30)
        self._ema_alpha = scaling.get("ema_alpha", 0.3)

        # Scaling event log for eval reproducibility
        self._scaling_events = []

        # Autotune selector (inner control loop)
        self._selector = ConfigSelector(config)

        self._stages = {}
        for stage in ("prefill", "decode"):
            s_cfg = scaling[stage]
            port = config[stage]["port"]
            host = config[stage].get("connect_host", config[stage].get("host", "localhost"))
            self._stages[stage] = {
                "target": f"{host}:{port}",
                "min_replicas": s_cfg["min_replicas"],
                "max_replicas": s_cfg["max_replicas"],
                "scale_up_threshold": s_cfg["scale_up_threshold"],
                "scale_down_threshold": s_cfg["scale_down_threshold"],
                "ema_load": 0.0,
                "last_scale_time": 0.0,
                "current_replicas": 1,
                # Oscillation detection: track recent scale directions
                "scale_history": deque(maxlen=10),  # +1 = up, -1 = down
                "adaptive_cooldown": scaling.get("cooldown", 30),
                # Multi-metric EMA
                "ema_service_time": 0.0,
                "ema_arrival_rate": 0.0,
            }

    @property
    def scaling_events(self):
        """Return scaling event log for eval analysis."""
        return list(self._scaling_events)

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
        """Compute composite load signal (0-1+) from multiple metrics.

        Weights: utilization 0.5, service_time 0.3, arrival_rate 0.2
        """
        info = self._stages[stage]
        if stage == "prefill":
            capacity = self._config["prefill"].get("max_concurrent", 4)
        else:
            capacity = self._config["decode"].get("max_batch_size", 32)

        # Signal 1: utilization (active / capacity)
        utilization = metrics.active_requests / max(capacity, 1)

        # Signal 2: service time pressure (normalized by a baseline of 100ms)
        alpha = self._ema_alpha
        info["ema_service_time"] = (
            alpha * metrics.avg_service_time_ms + (1 - alpha) * info["ema_service_time"]
        )
        stime_signal = min(info["ema_service_time"] / 100.0, 2.0)  # cap at 2x

        # Signal 3: arrival rate pressure
        info["ema_arrival_rate"] = (
            alpha * metrics.arrival_rate + (1 - alpha) * info["ema_arrival_rate"]
        )
        # Normalize arrival rate by capacity * replicas
        effective_capacity = capacity * info["current_replicas"]
        arrival_signal = info["ema_arrival_rate"] / max(effective_capacity, 1)

        # Weighted composite
        composite = 0.5 * utilization + 0.3 * stime_signal + 0.2 * arrival_signal
        return composite

    def _detect_oscillation(self, stage):
        """Check if recent scaling events show oscillation (alternating up/down)."""
        history = self._stages[stage]["scale_history"]
        if len(history) < 4:
            return False
        # Count direction changes in recent history
        changes = sum(1 for i in range(1, len(history)) if history[i] != history[i - 1])
        return changes >= 3  # 3+ direction changes in last 10 events = oscillating

    def _scale(self, stage, replicas):
        """Scale a deployment via kubectl with event logging."""
        replicas = int(replicas)
        info = self._stages[stage]
        replicas = max(info["min_replicas"], min(info["max_replicas"], replicas))
        if replicas == info["current_replicas"]:
            return

        direction = 1 if replicas > info["current_replicas"] else -1
        old_replicas = info["current_replicas"]

        logger.info("Scaling %s: %d -> %d replicas", stage, old_replicas, replicas)
        try:
            subprocess.run(
                ["kubectl", "-n", "llm-inference", "scale", "deployment",
                 stage, f"--replicas={replicas}"],
                check=True, capture_output=True, text=True,
            )
            info["current_replicas"] = replicas
            info["last_scale_time"] = time.monotonic()
            info["scale_history"].append(direction)

            # Log event for eval
            self._scaling_events.append({
                "timestamp": time.time(),
                "stage": stage,
                "from_replicas": old_replicas,
                "to_replicas": replicas,
                "direction": "up" if direction > 0 else "down",
                "ema_load": info["ema_load"],
            })

            # Adaptive cooldown: increase if oscillating
            if self._detect_oscillation(stage):
                info["adaptive_cooldown"] = min(info["adaptive_cooldown"] * 1.5, 120)
                logger.warning(
                    "Oscillation detected for %s — increasing cooldown to %.0fs",
                    stage, info["adaptive_cooldown"],
                )
            else:
                info["adaptive_cooldown"] = self._cooldown

        except subprocess.CalledProcessError as e:
            logger.error("kubectl scale failed for %s: %s", stage, e.stderr)

    def _get_gpu_pct(self, stage):
        """Query the current GPU% for a stage's workers."""
        # In production, read from MPS env or query NVML
        # Default: assume 100% if not set
        import os
        return int(os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", 100))

    def _apply_autotune_config(self, stage, config):
        """Push selected config to worker replicas via env or gRPC.

        In production, this would update a ConfigMap or call a SetConfig RPC.
        For now, log the selected config for eval tracking.
        """
        batch_size = config.get("batch_size")
        if batch_size:
            logger.info("Autotune %s: selected batch_size=%d (gpu_pct=%s)",
                        stage, batch_size, config.get("gpu_pct", "?"))

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

            # Cooldown check (uses adaptive cooldown if oscillating)
            if now - info["last_scale_time"] < info["adaptive_cooldown"]:
                continue

            current = info["current_replicas"]
            if ema > info["scale_up_threshold"]:
                # Fast scale-up: +2 if far above threshold, +1 otherwise
                step = 2 if ema > info["scale_up_threshold"] * 1.5 else 1
                self._scale(stage, current + step)
            elif ema < info["scale_down_threshold"] and current > info["min_replicas"]:
                # Slow scale-down: always -1
                self._scale(stage, current - 1)

            # Inner loop: autotune config selection based on current GPU%
            if self._selector.enabled:
                gpu_pct = self._get_gpu_pct(stage)
                selected = self._selector.select(stage, gpu_pct)
                if selected:
                    self._apply_autotune_config(stage, selected)

            logger.debug(
                "%s: load=%.2f ema=%.2f stime=%.1fms arr_rate=%.1f replicas=%d",
                stage, load, ema, info["ema_service_time"],
                info["ema_arrival_rate"], info["current_replicas"],
            )

    def export_events(self, path="scaling_events.json"):
        """Export scaling event log for evaluation analysis."""
        with open(path, "w") as f:
            json.dump(self._scaling_events, f, indent=2)
        logger.info("Exported %d scaling events to %s", len(self._scaling_events), path)

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
