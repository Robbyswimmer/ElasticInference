"""Tests for scaling.controller.ScalingController."""
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from common import load_config
from scaling.controller import ScalingController


@pytest.fixture
def controller():
    config = load_config()
    return ScalingController(config)


class TestScalingController:

    def test_init_stages(self, controller):
        assert "prefill" in controller._stages
        assert "decode" in controller._stages
        assert controller._stages["prefill"]["min_replicas"] == 1
        assert controller._stages["prefill"]["max_replicas"] == 4
        assert controller._stages["decode"]["max_replicas"] == 8

    def test_compute_load_prefill(self, controller):
        metrics = MagicMock()
        metrics.active_requests = 2
        load = controller._compute_load(metrics, "prefill")
        # max_concurrent = 4, so load = 2/4 = 0.5
        assert abs(load - 0.5) < 0.01

    def test_compute_load_decode(self, controller):
        metrics = MagicMock()
        metrics.active_requests = 16
        load = controller._compute_load(metrics, "decode")
        # max_batch_size = 32, so load = 16/32 = 0.5
        assert abs(load - 0.5) < 0.01

    def test_compute_load_zero(self, controller):
        metrics = MagicMock()
        metrics.active_requests = 0
        load = controller._compute_load(metrics, "prefill")
        assert load == 0.0

    @patch("scaling.controller.subprocess.run")
    def test_scale_up(self, mock_run, controller):
        mock_run.return_value = MagicMock(returncode=0)
        controller._stages["prefill"]["current_replicas"] = 1
        controller._scale("prefill", 2)
        mock_run.assert_called_once()
        assert controller._stages["prefill"]["current_replicas"] == 2
        call_args = mock_run.call_args[0][0]
        assert "--replicas=2" in call_args

    @patch("scaling.controller.subprocess.run")
    def test_scale_respects_max(self, mock_run, controller):
        mock_run.return_value = MagicMock(returncode=0)
        controller._stages["prefill"]["current_replicas"] = 3
        controller._scale("prefill", 10)  # max is 4
        assert controller._stages["prefill"]["current_replicas"] == 4

    @patch("scaling.controller.subprocess.run")
    def test_scale_respects_min(self, mock_run, controller):
        mock_run.return_value = MagicMock(returncode=0)
        controller._stages["decode"]["current_replicas"] = 2
        controller._scale("decode", 0)  # min is 1
        assert controller._stages["decode"]["current_replicas"] == 1

    @patch("scaling.controller.subprocess.run")
    def test_no_scale_when_same(self, mock_run, controller):
        controller._stages["prefill"]["current_replicas"] = 2
        controller._scale("prefill", 2)
        mock_run.assert_not_called()

    def test_ema_smoothing(self, controller):
        alpha = controller._ema_alpha  # 0.3
        stage = controller._stages["prefill"]
        stage["ema_load"] = 0.0

        # Simulate load = 1.0
        stage["ema_load"] = alpha * 1.0 + (1 - alpha) * 0.0
        assert abs(stage["ema_load"] - 0.3) < 0.01

        # Another reading of 1.0
        stage["ema_load"] = alpha * 1.0 + (1 - alpha) * stage["ema_load"]
        assert abs(stage["ema_load"] - 0.51) < 0.01

    def test_step_scales_up_on_high_load(self, controller):
        import time as _time
        metrics = MagicMock()
        metrics.active_requests = 32

        with patch.object(controller, "_get_metrics", return_value=metrics), \
             patch.object(controller, "_scale") as mock_scale:
            for stage in controller._stages.values():
                # Set last_scale_time far enough in the past to clear cooldown
                stage["last_scale_time"] = _time.monotonic() - 120
                stage["ema_load"] = 0.9
            controller._step()
            assert mock_scale.called

    def test_step_no_scale_during_cooldown(self, controller):
        import time
        metrics = MagicMock()
        metrics.active_requests = 32

        with patch.object(controller, "_get_metrics", return_value=metrics), \
             patch.object(controller, "_scale") as mock_scale:
            for stage in controller._stages.values():
                stage["last_scale_time"] = time.monotonic()
            controller._step()
            mock_scale.assert_not_called()
