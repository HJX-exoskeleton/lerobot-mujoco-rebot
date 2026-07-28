from __future__ import annotations

import torch
import numpy as np

from rebot_act_real.multimodal_policy import IMUEncoder, TactileEncoder
from rebot_act_real.multimodal_visualization import (
    AsyncPanelRenderer,
    add_multimodal_panel,
)
from rebot_act_real.workflow.train import build_argparser


def test_multimodal_training_flags_are_independent():
    args = build_argparser().parse_args(["--imu"])
    assert args.imu is True
    assert args.tactile is False
    args = build_argparser().parse_args(["--tactile"])
    assert args.imu is False
    assert args.tactile is True


def test_sensor_encoders_preserve_batch_and_embedding_dimensions():
    assert IMUEncoder(64)(torch.zeros(2, 10)).shape == (2, 64)
    assert TactileEncoder(64)(torch.zeros(2, 12, 30)).shape == (2, 64)


def test_multimodal_panel_appends_sensor_view():
    base = np.zeros((700, 960, 3), dtype=np.uint8)
    panel = add_multimodal_panel(
        base,
        {
            "sensor.imu": np.zeros(10, dtype=np.float32),
            "sensor.tactile": np.ones((12, 30), dtype=np.float32) * 0.5,
        },
        use_imu=True,
        use_tactile=True,
    )
    assert panel.shape == (700, 1300, 3)
    assert np.any(panel[:, 960:] != 24)


def test_multimodal_panel_draws_imu_history_curves():
    base = np.zeros((700, 960, 3), dtype=np.uint8)
    history = []
    for value in np.linspace(-1.0, 1.0, 20, dtype=np.float32):
        imu = np.zeros(10, dtype=np.float32)
        imu[4:7] = (value, -value, value * 0.5)
        imu[7:10] = (value * 2.0, value, -value)
        history.append({"sensor.imu": imu})
    panel = add_multimodal_panel(
        base,
        history[-1],
        use_imu=True,
        use_tactile=False,
        sensor_history=history,
    )
    # Red/green/blue axis traces are rendered inside both chart regions.
    chart = panel[150:303, 974:1287].astype(np.int16)
    assert np.any(chart[:, :, 0] > chart[:, :, 1] + 40)
    assert np.any(chart[:, :, 1] > chart[:, :, 0] + 40)
    assert np.any(chart[:, :, 2] > chart[:, :, 1] + 40)


def test_async_panel_renderer_returns_each_frame_only_once():
    renderer = AsyncPanelRenderer()
    assert renderer.submit_latest(
        lambda: np.ones((12, 16, 3), dtype=np.uint8)
    )
    renderer.close()
    assert renderer.take_latest().shape == (12, 16, 3)
    assert renderer.take_latest() is None
