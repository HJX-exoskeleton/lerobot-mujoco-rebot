import numpy as np

from rebot_aerohand_right_act_sim.visualization import (
    ReplaySensorVisualizer,
    hand_contact_metrics,
)


def test_hand_contact_metrics_locate_dominant_region():
    value = np.zeros(6, dtype=np.float32)
    value[3] = 2.0
    maximum, mean, dominant = hand_contact_metrics(value)
    assert maximum == 2.0
    assert mean > 0
    assert dominant == 3


def test_hand_contact_metrics_without_contact():
    value = np.zeros(6, dtype=np.float32)
    maximum, mean, dominant = hand_contact_metrics(value)
    assert maximum == 0.0
    assert dominant == -1


def test_replay_sensor_panel_contract():
    visualizer = ReplaySensorVisualizer(
        history_frames=10, contact_color_max=15.0
    )
    panel = visualizer.render(
        np.asarray([1, 0, 0, 0, 1, 2, 3, 4, 5, 6], dtype=np.float32),
        np.zeros(6, dtype=np.float32),
        np.asarray(
            [0.11, 0.11, 0.11, 0.11, 0.0, 0.038, 0.112], dtype=np.float32
        ),
        frame_index=3,
        timestamp=0.06,
    )
    assert panel.shape == (480, 640, 3)
    assert panel.dtype == np.uint8
    assert np.any(panel != 22)
    visualizer.reset()
    assert len(visualizer.imu_history) == 0
    assert visualizer.contact_scale == 15.0
