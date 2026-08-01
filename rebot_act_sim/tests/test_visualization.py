import numpy as np

from rebot_act_sim.visualization import ReplaySensorVisualizer, tactile_metrics


def test_tactile_metrics_locates_contact_center():
    value = np.zeros((8, 16), dtype=np.float32)
    value[3, 7] = 2.0
    maximum, mean, row, column = tactile_metrics(value)
    assert maximum == 2.0
    assert mean > 0
    assert row == 3.0
    assert column == 7.0


def test_replay_sensor_panel_contract():
    visualizer = ReplaySensorVisualizer(history_frames=10, tactile_color_max=15.0)
    panel = visualizer.render(
        np.asarray([1, 0, 0, 0, 1, 2, 3, 4, 5, 6], dtype=np.float32),
        np.zeros((8, 16), dtype=np.float32),
        np.ones((8, 16), dtype=np.float32),
        frame_index=3,
        timestamp=0.06,
    )
    assert panel.shape == (480, 640, 3)
    assert panel.dtype == np.uint8
    assert np.any(panel != 22)
    visualizer.reset()
    assert len(visualizer.imu_history) == 0
    assert visualizer.tactile_scale == 15.0
