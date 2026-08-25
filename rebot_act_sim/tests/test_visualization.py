import numpy as np

from rebot_act_sim.visualization import (
    ReplaySensorVisualizer,
    _reference_tactile_image,
    _stabilize_distance_map,
    tactile_metrics,
)


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
    assert panel.shape == (640, 640, 3)
    assert panel.dtype == np.uint8
    assert np.any(panel != 22)
    visualizer.reset()
    assert len(visualizer.imu_history) == 0
    assert visualizer.tactile_scale == 15.0


def test_reference_tactile_image_keeps_pads_independent():
    left = np.zeros((8, 16), dtype=np.float32)
    right = np.zeros_like(left)
    left[2, 5] = 1.0
    image = _reference_tactile_image(left, right)
    assert image.shape == (320, 320, 3)
    # Ignore labels and the central divider. A left-only taxel must not be
    # synthesized into the right panel.
    assert int(image[100:, :158].max()) > int(image[100:, 162:].max())


def test_displaced_cylinder_lines_are_not_merged_into_two_lines() -> None:
    left = np.zeros((8, 16), dtype=np.float32)
    right = np.zeros_like(left)
    left[1, 4:13] = 1.0
    right[4, 4:13] = 1.0
    image = _reference_tactile_image(left, right)
    left_panel = image[80:260, :158]
    right_panel = image[80:260, 162:]
    # Each physical pad contributes one vertical band to its own panel. The
    # panels must differ rather than both containing their averaged pair.
    assert not np.array_equal(left_panel, right_panel)


def test_new_xml_fingertip_display_is_at_bottom():
    left = np.zeros((8, 16), dtype=np.float32)
    right = np.zeros_like(left)
    # Display orientation observed for the new XML in the MuJoCo overlay.
    left[3:5, 0] = 1.0
    right[3:5, 15] = 1.0
    image = _reference_tactile_image(left, right)
    top = int(image[40:100, 20:140].max())
    bottom = int(image[280:320, 20:140].max())
    assert bottom > top


def test_distance_display_matches_auto_grasp_ema():
    value = np.zeros((8, 16), dtype=np.float32)
    value[1, 2] = 0.9
    value[4, 8:11] = (0.5, 0.8, 0.5)
    stable = _stabilize_distance_map(value, None)
    assert np.allclose(stable, 0.25 * value)


def test_distance_display_auto_grasp_ema_decays_when_object_leaves():
    previous = np.zeros((8, 16), dtype=np.float32)
    previous[4, 8:11] = 0.8
    stable = _stabilize_distance_map(np.zeros_like(previous), previous)
    assert np.allclose(stable, 0.75 * previous)
