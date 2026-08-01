import numpy as np
import pytest

from rebot_act_sim.sensors import (
    TactileProcessingConfig,
    TactileSignalProcessor,
    read_imu,
    read_tactile_normal,
    spatial_smooth,
    validate_processing_spec,
    write_or_validate_processing_spec,
)


class FakeParser:
    def __init__(self):
        self.sensor_names = [
            "orientation_left",
            "ang_vel_left",
            "accel_left",
            *[f"touch_point_left_{index:03d}" for index in range(128)],
            *[f"touch_point_right_{index:03d}" for index in range(128)],
        ]

    def get_sensor_value(self, name):
        if name == "orientation_left":
            return np.asarray([1, 0, 0, 0], dtype=np.float64)
        if name in {"ang_vel_left", "accel_left"}:
            return np.asarray([1, 2, 3], dtype=np.float64)
        index = int(name.rsplit("_", 1)[1])
        scale = 1 if "left" in name else 2
        return np.asarray([scale * index, 0, 0], dtype=np.float64)


def test_imu_contract():
    value = read_imu(FakeParser())
    assert value.shape == (10,)
    assert value.dtype == np.float32


def test_tactile_contract_preserves_left_right_halves():
    left, right = read_tactile_normal(FakeParser(), normal_axis=0)
    assert left.shape == right.shape == (8, 16)
    assert left.dtype == right.dtype == np.float32
    # XML stores 8 columns per physical row. Public shape is [column, row].
    assert left[0, 0] == 0
    assert left[7, 0] == 7
    assert left[0, 1] == 8
    assert right.mean() > left.mean()


def test_tactile_processor_averages_spatially_smooths_and_filters():
    processor = TactileSignalProcessor(
        TactileProcessingConfig(
            normal_axis=2,
            clip_max=10.0,
            temporal_ema_alpha=0.5,
            spatial_smoothing=True,
        )
    )
    impulse = np.zeros((8, 16), dtype=np.float32)
    impulse[4, 8] = 20.0
    processor.update(impulse, np.zeros_like(impulse))
    processor.update(np.zeros_like(impulse), np.zeros_like(impulse))
    left, right = processor.consume()
    # Per-physics-step clipping gives an averaged center of 5 before smoothing.
    assert left[4, 8] == 1.25
    assert left.sum() == 5.0
    assert np.count_nonzero(left) == 9
    assert np.all(right == 0)
    processor.update(np.zeros_like(impulse), np.zeros_like(impulse))
    left_next, _ = processor.consume()
    assert np.allclose(left_next, left * 0.5)


def test_spatial_smoothing_preserves_uniform_pressure():
    value = np.full((8, 16), 3.0, dtype=np.float32)
    assert np.allclose(spatial_smooth(value), value)


def test_processing_spec_rejects_collection_inference_mismatch(tmp_path):
    configured = {"temporal_ema_alpha": 0.25, "clip_max": 25.0}
    write_or_validate_processing_spec(tmp_path, configured)
    validate_processing_spec(tmp_path, configured)
    with pytest.raises(ValueError, match="mismatch"):
        validate_processing_spec(
            tmp_path, {"temporal_ema_alpha": 0.5, "clip_max": 25.0}
        )


def test_processing_config_rejects_invalid_projection():
    with pytest.raises(ValueError, match="projection_sigma"):
        TactileProcessingConfig.from_mapping({"projection_sigma": 0.0})
