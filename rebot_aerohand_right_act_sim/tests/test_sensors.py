from pathlib import Path

import numpy as np
import pytest

from rebot_aerohand_right_act_sim.sensors import (
    HAND_CONTACT_REGIONS,
    HandContactProcessingConfig,
    HandContactSignalProcessor,
    classify_hand_geom_regions,
    read_imu,
    validate_processing_spec,
    write_or_validate_processing_spec,
)

SCENE_XML = (
    Path(__file__).resolve().parents[2]
    / "asset_rebot_aerohand_right"
    / "mujoco_xml"
    / "rebotarm_aerohand_act_cylinder.xml"
)


class FakeParser:
    def __init__(self):
        self.sensor_names = [
            "orientation_left",
            "ang_vel_left",
            "accel_left",
        ]

    def get_sensor_value(self, name):
        if name == "orientation_left":
            return np.asarray([1, 0, 0, 0], dtype=np.float64)
        if name in {"ang_vel_left", "accel_left"}:
            return np.asarray([1, 2, 3], dtype=np.float64)
        raise KeyError(name)


def test_imu_contract():
    value = read_imu(FakeParser())
    assert value.shape == (10,)
    assert value.dtype == np.float32


def test_contact_processor_averages_clips_and_filters():
    processor = HandContactSignalProcessor(
        HandContactProcessingConfig(clip_max=10.0, temporal_ema_alpha=0.5)
    )
    impulse = np.zeros(6, dtype=np.float32)
    impulse[0] = 20.0  # clipped to 10 per step
    processor.update(impulse)
    processor.update(np.zeros(6, dtype=np.float32))
    value = processor.consume()
    assert value[0] == pytest.approx(5.0)   # (10 + 0) / 2
    assert np.all(value[1:] == 0)
    processor.update(np.zeros(6, dtype=np.float32))
    value_next = processor.consume()
    assert np.allclose(value_next, value * 0.5)  # EMA with alpha=0.5


def test_contact_processor_consume_without_updates():
    processor = HandContactSignalProcessor(HandContactProcessingConfig())
    value = processor.consume()
    assert value.shape == (6,)
    assert np.all(value == 0)


def test_processing_spec_rejects_collection_inference_mismatch(tmp_path):
    configured = {"temporal_ema_alpha": 0.25, "clip_max": 25.0}
    write_or_validate_processing_spec(tmp_path, configured)
    validate_processing_spec(tmp_path, configured)
    with pytest.raises(ValueError, match="mismatch"):
        validate_processing_spec(
            tmp_path, {"temporal_ema_alpha": 0.5, "clip_max": 25.0}
        )


def test_processing_config_rejects_invalid_clip():
    with pytest.raises(ValueError, match="clip_max"):
        HandContactProcessingConfig.from_mapping({"clip_max": 0.0})


def test_processing_config_rejects_unknown_regions():
    with pytest.raises(ValueError, match="regions"):
        HandContactProcessingConfig.from_mapping({"regions": ["a", "b"]})


def test_hand_geom_region_classification():
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    regions = classify_hand_geom_regions(model)
    assert len(regions) > 0
    assert set(regions.values()) <= set(range(len(HAND_CONTACT_REGIONS)))
    # The palm collision geoms map to the palm region.
    palm_geom = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "palm_collision_1"
    )
    assert regions[palm_geom] == len(HAND_CONTACT_REGIONS) - 1
