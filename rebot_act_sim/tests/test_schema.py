import numpy as np
import pytest

from rebot_act_sim.schema import FPS, TACTILE_SHAPE, build_lerobot_features, validate_frame


def valid_frame():
    return {
        "observation.image": np.zeros((256, 256, 3), dtype=np.uint8),
        "observation.wrist_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "observation.state": np.zeros(6, dtype=np.float32),
        "action": np.zeros(7, dtype=np.float32),
        "sensor.joint_velocity": np.zeros(6, dtype=np.float32),
        "sensor.gripper_feedback": np.zeros(2, dtype=np.float32),
        "sensor.imu": np.zeros(10, dtype=np.float32),
        "sensor.tactile_left": np.zeros((8, 16), dtype=np.float32),
        "sensor.tactile_right": np.zeros((8, 16), dtype=np.float32),
        "sensor.tactile_left_raw": np.zeros((8, 16), dtype=np.float32),
        "sensor.tactile_right_raw": np.zeros((8, 16), dtype=np.float32),
        "sensor.sim_time": np.zeros(1, dtype=np.float64),
        "episode.object_initial_position": np.zeros(6, dtype=np.float32),
    }


def test_schema_keeps_auxiliary_sensors_outside_observation_namespace():
    assert FPS == 50
    assert TACTILE_SHAPE == (8, 16)
    features = build_lerobot_features()
    assert set(key for key in features if key.startswith("observation.")) == {
        "observation.image",
        "observation.wrist_image",
        "observation.state",
    }
    validate_frame(valid_frame())


def test_invalid_gripper_target_is_rejected():
    frame = valid_frame()
    frame["action"][-1] = 1.5
    with pytest.raises(ValueError, match="gripper"):
        validate_frame(frame)
