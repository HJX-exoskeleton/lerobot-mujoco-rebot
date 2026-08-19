import numpy as np
import pytest

from rebot_aerohand_right_act_sim.schema import (
    ACTION_DIM,
    FPS,
    HAND_ACTUATOR_DIM,
    HAND_CONTACT_DIM,
    HAND_JOINT_DIM,
    STATE_DIM,
    build_lerobot_features,
    validate_frame,
)


def valid_frame():
    return {
        "observation.image": np.zeros((256, 256, 3), dtype=np.uint8),
        "observation.wrist_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "observation.state": np.zeros(6, dtype=np.float32),
        "action": np.asarray(
            [0, 0, 0, 0, 0, 0, 0.110387, 0.110387, 0.110387, 0.110387,
             0.0, 0.038389, 0.112138],
            dtype=np.float32,
        ),
        "sensor.joint_velocity": np.zeros(6, dtype=np.float32),
        "sensor.hand_feedback": np.zeros(7, dtype=np.float32),
        "sensor.hand_joint_position": np.zeros(16, dtype=np.float32),
        "sensor.imu": np.zeros(10, dtype=np.float32),
        "sensor.hand_contact": np.zeros(6, dtype=np.float32),
        "sensor.sim_time": np.zeros(1, dtype=np.float64),
        "episode.object_initial_position": np.zeros(7, dtype=np.float32),
    }


def test_schema_keeps_auxiliary_sensors_outside_observation_namespace():
    assert FPS == 50
    assert STATE_DIM == 6
    assert ACTION_DIM == 13
    assert HAND_ACTUATOR_DIM == 7
    assert HAND_JOINT_DIM == 16
    assert HAND_CONTACT_DIM == 6
    features = build_lerobot_features()
    assert set(key for key in features if key.startswith("observation.")) == {
        "observation.image",
        "observation.wrist_image",
        "observation.state",
    }
    validate_frame(valid_frame())


def test_invalid_hand_target_is_rejected():
    frame = valid_frame()
    # Finger tendon target above the XML ctrlrange upper bound.
    frame["action"][6] = 1.0
    with pytest.raises(ValueError, match="hand"):
        validate_frame(frame)


def test_valid_hand_target_is_accepted():
    frame = valid_frame()
    # Fully open hand targets: tendon maxima, zero thumb abduction.
    frame["action"][6:] = np.asarray(
        [0.110387, 0.110387, 0.110387, 0.110387, 0.0, 0.038389, 0.112138],
        dtype=np.float32,
    )
    validate_frame(frame)
