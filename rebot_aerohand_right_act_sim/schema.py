"""Canonical data contract shared by simulation collection and inference."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

FPS = 50
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
STATE_DIM = 6
ACTION_DIM = 13
ARM_DIM = 6
HAND_ACTUATOR_DIM = 7
HAND_JOINT_DIM = 16
HAND_CONTACT_DIM = 6
IMU_DIM = 10
OBJECT_QPOS_DIM = 7

CAMERA_KEYS = ("observation.image", "observation.wrist_image")
POLICY_FEATURE_KEYS = (*CAMERA_KEYS, "observation.state", "action")

# AeroHand actuator control ranges from aerohand_right.xml. Order matches the
# seven hand actuators: four finger flexor tendon lengths, thumb abduction
# angle, then the two thumb tendon lengths. High tendon length = open hand.
HAND_CTRL_MIN = np.asarray(
    [0.058520, 0.058520, 0.058520, 0.058520, -0.1, 0.026152, 0.081568],
    dtype=np.float64,
)
HAND_CTRL_MAX = np.asarray(
    [0.110387, 0.110387, 0.110387, 0.110387, 1.75, 0.038389, 0.112138],
    dtype=np.float64,
)

HAND_CONTACT_REGIONS = ("thumb", "index", "middle", "ring", "pinky", "palm")

AUXILIARY_FEATURES = {
    "sensor.joint_velocity": {
        "dtype": "float32",
        "shape": (STATE_DIM,),
        "names": ["joint"],
    },
    "sensor.hand_feedback": {
        "dtype": "float32",
        "shape": (HAND_ACTUATOR_DIM,),
        "names": [
            "four_finger_tendon_lengths_thumb_abduction_two_thumb_tendon_lengths"
        ],
    },
    "sensor.hand_joint_position": {
        "dtype": "float32",
        "shape": (HAND_JOINT_DIM,),
        "names": ["hand_joint"],
    },
    "sensor.imu": {
        "dtype": "float32",
        "shape": (IMU_DIM,),
        "names": ["quaternion_wxyz_gyro_xyz_accel_xyz"],
    },
    "sensor.hand_contact": {
        "dtype": "float32",
        "shape": (HAND_CONTACT_DIM,),
        "names": ["thumb_index_middle_ring_pinky_palm_normal_force"],
    },
    "sensor.sim_time": {
        "dtype": "float64",
        "shape": (1,),
        "names": ["seconds"],
    },
    "episode.object_initial_position": {
        "dtype": "float32",
        "shape": (OBJECT_QPOS_DIM,),
        "names": ["cylinder_xyz_quaternion_wxyz"],
    },
}


def build_lerobot_features(*, include_auxiliary: bool = True) -> dict[str, dict]:
    image = {
        "dtype": "image",
        "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        "names": ["height", "width", "channels"],
    }
    features = {
        CAMERA_KEYS[0]: dict(image),
        CAMERA_KEYS[1]: dict(image),
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": ["joint_position_rad"],
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": [
                "six_arm_joint_target_rad_and_seven_hand_actuator_target"
            ],
        },
    }
    if include_auxiliary:
        features.update({key: dict(value) for key, value in AUXILIARY_FEATURES.items()})
    return features


def validate_frame(frame: Mapping[str, object], *, require_auxiliary: bool = True) -> None:
    required = set(POLICY_FEATURE_KEYS)
    if require_auxiliary:
        required.update(AUXILIARY_FEATURES)
    missing = sorted(required.difference(frame))
    if missing:
        raise KeyError(f"ACT simulation frame is missing features: {missing}")

    for key in CAMERA_KEYS:
        value = np.asarray(frame[key])
        expected = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
        if value.shape != expected or value.dtype != np.uint8:
            raise ValueError(f"{key} must be uint8 RGB with shape {expected}, got {value.dtype} {value.shape}")

    declarations = build_lerobot_features(include_auxiliary=require_auxiliary)
    for key, feature in declarations.items():
        if key in CAMERA_KEYS:
            continue
        value = np.asarray(frame[key])
        expected_shape = tuple(feature["shape"])
        expected_dtype = np.dtype(feature["dtype"])
        if value.shape != expected_shape:
            raise ValueError(f"{key} must have shape {expected_shape}, got {value.shape}")
        if value.dtype != expected_dtype:
            raise ValueError(f"{key} must use {expected_dtype}, got {value.dtype}")
        if np.issubdtype(value.dtype, np.floating) and not np.all(np.isfinite(value)):
            raise ValueError(f"{key} contains NaN or Inf")

    hand_target = np.asarray(frame["action"], dtype=np.float64)[ARM_DIM:]
    # float32 storage rounds the XML ctrlrange endpoints slightly; tolerate
    # that rounding while still rejecting genuinely out-of-range targets.
    tolerance = 1e-5
    if np.any(hand_target < HAND_CTRL_MIN - tolerance) or np.any(
        hand_target > HAND_CTRL_MAX + tolerance
    ):
        raise ValueError(
            "hand actuator target must stay within the XML ctrlrange "
            f"[{HAND_CTRL_MIN}, {HAND_CTRL_MAX}], got {hand_target}"
        )
