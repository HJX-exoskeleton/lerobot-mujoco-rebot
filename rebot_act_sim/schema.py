"""Canonical data contract shared by simulation collection and inference."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

FPS = 50
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
STATE_DIM = 6
ACTION_DIM = 7
IMU_DIM = 10
TACTILE_SHAPE = (8, 16)

CAMERA_KEYS = ("observation.image", "observation.wrist_image")
POLICY_FEATURE_KEYS = (*CAMERA_KEYS, "observation.state", "action")
AUXILIARY_FEATURES = {
    "sensor.joint_velocity": {
        "dtype": "float32",
        "shape": (STATE_DIM,),
        "names": ["joint"],
    },
    "sensor.gripper_feedback": {
        "dtype": "float32",
        "shape": (2,),
        "names": ["position", "velocity"],
    },
    "sensor.imu": {
        "dtype": "float32",
        "shape": (IMU_DIM,),
        "names": ["quaternion_wxyz_gyro_xyz_accel_xyz"],
    },
    "sensor.tactile_left": {
        "dtype": "float32",
        "shape": TACTILE_SHAPE,
        "names": ["row", "column"],
    },
    "sensor.tactile_right": {
        "dtype": "float32",
        "shape": TACTILE_SHAPE,
        "names": ["row", "column"],
    },
    "sensor.tactile_left_raw": {
        "dtype": "float32",
        "shape": TACTILE_SHAPE,
        "names": ["row", "column"],
    },
    "sensor.tactile_right_raw": {
        "dtype": "float32",
        "shape": TACTILE_SHAPE,
        "names": ["row", "column"],
    },
    "sensor.sim_time": {
        "dtype": "float64",
        "shape": (1,),
        "names": ["seconds"],
    },
    "episode.object_initial_position": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["red_cube_xyz_plate_xyz"],
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
            "names": ["six_joint_target_rad_and_binary_gripper"],
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

    action = np.asarray(frame["action"])
    if not 0.0 <= float(action[-1]) <= 1.0:
        raise ValueError(f"binary gripper action must be in [0, 1], got {action[-1]}")
