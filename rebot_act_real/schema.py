"""Canonical LeRobot contract for reBot ACT multimodal demonstrations.

Only ``observation.*`` and ``action`` are policy features.  Auxiliary sensors
deliberately use the ``sensor.*`` namespace: LeRobot preserves every modality
in the dataset, while the first visual ACT baseline consumes only the two RGB
images, joint state, and action.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


FPS = 50
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
STATE_DIM = 6
ACTION_DIM = 7
IMU_DIM = 10
TACTILE_SHAPE = (12, 30)

CAMERA_KEYS = (
    "observation.image",
    "observation.wrist_image",
)

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
        "names": ["imu"],
    },
    "sensor.imu_magnetometer": {
        "dtype": "float32",
        "shape": (3,),
        "names": ["axis"],
    },
    "sensor.imu_euler": {
        "dtype": "float32",
        "shape": (3,),
        "names": ["axis"],
    },
    "sensor.imu_barometer": {
        "dtype": "float32",
        "shape": (4,),
        "names": ["value"],
    },
    "sensor.tactile": {
        "dtype": "float32",
        "shape": TACTILE_SHAPE,
        "names": ["row", "column"],
    },
    "sensor.frame_ids": {
        "dtype": "int64",
        "shape": (4,),
        "names": ["cam_high", "cam_wrist", "imu", "tactile"],
    },
    "sensor.timestamps": {
        "dtype": "float64",
        "shape": (4,),
        "names": ["cam_high", "cam_wrist", "imu", "tactile"],
    },
}


def build_lerobot_features(*, include_auxiliary: bool = True) -> dict[str, dict]:
    """Return the LeRobot feature declaration used by real data collection."""

    image_feature = {
        "dtype": "image",
        "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        "names": ["height", "width", "channels"],
    }
    features = {
        CAMERA_KEYS[0]: dict(image_feature),
        CAMERA_KEYS[1]: dict(image_feature),
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": ["joint"],
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": ["joint_target"],
        },
    }
    if include_auxiliary:
        features.update({key: dict(value) for key, value in AUXILIARY_FEATURES.items()})
    return features


def validate_frame(frame: Mapping[str, object], *, require_auxiliary: bool = True) -> None:
    """Validate one frame before it is added to a LeRobot episode buffer."""

    required = set(POLICY_FEATURE_KEYS)
    if require_auxiliary:
        required.update(AUXILIARY_FEATURES)
    missing = sorted(required.difference(frame))
    if missing:
        raise KeyError(f"ACT real frame is missing features: {missing}")

    for key in CAMERA_KEYS:
        image = np.asarray(frame[key])
        expected = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
        if image.shape != expected:
            raise ValueError(f"{key} must have shape {expected}, got {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"{key} must use uint8 RGB data, got {image.dtype}")

    state = np.asarray(frame["observation.state"], dtype=np.float32)
    if state.shape != (STATE_DIM,):
        raise ValueError(
            f"observation.state must have shape {(STATE_DIM,)}, got {state.shape}"
        )
    if not np.all(np.isfinite(state)):
        raise ValueError("observation.state contains NaN or Inf")

    action = np.asarray(frame["action"], dtype=np.float32)
    if action.shape != (ACTION_DIM,):
        raise ValueError(f"action must have shape {(ACTION_DIM,)}, got {action.shape}")
    if not np.all(np.isfinite(action)):
        raise ValueError("action contains NaN or Inf")

    if require_auxiliary:
        for key, feature in AUXILIARY_FEATURES.items():
            value = np.asarray(frame[key])
            expected_shape = tuple(feature["shape"])
            if value.shape != expected_shape:
                raise ValueError(f"{key} must have shape {expected_shape}, got {value.shape}")
            expected_dtype = np.dtype(feature["dtype"])
            if value.dtype != expected_dtype:
                raise ValueError(f"{key} must use {expected_dtype}, got {value.dtype}")
            if np.issubdtype(value.dtype, np.floating) and not np.all(np.isfinite(value)):
                raise ValueError(f"{key} contains NaN or Inf")
