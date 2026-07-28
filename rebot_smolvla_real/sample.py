"""Synchronized training samples for the real SmolVLA workflow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .contracts import CameraSample, RobotFeedback, RobotTarget
from .schema import IMAGE_HEIGHT, IMAGE_WIDTH, validate_frame


@dataclass(frozen=True)
class SynchronizedSample:
    agent: CameraSample
    wrist: CameraSample
    feedback: RobotFeedback
    target: RobotTarget
    imu: np.ndarray
    imu_magnetometer: np.ndarray
    imu_euler: np.ndarray
    imu_barometer: np.ndarray
    tactile: np.ndarray
    imu_frame_id: int
    tactile_frame_id: int
    imu_timestamp: float
    tactile_timestamp: float
    timestamp: float

    @property
    def camera_skew_ms(self) -> float:
        return abs(self.agent.timestamp - self.wrist.timestamp) * 1000.0

    @property
    def maximum_age_ms(self) -> float:
        oldest = min(
            self.agent.timestamp,
            self.wrist.timestamp,
            self.feedback.timestamp,
            self.target.timestamp,
        )
        return max(0.0, self.timestamp - oldest) * 1000.0


def _resize_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"RGB image must have HxWx3 shape, got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"RGB image must be uint8, got {image.dtype}")
    if image.shape[:2] == (IMAGE_HEIGHT, IMAGE_WIDTH):
        return image.copy()
    return np.asarray(
        Image.fromarray(image).resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.BILINEAR)
    )


def to_lerobot_frame(
    sample: SynchronizedSample,
    *,
    maximum_camera_skew_ms: float = 50.0,
    maximum_sample_age_ms: float = 150.0,
) -> dict[str, np.ndarray]:
    """Convert a synchronized hardware sample to the canonical dataset frame."""

    if sample.camera_skew_ms > maximum_camera_skew_ms:
        raise RuntimeError(
            f"camera skew is {sample.camera_skew_ms:.1f} ms, "
            f"limit is {maximum_camera_skew_ms:.1f} ms"
        )
    if sample.maximum_age_ms > maximum_sample_age_ms:
        raise RuntimeError(
            f"sample age is {sample.maximum_age_ms:.1f} ms, "
            f"limit is {maximum_sample_age_ms:.1f} ms"
        )

    state = np.asarray(sample.feedback.joint_position, dtype=np.float32)
    arm_target = np.asarray(sample.target.joint_position, dtype=np.float32)
    frame = {
        "observation.image": _resize_rgb(sample.agent.rgb),
        "observation.wrist_image": _resize_rgb(sample.wrist.rgb),
        "observation.state": state,
        "action": np.concatenate(
            [arm_target, np.asarray([sample.target.gripper_position], dtype=np.float32)]
        ),
        "sensor.joint_velocity": np.asarray(
            sample.feedback.joint_velocity, dtype=np.float32
        ),
        "sensor.gripper_feedback": np.asarray(
            [sample.feedback.gripper_position, sample.feedback.gripper_velocity],
            dtype=np.float32,
        ),
        "sensor.imu": np.asarray(sample.imu, dtype=np.float32),
        "sensor.imu_magnetometer": np.asarray(
            sample.imu_magnetometer, dtype=np.float32
        ),
        "sensor.imu_euler": np.asarray(sample.imu_euler, dtype=np.float32),
        "sensor.imu_barometer": np.asarray(sample.imu_barometer, dtype=np.float32),
        "sensor.tactile": np.asarray(sample.tactile, dtype=np.float32),
        "sensor.frame_ids": np.asarray(
            [
                sample.agent.frame_id,
                sample.wrist.frame_id,
                sample.imu_frame_id,
                sample.tactile_frame_id,
            ],
            dtype=np.int64,
        ),
        "sensor.timestamps": np.asarray(
            [
                sample.agent.timestamp,
                sample.wrist.timestamp,
                sample.imu_timestamp,
                sample.tactile_timestamp,
            ],
            dtype=np.float64,
        ),
    }
    validate_frame(frame)
    return frame
