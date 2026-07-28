"""Hardware-neutral interfaces used by collection and deployment workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class CameraSample:
    rgb: np.ndarray
    frame_id: int
    timestamp: float


@dataclass(frozen=True)
class RobotFeedback:
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    gripper_position: float
    gripper_velocity: float
    timestamp: float


@dataclass(frozen=True)
class RobotTarget:
    joint_position: np.ndarray
    gripper_position: float
    timestamp: float


@runtime_checkable
class Camera(Protocol):
    name: str

    def read(self) -> CameraSample:
        """Return the latest RGB sample without changing robot state."""

    def close(self) -> None:
        """Release camera resources."""


@runtime_checkable
class Leader(Protocol):
    def read_target(self) -> RobotTarget:
        """Return the latest operator-generated target."""

    def close(self) -> None:
        """Release leader serial resources."""


@runtime_checkable
class Robot(Protocol):
    def read_feedback(self) -> RobotFeedback:
        """Read six arm joints and gripper feedback."""

    def command(self, target: RobotTarget) -> None:
        """Send a target that has already passed safety filtering."""

    def disable(self) -> None:
        """Disable actuators when the configured safety policy requires it."""

    def close(self) -> None:
        """Release motor communication resources."""
