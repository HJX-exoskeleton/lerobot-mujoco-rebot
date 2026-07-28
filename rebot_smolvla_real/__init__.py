"""Real-robot SmolVLA workflow for reBot."""

from .schema import (
    ACTION_DIM,
    CAMERA_KEYS,
    STATE_DIM,
    build_lerobot_features,
    validate_frame,
)
__all__ = [
    "ACTION_DIM",
    "CAMERA_KEYS",
    "STATE_DIM",
    "build_lerobot_features",
    "validate_frame",
]
