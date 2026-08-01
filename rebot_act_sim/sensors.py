"""Canonical MuJoCo tactile signal processing for collection and deployment."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import mujoco
import numpy as np

from .schema import TACTILE_SHAPE

LEFT_PREFIX = "touch_point_left_"
RIGHT_PREFIX = "touch_point_right_"
TACTILE_PROCESSING_SPEC_NAME = "rebot_sim_tactile_processing.json"


def read_imu(parser) -> np.ndarray:
    values = [
        np.asarray(parser.get_sensor_value("orientation_left"), dtype=np.float32),
        np.asarray(parser.get_sensor_value("ang_vel_left"), dtype=np.float32),
        np.asarray(parser.get_sensor_value("accel_left"), dtype=np.float32),
    ]
    imu = np.concatenate(values).astype(np.float32, copy=False)
    if imu.shape != (10,):
        raise RuntimeError(f"MuJoCo IMU must have 10 values, got {imu.shape}")
    return imu


def _finger_normal_force_map(parser, prefix: str, normal_axis: int) -> np.ndarray:
    result = np.zeros(TACTILE_SHAPE[0] * TACTILE_SHAPE[1], dtype=np.float32)
    names = set(parser.sensor_names)
    for index in range(result.size):
        name = f"{prefix}{index:03d}"
        if name not in names:
            raise RuntimeError(f"MuJoCo tactile sensor is missing: {name}")
        # MuJoCo force sensors report in the site's local frame. The taxel
        # contact normal is local Z; absolute value makes left/right polarity
        # identical while excluding tangential inertial/shear components.
        force_xyz = np.asarray(parser.get_sensor_value(name), dtype=np.float32)
        result[index] = abs(float(force_xyz[normal_axis]))
    # XML indices advance over 8 physical columns first, then 16 rows.
    # The public [8, 16] convention is [column, row].
    return result.reshape(16, 8).T


def read_tactile_normal(
    parser, *, normal_axis: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    if normal_axis not in (0, 1, 2):
        raise ValueError("normal_axis must be 0, 1 or 2")
    return (
        _finger_normal_force_map(parser, LEFT_PREFIX, normal_axis),
        _finger_normal_force_map(parser, RIGHT_PREFIX, normal_axis),
    )


def read_tactile_contact_projection(
    parser, *, projection_sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Project a contact patch, rather than sparse solver points, onto [8, 16]."""

    if projection_sigma <= 0:
        raise ValueError("projection_sigma must be positive")
    model, data = parser.model, parser.data
    result = []
    wrench = np.zeros(6, dtype=np.float64)
    for side in ("left", "right"):
        pad_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"touch_base_{side}"
        )
        if pad_id < 0:
            raise RuntimeError(f"continuous tactile geom is missing: touch_base_{side}")
        site_ids = np.asarray(
            [
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_SITE,
                    f"touch_point_{side}_{index:03d}",
                )
                for index in range(128)
            ],
            dtype=np.int32,
        )
        if np.any(site_ids < 0):
            raise RuntimeError(f"{side} tactile projection sites are incomplete")
        pad_position = np.asarray(data.geom_xpos[pad_id], dtype=np.float64)
        pad_rotation = np.asarray(data.geom_xmat[pad_id], dtype=np.float64).reshape(3, 3)
        site_positions = np.asarray(data.site_xpos[site_ids], dtype=np.float64)
        site_local = (site_positions - pad_position) @ pad_rotation
        contact_points: list[np.ndarray] = []
        contact_forces: list[float] = []
        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]
            if pad_id not in (int(contact.geom1), int(contact.geom2)):
                continue
            mujoco.mj_contactForce(model, data, contact_index, wrench)
            normal_force = max(0.0, float(wrench[0]))
            if normal_force == 0.0:
                continue
            contact_points.append(
                (np.asarray(contact.pos, dtype=np.float64) - pad_position) @ pad_rotation
            )
            contact_forces.append(normal_force)
        if not contact_points:
            result.append(np.zeros((8, 16), dtype=np.float32))
            continue
        points = np.asarray(contact_points)
        # Box-box contact commonly returns four corner points. Treat their
        # bounding rectangle as a pressure patch and use a smooth distance to
        # its boundary, preventing four bright corner spots and contact-point
        # switching during lateral motion from becoming visible jumps.
        lo = points[:, :2].min(axis=0)
        hi = points[:, :2].max(axis=0)
        inside = np.maximum(np.maximum(lo - site_local[:, :2], site_local[:, :2] - hi), 0.0)
        distance_squared = np.sum(inside * inside, axis=1)
        weights = np.exp(-0.5 * distance_squared / (projection_sigma * projection_sigma))
        weight_sum = float(weights.sum())
        if weight_sum > 0:
            flat = weights * float(np.sum(contact_forces)) / weight_sum
            result.append(flat.reshape(16, 8).T.astype(np.float32))
        else:
            result.append(np.zeros((8, 16), dtype=np.float32))
    return result[0], result[1]


@dataclass(frozen=True)
class TactileProcessingConfig:
    signal_source: str = "continuous_contact_projection"
    normal_axis: int = 2
    projection_sigma: float = 0.0025
    clip_max: float = 25.0
    temporal_ema_alpha: float = 0.25
    spatial_smoothing: bool = True
    contact_time_constant: float = 0.01

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | None):
        value = dict(value or {})
        config = cls(
            signal_source=str(
                value.get("signal_source", "continuous_contact_projection")
            ),
            normal_axis=int(value.get("normal_axis", 2)),
            projection_sigma=float(value.get("projection_sigma", 0.0025)),
            clip_max=float(value.get("clip_max", 25.0)),
            temporal_ema_alpha=float(value.get("temporal_ema_alpha", 0.25)),
            spatial_smoothing=bool(value.get("spatial_smoothing", True)),
            contact_time_constant=float(
                value.get("contact_time_constant", 0.01)
            ),
        )
        if config.normal_axis not in (0, 1, 2):
            raise ValueError("tactile normal_axis must be 0, 1 or 2")
        if config.signal_source not in {
            "continuous_contact_projection",
            "legacy_force_sensor",
        }:
            raise ValueError(
                "tactile signal_source must be continuous_contact_projection "
                "or legacy_force_sensor"
            )
        if config.projection_sigma <= 0:
            raise ValueError("tactile projection_sigma must be positive")
        if config.clip_max <= 0:
            raise ValueError("tactile clip_max must be positive")
        if not 0 < config.temporal_ema_alpha <= 1:
            raise ValueError("tactile temporal_ema_alpha must be in (0, 1]")
        if config.contact_time_constant <= 0:
            raise ValueError("tactile contact_time_constant must be positive")
        return config

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": 2,
            "signal_source": self.signal_source,
            "grid_layout": "physical_columns_x_rows_8x16",
            "normal_axis": self.normal_axis,
            "projection_sigma": self.projection_sigma,
            "clip_max": self.clip_max,
            "temporal_ema_alpha": self.temporal_ema_alpha,
            "spatial_smoothing": self.spatial_smoothing,
            "contact_time_constant": self.contact_time_constant,
        }


def write_or_validate_processing_spec(
    directory: str | Path, value: Mapping[str, object] | None
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / TACTILE_PROCESSING_SPEC_NAME
    expected = TactileProcessingConfig.from_mapping(value).to_dict()
    if path.is_file():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(
                f"tactile processing mismatch at {path}: "
                f"stored={actual}, configured={expected}"
            )
    else:
        path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return path


def validate_processing_spec(
    directory: str | Path, value: Mapping[str, object] | None
) -> None:
    path = Path(directory) / TACTILE_PROCESSING_SPEC_NAME
    if not path.is_file():
        raise FileNotFoundError(f"tactile processing spec is missing: {path}")
    expected = TactileProcessingConfig.from_mapping(value).to_dict()
    actual = json.loads(path.read_text(encoding="utf-8"))
    if actual != expected:
        raise ValueError(
            f"tactile processing mismatch: stored={actual}, configured={expected}"
        )


def spatial_smooth(value: np.ndarray) -> np.ndarray:
    """Apply a normalized separable [1,2,1] Gaussian kernel."""

    value = np.asarray(value, dtype=np.float32).reshape(TACTILE_SHAPE)
    padded = np.pad(value, ((1, 1), (1, 1)), mode="edge")
    # Outer product([1,2,1], [1,2,1]) / 16.
    result = (
        padded[:-2, :-2]
        + 2 * padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + 2 * padded[1:-1, :-2]
        + 4 * padded[1:-1, 1:-1]
        + 2 * padded[1:-1, 2:]
        + padded[2:, :-2]
        + 2 * padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 16.0
    return result.astype(np.float32, copy=False)


class TactileSignalProcessor:
    """400 Hz accumulation followed by canonical 50 Hz filtering."""

    def __init__(self, config: TactileProcessingConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._left_sum = np.zeros(TACTILE_SHAPE, dtype=np.float64)
        self._right_sum = np.zeros(TACTILE_SHAPE, dtype=np.float64)
        self._sample_count = 0
        self._left_ema: np.ndarray | None = None
        self._right_ema: np.ndarray | None = None
        self.latest_raw_left = np.zeros(TACTILE_SHAPE, dtype=np.float32)
        self.latest_raw_right = np.zeros(TACTILE_SHAPE, dtype=np.float32)

    def update(self, left: np.ndarray, right: np.ndarray) -> None:
        left = np.asarray(left, dtype=np.float32).reshape(TACTILE_SHAPE)
        right = np.asarray(right, dtype=np.float32).reshape(TACTILE_SHAPE)
        self.latest_raw_left = left.copy()
        self.latest_raw_right = right.copy()
        self._left_sum += np.clip(left, 0.0, self.config.clip_max)
        self._right_sum += np.clip(right, 0.0, self.config.clip_max)
        self._sample_count += 1

    def consume(self) -> tuple[np.ndarray, np.ndarray]:
        if self._sample_count == 0:
            return (
                np.zeros(TACTILE_SHAPE, dtype=np.float32)
                if self._left_ema is None
                else self._left_ema.copy(),
                np.zeros(TACTILE_SHAPE, dtype=np.float32)
                if self._right_ema is None
                else self._right_ema.copy(),
            )
        left = (self._left_sum / self._sample_count).astype(np.float32)
        right = (self._right_sum / self._sample_count).astype(np.float32)
        self._left_sum.fill(0)
        self._right_sum.fill(0)
        self._sample_count = 0
        if self.config.spatial_smoothing:
            left, right = spatial_smooth(left), spatial_smooth(right)
        alpha = self.config.temporal_ema_alpha
        if self._left_ema is None:
            self._left_ema, self._right_ema = left, right
        else:
            self._left_ema = alpha * left + (1.0 - alpha) * self._left_ema
            self._right_ema = alpha * right + (1.0 - alpha) * self._right_ema
        return self._left_ema.copy(), self._right_ema.copy()
