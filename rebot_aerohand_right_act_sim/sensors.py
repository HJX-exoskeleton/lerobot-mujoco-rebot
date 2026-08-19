"""Canonical MuJoCo sensor reading: wrist IMU and per-region hand contact forces."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import mujoco
import numpy as np

from .schema import HAND_CONTACT_DIM, HAND_CONTACT_REGIONS

HAND_CONTACT_PROCESSING_SPEC_NAME = (
    "rebot_aerohand_right_hand_contact_processing.json"
)

# Hand contact geoms are classified by their body name. Bodies below ``palm``
# containing one of these keys map to that region; everything else on the hand
# (the palm itself and its collision geoms) maps to the last region.
_REGION_KEYS = ("thumb", "index", "middle", "ring", "pinky")


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


def read_hand_feedback(model, data) -> np.ndarray:
    """Measured value of each of the seven hand actuators.

    The XML declares ``tendonpos`` sensors for the six tendons and a
    ``jointpos`` sensor for thumb abduction, in actuator order.
    """
    values = np.asarray(
        [
            float(data.sensor(name).data[0])
            for name in (
                "len_if",
                "len_mf",
                "len_rf",
                "len_pf",
                "len_th_abd",
                "len_th1",
                "len_th2",
            )
        ],
        dtype=np.float32,
    )
    if values.shape != (7,):
        raise RuntimeError(f"hand feedback must have 7 values, got {values.shape}")
    return values


def classify_hand_geom_regions(model, *, hand_root_name: str = "palm") -> dict[int, int]:
    """Map each hand collision geom id to a contact region index.

    Geoms whose body is ``palm`` or one of its descendants are hand geoms.
    Bodies carrying a finger key map to that finger's region; all remaining
    hand geoms map to the palm region. Non-hand geoms are excluded.
    """
    hand_root = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, hand_root_name
    )
    if hand_root < 0:
        raise RuntimeError(f"model is missing hand root body: {hand_root_name}")
    body_region: dict[int, int] = {}
    for body_id in range(1, model.nbody):
        ancestor = body_id
        while ancestor > 0 and ancestor != hand_root:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor != hand_root:
            continue
        name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                or "").lower()
        region = next(
            (index for index, key in enumerate(_REGION_KEYS) if key in name),
            len(HAND_CONTACT_REGIONS) - 1,
        )
        body_region[body_id] = region
    return {
        geom_id: body_region[int(model.geom_bodyid[geom_id])]
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in body_region
    }


@dataclass(frozen=True)
class HandContactProcessingConfig:
    regions: tuple[str, ...] = HAND_CONTACT_REGIONS
    clip_max: float = 25.0
    temporal_ema_alpha: float = 0.25

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | None):
        value = dict(value or {})
        config = cls(
            regions=tuple(value.get("regions", HAND_CONTACT_REGIONS)),
            clip_max=float(value.get("clip_max", 25.0)),
            temporal_ema_alpha=float(value.get("temporal_ema_alpha", 0.25)),
        )
        if tuple(config.regions) != tuple(HAND_CONTACT_REGIONS):
            raise ValueError(
                f"hand contact regions must be {HAND_CONTACT_REGIONS}, "
                f"got {config.regions}"
            )
        if config.clip_max <= 0:
            raise ValueError("hand contact clip_max must be positive")
        if not 0 < config.temporal_ema_alpha <= 1:
            raise ValueError("hand contact temporal_ema_alpha must be in (0, 1]")
        return config

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": 1,
            "regions": list(self.regions),
            "clip_max": self.clip_max,
            "temporal_ema_alpha": self.temporal_ema_alpha,
        }


def write_or_validate_processing_spec(
    directory: str | Path, value: Mapping[str, object] | None
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / HAND_CONTACT_PROCESSING_SPEC_NAME
    expected = HandContactProcessingConfig.from_mapping(value).to_dict()
    if path.is_file():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(
                f"hand contact processing mismatch at {path}: "
                f"stored={actual}, configured={expected}"
            )
    else:
        path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return path


def validate_processing_spec(
    directory: str | Path, value: Mapping[str, object] | None
) -> None:
    path = Path(directory) / HAND_CONTACT_PROCESSING_SPEC_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"hand contact processing spec is missing: {path}"
        )
    expected = HandContactProcessingConfig.from_mapping(value).to_dict()
    actual = json.loads(path.read_text(encoding="utf-8"))
    if actual != expected:
        raise ValueError(
            f"hand contact processing mismatch: stored={actual}, configured={expected}"
        )


class HandContactSignalProcessor:
    """Per-physics-step region accumulation followed by 50 Hz EMA filtering.

    At each 1000 Hz physics step the environment calls :meth:`update` with a
    per-region normal-force vector.  At the 50 Hz control tick, :meth:`consume`
    returns the clipped step average smoothed by a first-order temporal EMA.
    """

    def __init__(self, config: HandContactProcessingConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._sum = np.zeros(HAND_CONTACT_DIM, dtype=np.float64)
        self._sample_count = 0
        self._ema: np.ndarray | None = None

    def update(self, region_forces: np.ndarray) -> None:
        forces = np.asarray(region_forces, dtype=np.float64).reshape(
            HAND_CONTACT_DIM
        )
        self._sum += np.clip(forces, 0.0, self.config.clip_max)
        self._sample_count += 1

    def consume(self) -> np.ndarray:
        if self._sample_count == 0:
            return (
                np.zeros(HAND_CONTACT_DIM, dtype=np.float32)
                if self._ema is None
                else self._ema.copy()
            )
        mean = (self._sum / self._sample_count).astype(np.float32)
        self._sum.fill(0)
        self._sample_count = 0
        alpha = self.config.temporal_ema_alpha
        if self._ema is None:
            self._ema = mean
        else:
            self._ema = alpha * mean + (1.0 - alpha) * self._ema
        return self._ema.copy()
