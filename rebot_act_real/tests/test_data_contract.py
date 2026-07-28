from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pytest

from rebot_act_real.contracts import CameraSample, RobotFeedback, RobotTarget
from rebot_act_real.dataset_writer import overwrite_lerobot_dataset
from rebot_act_real.sample import SynchronizedSample, to_lerobot_frame
from rebot_act_real.schema import build_lerobot_features, validate_frame


def make_sample(*, camera_skew_seconds: float = 0.0) -> SynchronizedSample:
    now = time.monotonic()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    return SynchronizedSample(
        agent=CameraSample(image, 1, now),
        wrist=CameraSample(image, 2, now - camera_skew_seconds),
        feedback=RobotFeedback(np.zeros(6), np.zeros(6), -5.8, 0.0, now),
        target=RobotTarget(np.ones(6), -3.0, now),
        imu=np.zeros(10, dtype=np.float32),
        imu_magnetometer=np.zeros(3, dtype=np.float32),
        imu_euler=np.zeros(3, dtype=np.float32),
        imu_barometer=np.zeros(4, dtype=np.float32),
        tactile=np.zeros((12, 30), dtype=np.float32),
        imu_frame_id=3,
        tactile_frame_id=4,
        imu_timestamp=now,
        tactile_timestamp=now,
        timestamp=now,
    )


def test_sample_converts_to_canonical_frame():
    frame = to_lerobot_frame(make_sample())
    validate_frame(frame)
    assert frame["observation.image"].shape == (256, 256, 3)
    assert frame["observation.wrist_image"].shape == (256, 256, 3)
    assert frame["observation.state"].shape == (6,)
    assert frame["action"].shape == (7,)
    assert frame["action"][-1] == pytest.approx(-3.0)
    assert frame["sensor.imu"].shape == (10,)
    assert frame["sensor.tactile"].shape == (12, 30)


def test_camera_skew_is_rejected():
    with pytest.raises(RuntimeError, match="camera skew"):
        to_lerobot_frame(make_sample(camera_skew_seconds=0.2))


def test_non_finite_action_is_rejected():
    sample = make_sample()
    invalid_target = RobotTarget(
        joint_position=np.full(6, np.nan),
        gripper_position=-3.0,
        timestamp=sample.timestamp,
    )
    invalid = replace(sample, target=invalid_target)
    with pytest.raises(ValueError, match="NaN or Inf"):
        to_lerobot_frame(invalid)


def test_auxiliary_sensors_are_not_policy_features():
    from lerobot.common.datasets.utils import dataset_to_policy_features

    policy_features = dataset_to_policy_features(build_lerobot_features())
    assert set(policy_features) == {
        "observation.image",
        "observation.wrist_image",
        "observation.state",
        "action",
    }


def test_overwrite_only_deletes_lerobot_dataset(tmp_path):
    dataset_root = tmp_path / "dataset"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "meta" / "info.json").write_text("{}", encoding="utf-8")
    (dataset_root / "data").mkdir()
    (dataset_root / "data" / "episode.parquet").write_bytes(b"test")

    assert overwrite_lerobot_dataset(dataset_root)
    assert not dataset_root.exists()

    ordinary_root = tmp_path / "ordinary"
    ordinary_root.mkdir()
    (ordinary_root / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="non-LeRobot"):
        overwrite_lerobot_dataset(ordinary_root)
    assert (ordinary_root / "keep.txt").read_text(encoding="utf-8") == "keep"
