"""Validate metadata and every decoded frame in a simulation dataset."""

from __future__ import annotations

import argparse
import json

import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from rebot_act_sim.config import DEFAULT_CONFIG, load_config, resolve_project_path
from rebot_act_sim.schema import POLICY_FEATURE_KEYS, validate_frame


def inspect(config_path: str) -> dict:
    raw = load_config(config_path)
    cfg = raw["dataset"]
    root = resolve_project_path(cfg["root"])
    dataset = LeRobotDataset(str(cfg["repo_id"]), root=root)
    ranges: dict[str, dict[str, float]] = {}
    zero_tactile_left = 0
    zero_tactile_right = 0
    for index in range(len(dataset)):
        sample = dataset[index]
        frame = {}
        for key in POLICY_FEATURE_KEYS:
            value = sample[key]
            if key.startswith("observation.") and value.ndim == 3:
                frame[key] = (
                    value.permute(1, 2, 0).mul(255).round().clamp(0, 255).byte().numpy()
                )
            else:
                frame[key] = value.numpy().astype(np.float32)
        for key in (
            "sensor.joint_velocity",
            "sensor.gripper_feedback",
            "sensor.imu",
            "sensor.tactile_left",
            "sensor.tactile_right",
            "sensor.tactile_left_raw",
            "sensor.tactile_right_raw",
            "sensor.sim_time",
            "episode.object_initial_position",
        ):
            expected = np.float64 if key == "sensor.sim_time" else np.float32
            value = sample[key].numpy().astype(expected)
            # LeRobot decodes one-element numeric features as scalar tensors.
            frame[key] = value.reshape(1) if key == "sensor.sim_time" else value
        validate_frame(frame)
        if not np.any(frame["sensor.tactile_left"]):
            zero_tactile_left += 1
        if not np.any(frame["sensor.tactile_right"]):
            zero_tactile_right += 1
        for key in (
            "observation.state",
            "action",
            "sensor.imu",
            "sensor.tactile_left",
            "sensor.tactile_right",
            "sensor.tactile_left_raw",
            "sensor.tactile_right_raw",
        ):
            value = np.asarray(frame[key], dtype=np.float64)
            item = ranges.setdefault(key, {"min": float("inf"), "max": float("-inf")})
            item["min"] = min(item["min"], float(value.min()))
            item["max"] = max(item["max"], float(value.max()))
    return {
        "root": str(root),
        "episodes": dataset.num_episodes,
        "frames": len(dataset),
        "fps": dataset.fps,
        "zero_tactile_left_frames": zero_tactile_left,
        "zero_tactile_right_frames": zero_tactile_right,
        "ranges": ranges,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--json-report")
    args = parser.parse_args()
    report = inspect(args.config)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json_report:
        path = resolve_project_path(args.json_report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
