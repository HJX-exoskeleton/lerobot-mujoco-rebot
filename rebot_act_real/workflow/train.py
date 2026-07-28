#!/usr/bin/env python3
"""Preflight and train LeRobot ACT on the reBot real-world dataset.

Preflight only::

    python -m rebot_act_real.workflow.train --check-only

Start training::

    python -m rebot_act_real.workflow.train
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "act_real_banana.yaml"
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "models"

os.environ.setdefault("HF_HOME", str(DEFAULT_CACHE_ROOT / ".hf_home"))
os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_CACHE_ROOT))
os.environ.setdefault("HF_DATASETS_CACHE", str(DEFAULT_CACHE_ROOT / "datasets"))
os.environ.setdefault("HF_XET_CACHE", str(DEFAULT_CACHE_ROOT / ".xet"))
os.environ.setdefault("HF_ASSETS_CACHE", str(DEFAULT_CACHE_ROOT / ".assets"))
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_CACHE_ROOT / ".matplotlib"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import yaml

from rebot_act_real.schema import FPS


EXPECTED_POLICY_FEATURES = {
    "observation.image",
    "observation.wrist_image",
    "observation.state",
    "action",
}


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def inspect_training_setup(
    config_path: Path,
    *,
    decode_sample: bool = True,
    use_imu: bool = False,
    use_tactile: bool = False,
) -> dict[str, object]:
    from lerobot.common.datasets.factory import resolve_delta_timestamps
    from lerobot.common.datasets.lerobot_dataset import (
        LeRobotDataset,
        LeRobotDatasetMetadata,
    )
    from lerobot.common.datasets.utils import (
        dataset_to_policy_features,
    )
    from lerobot.common.policies.act.configuration_act import ACTConfig

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_cfg = raw["dataset"]
    policy_cfg = raw["policy"]
    if policy_cfg.get("type") != "act":
        raise ValueError(f"policy.type必须为act，实际为{policy_cfg.get('type')!r}")

    root = _resolve_project_path(dataset_cfg["root"])
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot数据集不存在: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if int(info.get("fps", 0)) != FPS:
        raise ValueError(f"ACT真机数据集必须为{FPS}Hz，实际为{info.get('fps')}")
    if int(info.get("total_episodes", 0)) <= 0:
        raise ValueError("数据集没有已保存episode")

    policy_features = set(dataset_to_policy_features(info["features"]))
    if policy_features != EXPECTED_POLICY_FEATURES:
        raise ValueError(
            "ACT策略字段不符合预期："
            f"expected={sorted(EXPECTED_POLICY_FEATURES)}, "
            f"actual={sorted(policy_features)}"
        )

    chunk_size = int(policy_cfg["chunk_size"])
    n_action_steps = int(policy_cfg["n_action_steps"])
    if chunk_size <= 0 or n_action_steps <= 0 or n_action_steps > chunk_size:
        raise ValueError(
            f"ACT动作块参数非法: chunk_size={chunk_size}, "
            f"n_action_steps={n_action_steps}"
        )

    episode_lengths = [
        int(json.loads(line)["length"])
        for line in (root / "meta" / "episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if not episode_lengths or min(episode_lengths) < chunk_size:
        raise ValueError(
            f"最短episode长度必须不小于chunk_size={chunk_size}，"
            f"实际最短={min(episode_lengths) if episode_lengths else 0}"
        )

    metadata = LeRobotDatasetMetadata(str(dataset_cfg["repo_id"]), root=root)
    features = dataset_to_policy_features(metadata.features)
    act_config = ACTConfig(
        input_features={
            key: value for key, value in features.items() if key != "action"
        },
        output_features={"action": features["action"]},
        chunk_size=chunk_size,
        n_action_steps=n_action_steps,
        vision_backbone=str(policy_cfg.get("vision_backbone", "resnet18")),
        pretrained_backbone_weights=policy_cfg.get(
            "pretrained_backbone_weights", "ResNet18_Weights.IMAGENET1K_V1"
        ),
        use_vae=bool(policy_cfg.get("use_vae", True)),
        latent_dim=int(policy_cfg.get("latent_dim", 32)),
        kl_weight=float(policy_cfg.get("kl_weight", 10.0)),
        device=str(policy_cfg.get("device", "cuda")),
    )
    delta_timestamps = resolve_delta_timestamps(act_config, metadata)

    decoded_keys: list[str] = []
    sample_shapes: dict[str, list[int]] = {}
    if decode_sample:
        dataset = LeRobotDataset(
            str(dataset_cfg["repo_id"]),
            root=root,
            delta_timestamps=delta_timestamps,
        )
        sample = dataset[0]
        decoded_keys = sorted(sample)
        for key in EXPECTED_POLICY_FEATURES:
            value = sample[key]
            shape = tuple(getattr(value, "shape", np.asarray(value).shape))
            sample_shapes[key] = list(shape)
        if use_imu:
            sample_shapes["sensor.imu"] = list(sample["sensor.imu"].shape)
            if sample_shapes["sensor.imu"] != [10]:
                raise ValueError(f"sensor.imu形状必须为[10]: {sample_shapes['sensor.imu']}")
        if use_tactile:
            sample_shapes["sensor.tactile"] = list(sample["sensor.tactile"].shape)
            if sample_shapes["sensor.tactile"] != [12, 30]:
                raise ValueError(
                    f"sensor.tactile形状必须为[12,30]: {sample_shapes['sensor.tactile']}"
                )
        if sample_shapes["action"][0] != chunk_size:
            raise ValueError(
                "ACT未来动作窗口长度不匹配："
                f"expected={chunk_size}, actual={sample_shapes['action']}"
            )

    return {
        "root": str(root),
        "repo_id": str(dataset_cfg["repo_id"]),
        "episodes": int(info["total_episodes"]),
        "frames": int(info["total_frames"]),
        "fps": int(info["fps"]),
        "episode_length_min": min(episode_lengths),
        "episode_length_max": max(episode_lengths),
        "policy_features": sorted(policy_features),
        "auxiliary_features": sorted(
            key for key in info["features"] if key.startswith("sensor.")
        ),
        "chunk_size": chunk_size,
        "n_action_steps": n_action_steps,
        "use_imu": use_imu,
        "use_tactile": use_tactile,
        "decoded_keys": decoded_keys,
        "sample_shapes": sample_shapes,
        "output_dir": str(_resolve_project_path(raw["output_dir"])),
        "steps": int(raw["steps"]),
        "batch_size": int(raw["batch_size"]),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--imu", action="store_true", help="启用10维IMU编码分支")
    parser.add_argument("--tactile", action="store_true", help="启用12x30触觉CNN分支")
    parser.add_argument("--sensor-embed-dim", type=int, default=64)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    config = _resolve_project_path(args.config)
    if not config.is_file():
        raise FileNotFoundError(f"ACT训练配置不存在: {config}")
    if args.sensor_embed_dim <= 0:
        raise ValueError("--sensor-embed-dim必须大于0")
    summary = inspect_training_setup(
        config, use_imu=args.imu, use_tactile=args.tactile
    )
    print("ACT真机训练预检通过")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    if summary["episodes"] < 30:
        print("\n⚠️ episode少于30条，适合验证流程，不建议据此判断泛化能力。")
    if args.check_only:
        return

    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    command_module = "rebot_act_real.workflow.train_multimodal"
    temporary_config: Path | None = None
    if args.imu or args.tactile:
        raw = yaml.safe_load(config.read_text(encoding="utf-8"))
        suffix = "_imu_tactile" if args.imu and args.tactile else (
            "_imu" if args.imu else "_tactile"
        )
        output_dir = str(_resolve_project_path(raw["output_dir"]))
        job_name = str(raw.get("job_name", "act_rebot_real"))
        raw["output_dir"] = (
            output_dir if output_dir.endswith(suffix) else output_dir + suffix
        )
        raw["job_name"] = (
            job_name if job_name.endswith(suffix) else job_name + suffix
        )
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix="rebot_act_", delete=False
        )
        with handle:
            yaml.safe_dump(raw, handle, sort_keys=False, allow_unicode=True)
        temporary_config = Path(handle.name)
        env["REBOT_ACT_USE_IMU"] = "1" if args.imu else "0"
        env["REBOT_ACT_USE_TACTILE"] = "1" if args.tactile else "0"
        env["REBOT_ACT_SENSOR_EMBED_DIM"] = str(args.sensor_embed_dim)
        command = [
            sys.executable,
            "-m",
            command_module,
            "--config_path",
            str(temporary_config),
        ]
        print(f"多模态checkpoint输出目录: {raw['output_dir']}")
    else:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "train_model.py"),
            "--config_path",
            str(config),
        ]
    try:
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)
    finally:
        if temporary_config is not None:
            temporary_config.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
