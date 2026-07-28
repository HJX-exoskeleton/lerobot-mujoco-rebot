#!/usr/bin/env python3
"""Preflight and train SmolVLA on the reBot real-world LeRobot dataset.

Check the dataset and policy feature boundary without starting training::

    python -m rebot_smolvla_real.workflow.train --check-only

Start training with the project-local configuration::

    python -m rebot_smolvla_real.workflow.train

Only ``observation.image``, ``observation.wrist_image``,
``observation.state``, language task and ``action`` are used by SmolVLA.
Every ``sensor.*`` field remains stored in the dataset but is excluded by
LeRobot's policy feature conversion.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "smolvla_real_banana.yaml"
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "models"

# Must be configured before importing datasets/LeRobot. Otherwise their import
# may bind to ~/.cache, which is undesirable for this project and can be
# read-only in managed environments.
os.environ.setdefault("HF_HOME", str(DEFAULT_CACHE_ROOT / ".hf_home"))
os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_CACHE_ROOT))
os.environ.setdefault(
    "HF_DATASETS_CACHE", str(DEFAULT_CACHE_ROOT / "datasets")
)
os.environ.setdefault("HF_XET_CACHE", str(DEFAULT_CACHE_ROOT / ".xet"))
os.environ.setdefault("HF_ASSETS_CACHE", str(DEFAULT_CACHE_ROOT / ".assets"))
os.environ.setdefault("TORCH_HOME", str(DEFAULT_CACHE_ROOT / "torch"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import yaml


EXPECTED_POLICY_FEATURES = {
    "observation.image",
    "observation.wrist_image",
    "observation.state",
    "action",
}


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def inspect_training_setup(config_path: Path, *, decode_sample: bool = True) -> dict:
    import datasets
    import pyarrow
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.utils import dataset_to_policy_features

    datasets_major_minor = tuple(
        int(part) for part in datasets.__version__.split(".")[:2]
    )
    pyarrow_major = int(pyarrow.__version__.split(".", 1)[0])
    if datasets_major_minor <= (3, 4) and pyarrow_major >= 20:
        raise RuntimeError(
            f"当前 datasets=={datasets.__version__} 与 "
            f"pyarrow=={pyarrow.__version__} 不兼容本数据集的二维触觉字段。"
            "请在 lerobot_rebot 环境执行：pip install pyarrow==19.0.1"
        )

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_cfg = raw["dataset"]
    root = _resolve_project_path(dataset_cfg["root"])
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot 数据集不存在: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    policy_features = set(dataset_to_policy_features(info["features"]))
    if policy_features != EXPECTED_POLICY_FEATURES:
        raise ValueError(
            "SmolVLA 策略字段不符合预期："
            f"expected={sorted(EXPECTED_POLICY_FEATURES)}, "
            f"actual={sorted(policy_features)}"
        )
    if int(info.get("total_episodes", 0)) <= 0:
        raise ValueError("数据集没有已保存 episode")

    decoded_keys: list[str] = []
    if decode_sample:
        try:
            dataset = LeRobotDataset(str(dataset_cfg["repo_id"]), root=root)
            sample = dataset[0]
            decoded_keys = sorted(sample)
        except TypeError as exc:
            if "maps_as_pydicts" in str(exc):
                raise RuntimeError(
                    "当前 datasets/pyarrow 版本不兼容二维触觉字段。"
                    "请在 lerobot_rebot 环境执行："
                    "pip install pyarrow==19.0.1"
                ) from exc
            raise

    return {
        "root": root,
        "repo_id": str(dataset_cfg["repo_id"]),
        "episodes": int(info["total_episodes"]),
        "frames": int(info["total_frames"]),
        "fps": int(info["fps"]),
        "policy_features": sorted(policy_features),
        "auxiliary_features": sorted(
            key for key in info["features"] if key.startswith("sensor.")
        ),
        "decoded_keys": decoded_keys,
        "output_dir": _resolve_project_path(raw["output_dir"]),
        "steps": int(raw["steps"]),
        "batch_size": int(raw["batch_size"]),
    }


def _training_environment(cache_dir: Path, pretrained: str | None) -> dict[str, str]:
    env = os.environ.copy()
    cache_dir = cache_dir.resolve()
    env.update(
        {
            "HF_HOME": str(PROJECT_ROOT / "models" / ".hf_home"),
            "HF_HUB_CACHE": str(cache_dir),
            "HF_DATASETS_CACHE": str(PROJECT_ROOT / "models" / "datasets"),
            "HF_XET_CACHE": str(PROJECT_ROOT / "models" / ".xet"),
            "HF_ASSETS_CACHE": str(PROJECT_ROOT / "models" / ".assets"),
            "TORCH_HOME": str(PROJECT_ROOT / "models" / "torch"),
            "HF_HUB_DISABLE_XET": "1",
            "HF_HUB_DOWNLOAD_TIMEOUT": "300",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    if pretrained:
        candidate = Path(pretrained).expanduser()
        is_hub_id = (
            pretrained.count("/") == 1
            and not pretrained.startswith((".", "/"))
            and not candidate.exists()
        )
        env["SMOLVLA_PRETRAINED_PATH"] = (
            pretrained if is_hub_id else str(_resolve_project_path(candidate))
        )
    return env


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--pretrained",
        help="SmolVLA Hub ID、本地缓存根目录或 snapshot 目录",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    config = _resolve_project_path(args.config)
    if not config.is_file():
        raise FileNotFoundError(f"训练配置不存在: {config}")
    summary = inspect_training_setup(config)
    print("SmolVLA 真机训练预检通过")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    if summary["episodes"] < 10:
        print(
            "\n⚠️ 当前 episode 数量很少，入口可用于流程验证，"
            "但不建议据此判断真机泛化效果。"
        )
    if args.check_only:
        return

    command = [
        sys.executable,
        str(PROJECT_ROOT / "train_model.py"),
        "--config_path",
        str(config),
    ]
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=_training_environment(args.cache_dir, args.pretrained),
        check=True,
    )


if __name__ == "__main__":
    main()
