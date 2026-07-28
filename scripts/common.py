"""Shared helpers for the terminal versions of the tutorial notebooks."""

from __future__ import annotations

from pathlib import Path

import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EpisodeSampler(torch.utils.data.Sampler):
    """Select every frame belonging to one dataset episode."""

    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        if not 0 <= episode_index < dataset.num_episodes:
            raise ValueError(
                f"episode_index={episode_index} is outside [0, {dataset.num_episodes - 1}]"
            )
        start = dataset.episode_data_index["from"][episode_index].item()
        stop = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(start, stop)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return torch.device(requested)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path
