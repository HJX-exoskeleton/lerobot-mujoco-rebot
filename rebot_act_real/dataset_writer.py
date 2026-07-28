"""Small lifecycle wrapper around LeRobotDataset episode recording."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from .schema import FPS, build_lerobot_features, validate_frame


def overwrite_lerobot_dataset(root: str | Path) -> bool:
    """Delete an existing local LeRobot dataset after explicit user opt-in.

    Refuse to delete an arbitrary directory: an existing non-empty target must
    contain ``meta/info.json`` and therefore identify itself as a LeRobot
    dataset. Returns whether a directory was removed.
    """

    root = Path(root).expanduser()
    if not root.exists():
        return False
    if root.is_symlink():
        raise ValueError(f"refusing to overwrite a symlink dataset root: {root}")
    if not root.is_dir():
        raise ValueError(f"dataset root is not a directory: {root}")
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        if not any(root.iterdir()):
            root.rmdir()
            return True
        raise FileExistsError(
            "refusing to overwrite a non-LeRobot directory because "
            f"meta/info.json is missing: {root}"
        )
    shutil.rmtree(root)
    return True


def _reset_interrupted_empty_dataset(root: Path, info_path: Path) -> bool:
    """Reset a zero-episode dataset left by an interrupted first recording.

    This LeRobot version cannot reload a local dataset until its first Parquet
    file exists. It otherwise incorrectly falls back to the Hub. Recreating is
    safe only when metadata reports zero episodes and no committed data/video
    exists. The caller will immediately invoke ``LeRobotDataset.create``.
    """

    info = json.loads(info_path.read_text(encoding="utf-8"))
    total_episodes = int(info.get("total_episodes", 0))
    if total_episodes != 0:
        return False
    committed_files = [*root.rglob("*.parquet"), *root.rglob("*.mp4")]
    if committed_files:
        raise RuntimeError(
            "dataset reports zero episodes but contains committed data; refusing "
            f"automatic reset: {[str(path) for path in committed_files]}"
        )
    shutil.rmtree(root)
    return True


class RealACTDatasetWriter:
    def __init__(
        self,
        *,
        repo_id: str,
        root: str | Path,
        task: str,
        fps: int = FPS,
    ):
        self.root = Path(root)
        self.task = task.strip()
        if not self.task:
            raise ValueError("task cannot be empty")
        if fps <= 0:
            raise ValueError("fps must be positive")

        info_path = self.root / "meta" / "info.json"
        if self.root.exists() and not info_path.is_file():
            raise FileExistsError(
                f"dataset root exists but is not a LeRobot dataset: {self.root}"
            )
        if info_path.is_file():
            _reset_interrupted_empty_dataset(self.root, info_path)
        info_path = self.root / "meta" / "info.json"
        self.dataset = (
            LeRobotDataset(repo_id, root=self.root)
            if info_path.is_file()
            else LeRobotDataset.create(
                repo_id=repo_id,
                root=self.root,
                robot_type="rebot",
                fps=fps,
                features=build_lerobot_features(include_auxiliary=True),
                image_writer_threads=10,
                image_writer_processes=5,
            )
        )
        expected = set(build_lerobot_features(include_auxiliary=True))
        missing = expected.difference(self.dataset.meta.features)
        if missing:
            raise ValueError(
                "existing dataset schema is incompatible; missing features: "
                f"{sorted(missing)}. Use a new --root for the multimodal dataset."
            )
        self.frames_in_buffer = 0

    def add_frame(self, frame: dict) -> None:
        validate_frame(frame)
        self.dataset.add_frame(frame, task=self.task)
        self.frames_in_buffer += 1

    def save_episode(self) -> None:
        if self.frames_in_buffer == 0:
            raise RuntimeError("cannot save an empty episode")
        self.dataset.save_episode()
        self.frames_in_buffer = 0

    def discard_episode(self) -> None:
        self.dataset.clear_episode_buffer()
        self.frames_in_buffer = 0
