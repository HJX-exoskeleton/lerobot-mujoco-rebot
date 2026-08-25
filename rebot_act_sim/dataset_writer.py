"""Safe LeRobot dataset lifecycle for simulation demonstrations."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from .schema import FPS, build_lerobot_features, validate_frame
from .sensors import write_or_validate_processing_spec


def overwrite_lerobot_dataset(root: str | Path) -> bool:
    root = Path(root).expanduser()
    if not root.exists():
        return False
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"refusing to overwrite unsafe dataset path: {root}")
    info = root / "meta" / "info.json"
    if not info.is_file():
        if not any(root.iterdir()):
            root.rmdir()
            return True
        raise FileExistsError(f"refusing to remove non-LeRobot directory: {root}")
    shutil.rmtree(root)
    return True


def _reset_empty_interrupted_dataset(root: Path) -> None:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if int(info.get("total_episodes", 0)) != 0:
        return
    if list(root.rglob("*.parquet")) or list(root.rglob("*.mp4")):
        raise RuntimeError(f"zero-episode dataset contains committed files: {root}")
    shutil.rmtree(root)


class SimACTDatasetWriter:
    def __init__(
        self,
        *,
        repo_id: str,
        root: str | Path,
        task: str,
        fps: int = FPS,
        tactile_processing: dict | None = None,
    ):
        self.root = Path(root)
        self.task = task.strip()
        if not self.task:
            raise ValueError("task cannot be empty")
        if fps <= 0:
            raise ValueError("fps must be positive")
        if self.root.exists() and not (self.root / "meta" / "info.json").is_file():
            raise FileExistsError(f"dataset root is not a LeRobot dataset: {self.root}")
        _reset_empty_interrupted_dataset(self.root)
        info = self.root / "meta" / "info.json"
        self.dataset = (
            LeRobotDataset(repo_id, root=self.root)
            if info.is_file()
            else LeRobotDataset.create(
                repo_id=repo_id,
                root=self.root,
                robot_type="rebot_sim",
                fps=fps,
                features=build_lerobot_features(include_auxiliary=True),
                # Keep image encoding in this process. Forked workers and an
                # active GLFW/OpenGL context are an unsafe combination and can
                # terminate collection with SIGSEGV on Linux GPU drivers.
                # Four threads per camera follows LeRobot's recommendation.
                image_writer_threads=8,
                image_writer_processes=0,
            )
        )
        expected = set(build_lerobot_features(include_auxiliary=True))
        missing = expected.difference(self.dataset.meta.features)
        if missing:
            raise ValueError(f"incompatible existing dataset; missing {sorted(missing)}")
        if int(self.dataset.meta.fps) != int(fps):
            raise ValueError(
                f"existing dataset uses {self.dataset.meta.fps} Hz, expected {fps} Hz; "
                "use a new dataset root or explicitly overwrite it"
            )
        write_or_validate_processing_spec(self.root, tactile_processing)
        self.frames_in_buffer = 0

    def add_frame(self, frame: dict) -> None:
        validate_frame(frame)
        self.dataset.add_frame(frame, task=self.task)
        self.frames_in_buffer += 1

    def save_episode(self) -> None:
        if self.frames_in_buffer <= 0:
            raise RuntimeError("cannot save an empty episode")
        self.dataset.save_episode()
        self.frames_in_buffer = 0

    def discard_episode(self) -> None:
        # LeRobot writes camera frames asynchronously.  Deleting the episode
        # directory while workers still hold pending writes races with
        # shutil.rmtree and can leave the directory non-empty.
        image_writer = getattr(self.dataset, "image_writer", None)
        if image_writer is not None:
            image_writer.wait_until_done()
        self.dataset.clear_episode_buffer()
        self.frames_in_buffer = 0
