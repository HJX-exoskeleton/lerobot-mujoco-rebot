"""Non-blocking, self-describing recorder for ACT real-robot deployments.

The control thread only copies one sample into a bounded queue. A background
thread writes compressed, column-oriented NPZ chunks, so an interrupted run
keeps every chunk that was already committed.
"""

from __future__ import annotations

import argparse
import json
import platform
import queue
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


FORMAT_NAME = "rebot-policy-run"
FORMAT_VERSION = "1.0"


def add_recording_arguments(
    parser: argparse.ArgumentParser, *, default_root: Path
) -> None:
    group = parser.add_argument_group("策略推理实验记录")
    group.add_argument(
        "--record",
        action="store_true",
        help="显式开启真机状态、动作、时序和多模态传感器记录（默认关闭）",
    )
    group.add_argument(
        "--record-root",
        type=Path,
        default=default_root,
        help="每次运行自动创建唯一子目录",
    )
    group.add_argument("--run-name", default="", help="实验名/消融名，写入目录和元数据")
    group.add_argument(
        "--record-images",
        action="store_true",
        help="额外保存双相机MP4；默认关闭以降低实时I/O",
    )
    group.add_argument(
        "--record-chunk-size", type=int, default=250, help="每个NPZ分块的控制帧数"
    )
    group.add_argument(
        "--record-queue-size", type=int, default=1000, help="异步写盘队列容量"
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _git_info(project_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        result["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        result["dirty"] = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        result["commit"] = None
        result["dirty"] = None
    return result


@dataclass
class RecorderStats:
    enqueued: int = 0
    written: int = 0
    dropped: int = 0
    chunks: int = 0
    video_segments: int = 0


class PolicyRunRecorder:
    """Asynchronously persist synchronized policy/control samples."""

    def __init__(
        self,
        *,
        root: Path,
        run_name: str,
        entrypoint: str,
        args: argparse.Namespace,
        project_root: Path,
        record_images: bool,
        chunk_size: int,
        queue_size: int,
        rate_hz: float,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        if chunk_size <= 0 or queue_size <= 0:
            raise ValueError("record chunk/queue size必须大于0")
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_name)
        suffix = f"_{safe_name}" if safe_name else ""
        candidate = Path(root).resolve() / f"{stamp}{suffix}"
        index = 1
        while candidate.exists():
            candidate = Path(f"{candidate}_{index:02d}")
            index += 1
        self.run_dir = candidate
        self.data_dir = self.run_dir / "data"
        self.video_dir = self.run_dir / "videos"
        self.data_dir.mkdir(parents=True)
        if record_images:
            self.video_dir.mkdir()

        self.record_images = bool(record_images)
        self.chunk_size = int(chunk_size)
        self.rate_hz = float(rate_hz)
        self.stats = RecorderStats()
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(queue_size)
        self._error: BaseException | None = None
        self._closed = False
        self._start_perf = time.perf_counter()
        self._start_wall_ns = time.time_ns()
        self._metadata = {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "status": "recording",
            "entrypoint": entrypoint,
            "run_name": run_name,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "clock": {
                "monotonic_origin_s": self._start_perf,
                "unix_origin_ns": self._start_wall_ns,
                "timestamp_convention": "seconds from monotonic origin",
            },
            "command": sys.argv,
            "arguments": _jsonable(vars(args)),
            "host": {
                "hostname": platform.node(),
                "platform": platform.platform(),
                "python": platform.python_version(),
            },
            "git": _git_info(project_root),
            "modalities": {
                "joint_state": True,
                "policy_action": True,
                "imu": bool(getattr(args, "imu", False)),
                "tactile": bool(getattr(args, "tactile", False)),
                "camera_video": self.record_images,
            },
            "nominal_rate_hz": self.rate_hz,
            "units": {
                "joint_position": "rad",
                "joint_velocity": "rad/s",
                "action": "rad",
                "time": "s",
                "latency": "ms",
                "imu_angular_velocity": "sensor_native",
                "imu_acceleration": "sensor_native",
                "tactile": "baseline-subtracted_sensor_native",
            },
            "extra": _jsonable(extra_metadata or {}),
        }
        self._write_json("metadata.json", self._metadata)
        self._thread = threading.Thread(
            target=self._writer_loop, name="policy_run_writer", daemon=True
        )
        self._thread.start()

    def _write_json(self, name: str, value: dict[str, Any]) -> None:
        path = self.run_dir / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(_jsonable(value), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def record(self, sample: dict[str, Any]) -> None:
        if self._closed:
            return
        if self._error is not None:
            raise RuntimeError("策略实验后台写盘失败") from self._error
        copied: dict[str, Any] = {}
        for key, value in sample.items():
            copied[key] = np.asarray(value).copy() if isinstance(value, np.ndarray) else value
        copied.setdefault("time_s", time.perf_counter() - self._start_perf)
        try:
            self._queue.put_nowait(copied)
            self.stats.enqueued += 1
        except queue.Full:
            self.stats.dropped += 1

    def relative_timestamp(self, monotonic_timestamp: float) -> float:
        """Convert a ``time.perf_counter()`` timestamp to run-relative seconds."""
        return float(monotonic_timestamp) - self._start_perf

    @staticmethod
    def _stack_chunk(samples: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        keys = sorted(set().union(*(sample.keys() for sample in samples)))
        output: dict[str, np.ndarray] = {}
        for key in keys:
            present = [sample.get(key) for sample in samples]
            exemplar = next((value for value in present if value is not None), None)
            if exemplar is None:
                continue
            array = np.asarray(exemplar)
            fill = np.full(array.shape, np.nan, dtype=np.float32)
            values = [fill if value is None else np.asarray(value) for value in present]
            try:
                output[key] = np.stack(values)
            except ValueError as exc:
                raise ValueError(f"日志字段{key!r}形状在运行中发生变化") from exc
        return output

    def _write_chunk(self, samples: list[dict[str, Any]]) -> None:
        images: dict[str, list[np.ndarray]] = {
            "cam_high": [],
            "cam_wrist": [],
        }
        if self.record_images:
            for sample in samples:
                for key, camera_name in (
                    ("image_high_rgb", "cam_high"),
                    ("image_wrist_rgb", "cam_wrist"),
                ):
                    image = sample.pop(key, None)
                    if image is not None:
                        images[camera_name].append(np.asarray(image, dtype=np.uint8))
        payload = self._stack_chunk(samples)
        path = self.data_dir / f"chunk_{self.stats.chunks:06d}.npz"
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        temporary.replace(path)

        # MP4 stores its seek index at finalization. One run-long file is
        # therefore unusable after SIGKILL/os._exit. Finalize a short segment
        # for every data chunk and atomically publish it only after release().
        if self.record_images:
            segment_index = self.stats.chunks
            for camera_name, frames in images.items():
                if not frames:
                    continue
                final_path = (
                    self.video_dir / f"{camera_name}_{segment_index:06d}.mp4"
                )
                partial_path = (
                    self.video_dir / f".{camera_name}_{segment_index:06d}.partial.mp4"
                )
                height, width = frames[0].shape[:2]
                writer = cv2.VideoWriter(
                    str(partial_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.rate_hz,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"无法创建视频分段: {partial_path.name}")
                try:
                    for rgb in frames:
                        if rgb.shape[:2] != (height, width):
                            raise ValueError(
                                f"{camera_name}视频分辨率在运行中发生变化"
                            )
                        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                finally:
                    writer.release()
                partial_path.replace(final_path)
                self.stats.video_segments += 1
        self.stats.written += len(samples)
        self.stats.chunks += 1

    def _writer_loop(self) -> None:
        pending: list[dict[str, Any]] = []
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                pending.append(item)
                if len(pending) >= self.chunk_size:
                    self._write_chunk(pending)
                    pending = []
            if pending:
                self._write_chunk(pending)
        except BaseException as exc:
            self._error = exc

    def close(self, *, status: str = "completed") -> Path:
        if self._closed:
            return self.run_dir
        self._closed = True
        self._queue.put(None)
        self._thread.join()
        if self._error is not None:
            status = "write_error"
        summary = asdict(self.stats)
        summary.update(
            {
                "status": status,
                "duration_s": time.perf_counter() - self._start_perf,
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "disk_bytes": sum(
                    path.stat().st_size
                    for path in self.run_dir.rglob("*")
                    if path.is_file()
                ),
            }
        )
        self._metadata["status"] = status
        self._metadata["summary"] = summary
        self._write_json("metadata.json", self._metadata)
        self._write_json("summary.json", summary)
        if self._error is not None:
            raise RuntimeError("策略实验后台写盘失败") from self._error
        return self.run_dir
