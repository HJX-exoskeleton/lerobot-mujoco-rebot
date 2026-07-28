#!/usr/bin/env python3
"""Inspect a real reBot ACT LeRobot dataset without connecting to hardware.

Full integrity check of the default dataset::

    python -m rebot_act_real.workflow.inspect_dataset

Inspect another dataset and save a machine-readable report::

    python -m rebot_act_real.workflow.inspect_dataset \
      --root ./data_act_real/rebot_act_banana \
      --json-report ./data_act_real/rebot_act_banana_report.json

The checker streams Parquet row groups instead of loading the whole dataset
into RAM. It validates metadata, episode/frame continuity, embedded images,
robot state/action, IMU, tactile data, frame IDs and sensor timestamps.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_ROOT / "data_act_real" / "rebot_act_banana"
IMAGE_KEYS = ("observation.image", "observation.wrist_image")
NUMERIC_KEYS = (
    "observation.state",
    "action",
    "sensor.joint_velocity",
    "sensor.gripper_feedback",
    "sensor.imu",
    "sensor.imu_magnetometer",
    "sensor.imu_euler",
    "sensor.imu_barometer",
    "sensor.tactile",
    "sensor.frame_ids",
    "sensor.timestamps",
)
INDEX_KEYS = ("timestamp", "frame_index", "episode_index", "index", "task_index")


def _human_bytes(size: int | float) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def _read_jsonlines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSONL格式错误: {path}:{line_number}: {exc}") from exc
    return records


def _directory_sizes(root: Path) -> tuple[int, dict[str, int], int]:
    total = 0
    groups: dict[str, int] = defaultdict(int)
    file_count = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        size = path.stat().st_size
        total += size
        file_count += 1
        relative = path.relative_to(root)
        groups[relative.parts[0] if relative.parts else "."] += size
    return total, dict(sorted(groups.items())), file_count


def _episode_index_from_path(path: Path) -> int | None:
    stem = path.stem
    if not stem.startswith("episode_"):
        return None
    try:
        return int(stem.removeprefix("episode_"))
    except ValueError:
        return None


def _column_rows(column) -> list[Any]:
    # Scalar.as_py() works for both Arrow fixed lists and the datasets
    # Array2D extension used by tactile data, including with PyArrow 20+.
    return [column[index].as_py() for index in range(len(column))]


@dataclass
class NumericStats:
    expected_shape: tuple[int, ...]
    rows: int = 0
    finite_rows: int = 0
    zero_rows: int = 0
    bad_shapes: int = 0
    nonfinite_values: int = 0
    minimum: np.ndarray | None = None
    maximum: np.ndarray | None = None
    total_sum: np.ndarray | None = None
    total_sq_sum: np.ndarray | None = None

    def update(self, rows: list[Any]) -> None:
        for value in rows:
            self.rows += 1
            try:
                array = np.asarray(value)
            except Exception:
                self.bad_shapes += 1
                continue
            if array.shape != self.expected_shape:
                self.bad_shapes += 1
                continue
            numeric = array.astype(np.float64, copy=False)
            finite = np.isfinite(numeric)
            bad_count = int(numeric.size - finite.sum())
            self.nonfinite_values += bad_count
            if bad_count:
                continue
            self.finite_rows += 1
            if np.all(numeric == 0):
                self.zero_rows += 1
            flat = numeric.reshape(-1)
            if self.minimum is None:
                self.minimum = flat.copy()
                self.maximum = flat.copy()
                self.total_sum = np.zeros_like(flat)
                self.total_sq_sum = np.zeros_like(flat)
            self.minimum = np.minimum(self.minimum, flat)
            self.maximum = np.maximum(self.maximum, flat)
            self.total_sum += flat
            self.total_sq_sum += flat * flat

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "shape": list(self.expected_shape),
            "rows": self.rows,
            "bad_shapes": self.bad_shapes,
            "nonfinite_values": self.nonfinite_values,
            "all_zero_rows": self.zero_rows,
            "all_zero_ratio": self.zero_rows / self.rows if self.rows else 0.0,
        }
        if self.finite_rows and self.minimum is not None:
            mean = self.total_sum / self.finite_rows
            variance = np.maximum(
                self.total_sq_sum / self.finite_rows - mean * mean, 0.0
            )
            result.update(
                {
                    "min": self.minimum.tolist(),
                    "max": self.maximum.tolist(),
                    "mean": mean.tolist(),
                    "std": np.sqrt(variance).tolist(),
                }
            )
        return result


@dataclass
class ImageStats:
    expected_shape: tuple[int, int, int]
    checked: int = 0
    decoded: int = 0
    missing: int = 0
    corrupt: int = 0
    wrong_shape: int = 0
    total_encoded_bytes: int = 0
    consecutive_duplicates: int = 0
    unique_hashes: set[str] = field(default_factory=set)
    previous_hash_by_episode: dict[int, str] = field(default_factory=dict)

    def update(self, value: Any, episode_index: int, *, decode: bool) -> None:
        self.checked += 1
        if not isinstance(value, dict):
            self.missing += 1
            return
        payload = value.get("bytes")
        path = value.get("path")
        if payload is None and path:
            candidate = Path(path)
            try:
                payload = candidate.read_bytes()
            except OSError:
                self.missing += 1
                return
        if not payload:
            self.missing += 1
            return
        self.total_encoded_bytes += len(payload)
        digest = hashlib.sha1(payload).hexdigest()
        if self.previous_hash_by_episode.get(episode_index) == digest:
            self.consecutive_duplicates += 1
        self.previous_hash_by_episode[episode_index] = digest
        self.unique_hashes.add(digest)
        if not decode:
            return
        self.decoded += 1
        try:
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                actual_shape = (image.height, image.width, len(image.getbands()))
        except Exception:
            self.corrupt += 1
            return
        if actual_shape != self.expected_shape:
            self.wrong_shape += 1

    def summary(self) -> dict[str, Any]:
        return {
            "expected_shape": list(self.expected_shape),
            "checked": self.checked,
            "decoded": self.decoded,
            "missing": self.missing,
            "corrupt": self.corrupt,
            "wrong_shape": self.wrong_shape,
            "encoded_size": self.total_encoded_bytes,
            "encoded_size_human": _human_bytes(self.total_encoded_bytes),
            "unique_frames": len(self.unique_hashes),
            "consecutive_duplicates": self.consecutive_duplicates,
        }


def _shape_from_feature(feature: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(value) for value in feature.get("shape", []))


def inspect_dataset(
    root: Path,
    *,
    image_check: str = "all",
    max_camera_skew_ms: float = 80.0,
) -> tuple[dict[str, Any], list[str], list[str]]:
    root = root.expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot元数据不存在: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    required = set(IMAGE_KEYS + NUMERIC_KEYS + INDEX_KEYS)
    missing_features = sorted(required.difference(features))
    if missing_features:
        errors.append(f"元数据缺少字段: {missing_features}")

    episodes_meta = _read_jsonlines(root / "meta" / "episodes.jsonl")
    tasks_meta = _read_jsonlines(root / "meta" / "tasks.jsonl")
    total_size, size_groups, file_count = _directory_sizes(root)
    parquet_files = sorted(root.glob("data/chunk-*/episode_*.parquet"))
    file_indices = [_episode_index_from_path(path) for path in parquet_files]
    valid_file_indices = [index for index in file_indices if index is not None]
    expected_episode_count = int(info.get("total_episodes", 0))
    expected_frame_count = int(info.get("total_frames", 0))
    fps = int(info.get("fps", 0))
    if valid_file_indices != list(range(expected_episode_count)):
        errors.append(
            "Parquet episode编号不连续或数量不符: "
            f"actual={valid_file_indices}, expected=0..{expected_episode_count - 1}"
        )
    if len(episodes_meta) != expected_episode_count:
        errors.append(
            f"episodes.jsonl记录数={len(episodes_meta)}，元数据={expected_episode_count}"
        )

    numeric_stats = {
        key: NumericStats(_shape_from_feature(features[key]))
        for key in NUMERIC_KEYS
        if key in features
    }
    image_stats = {
        key: ImageStats(_shape_from_feature(features[key]))
        for key in IMAGE_KEYS
        if key in features
    }
    null_counts: Counter[str] = Counter()
    rows_by_episode: dict[int, int] = {}
    global_rows = 0
    parquet_uncompressed_bytes = 0
    timestamp_bad_steps = 0
    timestamp_large_steps = 0
    frame_id_non_increasing = np.zeros(4, dtype=np.int64)
    maximum_sensor_skew_ms = 0.0
    global_indices: list[int] = []

    for path in parquet_files:
        episode_from_path = _episode_index_from_path(path)
        parquet = pq.ParquetFile(path)
        episode_rows = int(parquet.metadata.num_rows)
        rows_by_episode[int(episode_from_path)] = episode_rows
        for row_group_index in range(parquet.metadata.num_row_groups):
            row_group_meta = parquet.metadata.row_group(row_group_index)
            parquet_uncompressed_bytes += int(row_group_meta.total_byte_size)
            table = parquet.read_row_group(row_group_index)
            columns = set(table.column_names)
            absent = sorted(required.difference(columns))
            if absent:
                errors.append(f"{path.name} row_group={row_group_index} 缺少字段: {absent}")
                continue
            for key in required:
                null_counts[key] += int(table[key].null_count)

            episode_values = np.asarray(
                _column_rows(table["episode_index"]), dtype=np.int64
            ).reshape(-1)
            frame_values = np.asarray(
                _column_rows(table["frame_index"]), dtype=np.int64
            ).reshape(-1)
            index_values = np.asarray(
                _column_rows(table["index"]), dtype=np.int64
            ).reshape(-1)
            global_indices.extend(index_values.tolist())
            if np.any(episode_values != episode_from_path):
                errors.append(f"{path.name} 内部episode_index与文件名不一致")

            row_offset = global_rows - sum(
                rows_by_episode.get(i, 0) for i in range(int(episode_from_path))
            )
            expected_frames = np.arange(row_offset, row_offset + len(frame_values))
            if not np.array_equal(frame_values, expected_frames):
                errors.append(
                    f"{path.name} frame_index在row group {row_group_index}不连续"
                )

            timestamps = np.asarray(
                _column_rows(table["timestamp"]), dtype=np.float64
            ).reshape(-1)
            if len(timestamps) > 1:
                delta = np.diff(timestamps)
                timestamp_bad_steps += int(np.count_nonzero(delta <= 0))
                if fps > 0:
                    timestamp_large_steps += int(
                        np.count_nonzero(delta > (2.5 / fps))
                    )

            sensor_times = np.asarray(
                _column_rows(table["sensor.timestamps"]), dtype=np.float64
            )
            if sensor_times.ndim == 2 and sensor_times.shape[1] == 4:
                skew = (np.max(sensor_times, axis=1) - np.min(sensor_times, axis=1))
                finite_skew = skew[np.isfinite(skew)]
                if finite_skew.size:
                    maximum_sensor_skew_ms = max(
                        maximum_sensor_skew_ms,
                        float(np.max(finite_skew) * 1000.0),
                    )

            frame_ids = np.asarray(
                _column_rows(table["sensor.frame_ids"]), dtype=np.int64
            )
            if frame_ids.ndim == 2 and frame_ids.shape[1] == 4 and len(frame_ids) > 1:
                frame_id_non_increasing += np.count_nonzero(
                    np.diff(frame_ids, axis=0) <= 0, axis=0
                )

            for key, stats in numeric_stats.items():
                stats.update(_column_rows(table[key]))

            for key, stats in image_stats.items():
                values = _column_rows(table[key])
                for row_index, value in enumerate(values):
                    absolute_row = global_rows + row_index
                    decode = image_check == "all" or (
                        image_check == "sample" and absolute_row % max(fps, 1) == 0
                    )
                    stats.update(value, int(episode_from_path), decode=decode)
            global_rows += len(table)

    if global_rows != expected_frame_count:
        errors.append(
            f"Parquet总帧数={global_rows}，meta/info.json声明={expected_frame_count}"
        )
    if global_indices and global_indices != list(range(expected_frame_count)):
        errors.append("全局index不是从0开始的连续序列")
    for key, count in sorted(null_counts.items()):
        if count:
            errors.append(f"{key} 存在 {count} 个null值")
    for key, stats in numeric_stats.items():
        if stats.bad_shapes:
            errors.append(f"{key} 有 {stats.bad_shapes} 帧shape错误")
        if stats.nonfinite_values:
            errors.append(f"{key} 有 {stats.nonfinite_values} 个NaN/Inf")
        if stats.rows and stats.zero_rows == stats.rows:
            warnings.append(f"{key} 全部为零，传感器可能未连接或被禁用")
    for key, stats in image_stats.items():
        if stats.missing or stats.corrupt or stats.wrong_shape:
            errors.append(
                f"{key}: missing={stats.missing}, corrupt={stats.corrupt}, "
                f"wrong_shape={stats.wrong_shape}"
            )
        if stats.checked and len(stats.unique_hashes) <= 1:
            warnings.append(f"{key} 所有帧完全相同，可能发生相机卡帧")
        duplicate_ratio = (
            stats.consecutive_duplicates / max(stats.checked - expected_episode_count, 1)
        )
        if duplicate_ratio > 0.05:
            warnings.append(
                f"{key} 连续重复帧比例约 {duplicate_ratio:.1%}，请检查相机帧率"
            )
    if timestamp_bad_steps:
        errors.append(f"episode时间戳有 {timestamp_bad_steps} 处未递增")
    if timestamp_large_steps:
        warnings.append(f"episode时间戳有 {timestamp_large_steps} 处间隔超过2.5帧")
    if maximum_sensor_skew_ms > max_camera_skew_ms:
        warnings.append(
            f"传感器最大时间偏差 {maximum_sensor_skew_ms:.1f}ms "
            f"超过参考阈值 {max_camera_skew_ms:.1f}ms"
        )
    for sensor_index, count in enumerate(frame_id_non_increasing.tolist()):
        if count:
            warnings.append(
                f"sensor.frame_ids第{sensor_index}路有 {count} 处未递增或重复"
            )

    episode_lengths = {
        int(record["episode_index"]): int(record["length"])
        for record in episodes_meta
        if "episode_index" in record and "length" in record
    }
    for episode_index, actual_rows in rows_by_episode.items():
        expected_rows = episode_lengths.get(episode_index)
        if expected_rows is None:
            errors.append(f"episode {episode_index} 缺少episodes.jsonl记录")
        elif actual_rows != expected_rows:
            errors.append(
                f"episode {episode_index}: Parquet={actual_rows}帧，"
                f"episodes.jsonl={expected_rows}帧"
            )

    decoded_image_memory = (
        expected_frame_count
        * sum(math.prod(stats.expected_shape) for stats in image_stats.values())
    )
    numeric_memory = sum(
        expected_frame_count * math.prod(stats.expected_shape) * 4
        for stats in numeric_stats.values()
    )
    report = {
        "root": str(root),
        "status": "ERROR" if errors else ("WARNING" if warnings else "OK"),
        "dataset": {
            "codebase_version": info.get("codebase_version"),
            "robot_type": info.get("robot_type"),
            "fps": fps,
            "episodes": expected_episode_count,
            "frames": expected_frame_count,
            "duration_seconds": expected_frame_count / fps if fps else None,
            "tasks": tasks_meta,
            "features": {
                key: {
                    "dtype": value.get("dtype"),
                    "shape": value.get("shape"),
                }
                for key, value in features.items()
            },
        },
        "storage": {
            "total_bytes": total_size,
            "total_human": _human_bytes(total_size),
            "file_count": file_count,
            "by_top_level": {
                key: {"bytes": value, "human": _human_bytes(value)}
                for key, value in size_groups.items()
            },
            "parquet_uncompressed_bytes": parquet_uncompressed_bytes,
            "parquet_uncompressed_human": _human_bytes(parquet_uncompressed_bytes),
            "estimated_fully_decoded_image_memory": decoded_image_memory,
            "estimated_fully_decoded_image_memory_human": _human_bytes(
                decoded_image_memory
            ),
            "estimated_numeric_memory": numeric_memory,
            "estimated_numeric_memory_human": _human_bytes(numeric_memory),
            "note": "检查过程按Parquet row group流式读取，不会同时占用上述完整解码内存。",
        },
        "episodes": {
            "parquet_files": len(parquet_files),
            "rows_by_episode": rows_by_episode,
        },
        "images": {key: stats.summary() for key, stats in image_stats.items()},
        "image_check_mode": image_check,
        "numeric": {key: stats.summary() for key, stats in numeric_stats.items()},
        "timing": {
            "timestamp_non_increasing": timestamp_bad_steps,
            "timestamp_large_gaps": timestamp_large_steps,
            "maximum_sensor_skew_ms": maximum_sensor_skew_ms,
            "frame_id_non_increasing": frame_id_non_increasing.tolist(),
        },
        "errors": errors,
        "warnings": warnings,
    }
    return report, errors, warnings


def _short_vector(value: Any, limit: int = 8) -> str:
    if not isinstance(value, list):
        return str(value)
    shown = value[:limit]
    suffix = ", ..." if len(value) > limit else ""
    return "[" + ", ".join(f"{item:.4g}" for item in shown) + suffix + "]"


def print_report(report: dict[str, Any]) -> None:
    dataset = report["dataset"]
    storage = report["storage"]
    print("\n" + "=" * 96)
    print("reBot ACT / LeRobot 多模态数据集检查报告")
    print("=" * 96)
    print(f"路径:       {report['root']}")
    print(f"总体状态:   {report['status']}")
    print(
        f"数据规模:   {dataset['episodes']} episodes, {dataset['frames']} frames, "
        f"{dataset['fps']} Hz, {dataset['duration_seconds']:.1f} s"
    )
    print(
        f"格式/机器人: LeRobot {dataset['codebase_version']} / {dataset['robot_type']}"
    )
    tasks = [record.get("task") for record in dataset["tasks"]]
    print(f"ACT任务标签: {tasks}")
    print(f"磁盘占用:   {storage['total_human']} / {storage['file_count']} files")
    for key, value in storage["by_top_level"].items():
        print(f"  - {key:<10} {value['human']}")
    print(f"Parquet解压数据量:       {storage['parquet_uncompressed_human']}")
    print(
        "双相机全部解码估算内存: "
        f"{storage['estimated_fully_decoded_image_memory_human']}"
    )
    print(f"数值字段估算内存:       {storage['estimated_numeric_memory_human']}")
    print("检查器采用row group流式读取，不会一次占用全部估算内存。")

    print("\nEpisode:")
    lengths = list(report["episodes"]["rows_by_episode"].values())
    print(
        f"  Parquet文件: {report['episodes']['parquet_files']}, "
        f"长度范围: {min(lengths) if lengths else 0}～{max(lengths) if lengths else 0}"
    )

    print("\n图像:")
    print(f"  解码检查模式: {report['image_check_mode']}")
    for key, stats in report["images"].items():
        print(
            f"  {key}: checked={stats['checked']}, decoded={stats['decoded']}, "
            f"missing={stats['missing']}, "
            f"corrupt={stats['corrupt']}, wrong_shape={stats['wrong_shape']}, "
            f"unique={stats['unique_frames']}, "
            f"consecutive_duplicates={stats['consecutive_duplicates']}, "
            f"encoded={stats['encoded_size_human']}"
        )

    print("\n数值传感器/动作:")
    for key, stats in report["numeric"].items():
        print(
            f"  {key:<29} shape={stats['shape']!s:<10} "
            f"bad_shape={stats['bad_shapes']:<4} "
            f"NaN/Inf={stats['nonfinite_values']:<4} "
            f"zero={stats['all_zero_ratio']:.1%} "
            f"min={_short_vector(stats.get('min'))} "
            f"max={_short_vector(stats.get('max'))}"
        )

    timing = report["timing"]
    print("\n时序:")
    print(
        f"  timestamp未递增={timing['timestamp_non_increasing']}, "
        f"大间隔={timing['timestamp_large_gaps']}, "
        f"传感器最大偏差={timing['maximum_sensor_skew_ms']:.2f} ms"
    )
    print(f"  四路frame_id未递增/重复={timing['frame_id_non_increasing']}")

    print("\n问题:")
    if not report["errors"] and not report["warnings"]:
        print("  未发现数据缺失、损坏、形状错误、NaN/Inf或连续性问题。")
    for message in report["errors"]:
        print(f"  [ERROR] {message}")
    for message in report["warnings"]:
        print(f"  [WARN]  {message}")
    print("=" * 96)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="检查reBot ACT LeRobot多模态数据集")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--image-check",
        choices=("all", "sample", "none"),
        default="all",
        help="图像解码检查范围；all最完整，sample每秒一帧，none仅查字节是否存在",
    )
    parser.add_argument("--max-camera-skew-ms", type=float, default=80.0)
    parser.add_argument("--json-report", type=Path, help="额外保存JSON检查报告")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="存在warning时也返回非零退出码",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    report, errors, warnings = inspect_dataset(
        args.root,
        image_check=args.image_check,
        max_camera_skew_ms=args.max_camera_skew_ms,
    )
    print_report(report)
    if args.json_report is not None:
        output = args.json_report.expanduser()
        if not output.is_absolute():
            output = (PROJECT_ROOT / output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"JSON报告已保存: {output}")
    if errors or (args.strict and warnings):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
