"""Strict real-time sensor input used by multimodal ACT deployment."""

from __future__ import annotations

import time

import numpy as np

from rebot_scripts.Servo_control import record_rebot_episodes as sensor_hw


class RealTimeMultimodalSensors:
    def __init__(
        self,
        *,
        use_imu: bool,
        use_tactile: bool,
        imu_port: str,
        tactile_port: str,
        tactile_baud: int,
        tactile_init_frames: int,
    ):
        self.use_imu = use_imu
        self.use_tactile = use_tactile
        self.imu = (
            sensor_hw.ThreadedYbImuReader(
                port=imu_port, report_rate=50, alpha=0.9, name="act_imu"
            )
            if use_imu
            else None
        )
        self.tactile = (
            sensor_hw.ThreadedFlexiTacReader(
                port=tactile_port,
                baud=tactile_baud,
                init_frames=tactile_init_frames,
                name="act_tactile",
            )
            if use_tactile
            else None
        )

    def wait_until_ready(
        self, *, timeout_s: float, maximum_age_ms: float
    ) -> dict[str, np.ndarray]:
        """Wait for initial IMU sample and tactile baseline before arm connection."""
        deadline = time.perf_counter() + max(float(timeout_s), 0.1)
        last_error = "传感器尚未输出首帧"
        while time.perf_counter() < deadline:
            try:
                values = self.read(maximum_age_ms)
                enabled_count = int(self.use_imu) + int(self.use_tactile)
                if len(values) == enabled_count:
                    if enabled_count:
                        print("✅ 多模态传感器预检通过，IMU/触觉数据已就绪。")
                    return values
            except RuntimeError as exc:
                last_error = str(exc)
            time.sleep(0.02)
        statuses = []
        if self.imu is not None:
            statuses.append(self.imu.status())
        if self.tactile is not None:
            statuses.append(self.tactile.status())
        detail = "；".join(statuses) if statuses else "没有启用额外传感器"
        raise RuntimeError(
            f"多模态传感器启动超时({timeout_s:.1f}s)：{last_error}；{detail}"
        )

    def read(self, maximum_age_ms: float) -> dict[str, np.ndarray]:
        values, _ = self.read_with_metadata(maximum_age_ms)
        return values

    def read_with_metadata(
        self, maximum_age_ms: float
    ) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
        now = time.perf_counter()
        result: dict[str, np.ndarray] = {}
        metadata: dict[str, float | int] = {}
        if self.imu is not None:
            sample = self.imu.read()
            timestamp = float(sample["timestamp"])
            frame_id = int(sample["frame_id"])
            if frame_id <= 0 or timestamp <= 0:
                raise RuntimeError("IMU尚无有效首帧")
            age_ms = (now - timestamp) * 1000.0
            if age_ms > maximum_age_ms:
                raise RuntimeError(f"IMU数据过期: age={age_ms:.1f}ms")
            value = np.asarray(sample["imu_left"], dtype=np.float32)
            if value.shape != (10,) or not np.all(np.isfinite(value)):
                raise RuntimeError("IMU数据形状非法或包含NaN/Inf")
            result["sensor.imu"] = value
            metadata["imu_frame_id"] = frame_id
            metadata["imu_timestamp"] = timestamp
        if self.tactile is not None:
            value, frame_id, timestamp = self.tactile.read()
            if int(frame_id) <= 0 or float(timestamp) <= 0:
                raise RuntimeError("触觉baseline尚未完成或尚无有效首帧")
            age_ms = (now - float(timestamp)) * 1000.0
            if age_ms > maximum_age_ms:
                raise RuntimeError(f"触觉数据过期: age={age_ms:.1f}ms")
            value = np.asarray(value, dtype=np.float32)
            if value.shape != (12, 30) or not np.all(np.isfinite(value)):
                raise RuntimeError("触觉数据形状非法或包含NaN/Inf")
            result["sensor.tactile"] = value
            metadata["tactile_frame_id"] = int(frame_id)
            metadata["tactile_timestamp"] = float(timestamp)
        return result, metadata

    def release(self) -> None:
        for sensor in (self.imu, self.tactile):
            if sensor is not None:
                sensor.release()
