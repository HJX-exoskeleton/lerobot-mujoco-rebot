"""Camera backends owned by the real ACT project."""

from __future__ import annotations

import threading
import time
from typing import Any

import cv2
import numpy as np
from pyorbbecsdk import Config, OBError, OBFormat, OBSensorType, Pipeline


class ThreadedGemini336LCamera:
    """Non-blocking Gemini 336L color camera with the existing camera API.

    The Orbbec pipeline explicitly enables only the color sensor. ``read``
    returns the latest independent RGB array, frame id and monotonic timestamp.
    """

    def __init__(
        self,
        name: str = "gemini336l",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        startup_timeout: float = 8.0,
        frame_timeout_ms: int = 1000,
    ) -> None:
        self.name = name
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.startup_timeout = float(startup_timeout)
        self.frame_timeout_ms = int(frame_timeout_ms)

        self.frame_rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.frame_id = 0
        self.frame_timestamp = 0.0
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._valid = False
        self._error = "正在启动"
        self._active_mode = "unknown"

        self.thread = threading.Thread(
            target=self._capture_worker,
            name=f"gemini336l-{self.name}",
            daemon=True,
        )
        print(f"⏳ 正在初始化 Gemini 336L RGB 相机 [{self.name}]...")
        self.thread.start()
        self._ready_event.wait(timeout=self.startup_timeout)

        if self.valid:
            print(
                f"✅ [成功] Gemini 336L [{self.name}] RGB 已就绪，"
                f"输出尺寸 {self.width}x{self.height}。"
            )
        else:
            print(
                f"⚠️ [Gemini336L:{self.name}] 启动后暂未读到 RGB 图像："
                f"{self._error}。先输出黑图，主循环继续。"
            )

    @property
    def valid(self) -> bool:
        with self.lock:
            return bool(self._valid and self.frame_id > 0 and self.thread.is_alive())

    def _select_color_profile(self, pipeline: Pipeline) -> Any:
        profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        if profiles is None or profiles.get_count() == 0:
            raise RuntimeError("设备没有可用的彩色流")
        try:
            profile = profiles.get_video_stream_profile(
                self.width,
                self.height,
                OBFormat.RGB,
                self.fps,
            )
        except OBError:
            profile = profiles.get_default_video_stream_profile()
        self._active_mode = (
            f"{profile.get_width()}x{profile.get_height()}@{profile.get_fps()}"
            f"/{profile.get_format()}"
        )
        return profile

    @staticmethod
    def _frame_to_rgb(frame: Any) -> np.ndarray:
        width = frame.get_width()
        height = frame.get_height()
        frame_format = frame.get_format()
        data = np.asanyarray(frame.get_data())

        if frame_format == OBFormat.RGB:
            return data.reshape((height, width, 3)).copy()
        if frame_format == OBFormat.BGR:
            bgr = data.reshape((height, width, 3))
        elif frame_format == OBFormat.MJPG:
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError("MJPG 彩色帧解码失败")
        elif frame_format == OBFormat.YUYV:
            yuv = data.reshape((height, width, 2))
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_YUYV)
        elif frame_format == OBFormat.UYVY:
            yuv = data.reshape((height, width, 2))
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_UYVY)
        elif frame_format == OBFormat.I420:
            yuv = data.reshape((height * 3 // 2, width))
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        elif frame_format == OBFormat.NV12:
            yuv = data.reshape((height * 3 // 2, width))
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        elif frame_format == OBFormat.NV21:
            yuv = data.reshape((height * 3 // 2, width))
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
        else:
            raise RuntimeError(f"不支持的 Gemini 336L 彩色格式: {frame_format}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _capture_worker(self) -> None:
        pipeline: Pipeline | None = None
        started = False
        try:
            pipeline = Pipeline()
            config = Config()
            config.disable_all_stream()
            config.enable_stream(self._select_color_profile(pipeline))
            pipeline.start(config)
            started = True

            while not self._stop_event.is_set():
                frames = pipeline.wait_for_frames(self.frame_timeout_ms)
                if frames is None:
                    with self.lock:
                        self._error = "等待 RGB 帧超时"
                    continue
                color_frame = frames.get_color_frame()
                if color_frame is None:
                    continue

                rgb = self._frame_to_rgb(color_frame)
                if rgb.shape[:2] != (self.height, self.width):
                    rgb = cv2.resize(
                        rgb,
                        (self.width, self.height),
                        interpolation=cv2.INTER_AREA,
                    )
                rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
                timestamp = time.perf_counter()
                with self.lock:
                    self.frame_rgb = rgb
                    self.frame_id += 1
                    self.frame_timestamp = timestamp
                    self._valid = True
                    self._error = ""
                self._ready_event.set()
        except Exception as exc:  # native SDK failures must not kill the process
            with self.lock:
                self._valid = False
                self._error = f"{type(exc).__name__}: {exc}"
            self._ready_event.set()
        finally:
            if started and pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass
            with self.lock:
                self._valid = False
            self._ready_event.set()

    def read(self) -> tuple[np.ndarray, int, float]:
        with self.lock:
            return (
                self.frame_rgb.copy(),
                int(self.frame_id),
                float(self.frame_timestamp),
            )

    def read_rgb(self) -> np.ndarray:
        return self.read()[0]

    def status(self) -> str:
        with self.lock:
            age_ms = (
                (time.perf_counter() - self.frame_timestamp) * 1000.0
                if self.frame_timestamp > 0
                else float("inf")
            )
            return (
                f"{self.name}:backend=Gemini336L-RGB,valid={self._valid},"
                f"fid={self.frame_id},age={age_ms:.0f}ms,"
                f"mode={self._active_mode},err={self._error or '-'}"
            )

    def release(self) -> None:
        self._stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=max(self.frame_timeout_ms / 1000.0 + 1.0, 2.0))

