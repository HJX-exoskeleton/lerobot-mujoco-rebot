#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
舵机主手 -> 达妙真机机械臂 + 达妙真机夹爪 遥操作程序 (集成多线程真实RGB流)

控制链路：
    ID1~ID6 ST/SMS_STS 舵机主手 -> q_sim(rad) -> 达妙真机 6 轴 q_real(rad) -> SafetyGuard -> pos_vel
    ID7 ST/SMS_STS 舵机夹爪 -> gripper_norm -> 达妙夹爪目标角度 -> send_mit

数据采集特性：
    1. 按下【Enter】键触发录制，支持 --time 或 --episode_len。
    2. 自动保存为标准 ALOHA 规范 HDF5 (time, qpos, qvel, action, images/*)。
    3. 后台多线程读取 MJPG 视频流并实时转换为 RGB，彻底解耦，绝不阻塞 50Hz 控制主循环。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import signal
import sys
import threading
import time
import subprocess
import struct
from multiprocessing import shared_memory
from pathlib import Path

# 减少 OpenCV Qt 字体警告：必须尽量放在 import cv2 前
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2  # OpenCV 用于图像处理/显示/格式转换
# cv2 导入后再设置一次，防止 opencv-python 覆盖字体路径
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu"

import pyrealsense2 as rs  # Intel RealSense D405 官方库
import h5py
from tqdm import tqdm

import mujoco
import numpy as np

# =============================================================================
# 0. 全局运行标志与数据采集配置
# =============================================================================

_running = True
_is_recording = False

# 基础数据缓存
recorded_timestamps = []
recorded_qpos = []
recorded_qvel = []
recorded_action = []
recorded_imu_left = []
recorded_imu_quat = []
recorded_imu_gyro = []
recorded_imu_accel = []
recorded_imu_mag = []
recorded_imu_euler = []
recorded_imu_baro = []
recorded_imu_frame_ids = []
recorded_imu_timestamps = []
recorded_tactile = []
recorded_tactile_frame_ids = []
recorded_tactile_timestamps = []

# 相机图像流缓存字典
recorded_images = {}
# 每个保存帧对应的真实相机帧号与相机时间戳，用于诊断重复帧/卡顿
recorded_image_frame_ids = {}
recorded_image_timestamps = {}

_config = {
    "task_name": "teleop_task",
    "base_save_dir": Path("./collected_data"),
    "episode_len": 500,
    "dt": 0.02,
    "rate": 50,
    "episode_idx": None,
    # 🌟 修改：目前先挂载全局高空相机 cam_high
    "camera_names": ["cam_high", "cam_wrist"]
}

TACTILE_RAW_ROWS = 12
TACTILE_RAW_COLS = 30
IMU_LEFT_DIM = 10

def _sigint_handler(signum, frame) -> None:
    global _running
    print("\n[teleop] 收到退出信号，准备安全关闭...")
    _running = False


signal.signal(signal.SIGINT, _sigint_handler)
signal.signal(signal.SIGTERM, _sigint_handler)


# =============================================================================
# 0.5 🌟 新增：独立的多线程相机读取类
# =============================================================================
class ThreadedCamera:
    """独立的后台相机读取线程，避免 OpenCV I/O 阻塞 50Hz 的遥操作主循环"""

    def __init__(self, src=2, width=640, height=480, fps=30, name="camera"):
        self.name = name
        self.src = src
        self.width = width
        self.height = height
        self.fps = int(fps)
        self.capture = cv2.VideoCapture(src)

        # 强制使用 MJPG 编码压缩 USB 带宽
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret, self.frame = self.capture.read()
        self.valid = self.ret
        self.frame_count = 1 if self.valid else 0
        self.start_time = time.time()
        self.last_frame_time = self.start_time if self.valid else 0.0

        if not self.valid:
            print(f"⚠️ [警告] 相机 {self.name} (src={src}) 初始化失败，将输出全黑图像以维持时序对齐。")
            self.frame = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            actual_fps = self.capture.get(cv2.CAP_PROP_FPS)
            print(
                f"📷 [成功] 相机 {self.name} (src={src}) 后台读取线程已启动，"
                f"request_fps={self.fps}, actual_fps≈{actual_fps:.1f}。"
            )

        self.running = True
        self.lock = threading.Lock()

        if self.valid:
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.capture.read()
            if ret:
                with self.lock:
                    self.frame = frame
                    self.frame_count += 1
                    self.last_frame_time = time.time()
            else:
                time.sleep(0.005)

    def status(self) -> str:
        with self.lock:
            frame = self.frame.copy()
            frame_count = int(self.frame_count)
            last_frame_time = float(self.last_frame_time)
        elapsed = max(time.time() - self.start_time, 1e-6)
        fps = frame_count / elapsed
        age_ms = (time.time() - last_frame_time) * 1000.0 if last_frame_time > 0 else float("inf")
        h, w = frame.shape[:2]
        return (
            f"{self.name}: valid={self.valid}, src={self.src}, frame={w}x{h}, "
            f"count={frame_count}, fps≈{fps:.1f}, age={age_ms:.0f}ms, "
            f"mean={frame.mean():.1f}, std={frame.std():.1f}"
        )

    def info(self) -> dict:
        with self.lock:
            frame_count = int(self.frame_count)
            last_frame_time = float(self.last_frame_time)
            valid = bool(self.valid)
        age_ms = (time.time() - last_frame_time) * 1000.0 if last_frame_time > 0 else float("inf")
        elapsed = max(time.time() - self.start_time, 1e-6)
        fps = frame_count / elapsed
        return {
            "name": self.name,
            "backend": "OpenCV",
            "valid": valid,
            "frame_id": frame_count,
            "fps": fps,
            "age_ms": age_ms,
            "error": "",
            "exitcode": None,
            "alive": valid,
            "timestamp": last_frame_time,
        }

    def read_rgb(self):
        """主线程调用的极速读取接口，直接返回内存最新帧的 RGB 格式"""
        with self.lock:
            current_frame = self.frame.copy()
        if self.valid:
            return cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        return current_frame

    def read(self) -> tuple[np.ndarray, int, float]:
        with self.lock:
            current_frame = self.frame.copy()
            frame_count = int(self.frame_count)
            last_frame_time = float(self.last_frame_time)
        if self.valid:
            return cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB), frame_count, last_frame_time
        return current_frame, frame_count, last_frame_time

    def release(self):
        self.running = False
        if self.capture.isOpened():
            self.capture.release()


# =============================================================================
# 0.5.1 🌟 新增：Astra-S 完全外部进程 RGB 读取类
# =============================================================================
class ThreadedAstraSCamera:
    """
    Astra-S RGB 读取类：主进程接口 + 完全外部采集脚本。

    为什么不用 multiprocessing target 函数：
        Linux spawn/fork 子进程仍可能重新导入当前复杂主脚本，包括 MuJoCo、RealSense、
        机器人驱动等库。Orbbec OpenNI beta 在这种环境下容易 SIGSEGV/SIGABRT。
        因此这里用 subprocess 启动一个最小 helper：astra_s_shm_server.py。
        helper 只 import OpenNI/cv2/numpy，并通过 shared_memory 输出最新 RGB 帧。

    read()/read_rgb()/status()/release() 接口与其他相机类保持一致。
    """

    DEFAULT_OPENNI2_REDIST = "/home/hjx/orbbec_openni_redist"
    DEFAULT_WIDTH = 640
    DEFAULT_HEIGHT = 480
    DEFAULT_FPS = 30  # helper 与独立预览脚本一致：RGB888 640x480 @ 30 FPS。
    DEFAULT_FLIP_HORIZONTAL = True
    DEFAULT_FLIP_VERTICAL = False
    DEFAULT_STARTUP_TIMEOUT = 8.0
    DEFAULT_RESTART_ON_CRASH = True

    # 与 astra_s_shm_server.py 保持一致。
    META_FORMAT = "<qqdii"
    META_SIZE = struct.calcsize(META_FORMAT)

    def __init__(
        self,
        name: str = "astra_s",
        openni2_redist: str | None = None,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
        flip_horizontal: bool | None = None,
        flip_vertical: bool | None = None,
        startup_timeout: float | None = None,
        restart_on_crash: bool | None = None,
        helper_path: str | Path | None = None,
    ):
        self.name = name
        self.openni2_redist = str(openni2_redist or self.DEFAULT_OPENNI2_REDIST)
        self.width = int(width or self.DEFAULT_WIDTH)
        self.height = int(height or self.DEFAULT_HEIGHT)
        self.fps = int(fps or self.DEFAULT_FPS)
        self.flip_horizontal = self.DEFAULT_FLIP_HORIZONTAL if flip_horizontal is None else bool(flip_horizontal)
        self.flip_vertical = self.DEFAULT_FLIP_VERTICAL if flip_vertical is None else bool(flip_vertical)
        self.startup_timeout = float(startup_timeout or self.DEFAULT_STARTUP_TIMEOUT)
        self.restart_on_crash = self.DEFAULT_RESTART_ON_CRASH if restart_on_crash is None else bool(restart_on_crash)

        default_helper = Path(__file__).resolve().with_name("astra_s_shm_server.py")
        self.helper_path = Path(helper_path).resolve() if helper_path is not None else default_helper

        self.frame_shm = shared_memory.SharedMemory(create=True, size=self.height * self.width * 3)
        self.meta_shm = shared_memory.SharedMemory(create=True, size=self.META_SIZE)

        self.frame_rgb = np.ndarray((self.height, self.width, 3), dtype=np.uint8, buffer=self.frame_shm.buf)
        self.frame_rgb[:] = 0
        self._write_empty_meta()

        self.process: subprocess.Popen | None = None
        self._last_restart_time = 0.0
        self._last_stderr_report = 0.0
        self._fallback_camera: ThreadedCamera | None = None
        self._backend = "Astra-S"

        print(f"⏳ 正在初始化 Astra-S 相机 [{self.name}]，完全外部进程采集中...")
        print(f"📷 [Astra-S:{self.name}] helper = {self.helper_path}")

        if not self.helper_path.exists():
            print(f"❌ [Astra-S:{self.name}] 找不到 helper 脚本: {self.helper_path}")
            self._enter_fallback("helper 脚本不存在")
            return

        self._start_process()

        t0 = time.perf_counter()
        while time.perf_counter() - t0 < self.startup_timeout:
            _, fid, _, valid, _ = self._read_meta_once()
            if valid and fid > 0:
                break
            if self.process is not None and self.process.poll() is not None:
                break
            time.sleep(0.05)

        _, fid, _, valid, _ = self._read_meta_once()
        if valid and fid > 0 and self.process is not None and self.process.poll() is None:
            print(
                f"✅ [成功] Astra-S [{self.name}] 外部进程已输出图像，"
                f"共享帧尺寸 {self.width}x{self.height}。"
            )
        else:
            exitcode = self.process.poll() if self.process is not None else None
            print(
                f"⚠️ [Astra-S:{self.name}] 启动后暂未读到图像，"
                f"fid={fid}, valid={valid}, exitcode={exitcode}。先输出黑图，主循环继续。"
            )
            self._drain_stderr(force=True)
            if exitcode is not None or not valid:
                self._enter_fallback("Astra-S OpenNI 采集未就绪")

    def _write_empty_meta(self) -> None:
        struct.pack_into(self.META_FORMAT, self.meta_shm.buf, 0, 0, 0, 0.0, 0, 0)

    def _cleanup_shared_memory(self) -> None:
        for shm in (getattr(self, "frame_shm", None), getattr(self, "meta_shm", None)):
            if shm is None:
                continue
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass

    def _enter_fallback(self, reason: str) -> None:
        if self._fallback_camera is not None:
            return

        print(f"⚠️ [Astra-S:{self.name}] {reason}，回退到 src=2 普通 RGB 相机。")
        self._backend = "OpenCV-src2"

        if self.process is not None and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=0.5)
            except Exception:
                pass

        self._cleanup_shared_memory()
        self._fallback_camera = ThreadedCamera(src=2, name=self.name, fps=30)

    def _start_process(self) -> None:
        self._write_empty_meta()

        cmd = [
            sys.executable,
            str(self.helper_path),
            "--frame-shm", self.frame_shm.name,
            "--meta-shm", self.meta_shm.name,
            "--width", str(self.width),
            "--height", str(self.height),
            "--openni2-redist", self.openni2_redist,
            "--flip-horizontal", "1" if self.flip_horizontal else "0",
            "--flip-vertical", "1" if self.flip_vertical else "0",
        ]

        env = os.environ.copy()
        env.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")
        # 不强行覆盖 LD_LIBRARY_PATH；openni2.initialize(redist) 会加载指定路径。

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        self._last_restart_time = time.perf_counter()

    def _read_meta_once(self) -> tuple[int, int, float, int, int]:
        if self._fallback_camera is not None:
            return 0, 1, time.perf_counter(), 1, 0
        try:
            return struct.unpack_from(self.META_FORMAT, self.meta_shm.buf, 0)
        except Exception:
            return 0, 0, 0.0, 0, 9

    def _read_stable_meta_and_frame(self) -> tuple[np.ndarray, int, float, int, int]:
        """Seqlock 方式读取一帧，尽量避免读到写入中的半帧。"""
        if self._fallback_camera is not None:
            rgb = self._fallback_camera.read_rgb()
            return rgb, 1, time.perf_counter(), 1, 0
        for _ in range(3):
            seq1, fid1, ts1, valid1, err1 = self._read_meta_once()
            if seq1 % 2 == 1:
                time.sleep(0.0002)
                continue
            image = self.frame_rgb.copy()
            seq2, fid2, ts2, valid2, err2 = self._read_meta_once()
            if seq1 == seq2 and seq2 % 2 == 0:
                return image, int(fid2), float(ts2), int(valid2), int(err2)
        image = self.frame_rgb.copy()
        _, fid, ts, valid, err = self._read_meta_once()
        return image, int(fid), float(ts), int(valid), int(err)

    @property
    def valid(self) -> bool:
        if self._fallback_camera is not None:
            return bool(getattr(self._fallback_camera, "valid", True))
        _, fid, _, valid, _ = self._read_meta_once()
        return bool(valid) and fid > 0 and self.process is not None and self.process.poll() is None

    def _drain_stderr(self, force: bool = False) -> None:
        """少量读取 helper 输出，避免管道塞满。只做诊断，不阻塞主循环。"""
        if self._fallback_camera is not None:
            return
        if self.process is None:
            return
        now = time.perf_counter()
        if not force and now - self._last_stderr_report < 2.0:
            return
        self._last_stderr_report = now

        # 进程还活着时不能安全阻塞 readline，这里只在已退出或 force 时尝试 communicate(timeout=0.01)。
        if self.process.poll() is not None:
            try:
                out, err = self.process.communicate(timeout=0.05)
                if out.strip():
                    print(out.strip())
                if err.strip():
                    print(err.strip())
            except Exception:
                pass

    def _maybe_restart(self) -> None:
        if self._fallback_camera is not None:
            return
        if not self.restart_on_crash:
            return
        if self.process is not None and self.process.poll() is None:
            return

        now = time.perf_counter()
        if now - self._last_restart_time < 3.0:
            return

        exitcode = self.process.poll() if self.process is not None else None
        print(f"⚠️ [Astra-S:{self.name}] 外部进程已退出 exitcode={exitcode}，尝试重启...", flush=True)
        self._drain_stderr(force=True)
        self._start_process()

    def read(self) -> tuple[np.ndarray, int, float]:
        if self._fallback_camera is not None:
            return self._fallback_camera.read()
        self._maybe_restart()
        image, fid, ts, _, _ = self._read_stable_meta_and_frame()
        return image, fid, ts

    def read_rgb(self) -> np.ndarray:
        if self._fallback_camera is not None:
            return self._fallback_camera.read_rgb()
        return self.read()[0]

    def status(self) -> str:
        if self._fallback_camera is not None:
            return f"{self.name}:backend={self._backend},valid={self.valid},fid=1,age=0ms,err=0,exit=None"
        self._maybe_restart()
        _, fid, ts, valid, err = self._read_meta_once()
        alive = self.process is not None and self.process.poll() is None
        exitcode = self.process.poll() if self.process is not None else None
        age_ms = (time.perf_counter() - ts) * 1000.0 if ts > 0 else float("inf")
        return (
            f"{self.name}:ext_alive={alive},valid={bool(valid)},"
            f"fid={int(fid)},age={age_ms:.0f}ms,err={int(err)},exit={exitcode}"
        )

    def release(self) -> None:
        if self._fallback_camera is not None:
            try:
                self._fallback_camera.release()
            except Exception:
                pass
            self._cleanup_shared_memory()
            return
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1.0)
            except Exception:
                pass

        self._cleanup_shared_memory()


# =============================================================================
# 0.6 🌟 新增：专为 RealSense D405 设计的后台多线程读取类
# =============================================================================
class ThreadedRealSenseCamera:
    """
    RealSense D405 后台 RGB 读取线程。

    关键改进：
        1. 后台线程内完成 BGR/RGB 转换，主循环只 copy 缓存图。
        2. 使用 wait_for_frames(timeout_ms=200)，让相机线程阻塞等待真实新帧，
           不在 20ms try_wait 轮询中频繁空转。
        3. 维护 frame_id / frame_timestamp，用于诊断保存数据中的重复帧与卡顿。
    """

    def __init__(self, name="realsense_cam", width=640, height=480, fps=30):
        self.name = name
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        self.valid = False
        self.frame_rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.frame_id = 0
        self.frame_timestamp = 0.0
        self.lock = threading.Lock()
        self.running = True
        self.thread = None
        self.last_error_time = 0.0
        self.error_msg = ""

        print(f"⏳ 正在初始化 RealSense 相机 [{self.name}]...")
        try:
            self.profile = self.pipeline.start(self.config)
            self.valid = True
            print(f"✅ [成功] RealSense [{self.name}] 已启动: {self.width}x{self.height} @ {self.fps} FPS")
        except RuntimeError as e:
            print(f"⚠️ [警告] RealSense [{self.name}] 标准模式启动失败，尝试自动兼容模式: {e}")
            self.pipeline = rs.pipeline()
            fallback_config = rs.config()
            fallback_config.enable_stream(rs.stream.color)
            try:
                self.profile = self.pipeline.start(fallback_config)
                self.valid = True
                print(f"✅ [成功] RealSense [{self.name}] 已在自动兼容模式下启动。")
            except Exception as e2:
                print(f"❌ [失败] RealSense [{self.name}] 初始化失败: {e2}")
                self.error_msg = str(e2)
                self.valid = False

        if self.valid:
            self.thread = threading.Thread(target=self._update, name=f"RealSense-{self.name}", daemon=True)
            self.thread.start()

    def _update(self):
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=200)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                img = np.asanyarray(color_frame.get_data())

                # 后台线程统一转成 RGB，主循环不再 cvtColor。
                fmt = color_frame.profile.format()
                if fmt == rs.format.bgr8:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif fmt == rs.format.rgb8:
                    rgb = img.copy()
                else:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if rgb.shape[1] != self.width or rgb.shape[0] != self.height:
                    rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

                now = time.perf_counter()
                with self.lock:
                    self.frame_rgb = rgb.copy()
                    self.frame_id += 1
                    self.frame_timestamp = now
                    self.error_msg = ""

            except Exception as e:
                now = time.perf_counter()
                with self.lock:
                    self.error_msg = str(e)
                if now - self.last_error_time > 2.0:
                    print(f"⚠️ [RealSense:{self.name}] 后台读帧异常: {e}")
                    self.last_error_time = now
                time.sleep(0.002)

    def read(self) -> tuple[np.ndarray, int, float]:
        with self.lock:
            return self.frame_rgb.copy(), int(self.frame_id), float(self.frame_timestamp)

    def read_rgb(self):
        return self.read()[0]

    def status(self) -> str:
        with self.lock:
            fid = int(self.frame_id)
            ts = float(self.frame_timestamp)
            ok = bool(self.valid)
            err = self.error_msg
        age_ms = (time.perf_counter() - ts) * 1000.0 if ts > 0 else float("inf")
        if ok:
            return f"{self.name}:valid=True,fid={fid},age={age_ms:.0f}ms"
        return f"{self.name}:valid=False,fid={fid},err={err[:30]}"

    def release(self):
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.valid:
            try:
                self.pipeline.stop()
            except Exception as e:
                print(f"⚠️ [RealSense] 停止 pipeline 时出现异常: {e}")


# =============================================================================
# 0.7 真实触觉与 IMU 读取类
# =============================================================================
class ThreadedYbImuReader:
    """Yb IMU 后台接收封装，read() 返回训练兼容 10 维和完整诊断字段。"""

    def __init__(self, port="/dev/ttyUSB1", report_rate=50, alpha=0.9, name="imu_left"):
        self.name = name
        self.port = port
        self.alpha = clamp(float(alpha), 0.0, 1.0)
        self.imu = None
        self.valid = False
        self.frame_id = 0
        self.timestamp = 0.0
        self.error_msg = ""
        self._filtered_quat = None
        self._filtered_gyro = None
        self._filtered_accel = None

        try:
            imu_dir = _find_first_existing(["IMU_Yb"])
            if imu_dir is not None and str(imu_dir) not in sys.path:
                sys.path.insert(0, str(imu_dir))
            from YbImuLib import YbImuSerial

            print(f"⏳ 正在初始化 Yb IMU [{self.name}] port={self.port} ...")
            self.imu = YbImuSerial(self.port, debug=False)
            self.imu.create_receive_threading()
            try:
                self.imu.set_report_rate(int(report_rate))
            except Exception as exc:
                print(f"⚠️ [IMU:{self.name}] 设置上报频率失败，继续使用设备默认频率: {exc}")
            time.sleep(0.1)
            self.valid = True
            print(f"✅ [成功] Yb IMU [{self.name}] 已启动。")
        except Exception as exc:
            self.error_msg = str(exc)
            self.valid = False
            print(f"⚠️ [IMU:{self.name}] 初始化失败，将保存零 IMU 数据: {exc}")

    def read(self) -> dict:
        if not self.valid or self.imu is None:
            now = time.perf_counter()
            return {
                "imu_left": np.zeros(IMU_LEFT_DIM, dtype=np.float32),
                "quat": np.zeros(4, dtype=np.float32),
                "gyro": np.zeros(3, dtype=np.float32),
                "accel": np.zeros(3, dtype=np.float32),
                "mag": np.zeros(3, dtype=np.float32),
                "euler": np.zeros(3, dtype=np.float32),
                "baro": np.zeros(4, dtype=np.float32),
                "frame_id": -1,
                "timestamp": now,
            }

        try:
            accel = np.asarray(self.imu.get_accelerometer_data(), dtype=np.float32)
            gyro = np.asarray(self.imu.get_gyroscope_data(), dtype=np.float32)
            mag = np.asarray(self.imu.get_magnetometer_data(), dtype=np.float32)
            euler = np.asarray(self.imu.get_imu_attitude_data(ToAngle=True), dtype=np.float32)
            quat = np.asarray(self.imu.get_imu_quaternion_data(), dtype=np.float32)
            baro = np.asarray(self.imu.get_baro_data(), dtype=np.float32)

            if self.alpha < 1.0:
                if self._filtered_quat is not None and float(np.dot(self._filtered_quat, quat)) < 0.0:
                    quat = -quat
                quat_f = self.alpha * quat + (1.0 - self.alpha) * (
                    self._filtered_quat if self._filtered_quat is not None else quat
                )
                gyro_f = self.alpha * gyro + (1.0 - self.alpha) * (
                    self._filtered_gyro if self._filtered_gyro is not None else gyro
                )
                accel_f = self.alpha * accel + (1.0 - self.alpha) * (
                    self._filtered_accel if self._filtered_accel is not None else accel
                )
                quat_norm = float(np.linalg.norm(quat_f))
                if quat_norm > 1e-6:
                    quat_f = quat_f / quat_norm
            else:
                quat_f = quat
                gyro_f = gyro
                accel_f = accel

            imu_left = np.concatenate([quat_f, gyro_f, accel_f]).astype(np.float32)
            self._filtered_quat = quat_f.astype(np.float32, copy=True)
            self._filtered_gyro = gyro_f.astype(np.float32, copy=True)
            self._filtered_accel = accel_f.astype(np.float32, copy=True)

            self.frame_id += 1
            self.timestamp = time.perf_counter()
            self.error_msg = ""
            return {
                "imu_left": imu_left.astype(np.float32, copy=False),
                "quat": quat_f.astype(np.float32, copy=False),
                "gyro": gyro_f.astype(np.float32, copy=False),
                "accel": accel_f.astype(np.float32, copy=False),
                "mag": mag,
                "euler": euler,
                "baro": baro,
                "frame_id": self.frame_id,
                "timestamp": self.timestamp,
            }
        except Exception as exc:
            self.error_msg = str(exc)
            self.valid = False
            return self.read()

    def status(self) -> str:
        if not self.valid:
            return f"{self.name}:valid=False,err={self.error_msg[:30]}"
        age_ms = (time.perf_counter() - self.timestamp) * 1000.0 if self.timestamp > 0 else float("inf")
        return f"{self.name}:valid=True,fid={self.frame_id},age={age_ms:.0f}ms"

    def release(self) -> None:
        if self.imu is None:
            return
        try:
            if hasattr(self.imu, "close"):
                self.imu.close()
            else:
                self.imu._dev.close()
        except Exception:
            pass


class ThreadedFlexiTacReader:
    """FlexiTac 16x32 串口帧读取，直接输出右侧单片触觉裁切后的 12x30。"""

    ROWS = 16
    COLS = 32
    FRAME_BYTES = ROWS * COLS
    MAGIC = b"\xAA\x55"
    ROW_SLICE = slice(-TACTILE_RAW_ROWS, None)
    COL_SLICE = slice(1, -1)

    def __init__(
        self,
        port="/dev/ttyUSB2",
        baud=2_000_000,
        init_frames=30,
        threshold=20.0,
        noise_scale=60.0,
        alpha=1.0,
        name="flexitac",
    ):
        self.name = name
        self.port = port
        self.baud = int(baud)
        self.init_frames = max(int(init_frames), 1)
        self.threshold = float(threshold)
        self.noise_scale = max(float(noise_scale), 1e-6)
        self.alpha = clamp(float(alpha), 0.0, 1.0)
        self.valid = False
        self.running = False
        self.error_msg = ""
        self.frame_id = 0
        self.timestamp = 0.0
        self.lock = threading.Lock()
        self.latest_tactile = np.zeros((TACTILE_RAW_ROWS, TACTILE_RAW_COLS), dtype=np.float32)
        self._filtered = None
        self._serial = None
        self._thread = None

        try:
            import serial

            print(f"⏳ 正在初始化 FlexiTac [{self.name}] port={self.port}, baud={self.baud} ...")
            self._serial = serial.Serial(self.port, self.baud, timeout=0.001)
            self._serial.flush()
            self._serial.reset_input_buffer()
            try:
                self._serial.set_buffer_size(rx_size=262144, tx_size=262144)
            except Exception:
                pass
            self.running = True
            self._thread = threading.Thread(target=self._read_loop, name=f"FlexiTac-{self.name}", daemon=True)
            self._thread.start()
        except Exception as exc:
            self.error_msg = str(exc)
            self.valid = False
            print(f"⚠️ [FlexiTac:{self.name}] 初始化失败，将保存零触觉数据: {exc}")

    def _extract_next_frame(self, ring):
        idx = ring.find(self.MAGIC)
        if idx < 0:
            return None, bytearray(ring[-1:] if len(ring) > 1 else ring)
        end = idx + 2 + self.FRAME_BYTES
        if len(ring) >= end:
            return bytes(ring[idx + 2:end]), bytearray(ring[end:])
        return None, bytearray(ring[idx:])

    def _extract_latest_complete_frame(self, ring):
        positions = []
        start = 0
        while True:
            idx = ring.find(self.MAGIC, start)
            if idx < 0:
                break
            positions.append(idx)
            start = idx + 2
        if not positions:
            return None, bytearray(ring[-1:] if len(ring) > 1 else ring)
        for idx in reversed(positions):
            end = idx + 2 + self.FRAME_BYTES
            if len(ring) >= end:
                return bytes(ring[idx + 2:end]), bytearray(ring[end:])
        return None, bytearray(ring[positions[-1]:])

    def _frame_from_bytes(self, frame_bytes):
        return np.frombuffer(frame_bytes, dtype=np.uint8).reshape((self.ROWS, self.COLS)).astype(np.float32)

    def _normalize(self, contact_crop):
        contact_crop = contact_crop.astype(np.float32, copy=False)
        max_val = float(np.max(contact_crop))
        scale = max_val + 1e-6 if max_val >= self.threshold else self.noise_scale
        norm = contact_crop / scale
        np.clip(norm, 0.0, 1.0, out=norm)
        return norm.astype(np.float32, copy=False)

    def _read_loop(self):
        ring = bytearray()
        baseline_frames = []
        assert self._serial is not None
        print(f"⏳ [FlexiTac:{self.name}] 初始化 baseline，请保持触觉阵列无接触...")

        while self.running and len(baseline_frames) < self.init_frames:
            chunk = self._serial.read(65536)
            if not chunk:
                continue
            ring.extend(chunk)
            if len(ring) > 50000:
                ring = ring[-50000:]
            while self.running and len(baseline_frames) < self.init_frames:
                frame_bytes, ring = self._extract_next_frame(ring)
                if frame_bytes is None:
                    break
                baseline_frames.append(self._frame_from_bytes(frame_bytes))

        if not baseline_frames:
            self.error_msg = "baseline init failed"
            print(f"⚠️ [FlexiTac:{self.name}] baseline 初始化失败。")
            return

        baseline = np.median(np.stack(baseline_frames, axis=0), axis=0).astype(np.float32)
        self.valid = True
        print(f"✅ [成功] FlexiTac [{self.name}] baseline 完成，开始实时采集右侧触觉 12x30。")

        while self.running:
            waiting = self._serial.in_waiting
            chunk = self._serial.read(waiting if waiting > 0 else 4096)
            if not chunk:
                continue
            ring.extend(chunk)
            if len(ring) > 50000:
                ring = ring[-50000:]

            frame_bytes, ring = self._extract_latest_complete_frame(ring)
            if frame_bytes is None:
                continue

            raw_frame = self._frame_from_bytes(frame_bytes)
            contact = raw_frame - baseline - self.threshold
            np.clip(contact, 0.0, 100.0, out=contact)
            raw_30x12 = self._normalize(contact[self.ROW_SLICE, self.COL_SLICE])
            if self._filtered is None or self.alpha >= 1.0:
                filtered = raw_30x12
            else:
                filtered = self.alpha * raw_30x12 + (1.0 - self.alpha) * self._filtered
            self._filtered = filtered.astype(np.float32, copy=True)

            with self.lock:
                self.latest_tactile = filtered.astype(np.float32, copy=True)
                self.frame_id += 1
                self.timestamp = time.perf_counter()

    def read(self) -> tuple[np.ndarray, int, float]:
        with self.lock:
            return self.latest_tactile.copy(), int(self.frame_id), float(self.timestamp)

    def status(self) -> str:
        if not self.valid:
            return f"{self.name}:valid=False,fid={self.frame_id},err={self.error_msg[:30]}"
        age_ms = (time.perf_counter() - self.timestamp) * 1000.0 if self.timestamp > 0 else float("inf")
        return f"{self.name}:valid=True,fid={self.frame_id},age={age_ms:.0f}ms,shape=12x30"

    def release(self) -> None:
        self.running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass


# =============================================================================
# 1. 数据保存与终端交互逻辑 (ALOHA 规范 + 相机流)
# =============================================================================

def save_to_hdf5():
    """标准 ALOHA 规范 .hdf5 数据集安全持久化落盘（含影像流）"""
    global recorded_timestamps, recorded_qpos, recorded_qvel, recorded_action
    global recorded_imu_left, recorded_imu_quat, recorded_imu_gyro, recorded_imu_accel, recorded_imu_mag
    global recorded_imu_euler, recorded_imu_baro, recorded_imu_frame_ids, recorded_imu_timestamps
    global recorded_tactile, recorded_tactile_frame_ids, recorded_tactile_timestamps
    global recorded_images, recorded_image_frame_ids, recorded_image_timestamps

    if not recorded_timestamps:
        print("\n[💾 导出失败] 未采集到有效数据。")
        return

    task_name = _config["task_name"]
    task_sub_dir = _config["base_save_dir"] / task_name
    task_sub_dir.mkdir(parents=True, exist_ok=True)

    if _config["episode_idx"] is not None:
        final_episode_idx = _config["episode_idx"]
    else:
        existing_episodes = []
        for p in task_sub_dir.glob("episode_*.hdf5"):
            try:
                idx = int(p.stem.split("_")[1])
                existing_episodes.append(idx)
            except (IndexError, ValueError):
                continue
        final_episode_idx = max(existing_episodes) + 1 if existing_episodes else 0

    file_name = f"episode_{final_episode_idx}.hdf5"
    file_path = task_sub_dir / file_name
    total_frames = len(recorded_timestamps)

    print(f"\n\n[💾 存储线程] 正在向硬盘写入 {file_name} ({total_frames} 帧 ALOHA 规范数据)...")
    t0 = time.time()

    num_datasets = 16 + 3 * len(_config["camera_names"])

    try:
        with h5py.File(file_path, 'w') as f:
            with tqdm(total=num_datasets, desc="📝 HDF5数据集落盘", bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
                f.create_dataset('/time', data=np.array(recorded_timestamps, dtype=np.float32), compression="gzip")
                pbar.update(1)

                f.create_dataset('/observations/qpos', data=np.array(recorded_qpos, dtype=np.float32),
                                 compression="gzip")
                pbar.update(1)

                f.create_dataset('/observations/qvel', data=np.array(recorded_qvel, dtype=np.float32),
                                 compression="gzip")
                pbar.update(1)

                f.create_dataset('/action/target_pos', data=np.array(recorded_action, dtype=np.float32),
                                 compression="gzip")
                pbar.update(1)

                f.create_dataset('/observations/imu_left', data=np.asarray(recorded_imu_left, dtype=np.float32),
                                 compression="gzip")
                pbar.update(1)
                f.create_dataset('/observations/imu_quat', data=np.asarray(recorded_imu_quat, dtype=np.float32),
                                 compression="gzip")
                pbar.update(1)
                f.create_dataset('/observations/imu_gyro', data=np.asarray(recorded_imu_gyro, dtype=np.float32),
                                 compression="gzip")
                pbar.update(1)
                f.create_dataset('/observations/imu_accel', data=np.asarray(recorded_imu_accel, dtype=np.float32),
                                 compression="gzip")
                pbar.update(1)
                f.create_dataset('/observations/imu_mag', data=np.asarray(recorded_imu_mag, dtype=np.float32),
                                 compression="gzip")
                pbar.update(1)
                f.create_dataset('/observations/imu_euler', data=np.asarray(recorded_imu_euler, dtype=np.float32),
                                 compression="gzip")
                pbar.update(1)
                f.create_dataset('/observations/imu_baro', data=np.asarray(recorded_imu_baro, dtype=np.float32),
                                 compression="gzip")
                pbar.update(1)
                f.create_dataset('/observations/imu_frame_ids', data=np.asarray(recorded_imu_frame_ids, dtype=np.int64),
                                 compression="gzip")
                pbar.update(1)
                f.create_dataset('/observations/imu_timestamps', data=np.asarray(recorded_imu_timestamps, dtype=np.float64),
                                 compression="gzip")
                pbar.update(1)

                tactile_ds = f.create_dataset(
                    '/observations/tactile',
                    data=np.asarray(recorded_tactile, dtype=np.float32),
                    compression="gzip",
                )
                tactile_ds.attrs['sensor_mount'] = 'right_gripper'
                tactile_ds.attrs['layout'] = 'height,width'
                tactile_ds.attrs['source'] = 'FlexiTac 16x32 cropped to last 12 rows and middle 30 columns'
                pbar.update(1)

                f.create_dataset(
                    '/observations/tactile_frame_ids',
                    data=np.asarray(recorded_tactile_frame_ids, dtype=np.int64),
                    compression="gzip",
                )
                pbar.update(1)
                f.create_dataset(
                    '/observations/tactile_timestamps',
                    data=np.asarray(recorded_tactile_timestamps, dtype=np.float64),
                    compression="gzip",
                )
                pbar.update(1)

                for cam_name in _config["camera_names"]:
                    img_array = np.stack(recorded_images[cam_name], axis=0).astype(np.uint8)
                    f.create_dataset(
                        f'/observations/images/{cam_name}',
                        data=img_array,
                        compression="gzip",
                        chunks=(1, img_array.shape[1], img_array.shape[2], img_array.shape[3]),
                    )
                    pbar.update(1)

                    frame_ids = np.asarray(recorded_image_frame_ids[cam_name], dtype=np.int64)
                    f.create_dataset(f'/observations/image_frame_ids/{cam_name}', data=frame_ids, compression="gzip")
                    pbar.update(1)

                    frame_ts = np.asarray(recorded_image_timestamps[cam_name], dtype=np.float64)
                    f.create_dataset(f'/observations/image_timestamps/{cam_name}', data=frame_ts, compression="gzip")
                    pbar.update(1)

                    unique_count = len(np.unique(frame_ids)) if frame_ids.size else 0
                    duplicate_rate = 1.0 - unique_count / max(len(frame_ids), 1)
                    print(
                        f"📷 [{cam_name}] 保存帧数={len(frame_ids)}, "
                        f"真实相机新帧数={unique_count}, 重复率={duplicate_rate * 100:.1f}%"
                    )

            f.attrs['task_name'] = task_name
            f.attrs['episode_idx'] = final_episode_idx
            f.attrs['episode_len'] = _config["episode_len"]
            f.attrs['total_frames'] = total_frames
            f.attrs['hz_rate'] = _config["rate"]
            f.attrs['duration_seconds'] = total_frames * _config["dt"]
            f.attrs['robot_name'] = 'wheeled_dual_arm_robot'
            f.attrs['sim'] = False
            f.attrs['imu_left_layout'] = 'quat_wxyz,gyro_xyz_rad_s,accel_xyz_g'
            f.attrs['tactile_shape'] = '12,30'
            f.attrs['tactile_sensor_mount'] = 'right_gripper'

        print(f"🎉 [💾 导出成功] 数据集固化完成！耗时: {time.time() - t0:.2f}s")
        print(f"📄 数据文件路径: {file_path.resolve()}\n")

    except Exception as e:
        print(f"❌ [💾 导出异常] 写入 HDF5 失败: {e}\n")

    recorded_timestamps.clear()
    recorded_qpos.clear()
    recorded_qvel.clear()
    recorded_action.clear()
    recorded_imu_left.clear()
    recorded_imu_quat.clear()
    recorded_imu_gyro.clear()
    recorded_imu_accel.clear()
    recorded_imu_mag.clear()
    recorded_imu_euler.clear()
    recorded_imu_baro.clear()
    recorded_imu_frame_ids.clear()
    recorded_imu_timestamps.clear()
    recorded_tactile.clear()
    recorded_tactile_frame_ids.clear()
    recorded_tactile_timestamps.clear()
    for cam_name in _config["camera_names"]:
        recorded_images[cam_name].clear()
        recorded_image_frame_ids[cam_name].clear()
        recorded_image_timestamps[cam_name].clear()


def terminal_keyboard_listener():
    global _is_recording, _running
    while _running:
        try:
            input()
        except (KeyboardInterrupt, EOFError):
            break

        if not _running: break

        if not _is_recording:
            total_seconds = _config["episode_len"] * _config["dt"]
            print(
                f"\n🚀 [采集触发] 开始录制！频率: {_config['rate']}Hz, 目标长度: {_config['episode_len']} 步 ({total_seconds:.2f} 秒)")
            _is_recording = True

            def progress_bar_runner():
                total_steps = _config["episode_len"]
                with tqdm(total=total_steps, desc=f"🔴 [{_config['task_name']}] 运动轨迹录制中",
                          bar_format="{l_bar}{bar:40}{r_bar} [{elapsed}<{remaining}]") as pbar:

                    last_count = 0
                    while _is_recording and _running:
                        time.sleep(0.05)
                        current_count = len(recorded_timestamps)
                        pbar.update(current_count - last_count)
                        last_count = current_count

                    if last_count < total_steps:
                        pbar.update(total_steps - last_count)

                save_to_hdf5()
                print("💡 [提示] 随时再次按下【Enter (回车键)】可录制下一段数据。")

            threading.Thread(target=progress_bar_runner, daemon=True).start()
        else:
            print("\n⚠️ [警告] 系统当前正处于录制中，请勿重复操作。")


# =============================================================================
# 1.5 ~ 8. 路径导入、工具函数、命令行参数 (与原版保持一致)
# =============================================================================
CURRENT_DIR = Path(__file__).resolve().parent


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    for p in [CURRENT_DIR, *CURRENT_DIR.parents]: roots.append(p)
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]: roots.append(p)
    unique_roots = []
    seen = set()
    for r in roots:
        if r not in seen:
            unique_roots.append(r)
            seen.add(r)
    return unique_roots


def _find_first_existing(relative_paths: list[str]) -> Path | None:
    for root in _candidate_roots():
        for rel in relative_paths:
            p = root / rel
            if p.exists(): return p
    return None


def _inject_paths() -> tuple[Path | None, Path | None]:
    sdk_dir = _find_first_existing(["STservo_sdk", "Python/STservo_sdk"])
    robot_pkg_dir = _find_first_existing(["reBotArm_control_py", "Python/reBotArm_control_py"])
    add_paths: list[Path] = []
    if sdk_dir is not None:
        add_paths.append(sdk_dir);
        add_paths.append(sdk_dir.parent)
    if robot_pkg_dir is not None: add_paths.append(robot_pkg_dir.parent)
    add_paths.extend(_candidate_roots()[:5])
    for p in add_paths:
        sp = str(p)
        if sp not in sys.path: sys.path.insert(0, sp)
    return sdk_dir, robot_pkg_dir


SDK_DIR, ROBOT_PKG_DIR = _inject_paths()

try:
    from STservo_sdk import *  # noqa: F401,F403
except Exception as exc:
    print("❌ 无法导入 STservo_sdk。")
    raise exc


def _load_robot_arm_class():
    try:
        from reBotArm_control_py.actuator import RobotArm
        return RobotArm
    except ImportError:
        pass
    possible_arm_py: list[Path] = []
    if ROBOT_PKG_DIR is not None: possible_arm_py.append(ROBOT_PKG_DIR / "actuator" / "arm.py")
    for root in _candidate_roots():
        possible_arm_py.append(root / "reBotArm_control_py" / "actuator" / "arm.py")
        possible_arm_py.append(root / "Python" / "reBotArm_control_py" / "actuator" / "arm.py")
    for arm_py in possible_arm_py:
        if arm_py.exists():
            spec = importlib.util.spec_from_file_location("_rebotarm_actuator_arm", arm_py)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                return module.RobotArm
    raise ImportError("无法加载 RobotArm。")


def _load_gripper_cfg_func():
    try:
        from reBotArm_control_py.actuator.gripper import load_cfg
        return load_cfg
    except ImportError:
        pass
    possible_gripper_py: list[Path] = []
    if ROBOT_PKG_DIR is not None: possible_gripper_py.append(ROBOT_PKG_DIR / "actuator" / "gripper.py")
    for root in _candidate_roots():
        possible_gripper_py.append(root / "reBotArm_control_py" / "actuator" / "gripper.py")
        possible_gripper_py.append(root / "Python" / "reBotArm_control_py" / "actuator" / "gripper.py")
    for gripper_py in possible_gripper_py:
        if gripper_py.exists():
            spec = importlib.util.spec_from_file_location("_rebotarm_actuator_gripper", gripper_py)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                return module.load_cfg
    raise ImportError("无法加载 gripper.py 中的 load_cfg。")


def _default_xml_path() -> Path:
    candidates = ["mujoco/xml/rebot_gripper/sim_reBot_grasp.xml",
                  "Python/Servo_control/xml/rebot_gripper/sim_reBot_grasp.xml"]
    p = _find_first_existing(candidates)
    return p if p else CURRENT_DIR / "xml" / "rebot_gripper" / "sim_reBot_grasp.xml"


def _default_gripper_cfg_path() -> Path:
    candidates = ["config/gripper.yaml", "Python/config/gripper.yaml"]
    p = _find_first_existing(candidates)
    return p if p else CURRENT_DIR / "config" / "gripper.yaml"


DEFAULT_XML = _default_xml_path()
DEFAULT_GRIPPER_CFG = _default_gripper_cfg_path()
DEFAULT_SERVO_PORT = "COM6" if os.name == "nt" else "/dev/ttyUSB0"

ARM_SERVO_IDS = [1, 2, 3, 4, 5, 6]
GRIPPER_SERVO_ID = 7
ARM_DOF = len(ARM_SERVO_IDS)
SERVO_DIGITAL_RANGE = 4095.0
SERVO_ANGLE_RANGE = 360.0

JOINT_LIMITS_DEG = {
    1: {"min_deg": 50.0, "max_deg": 300.0, "home_deg": 180.0},
    2: {"min_deg": 10.0, "max_deg": 180.0, "home_deg": 180.0},
    3: {"min_deg": 22.0, "max_deg": 180.0, "home_deg": 180.0},
    4: {"min_deg": 100.0, "max_deg": 270.0, "home_deg": 180.0},
    5: {"min_deg": 90.0, "max_deg": 270.0, "home_deg": 180.0},
    6: {"min_deg": 90.0, "max_deg": 270.0, "home_deg": 180.0},
    7: {"min_deg": 90.0, "max_deg": 180.0, "home_deg": 180.0},
}
SAFETY_MARGIN_DEG = 0.0
DEFAULT_SERVO_TO_SIM_SIGN = np.array([-1.0, 1.0, 1.0, -1.0, -1.0, -1.0], dtype=np.float64)
DEFAULT_SIM_HOME_RAD = np.zeros(6, dtype=np.float64)
GRIPPER_SERVO_CLOSED_DEG = 90.0
GRIPPER_SERVO_OPEN_DEG = 180.0
DEFAULT_GRIPPER_REAL_CLOSED_RAD = 0.2
DEFAULT_GRIPPER_REAL_OPEN_RAD = -5.8


def clamp(val: float, min_val: float, max_val: float) -> float: return max(float(min_val),
                                                                           min(float(val), float(max_val)))


def servo_pos_to_deg(pos: int | float) -> float: return float(pos) / SERVO_DIGITAL_RANGE * SERVO_ANGLE_RANGE


def deg_to_rad(deg: float) -> float: return float(deg) * np.pi / 180.0


def limit_servo_deg(servo_id: int, angle_deg: float) -> float:
    cfg = JOINT_LIMITS_DEG[servo_id]
    return clamp(float(angle_deg), cfg["min_deg"] + SAFETY_MARGIN_DEG, cfg["max_deg"] - SAFETY_MARGIN_DEG)


def servo_deg_to_sim_rad(servo_id: int, angle_deg: float, arm_index: int, servo_to_sim_sign: np.ndarray,
                         sim_home_rad: np.ndarray) -> float:
    delta_deg = limit_servo_deg(servo_id, angle_deg) - JOINT_LIMITS_DEG[servo_id]["home_deg"]
    return float(sim_home_rad[arm_index] + servo_to_sim_sign[arm_index] * deg_to_rad(delta_deg))


def servo_deg_array_to_sim_rad(arm_deg_array: np.ndarray, servo_to_sim_sign: np.ndarray,
                               sim_home_rad: np.ndarray) -> np.ndarray:
    q_sim = np.zeros(ARM_DOF, dtype=np.float64)
    for i, servo_id in enumerate(ARM_SERVO_IDS):
        q_sim[i] = servo_deg_to_sim_rad(servo_id, float(arm_deg_array[i]), i, servo_to_sim_sign, sim_home_rad)
    return q_sim


def gripper_servo_deg_to_norm(angle_deg: float, invert_gripper: bool) -> float:
    angle_deg = limit_servo_deg(GRIPPER_SERVO_ID, angle_deg)
    denom = GRIPPER_SERVO_OPEN_DEG - GRIPPER_SERVO_CLOSED_DEG
    norm = 0.0 if abs(denom) < 1e-9 else clamp((angle_deg - GRIPPER_SERVO_CLOSED_DEG) / denom, 0.0, 1.0)
    return float(1.0 - norm) if invert_gripper else float(norm)


def gripper_norm_to_real_rad(norm: float, closed_rad: float, open_rad: float) -> float:
    target = float(closed_rad) + clamp(norm, 0.0, 1.0) * (float(open_rad) - float(closed_rad))
    return clamp(target, min(float(closed_rad), float(open_rad)), max(float(closed_rad), float(open_rad)))


def smooth_update(prev: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    return clamp(float(alpha), 0.0, 1.0) * target + (1.0 - clamp(float(alpha), 0.0, 1.0)) * prev


def smooth_update_scalar(prev: float, target: float, alpha: float) -> float:
    return float(clamp(float(alpha), 0.0, 1.0) * target + (1.0 - clamp(float(alpha), 0.0, 1.0)) * prev)


def read_servo_angle(scs, servo_id: int, last_angle: float) -> tuple[float, bool]:
    try:
        pos, speed, result, error = scs.ReadPosSpeed(servo_id)
        if result == COMM_SUCCESS: return float(limit_servo_deg(servo_id, servo_pos_to_deg(pos))), True
        return float(last_angle), False
    except:
        return float(last_angle), False


def release_servo_torque(scs, servo_ids: list[int]) -> None:
    print("\n🔓 正在释放舵机主手力矩，用于手动拖动遥操作...")
    for servo_id in servo_ids:
        try:
            scs.write1ByteTxRx(servo_id, STS_TORQUE_ENABLE, 0)
        except:
            pass
        time.sleep(0.02)


DEFAULT_CMD_VLIM = np.array([0.8, 0.8, 0.8, 1.2, 1.2, 1.2], dtype=np.float64)
DEFAULT_MAX_STEP = np.array([0.015, 0.015, 0.015, 0.020, 0.020, 0.020], dtype=np.float64)
DEFAULT_SOFT_MARGIN = 0.0
DEFAULT_SETTLE_SAMPLES = 30
DEFAULT_SETTLE_INTERVAL = 0.02
DEFAULT_TRACKING_BREACH_SAMPLES = 20


def _parse_vector(values: list[float] | None, default: np.ndarray, name: str) -> np.ndarray:
    arr = default.astype(np.float64) if values is None else np.asarray(values, dtype=np.float64)
    if arr.shape != default.shape: raise ValueError(f"{name} 长度不匹配")
    return arr


def _joint_id(model: mujoco.MjModel, joint_name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0: raise RuntimeError(f"XML 找不到: {joint_name}")
    return int(jid)


def _clip_rate(target: np.ndarray, previous: np.ndarray, max_step: np.ndarray) -> np.ndarray:
    return previous + np.clip(target - previous, -max_step, max_step)


def _unwrap_near(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) + 2.0 * np.pi * np.round(
        (np.asarray(reference, dtype=np.float64) - np.asarray(values, dtype=np.float64)) / (2.0 * np.pi))


def _sim_to_real_unclipped(q_sim: np.ndarray, signs: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return (np.asarray(q_sim, dtype=np.float64)[:6] - offsets) / signs


def read_stable_positions(arm, reference: np.ndarray, samples: int, interval: float) -> np.ndarray:
    values = []
    for _ in range(max(int(samples), 1)):
        values.append(_unwrap_near(np.asarray(arm.get_positions(request=True)[:6], dtype=np.float64), reference[:6]))
        time.sleep(max(float(interval), 0.0))
    return np.median(np.vstack(values), axis=0)


def close_arm_fast(arm) -> None:
    if arm is None: return
    try:
        arm.disable(retries=0); time.sleep(0.1)
    except:
        pass
    for ctrl in list(getattr(arm, "_ctrl_map", {}).values()):
        try:
            ctrl.shutdown(); time.sleep(0.02); ctrl.close()
        except:
            pass


class SimToRealMapper:
    def __init__(self, model, joint_names, signs, offsets, soft_margin):
        self.joint_names = tuple(joint_names)
        self.signs = np.asarray(signs, dtype=np.float64)
        self.offsets = np.asarray(offsets, dtype=np.float64)
        self.joint_ids = np.array([_joint_id(model, name) for name in self.joint_names], dtype=np.int32)
        sim_ranges = []
        for jid in self.joint_ids:
            sim_ranges.append(
                model.jnt_range[jid].copy() if int(model.jnt_limited[jid]) == 1 else np.array([-np.inf, np.inf],
                                                                                              dtype=np.float64))
        real_limits = (np.asarray(sim_ranges, dtype=np.float64) - self.offsets[:, None]) / self.signs[:, None]
        self.real_lower = np.minimum(real_limits[:, 0], real_limits[:, 1]) + soft_margin
        self.real_upper = np.maximum(real_limits[:, 0], real_limits[:, 1]) - soft_margin

    def real_to_sim(self, q_real: np.ndarray) -> np.ndarray: return np.asarray(q_real, dtype=np.float64)[
                                                                    :len(self.joint_names)] * self.signs + self.offsets

    def sim_to_real(self, q_sim: np.ndarray) -> np.ndarray: return np.clip(
        (np.asarray(q_sim, dtype=np.float64)[:len(self.joint_names)] - self.offsets) / self.signs, self.real_lower,
        self.real_upper)


class SafetyGuard:
    def __init__(self, mapper, max_step, max_start_error, max_tracking_error, tracking_breach_samples):
        self.mapper = mapper;
        self.max_step = np.asarray(max_step, dtype=np.float64)
        self.max_start_error = float(max_start_error);
        self.max_tracking_error = float(max_tracking_error)
        self.tracking_breach_samples = max(int(tracking_breach_samples), 1)
        self.command: np.ndarray | None = None;
        self._tracking_breach_count = 0

    def initialize(self, q_real_now, q_target, allow_large_start) -> np.ndarray:
        self.command = np.asarray(q_real_now, dtype=np.float64)[:6].copy();
        return self.command.copy()

    def next_command(self, q_target, q_feedback) -> np.ndarray:
        q_target = np.clip(np.asarray(q_target, dtype=np.float64)[:6], self.mapper.real_lower, self.mapper.real_upper)
        q_target_cmd = _unwrap_near(q_target, q_feedback)
        tracking_error = np.max(np.abs(q_target_cmd - q_feedback))
        if tracking_error > self.max_tracking_error:
            self._tracking_breach_count += 1
            if self._tracking_breach_count >= self.tracking_breach_samples: raise RuntimeError("达妙真机跟踪误差过大。")
        else:
            self._tracking_breach_count = 0
        self.command = _clip_rate(q_target_cmd, _unwrap_near(self.command, q_feedback), self.max_step)
        return self.command.copy()


def setup_damiao_gripper(arm, gripper_cfg_path, gripper_name_fallback="gripper"):
    if arm is None: return None, None, None
    load_gripper_cfg = _load_gripper_cfg_func()
    g_cfg = load_gripper_cfg(str(gripper_cfg_path))["gripper"]
    shared_damiao_controller = arm._ctrl_map["damiao"]
    gripper_name = getattr(g_cfg, "name", gripper_name_fallback)
    if gripper_name in arm._motor_map:
        g_mot = arm._motor_map[gripper_name]
    else:
        g_mot = shared_damiao_controller.add_damiao_motor(g_cfg.motor_id, g_cfg.feedback_id, g_cfg.model)
        arm._motor_map[gripper_name] = g_mot
    from motorbridge import Mode
    g_mot.ensure_mode(Mode.MIT, 1000)
    shared_damiao_controller.enable_all()
    time.sleep(0.2)
    return g_mot, shared_damiao_controller, gripper_name


def send_damiao_gripper_mit(g_mot, controller, target_rad, kp, kd, tau, request_feedback=True) -> bool:
    if g_mot is None: return False
    try:
        g_mot.send_mit(float(target_rad), 0.0, float(kp), float(kd), float(tau))
        if request_feedback:
            try:
                g_mot.request_feedback()
            except:
                pass
            if controller:
                try:
                    controller.poll_feedback_once()
                except:
                    pass
        return True
    except:
        return False


def get_gripper_feedback_pos(g_mot) -> float | None:
    if g_mot is None: return None
    try:
        return float(g_mot.get_state().pos)
    except:
        return None


def get_gripper_feedback_vel(g_mot) -> float | None:
    if g_mot is None: return None
    try:
        return float(g_mot.get_state().vel)
    except:
        return 0.0


def servo_reader_worker(scs, state_lock, shared_state, read_rate, servo_to_sim_sign, sim_home_rad, enable_gripper,
                        invert_gripper, closed_rad, open_rad):
    global _running
    read_period = 1.0 / max(float(read_rate), 1e-6)
    last_arm_deg = np.array([JOINT_LIMITS_DEG[i]["home_deg"] for i in ARM_SERVO_IDS], dtype=np.float64)
    last_gripper_deg = JOINT_LIMITS_DEG[GRIPPER_SERVO_ID]["home_deg"]

    while _running:
        loop_start = time.perf_counter()
        arm_deg = np.array(last_arm_deg, dtype=np.float64)
        success_count = 0
        failed_ids = []
        for i, servo_id in enumerate(ARM_SERVO_IDS):
            angle_deg, ok = read_servo_angle(scs, servo_id, arm_deg[i])
            arm_deg[i] = angle_deg
            if ok:
                success_count += 1
            else:
                failed_ids.append(servo_id)
        last_arm_deg = arm_deg.copy()
        q_sim = servo_deg_array_to_sim_rad(arm_deg, servo_to_sim_sign, sim_home_rad)

        if enable_gripper:
            gripper_deg, gripper_ok = read_servo_angle(scs, GRIPPER_SERVO_ID, last_gripper_deg)
            if gripper_ok:
                success_count += 1
            else:
                failed_ids.append(GRIPPER_SERVO_ID)
            last_gripper_deg = float(gripper_deg)
            gripper_norm = gripper_servo_deg_to_norm(gripper_deg, invert_gripper)
            gripper_target_rad = gripper_norm_to_real_rad(gripper_norm, closed_rad, open_rad)
        else:
            gripper_deg, gripper_norm, gripper_target_rad = last_gripper_deg, 1.0, open_rad

        with state_lock:
            shared_state.update(
                {"arm_deg": arm_deg.copy(), "target_q_sim": q_sim.copy(), "gripper_deg": float(gripper_deg),
                 "gripper_norm": float(gripper_norm), "gripper_target_rad": float(gripper_target_rad),
                 "success_count": int(success_count), "failed_ids": list(failed_ids),
                 "timestamp": time.perf_counter(), "read_frame": shared_state["read_frame"] + 1})
        sleep_time = read_period - (time.perf_counter() - loop_start)
        if sleep_time > 0: time.sleep(sleep_time)


def wait_for_servo_ready(state_lock, shared_state, min_success_count, timeout=5.0) -> bool:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        with state_lock:
            if shared_state.get("read_frame", 0) > 0 and shared_state.get("success_count",
                                                                          0) >= min_success_count: return True
        time.sleep(0.02)
    return False


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="舵机主手 -> 达妙真机遥操作 (带多相机真实数据采集)")
    parser.add_argument("--task_name", "-t", type=str, default="teleop_task", help="采集任务名称")
    parser.add_argument("--save_dir", "-d", type=str, default="./data", help="数据保存根目录")
    parser.add_argument("--episode_len", "-l", type=int, default=500, help="设定单次录制的时间步数")
    parser.add_argument("--time", "-sec", type=float, default=None, help="以秒为单位设定录制时长")
    parser.add_argument("--episode_idx", "-idx", type=int, default=None, help="显式指定录制索引号")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--joint-names", type=str, default="joint1,joint2,joint3,joint4,joint5,joint6")
    parser.add_argument("--port", type=str, default=DEFAULT_SERVO_PORT)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--read-rate", type=float, default=60.0)
    parser.add_argument("--keep-servo-torque", action="store_true")
    parser.add_argument("--cfg", type=Path, default=None)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--vlim", type=float, nargs=6, default=None)
    parser.add_argument("--max-step", type=float, nargs=6, default=None)
    parser.add_argument("--alpha-master", type=float, default=0.85)
    parser.add_argument("--servo-to-sim-signs", type=float, nargs=6, default=None)
    parser.add_argument("--sim-home", type=float, nargs=6, default=None)
    parser.add_argument("--signs", type=float, nargs=6, default=None)
    parser.add_argument("--offsets", type=float, nargs=6, default=None)
    parser.add_argument("--calibrate-current-as-master", action="store_true")
    parser.add_argument("--soft-margin", type=float, default=DEFAULT_SOFT_MARGIN)
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--gripper-cfg", type=Path, default=DEFAULT_GRIPPER_CFG)
    parser.add_argument("--invert-gripper", action="store_true")
    parser.add_argument("--alpha-gripper", type=float, default=0.85)
    parser.add_argument("--gripper-real-closed-rad", type=float, default=DEFAULT_GRIPPER_REAL_CLOSED_RAD)
    parser.add_argument("--gripper-real-open-rad", type=float, default=DEFAULT_GRIPPER_REAL_OPEN_RAD)
    parser.add_argument("--gripper-kp", type=float, default=1.0)
    parser.add_argument("--gripper-kd", type=float, default=0.05)
    parser.add_argument("--gripper-tau", type=float, default=0.0)
    parser.add_argument("--gripper-send-every", type=int, default=1)
    parser.add_argument("--gripper-delta-threshold", type=float, default=0.003)
    parser.add_argument("--allow-large-start", action="store_true")
    parser.add_argument("--max-start-error", type=float, default=0.25)
    parser.add_argument("--max-tracking-error", type=float, default=1.50)
    parser.add_argument("--tracking-breach-samples", type=int, default=DEFAULT_TRACKING_BREACH_SAMPLES)
    parser.add_argument("--settle-samples", type=int, default=DEFAULT_SETTLE_SAMPLES)
    parser.add_argument("--settle-interval", type=float, default=DEFAULT_SETTLE_INTERVAL)
    parser.add_argument("--max-servo-age", type=float, default=0.5)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--no-imu", action="store_true", help="不采集 Yb IMU，HDF5 中保存零 IMU 数据")
    parser.add_argument("--imu-port", type=str, default="/dev/ttyUSB1", help="Yb IMU 串口")
    parser.add_argument("--imu-report-rate", type=int, default=50, help="Yb IMU 上报频率，设备支持范围约 10~100Hz")
    parser.add_argument("--imu-alpha", type=float, default=0.9, help="IMU EMA 滤波系数，1.0 表示不滤波，0.9 左右可轻微降噪")
    parser.add_argument("--no-tactile", action="store_true", help="不采集 FlexiTac，HDF5 中保存零触觉数据")
    parser.add_argument("--tactile-port", type=str, default="/dev/ttyUSB2", help="FlexiTac 触觉串口")
    parser.add_argument("--tactile-baud", type=int, default=2_000_000, help="FlexiTac 串口波特率")
    parser.add_argument("--tactile-init-frames", type=int, default=30, help="FlexiTac baseline 初始化帧数")
    parser.add_argument("--tactile-threshold", type=float, default=20.0, help="FlexiTac baseline 扣除后的接触阈值")
    parser.add_argument("--tactile-noise-scale", type=float, default=60.0, help="FlexiTac 无接触时归一化尺度")
    parser.add_argument("--tactile-alpha", type=float, default=1.0, help="FlexiTac EMA 滤波系数，1.0 表示不滤波")
    return parser


# =============================================================================
# 10. 主程序
# =============================================================================
def main() -> None:
    global _running, _is_recording, recorded_images, recorded_image_frame_ids, recorded_image_timestamps

    args = build_argparser().parse_args()

    dt_actual = 1.0 / args.rate
    if args.time is not None:
        final_episode_len = int(args.time * args.rate)
    else:
        final_episode_len = args.episode_len

    _config["task_name"] = args.task_name
    _config["base_save_dir"] = Path(args.save_dir)
    _config["episode_len"] = final_episode_len
    _config["dt"] = dt_actual
    _config["rate"] = args.rate
    _config["episode_idx"] = args.episode_idx

    # 🌟 1. 初始化相机流全局缓存槽
    for cam_name in _config["camera_names"]:
        recorded_images[cam_name] = []
        recorded_image_frame_ids[cam_name] = []
        recorded_image_timestamps[cam_name] = []

    # 🌟 2. 构建真实的相机读取对象字典
    # cam_high：外部高空相机，已由普通 USB 摄像头替换为 Orbbec Astra-S。
    # cam_wrist：继续保留 RealSense D405 腕部相机。
    # 如果某个相机打不开，对应 read_rgb() 会返回安全黑图，保证 50Hz 主循环不中断。
    cameras = {}
    if "cam_high" in _config["camera_names"]:
        cameras["cam_high"] = ThreadedAstraSCamera(name="cam_high")
    if "cam_wrist" in _config["camera_names"]:
        cameras["cam_wrist"] = ThreadedRealSenseCamera(name="cam_wrist")

    imu_reader = None if args.no_imu else ThreadedYbImuReader(
        port=args.imu_port,
        report_rate=args.imu_report_rate,
        alpha=args.imu_alpha,
        name="imu_left",
    )
    tactile_reader = None if args.no_tactile else ThreadedFlexiTacReader(
        port=args.tactile_port,
        baud=args.tactile_baud,
        init_frames=args.tactile_init_frames,
        threshold=args.tactile_threshold,
        noise_scale=args.tactile_noise_scale,
        alpha=args.tactile_alpha,
        name="gripper_tactile",
    )

    enable_gripper = not args.no_gripper
    servo_to_sim_sign = _parse_vector(args.servo_to_sim_signs, DEFAULT_SERVO_TO_SIM_SIGN, "signs")
    sim_home_rad = _parse_vector(args.sim_home, DEFAULT_SIM_HOME_RAD, "home")
    signs = _parse_vector(args.signs, np.ones(6, dtype=np.float64), "signs")
    offsets = _parse_vector(args.offsets, np.zeros(6, dtype=np.float64), "offsets")
    vlim = _parse_vector(args.vlim, DEFAULT_CMD_VLIM, "vlim")
    max_step = _parse_vector(args.max_step, DEFAULT_MAX_STEP, "max-step")
    joint_names = tuple(x.strip() for x in args.joint_names.split(",") if x.strip())

    print("\n" + "=" * 60)
    print("  🚀 主从遥操作 + ALOHA 多相机规范数据采集系统就绪")
    print(f"  📝 目标任务名称: {args.task_name}")
    print(f"  ⚡ 控制与录制频率: {args.rate} Hz")
    print(f"  ⏱️ 单次录制规模: {final_episode_len} 步 ({final_episode_len * dt_actual:.2f} 秒)")
    print(f"  📷 已挂载真实相机: {_config['camera_names']}")
    print(f"  🧭 IMU: {'disabled' if imu_reader is None else args.imu_port}")
    print(f"  🖐️ 触觉: {'disabled' if tactile_reader is None else args.tactile_port} -> right gripper 12x30")
    print("=" * 60)

    model = mujoco.MjModel.from_xml_path(str(args.xml))

    portHandler = PortHandler(args.port)
    scs = sts(portHandler)
    if not portHandler.openPort() or not portHandler.setBaudRate(args.baudrate): return

    time.sleep(2.5)
    if not args.keep_servo_torque:
        release_servo_torque(scs, ARM_SERVO_IDS + ([GRIPPER_SERVO_ID] if enable_gripper else []))

    home_arm_deg = np.array([JOINT_LIMITS_DEG[i]["home_deg"] for i in ARM_SERVO_IDS], dtype=np.float64)
    home_q_sim = servo_deg_array_to_sim_rad(home_arm_deg, servo_to_sim_sign, sim_home_rad)
    home_gripper_target_rad = gripper_norm_to_real_rad(
        gripper_servo_deg_to_norm(JOINT_LIMITS_DEG[GRIPPER_SERVO_ID]["home_deg"], args.invert_gripper),
        args.gripper_real_closed_rad, args.gripper_real_open_rad)

    state_lock = threading.Lock()
    shared_state = {
        "arm_deg": home_arm_deg.copy(), "target_q_sim": home_q_sim.copy(),
        "gripper_deg": float(JOINT_LIMITS_DEG[GRIPPER_SERVO_ID]["home_deg"]),
        "gripper_norm": float(
            gripper_servo_deg_to_norm(JOINT_LIMITS_DEG[GRIPPER_SERVO_ID]["home_deg"], args.invert_gripper)),
        "gripper_target_rad": float(home_gripper_target_rad),
        "success_count": 0, "failed_ids": [], "timestamp": time.perf_counter(), "read_frame": 0,
    }

    reader_thread = threading.Thread(target=servo_reader_worker, args=(
        scs, state_lock, shared_state, args.read_rate, servo_to_sim_sign, sim_home_rad, enable_gripper,
        args.invert_gripper,
        args.gripper_real_closed_rad, args.gripper_real_open_rad), daemon=True)
    reader_thread.start()

    if not wait_for_servo_ready(state_lock, shared_state, ARM_DOF + (1 if enable_gripper else 0)): return

    with state_lock:
        initial_q_sim = shared_state["target_q_sim"].copy()
        initial_gripper_target_rad = float(shared_state["gripper_target_rad"])

    RobotArm = _load_robot_arm_class()
    arm = RobotArm(cfg_path=str(args.cfg) if args.cfg else None)
    arm.connect()
    arm.enable()
    arm.mode_pos_vel(vlim=vlim)

    gripper_motor, gripper_controller = None, None
    if enable_gripper:
        gripper_motor, gripper_controller, _ = setup_damiao_gripper(arm, args.gripper_cfg)

    q_feedback = read_stable_positions(arm, _sim_to_real_unclipped(initial_q_sim, signs, offsets), args.settle_samples,
                                       args.settle_interval)
    if args.calibrate_current_as_master: offsets = initial_q_sim.copy() - signs * q_feedback[:6]

    mapper = SimToRealMapper(model, joint_names, signs, offsets, args.soft_margin)
    guard = SafetyGuard(mapper, max_step, args.max_start_error, args.max_tracking_error, args.tracking_breach_samples)
    q_cmd = guard.initialize(q_feedback, mapper.sim_to_real(initial_q_sim), args.allow_large_start)
    arm.pos_vel(q_cmd, vlim=vlim)

    filtered_gripper_target_rad = initial_gripper_target_rad
    if enable_gripper: send_damiao_gripper_mit(gripper_motor, gripper_controller, filtered_gripper_target_rad,
                                               args.gripper_kp, args.gripper_kd, args.gripper_tau)

    listener_thread = threading.Thread(target=terminal_keyboard_listener, daemon=True)
    listener_thread.start()
    print("\n📌 [录制就绪] 控制台按下【回车键】开始采集当前 Episode 数据...")

    cmd_period = 1.0 / args.rate
    filtered_q_sim = initial_q_sim.copy()
    frame = 0

    try:
        while _running:
            loop_start = time.perf_counter()

            with state_lock:
                target_q_sim_raw = shared_state["target_q_sim"].copy()
                gripper_target_rad_raw = float(shared_state["gripper_target_rad"])
                servo_age = loop_start - float(shared_state["timestamp"])

            if servo_age > args.max_servo_age: raise RuntimeError("舵机主手数据超时")

            filtered_q_sim = smooth_update(filtered_q_sim, target_q_sim_raw, args.alpha_master)
            q_target_real = mapper.sim_to_real(filtered_q_sim)

            q_feedback = np.asarray(arm.get_positions(request=True)[:6], dtype=np.float64)
            try:
                v_feedback = np.asarray(arm.get_velocities(request=True)[:6], dtype=np.float64)
            except:
                v_feedback = np.zeros(6, dtype=np.float64)

            q_feedback_unwrapped = _unwrap_near(q_feedback, q_cmd)
            q_cmd = guard.next_command(q_target_real, q_feedback_unwrapped)
            arm.pos_vel(q_cmd, vlim=vlim)

            gripper_fb_pos, gripper_fb_vel = 0.0, 0.0
            if enable_gripper:
                filtered_gripper_target_rad = smooth_update_scalar(filtered_gripper_target_rad, gripper_target_rad_raw,
                                                                   args.alpha_gripper)
                if frame % max(int(args.gripper_send_every), 1) == 0:
                    send_damiao_gripper_mit(gripper_motor, gripper_controller, filtered_gripper_target_rad,
                                            args.gripper_kp, args.gripper_kd, args.gripper_tau, request_feedback=True)

                g_pos = get_gripper_feedback_pos(gripper_motor)
                if g_pos is not None: gripper_fb_pos = g_pos
                g_vel = get_gripper_feedback_vel(gripper_motor)
                if g_vel is not None: gripper_fb_vel = g_vel

            # 🌟 3. 核心截断式采集层（读取真实的 RGB 相机流）
            if _is_recording:
                current_frame_count = len(recorded_timestamps)
                if current_frame_count < _config["episode_len"]:
                    rec_time = current_frame_count * _config["dt"]
                    recorded_timestamps.append(rec_time)

                    recorded_qpos.append(np.concatenate([q_feedback, [gripper_fb_pos]]))
                    recorded_qvel.append(np.concatenate([v_feedback, [gripper_fb_vel]]))
                    recorded_action.append(np.concatenate([q_cmd, [filtered_gripper_target_rad]]))

                    if imu_reader is not None:
                        imu_sample = imu_reader.read()
                    else:
                        imu_sample = {
                            "imu_left": np.zeros(IMU_LEFT_DIM, dtype=np.float32),
                            "quat": np.zeros(4, dtype=np.float32),
                            "gyro": np.zeros(3, dtype=np.float32),
                            "accel": np.zeros(3, dtype=np.float32),
                            "mag": np.zeros(3, dtype=np.float32),
                            "euler": np.zeros(3, dtype=np.float32),
                            "baro": np.zeros(4, dtype=np.float32),
                            "frame_id": -1,
                            "timestamp": 0.0,
                        }
                    recorded_imu_left.append(imu_sample["imu_left"])
                    recorded_imu_quat.append(imu_sample["quat"])
                    recorded_imu_gyro.append(imu_sample["gyro"])
                    recorded_imu_accel.append(imu_sample["accel"])
                    recorded_imu_mag.append(imu_sample["mag"])
                    recorded_imu_euler.append(imu_sample["euler"])
                    recorded_imu_baro.append(imu_sample["baro"])
                    recorded_imu_frame_ids.append(int(imu_sample["frame_id"]))
                    recorded_imu_timestamps.append(float(imu_sample["timestamp"]))

                    if tactile_reader is not None:
                        tactile_frame, tactile_frame_id, tactile_ts = tactile_reader.read()
                    else:
                        tactile_frame = np.zeros((TACTILE_RAW_ROWS, TACTILE_RAW_COLS), dtype=np.float32)
                        tactile_frame_id = -1
                        tactile_ts = 0.0
                    recorded_tactile.append(tactile_frame)
                    recorded_tactile_frame_ids.append(int(tactile_frame_id))
                    recorded_tactile_timestamps.append(float(tactile_ts))

                    # 🌟 动态匹配已挂载的真实相机并获取 RGB 数据。
                    # read() 会返回：最新 RGB 图像、真实相机帧号、真实相机线程时间戳。
                    # 注意：主循环 50Hz，而 Astra-S/D405 多数为 30FPS，因此 HDF5 中有重复 frame_id 是正常的。
                    for cam_name in _config["camera_names"]:
                        if cam_name in cameras:
                            cam_obj = cameras[cam_name]
                            if hasattr(cam_obj, "read"):
                                img, cam_frame_id, cam_ts = cam_obj.read()
                            else:
                                img = cam_obj.read_rgb()
                                cam_frame_id, cam_ts = -1, 0.0
                            recorded_images[cam_name].append(img)
                            recorded_image_frame_ids[cam_name].append(int(cam_frame_id))
                            recorded_image_timestamps[cam_name].append(float(cam_ts))
                        else:
                            recorded_images[cam_name].append(np.zeros((480, 640, 3), dtype=np.uint8))
                            recorded_image_frame_ids[cam_name].append(-1)
                            recorded_image_timestamps[cam_name].append(0.0)
                else:
                    _is_recording = False

            if frame % args.print_every == 0 and not _is_recording:
                cam_status = " | ".join(
                    cam.status() if hasattr(cam, "status") else f"{name}:no-status"
                    for name, cam in cameras.items()
                )
                imu_status = "imu:disabled" if imu_reader is None else imu_reader.status()
                tactile_status = "tactile:disabled" if tactile_reader is None else tactile_reader.status()
                print(
                    f"[{frame:06d}] fb_J1={q_feedback[0]:+.2f} | cmd_J1={q_cmd[0]:+.2f} | "
                    f"gripper_cmd={filtered_gripper_target_rad:+.2f} | {cam_status} | {imu_status} | {tactile_status}",
                    end="\r")

            frame += 1
            sleep_time = cmd_period - (time.perf_counter() - loop_start)
            if sleep_time > 0: time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"\n[停机] {exc}")
    finally:
        _running = False
        close_arm_fast(arm)
        portHandler.closePort()

        # 🌟 4. 安全退出：释放所有硬件相机占用
        for cam in cameras.values():
            cam.release()
        if imu_reader is not None:
            imu_reader.release()
        if tactile_reader is not None:
            tactile_reader.release()
        print("\n🧹 所有硬件资源已安全释放。")


if __name__ == "__main__":
    main()

# python record_rebot_episodes.py --task_name rebot_real_grasp_banana --xml /home/hjx/hjx_file/rebot_devarm_ws/rebotArm_policy_learning/act_tactile/assets/rebotarm_sim_cupboard.xml --port /dev/ttyUSB0 --baudrate 115200 --rate 50 --read-rate 60 --calibrate-current-as-master --save_dir /media/hjx/PSSD/hjx_ws/data/rebot/data_real_tactile --imu-port /dev/ttyUSB1 --tactile-port /dev/ttyUSB2 --tactile-baud 2000000 --tactile-init-frames 30 --episode_len 800 --episode_idx 0
