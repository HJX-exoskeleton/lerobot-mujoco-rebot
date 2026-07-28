#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rebot 真机策略部署入口。

功能:
1. 从 ckpt_dir 读取 policy_best.ckpt / policy_config.pkl / dataset_stats.pkl
2. 读取真机机械臂与夹爪反馈
3. 读取真实相机流
4. 使用 ACT/CNNMLP checkpoint 做实时推理并下发控制
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import struct
import subprocess
import signal
import sys
import threading
import time
from collections import deque
from multiprocessing import shared_memory
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    import pyrealsense2 as rs
except Exception:  # pragma: no cover - 运行环境没有 RealSense SDK 时允许脚本继续加载
    rs = None

CURRENT_DIR = Path(__file__).resolve().parent

_running = True
IMU_DIM = 10
TACTILE_ROWS = 12
TACTILE_COLS = 30


def _sigint_handler(signum, frame) -> None:
    global _running
    print("\n[deploy] 收到退出信号，准备安全关闭...")
    _running = False


signal.signal(signal.SIGINT, _sigint_handler)
signal.signal(signal.SIGTERM, _sigint_handler)


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    for p in [CURRENT_DIR, *CURRENT_DIR.parents]:
        roots.append(p)
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        roots.append(p)

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
            if p.exists():
                return p
    return None


def _inject_paths() -> tuple[Path | None, Path | None]:
    sdk_dir = _find_first_existing(["STservo_sdk", "Python/STservo_sdk"])
    robot_pkg_dir = _find_first_existing(["reBotArm_control_py", "Python/reBotArm_control_py"])

    add_paths: list[Path] = []
    if sdk_dir is not None:
        add_paths.append(sdk_dir)
        add_paths.append(sdk_dir.parent)
    if robot_pkg_dir is not None:
        add_paths.append(robot_pkg_dir.parent)
    add_paths.extend(_candidate_roots()[:6])

    for p in add_paths:
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)

    return sdk_dir, robot_pkg_dir


SDK_DIR, ROBOT_PKG_DIR = _inject_paths()


def _default_act_root() -> Path | None:
    configured = os.environ.get("ACT_TACTILE_ROOT") or os.environ.get("ACT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()

    policy_file = _find_first_existing(
        [
            "act_tactile_rebot/policy.py",
            "ACT_tactile/policy.py",
        ]
    )
    return policy_file.parent if policy_file is not None else None


def _activate_act_root(act_root: Path | None) -> Path:
    if act_root is None:
        raise FileNotFoundError(
            "未找到触觉 ACT 源码目录。请通过 --act-root 或 ACT_TACTILE_ROOT "
            "指定包含触觉版 policy.py 和 detr/ 的 act_tactile_rebot 目录。"
        )

    act_root = act_root.expanduser().resolve()
    required_files = (act_root / "policy.py", act_root / "detr" / "main.py")
    missing = [str(path.relative_to(act_root)) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"触觉 ACT 源码目录不完整: {act_root}，缺少: {', '.join(missing)}"
        )

    policy_source = (act_root / "policy.py").read_text(encoding="utf-8")
    if "tactile" not in policy_source:
        raise RuntimeError(
            f"指定的 ACT policy.py 不包含触觉输入支持: {act_root / 'policy.py'}"
        )

    root_string = str(act_root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    return act_root


try:
    from STservo_sdk import *  # noqa: F401,F403
except Exception as exc:
    print("❌ 无法导入 STservo_sdk。")
    raise exc


def _load_robot_arm_class():
    try:
        from reBotArm_control_py.actuator import RobotArm
        return RobotArm
    except Exception as exc:
        print(f"[导入] 常规导入 RobotArm 失败，尝试直接加载 arm.py: {exc}")

    possible_arm_py: list[Path] = []
    if ROBOT_PKG_DIR is not None:
        possible_arm_py.append(ROBOT_PKG_DIR / "actuator" / "arm.py")
    for root in _candidate_roots():
        possible_arm_py.append(root / "reBotArm_control_py" / "actuator" / "arm.py")
        possible_arm_py.append(root / "Servo_control" / "reBotArm_control_py" / "actuator" / "arm.py")
        possible_arm_py.append(root / "Python" / "reBotArm_control_py" / "actuator" / "arm.py")

    for arm_py in possible_arm_py:
        if arm_py.exists():
            spec = importlib.util.spec_from_file_location("_rebotarm_actuator_arm", arm_py)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                return module.RobotArm

    raise ImportError("无法加载 RobotArm，请检查 reBotArm_control_py/actuator/arm.py 是否存在。")


def _load_gripper_cfg_func():
    try:
        from reBotArm_control_py.actuator.gripper import load_cfg
        return load_cfg
    except Exception:
        pass

    possible_gripper_py: list[Path] = []
    if ROBOT_PKG_DIR is not None:
        possible_gripper_py.append(ROBOT_PKG_DIR / "actuator" / "gripper.py")
    for root in _candidate_roots():
        possible_gripper_py.append(root / "reBotArm_control_py" / "actuator" / "gripper.py")
        possible_gripper_py.append(root / "Servo_control" / "reBotArm_control_py" / "actuator" / "gripper.py")
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


def _load_task_configs():
    candidates = [
        "rebot_scripts/constants.py",
        "act/rebot_scripts/constants.py",
    ]
    path = _find_first_existing(candidates)
    if path is None:
        return {}

    spec = importlib.util.spec_from_file_location("_rebot_task_constants", path)
    if spec is None or spec.loader is None:
        return {}

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "TASK_CONFIGS", {})


def _load_policy_config(ckpt_dir: Path, args: argparse.Namespace):
    cfg_path = ckpt_dir / "policy_config.pkl"
    if cfg_path.exists():
        with open(cfg_path, "rb") as f:
            return pickle.load(f)

    required = {
        "kl_weight": args.kl_weight,
        "chunk_size": args.chunk_size,
        "hidden_dim": args.hidden_dim,
        "dim_feedforward": args.dim_feedforward,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise FileNotFoundError(
            f"找不到 {cfg_path}，且命令行没有提供 {missing}。"
            "请先从训练目录读取 policy_config.pkl，或者补齐这些参数。"
        )

    return {
        "lr": args.lr,
        "num_queries": args.chunk_size,
        "kl_weight": args.kl_weight,
        "hidden_dim": args.hidden_dim,
        "dim_feedforward": args.dim_feedforward,
        "lr_backbone": args.lr_backbone,
        "backbone": args.backbone,
        "enc_layers": args.enc_layers,
        "dec_layers": args.dec_layers,
        "nheads": args.nheads,
        "camera_names": args.camera_names.split(",") if args.camera_names else ["cam_high"],
    }


def _load_stats(ckpt_dir: Path):
    stats_path = ckpt_dir / "dataset_stats.pkl"
    if not stats_path.exists():
        raise FileNotFoundError(f"找不到统计文件: {stats_path}")
    with open(stats_path, "rb") as f:
        return pickle.load(f)


def _default_xml_path() -> Path:
    candidates = [
        "mujoco/xml/rebot_gripper/sim_reBot_grasp.xml",
        "Python/Servo_control/xml/rebot_gripper/sim_reBot_grasp.xml",
        "Servo_control/xml/rebot_gripper/sim_reBot_grasp.xml",
        "mujoco/xml/rebot_fixend/reBot-DevArm_fixend.xml",
        "Python/mujoco/xml/rebot_fixend/reBot-DevArm_fixend.xml",
    ]
    p = _find_first_existing(candidates)
    return p if p else CURRENT_DIR / "xml" / "rebot_gripper" / "sim_reBot_grasp.xml"


def _default_gripper_cfg_path() -> Path:
    candidates = [
        "config/gripper.yaml",
        "Python/config/gripper.yaml",
        "Servo_control/config/gripper.yaml",
    ]
    p = _find_first_existing(candidates)
    return p if p else CURRENT_DIR / "config" / "gripper.yaml"


def _parse_vector(values: list[float] | None, default: np.ndarray, name: str) -> np.ndarray:
    arr = default.astype(np.float64) if values is None else np.asarray(values, dtype=np.float64)
    if arr.shape != default.shape:
        raise ValueError(f"{name} 必须提供 {default.size} 个数，当前为 {arr.size} 个")
    return arr


def _unwrap_near(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    return values + 2.0 * np.pi * np.round((reference - values) / (2.0 * np.pi))


def _clip_rate(target: np.ndarray, previous: np.ndarray, max_step: np.ndarray) -> np.ndarray:
    return previous + np.clip(target - previous, -max_step, max_step)


def _read_arm_velocity_or_zero(arm, request: bool = False) -> np.ndarray:
    try:
        return np.asarray(arm.get_velocities(request=request)[:6], dtype=np.float64)
    except Exception:
        return np.zeros(6, dtype=np.float64)


def close_arm_fast(arm) -> None:
    if arm is None:
        return

    try:
        arm.disable(retries=0)
        time.sleep(0.1)
    except Exception:
        pass

    for ctrl in list(getattr(arm, "_ctrl_map", {}).values()):
        try:
            ctrl.shutdown()
            time.sleep(0.02)
            ctrl.close()
        except Exception:
            pass


def setup_damiao_gripper(arm, gripper_cfg_path: Path):
    if arm is None:
        return None, None
    if not gripper_cfg_path.exists():
        raise FileNotFoundError(f"夹爪配置文件不存在: {gripper_cfg_path}")

    load_gripper_cfg = _load_gripper_cfg_func()
    g_cfg = load_gripper_cfg(str(gripper_cfg_path))["gripper"]
    shared_damiao_controller = arm._ctrl_map.get("damiao")
    if shared_damiao_controller is None:
        raise RuntimeError("未找到 damiao 控制器，无法添加达妙夹爪。")

    gripper_name = getattr(g_cfg, "name", "gripper")
    if gripper_name in arm._motor_map:
        g_mot = arm._motor_map[gripper_name]
    else:
        g_mot = shared_damiao_controller.add_damiao_motor(
            g_cfg.motor_id,
            g_cfg.feedback_id,
            g_cfg.model,
        )
        arm._motor_map[gripper_name] = g_mot

    from motorbridge import Mode

    g_mot.ensure_mode(Mode.MIT, 1000)
    shared_damiao_controller.enable_all()
    time.sleep(0.2)
    return g_mot, shared_damiao_controller


def send_damiao_gripper_mit(g_mot, controller, target_rad: float, kp: float = 1.0, kd: float = 0.05, tau: float = 0.0,
                            request_feedback: bool = True) -> bool:
    if g_mot is None:
        return False
    try:
        g_mot.send_mit(float(target_rad), 0.0, float(kp), float(kd), float(tau))
        if request_feedback:
            try:
                g_mot.request_feedback()
            except Exception:
                pass
            if controller is not None:
                try:
                    controller.poll_feedback_once()
                except Exception:
                    pass
        return True
    except Exception as e:
        print(f"\n⚠️ 夹爪 MIT 命令发送失败: {e}")
        return False


def get_gripper_feedback_pos(g_mot) -> float | None:
    if g_mot is None:
        return None
    try:
        st = g_mot.get_state()
        return None if st is None else float(st.pos)
    except Exception:
        return None


class SafetyGuard:
    def __init__(self, max_step_per_sec: np.ndarray, max_tracking_error: float = 1.0, breach_samples: int = 20):
        self.max_step_per_sec = np.asarray(max_step_per_sec, dtype=np.float64)
        self.max_tracking_error = float(max_tracking_error)
        self.breach_samples = max(int(breach_samples), 1)
        self.command: np.ndarray | None = None
        self._breach_count = 0

    def initialize(self, q_real_now: np.ndarray) -> np.ndarray:
        self.command = np.asarray(q_real_now, dtype=np.float64)[:6].copy()
        return self.command.copy()

    def next_command(self, q_target: np.ndarray, q_feedback: np.ndarray, dt: float) -> np.ndarray:
        if self.command is None:
            raise RuntimeError("SafetyGuard 尚未 initialize。")

        q_target = np.asarray(q_target, dtype=np.float64)[:6]
        q_feedback = np.asarray(q_feedback, dtype=np.float64)[:6]
        dt = max(float(dt), 1e-4)

        q_target_cmd = _unwrap_near(q_target, q_feedback)
        previous_cmd = _unwrap_near(self.command, q_feedback)
        tracking_error = float(np.max(np.abs(q_target_cmd - q_feedback)))

        if tracking_error > self.max_tracking_error:
            self._breach_count += 1
            if self._breach_count >= self.breach_samples:
                raise RuntimeError(f"⚠️ 真机跟踪误差过大 ({tracking_error:.2f} rad)，触发保护。")
        else:
            self._breach_count = 0

        self.command = _clip_rate(q_target_cmd, previous_cmd, self.max_step_per_sec * dt)
        return self.command.copy()


class ThreadedCamera:
    def __init__(self, src=2, width=640, height=480, fps=30, name="camera"):
        self.name = name
        self.src = src
        self.width = width
        self.height = height
        self.fps = int(fps)
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.capture.read()
        self.valid = bool(self.ret)
        self.running = True
        self.lock = threading.Lock()
        self.frame_count = 1 if self.valid else 0
        self.start_time = time.time()
        self.last_frame_time = self.start_time if self.valid else 0.0

        if not self.valid:
            print(f"⚠️ 相机 {self.name} (src={src}) 初始化失败，回退黑屏。")
            self.frame = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            h, w = self.frame.shape[:2]
            print(
                f"✅ 相机 {self.name} (src={src}) 初始化成功: "
                f"frame={w}x{h}, request_fps={self.fps}, actual_fps≈{self.capture.get(cv2.CAP_PROP_FPS):.1f}, "
                f"mean={self.frame.mean():.1f}, std={self.frame.std():.1f}"
            )
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

    def read_rgb(self):
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

    def status_string(self) -> str:
        with self.lock:
            frame = self.frame.copy()
            frame_count = self.frame_count
            last_frame_time = self.last_frame_time
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

    def save_debug_frame(self, save_dir: Path, step: int | str = "init") -> Path:
        save_dir.mkdir(parents=True, exist_ok=True)
        rgb = self.read_rgb()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        out_path = save_dir / f"{self.name}_step_{step}.jpg"
        cv2.imwrite(str(out_path), bgr)
        return out_path

    def release(self):
        self.running = False
        if self.capture.isOpened():
            self.capture.release()

class ThreadedAstraSCamera:
    """
    Astra-S RGB 读取类：主进程接口 + 外部 helper 进程。

    主进程不 import OpenNI；通过 astra_s_shm_server.py 将最新 RGB 帧写入 shared memory。
    """

    DEFAULT_OPENNI2_REDIST = "/home/hjx/orbbec_openni_redist"
    DEFAULT_WIDTH = 640
    DEFAULT_HEIGHT = 480
    DEFAULT_FPS = 30
    DEFAULT_FLIP_HORIZONTAL = True
    DEFAULT_FLIP_VERTICAL = False
    DEFAULT_STARTUP_TIMEOUT = 8.0
    DEFAULT_RESTART_ON_CRASH = True
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
        self._start_time = time.perf_counter()
        self._warned_missing_helper = False
        self._fallback_camera: ThreadedCamera | None = None
        self._backend = "Astra-S"

        print(f"⏳ 正在初始化 Astra-S 相机 [{self.name}]，通过外部 helper 进程读取...")
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
            print(f"✅ [成功] Astra-S [{self.name}] 已输出图像，分辨率 {self.width}x{self.height}")
        else:
            exitcode = self.process.poll() if self.process is not None else None
            print(
                f"⚠️ [Astra-S:{self.name}] 启动后暂未读到图像，fid={fid}, valid={valid}, exitcode={exitcode}。"
                " 主循环将继续输出黑图。"
            )
            self._drain_stderr(force=True)
            if exitcode is not None or not valid:
                self._enter_fallback("未检测到 Astra-S 设备")

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
        if self._fallback_camera is not None:
            return
        if self.process is None:
            return
        now = time.perf_counter()
        if not force and now - self._last_stderr_report < 2.0:
            return
        self._last_stderr_report = now

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
        if not self.helper_path.exists():
            if not self._warned_missing_helper:
                print(f"⚠️ [Astra-S:{self.name}] helper 脚本不存在，保持黑图输出: {self.helper_path}")
                self._warned_missing_helper = True
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
        if fid <= 0:
            image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return image, fid, ts

    def read_rgb(self) -> np.ndarray:
        if self._fallback_camera is not None:
            return self._fallback_camera.read_rgb()
        return self.read()[0]

    def info(self) -> dict:
        if self._fallback_camera is not None:
            info_fn = getattr(self._fallback_camera, "info", None)
            if callable(info_fn):
                info = dict(info_fn())
            else:
                info = {
                    "name": self.name,
                    "backend": "OpenCV",
                    "valid": bool(getattr(self._fallback_camera, "valid", True)),
                    "frame_id": 1,
                    "fps": 0.0,
                    "age_ms": 0.0,
                    "error": "",
                    "exitcode": None,
                    "alive": True,
                    "timestamp": time.perf_counter(),
                }
            info["name"] = self.name
            info["backend"] = self._backend
            return info
        self._maybe_restart()
        _, fid, ts, valid, err = self._read_meta_once()
        alive = self.process is not None and self.process.poll() is None
        exitcode = self.process.poll() if self.process is not None else None
        age_ms = (time.perf_counter() - ts) * 1000.0 if ts > 0 else float("inf")
        elapsed = max(time.perf_counter() - self._start_time, 1e-6)
        fps = float(fid) / elapsed if fid > 0 else 0.0
        return {
            "name": self.name,
            "backend": "Astra-S",
            "valid": bool(valid) and alive,
            "frame_id": int(fid),
            "fps": fps,
            "age_ms": age_ms,
            "error": int(err),
            "exitcode": exitcode,
            "alive": alive,
            "timestamp": float(ts),
        }

    def status(self) -> str:
        if self._fallback_camera is not None:
            return f"{self.name}:backend={self._backend},valid={self.valid},fid=1,fps=0.0,age=0ms,err=0,exit=None"
        info = self.info()
        return (
            f"{info['name']}:backend={info['backend']},valid={info['valid']},fid={info['frame_id']},"
            f"fps={info['fps']:.1f},age={info['age_ms']:.0f}ms,err={info['error']},exit={info['exitcode']}"
        )

    def save_debug_frame(self, save_dir: Path, step: int | str = "init") -> Path:
        save_dir.mkdir(parents=True, exist_ok=True)
        rgb = self.read_rgb()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        out_path = save_dir / f"{self.name}_step_{step}.jpg"
        cv2.imwrite(str(out_path), bgr)
        return out_path

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


class ThreadedRealSenseCamera:
    """
    RealSense D405 后台 RGB 读取线程。

    读取与格式转换均在后台线程完成，主循环只 copy 缓存图。
    """

    def __init__(self, name="realsense_cam", width=640, height=480, fps=30):
        if rs is None:
            raise RuntimeError("pyrealsense2 不可用，无法启动 RealSense 相机。")

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
        self._start_time = time.perf_counter()

        print(f"⏳ 正在初始化 RealSense 相机 [{self.name}]...")
        try:
            self.pipeline.start(self.config)
            self.valid = True
            print(f"✅ [成功] RealSense [{self.name}] 已启动: {self.width}x{self.height} @ {self.fps} FPS")
        except RuntimeError as e:
            print(f"⚠️ [警告] RealSense [{self.name}] 标准模式启动失败，尝试自动兼容模式: {e}")
            self.pipeline = rs.pipeline()
            fallback_config = rs.config()
            fallback_config.enable_stream(rs.stream.color)
            try:
                self.pipeline.start(fallback_config)
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

    def info(self) -> dict:
        with self.lock:
            fid = int(self.frame_id)
            ts = float(self.frame_timestamp)
            ok = bool(self.valid)
            err = self.error_msg
        age_ms = (time.perf_counter() - ts) * 1000.0 if ts > 0 else float("inf")
        elapsed = max(time.perf_counter() - self._start_time, 1e-6)
        fps = float(fid) / elapsed if fid > 0 else 0.0
        return {
            "name": self.name,
            "backend": "RealSense",
            "valid": ok,
            "frame_id": fid,
            "fps": fps,
            "age_ms": age_ms,
            "error": err,
            "exitcode": None,
            "alive": ok,
            "timestamp": ts,
        }

    def status(self) -> str:
        info = self.info()
        if not info["valid"]:
            return f"{info['name']}:backend={info['backend']},valid=False,fid={info['frame_id']},err={info['error'][:30]}"
        return (
            f"{info['name']}:backend={info['backend']},valid=True,fid={info['frame_id']},"
            f"fps={info['fps']:.1f},age={info['age_ms']:.0f}ms"
        )

    def save_debug_frame(self, save_dir: Path, step: int | str = "init") -> Path:
        save_dir.mkdir(parents=True, exist_ok=True)
        rgb = self.read_rgb()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        out_path = save_dir / f"{self.name}_step_{step}.jpg"
        cv2.imwrite(str(out_path), bgr)
        return out_path

    def release(self):
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.valid:
            try:
                self.pipeline.stop()
            except Exception as e:
                print(f"⚠️ [RealSense] 停止 pipeline 时出现异常: {e}")


class ThreadedYbImuReader:
    def __init__(self, port="/dev/ttyUSB1", report_rate=50, alpha=0.9, name="imu_left"):
        self.name = name
        self.port = port
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
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
            print(f"⚠️ [IMU:{self.name}] 初始化失败，将使用零 IMU: {exc}")

    def read(self) -> dict:
        if not self.valid or self.imu is None:
            return {
                "imu_left": np.zeros(IMU_DIM, dtype=np.float32),
                "quat": np.zeros(4, dtype=np.float32),
                "gyro": np.zeros(3, dtype=np.float32),
                "accel": np.zeros(3, dtype=np.float32),
                "frame_id": -1,
                "timestamp": time.perf_counter(),
                "valid": False,
            }
        try:
            quat = np.asarray(self.imu.get_imu_quaternion_data(), dtype=np.float32)
            gyro = np.asarray(self.imu.get_gyroscope_data(), dtype=np.float32)
            accel = np.asarray(self.imu.get_accelerometer_data(), dtype=np.float32)
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
                "frame_id": self.frame_id,
                "timestamp": self.timestamp,
                "valid": True,
            }
        except Exception as exc:
            self.error_msg = str(exc)
            self.valid = False
            return self.read()

    def info(self) -> dict:
        age_ms = (time.perf_counter() - self.timestamp) * 1000.0 if self.timestamp > 0 else float("inf")
        return {
            "name": self.name,
            "valid": bool(self.valid),
            "frame_id": int(self.frame_id),
            "age_ms": age_ms,
            "error": self.error_msg,
        }

    def status(self) -> str:
        info = self.info()
        if not info["valid"]:
            return f"{self.name}:valid=False,fid={info['frame_id']},err={str(info['error'])[:30]}"
        return f"{self.name}:valid=True,fid={info['frame_id']},age={info['age_ms']:.0f}ms"

    def release(self) -> None:
        if self.imu is None:
            return
        try:
            self.imu._dev.close()
        except Exception:
            pass


class ThreadedFlexiTacReader:
    ROWS = 16
    COLS = 32
    FRAME_BYTES = ROWS * COLS
    MAGIC = b"\xAA\x55"
    ROW_SLICE = slice(-TACTILE_ROWS, None)
    COL_SLICE = slice(1, -1)

    def __init__(
        self,
        port="/dev/ttyUSB2",
        baud=2_000_000,
        init_frames=30,
        threshold=20.0,
        noise_scale=60.0,
        alpha=1.0,
        name="right_tactile",
    ):
        self.name = name
        self.port = port
        self.baud = int(baud)
        self.init_frames = max(int(init_frames), 1)
        self.threshold = float(threshold)
        self.noise_scale = max(float(noise_scale), 1e-6)
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.valid = False
        self.running = False
        self.error_msg = ""
        self.frame_id = 0
        self.timestamp = 0.0
        self.lock = threading.Lock()
        self.latest_tactile = np.zeros((TACTILE_ROWS, TACTILE_COLS), dtype=np.float32)
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
            print(f"⚠️ [FlexiTac:{self.name}] 初始化失败，将使用零触觉: {exc}")

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
        print(f"⏳ [FlexiTac:{self.name}] 初始化 baseline，请保持右侧触觉阵列无接触...")
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
            tactile = self._normalize(contact[self.ROW_SLICE, self.COL_SLICE])
            if self._filtered is None or self.alpha >= 1.0:
                filtered = tactile
            else:
                filtered = self.alpha * tactile + (1.0 - self.alpha) * self._filtered
            self._filtered = filtered.astype(np.float32, copy=True)
            with self.lock:
                self.latest_tactile = filtered.astype(np.float32, copy=True)
                self.frame_id += 1
                self.timestamp = time.perf_counter()

    def read(self) -> tuple[np.ndarray, int, float]:
        with self.lock:
            return self.latest_tactile.copy(), int(self.frame_id), float(self.timestamp)

    def info(self) -> dict:
        with self.lock:
            tactile = self.latest_tactile.copy()
            fid = int(self.frame_id)
            ts = float(self.timestamp)
        age_ms = (time.perf_counter() - ts) * 1000.0 if ts > 0 else float("inf")
        return {
            "name": self.name,
            "valid": bool(self.valid),
            "frame_id": fid,
            "age_ms": age_ms,
            "max": float(np.max(tactile)),
            "mean": float(np.mean(tactile)),
            "sum": float(np.sum(tactile)),
            "error": self.error_msg,
        }

    def status(self) -> str:
        info = self.info()
        if not info["valid"]:
            return f"{self.name}:valid=False,fid={info['frame_id']},err={str(info['error'])[:30]}"
        return (
            f"{self.name}:valid=True,fid={info['frame_id']},age={info['age_ms']:.0f}ms,"
            f"max={info['max']:.3f},sum={info['sum']:.2f}"
        )

    def release(self) -> None:
        self.running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass


class AsyncPolicyRunner:
    def __init__(self, policy):
        self.policy = policy
        self.cond = threading.Condition()
        self.pending: tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self.results: deque[dict] = deque(maxlen=16)
        self.running = False
        self.thread: threading.Thread | None = None
        self.last_infer_ms = 0.0
        self.last_origin_step = -1
        self.error: BaseException | None = None

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        with self.cond:
            self.running = False
            self.cond.notify_all()
        if self.thread is not None:
            self.thread.join(timeout=1.0)

    def submit(
        self,
        origin_step: int,
        qpos_np: np.ndarray,
        image_np: np.ndarray,
        imu_np: np.ndarray,
        tactile_np: np.ndarray,
    ) -> None:
        if not self.running:
            return
        with self.cond:
            self.pending = (
                int(origin_step),
                np.asarray(qpos_np, dtype=np.float32).copy(),
                np.asarray(image_np, dtype=np.uint8).copy(),
                np.asarray(imu_np, dtype=np.float32).copy(),
                np.asarray(tactile_np, dtype=np.float32).copy(),
            )
            self.cond.notify()

    def pop_results(self) -> list[dict]:
        with self.cond:
            results = list(self.results)
            self.results.clear()
            return results

    def _run(self) -> None:
        while True:
            with self.cond:
                while self.running and self.pending is None:
                    self.cond.wait(timeout=0.05)
                if not self.running:
                    return
                job = self.pending
                self.pending = None

            if job is None:
                continue

            origin_step, qpos_np, image_np, imu_np, tactile_np = job
            try:
                with torch.inference_mode():
                    image = torch.from_numpy(image_np).float().div_(255.0).permute(0, 3, 1, 2).cuda().unsqueeze(0)
                    qpos = torch.from_numpy(qpos_np).float().cuda().unsqueeze(0)
                    imu = torch.from_numpy(imu_np).float().cuda().unsqueeze(0)
                    tactile = torch.from_numpy(tactile_np).float().cuda().unsqueeze(0)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    chunk = self.policy(qpos, image, imu, tactile)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    infer_ms = (time.perf_counter() - t0) * 1000.0
                    chunk_np = chunk.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=True)

                with self.cond:
                    self.last_infer_ms = infer_ms
                    self.last_origin_step = origin_step
                    self.results.append(
                        {
                            "origin_step": origin_step,
                            "chunk": chunk_np,
                            "infer_ms": infer_ms,
                        }
                    )
            except BaseException as exc:
                with self.cond:
                    self.error = exc
                    self.running = False
                return


class LatestVisualizer:
    """
    单窗口实时可视化：
    1) 相机画面 + 运行状态
    2) 关节反馈 / 命令 / 策略动作曲线
    3) 新增 IMU 曲线 + 触觉热力图 + 触觉统计曲线

    设计原则：
    - 只在可视化线程中绘图，不阻塞主控制循环。
    - update() 只拷贝最新快照，历史缓存由可视化刷新频率决定。
    - IMU/触觉使用原始读数显示，策略仍使用归一化后的输入推理。
    """

    def __init__(
        self,
        enabled: bool = True,
        fps: float = 10.0,
        scale: float = 0.5,
        history_seconds: float = 8.0,
        window_name: str = "rebot deploy",
        show_sensor_panel: bool = True,
        sensor_panel_height: int = 360,
    ):
        self.enabled = bool(enabled)
        self.period = 1.0 / max(float(fps), 1e-6)
        self.scale = float(scale)
        self.history_maxlen = max(20, int(max(float(history_seconds), 1.0) * max(float(fps), 1.0)))
        self.window_name = window_name
        self.show_sensor_panel = bool(show_sensor_panel)
        self.sensor_panel_height = int(np.clip(sensor_panel_height, 240, 720))

        self.lock = threading.Lock()
        self.latest: dict | None = None
        self.history = deque(maxlen=self.history_maxlen)
        self.sensor_history = deque(maxlen=self.history_maxlen)
        self.running = False
        self.thread: threading.Thread | None = None
        self._tactile_gamma_lut = self._build_gamma_lut(0.82)
        self._vignette_cache: dict[tuple[int, int], np.ndarray] = {}

    def start(self) -> None:
        if not self.enabled or self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=0.5)
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass

    def update(self, **kwargs) -> None:
        if not self.enabled:
            return
        # 尽量只做轻量拷贝，避免在主控制循环中进行绘图/resize/滤波。
        with self.lock:
            self.latest = kwargs

    def _snapshot(self) -> dict | None:
        with self.lock:
            return self.latest

    def _run(self) -> None:
        global _running
        while self.running and _running:
            start = time.perf_counter()
            snapshot = self._snapshot()
            if snapshot is not None:
                try:
                    cv2.imshow(self.window_name, self._render(snapshot))
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        print("\n[visualize] 收到窗口退出命令，准备安全关闭...")
                        _running = False
                        self.running = False
                        break
                except Exception as exc:
                    print(f"\n⚠️ 可视化窗口异常，已关闭显示线程: {exc}")
                    self.running = False
                    break
            sleep_time = self.period - (time.perf_counter() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _render(self, snapshot: dict) -> np.ndarray:
        self._append_history(snapshot)
        frames = snapshot.get("frames") or []
        camera_names = snapshot.get("camera_names") or []
        camera_infos = snapshot.get("camera_infos") or []
        image_panel = self._make_image_panel(frames, camera_names, camera_infos)
        status_panel = self._make_status_panel(snapshot, camera_infos, height=image_panel.shape[0])
        top_panel = np.hstack([image_panel, status_panel])
        curve_panel = self._make_curve_panel(width=top_panel.shape[1], height=300)
        if not self.show_sensor_panel:
            return np.vstack([top_panel, curve_panel])
        sensor_panel = self._make_sensor_panel(snapshot, width=top_panel.shape[1], height=self.sensor_panel_height)
        return np.vstack([top_panel, curve_panel, sensor_panel])

    def _append_history(self, snapshot: dict) -> None:
        now = time.perf_counter()

        q_feedback = snapshot.get("q_feedback")
        target_q = snapshot.get("target_q")
        action = snapshot.get("action")
        if q_feedback is not None:
            self.history.append(
                {
                    "t": now,
                    "q_feedback": np.asarray(q_feedback, dtype=np.float64).reshape(-1)[:6].copy(),
                    "target_q": None if target_q is None else np.asarray(target_q, dtype=np.float64).reshape(-1)[:6].copy(),
                    "action": None if action is None else np.asarray(action, dtype=np.float64).reshape(-1)[:6].copy(),
                }
            )

        imu_sample = snapshot.get("imu_sample") or {}
        tactile_frame = snapshot.get("tactile_frame")
        if imu_sample or tactile_frame is not None:
            quat = self._safe_vec(imu_sample.get("quat", None), 4)
            gyro = self._safe_vec(imu_sample.get("gyro", None), 3)
            accel = self._safe_vec(imu_sample.get("accel", None), 3)
            tactile_stats = self._tactile_stats(tactile_frame)
            self.sensor_history.append(
                {
                    "t": now,
                    "quat": quat,
                    "gyro": gyro,
                    "accel": accel,
                    **tactile_stats,
                }
            )

    def _safe_vec(self, value, size: int) -> np.ndarray:
        if value is None:
            return np.zeros(size, dtype=np.float64)
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        out = np.zeros(size, dtype=np.float64)
        n = min(size, arr.size)
        if n > 0:
            out[:n] = np.nan_to_num(arr[:n], nan=0.0, posinf=0.0, neginf=0.0)
        return out

    def _tactile_stats(self, tactile_frame) -> dict:
        if tactile_frame is None:
            return {
                "tactile_max": 0.0,
                "tactile_mean": 0.0,
                "tactile_sum": 0.0,
                "tactile_contact_ratio": 0.0,
                "tactile_cx": 0.5,
                "tactile_cy": 0.5,
            }
        tactile = np.asarray(tactile_frame, dtype=np.float32)
        if tactile.ndim == 3:
            tactile = tactile[0]
        tactile = np.nan_to_num(tactile, nan=0.0, posinf=0.0, neginf=0.0)
        if tactile.size == 0:
            return {
                "tactile_max": 0.0,
                "tactile_mean": 0.0,
                "tactile_sum": 0.0,
                "tactile_contact_ratio": 0.0,
                "tactile_cx": 0.5,
                "tactile_cy": 0.5,
            }
        max_val = float(np.max(tactile))
        mean_val = float(np.mean(tactile))
        sum_val = float(np.sum(tactile))
        contact_ratio = float(np.mean(tactile > 0.05))
        if sum_val > 1e-6 and tactile.ndim == 2:
            h, w = tactile.shape
            yy, xx = np.mgrid[0:h, 0:w]
            cx = float(np.sum(xx * tactile) / sum_val / max(w - 1, 1))
            cy = float(np.sum(yy * tactile) / sum_val / max(h - 1, 1))
        else:
            cx, cy = 0.5, 0.5
        return {
            "tactile_max": max_val,
            "tactile_mean": mean_val,
            "tactile_sum": sum_val,
            "tactile_contact_ratio": contact_ratio,
            "tactile_cx": cx,
            "tactile_cy": cy,
        }

    def _make_image_panel(self, frames: list[np.ndarray], camera_names: list[str], camera_infos: list[dict]) -> np.ndarray:
        if not frames:
            return np.zeros((360, 480, 3), dtype=np.uint8)

        bgr_frames = []
        for i, rgb in enumerate(frames):
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if self.scale > 0 and self.scale != 1.0:
                frame = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
            info = camera_infos[i] if i < len(camera_infos) else {}
            label = camera_names[i] if i < len(camera_names) else f"cam{i}"
            fid = int(info.get("frame_id", -1))
            fps = float(info.get("fps", 0.0))
            age_ms = float(info.get("age_ms", float("inf")))
            backend = str(info.get("backend", ""))
            valid = bool(info.get("valid", True))
            overlay_h = 68
            overlay_w = min(frame.shape[1] - 12, 320)
            cv2.rectangle(frame, (6, 6), (6 + overlay_w, 6 + overlay_h), (0, 0, 0), -1)
            cv2.putText(frame, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"{backend} fid={fid} fps={fps:.1f}",
                (14, 49),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"age={age_ms:.0f}ms valid={valid}",
                (14, 66),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            bgr_frames.append(frame)

        target_h = max(frame.shape[0] for frame in bgr_frames)
        padded = []
        for frame in bgr_frames:
            if frame.shape[0] < target_h:
                pad = np.zeros((target_h - frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
                frame = np.vstack([frame, pad])
            padded.append(frame)
        return np.hstack(padded)

    def _make_status_panel(self, snapshot: dict, camera_infos: list[dict], height: int) -> np.ndarray:
        panel_w = 520
        panel = np.full((height, panel_w, 3), 24, dtype=np.uint8)
        lines = self._status_lines(snapshot)
        if camera_infos:
            lines.append(("", (255, 255, 255)))
            for info in camera_infos:
                lines.append((self._camera_info_line(info), (200, 230, 255)))
        y = 30
        for text, color in lines:
            cv2.putText(panel, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
            y += 24
            if y > height - 12:
                break

        # 保留原状态面板的小型触觉热力图；新增的大型触觉图在 sensor panel 中。
        tactile_frame = snapshot.get("tactile_frame")
        if tactile_frame is not None and y < height - 80:
            heatmap = self._make_tactile_heatmap(tactile_frame, width=panel_w - 28, height=min(150, height - y - 14))
            panel[y : y + heatmap.shape[0], 14 : 14 + heatmap.shape[1]] = heatmap
        return panel

    def _make_sensor_panel(self, snapshot: dict, width: int, height: int) -> np.ndarray:
        panel = np.full((height, width, 3), 16, dtype=np.uint8)
        cv2.putText(
            panel,
            "sensor dashboard: IMU raw curves + tactile heatmap/statistics",
            (14, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

        left_w = max(520, int(width * 0.56))
        right_x = left_w + 10
        right_w = max(width - right_x - 14, 320)
        top = 44
        bottom = height - 12
        cv2.line(panel, (left_w, top - 8), (left_w, bottom), (48, 48, 48), 1)

        hist = list(self.sensor_history)
        if len(hist) < 2:
            cv2.putText(panel, "waiting for IMU/tactile history...", (18, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (160, 160, 160), 1, cv2.LINE_AA)
        else:
            imu_sample = snapshot.get("imu_sample") or {}
            gyro_now = self._safe_vec(imu_sample.get("gyro", None), 3)
            accel_now = self._safe_vec(imu_sample.get("accel", None), 3)
            quat_now = self._safe_vec(imu_sample.get("quat", None), 4)

            left = 54
            right = left_w - 18
            row_h = max(70, (bottom - top - 6) // 3)
            self._draw_vector_group(
                panel,
                hist,
                vector_key="gyro",
                labels=("gx", "gy", "gz"),
                title=f"IMU gyro xyz  now [{gyro_now[0]:+.2f}, {gyro_now[1]:+.2f}, {gyro_now[2]:+.2f}]",
                rect=(left, top, right, min(top + row_h - 8, bottom)),
                colors=((80, 220, 255), (120, 255, 140), (120, 160, 255)),
            )
            y2 = top + row_h
            self._draw_vector_group(
                panel,
                hist,
                vector_key="accel",
                labels=("ax", "ay", "az"),
                title=f"IMU accel xyz now [{accel_now[0]:+.2f}, {accel_now[1]:+.2f}, {accel_now[2]:+.2f}]",
                rect=(left, y2, right, min(y2 + row_h - 8, bottom)),
                colors=((255, 210, 90), (130, 230, 255), (255, 130, 130)),
            )
            y3 = top + row_h * 2
            self._draw_vector_group(
                panel,
                hist,
                vector_key="quat",
                labels=("qw", "qx", "qy", "qz"),
                title=f"IMU quat wxyz now [{quat_now[0]:+.2f}, {quat_now[1]:+.2f}, {quat_now[2]:+.2f}, {quat_now[3]:+.2f}]",
                rect=(left, y3, right, min(y3 + row_h - 8, bottom)),
                colors=((240, 240, 240), (80, 220, 255), (120, 255, 140), (120, 160, 255)),
                fixed_range=(-1.05, 1.05),
            )

        tactile_frame = snapshot.get("tactile_frame")
        tactile_info = snapshot.get("tactile_info") or {}
        heat_h = int((bottom - top) * 0.58)
        heat_h = max(150, min(heat_h, bottom - top - 110))
        heatmap = self._make_tactile_heatmap(tactile_frame, width=right_w, height=heat_h)
        panel[top : top + heat_h, right_x : right_x + right_w] = heatmap

        tactile_stats = self._tactile_stats(tactile_frame)
        status_y = top + heat_h + 24
        cv2.putText(
            panel,
            f"tactile fid={int(tactile_info.get('frame_id', -1))} age={float(tactile_info.get('age_ms', float('inf'))):.0f}ms "
            f"max={tactile_stats['tactile_max']:.3f} mean={tactile_stats['tactile_mean']:.3f} "
            f"contact={tactile_stats['tactile_contact_ratio'] * 100:.1f}%",
            (right_x, status_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        curve_top = status_y + 12
        curve_bottom = bottom
        if curve_bottom - curve_top > 50 and len(hist) >= 2:
            self._draw_scalar_group(
                panel,
                hist,
                keys=("tactile_max", "tactile_mean", "tactile_contact_ratio"),
                labels=("max", "mean", "contact"),
                title="tactile max / mean / contact-ratio",
                rect=(right_x + 36, curve_top, right_x + right_w - 8, curve_bottom),
                colors=((255, 210, 90), (80, 220, 255), (120, 255, 140)),
                fixed_range=(0.0, 1.0),
            )
        return panel

    def _build_gamma_lut(self, gamma: float) -> np.ndarray:
        lut = np.arange(256, dtype=np.float32) / 255.0
        lut = np.power(lut, gamma)
        return np.clip(lut * 255.0, 0, 255).astype(np.uint8)

    def _build_vignette_mask(self, width: int, height: int, strength: float = 0.22) -> np.ndarray:
        cache_key = (int(width), int(height))
        cached = self._vignette_cache.get(cache_key)
        if cached is not None:
            return cached
        y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
        x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        radius = np.sqrt(xx * xx + yy * yy)
        radius = np.clip(radius, 0.0, 1.0)
        mask = 1.0 - strength * np.power(radius, 1.7)
        mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
        self._vignette_cache[cache_key] = mask
        return mask

    def _soft_low_cut(self, frame: np.ndarray, low_cut: float = 0.025) -> np.ndarray:
        out = (frame - low_cut) / max(1e-6, 1.0 - low_cut)
        np.clip(out, 0.0, 1.0, out=out)
        return out.astype(np.float32, copy=False)

    def _make_tactile_heatmap(self, tactile_frame, width: int, height: int) -> np.ndarray:
        if tactile_frame is None:
            tactile = np.zeros((TACTILE_ROWS, TACTILE_COLS), dtype=np.float32)
        else:
            tactile = np.asarray(tactile_frame, dtype=np.float32)
            if tactile.ndim == 3:
                tactile = tactile[0]

        tactile = np.nan_to_num(tactile, nan=0.0, posinf=0.0, neginf=0.0)
        max_val = float(np.max(tactile)) if tactile.size else 0.0
        force_max = max(max_val, 1e-6)
        base = np.clip(tactile, 0.0, force_max) / force_max
        base = self._soft_low_cut(base, low_cut=0.025)
        base *= 1.12
        np.clip(base, 0.0, 1.0, out=base)

        up = cv2.resize(base, (width, height), interpolation=cv2.INTER_CUBIC)
        up = np.clip(up, 0.0, 1.0).astype(np.float32, copy=False)

        smooth = cv2.GaussianBlur(up, (0, 0), 1.15)
        glow = cv2.GaussianBlur(smooth, (0, 0), 4.6)
        mixed = smooth + 0.48 * glow
        np.clip(mixed, 0.0, 1.0, out=mixed)

        gray = np.clip(mixed * 255.0, 0, 255).astype(np.uint8)
        gray = cv2.LUT(gray, self._tactile_gamma_lut)
        colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)

        intensity = gray.astype(np.float32) / 255.0
        alpha = np.power(np.clip(intensity, 0.0, 1.0), 0.72)
        colored_float = colored.astype(np.float32) * alpha[..., None]
        vignette = self._build_vignette_mask(width, height, strength=0.22)
        colored_float *= vignette[..., None]
        colored_out = np.clip(colored_float, 0, 255).astype(np.uint8)

        stats = self._tactile_stats(tactile)
        cx = int(np.clip(stats["tactile_cx"], 0.0, 1.0) * (width - 1))
        cy = int(np.clip(stats["tactile_cy"], 0.0, 1.0) * (height - 1))
        if stats["tactile_sum"] > 1e-6:
            cv2.drawMarker(colored_out, (cx, cy), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=1, line_type=cv2.LINE_AA)

        cv2.putText(colored_out, "right tactile 12x30", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(
            colored_out,
            f"max={max_val:.3f} sum={float(np.sum(tactile)):.2f} cop=({stats['tactile_cx']:.2f},{stats['tactile_cy']:.2f})",
            (10, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        return colored_out

    def _camera_info_line(self, info: dict) -> str:
        name = info.get("name", "cam")
        backend = info.get("backend", "")
        fid = int(info.get("frame_id", -1))
        fps = float(info.get("fps", 0.0))
        age_ms = float(info.get("age_ms", float("inf")))
        valid = bool(info.get("valid", True))
        return f"{name}: {backend} fid={fid} fps={fps:.1f} age={age_ms:.0f}ms valid={valid}"

    def _make_curve_panel(self, width: int, height: int) -> np.ndarray:
        panel = np.full((height, width, 3), 18, dtype=np.uint8)
        cv2.putText(
            panel,
            "joint curves: q_feedback=white  q_cmd=cyan  policy_action=orange",
            (14, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
        hist = list(self.history)
        if len(hist) < 2:
            cv2.putText(panel, "waiting for joint history...", (14, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)
            return panel

        left = 54
        right = width - 16
        top = 42
        row_h = max(32, (height - top - 10) // 6)
        xs = np.linspace(left, right, len(hist)).astype(np.int32)

        q_fb = np.stack([item["q_feedback"] for item in hist], axis=0)
        q_cmd = self._stack_optional_history(hist, "target_q")
        action = self._stack_optional_history(hist, "action")

        for joint_idx in range(6):
            y0 = top + joint_idx * row_h
            y1 = min(y0 + row_h - 6, height - 10)
            if y1 <= y0 + 4:
                break

            values = [q_fb[:, joint_idx]]
            if q_cmd is not None:
                values.append(q_cmd[:, joint_idx])
            if action is not None:
                values.append(action[:, joint_idx])
            all_values = np.concatenate(values)
            vmin = float(np.nanmin(all_values))
            vmax = float(np.nanmax(all_values))
            pad = max((vmax - vmin) * 0.15, 0.05)
            vmin -= pad
            vmax += pad

            cv2.rectangle(panel, (left, y0), (right, y1), (48, 48, 48), 1)
            mid_y = (y0 + y1) // 2
            cv2.line(panel, (left, mid_y), (right, mid_y), (34, 34, 34), 1)
            cv2.putText(panel, f"J{joint_idx + 1}", (14, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(panel, f"{q_fb[-1, joint_idx]:+.2f}", (14, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

            self._draw_curve(panel, xs, q_fb[:, joint_idx], vmin, vmax, y0, y1, (235, 235, 235), thickness=2)
            if q_cmd is not None:
                self._draw_curve(panel, xs, q_cmd[:, joint_idx], vmin, vmax, y0, y1, (80, 220, 255), thickness=1)
            if action is not None:
                self._draw_curve(panel, xs, action[:, joint_idx], vmin, vmax, y0, y1, (70, 160, 255), thickness=1)

        cv2.putText(panel, f"history {len(hist)} samples", (width - 170, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
        return panel

    def _stack_optional_history(self, hist: list[dict], key: str) -> np.ndarray | None:
        values = [item[key] for item in hist]
        if any(value is None for value in values):
            return None
        return np.stack(values, axis=0)

    def _draw_vector_group(
        self,
        panel: np.ndarray,
        hist: list[dict],
        vector_key: str,
        labels: tuple[str, ...],
        title: str,
        rect: tuple[int, int, int, int],
        colors: tuple[tuple[int, int, int], ...],
        fixed_range: tuple[float, float] | None = None,
    ) -> None:
        left, top, right, bottom = rect
        if bottom <= top + 8 or right <= left + 8:
            return
        cv2.rectangle(panel, (left, top), (right, bottom), (50, 50, 50), 1)
        cv2.putText(panel, title, (left + 6, top + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        values = np.stack([np.asarray(item[vector_key], dtype=np.float64).reshape(-1)[: len(labels)] for item in hist], axis=0)
        if fixed_range is None:
            vmin = float(np.nanmin(values))
            vmax = float(np.nanmax(values))
            pad = max((vmax - vmin) * 0.15, 0.05)
            vmin -= pad
            vmax += pad
        else:
            vmin, vmax = fixed_range
        xs = np.linspace(left + 6, right - 6, values.shape[0]).astype(np.int32)
        zero_y = self._value_to_y(0.0, vmin, vmax, top + 24, bottom - 8)
        cv2.line(panel, (left + 6, zero_y), (right - 6, zero_y), (38, 38, 38), 1)
        for i, label in enumerate(labels):
            color = colors[i % len(colors)]
            self._draw_curve(panel, xs, values[:, i], vmin, vmax, top + 24, bottom - 8, color, thickness=1)
            cv2.putText(panel, label, (right - 44, top + 18 + i * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    def _draw_scalar_group(
        self,
        panel: np.ndarray,
        hist: list[dict],
        keys: tuple[str, ...],
        labels: tuple[str, ...],
        title: str,
        rect: tuple[int, int, int, int],
        colors: tuple[tuple[int, int, int], ...],
        fixed_range: tuple[float, float] | None = None,
    ) -> None:
        left, top, right, bottom = rect
        if bottom <= top + 8 or right <= left + 8:
            return
        cv2.rectangle(panel, (left, top), (right, bottom), (50, 50, 50), 1)
        cv2.putText(panel, title, (left + 6, top + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        values = np.stack([[float(item.get(k, 0.0)) for k in keys] for item in hist], axis=0)
        if fixed_range is None:
            vmin = float(np.nanmin(values))
            vmax = float(np.nanmax(values))
            pad = max((vmax - vmin) * 0.15, 0.02)
            vmin -= pad
            vmax += pad
        else:
            vmin, vmax = fixed_range
        xs = np.linspace(left + 6, right - 6, values.shape[0]).astype(np.int32)
        for i, label in enumerate(labels):
            color = colors[i % len(colors)]
            self._draw_curve(panel, xs, values[:, i], vmin, vmax, top + 24, bottom - 8, color, thickness=1)
            cv2.putText(panel, label, (right - 80, top + 18 + i * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    def _value_to_y(self, value: float, vmin: float, vmax: float, y0: int, y1: int) -> int:
        denom = max(vmax - vmin, 1e-6)
        return int(y1 - ((float(value) - vmin) / denom * (y1 - y0)))

    def _draw_curve(
        self,
        panel: np.ndarray,
        xs: np.ndarray,
        values: np.ndarray,
        vmin: float,
        vmax: float,
        y0: int,
        y1: int,
        color: tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        denom = max(vmax - vmin, 1e-6)
        values = np.asarray(values, dtype=np.float64)
        values = np.nan_to_num(values, nan=0.0, posinf=vmax, neginf=vmin)
        ys = y1 - ((values - vmin) / denom * (y1 - y0)).astype(np.int32)
        ys = np.clip(ys, min(y0, y1), max(y0, y1))
        pts = np.column_stack([xs, ys]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(panel, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    def _status_lines(self, snapshot: dict) -> list[tuple[str, tuple[int, int, int]]]:
        def fmt_vec(name: str, value, precision: int = 3) -> list[str]:
            if value is None:
                return [f"{name}: None"]
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
            parts = [f"{x:+.{precision}f}" for x in arr]
            return [f"{name}: [{', '.join(parts[:3])}]", f"      [{', '.join(parts[3:6])}]"]

        actual_hz = snapshot.get("actual_hz", 0.0)
        lines: list[tuple[str, tuple[int, int, int]]] = [
            (
                f"step {snapshot.get('step', 0)} | actual {actual_hz:.1f} Hz | "
                f"dt {snapshot.get('dt_ms', 0.0):.1f} ms | async_age {snapshot.get('async_age', -1)}",
                (240, 240, 240),
            ),
            (
                f"read {snapshot.get('read_ms', 0.0):.1f}  cam {snapshot.get('camera_ms', 0.0):.1f}  "
                f"infer {snapshot.get('infer_ms', 0.0):.1f}  cmd {snapshot.get('command_ms', 0.0):.1f} ms",
                (180, 220, 255),
            ),
            (f"max_err {snapshot.get('arm_err_max', 0.0):.4f} rad", (120, 220, 120)),
            ("", (255, 255, 255)),
        ]

        for line in fmt_vec("q_fb ", snapshot.get("q_feedback")):
            lines.append((line, (230, 230, 230)))
        for line in fmt_vec("q_cmd", snapshot.get("target_q")):
            lines.append((line, (80, 220, 255)))
        for line in fmt_vec("action", snapshot.get("action")):
            lines.append((line, (120, 180, 255)))

        gripper_fb = snapshot.get("gripper_fb")
        gripper_fb_str = "None" if gripper_fb is None else f"{float(gripper_fb):+.3f}"
        imu_info = snapshot.get("imu_info") or {}
        tactile_info = snapshot.get("tactile_info") or {}
        imu_sample = snapshot.get("imu_sample") or {}
        gyro = self._safe_vec(imu_sample.get("gyro", None), 3)
        accel = self._safe_vec(imu_sample.get("accel", None), 3)
        tactile_stats = self._tactile_stats(snapshot.get("tactile_frame"))
        lines.extend(
            [
                ("", (255, 255, 255)),
                (f"gripper cmd {snapshot.get('gripper_cmd', 0.0):+.3f} | fb {gripper_fb_str}", (180, 220, 255)),
                (
                    f"imu fid={int(imu_info.get('frame_id', -1))} "
                    f"age={float(imu_info.get('age_ms', float('inf'))):.0f}ms "
                    f"valid={bool(imu_info.get('valid', False))}",
                    (200, 230, 255),
                ),
                (
                    f"gyro=[{gyro[0]:+.2f},{gyro[1]:+.2f},{gyro[2]:+.2f}] "
                    f"acc=[{accel[0]:+.2f},{accel[1]:+.2f},{accel[2]:+.2f}]",
                    (200, 230, 255),
                ),
                (
                    f"tactile fid={int(tactile_info.get('frame_id', -1))} "
                    f"age={float(tactile_info.get('age_ms', float('inf'))):.0f}ms "
                    f"max={tactile_stats['tactile_max']:.3f} sum={tactile_stats['tactile_sum']:.2f}",
                    (200, 230, 255),
                ),
                ("press q or ESC to stop", (160, 160, 160)),
            ]
        )
        return lines

def parse_camera_srcs(value: str | None) -> dict[str, int | str]:
    if not value:
        return {}
    result = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        name, src = item.split("=", 1)
        src_value = src.strip()
        result[name.strip()] = int(src_value) if src_value.isdigit() else src_value.lower()
    return result


def read_action_stats(stats, action):
    return action * stats["action_std"] + stats["action_mean"]


def infer_task_config(task_name: str, camera_names_override: list[str] | None):
    task_configs = _load_task_configs()
    task_config = dict(task_configs.get(task_name, {}))
    if camera_names_override is not None:
        task_config["camera_names"] = camera_names_override
    if "camera_names" not in task_config or not task_config["camera_names"]:
        task_config["camera_names"] = ["cam_high", "cam_wrist"]
    return task_config


def _make_camera_backend(cam_name: str, source: int | str):
    source_token = str(source).strip().lower() if not isinstance(source, int) else str(source)

    if cam_name == "cam_high":
        if isinstance(source, int) or source_token in {"astra", "astra_s", "openni", "shm"}:
            return ThreadedAstraSCamera(name=cam_name)
        print(f"⚠️ [camera] {cam_name} 收到旧的源配置 {source!r}，已强制切换为 Astra-S helper。")
        return ThreadedAstraSCamera(name=cam_name)

    if cam_name == "cam_wrist":
        if isinstance(source, int) or source_token in {"d405", "realsense", "rs", "realsense_d405"}:
            return ThreadedRealSenseCamera(name=cam_name)
        print(f"⚠️ [camera] {cam_name} 收到旧的源配置 {source!r}，已强制切换为 RealSense D405。")
        return ThreadedRealSenseCamera(name=cam_name)

    if isinstance(source, int):
        return ThreadedCamera(src=source, name=cam_name)
    if source_token in {"realsense", "rs", "d405", "realsense_d405"}:
        return ThreadedRealSenseCamera(name=cam_name)
    if source_token in {"astra", "astra_s", "openni", "shm"}:
        return ThreadedAstraSCamera(name=cam_name)

    raise ValueError(
        f"不支持的相机源配置: {cam_name}={source}. "
        "整数会走 OpenCV VideoCapture，astra/d405 会走专用后端。"
    )


def main():
    parser = argparse.ArgumentParser(description="rebot policy deployment")
    parser.add_argument("--ckpt_dir", required=True, type=str)
    parser.add_argument("--task_name", default="rebot_real_test", type=str)
    parser.add_argument("--policy_class", default="ACT", choices=["ACT", "CNNMLP"])
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument(
        "--act-root",
        type=Path,
        default=_default_act_root(),
        help="包含触觉版 policy.py 和 detr/ 的 ACT 源码目录；也可设置 ACT_TACTILE_ROOT。",
    )

    parser.add_argument("--xml", type=Path, default=_default_xml_path())
    parser.add_argument("--cfg", type=Path, default=None)
    parser.add_argument("--gripper-cfg", type=Path, default=_default_gripper_cfg_path())
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--read-rate", type=float, default=60.0)
    parser.add_argument("--vlim", type=float, nargs=6, default=None)
    parser.add_argument("--max-step", type=float, nargs=6, default=None)
    parser.add_argument("--max-tracking-error", type=float, default=1.0)
    parser.add_argument("--breach-samples", type=int, default=20)
    parser.add_argument("--gripper-kp", type=float, default=1.0)
    parser.add_argument("--gripper-kd", type=float, default=0.05)
    parser.add_argument("--gripper-tau", type=float, default=0.0)
    parser.add_argument("--gripper-send-every", type=int, default=1)
    parser.add_argument("--gripper-default", type=float, default=-5.8)
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--no-final-disable", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--temporal-agg", action="store_true")
    parser.add_argument("--temporal-agg-k", type=float, default=0.2, help="时间集合动作平滑指数衰减系数，越大越跟随最新预测")
    parser.add_argument("--temporal-agg-max-history", type=int, default=10, help="时间集合动作平滑最多融合最近 N 个预测，<=0 表示不限制")
    parser.add_argument("--sync-inference", action="store_true", help="关闭后台异步推理，恢复主循环同步推理")
    parser.add_argument("--async-submit-every", type=int, default=1, help="异步推理每隔 N 个控制步提交一次最新观测")
    parser.add_argument("--camera-src", type=str, default=None, help="例如 cam_high=2, cam_high=astra,cam_wrist=d405")
    parser.add_argument("--camera-names", type=str, default=None, help="例如 cam_high,cam_wrist")
    parser.add_argument("--camera-debug", action="store_true", help="启动时打印相机状态并保存一张调试图")
    parser.add_argument("--camera-save-dir", type=Path, default=Path("/tmp/rebot_cam_debug"))
    parser.add_argument("--camera-save-every", type=int, default=0, help=">0 时每隔 N 步保存一次相机图像")
    parser.add_argument("--no-visualize", action="store_true", help="关闭实时相机和关节状态窗口")
    parser.add_argument("--visualize-fps", type=float, default=10.0, help="实时窗口刷新率，建议 10-15Hz")
    parser.add_argument("--visualize-scale", type=float, default=0.5, help="相机画面显示缩放比例")
    parser.add_argument("--visualize-history-sec", type=float, default=8.0, help="关节曲线显示最近多少秒的低频历史")
    parser.add_argument("--no-sensor-visualize", action="store_true", help="关闭新增 IMU/触觉传感器可视化面板")
    parser.add_argument("--sensor-panel-height", type=int, default=360, help="新增 IMU/触觉传感器面板高度，建议 320-420")
    parser.add_argument("--no-imu", action="store_true", help="关闭实时 IMU，使用零 IMU 输入")
    parser.add_argument("--imu-port", type=str, default="/dev/ttyUSB1", help="Yb IMU 串口")
    parser.add_argument("--imu-report-rate", type=int, default=50)
    parser.add_argument("--imu-alpha", type=float, default=0.9)
    parser.add_argument("--no-tactile", action="store_true", help="关闭实时 FlexiTac，使用零触觉输入")
    parser.add_argument("--tactile-port", type=str, default="/dev/ttyUSB2", help="FlexiTac 右侧触觉串口")
    parser.add_argument("--tactile-baud", type=int, default=2_000_000)
    parser.add_argument("--tactile-init-frames", type=int, default=30)
    parser.add_argument("--tactile-threshold", type=float, default=20.0)
    parser.add_argument("--tactile-noise-scale", type=float, default=60.0)
    parser.add_argument("--tactile-alpha", type=float, default=1.0)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--enc_layers", type=int, default=4)
    parser.add_argument("--dec_layers", type=int, default=7)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--kl_weight", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--dim_feedforward", type=int, default=None)

    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"ckpt_dir 不存在: {ckpt_dir}")

    task_config = infer_task_config(
        args.task_name,
        args.camera_names.split(",") if args.camera_names else None,
    )
    camera_names = task_config["camera_names"]
    camera_sources = parse_camera_srcs(args.camera_src)
    for cam_name in camera_names:
        if cam_name == "cam_high":
            camera_sources.setdefault(cam_name, "astra")
        elif cam_name == "cam_wrist":
            camera_sources.setdefault(cam_name, "d405")
        else:
            camera_sources.setdefault(cam_name, 0)

    policy_config = _load_policy_config(ckpt_dir, args)
    policy_config["camera_names"] = camera_names

    stats = _load_stats(ckpt_dir)

    qpos_mean = np.asarray(stats["qpos_mean"], dtype=np.float32)
    qpos_std = np.asarray(stats["qpos_std"], dtype=np.float32)
    imu_mean = np.asarray(stats["imu_mean"], dtype=np.float32)
    imu_std = np.asarray(stats["imu_std"], dtype=np.float32)
    tactile_mean = np.asarray(stats.get("tactile_mean", np.zeros((1, TACTILE_ROWS, TACTILE_COLS), dtype=np.float32)), dtype=np.float32)
    tactile_std = np.asarray(stats.get("tactile_std", np.ones_like(tactile_mean)), dtype=np.float32)
    action_mean = np.asarray(stats["action_mean"], dtype=np.float32)
    action_std = np.asarray(stats["action_std"], dtype=np.float32)
    tactile_shape = tuple(int(v) for v in stats.get("tactile_shape", (TACTILE_ROWS, TACTILE_COLS)))
    tactile_model_shape = tuple(int(v) for v in stats.get("tactile_model_shape", (1, *tactile_shape)))
    tactile_in_channels = int(stats.get("tactile_in_channels", tactile_model_shape[0]))
    policy_config.setdefault("use_tactile", bool(stats.get("has_tactile", False)))
    policy_config.setdefault("tactile_in_channels", tactile_in_channels)
    policy_config.setdefault("tactile_model_shape", tactile_model_shape)

    qpos_dim = int(qpos_mean.shape[0])
    action_dim = int(action_mean.shape[0])
    print(
        f"[deploy] stats: qpos_dim={qpos_dim}, action_dim={action_dim}, imu_dim={imu_mean.shape[0]}, "
        f"tactile_shape={tactile_shape}, tactile_model_shape={tactile_model_shape}, "
        f"tactile_in_channels={policy_config.get('tactile_in_channels')}"
    )

    def build_policy_qpos(q_arm_6d: np.ndarray, gripper_fb: float | None, gripper_cmd: float) -> np.ndarray:
        q_arm_6d = np.asarray(q_arm_6d, dtype=np.float32).reshape(-1)
        if q_arm_6d.size != 6:
            raise ValueError(f"机械臂反馈应为6维，当前为 {q_arm_6d.size} 维")
        if qpos_dim == 6:
            return q_arm_6d
        if qpos_dim == 7:
            g = gripper_fb if gripper_fb is not None else gripper_cmd
            return np.concatenate([q_arm_6d, np.array([float(g)], dtype=np.float32)], axis=0)
        raise ValueError(f"暂不支持 qpos_dim={qpos_dim}。请确认训练数据 qpos 维度。")

    def pre_process_qpos(s_qpos):
        s_qpos = np.asarray(s_qpos, dtype=np.float32).reshape(-1)
        if s_qpos.shape != qpos_mean.shape:
            raise ValueError(f"qpos维度不匹配: 当前 {s_qpos.shape}, 训练统计 {qpos_mean.shape}")
        return (s_qpos - qpos_mean) / qpos_std

    def pre_process_imu(s_imu):
        return (s_imu - imu_mean) / imu_std

    def tactile_to_model_input(tactile_frame):
        tactile_frame = np.asarray(tactile_frame, dtype=np.float32)
        if tactile_frame.shape == tactile_model_shape:
            return tactile_frame
        if tactile_frame.shape == tactile_shape and len(tactile_shape) == 2:
            return tactile_frame[None, :, :]
        if tactile_frame.shape == tactile_shape and len(tactile_shape) == 3:
            return tactile_frame
        raise ValueError(
            f"tactile维度不匹配: 当前 {tactile_frame.shape}, "
            f"训练 tactile_shape={tactile_shape}, tactile_model_shape={tactile_model_shape}"
        )

    def pre_process_tactile(tactile_frame):
        tactile_model = tactile_to_model_input(tactile_frame)
        return (tactile_model - tactile_mean) / tactile_std

    def post_process(action):
        return action * action_std + action_mean

    act_root = _activate_act_root(args.act_root)
    print(f"[deploy] tactile ACT source: {act_root}")
    from policy import ACTPolicy, CNNMLPPolicy

    if args.policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    else:
        policy = CNNMLPPolicy(policy_config)

    policy.load_state_dict(torch.load(ckpt_dir / "policy_best.ckpt"))
    policy.cuda()
    policy.eval()
    print(f"Loaded policy from {ckpt_dir / 'policy_best.ckpt'}")

    vlim = _parse_vector(args.vlim, np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=np.float64), "--vlim")
    # max_step = _parse_vector(args.max_step, np.array([0.02, 0.02, 0.02, 0.03, 0.03, 0.03], dtype=np.float64), "--max-step")
    max_step = _parse_vector(args.max_step, np.array([0.05, 0.05, 0.05, 0.08, 0.08, 0.08], dtype=np.float64), "--max-step")
    max_step_per_sec = max_step * max(float(args.rate), 1e-6)
    print(f"[deploy] vlim={np.round(vlim, 4).tolist()}")
    print(
        f"[deploy] max_step@{args.rate:.1f}Hz={np.round(max_step, 4).tolist()}, "
        f"max_step_per_sec={np.round(max_step_per_sec, 4).tolist()}"
    )

    RobotArm = _load_robot_arm_class()
    arm = None
    gripper_motor = None
    gripper_controller = None
    cameras = {}
    imu_reader = None
    tactile_reader = None
    visualizer = LatestVisualizer(
        enabled=not args.no_visualize,
        fps=args.visualize_fps,
        scale=args.visualize_scale,
        history_seconds=args.visualize_history_sec,
        show_sensor_panel=not args.no_sensor_visualize,
        sensor_panel_height=args.sensor_panel_height,
    )
    q_cmd = None
    async_runner = None

    try:
        arm = RobotArm(cfg_path=str(args.cfg) if args.cfg else None)
        arm.connect()
        arm.enable()
        arm.mode_pos_vel(vlim=vlim)

        if not args.no_gripper:
            gripper_motor, gripper_controller = setup_damiao_gripper(arm, args.gripper_cfg)

        for cam_name in camera_names:
            cameras[cam_name] = _make_camera_backend(cam_name, camera_sources[cam_name])

        time.sleep(0.5)
        for cam_name, cam in cameras.items():
            status_fn = getattr(cam, "status", None)
            status_text = status_fn() if callable(status_fn) else getattr(cam, "status_string", lambda: f"{cam_name}:unknown")()
            print(f"[camera] {status_text}")
            if args.camera_debug:
                debug_path = cam.save_debug_frame(args.camera_save_dir, step="init")
                print(f"[camera] 已保存启动调试图: {debug_path}")

        if not args.no_imu and bool(stats.get("has_imu", True)):
            imu_reader = ThreadedYbImuReader(
                port=args.imu_port,
                report_rate=args.imu_report_rate,
                alpha=args.imu_alpha,
                name="imu_left",
            )
        else:
            print("[deploy] IMU disabled: using zero imu input.")

        if not args.no_tactile and bool(stats.get("has_tactile", policy_config.get("use_tactile", False))):
            tactile_reader = ThreadedFlexiTacReader(
                port=args.tactile_port,
                baud=args.tactile_baud,
                init_frames=args.tactile_init_frames,
                threshold=args.tactile_threshold,
                noise_scale=args.tactile_noise_scale,
                alpha=args.tactile_alpha,
                name="right_tactile",
            )
        else:
            print("[deploy] tactile disabled: using zero tactile input.")
        visualizer.start()

        q_feedback = np.asarray(arm.get_positions(request=True)[:6], dtype=np.float64)
        guard = SafetyGuard(max_step_per_sec=max_step_per_sec, max_tracking_error=args.max_tracking_error, breach_samples=args.breach_samples)
        q_cmd = guard.initialize(q_feedback)

        gripper_fb = get_gripper_feedback_pos(gripper_motor) if gripper_motor is not None else None
        gripper_cmd = float(gripper_fb if gripper_fb is not None else args.gripper_default)

        async_inference = (not args.sync_inference) and args.policy_class == "ACT"
        if not async_inference and not args.sync_inference and args.policy_class != "ACT":
            print("[deploy] 当前仅 ACT 支持异步推理，CNNMLP 将使用同步推理。")

        query_frequency = policy_config["num_queries"]
        if args.temporal_agg:
            query_frequency = 1
            num_queries = policy_config["num_queries"]
            chunk_history = deque(maxlen=max(int(args.temporal_agg_max_history), 1) if args.temporal_agg_max_history > 0 else num_queries)
        else:
            num_queries = policy_config["num_queries"]
            chunk_history = None

        async_runner = AsyncPolicyRunner(policy) if async_inference else None
        if async_runner is not None:
            async_runner.start()
            print("[deploy] 已启用后台异步推理：控制循环不会等待 policy() 完成。")

        prev_chunk = None
        prev_chunk_origin = -1
        step = 0
        last_loop_time = time.perf_counter()
        print("\n[deploy] 开始闭环推理。按 Ctrl+C 停止。")

        with torch.inference_mode():
            while _running and (args.max_steps is None or step < args.max_steps):
                loop_start = time.perf_counter()
                control_dt = loop_start - last_loop_time
                last_loop_time = loop_start

                t0 = time.perf_counter()
                q_feedback_raw = np.asarray(arm.get_positions(request=True)[:6], dtype=np.float64)
                q_feedback = _unwrap_near(q_feedback_raw, q_cmd)
                read_ms = (time.perf_counter() - t0) * 1000.0

                t0 = time.perf_counter()
                cam_frames = []
                camera_infos = []
                for cam_name in camera_names:
                    cam = cameras[cam_name]
                    if hasattr(cam, "read"):
                        frame_rgb, frame_id, frame_ts = cam.read()
                    else:
                        frame_rgb = cam.read_rgb()
                        frame_id = 0
                        frame_ts = 0.0
                    cam_frames.append(frame_rgb)
                    info_fn = getattr(cam, "info", None)
                    if callable(info_fn):
                        camera_infos.append(info_fn())
                    else:
                        camera_infos.append(
                            {
                                "name": cam_name,
                                "backend": "unknown",
                                "valid": True,
                                "frame_id": int(frame_id),
                                "fps": 0.0,
                                "age_ms": (time.perf_counter() - frame_ts) * 1000.0 if frame_ts > 0 else float("inf"),
                                "error": "",
                                "exitcode": None,
                                "alive": True,
                                "timestamp": float(frame_ts),
                            }
                        )
                image_np = np.stack(cam_frames, axis=0)
                if async_runner is None:
                    image = torch.from_numpy(image_np).float().div_(255.0).permute(0, 3, 1, 2).cuda().unsqueeze(0)
                else:
                    image = None
                camera_ms = (time.perf_counter() - t0) * 1000.0

                if args.camera_save_every > 0 and step % args.camera_save_every == 0:
                    for cam_name, cam in cameras.items():
                        debug_path = cam.save_debug_frame(args.camera_save_dir, step=step)
                        print(f"[camera] step={step} 保存调试图: {debug_path}")

                gripper_fb_now = get_gripper_feedback_pos(gripper_motor) if gripper_motor is not None else None
                policy_qpos = build_policy_qpos(q_feedback.astype(np.float32), gripper_fb_now, gripper_cmd)
                qpos_np = pre_process_qpos(policy_qpos)
                imu_sample = imu_reader.read() if imu_reader is not None else {
                    "imu_left": np.zeros(IMU_DIM, dtype=np.float32),
                    "quat": np.zeros(4, dtype=np.float32),
                    "gyro": np.zeros(3, dtype=np.float32),
                    "accel": np.zeros(3, dtype=np.float32),
                    "frame_id": -1,
                    "timestamp": 0.0,
                    "valid": False,
                }
                tactile_frame, tactile_frame_id, tactile_ts = (
                    tactile_reader.read()
                    if tactile_reader is not None
                    else (np.zeros(tactile_shape, dtype=np.float32), -1, 0.0)
                )
                imu_np = pre_process_imu(imu_sample["imu_left"])
                tactile_np = pre_process_tactile(tactile_frame)
                imu_info = imu_reader.info() if imu_reader is not None else {
                    "name": "imu_left",
                    "valid": False,
                    "frame_id": -1,
                    "age_ms": float("inf"),
                    "error": "disabled",
                }
                tactile_info = tactile_reader.info() if tactile_reader is not None else {
                    "name": "right_tactile",
                    "valid": False,
                    "frame_id": int(tactile_frame_id),
                    "age_ms": (time.perf_counter() - tactile_ts) * 1000.0 if tactile_ts > 0 else float("inf"),
                    "max": float(np.max(tactile_frame)),
                    "mean": float(np.mean(tactile_frame)),
                    "sum": float(np.sum(tactile_frame)),
                    "error": "disabled",
                }

                infer_ms = 0.0
                async_age = -1
                if async_runner is not None:
                    if step % max(int(args.async_submit_every), 1) == 0:
                        async_runner.submit(step, qpos_np, image_np, imu_np, tactile_np)
                    if async_runner.error is not None:
                        raise RuntimeError(f"异步推理线程异常: {async_runner.error}") from async_runner.error
                    for result in async_runner.pop_results():
                        prev_chunk = result["chunk"]
                        prev_chunk_origin = int(result["origin_step"])
                        infer_ms = float(result["infer_ms"])
                        if args.temporal_agg and chunk_history is not None:
                            chunk_history.append((prev_chunk_origin, prev_chunk))
                    if async_runner.last_origin_step >= 0:
                        async_age = step - async_runner.last_origin_step
                    infer_ms = async_runner.last_infer_ms
                else:
                    qpos = torch.from_numpy(qpos_np).float().cuda().unsqueeze(0)
                    imu = torch.from_numpy(imu_np).float().cuda().unsqueeze(0)
                    tactile = torch.from_numpy(tactile_np).float().cuda().unsqueeze(0)
                    if step % query_frequency == 0 or prev_chunk is None:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        if args.policy_class == "ACT":
                            prev_chunk = policy(qpos, image, imu, tactile)
                        else:
                            prev_chunk = policy(qpos, image)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        infer_ms = (time.perf_counter() - t0) * 1000.0
                        prev_chunk_origin = step
                        if args.temporal_agg and chunk_history is not None:
                            chunk_history.append((prev_chunk_origin, prev_chunk.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=True)))

                t0 = time.perf_counter()
                fallback_action = np.zeros(action_dim, dtype=np.float32)
                fallback_action[:6] = q_feedback.astype(np.float32)
                if action_dim > 6:
                    fallback_action[6] = float(gripper_cmd)
                if args.temporal_agg:
                    if chunk_history is not None:
                        while chunk_history and chunk_history[0][0] + num_queries <= step:
                            chunk_history.popleft()
                        actions_for_curr_step = []
                        for origin, chunk in chunk_history:
                            chunk_idx = step - origin
                            if 0 <= chunk_idx < chunk.shape[0]:
                                actions_for_curr_step.append(chunk[chunk_idx])
                        if args.temporal_agg_max_history > 0:
                            actions_for_curr_step = actions_for_curr_step[-args.temporal_agg_max_history:]
                    else:
                        actions_for_curr_step = []

                    if len(actions_for_curr_step) == 0:
                        action = fallback_action
                    else:
                        actions_for_curr_step = np.stack(actions_for_curr_step, axis=0)
                        k = max(float(args.temporal_agg_k), 0.0)
                        ages = np.arange(len(actions_for_curr_step) - 1, -1, -1)
                        exp_weights = np.exp(-k * ages)
                        exp_weights = exp_weights / exp_weights.sum()
                        raw_action_np = (actions_for_curr_step * exp_weights[:, None]).sum(axis=0)
                        action = post_process(raw_action_np)
                else:
                    if async_runner is not None:
                        if prev_chunk is None:
                            action = fallback_action
                        else:
                            chunk_idx = np.clip(step - prev_chunk_origin, 0, prev_chunk.shape[0] - 1)
                            action = post_process(prev_chunk[int(chunk_idx)])
                    else:
                        raw_action = prev_chunk[:, step % query_frequency]
                        action = post_process(raw_action.squeeze(0).cpu().numpy())

                target_q = guard.next_command(action[:6], q_feedback, control_dt)
                arm.pos_vel(target_q, vlim=vlim)
                command_ms = (time.perf_counter() - t0) * 1000.0

                if gripper_motor is not None and action.shape[0] > 6 and step % max(int(args.gripper_send_every), 1) == 0:
                    t0 = time.perf_counter()
                    gripper_cmd = float(action[6])
                    send_damiao_gripper_mit(
                        g_mot=gripper_motor,
                        controller=gripper_controller,
                        target_rad=gripper_cmd,
                        kp=args.gripper_kp,
                        kd=args.gripper_kd,
                        tau=args.gripper_tau,
                        request_feedback=True,
                    )
                    command_ms += (time.perf_counter() - t0) * 1000.0

                if args.print_every > 0 and step % args.print_every == 0:
                    q_vel_feedback = _read_arm_velocity_or_zero(arm)
                    arm_err = np.abs(target_q - q_feedback)
                    arm_err_max = float(np.max(arm_err))
                    g_fb = get_gripper_feedback_pos(gripper_motor) if gripper_motor is not None else None
                    g_fb_str = "None" if g_fb is None else f"{g_fb:+.3f}"
                    camera_status = " | ".join(
                        (cam.status() if callable(getattr(cam, "status", None)) else getattr(cam, "status_string", lambda: "unknown")())
                        for cam in cameras.values()
                    )
                    imu_status = imu_reader.status() if imu_reader is not None else "imu_left:disabled"
                    tactile_status = tactile_reader.status() if tactile_reader is not None else "right_tactile:disabled"
                    loop_ms = (time.perf_counter() - loop_start) * 1000.0
                    work_hz = 1000.0 / max(loop_ms, 1e-6)
                    actual_hz = 1.0 / max(control_dt, 1e-6)
                    print(
                        f"[step={step}] "
                        f"work={loop_ms:.1f}ms({work_hz:.1f}Hz) | actual={actual_hz:.1f}Hz | "
                        f"dt={control_dt * 1000.0:.1f}ms | "
                        f"read={read_ms:.1f}ms cam={camera_ms:.1f}ms infer={infer_ms:.1f}ms cmd={command_ms:.1f}ms | "
                        f"async_age={async_age} | "
                        f"fb_J1={q_feedback[0]:+.2f} | cmd_J1={target_q[0]:+.2f} | "
                        f"vel_J1={q_vel_feedback[0]:+.2f} | "
                        f"max_err={arm_err_max:.3f} rad | "
                        f"gripper_cmd={gripper_cmd:+.3f}, fb={g_fb_str} | "
                        f"{camera_status} | {imu_status} | {tactile_status}"
                    )

                visualizer.update(
                    frames=cam_frames,
                    camera_names=camera_names,
                    camera_infos=camera_infos,
                    step=step,
                    q_feedback=q_feedback.copy(),
                    target_q=target_q.copy(),
                    action=action.copy(),
                    gripper_cmd=gripper_cmd,
                    gripper_fb=get_gripper_feedback_pos(gripper_motor) if gripper_motor is not None else None,
                    arm_err_max=float(np.max(np.abs(target_q - q_feedback))),
                    actual_hz=1.0 / max(control_dt, 1e-6),
                    dt_ms=control_dt * 1000.0,
                    read_ms=read_ms,
                    camera_ms=camera_ms,
                    infer_ms=infer_ms,
                    command_ms=command_ms,
                    async_age=async_age,
                    imu_info=imu_info,
                    imu_sample=imu_sample,
                    imu_raw=imu_sample.get("imu_left", np.zeros(IMU_DIM, dtype=np.float32)),
                    imu_model_input=imu_np,
                    tactile_info=tactile_info,
                    tactile_frame=tactile_frame,
                    tactile_model_input=tactile_np,
                )

                step += 1
                sleep_time = (1.0 / max(float(args.rate), 1e-6)) - (time.perf_counter() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[deploy] 用户中断。")
    except Exception as exc:
        print(f"\n❌ [deploy] 异常中止: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        if async_runner is not None:
            async_runner.stop()
        visualizer.stop()
        for cam in cameras.values():
            cam.release()
        if imu_reader is not None:
            imu_reader.release()
        if tactile_reader is not None:
            tactile_reader.release()
        if args.no_final_disable:
            try:
                if arm is not None:
                    q_hold = None
                    try:
                        q_hold = np.asarray(arm.get_positions(request=True)[:6], dtype=np.float64)
                    except Exception:
                        q_hold = q_cmd
                    if q_hold is not None:
                        hold_vlim = np.minimum(vlim, np.ones(6, dtype=np.float64) * 0.3) if vlim is not None else None
                        for _ in range(5):
                            arm.pos_vel(q_hold, vlim=hold_vlim)
                            time.sleep(0.02)
                        print(f"[deploy] 已下发保持当前位置命令: {np.round(q_hold, 3)}")
            except Exception as hold_exc:
                print(f"⚠️ 保持当前位置命令发送失败: {hold_exc}")
            print("[deploy] 保持使能状态，未执行 arm.disable()。")
        else:
            close_arm_fast(arm)
            print("[deploy] 机械臂已失能。")


if __name__ == "__main__":
    main()

# 模型部署推理
# python /act_tactile/rebot_scripts/Servo_control/deploy_rebot_real_policy_visualize_tactile.py --ckpt_dir /media/hjx/PSSD/hjx_ws/data/rebot/data_real_tactile/ckpt/ACT/rebot_real_grasp_banana --task_name rebot_real_grasp_banana --policy_class ACT --device cuda --camera-names cam_high,cam_wrist --camera-src cam_high=astra,cam_wrist=d405 --no-final-disable --print-every 10 --visualize-fps 10 --visualize-scale 0.5 --visualize-history-sec 8 --sensor-panel-height 360 --imu-port /dev/ttyUSB1 --tactile-port /dev/ttyUSB2 --tactile-baud 2000000 --tactile-init-frames 30

# 启动时间集合动作平滑【--temporal-agg-k 越大，越相信最新预测，--temporal-agg-max-history 越大，越平滑但越滞后】
# python /act_tactile/rebot_scripts/Servo_control/deploy_rebot_real_policy_visualize_tactile.py --ckpt_dir /media/hjx/PSSD/hjx_ws/data/rebot/data_real_tactile/ckpt/ACT/rebot_real_grasp_banana --task_name rebot_real_grasp_banana --policy_class ACT --device cuda --camera-names cam_high,cam_wrist --camera-src cam_high=astra,cam_wrist=d405 --no-final-disable --print-every 10 --visualize-fps 10 --visualize-scale 0.5 --visualize-history-sec 8 --sensor-panel-height 360 --imu-port /dev/ttyUSB1 --tactile-port /dev/ttyUSB2 --tactile-baud 2000000 --tactile-init-frames 30 --temporal-agg --temporal-agg-k 0.2 --temporal-agg-max-history 10
