#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
reBotArm 达妙机械臂 + 达妙夹爪 纯物理回零程序

特点：
1. 参考 rebot_position_replay_hjx.py 的控制参数：
   - DEFAULT_CMD_VLIM = [1.5, 1.5, 1.5, 3.0, 3.0, 3.0]
   - DEFAULT_MAX_STEP = [0.05, 0.05, 0.05, 0.06, 0.06, 0.06]
   - SafetyGuard 跟踪误差保护
   - 4 秒平滑预引导回零
   - 回零完成后保持 2 秒，然后自动失能保护

2. 机械臂：
   - POS_VEL 模式
   - 目标零位默认 q = [0, 0, 0, 0, 0, 0] rad

3. 夹爪：
   - 使用达妙 MIT 模式
   - 默认夹爪零位 gripper_home_rad = 0.0 rad
   - 可通过 --gripper-home-rad 修改

运行：
    python rebot_go_home_physical_with_gripper.py

指定夹爪配置：
    python home_rebot.py \
        --gripper-cfg rebot_smolvla_real/workflow/config/gripper.yaml

让夹爪回到张开位：
    python home_rebot.py --gripper-home-rad -5.8

让夹爪回到闭合附近：
    python home_rebot.py --gripper-home-rad 0.2
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# 路径注入
# --------------------------------------------------------------------------- #

CURRENT_DIR = Path(__file__).resolve().parent


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


def _inject_paths() -> Path | None:
    robot_pkg_dir = _find_first_existing(
        [
            "reBotArm_control_py",
            "Python/reBotArm_control_py",
        ]
    )

    add_paths: list[Path] = []

    if robot_pkg_dir is not None:
        add_paths.append(robot_pkg_dir.parent)

    add_paths.extend(_candidate_roots()[:5])

    for p in add_paths:
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)

    return robot_pkg_dir


ROBOT_PKG_DIR = _inject_paths()


def _load_robot_arm_class():
    try:
        from reBotArm_control_py.actuator import RobotArm
        return RobotArm

    except ImportError as exc:
        print(f"[导入] 常规导入 RobotArm 失败，尝试直接加载 arm.py: {exc}")

    possible_arm_py: list[Path] = []

    if ROBOT_PKG_DIR is not None:
        possible_arm_py.append(ROBOT_PKG_DIR / "actuator" / "arm.py")

    for root in _candidate_roots():
        possible_arm_py.append(root / "reBotArm_control_py" / "actuator" / "arm.py")
        possible_arm_py.append(root / "Python" / "reBotArm_control_py" / "actuator" / "arm.py")

    for arm_py in possible_arm_py:
        if arm_py.exists():
            spec = importlib.util.spec_from_file_location(
                "_rebotarm_actuator_arm",
                arm_py,
            )

            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            return module.RobotArm

    raise ImportError("无法加载 RobotArm，请检查 reBotArm_control_py/actuator/arm.py 是否存在。")


def _load_gripper_cfg_func():
    try:
        from reBotArm_control_py.actuator.gripper import load_cfg
        return load_cfg

    except ImportError:
        pass

    possible_gripper_py: list[Path] = []

    if ROBOT_PKG_DIR is not None:
        possible_gripper_py.append(ROBOT_PKG_DIR / "actuator" / "gripper.py")

    for root in _candidate_roots():
        possible_gripper_py.append(root / "reBotArm_control_py" / "actuator" / "gripper.py")
        possible_gripper_py.append(root / "Python" / "reBotArm_control_py" / "actuator" / "gripper.py")

    for gripper_py in possible_gripper_py:
        if gripper_py.exists():
            spec = importlib.util.spec_from_file_location(
                "_rebotarm_actuator_gripper",
                gripper_py,
            )

            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            return module.load_cfg

    raise ImportError("无法加载 gripper.py 中的 load_cfg，请检查 reBotArm_control_py/actuator/gripper.py。")


# --------------------------------------------------------------------------- #
# 全局运行标志
# --------------------------------------------------------------------------- #

_running = True


def _sigint_handler(signum, frame) -> None:
    global _running
    print("\n[GoHome] 收到退出信号，准备安全关闭...")
    _running = False


signal.signal(signal.SIGINT, _sigint_handler)
signal.signal(signal.SIGTERM, _sigint_handler)


# --------------------------------------------------------------------------- #
# 控制参数：参考你的重播代码
# --------------------------------------------------------------------------- #

DEFAULT_CMD_VLIM = np.array(
    [1.5, 1.5, 1.5, 3.0, 3.0, 3.0],
    dtype=np.float64,
)

DEFAULT_MAX_STEP = np.array(
    [0.05, 0.05, 0.05, 0.06, 0.06, 0.06],
    dtype=np.float64,
)

DEFAULT_TRACKING_BREACH_SAMPLES = 20

DEFAULT_TARGET_Q = np.zeros(6, dtype=np.float64)

DEFAULT_RATE = 50.0
DEFAULT_PREPARE_DURATION = 4.0
DEFAULT_PROTECT_DURATION = 2.0
DEFAULT_MAX_TRACKING_ERROR = 1.0

# 夹爪默认零位：0 rad
# 需要张开位时使用 --gripper-home-rad -5.8
# 需要闭合附近时使用 --gripper-home-rad 0.2
DEFAULT_GRIPPER_HOME_RAD = 0.0
DEFAULT_ARM_CFG = CURRENT_DIR / "config" / "arm.yaml"


def _default_gripper_cfg_path() -> Path:
    return CURRENT_DIR / "config" / "gripper.yaml"


DEFAULT_GRIPPER_CFG = _default_gripper_cfg_path()


# --------------------------------------------------------------------------- #
# 基础工具函数
# --------------------------------------------------------------------------- #

def _parse_vector(values: list[float] | None, default: np.ndarray, name: str) -> np.ndarray:
    arr = default.astype(np.float64) if values is None else np.asarray(values, dtype=np.float64)

    if arr.shape != default.shape:
        raise ValueError(f"{name} 必须提供 {default.size} 个数，当前为 {arr.size} 个")

    return arr


def _clip_rate(target: np.ndarray, previous: np.ndarray, max_step: np.ndarray) -> np.ndarray:
    return previous + np.clip(target - previous, -max_step, max_step)


def _unwrap_near(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    return values + 2.0 * np.pi * np.round((reference - values) / (2.0 * np.pi))


def _cosine_smooth(progress: float) -> float:
    """
    与重播代码一致的余弦平滑：
        smooth = (1 - cos(progress*pi)) / 2
    """

    progress = float(np.clip(progress, 0.0, 1.0))
    return (1.0 - math.cos(progress * math.pi)) / 2.0


def read_stable_positions(
    arm,
    samples: int = 20,
    interval: float = 0.02,
) -> np.ndarray:
    """
    启动时多次读取当前位置，取中位数作为稳定初始位置。
    """

    first = np.asarray(arm.get_positions(request=True)[:6], dtype=np.float64)
    reference = first.copy()
    values = []

    for _ in range(max(int(samples), 1)):
        q = np.asarray(arm.get_positions(request=True)[:6], dtype=np.float64)
        q = _unwrap_near(q, reference)
        values.append(q)
        reference = q.copy()
        time.sleep(max(float(interval), 0.0))

    return np.median(np.vstack(values), axis=0)


def read_arm_velocity_or_zero(arm) -> np.ndarray:
    try:
        return np.asarray(arm.get_velocities(request=True)[:6], dtype=np.float64)
    except Exception:
        return np.zeros(6, dtype=np.float64)


def close_arm_fast(arm) -> None:
    if arm is None:
        return

    try:
        print("[硬件保护] 执行失能指令...")
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

    try:
        getattr(arm, "_ctrl_map", {}).clear()
        getattr(arm, "_motor_map", {}).clear()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# SafetyGuard：参考你的重播代码
# --------------------------------------------------------------------------- #

class SafetyGuard:
    def __init__(
        self,
        num_joints: int,
        max_step: np.ndarray,
        max_tracking_error: float,
        tracking_breach_samples: int,
    ):
        self.num_joints = int(num_joints)
        self.max_step = np.asarray(max_step, dtype=np.float64)[: self.num_joints]
        self.max_tracking_error = float(max_tracking_error)
        self.tracking_breach_samples = max(int(tracking_breach_samples), 1)

        self.command: np.ndarray | None = None
        self._tracking_breach_count = 0

    def initialize(self, q_real_now: np.ndarray) -> np.ndarray:
        self.command = np.asarray(q_real_now, dtype=np.float64)[: self.num_joints].copy()
        return self.command.copy()

    def next_command(self, q_target: np.ndarray, q_feedback: np.ndarray) -> np.ndarray:
        if self.command is None:
            raise RuntimeError("SafetyGuard 尚未 initialize。")

        q_target = np.asarray(q_target, dtype=np.float64)[: self.num_joints]
        q_feedback = np.asarray(q_feedback, dtype=np.float64)[: self.num_joints]

        q_target_cmd = _unwrap_near(q_target, q_feedback)
        previous_cmd = _unwrap_near(self.command, q_feedback)

        tracking_error = float(np.max(np.abs(q_target_cmd - q_feedback)))

        if tracking_error > self.max_tracking_error:
            self._tracking_breach_count += 1

            if self._tracking_breach_count >= self.tracking_breach_samples:
                raise RuntimeError(
                    f"真机跟踪误差过大: {tracking_error:.3f} rad，触发保护停机。"
                )
        else:
            self._tracking_breach_count = 0

        self.command = _clip_rate(q_target_cmd, previous_cmd, self.max_step)

        return self.command.copy()


# --------------------------------------------------------------------------- #
# 达妙夹爪初始化与 MIT 控制
# --------------------------------------------------------------------------- #

def setup_damiao_gripper(
    arm,
    gripper_cfg_path: Path,
    gripper_name_fallback: str = "gripper",
):
    if arm is None:
        return None, None, None

    if not gripper_cfg_path.exists():
        raise FileNotFoundError(f"夹爪配置文件不存在: {gripper_cfg_path}")

    load_gripper_cfg = _load_gripper_cfg_func()
    g_cfg = load_gripper_cfg(str(gripper_cfg_path))["gripper"]

    if "damiao" not in arm._ctrl_map:
        raise RuntimeError(
            "arm._ctrl_map 中没有 'damiao' 控制器，无法添加达妙夹爪电机。"
        )

    shared_damiao_controller = arm._ctrl_map["damiao"]
    gripper_name = getattr(g_cfg, "name", gripper_name_fallback)

    if gripper_name in arm._motor_map:
        g_mot = arm._motor_map[gripper_name]
        print(f"✅ 夹爪电机已存在于 arm._motor_map: {gripper_name}")
    else:
        g_mot = shared_damiao_controller.add_damiao_motor(
            g_cfg.motor_id,
            g_cfg.feedback_id,
            g_cfg.model,
        )

        arm._motor_map[gripper_name] = g_mot

        print(
            f"✅ 已添加达妙夹爪电机: "
            f"name={gripper_name}, motor_id={g_cfg.motor_id}, "
            f"feedback_id={g_cfg.feedback_id}, model={g_cfg.model}"
        )

    try:
        from motorbridge import Mode

        g_mot.ensure_mode(Mode.MIT, 1000)
        shared_damiao_controller.enable_all()
        time.sleep(0.2)

        print("✅ 夹爪已切入 MIT 模式并完成使能")

        st = g_mot.get_state()

        if st is not None:
            print(f"✅ 夹爪当前反馈位置: {st.pos:.3f} rad")
        else:
            print("⚠️ 夹爪初始状态读取为空，但电机已尝试使能。")

    except Exception as e:
        raise RuntimeError(f"夹爪 MIT 模式配置或使能失败: {e}") from e

    return g_mot, shared_damiao_controller, gripper_name


def send_damiao_gripper_mit(
    g_mot,
    controller,
    target_rad: float,
    kp: float,
    kd: float,
    tau: float,
    request_feedback: bool = True,
) -> bool:
    if g_mot is None:
        return False

    try:
        g_mot.send_mit(
            float(target_rad),
            0.0,
            float(kp),
            float(kd),
            float(tau),
        )

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
        print(f"⚠️ 夹爪 MIT 命令发送失败: {e}")
        return False


def get_gripper_feedback_pos(g_mot) -> float | None:
    if g_mot is None:
        return None

    try:
        st = g_mot.get_state()

        if st is None:
            return None

        return float(st.pos)

    except Exception:
        return None


# --------------------------------------------------------------------------- #
# 参数
# --------------------------------------------------------------------------- #

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="reBotArm 达妙机械臂 + 达妙夹爪纯物理回零程序"
    )

    parser.add_argument(
        "--cfg",
        type=Path,
        default=DEFAULT_ARM_CFG,
        help="RobotArm 配置文件路径，默认使用 workflow/config/arm.yaml",
    )

    parser.add_argument(
        "--target",
        type=float,
        nargs=6,
        default=None,
        help="机械臂 6 轴目标零位，单位 rad，默认 0 0 0 0 0 0",
    )

    parser.add_argument(
        "--rate",
        type=float,
        default=DEFAULT_RATE,
        help="控制频率 Hz，默认 50",
    )

    parser.add_argument(
        "--prepare-duration",
        type=float,
        default=DEFAULT_PREPARE_DURATION,
        help="从当前位置平滑回零的时间，单位 s，默认 4.0",
    )

    parser.add_argument(
        "--protect-duration",
        type=float,
        default=DEFAULT_PROTECT_DURATION,
        help="到零位后保持时间，单位 s，默认 2.0；随后自动失能",
    )

    parser.add_argument(
        "--max-tracking-error",
        type=float,
        default=DEFAULT_MAX_TRACKING_ERROR,
        help="最大跟踪误差，单位 rad，默认 1.0",
    )

    parser.add_argument(
        "--tracking-breach-samples",
        type=int,
        default=DEFAULT_TRACKING_BREACH_SAMPLES,
        help="连续多少帧超误差后触发保护，默认 20",
    )

    parser.add_argument(
        "--vlim",
        type=float,
        nargs=6,
        default=None,
        help="6 轴速度限制，单位 rad/s",
    )

    parser.add_argument(
        "--max-step",
        type=float,
        nargs=6,
        default=None,
        help="每个控制周期最大命令变化量，单位 rad",
    )

    parser.add_argument(
        "--settle-samples",
        type=int,
        default=20,
        help="启动时读取当前位置的采样次数，默认 20",
    )

    parser.add_argument(
        "--settle-interval",
        type=float,
        default=0.02,
        help="启动读取当前位置的采样间隔，单位 s，默认 0.02",
    )

    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="每隔多少帧打印一次状态，默认 10",
    )

    # 夹爪参数
    parser.add_argument(
        "--no-gripper",
        action="store_true",
        help="只回零机械臂，不控制夹爪",
    )

    parser.add_argument(
        "--gripper-cfg",
        type=Path,
        default=DEFAULT_GRIPPER_CFG,
        help="夹爪配置文件路径，默认自动寻找 config/gripper.yaml",
    )

    parser.add_argument(
        "--gripper-home-rad",
        type=float,
        default=DEFAULT_GRIPPER_HOME_RAD,
        help="夹爪零位，单位 rad，默认 0.0",
    )

    parser.add_argument(
        "--gripper-kp",
        type=float,
        default=1.0,
        help="夹爪 MIT kp，默认 1.0",
    )

    parser.add_argument(
        "--gripper-kd",
        type=float,
        default=0.05,
        help="夹爪 MIT kd，默认 0.05",
    )

    parser.add_argument(
        "--gripper-tau",
        type=float,
        default=0.0,
        help="夹爪 MIT tau，默认 0.0",
    )

    parser.add_argument(
        "--gripper-send-every",
        type=int,
        default=2,
        help="每隔多少帧发送一次夹爪 MIT 命令，默认 2",
    )

    return parser


# --------------------------------------------------------------------------- #
# 主程序
# --------------------------------------------------------------------------- #

def main() -> None:
    global _running

    args = build_argparser().parse_args()

    target_q = _parse_vector(args.target, DEFAULT_TARGET_Q, "--target")
    vlim = _parse_vector(args.vlim, DEFAULT_CMD_VLIM, "--vlim")
    max_step = _parse_vector(args.max_step, DEFAULT_MAX_STEP, "--max-step")

    enable_gripper = not args.no_gripper

    rate = max(float(args.rate), 1e-6)
    cmd_period = 1.0 / rate

    prepare_duration = max(float(args.prepare_duration), 0.1)
    protect_duration = max(float(args.protect_duration), 0.0)

    print("\n" + "=" * 100)
    print("reBotArm 达妙机械臂 + 达妙夹爪 纯物理回零系统")
    print("=" * 100)
    print(f"RobotArm cfg: {args.cfg}")
    print(f"机械臂目标零位 target_q(rad): {np.round(target_q, 4).tolist()}")
    print(f"控制频率 rate: {rate:.1f} Hz")
    print(f"准备回零时间 prepare_duration: {prepare_duration:.2f} s")
    print(f"零位保持时间 protect_duration: {protect_duration:.2f} s")
    print(f"vlim(rad/s): {np.round(vlim, 4).tolist()}")
    print(f"max_step(rad/frame): {np.round(max_step, 4).tolist()}")
    print(f"max_tracking_error: {args.max_tracking_error:.3f} rad")
    print(f"tracking_breach_samples: {args.tracking_breach_samples}")
    print(f"夹爪启用: {enable_gripper}")

    if enable_gripper:
        print(f"夹爪 cfg: {args.gripper_cfg}")
        print(f"夹爪零位 gripper_home_rad: {args.gripper_home_rad:.3f}")
        print(
            f"夹爪 MIT 参数: "
            f"kp={args.gripper_kp:.3f}, "
            f"kd={args.gripper_kd:.3f}, "
            f"tau={args.gripper_tau:.3f}"
        )

    print("=" * 100)

    if not args.cfg.is_file():
        print(f"❌ RobotArm 配置文件不存在: {args.cfg}")
        return

    if enable_gripper and not args.gripper_cfg.is_file():
        print(f"❌ 夹爪配置文件不存在: {args.gripper_cfg}")
        return

    arm = None
    gripper_motor = None
    gripper_controller = None

    try:
        # ---------------- 初始化机械臂 ----------------
        print("\n🤖 正在初始化达妙机械臂 RobotArm...")

        RobotArm = _load_robot_arm_class()
        arm = RobotArm(cfg_path=str(args.cfg) if args.cfg is not None else None)

        arm.connect()
        arm.enable()
        arm.mode_pos_vel(vlim=vlim)

        time.sleep(0.5)

        print("✅ 达妙机械臂已连接、使能，并进入 POS_VEL 模式")

        # ---------------- 初始化夹爪 ----------------
        if enable_gripper:
            print("\n🦾 正在初始化达妙夹爪...")

            gripper_motor, gripper_controller, gripper_name = setup_damiao_gripper(
                arm=arm,
                gripper_cfg_path=args.gripper_cfg,
            )

            print(f"✅ 达妙夹爪已准备完成: {gripper_name}")

            send_damiao_gripper_mit(
                g_mot=gripper_motor,
                controller=gripper_controller,
                target_rad=args.gripper_home_rad,
                kp=args.gripper_kp,
                kd=args.gripper_kd,
                tau=args.gripper_tau,
                request_feedback=True,
            )

            print(f"✅ 已发送初始夹爪零位命令: {args.gripper_home_rad:.3f} rad")

        # ---------------- 读取当前位姿 ----------------
        print("\n📡 正在读取当前机械臂 6 轴反馈位置...")

        q_start = read_stable_positions(
            arm=arm,
            samples=args.settle_samples,
            interval=args.settle_interval,
        )

        q_start = _unwrap_near(q_start, target_q)

        print(f"当前反馈 q_start(rad): {np.round(q_start, 4).tolist()}")
        print(f"目标零位 q_home(rad):  {np.round(target_q, 4).tolist()}")

        start_error = np.abs(target_q - q_start)

        print(
            "初始误差(deg): "
            + " | ".join(
                [
                    f"J{i + 1}:{start_error[i] * 180.0 / np.pi:.1f}°"
                    for i in range(6)
                ]
            )
        )

        guard = SafetyGuard(
            num_joints=6,
            max_step=max_step,
            max_tracking_error=args.max_tracking_error,
            tracking_breach_samples=args.tracking_breach_samples,
        )

        q_cmd = guard.initialize(q_start)

        # ---------------- 回零主循环 ----------------
        print("\n🚀 [回零启动] 使用余弦平滑轨迹，从当前位置引导到零位")
        print("按 Ctrl+C 可中断，程序会执行失能保护。\n")

        t_start = time.perf_counter()
        frame = 0

        while _running:
            loop_t0 = time.perf_counter()
            elapsed = loop_t0 - t_start

            q_raw_feedback = np.asarray(
                arm.get_positions(request=True)[:6],
                dtype=np.float64,
            )

            q_feedback = _unwrap_near(q_raw_feedback, q_cmd)
            q_vel_feedback = read_arm_velocity_or_zero(arm)

            if elapsed < prepare_duration:
                progress = elapsed / prepare_duration
                smooth = _cosine_smooth(progress)

                q_target = q_start + (target_q - q_start) * smooth
                stage = "回零"
            else:
                q_target = target_q.copy()
                stage = "保持"

                if elapsed >= prepare_duration + protect_duration:
                    print(
                        f"\n✅ [回零完成] elapsed={elapsed:.2f}s，"
                        f"已保持 {protect_duration:.2f}s，准备自动失能。"
                    )
                    break

            q_cmd = guard.next_command(q_target, q_feedback)
            arm.pos_vel(q_cmd, vlim=vlim)

            # 夹爪持续发送零位 MIT 命令
            if enable_gripper and frame % max(int(args.gripper_send_every), 1) == 0:
                send_damiao_gripper_mit(
                    g_mot=gripper_motor,
                    controller=gripper_controller,
                    target_rad=args.gripper_home_rad,
                    kp=args.gripper_kp,
                    kd=args.gripper_kd,
                    tau=args.gripper_tau,
                    request_feedback=True,
                )

            # 打印监控
            if args.print_every > 0 and frame % args.print_every == 0:
                tracking_error = np.abs(q_cmd - q_feedback)

                arm_info = []

                for i in range(6):
                    arm_info.append(
                        f"J{i + 1}: "
                        f"pos {q_feedback[i] * 180.0 / np.pi:.1f}° "
                        f"(cmd {q_cmd[i] * 180.0 / np.pi:.1f}°) "
                        f"vel {q_vel_feedback[i]:.2f} rad/s "
                        f"err {tracking_error[i] * 180.0 / np.pi:.1f}°"
                    )

                if enable_gripper:
                    g_fb = get_gripper_feedback_pos(gripper_motor)
                    g_fb_str = "None" if g_fb is None else f"{g_fb:+.3f} rad"

                    print(
                        f"[t={elapsed:.2f}s][{stage}] "
                        + " | ".join(arm_info)
                        + f" | Gripper cmd {args.gripper_home_rad:+.3f} rad "
                        + f"fb {g_fb_str}"
                    )

                else:
                    print(
                        f"[t={elapsed:.2f}s][{stage}] "
                        + " | ".join(arm_info)
                    )

            frame += 1

            time.sleep(max(0.0, cmd_period - (time.perf_counter() - loop_t0)))

    except KeyboardInterrupt:
        print("\n[GoHome] 用户中断。")

    except Exception as exc:
        print(f"\n[运行异常] {exc}")

        import traceback
        traceback.print_exc()

    finally:
        print("\n[退出流程] 正在执行机械臂失能保护...")
        close_arm_fast(arm)
        print("[退出] 机械臂与夹爪控制器已关闭。")


if __name__ == "__main__":
    main()
