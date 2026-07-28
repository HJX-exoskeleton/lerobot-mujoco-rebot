#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
达妙真机机械臂 + 夹爪：ALOHA 规范轨迹重播系统

功能特性：
1. 兼容 ALOHA 规范：优先读取 HDF5 中的 /observations/qpos。
2. 7-DoF 同步控制：前 6 维 POS_VEL 控制机械臂，第 7 维 MIT 控制夹爪。
3. 安全预引导：重播前 4 秒余弦平滑引导，防止第一帧位置突变。
4. 物理防暴走：SafetyGuard 步长削峰 + 真机跟踪误差监控。
5. 支持 sh 串联：
   - 默认结束后失能；
   - 使用 --no-final-disable 时，replay 结束后不失能，便于 home_rebot.py 接管。
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import signal
import sys
import time
from pathlib import Path

import h5py
import numpy as np


# =============================================================================
# 路径与动态导入
# =============================================================================

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
            "Servo_control/reBotArm_control_py",
        ]
    )

    add_paths: list[Path] = []

    if robot_pkg_dir is not None:
        add_paths.append(robot_pkg_dir.parent)

    add_paths.extend(_candidate_roots()[:6])

    for p in add_paths:
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)

    return robot_pkg_dir


ROBOT_PKG_DIR = _inject_paths()


def _load_robot_arm_class():
    """
    优先常规导入 RobotArm。
    如果 reBotArm_control_py/__init__.py 里 kinematics/pinocchio 有问题，
    则直接从 actuator/arm.py 加载 RobotArm。
    """

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
            print(f"[导入] 直接加载 RobotArm: {arm_py}")

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
    """
    加载 gripper.py 中的 load_cfg。
    """

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
            print(f"[导入] 直接加载 gripper load_cfg: {gripper_py}")

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

    raise ImportError("无法加载 gripper.py 中的 load_cfg，请检查环境。")


# =============================================================================
# 全局标志与配置
# =============================================================================

_running = True

# 机械臂 6 轴安全限制
# 如果你想更接近之前的 replay 参数，可以运行时传 --vlim / --max-step 覆盖
DEFAULT_CMD_VLIM = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=np.float64)
DEFAULT_MAX_STEP = np.array([0.02, 0.02, 0.02, 0.03, 0.03, 0.03], dtype=np.float64)

DEFAULT_TRACKING_BREACH_SAMPLES = 20


def _sigint_handler(signum, frame) -> None:
    global _running
    print("\n[Replay] 收到退出信号，准备安全停机...")
    _running = False


signal.signal(signal.SIGINT, _sigint_handler)
signal.signal(signal.SIGTERM, _sigint_handler)


# =============================================================================
# 基础工具函数
# =============================================================================

def _parse_vector(values: list[float] | None, default: np.ndarray, name: str) -> np.ndarray:
    arr = default.astype(np.float64) if values is None else np.asarray(values, dtype=np.float64)

    if arr.shape != default.shape:
        raise ValueError(f"{name} 必须提供 {default.size} 个数，当前为 {arr.size} 个")

    return arr


def _unwrap_near(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    防止关节跨越 2pi 边界时发生反向跳变。
    """

    values = np.asarray(values, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    return values + 2.0 * np.pi * np.round((reference - values) / (2.0 * np.pi))


def _clip_rate(target: np.ndarray, previous: np.ndarray, max_step: np.ndarray) -> np.ndarray:
    """
    指令平滑：限制单控制周期最大变化量。
    """

    return previous + np.clip(target - previous, -max_step, max_step)


def _cosine_smooth(progress: float) -> float:
    progress = float(np.clip(progress, 0.0, 1.0))
    return (1.0 - math.cos(progress * math.pi)) / 2.0


def _read_arm_velocity_or_zero(arm) -> np.ndarray:
    try:
        return np.asarray(arm.get_velocities(request=True)[:6], dtype=np.float64)
    except Exception:
        return np.zeros(6, dtype=np.float64)


# =============================================================================
# 夹爪控制函数
# =============================================================================

def _default_gripper_cfg_path() -> Path:
    candidates = [
        "config/gripper.yaml",
        "Python/config/gripper.yaml",
        "Servo_control/config/gripper.yaml",
        "../config/gripper.yaml",
        "../../config/gripper.yaml",
    ]

    p = _find_first_existing(candidates)

    if p is not None:
        return p

    return CURRENT_DIR / "config" / "gripper.yaml"


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
        print(f"✅ 夹爪电机已存在: {gripper_name}")
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

    from motorbridge import Mode

    g_mot.ensure_mode(Mode.MIT, 1000)
    shared_damiao_controller.enable_all()
    time.sleep(0.2)

    st = g_mot.get_state()

    if st is not None:
        print(f"✅ 夹爪已切入 MIT 模式，当前反馈位置: {st.pos:.3f} rad")
    else:
        print("⚠️ 夹爪已尝试切入 MIT 模式，但初始反馈为空。")

    return g_mot, shared_damiao_controller


def send_damiao_gripper_mit(
    g_mot,
    controller,
    target_rad: float,
    kp: float = 1.0,
    kd: float = 0.05,
    tau: float = 0.0,
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
        print(f"\n⚠️ 夹爪 MIT 命令发送失败: {e}")
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


# =============================================================================
# 核心防护网
# =============================================================================

class SafetyGuard:
    def __init__(
        self,
        max_step: np.ndarray,
        max_tracking_error: float = 1.0,
        breach_samples: int = DEFAULT_TRACKING_BREACH_SAMPLES,
    ):
        self.max_step = np.asarray(max_step, dtype=np.float64)
        self.max_tracking_error = float(max_tracking_error)
        self.breach_samples = max(int(breach_samples), 1)

        self.command: np.ndarray | None = None
        self._breach_count = 0

    def initialize(self, q_real_now: np.ndarray) -> np.ndarray:
        self.command = np.asarray(q_real_now, dtype=np.float64)[:6].copy()
        return self.command.copy()

    def next_command(self, q_target: np.ndarray, q_feedback: np.ndarray) -> np.ndarray:
        if self.command is None:
            raise RuntimeError("SafetyGuard 尚未 initialize。")

        q_target = np.asarray(q_target, dtype=np.float64)[:6]
        q_feedback = np.asarray(q_feedback, dtype=np.float64)[:6]

        q_target_cmd = _unwrap_near(q_target, q_feedback)
        previous_cmd = _unwrap_near(self.command, q_feedback)

        tracking_error = float(np.max(np.abs(q_target_cmd - q_feedback)))

        if tracking_error > self.max_tracking_error:
            self._breach_count += 1

            if self._breach_count >= self.breach_samples:
                raise RuntimeError(
                    f"⚠️ 真机跟踪误差过大 ({tracking_error:.2f} rad)，触发保护。"
                )
        else:
            self._breach_count = 0

        self.command = _clip_rate(q_target_cmd, previous_cmd, self.max_step)

        return self.command.copy()


def close_arm_fast(arm) -> None:
    if arm is None:
        return

    try:
        print("[硬件保护] 执行 arm.disable() ...")
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


# =============================================================================
# HDF5 数据读取
# =============================================================================

def load_qpos_from_hdf5(dataset_path: str, fallback_rate: float) -> tuple[np.ndarray, float]:
    with h5py.File(dataset_path, "r") as f:
        if "/observations/qpos" in f:
            qpos_data = np.asarray(f["/observations/qpos"], dtype=np.float64)
            print("✅ 已读取 /observations/qpos")
        elif "qpos" in f:
            qpos_data = np.asarray(f["qpos"], dtype=np.float64)
            print("✅ 已读取 /qpos")
        else:
            raise KeyError("HDF5 中找不到 /observations/qpos 或 /qpos")

        record_rate = float(f.attrs.get("hz_rate", fallback_rate))

    if qpos_data.ndim != 2:
        raise ValueError(f"qpos_data 应该是二维数组，当前 shape={qpos_data.shape}")

    if qpos_data.shape[1] < 6:
        raise ValueError(f"qpos 至少需要 6 维机械臂数据，当前 shape={qpos_data.shape}")

    return qpos_data, record_rate


# =============================================================================
# 主逻辑
# =============================================================================

def main() -> None:
    global _running

    parser = argparse.ArgumentParser(description="达妙真机 ALOHA 规范重播系统")

    parser.add_argument("--dataset", "-d", type=str, required=True, help="HDF5 文件路径")
    parser.add_argument("--cfg", type=Path, default=None, help="RobotArm 配置文件")

    parser.add_argument(
        "--gripper-cfg",
        type=Path,
        default=_default_gripper_cfg_path(),
        help="夹爪配置文件路径",
    )

    parser.add_argument("--no-gripper", action="store_true", help="不控制夹爪")
    parser.add_argument("--speed-scale", type=float, default=1.0, help="重播速度倍率")
    parser.add_argument("--rate", type=float, default=50.0, help="控制频率 Hz")
    parser.add_argument("--prepare-duration", type=float, default=4.0, help="预引导时间 s")
    parser.add_argument("--post-hold", type=float, default=0.5, help="重播结束后保持最后一帧的时间 s")

    parser.add_argument("--max-tracking-error", type=float, default=1.0, help="最大跟踪误差 rad")
    parser.add_argument("--breach-samples", type=int, default=20, help="连续超误差多少帧触发保护")
    parser.add_argument("--vlim", type=float, nargs=6, default=None, help="机械臂 6 轴速度限制 rad/s")
    parser.add_argument("--max-step", type=float, nargs=6, default=None, help="每帧最大命令变化 rad")

    parser.add_argument("--gripper-kp", type=float, default=1.0)
    parser.add_argument("--gripper-kd", type=float, default=0.05)
    parser.add_argument("--gripper-tau", type=float, default=0.0)
    parser.add_argument("--gripper-send-every", type=int, default=1)

    parser.add_argument(
        "--gripper-index",
        type=int,
        default=6,
        help="qpos 中夹爪维度索引，默认 6，即第 7 维",
    )

    parser.add_argument(
        "--gripper-default",
        type=float,
        default=-5.8,
        help="当数据没有第 7 维时，夹爪保持的默认目标，默认 -5.8 rad 张开位",
    )

    parser.add_argument(
        "--no-final-disable",
        action="store_true",
        help="重播结束后不执行 arm.disable()，用于 sh 脚本后续运行 home_rebot.py 接管",
    )

    parser.add_argument("--print-every", type=int, default=25)

    args = parser.parse_args()

    vlim = _parse_vector(args.vlim, DEFAULT_CMD_VLIM, "--vlim")
    max_step = _parse_vector(args.max_step, DEFAULT_MAX_STEP, "--max-step")

    enable_gripper = not args.no_gripper

    # 1. 读取数据
    dataset_path = Path(args.dataset)

    if not dataset_path.exists():
        print(f"❌ 找不到数据集: {dataset_path}")
        return

    print(f"\n📂 正在加载数据集: {dataset_path}")

    qpos_data, record_rate = load_qpos_from_hdf5(
        dataset_path=str(dataset_path),
        fallback_rate=args.rate,
    )

    total_frames = len(qpos_data)
    record_dt = 1.0 / max(float(record_rate), 1e-6)

    has_gripper_data = qpos_data.shape[1] > int(args.gripper_index)

    print(f"✅ 数据加载完成: shape={qpos_data.shape}")
    print(f"✅ 总帧数: {total_frames}, 原始录制频率: {record_rate:.1f} Hz")
    print(f"✅ 是否包含夹爪数据: {has_gripper_data}")

    if enable_gripper:
        print(f"✅ 夹爪配置: {args.gripper_cfg}")

        if not args.gripper_cfg.exists():
            print(f"❌ 夹爪配置文件不存在: {args.gripper_cfg}")
            return

    # 2. 初始化硬件
    arm = None
    gripper_motor = None
    gripper_controller = None

    try:
        print("\n🤖 正在初始化达妙机械臂...")

        RobotArm = _load_robot_arm_class()
        arm = RobotArm(cfg_path=str(args.cfg) if args.cfg else None)

        arm.connect()
        arm.enable()
        arm.mode_pos_vel(vlim=vlim)

        time.sleep(0.5)

        print("✅ 机械臂已连接、使能，并进入 POS_VEL 模式")

        if enable_gripper:
            print("\n🦾 正在初始化达妙夹爪...")

            gripper_motor, gripper_controller = setup_damiao_gripper(
                arm=arm,
                gripper_cfg_path=args.gripper_cfg,
            )

            print("✅ 夹爪初始化完成")

        # 3. 状态初始化
        q_feedback_raw = np.asarray(arm.get_positions(request=True)[:6], dtype=np.float64)

        first_target_arm = _unwrap_near(qpos_data[0][:6], q_feedback_raw)
        q_feedback = _unwrap_near(q_feedback_raw, first_target_arm)

        gripper_fb = get_gripper_feedback_pos(gripper_motor) if enable_gripper else None

        if gripper_fb is None:
            gripper_fb_start = float(args.gripper_default)
        else:
            gripper_fb_start = float(gripper_fb)

        if has_gripper_data:
            first_target_gripper = float(qpos_data[0][int(args.gripper_index)])
        else:
            first_target_gripper = float(args.gripper_default)

        guard = SafetyGuard(
            max_step=max_step,
            max_tracking_error=args.max_tracking_error,
            breach_samples=args.breach_samples,
        )

        q_cmd = guard.initialize(q_feedback)
        gripper_cmd = gripper_fb_start

        # 4. 时钟配置
        prepare_duration = max(float(args.prepare_duration), 0.0)
        replay_duration = (total_frames * record_dt) / max(float(args.speed_scale), 1e-6)
        post_hold = max(float(args.post_hold), 0.0)

        cmd_period = 1.0 / max(float(args.rate), 1e-6)

        print(
            f"\n🚀 [重播启动] "
            f"speed={args.speed_scale}x | "
            f"prepare={prepare_duration:.2f}s | "
            f"replay={replay_duration:.2f}s | "
            f"post_hold={post_hold:.2f}s"
        )

        print(f"机械臂 vlim: {np.round(vlim, 4).tolist()}")
        print(f"机械臂 max_step: {np.round(max_step, 4).tolist()}")
        print(f"初始机械臂反馈: {np.round(q_feedback, 4).tolist()}")
        print(f"轨迹第 0 帧机械臂目标: {np.round(first_target_arm, 4).tolist()}")

        if enable_gripper:
            print(f"初始夹爪反馈: {gripper_fb_start:+.4f} rad")
            print(f"轨迹第 0 帧夹爪目标: {first_target_gripper:+.4f} rad")

        t_start = time.perf_counter()
        frame = 0

        last_target_q_6dof = first_target_arm.copy()
        last_target_gripper = first_target_gripper

        while _running:
            loop_start = time.perf_counter()
            elapsed = loop_start - t_start

            # A. 获取实时反馈
            q_feedback_raw = np.asarray(arm.get_positions(request=True)[:6], dtype=np.float64)
            q_feedback = _unwrap_near(q_feedback_raw, q_cmd)
            q_vel_feedback = _read_arm_velocity_or_zero(arm)

            # B. 计算轨迹目标
            if elapsed < prepare_duration:
                progress = elapsed / max(prepare_duration, 1e-6)
                smooth = _cosine_smooth(progress)

                target_q_6dof = q_feedback + (first_target_arm - q_feedback) * smooth
                target_gripper = gripper_fb_start + (first_target_gripper - gripper_fb_start) * smooth

                stage = "预引导"

            elif elapsed < prepare_duration + replay_duration:
                replay_time = elapsed - prepare_duration
                idx = min(
                    int((replay_time * float(args.speed_scale)) / record_dt),
                    total_frames - 1,
                )

                target_q_6dof = qpos_data[idx][:6]

                if has_gripper_data:
                    target_gripper = float(qpos_data[idx][int(args.gripper_index)])
                else:
                    target_gripper = float(args.gripper_default)

                stage = f"重播 idx={idx}"

            elif elapsed < prepare_duration + replay_duration + post_hold:
                target_q_6dof = last_target_q_6dof
                target_gripper = last_target_gripper
                stage = "结束保持"

            else:
                print(f"\n✅ [重播完成] elapsed={elapsed:.2f}s")
                break

            # 记录最后目标，供 post_hold 使用
            last_target_q_6dof = np.asarray(target_q_6dof, dtype=np.float64)[:6].copy()
            last_target_gripper = float(target_gripper)

            # C. SafetyGuard + 下发机械臂命令
            q_cmd = guard.next_command(target_q_6dof, q_feedback)
            arm.pos_vel(q_cmd, vlim=vlim)

            # D. 持续发送夹爪 MIT 命令
            if enable_gripper and frame % max(int(args.gripper_send_every), 1) == 0:
                gripper_cmd = float(target_gripper)

                send_damiao_gripper_mit(
                    g_mot=gripper_motor,
                    controller=gripper_controller,
                    target_rad=gripper_cmd,
                    kp=args.gripper_kp,
                    kd=args.gripper_kd,
                    tau=args.gripper_tau,
                    request_feedback=True,
                )

            # E. 打印监控
            if args.print_every > 0 and frame % args.print_every == 0:
                arm_err = np.abs(q_cmd - q_feedback)
                arm_err_max = float(np.max(arm_err))

                g_fb = get_gripper_feedback_pos(gripper_motor) if enable_gripper else None
                g_fb_str = "None" if g_fb is None else f"{g_fb:+.3f}"

                print(
                    f"[{stage}] "
                    f"t={elapsed:6.2f}s | "
                    f"J1 fb={q_feedback[0]:+.2f}, cmd={q_cmd[0]:+.2f} | "
                    f"J1 vel={q_vel_feedback[0]:+.2f} | "
                    f"max_err={arm_err_max:.3f} rad | "
                    f"gripper_cmd={gripper_cmd:+.3f}, fb={g_fb_str}"
                )

            frame += 1

            sleep_time = cmd_period - (time.perf_counter() - loop_start)

            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[Replay] 用户中断。")

    except Exception as e:
        print(f"\n❌ [异常中止] {e}")

        import traceback
        traceback.print_exc()

    finally:
        if args.no_final_disable:
            print("\n[退出流程] 已跳过 arm.disable()。")
            print("[退出流程] 机械臂/夹爪保持最后命令，交给后续 home_rebot.py 接管。")
        else:
            print("\n[退出流程] 切断硬件连接并卸载电机力矩...")
            close_arm_fast(arm)
            print("[退出流程] 机械臂已失能。")


if __name__ == "__main__":
    main()

# python replay_rebot_episodes.py --dataset /media/hjx/PSSD/hjx_ws/data/rebot/data_real/rebot_real_grasp_banana/episode_0.hdf5
