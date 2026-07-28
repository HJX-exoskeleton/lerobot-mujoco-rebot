#!/usr/bin/env python3
"""Deploy ACT with a real-time MIT joint-impedance execution layer.

Shadow mode reads real cameras/joints and runs inference without motor commands::

    python -m rebot_act_real.workflow.deploy_impedance_control \
      --checkpoint rebot_act_real/ckpt/act_rebot_real_banana_chunk50/checkpoints/last/pretrained_model \
      --shadow --temporal-ensemble --visualize

ACT remains the only source of target positions. The impedance layer sends
those targets through motor MIT position/velocity/stiffness/damping/torque
commands; it does not generate a task trajectory or replace policy inference.
"""

from __future__ import annotations

import argparse
import ctypes
import functools
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = Path(__file__).resolve().parent
MODEL_CACHE_ROOT = PROJECT_ROOT / "models"

os.environ.setdefault("HF_HOME", str(MODEL_CACHE_ROOT / ".hf_home"))
os.environ.setdefault("HF_HUB_CACHE", str(MODEL_CACHE_ROOT))
os.environ.setdefault("HF_DATASETS_CACHE", str(MODEL_CACHE_ROOT / "datasets"))
os.environ.setdefault("HF_XET_CACHE", str(MODEL_CACHE_ROOT / ".xet"))
os.environ.setdefault("HF_ASSETS_CACHE", str(MODEL_CACHE_ROOT / ".assets"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.policies import PreTrainedConfig

from rebot_scripts.Servo_control import record_rebot_episodes as camera_hw
from rebot_scripts.Servo_control import replay_rebot_episodes as robot_hw
from rebot_act_real.multimodal_policy import (
    MultimodalACTPolicy,
    is_multimodal_checkpoint,
    load_multimodal_spec,
)
from rebot_act_real.realtime_sensors import RealTimeMultimodalSensors
from rebot_act_real.realtime_inference import InferenceResult, LatestInferenceWorker
from rebot_act_real.multimodal_visualization import (
    AsyncPanelRenderer,
    add_multimodal_panel,
)
from rebot_act_real.policy_logging import PolicyRunRecorder, add_recording_arguments


DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data_act_real" / "rebot_act_banana"
DEFAULT_CHECKPOINT = (
    PACKAGE_ROOT
    / "ckpt"
    / "act_rebot_real_banana_chunk50"
    / "checkpoints"
    / "last"
    / "pretrained_model"
)
DEFAULT_ARM_CFG = WORKFLOW_ROOT / "config" / "arm.yaml"
DEFAULT_GRIPPER_CFG = WORKFLOW_ROOT / "config" / "gripper.yaml"
DEFAULT_RECORD_ROOT = PACKAGE_ROOT / "policy_logging" / "runs"
DEFAULT_VLIM = np.asarray([0.8, 0.8, 0.8, 1.2, 1.2, 1.2], dtype=np.float64)
DEFAULT_MAX_STEP = np.asarray(
    [0.015, 0.015, 0.015, 0.020, 0.020, 0.020], dtype=np.float64
)
DEFAULT_MAX_ACCEL = np.asarray(
    [2.5, 2.5, 2.5, 3.5, 3.5, 3.5], dtype=np.float64
)
DEFAULT_MIT_KP = np.asarray([10.5, 7.5, 10.5, 4.5, 3.0, 3.0], dtype=np.float64)
DEFAULT_MIT_KD = np.asarray([1.0, 2.5, 1.5, 1.5, 0.8, 0.6], dtype=np.float64)
DEFAULT_MIT_VEL_LIMIT = np.asarray(
    [0.4, 0.4, 0.4, 0.6, 0.6, 0.6], dtype=np.float64
)


def _configure_inference_backend(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
DEFAULT_GRAVITY_SCALES = np.asarray(
    [1.5, 1.0, 0.95, 0.85, 1.0, 1.0], dtype=np.float64
)
DEFAULT_TORQUE_LIMITS = np.asarray(
    [10.0, 10.0, 10.0, 5.0, 5.0, 5.0], dtype=np.float64
)
EXPECTED_INPUTS = {
    "observation.image",
    "observation.wrist_image",
    "observation.state",
}


class AccelerationLimitedCommand:
    """Make position-command velocity continuous without replacing ACT targets."""

    def __init__(self, max_velocity: np.ndarray, max_acceleration: np.ndarray):
        self.max_velocity = np.asarray(max_velocity, dtype=np.float64)
        self.max_acceleration = np.asarray(max_acceleration, dtype=np.float64)
        self.position: np.ndarray | None = None
        self.velocity = np.zeros(6, dtype=np.float64)

    def initialize(self, position: np.ndarray) -> np.ndarray:
        self.position = np.asarray(position, dtype=np.float64)[:6].copy()
        self.velocity.fill(0.0)
        return self.position.copy()

    def update(self, target: np.ndarray, dt: float) -> np.ndarray:
        if self.position is None:
            raise RuntimeError("AccelerationLimitedCommand尚未初始化")
        dt = max(float(dt), 1e-4)
        target = np.asarray(target, dtype=np.float64)[:6]
        error = target - self.position
        desired_velocity = np.clip(error / dt, -self.max_velocity, self.max_velocity)
        max_delta_velocity = self.max_acceleration * dt
        self.velocity += np.clip(
            desired_velocity - self.velocity,
            -max_delta_velocity,
            max_delta_velocity,
        )
        displacement = self.velocity * dt
        crossed = np.abs(displacement) > np.abs(error)
        displacement[crossed] = error[crossed]
        self.velocity[crossed] = 0.0
        self.position += displacement
        return self.position.copy()


class GripperCommandFilter:
    def __init__(self, *, alpha: float, deadband: float, max_step: float):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.deadband = max(float(deadband), 0.0)
        self.max_step = max(float(max_step), 0.0)
        self.command: float | None = None

    def initialize(self, command: float) -> float:
        self.command = float(command)
        return self.command

    def update(self, target: float) -> float:
        if self.command is None:
            return self.initialize(target)
        error = float(target) - self.command
        if abs(error) <= self.deadband:
            return self.command
        delta = float(np.clip(self.alpha * error, -self.max_step, self.max_step))
        self.command += delta
        return self.command


class PolicyJointImpedanceController:
    """ACT joint reference with optional gravity feedforward in MIT mode."""

    @staticmethod
    def _preload_conda_pinocchio_dependencies() -> None:
        """Prevent ROS Noetic's Python 3.8 eigenpy from entering Python 3.10."""
        python_tag = f"{sys.version_info.major}{sys.version_info.minor}"
        candidate_lib_dirs: list[Path] = []
        for entry in sys.path:
            if not entry:
                continue
            candidate = Path(entry) / "cmeel.prefix" / "lib"
            if candidate.is_dir() and candidate not in candidate_lib_dirs:
                candidate_lib_dirs.append(candidate)

        errors: list[str] = []
        for lib_dir in candidate_lib_dirs:
            boost_candidates = sorted(
                lib_dir.glob(f"libboost_python{python_tag}.so.*"),
                reverse=True,
            )
            eigenpy = lib_dir / "libeigenpy.so"
            if not boost_candidates or not eigenpy.is_file():
                continue
            try:
                ctypes.CDLL(str(boost_candidates[0]), mode=ctypes.RTLD_GLOBAL)
                ctypes.CDLL(str(eigenpy), mode=ctypes.RTLD_GLOBAL)
                return
            except OSError as exc:
                errors.append(f"{lib_dir}: {exc}")

        if errors:
            raise RuntimeError(
                "找到了CMeel Pinocchio依赖，但预加载失败：" + "；".join(errors)
            )

    def __init__(
        self,
        *,
        gravity_compensation: bool,
        gravity_scales: np.ndarray,
        torque_limits: np.ndarray,
    ):
        self.gravity_compensation = bool(gravity_compensation)
        self.gravity_scales = np.asarray(gravity_scales, dtype=np.float64)
        self.torque_limits = np.asarray(torque_limits, dtype=np.float64)
        self.model = None
        self.compute_generalized_gravity = None
        if self.gravity_compensation:
            try:
                self._preload_conda_pinocchio_dependencies()
                from reBotArm_control_py.dynamics import (
                    compute_generalized_gravity,
                    load_dynamics_model,
                )
            except Exception as exc:
                raise RuntimeError(
                    "已请求重力补偿，但reBotArm_control_py.dynamics不可用。"
                    "程序已尝试优先加载当前Python版本对应的CMeel eigenpy/Boost.Python；"
                    "请检查Pinocchio安装和动力学模型路径。"
                    "仅排查问题时可显式传入--no-gravity-compensation运行，"
                    "但不能静默伪造重力力矩。"
                ) from exc
            self.model = load_dynamics_model()
            self.compute_generalized_gravity = compute_generalized_gravity

    def compute(self, q_feedback: np.ndarray, q_target: np.ndarray) -> dict:
        q = np.asarray(q_feedback, dtype=np.float64).reshape(-1)
        target = np.asarray(q_target, dtype=np.float64).reshape(-1)[: q.size]
        tau_g = np.zeros(q.size, dtype=np.float64)
        if self.gravity_compensation:
            if self.model is None or self.compute_generalized_gravity is None:
                raise RuntimeError("重力补偿控制器未初始化")
            if q.size < self.model.nq:
                raise RuntimeError(
                    f"机械臂反馈关节数{q.size}小于动力学模型nq={self.model.nq}"
                )
            tau_raw = np.asarray(
                self.compute_generalized_gravity(
                    model=self.model, q=q[: self.model.nq]
                ),
                dtype=np.float64,
            ).reshape(-1)
            count = min(tau_raw.size, self.gravity_scales.size, tau_g.size)
            tau_g[:count] = tau_raw[:count] * self.gravity_scales[:count]
        limits = self.torque_limits[: q.size]
        tau = np.clip(tau_g, -limits, limits)
        return {"tau": tau, "tau_g": tau_g, "q_err": target - q[: target.size]}


def _device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda，但当前PyTorch无法使用CUDA")
    return torch.device(value)


def _image_tensor(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    resized = cv2.resize(np.asarray(rgb), (256, 256), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1)
    return tensor.to(device=device, dtype=torch.float32).div_(255.0).unsqueeze(0)


def _build_policy_batch(
    high_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    q_feedback: np.ndarray,
    sensor_values: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch = {
        "observation.state": torch.as_tensor(
            q_feedback, device=device, dtype=torch.float32
        ).unsqueeze(0),
        "observation.image": _image_tensor(high_rgb, device),
        "observation.wrist_image": _image_tensor(wrist_rgb, device),
    }
    for key, value in sensor_values.items():
        batch[key] = torch.as_tensor(
            value, device=device, dtype=torch.float32
        ).unsqueeze(0)
    return batch


def _select_action(
    policy: ACTPolicy,
    high_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    q_feedback: np.ndarray,
    sensor_values: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    batch = _build_policy_batch(
        high_rgb, wrist_rgb, q_feedback, sensor_values, device
    )
    with torch.inference_mode():
        action = policy.select_action(batch)[0, :7].detach().cpu().numpy()
    if action.shape != (7,) or not np.all(np.isfinite(action)):
        raise RuntimeError(f"ACT输出非法: {action}")
    return action, (time.perf_counter() - started) * 1000.0


def _read_camera(camera) -> tuple[np.ndarray, int, float]:
    image, frame_id, timestamp = camera.read()
    return np.asarray(image, dtype=np.uint8), int(frame_id), float(timestamp)


def _sample_cameras(cameras: dict[str, object]) -> dict[str, tuple]:
    return {name: _read_camera(camera) for name, camera in cameras.items()}


def _check_cameras(samples: dict[str, tuple], maximum_age_ms: float) -> None:
    now = time.perf_counter()
    problems: list[str] = []
    timestamps: list[float] = []
    for name, (_, frame_id, timestamp) in samples.items():
        age_ms = (now - timestamp) * 1000.0 if timestamp > 0 else float("inf")
        if frame_id <= 0 or timestamp <= 0:
            problems.append(f"{name}无有效帧")
        elif age_ms > maximum_age_ms:
            problems.append(f"{name}帧过期{age_ms:.0f}ms")
        timestamps.append(timestamp)
    if timestamps and (max(timestamps) - min(timestamps)) * 1000 > maximum_age_ms:
        problems.append("双相机时间偏差过大")
    if problems:
        raise RuntimeError("；".join(problems))


def _wait_for_camera_ready(
    cameras: dict[str, object], *, timeout_s: float, maximum_age_ms: float
) -> None:
    start = time.perf_counter()
    last_error = ""
    while time.perf_counter() - start < timeout_s:
        samples = _sample_cameras(cameras)
        try:
            _check_cameras(samples, maximum_age_ms)
            return
        except RuntimeError as exc:
            last_error = str(exc)
            time.sleep(0.05)
    raise RuntimeError(f"相机启动超时({timeout_s:.1f}s)：{last_error}")


def _read_stable_cameras(
    cameras: dict[str, object],
    *,
    maximum_age_ms: float,
    retry_count: int,
    retry_sleep_s: float,
) -> dict[str, tuple]:
    for attempt in range(max(retry_count, 0) + 1):
        samples = _sample_cameras(cameras)
        try:
            _check_cameras(samples, maximum_age_ms)
            return samples
        except RuntimeError:
            if attempt >= max(retry_count, 0):
                raise
            time.sleep(max(retry_sleep_s, 0.0))
    raise RuntimeError("相机读取失败")


def _draw_panel(
    high_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    state: np.ndarray,
    raw_action: np.ndarray,
    safe_action: np.ndarray,
    history: deque[dict],
    *,
    mode: str,
    step: int,
    inference_ms: float,
    queue_remaining: int,
    control_hz: float,
) -> np.ndarray:
    high = cv2.cvtColor(cv2.resize(high_rgb, (480, 360)), cv2.COLOR_RGB2BGR)
    wrist = cv2.cvtColor(cv2.resize(wrist_rgb, (480, 360)), cv2.COLOR_RGB2BGR)
    panel = np.full((700, 960, 3), 24, dtype=np.uint8)
    panel[:360, :480] = high
    panel[:360, 480:] = wrist
    lines = [
        f"ACT {mode} step={step} inference={inference_ms:.1f}ms control_hz={control_hz:.1f}",
        f"queue_remaining={queue_remaining}",
        f"state: {np.array2string(state, precision=3, suppress_small=True)}",
        f"raw:   {np.array2string(raw_action, precision=3, suppress_small=True)}",
        f"safe:  {np.array2string(safe_action, precision=3, suppress_small=True)}",
        "q / ESC: stop",
    ]
    y = 390
    for line in lines:
        cv2.putText(
            panel,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        y += 34

    if history:
        width = 900
        left = 30
        xs = np.linspace(left, left + width, len(history)).astype(np.int32)
        raw = np.stack([item["raw"] for item in history])
        safe = np.stack([item["safe"] for item in history])
        for joint in range(6):
            y0 = 600 + joint * 14
            combined = np.concatenate([raw[:, joint], safe[:, joint]])
            low, high_value = float(combined.min()), float(combined.max())
            span = max(high_value - low, 1e-4)
            raw_y = y0 + 10 - ((raw[:, joint] - low) / span * 10).astype(np.int32)
            safe_y = y0 + 10 - ((safe[:, joint] - low) / span * 10).astype(np.int32)
            cv2.polylines(
                panel,
                [np.column_stack([xs, raw_y]).astype(np.int32)],
                False,
                (70, 160, 255),
                1,
            )
            cv2.polylines(
                panel,
                [np.column_stack([xs, safe_y]).astype(np.int32)],
                False,
                (80, 220, 255),
                1,
            )
    return panel


def _render_visualization(
    high_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    state: np.ndarray,
    raw_action: np.ndarray,
    safe_action: np.ndarray,
    history: list[dict],
    sensor_values: dict[str, np.ndarray],
    *,
    mode: str,
    step: int,
    inference_ms: float,
    queue_remaining: int,
    control_hz: float,
    use_imu: bool,
    use_tactile: bool,
) -> np.ndarray:
    panel = _draw_panel(
        high_rgb,
        wrist_rgb,
        state,
        raw_action,
        safe_action,
        history,
        mode=mode,
        step=step,
        inference_ms=inference_ms,
        queue_remaining=queue_remaining,
        control_hz=control_hz,
    )
    return add_multimodal_panel(
        panel,
        sensor_values,
        use_imu=use_imu,
        use_tactile=use_tactile,
        sensor_history=history,
    )


def _close_connections_without_disabling(arm) -> None:
    if arm is None:
        return
    for controller in list(getattr(arm, "_ctrl_map", {}).values()):
        try:
            controller.shutdown()
            controller.close()
        except Exception:
            pass


def _load_policy(
    checkpoint: Path,
    metadata: LeRobotDatasetMetadata,
    device: torch.device,
    *,
    temporal_ensemble_coeff: float,
    use_imu: bool,
    use_tactile: bool,
) -> ACTPolicy:
    config = PreTrainedConfig.from_pretrained(checkpoint)
    if not isinstance(config, ACTConfig):
        raise ValueError(f"checkpoint不是ACT策略: {type(config).__name__}")
    config.device = str(device)
    # This dedicated impedance entrypoint always uses ACT's native temporal
    # ensemble. n_action_steps=1 makes select_action run a complete policy
    # inference on every control step instead of consuming an action queue.
    config.n_action_steps = 1
    config.temporal_ensemble_coeff = float(temporal_ensemble_coeff)
    config.__post_init__()
    if is_multimodal_checkpoint(checkpoint):
        spec = load_multimodal_spec(checkpoint)
        if (use_imu, use_tactile) != (spec["use_imu"], spec["use_tactile"]):
            raise ValueError(
                "部署模态与checkpoint不一致："
                f"checkpoint imu={spec['use_imu']} tactile={spec['use_tactile']}，"
                f"命令行 imu={use_imu} tactile={use_tactile}"
            )
        policy = MultimodalACTPolicy.from_multimodal_pretrained(
            checkpoint, config=config, dataset_stats=metadata.stats
        ).to(device).eval()
    else:
        if use_imu or use_tactile:
            raise ValueError("纯相机ACT checkpoint不能传入--imu或--tactile")
        policy = ACTPolicy.from_pretrained(
            checkpoint,
            config=config,
            dataset_stats=metadata.stats,
            local_files_only=True,
        ).to(device).eval()
    policy.reset()
    expected_inputs = set(EXPECTED_INPUTS)
    if use_imu or use_tactile:
        expected_inputs.add("observation.environment_state")
    if set(policy.config.input_features) != expected_inputs:
        raise ValueError(
            "checkpoint输入字段不匹配："
            f"expected={sorted(expected_inputs)}, "
            f"actual={sorted(policy.config.input_features)}"
        )
    if set(policy.config.output_features) != {"action"}:
        raise ValueError(
            f"checkpoint输出字段必须只有action: {policy.config.output_features}"
        )
    return policy


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ACT reBot实时MIT关节阻抗部署")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", default="rebot_real/rebot_act_banana")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--shadow", action="store_true", help="推理但不下发动作")
    mode.add_argument("--execute", action="store_true", help="允许下发真机动作")
    parser.add_argument("--yes", action="store_true", help="跳过DEPLOY文本确认")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--imu", action="store_true", help="使用checkpoint的IMU分支")
    parser.add_argument("--tactile", action="store_true", help="使用checkpoint的触觉分支")
    parser.add_argument("--imu-port", default="/dev/ttyUSB1")
    parser.add_argument("--tactile-port", default="/dev/ttyUSB2")
    parser.add_argument("--tactile-baud", type=int, default=2_000_000)
    parser.add_argument("--tactile-init-frames", type=int, default=30)
    parser.add_argument("--max-sensor-age-ms", type=float, default=250.0)
    parser.add_argument("--sensor-ready-timeout-s", type=float, default=15.0)
    parser.add_argument("--cfg", type=Path, default=DEFAULT_ARM_CFG)
    parser.add_argument("--gripper-cfg", type=Path, default=DEFAULT_GRIPPER_CFG)
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="控制循环最大步数；0表示不限制，例如50Hz下800步约16秒",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=1,
        help="兼容旧命令；本阻抗入口固定为1，不使用多步动作队列",
    )
    parser.add_argument(
        "--temporal-ensemble",
        action="store_true",
        help="兼容旧命令；本阻抗入口始终启用ACT原生时间集合",
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=0.01,
        help="ACT指数时间集合系数，原论文常用0.01",
    )
    parser.add_argument(
        "--impedance-control",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="使用实时MIT关节阻抗执行ACT目标（本脚本默认启用）",
    )
    parser.add_argument(
        "--mit-kp",
        type=float,
        nargs=6,
        default=None,
        help="MIT逐关节位置刚度；默认10.5 7.5 10.5 4.5 3 3",
    )
    parser.add_argument(
        "--mit-kd",
        type=float,
        nargs=6,
        default=None,
        help="MIT逐关节阻尼；默认1 2.5 1.5 1.5 0.8 0.6",
    )
    parser.add_argument(
        "--mit-vel-feedforward",
        action="store_true",
        help="按ACT命令变化计算MIT目标速度；默认关闭以避免chunk边界冲击",
    )
    parser.add_argument(
        "--mit-vel-limit",
        type=float,
        nargs=6,
        default=None,
        help="MIT速度前馈逐关节绝对限幅",
    )
    parser.add_argument(
        "--mit-kp-ramp-sec",
        type=float,
        default=1.0,
        help="MIT刚度从起始比例渐入目标值的时间，默认1秒",
    )
    parser.add_argument(
        "--mit-kp-ramp-start",
        type=float,
        default=0.7,
        help="MIT刚度渐入起始比例，默认0.7",
    )
    parser.add_argument(
        "--gravity-compensation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用Pinocchio实时重力前馈（阻抗模式默认开启；可用--no-gravity-compensation关闭）",
    )
    parser.add_argument(
        "--gravity-scales",
        type=float,
        nargs=6,
        default=None,
        help="重力补偿逐关节缩放",
    )
    parser.add_argument(
        "--torque-limits",
        type=float,
        nargs=6,
        default=None,
        help="MIT附加前馈力矩逐关节绝对限幅",
    )
    parser.add_argument("--vlim", type=float, nargs=6, default=None)
    parser.add_argument("--max-step", type=float, nargs=6, default=None)
    parser.add_argument(
        "--accel-limit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="启用最终关节命令加速度连续整形",
    )
    parser.add_argument(
        "--max-accel",
        type=float,
        nargs=6,
        default=None,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="最大命令加速度rad/s²；默认2.5 2.5 2.5 3.5 3.5 3.5",
    )
    parser.add_argument("--max-tracking-error", type=float, default=1.0)
    parser.add_argument("--tracking-breach-samples", type=int, default=20)
    parser.add_argument("--dataset-action-margin", type=float, default=0.10)
    parser.add_argument("--no-dataset-action-clip", action="store_true")
    parser.add_argument("--gripper-max-step", type=float, default=0.15)
    parser.add_argument("--gripper-filter-alpha", type=float, default=0.7)
    parser.add_argument("--gripper-deadband", type=float, default=0.01)
    parser.add_argument("--no-gripper-filter", action="store_true")
    parser.add_argument("--gripper-kp", type=float, default=1.0)
    parser.add_argument("--gripper-kd", type=float, default=0.05)
    parser.add_argument("--gripper-tau", type=float, default=0.0)
    parser.add_argument("--max-camera-age-ms", type=float, default=250.0)
    parser.add_argument("--camera-ready-timeout-s", type=float, default=10.0)
    parser.add_argument("--camera-retry-count", type=int, default=2)
    parser.add_argument("--camera-retry-sleep-ms", type=float, default=30.0)
    parser.add_argument("--visualize-history-sec", type=float, default=8.0)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--final-disable",
        action="store_true",
        help="退出时失能机械臂；默认保持使能以防下坠",
    )
    add_recording_arguments(parser, default_root=DEFAULT_RECORD_ROOT)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.record_images and not args.record:
        raise ValueError("--record-images必须与--record同时使用")
    if args.max_steps < 0:
        raise ValueError("--max-steps不能小于0")
    if args.rate <= 0:
        raise ValueError("--rate必须大于0")
    if args.temporal_ensemble_coeff <= 0:
        raise ValueError("--temporal-ensemble-coeff必须大于0")
    if args.n_action_steps != 1:
        print(
            f"[ACT] 忽略--n-action-steps={args.n_action_steps}；"
            "阻抗部署固定使用原生时间集合和n_action_steps=1。"
        )
    if args.mit_kp_ramp_sec < 0:
        raise ValueError("--mit-kp-ramp-sec不能小于0")
    if not 0.0 <= args.mit_kp_ramp_start <= 1.0:
        raise ValueError("--mit-kp-ramp-start必须在[0,1]内")
    if args.max_accel is not None and np.any(np.asarray(args.max_accel) <= 0):
        raise ValueError("--max-accel的六个值必须大于0")
    if not 0 < args.gripper_filter_alpha <= 1:
        raise ValueError("--gripper-filter-alpha必须在(0,1]内")
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(f"ACT checkpoint不存在: {args.checkpoint}")
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"LeRobot数据集不存在: {args.dataset_root}")
    if not args.cfg.is_file() or (
        not args.no_gripper and not args.gripper_cfg.is_file()
    ):
        raise FileNotFoundError("workflow/config下的机械臂或夹爪配置不存在")

    stop = False
    interrupts = 0

    def handle_signal(_signum, _frame):
        nonlocal stop, interrupts
        interrupts += 1
        stop = True
        if interrupts > 1:
            os._exit(130)
        print("\n收到退出信号，正在停止ACT策略循环。")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    device = _device(args.device)
    _configure_inference_backend(device)
    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root.resolve())
    policy = _load_policy(
        args.checkpoint.resolve(),
        metadata,
        device,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
        use_imu=args.imu,
        use_tactile=args.tactile,
    )
    action_stats = metadata.stats["action"]
    action_min = np.asarray(action_stats["min"], dtype=np.float64)
    action_max = np.asarray(action_stats["max"], dtype=np.float64)
    margin = max(float(args.dataset_action_margin), 0.0)
    lower, upper = action_min - margin, action_max + margin

    mit_kp = robot_hw._parse_vector(args.mit_kp, DEFAULT_MIT_KP, "--mit-kp")
    mit_kd = robot_hw._parse_vector(args.mit_kd, DEFAULT_MIT_KD, "--mit-kd")
    mit_vel_limit = robot_hw._parse_vector(
        args.mit_vel_limit, DEFAULT_MIT_VEL_LIMIT, "--mit-vel-limit"
    )
    gravity_scales = robot_hw._parse_vector(
        args.gravity_scales, DEFAULT_GRAVITY_SCALES, "--gravity-scales"
    )
    torque_limits = robot_hw._parse_vector(
        args.torque_limits, DEFAULT_TORQUE_LIMITS, "--torque-limits"
    )
    if np.any(mit_kp < 0) or np.any(mit_kd < 0):
        raise ValueError("--mit-kp和--mit-kd不能包含负数")
    if np.any(mit_vel_limit <= 0) or np.any(torque_limits <= 0):
        raise ValueError("--mit-vel-limit和--torque-limits必须全部大于0")

    cameras: dict[str, object] = {}
    sensors: RealTimeMultimodalSensors | None = None
    visualizer = AsyncPanelRenderer() if args.visualize else None
    inference_worker: LatestInferenceWorker | None = None
    recorder: PolicyRunRecorder | None = None
    arm = gripper_motor = gripper_controller = None
    execute = bool(args.execute)
    step = 0
    impedance_controller: PolicyJointImpedanceController | None = None
    if execute and args.impedance_control:
        # Do this before camera/arm initialization so a native dynamics failure
        # cannot leave hardware in a partially mutated state.
        impedance_controller = PolicyJointImpedanceController(
            gravity_compensation=args.gravity_compensation,
            gravity_scales=gravity_scales,
            torque_limits=torque_limits,
        )
    q_cmd: np.ndarray | None = None
    try:
        cameras["cam_high"] = camera_hw.ThreadedAstraSCamera(name="cam_high")
        cameras["cam_wrist"] = camera_hw.ThreadedRealSenseCamera(name="cam_wrist")
        sensors = RealTimeMultimodalSensors(
            use_imu=args.imu,
            use_tactile=args.tactile,
            imu_port=args.imu_port,
            tactile_port=args.tactile_port,
            tactile_baud=args.tactile_baud,
            tactile_init_frames=args.tactile_init_frames,
        )
        sensors.wait_until_ready(
            timeout_s=args.sensor_ready_timeout_s,
            maximum_age_ms=args.max_sensor_age_ms,
        )
        RobotArm = robot_hw._load_robot_arm_class()
        arm = RobotArm(cfg_path=str(args.cfg))
        arm.connect()
        if execute:
            print("\n⚠️ 即将由ACT控制真实机械臂。确认急停可用且工作空间安全。")
            if not args.yes and input("输入 DEPLOY 后回车开始：").strip() != "DEPLOY":
                print("已取消部署。")
                return
            vlim = robot_hw._parse_vector(args.vlim, DEFAULT_VLIM, "--vlim")
            max_step = robot_hw._parse_vector(
                args.max_step, DEFAULT_MAX_STEP, "--max-step"
            )
            max_accel = robot_hw._parse_vector(
                args.max_accel, DEFAULT_MAX_ACCEL, "--max-accel"
            )
            arm.enable()
            if args.impedance_control:
                arm.mode_mit(kp=mit_kp, kd=mit_kd)
            else:
                arm.mode_pos_vel(vlim=vlim)
            if not args.no_gripper:
                gripper_motor, gripper_controller = robot_hw.setup_damiao_gripper(
                    arm, args.gripper_cfg
                )
        else:
            vlim = DEFAULT_VLIM.copy()
            max_step = DEFAULT_MAX_STEP.copy()
            max_accel = DEFAULT_MAX_ACCEL.copy()

        q_feedback = np.asarray(arm.get_positions(request=True)[:6], dtype=np.float64)
        guard = robot_hw.SafetyGuard(
            max_step=max_step,
            max_tracking_error=args.max_tracking_error,
            breach_samples=args.tracking_breach_samples,
        )
        q_cmd = guard.initialize(q_feedback)
        command_filter = AccelerationLimitedCommand(
            max_velocity=np.minimum(vlim, max_step * args.rate),
            max_acceleration=max_accel,
        )
        command_filter.initialize(q_cmd)
        gripper_cmd = (
            robot_hw.get_gripper_feedback_pos(gripper_motor)
            if gripper_motor is not None
            else None
        )
        if gripper_cmd is None:
            gripper_cmd = float(np.clip(0.2, lower[6], upper[6]))
        gripper_filter = GripperCommandFilter(
            alpha=args.gripper_filter_alpha,
            deadband=args.gripper_deadband,
            max_step=args.gripper_max_step,
        )
        gripper_filter.initialize(gripper_cmd)

        _wait_for_camera_ready(
            cameras,
            timeout_s=args.camera_ready_timeout_s,
            maximum_age_ms=args.max_camera_age_ms,
        )
        initial_samples = _read_stable_cameras(
            cameras,
            maximum_age_ms=args.max_camera_age_ms,
            retry_count=args.camera_retry_count,
            retry_sleep_s=args.camera_retry_sleep_ms / 1000.0,
        )
        initial_q = np.asarray(
            arm.get_positions(request=True)[:6], dtype=np.float64
        )
        initial_sensors = (
            sensors.read(args.max_sensor_age_ms) if sensors is not None else {}
        )
        initial_action, initial_inference_ms = _select_action(
            policy,
            initial_samples["cam_high"][0],
            initial_samples["cam_wrist"][0],
            initial_q,
            initial_sensors,
            device,
        )
        if initial_action.shape != (7,) or not np.all(np.isfinite(initial_action)):
            raise RuntimeError(f"ACT首次输出非法: {initial_action}")
        inference_worker = LatestInferenceWorker(
            InferenceResult(
                action=initial_action,
                inference_ms=initial_inference_ms,
                completed_at=time.perf_counter(),
                sequence=0,
            )
        )
        period = 1.0 / args.rate
        mode_name = "EXECUTE" if execute else "SHADOW"
        impedance_start_time = time.perf_counter()
        history: deque[dict] = deque(
            maxlen=max(20, int(args.visualize_history_sec * args.rate))
        )
        loop_periods: deque[float] = deque(maxlen=max(int(args.rate), 10))
        inference_periods: deque[float] = deque(maxlen=20)
        last_inference_sequence = 0
        last_inference_completed_at = inference_worker.latest().completed_at
        previous_loop_start: float | None = None
        overrun = 0
        next_log_time = 0.0
        previous_q_for_log = initial_q.copy()
        if args.record:
            recorder = PolicyRunRecorder(
                root=args.record_root,
                run_name=args.run_name,
                entrypoint="deploy_impedance_control",
                args=args,
                project_root=PROJECT_ROOT,
                record_images=args.record_images,
                chunk_size=args.record_chunk_size,
                queue_size=args.record_queue_size,
                rate_hz=args.rate,
                extra_metadata={
                    "checkpoint": args.checkpoint.resolve(),
                    "dataset_root": args.dataset_root.resolve(),
                    "control_layer": (
                        "mit_impedance" if args.impedance_control else "position_velocity"
                    ),
                    "execute": execute,
                    "mit_kp": mit_kp,
                    "mit_kd": mit_kd,
                    "gravity_scales": gravity_scales,
                    "torque_limits": torque_limits,
                },
            )
            print(f"实验记录目录: {recorder.run_dir}")
        print(
            f"\nACT {mode_name}: rate={args.rate:g}Hz, device={device}, "
            f"chunk_size={policy.config.chunk_size}, "
            f"n_action_steps={policy.config.n_action_steps}, "
            f"temporal_ensemble={policy.config.temporal_ensemble_coeff}, "
            f"accel_limit={args.accel_limit}, "
            f"joint_layer={'MIT-impedance' if args.impedance_control else 'pos-vel'}"
        )
        if execute and args.impedance_control:
            print(
                "实时关节阻抗已启用: "
                f"kp={np.round(mit_kp, 3).tolist()}, "
                f"kd={np.round(mit_kd, 3).tolist()}, "
                f"vel_ff={args.mit_vel_feedforward}, "
                f"gravity={args.gravity_compensation}, "
                f"kp_ramp={args.mit_kp_ramp_sec:g}s"
            )
        with torch.inference_mode():
            while not stop and (args.max_steps <= 0 or step < args.max_steps):
                loop_start = time.perf_counter()
                loop_dt = (
                    period
                    if previous_loop_start is None
                    else loop_start - previous_loop_start
                )
                previous_loop_start = loop_start
                loop_periods.append(loop_dt)
                samples = _read_stable_cameras(
                    cameras,
                    maximum_age_ms=args.max_camera_age_ms,
                    retry_count=args.camera_retry_count,
                    retry_sleep_s=args.camera_retry_sleep_ms / 1000.0,
                )
                q_feedback = np.asarray(
                    arm.get_positions(request=True)[:6], dtype=np.float64
                )
                sensor_values, sensor_metadata = (
                    sensors.read_with_metadata(args.max_sensor_age_ms)
                    if sensors is not None
                    else ({}, {})
                )
                inference_worker.submit_latest(
                    functools.partial(
                        _select_action,
                        policy,
                        samples["cam_high"][0].copy(),
                        samples["cam_wrist"][0].copy(),
                        q_feedback.copy(),
                        {
                            key: value.copy()
                            for key, value in sensor_values.items()
                        },
                        device,
                    )
                )
                inference_result = inference_worker.latest()
                if inference_result.sequence != last_inference_sequence:
                    inference_periods.append(
                        inference_result.completed_at - last_inference_completed_at
                    )
                    last_inference_sequence = inference_result.sequence
                    last_inference_completed_at = inference_result.completed_at
                raw_action = inference_result.action.copy()
                inference_ms = inference_result.inference_ms
                action_age_ms = (
                    time.perf_counter() - inference_result.completed_at
                ) * 1000.0
                if raw_action.shape != (7,) or not np.all(np.isfinite(raw_action)):
                    raise RuntimeError(f"ACT输出非法: {raw_action}")

                safe_action = raw_action.astype(np.float64, copy=True)
                if not args.no_dataset_action_clip:
                    safe_action = np.clip(safe_action, lower, upper)
                impedance_debug = None
                if execute:
                    q_feedback_unwrapped = robot_hw._unwrap_near(q_feedback, q_cmd)
                    target_unwrapped = robot_hw._unwrap_near(
                        safe_action[:6], q_feedback_unwrapped
                    )
                    shaped_target = (
                        command_filter.update(target_unwrapped, loop_dt)
                        if args.accel_limit
                        else target_unwrapped
                    )
                    previous_q_cmd = q_cmd.copy()
                    q_cmd = guard.next_command(shaped_target, q_feedback_unwrapped)
                    if args.impedance_control:
                        if impedance_controller is None:
                            raise RuntimeError("MIT关节阻抗控制器未初始化")
                        mit_velocity = np.zeros(6, dtype=np.float64)
                        if args.mit_vel_feedforward:
                            mit_velocity = (q_cmd - previous_q_cmd) / max(
                                loop_dt, 1e-4
                            )
                            mit_velocity = np.clip(
                                mit_velocity, -mit_vel_limit, mit_vel_limit
                            )
                        kp_command = mit_kp
                        if args.mit_kp_ramp_sec > 0:
                            ramp = float(
                                np.clip(
                                    (
                                        time.perf_counter()
                                        - impedance_start_time
                                    )
                                    / args.mit_kp_ramp_sec,
                                    0.0,
                                    1.0,
                                )
                            )
                            kp_command = mit_kp * (
                                args.mit_kp_ramp_start
                                + (1.0 - args.mit_kp_ramp_start) * ramp
                            )
                        impedance_debug = impedance_controller.compute(
                            q_feedback_unwrapped, q_cmd
                        )
                        arm.mit(
                            pos=q_cmd,
                            vel=mit_velocity,
                            kp=kp_command,
                            kd=mit_kd,
                            tau=impedance_debug["tau"],
                            request_feedback=True,
                        )
                    else:
                        arm.pos_vel(q_cmd, vlim=vlim)
                    safe_action[:6] = q_cmd
                    if gripper_motor is not None:
                        target = (
                            float(
                                np.clip(
                                    safe_action[6],
                                    gripper_cmd - args.gripper_max_step,
                                    gripper_cmd + args.gripper_max_step,
                                )
                            )
                            if args.no_gripper_filter
                            else gripper_filter.update(safe_action[6])
                        )
                        robot_hw.send_damiao_gripper_mit(
                            gripper_motor,
                            gripper_controller,
                            target,
                            args.gripper_kp,
                            args.gripper_kd,
                            args.gripper_tau,
                            request_feedback=True,
                        )
                        gripper_cmd = target
                        safe_action[6] = target

                elapsed = time.perf_counter() - loop_start
                if elapsed > period:
                    overrun += 1
                control_hz = 1.0 / max(float(np.mean(loop_periods)), 1e-6)
                inference_hz = (
                    1.0 / max(float(np.mean(inference_periods)), 1e-6)
                    if inference_periods
                    else 0.0
                )
                queue_remaining = len(getattr(policy, "_action_queue", ()))
                if recorder is not None:
                    q_velocity = (q_feedback - previous_q_for_log) / max(loop_dt, 1e-4)
                    previous_q_for_log = q_feedback.copy()
                    record_sample = {
                        "step": step,
                        "loop_dt_s": loop_dt,
                        "joint_position": q_feedback,
                        "joint_velocity": q_velocity,
                        "raw_action": raw_action,
                        "safe_action": safe_action,
                        "tracking_error": safe_action[:6] - q_feedback,
                        "inference_ms": inference_ms,
                        "inference_sequence": inference_result.sequence,
                        "action_age_ms": action_age_ms,
                        "control_hz": control_hz,
                        "inference_hz": inference_hz,
                        "overrun_count": overrun,
                        "queue_remaining": queue_remaining,
                        "camera_frame_ids": np.asarray(
                            [samples["cam_high"][1], samples["cam_wrist"][1]],
                            dtype=np.int64,
                        ),
                        "camera_timestamp_s": np.asarray(
                            [
                                recorder.relative_timestamp(samples["cam_high"][2]),
                                recorder.relative_timestamp(samples["cam_wrist"][2]),
                            ]
                        ),
                        "sensor_frame_ids": np.asarray(
                            [
                                sensor_metadata.get("imu_frame_id", -1),
                                sensor_metadata.get("tactile_frame_id", -1),
                            ],
                            dtype=np.int64,
                        ),
                        "sensor_timestamp_s": np.asarray(
                            [
                                recorder.relative_timestamp(
                                    sensor_metadata["imu_timestamp"]
                                )
                                if "imu_timestamp" in sensor_metadata
                                else np.nan,
                                recorder.relative_timestamp(
                                    sensor_metadata["tactile_timestamp"]
                                )
                                if "tactile_timestamp" in sensor_metadata
                                else np.nan,
                            ]
                        ),
                    }
                    if "sensor.imu" in sensor_values:
                        record_sample["imu"] = sensor_values["sensor.imu"]
                    if "sensor.tactile" in sensor_values:
                        record_sample["tactile"] = sensor_values["sensor.tactile"]
                    if impedance_debug is not None:
                        record_sample["mit_tau"] = impedance_debug["tau"]
                        record_sample["gravity_tau"] = impedance_debug["tau_g"]
                        record_sample["impedance_position_error"] = impedance_debug["q_err"]
                    if args.record_images:
                        record_sample["image_high_rgb"] = samples["cam_high"][0]
                        record_sample["image_wrist_rgb"] = samples["cam_wrist"][0]
                    recorder.record(record_sample)
                if loop_start >= next_log_time:
                    print(
                        f"[{mode_name} {step:06d}] inference={inference_ms:6.1f}ms "
                        f"control_hz={control_hz:5.1f} "
                        f"infer_hz={inference_hz:5.1f} overrun={overrun} "
                        f"action_age={action_age_ms:5.1f}ms "
                        f"queue={queue_remaining:2d} "
                        f"q1={q_feedback[0]:+.3f}->{safe_action[0]:+.3f} "
                        f"gripper={safe_action[6]:+.3f}"
                        + (
                            f" tau|max|={np.max(np.abs(impedance_debug['tau'])):.2f}"
                            if impedance_debug is not None
                            else ""
                        ),
                        end="\r",
                    )
                    next_log_time = loop_start + 0.2
                if args.visualize:
                    history_item = {
                        "raw": raw_action.copy(),
                        "safe": safe_action.copy(),
                    }
                    if "sensor.imu" in sensor_values:
                        history_item["sensor.imu"] = sensor_values["sensor.imu"].copy()
                    history.append(history_item)
                    if visualizer.idle:
                        render = functools.partial(
                            _render_visualization,
                            samples["cam_high"][0].copy(),
                            samples["cam_wrist"][0].copy(),
                            q_feedback.copy(),
                            raw_action.copy(),
                            safe_action.copy(),
                            list(history),
                            {
                                key: value.copy()
                                for key, value in sensor_values.items()
                            },
                            mode=mode_name,
                            step=step,
                            inference_ms=inference_ms,
                            queue_remaining=queue_remaining,
                            control_hz=control_hz,
                            use_imu=args.imu,
                            use_tactile=args.tactile,
                        )
                        visualizer.submit_latest(render)
                    panel = visualizer.take_latest()
                    if panel is not None:
                        cv2.imshow("reBot ACT Deploy", panel)
                        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                            break
                step += 1
                delay = period - (time.perf_counter() - loop_start)
                if delay > 0:
                    time.sleep(delay)
    finally:
        if recorder is not None:
            status = "failed" if sys.exc_info()[0] is not None else (
                "completed" if step > 0 else "aborted"
            )
            run_dir = recorder.close(status=status)
            print(f"\n实验记录已保存: {run_dir}")
        if inference_worker is not None:
            inference_worker.close()
        if visualizer is not None:
            visualizer.close()
        if sensors is not None:
            sensors.release()
        for camera in cameras.values():
            camera.release()
        cv2.destroyAllWindows()
        if execute and args.final_disable:
            robot_hw.close_arm_fast(arm)
        elif execute:
            if (
                args.impedance_control
                and arm is not None
                and impedance_controller is not None
            ):
                try:
                    q_hold = np.asarray(
                        arm.get_positions(request=True)[:6], dtype=np.float64
                    )
                    hold_debug = impedance_controller.compute(q_hold, q_hold)
                    for _ in range(5):
                        arm.mit(
                            pos=q_hold,
                            vel=np.zeros(6, dtype=np.float64),
                            kp=mit_kp,
                            kd=mit_kd,
                            tau=hold_debug["tau"],
                            request_feedback=True,
                        )
                        time.sleep(0.02)
                    print(
                        "\n[安全保持] 已下发MIT当前位置阻抗保持命令"
                        + (
                            "和重力补偿。"
                            if args.gravity_compensation
                            else "（无重力前馈）。"
                        )
                    )
                except Exception as hold_exc:
                    print(f"\n⚠️ MIT退出保持命令失败: {hold_exc}")
            print(
                "\n[安全保持] ACT阻抗部署已停止，未调用arm.disable()；"
                "机械臂保持使能，请由回零程序平稳接管。"
            )
        else:
            _close_connections_without_disabling(arm)


if __name__ == "__main__":
    main()
