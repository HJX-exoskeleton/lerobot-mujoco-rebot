#!/usr/bin/env python3
"""Deploy a trained SmolVLA checkpoint on the real reBot arm.

Shadow mode performs camera/state inference without sending motor commands::

    python -m rebot_smolvla_real.workflow.deploy_new_impedance_control \
      --checkpoint rebot_smolvla_real/ckpt/smolvla_rebot_real_banana/checkpoints/last/pretrained_model \
      --shadow

Real execution requires an explicit flag and ``DEPLOY`` confirmation::

    python -m rebot_smolvla_real.workflow.deploy_new_impedance_control \
      --checkpoint rebot_smolvla_real/ckpt/smolvla_rebot_real_banana/checkpoints/last/pretrained_model \
      --instruction "抓取香蕉并放置到盘子中" \
      --execute

The policy output is already in the real dataset action space: six absolute
joint targets in radians plus the Damiao gripper motor target in radians. No
simulation gripper mapping is applied.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path

# ROS Noetic can place its Python-3.8 octomap/Boost libraries before the
# Python-3.10 CMeel libraries required by Pinocchio. LD_LIBRARY_PATH is read by
# the dynamic loader at process startup, so re-exec once before importing any
# native modules.
_CMEEL_LIB = (
    Path(sys.prefix)
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
    / "cmeel.prefix"
    / "lib"
)
if (
    __name__ == "__main__"
    and _CMEEL_LIB.is_dir()
    and os.environ.get("REBOT_CMEEL_LIB_READY") != "1"
):
    _env = os.environ.copy()
    _current_ld_path = _env.get("LD_LIBRARY_PATH", "")
    _entries = [entry for entry in _current_ld_path.split(":") if entry]
    _cmeel_text = str(_CMEEL_LIB)
    _entries = [_cmeel_text] + [entry for entry in _entries if entry != _cmeel_text]
    _env["LD_LIBRARY_PATH"] = ":".join(_entries)
    _env["REBOT_CMEEL_LIB_READY"] = "1"
    _project_root_text = str(Path(__file__).resolve().parents[2])
    _current_pythonpath = _env.get("PYTHONPATH", "")
    _env["PYTHONPATH"] = (
        f"{_project_root_text}:{_current_pythonpath}"
        if _current_pythonpath
        else _project_root_text
    )
    if __spec__ is not None and __spec__.name:
        _restart_argv = [
            sys.executable,
            "-m",
            __spec__.name,
            *sys.argv[1:],
        ]
    else:
        _restart_argv = [sys.executable, *sys.argv]
    os.execvpe(sys.executable, _restart_argv, _env)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = Path(__file__).resolve().parent
MODEL_CACHE_ROOT = PROJECT_ROOT / "models"

# Keep model/dataset lookup inside the project. These variables must be set
# before importing LeRobot and Hugging Face-backed modules.
os.environ.setdefault("HF_HOME", str(MODEL_CACHE_ROOT / ".hf_home"))
os.environ.setdefault("HF_HUB_CACHE", str(MODEL_CACHE_ROOT))
os.environ.setdefault(
    "HF_DATASETS_CACHE", str(MODEL_CACHE_ROOT / "datasets")
)
os.environ.setdefault("HF_XET_CACHE", str(MODEL_CACHE_ROOT / ".xet"))
os.environ.setdefault("HF_ASSETS_CACHE", str(MODEL_CACHE_ROOT / ".assets"))
os.environ.setdefault("TORCH_HOME", str(MODEL_CACHE_ROOT / "torch"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from rebot_scripts.Servo_control import record_rebot_episodes as camera_hw
from rebot_scripts.Servo_control import replay_rebot_episodes as robot_hw


DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data_vla_real" / "rebot_smolvla_multimodal"
DEFAULT_CHECKPOINT = (
    PACKAGE_ROOT
    / "ckpt"
    / "smolvla_rebot_real_banana"
    / "checkpoints"
    / "last"
    / "pretrained_model"
)
DEFAULT_ARM_CFG = WORKFLOW_ROOT / "config" / "arm.yaml"
DEFAULT_GRIPPER_CFG = WORKFLOW_ROOT / "config" / "gripper.yaml"
DEFAULT_VLIM = np.asarray([0.8, 0.8, 0.8, 1.2, 1.2, 1.2], dtype=np.float64)
DEFAULT_MAX_STEP = np.asarray(
    [0.015, 0.015, 0.015, 0.020, 0.020, 0.020], dtype=np.float64
)
DEFAULT_CAMERA_READY_TIMEOUT_S = 10.0
DEFAULT_CAMERA_RETRY_COUNT = 2
DEFAULT_CAMERA_RETRY_SLEEP_S = 0.03
DEFAULT_VISUALIZE_HISTORY_SEC = 8.0
DEFAULT_TEMPORAL_ENSEMBLE_K = 0.2
DEFAULT_TEMPORAL_ENSEMBLE_HISTORY = 5
DEFAULT_MAX_ACCEL = np.asarray(
    [1.0, 1.0, 0.8, 2.0, 2.0, 2.0], dtype=np.float64
)
DEFAULT_MIT_KP = np.asarray([10.5, 7.5, 10.5, 4.5, 3.0, 3.0], dtype=np.float64)
DEFAULT_MIT_KD = np.asarray([1.0, 2.5, 1.5, 1.5, 0.8, 0.6], dtype=np.float64)
DEFAULT_MIT_VEL_LIMIT = np.asarray(
    [0.4, 0.4, 0.4, 0.6, 0.6, 0.6], dtype=np.float64
)
DEFAULT_GRAVITY_SCALES = np.asarray(
    [1.5, 1.0, 0.95, 0.85, 1.0, 1.0], dtype=np.float64
)
DEFAULT_TORQUE_LIMITS = np.asarray(
    [10.0, 10.0, 10.0, 5.0, 5.0, 5.0], dtype=np.float64
)


class TemporalActionEnsembler:
    """ACT-style aggregation of overlapping action chunks for the current step.

    Chunks are stored with the control step at which they were predicted.  At
    step ``t`` only chunk entries that also target ``t`` are blended, so this
    smooths disagreements between predictions without shifting actions in time.
    """

    def __init__(self, *, decay: float, max_history: int):
        self.decay = max(float(decay), 0.0)
        self._chunks: deque[tuple[int, np.ndarray]] = deque(
            maxlen=max(int(max_history), 1)
        )

    def add(self, origin_step: int, chunk: np.ndarray) -> None:
        values = np.asarray(chunk, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] < 1:
            raise ValueError(f"动作块形状非法: {values.shape}")
        self._chunks.append((int(origin_step), values.copy()))

    def action(self, step: int) -> tuple[np.ndarray | None, int]:
        while self._chunks and self._chunks[0][0] + len(self._chunks[0][1]) <= step:
            self._chunks.popleft()

        aligned = []
        for origin, chunk in self._chunks:
            index = int(step) - origin
            if 0 <= index < len(chunk):
                aligned.append(chunk[index])
        if not aligned:
            return None, 0

        actions = np.stack(aligned, axis=0)
        # deque order is oldest -> newest. age=0 therefore belongs to newest.
        ages = np.arange(len(actions) - 1, -1, -1, dtype=np.float32)
        weights = np.exp(-self.decay * ages)
        weights /= weights.sum()
        return (actions * weights[:, None]).sum(axis=0), len(actions)


def _predict_action_chunk(
    policy: SmolVLAPolicy,
    batch: dict[str, object],
    *,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one SmolVLA inference and return its full unnormalized action chunk."""

    policy.eval()
    normalized = policy.normalize_inputs(batch)
    images, image_masks = policy.prepare_images(normalized)
    state = policy.prepare_state(normalized)
    language_tokens, language_masks = policy.prepare_language(normalized)
    actions = policy.model.sample_actions(
        images,
        image_masks,
        language_tokens,
        language_masks,
        state,
        noise=noise.clone() if noise is not None else None,
    )
    action_dim = policy.config.action_feature.shape[0]
    actions = actions[:, :, :action_dim]
    return policy.unnormalize_outputs({"action": actions})["action"]


def _next_correlated_noise(
    previous: torch.Tensor | None,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    correlation: float,
) -> torch.Tensor:
    innovation = torch.randn(shape, device=device, dtype=torch.float32)
    if previous is None or correlation <= 0.0:
        return innovation
    rho = float(np.clip(correlation, 0.0, 1.0))
    return rho * previous + float(np.sqrt(max(1.0 - rho * rho, 0.0))) * innovation


def _apply_joint_chunk_takeover(
    chunk: np.ndarray,
    previous_action: np.ndarray | None,
    steps: int,
) -> np.ndarray:
    """Blend only the first two joint targets without changing chunk timing."""

    result = np.asarray(chunk, dtype=np.float32).copy()
    if previous_action is None or steps <= 0:
        return result
    previous_joints = np.asarray(previous_action, dtype=np.float32)[:6]
    new_weights = (0.4, 0.8)
    count = min(int(steps), len(result), len(new_weights))
    for index in range(count):
        alpha = new_weights[index]
        result[index, :6] = (
            (1.0 - alpha) * previous_joints + alpha * result[index, :6]
        )
    return result


class AccelerationLimitedCommand:
    """Shape position targets while keeping commanded velocity continuous."""

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
            raise RuntimeError("AccelerationLimitedCommand 尚未 initialize")
        dt = max(float(dt), 1e-4)
        target = np.asarray(target, dtype=np.float64)[:6]
        error = target - self.position
        desired_velocity = np.clip(error / dt, -self.max_velocity, self.max_velocity)
        max_dv = self.max_acceleration * dt
        self.velocity += np.clip(
            desired_velocity - self.velocity, -max_dv, max_dv
        )
        displacement = self.velocity * dt
        # Do not cross the target when the remaining error is smaller.
        crossed = np.abs(displacement) > np.abs(error)
        displacement[crossed] = error[crossed]
        self.velocity[crossed] = 0.0
        self.position += displacement
        return self.position.copy()


class GripperCommandFilter:
    """Small deadband plus low-pass/rate limiting for the open-loop gripper."""

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
        delta = self.alpha * error
        delta = float(np.clip(delta, -self.max_step, self.max_step))
        self.command += delta
        return self.command


class PolicyJointImpedanceController:
    """SmolVLA joint reference with gravity feedforward in MIT mode."""

    def __init__(self, *, gravity_scales: np.ndarray, torque_limits: np.ndarray):
        try:
            from reBotArm_control_py.dynamics import (
                compute_generalized_gravity,
                load_dynamics_model,
            )
        except Exception as exc:
            raise RuntimeError(
                "关节阻抗控制需要可用的 reBotArm_control_py.dynamics/Pinocchio"
            ) from exc
        self.model = load_dynamics_model()
        self.compute_generalized_gravity = compute_generalized_gravity
        self.gravity_scales = np.asarray(gravity_scales, dtype=np.float64)
        self.torque_limits = np.asarray(torque_limits, dtype=np.float64)

    def compute(self, q_feedback: np.ndarray, q_target: np.ndarray) -> dict:
        q = np.asarray(q_feedback, dtype=np.float64).reshape(-1)
        if q.size < self.model.nq:
            raise RuntimeError(
                f"机械臂反馈关节数 {q.size} 小于动力学模型 nq={self.model.nq}"
            )
        tau_raw = np.asarray(
            self.compute_generalized_gravity(
                model=self.model, q=q[: self.model.nq]
            ),
            dtype=np.float64,
        ).reshape(-1)
        tau_g = np.zeros(q.size, dtype=np.float64)
        count = min(tau_raw.size, self.gravity_scales.size, tau_g.size)
        tau_g[:count] = tau_raw[:count] * self.gravity_scales[:count]
        limits = self.torque_limits[: q.size]
        tau = np.clip(tau_g, -limits, limits)
        target = np.asarray(q_target, dtype=np.float64).reshape(-1)[: q.size]
        return {
            "tau": tau,
            "tau_g": tau_g,
            "q_err": target - q[: target.size],
        }


class AsyncSmolVLAInference:
    """Latest-only inference worker; observations and results never queue up."""

    def __init__(
        self,
        policy: SmolVLAPolicy,
        device: torch.device,
        fixed_noise: torch.Tensor | None,
        noise_correlation: float,
    ):
        self.policy = policy
        self.device = device
        self.fixed_noise = fixed_noise
        self.noise_correlation = float(noise_correlation)
        self._previous_noise: torch.Tensor | None = None
        self._condition = threading.Condition()
        self._pending: tuple[int, np.ndarray, np.ndarray, np.ndarray, str] | None = None
        self._result: dict | None = None
        self._error: BaseException | None = None
        self._stopping = False
        self._busy = False
        self._thread = threading.Thread(
            target=self._run, name="smolvla-inference", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def submit(
        self,
        step: int,
        q_feedback: np.ndarray,
        high_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        instruction: str,
    ) -> None:
        request = (
            int(step),
            np.asarray(q_feedback, dtype=np.float32).copy(),
            np.asarray(high_rgb, dtype=np.uint8).copy(),
            np.asarray(wrist_rgb, dtype=np.uint8).copy(),
            str(instruction),
        )
        with self._condition:
            # Replacing the pending request is intentional: stale observations
            # must never build an inference backlog.
            self._pending = request
            self._condition.notify()

    def pop_latest(self) -> dict | None:
        with self._condition:
            result = self._result
            self._result = None
            return result

    @property
    def error(self) -> BaseException | None:
        with self._condition:
            return self._error

    @property
    def busy(self) -> bool:
        with self._condition:
            return self._busy

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            self._pending = None
            self._condition.notify()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        try:
            with torch.inference_mode():
                while True:
                    with self._condition:
                        while self._pending is None and not self._stopping:
                            self._condition.wait()
                        if self._stopping:
                            return
                        request = self._pending
                        self._pending = None
                        self._busy = True

                    submit_step, q_feedback, high_rgb, wrist_rgb, instruction = request
                    batch = {
                        "observation.state": torch.as_tensor(
                            q_feedback, device=self.device, dtype=torch.float32
                        ).unsqueeze(0),
                        "observation.image": _image_tensor(high_rgb, self.device),
                        "observation.wrist_image": _image_tensor(
                            wrist_rgb, self.device
                        ),
                        "task": [instruction],
                    }
                    infer_start = time.perf_counter()
                    inference_noise = self.fixed_noise
                    if inference_noise is None:
                        self._previous_noise = _next_correlated_noise(
                            self._previous_noise,
                            shape=(
                                1,
                                int(self.policy.config.chunk_size),
                                int(self.policy.config.max_action_dim),
                            ),
                            device=self.device,
                            correlation=self.noise_correlation,
                        )
                        inference_noise = self._previous_noise
                    chunk = (
                        _predict_action_chunk(
                            self.policy, batch, noise=inference_noise
                        )[0, :, :7]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    result = {
                        "submit_step": submit_step,
                        "chunk": chunk,
                        "inference_ms": (
                            time.perf_counter() - infer_start
                        )
                        * 1000.0,
                    }
                    with self._condition:
                        self._result = result
                        self._busy = False
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._busy = False


def _device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda，但当前 PyTorch 无法使用 CUDA")
    return torch.device(value)


def _image_tensor(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    resized = cv2.resize(np.asarray(rgb), (256, 256), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1)
    return tensor.to(device=device, dtype=torch.float32).div_(255.0).unsqueeze(0)


def _read_camera(camera) -> tuple[np.ndarray, int, float]:
    image, frame_id, timestamp = camera.read()
    return np.asarray(image, dtype=np.uint8), int(frame_id), float(timestamp)


def _sample_cameras(cameras: dict[str, object]) -> dict[str, tuple[np.ndarray, int, float]]:
    return {name: _read_camera(camera) for name, camera in cameras.items()}


def _check_cameras(samples: dict[str, tuple], maximum_age_ms: float) -> None:
    now = time.perf_counter()
    problems = []
    timestamps = []
    for name, (_, frame_id, timestamp) in samples.items():
        age_ms = (now - timestamp) * 1000.0 if timestamp > 0 else float("inf")
        if frame_id <= 0 or timestamp <= 0:
            problems.append(f"{name} 无有效帧")
        elif age_ms > maximum_age_ms:
            problems.append(f"{name} 帧过期 {age_ms:.0f}ms")
        timestamps.append(timestamp)
    if timestamps and (max(timestamps) - min(timestamps)) * 1000.0 > maximum_age_ms:
        problems.append("双相机时间偏差过大")
    if problems:
        raise RuntimeError("；".join(problems))


def _wait_for_camera_ready(
    cameras: dict[str, object],
    *,
    timeout_s: float,
    maximum_age_ms: float,
    poll_s: float = 0.05,
) -> dict[str, tuple[np.ndarray, int, float]]:
    """Block until both cameras have valid, recent frames."""

    start = time.perf_counter()
    last_error = ""
    while True:
        samples = _sample_cameras(cameras)
        try:
            _check_cameras(samples, maximum_age_ms)
            return samples
        except RuntimeError as exc:
            last_error = str(exc)

        elapsed = time.perf_counter() - start
        if elapsed >= timeout_s:
            camera_status = ", ".join(
                f"{name}:fid={frame_id},ts={timestamp:.3f}"
                for name, (_, frame_id, timestamp) in samples.items()
            )
            raise RuntimeError(
                f"相机启动超时({timeout_s:.1f}s)：{last_error}；当前状态: {camera_status}"
            ) from None
        time.sleep(poll_s)


def _read_stable_camera_samples(
    cameras: dict[str, object],
    *,
    maximum_age_ms: float,
    retry_count: int,
    retry_sleep_s: float,
) -> dict[str, tuple[np.ndarray, int, float]]:
    last_error = None
    for attempt in range(max(int(retry_count), 0) + 1):
        samples = _sample_cameras(cameras)
        try:
            _check_cameras(samples, maximum_age_ms)
            return samples
        except RuntimeError as exc:
            last_error = exc
            if attempt >= max(int(retry_count), 0):
                raise
            time.sleep(max(float(retry_sleep_s), 0.0))
    raise RuntimeError(str(last_error) if last_error is not None else "相机读取失败")


def _draw_status(
    high_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    state: np.ndarray,
    raw_action: np.ndarray,
    safe_action: np.ndarray,
    *,
    mode: str,
    step: int,
    inference_ms: float,
) -> np.ndarray:
    high = cv2.cvtColor(cv2.resize(high_rgb, (480, 360)), cv2.COLOR_RGB2BGR)
    wrist = cv2.cvtColor(cv2.resize(wrist_rgb, (480, 360)), cv2.COLOR_RGB2BGR)
    panel = np.full((590, 960, 3), 25, dtype=np.uint8)
    panel[:360, :480] = high
    panel[:360, 480:] = wrist
    lines = [
        f"{mode}  step={step}  inference={inference_ms:.1f}ms",
        f"state: {np.array2string(state, precision=3, suppress_small=True)}",
        f"raw action: {np.array2string(raw_action, precision=3, suppress_small=True)}",
        f"safe action: {np.array2string(safe_action, precision=3, suppress_small=True)}",
        "q / ESC: stop",
    ]
    y = 395
    for line in lines:
        cv2.putText(
            panel,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        y += 38
    return panel


def _value_to_y(value: float, vmin: float, vmax: float, y0: int, y1: int) -> int:
    denom = max(vmax - vmin, 1e-6)
    return int(y1 - ((float(value) - vmin) / denom * (y1 - y0)))


def _draw_curve(
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


def _make_joint_curve_panel(
    history: list[dict],
    *,
    width: int,
    height: int,
) -> np.ndarray:
    panel = np.full((height, width, 3), 18, dtype=np.uint8)
    cv2.putText(
        panel,
        "joint curves: q_feedback=white  policy_raw=orange  policy_safe=cyan",
        (14, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    if len(history) < 2:
        cv2.putText(
            panel,
            "waiting for joint history...",
            (14, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )
        return panel

    left = 54
    right = width - 16
    top = 42
    bottom_margin = 10
    row_h = max(28, (height - top - bottom_margin) // 7)
    xs = np.linspace(left, right, len(history)).astype(np.int32)

    q_fb = np.stack([item["q_feedback"] for item in history], axis=0)
    raw_action = np.stack([item["raw_action"] for item in history], axis=0)
    safe_action = np.stack([item["safe_action"] for item in history], axis=0)

    for joint_idx in range(6):
        y0 = top + joint_idx * row_h
        y1 = min(y0 + row_h - 6, height - bottom_margin)
        if y1 <= y0 + 4:
            break

        values = np.concatenate(
            [q_fb[:, joint_idx], raw_action[:, joint_idx], safe_action[:, joint_idx]]
        )
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        pad = max((vmax - vmin) * 0.15, 0.04)
        vmin -= pad
        vmax += pad

        cv2.rectangle(panel, (left, y0), (right, y1), (48, 48, 48), 1)
        mid_y = (y0 + y1) // 2
        cv2.line(panel, (left, mid_y), (right, mid_y), (34, 34, 34), 1)
        label = "G" if joint_idx == 6 else f"J{joint_idx + 1}"
        cv2.putText(
            panel,
            label,
            (14, y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"{q_fb[-1, joint_idx]:+.2f}",
            (14, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        _draw_curve(panel, xs, q_fb[:, joint_idx], vmin, vmax, y0, y1, (235, 235, 235), thickness=2)
        _draw_curve(panel, xs, raw_action[:, joint_idx], vmin, vmax, y0, y1, (70, 160, 255), thickness=1)
        _draw_curve(panel, xs, safe_action[:, joint_idx], vmin, vmax, y0, y1, (80, 220, 255), thickness=1)

    gripper_idx = 6
    y0 = top + 6 * row_h
    y1 = min(y0 + row_h - 6, height - bottom_margin)
    if y1 > y0 + 4:
        values = np.concatenate([raw_action[:, gripper_idx], safe_action[:, gripper_idx]])
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        pad = max((vmax - vmin) * 0.15, 0.02)
        vmin -= pad
        vmax += pad

        cv2.rectangle(panel, (left, y0), (right, y1), (48, 48, 48), 1)
        mid_y = (y0 + y1) // 2
        cv2.line(panel, (left, mid_y), (right, mid_y), (34, 34, 34), 1)
        cv2.putText(
            panel,
            "G",
            (14, y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"{safe_action[-1, gripper_idx]:+.2f}",
            (14, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        _draw_curve(panel, xs, raw_action[:, gripper_idx], vmin, vmax, y0, y1, (70, 160, 255), thickness=1)
        _draw_curve(panel, xs, safe_action[:, gripper_idx], vmin, vmax, y0, y1, (80, 220, 255), thickness=2)

    cv2.putText(
        panel,
        f"history {len(history)} samples",
        (width - 170, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )
    return panel


def _compose_visual_panel(
    high_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    state: np.ndarray,
    raw_action: np.ndarray,
    safe_action: np.ndarray,
    history: list[dict],
    *,
    mode: str,
    step: int,
    inference_ms: float,
) -> np.ndarray:
    status_panel = _draw_status(
        high_rgb,
        wrist_rgb,
        state,
        raw_action,
        safe_action,
        mode=mode,
        step=step,
        inference_ms=inference_ms,
    )
    curve_panel = _make_joint_curve_panel(history, width=status_panel.shape[1], height=280)
    return np.vstack([status_panel, curve_panel])


def _close_connections_without_disabling(arm) -> None:
    if arm is None:
        return
    for controller in list(getattr(arm, "_ctrl_map", {}).values()):
        try:
            controller.shutdown()
            controller.close()
        except Exception:
            pass


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SmolVLA reBot 真机部署")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--repo-id", default="rebot_real/rebot_smolvla_multimodal"
    )
    parser.add_argument(
        "--instruction", default="抓取香蕉并放置到盘子中"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--shadow", action="store_true", help="推理但不下发动作")
    mode.add_argument("--execute", action="store_true", help="允许下发真机动作")
    parser.add_argument("--yes", action="store_true", help="跳过 DEPLOY 文本确认")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cfg", type=Path, default=DEFAULT_ARM_CFG)
    parser.add_argument("--gripper-cfg", type=Path, default=DEFAULT_GRIPPER_CFG)
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--rate", type=float, default=50.0)  # 20
    parser.add_argument("--max-steps", type=int, default=0, help="0 表示不限制")
    inference_mode = parser.add_mutually_exclusive_group()
    inference_mode.add_argument(
        "--async-inference",
        dest="async_inference",
        action="store_true",
        default=True,
        help="最新观测双缓冲异步推理，控制循环不等待GPU（deploy_new默认）",
    )
    inference_mode.add_argument(
        "--sync-inference",
        dest="async_inference",
        action="store_false",
        help="恢复同步推理，用于对照诊断",
    )
    parser.add_argument(
        "--temporal-ensemble",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ACT风格重叠动作块时间集合（deploy_new默认启用）",
    )
    parser.add_argument(
        "--temporal-ensemble-k",
        type=float,
        default=DEFAULT_TEMPORAL_ENSEMBLE_K,
        help="指数衰减系数；越大越信任最新预测，默认 0.2",
    )
    parser.add_argument(
        "--temporal-ensemble-history",
        type=int,
        default=DEFAULT_TEMPORAL_ENSEMBLE_HISTORY,
        help="最多融合的预测块数，默认 5",
    )
    parser.add_argument(
        "--inference-every",
        type=int,
        default=1,
        help="每N个控制步提交最新观测；异步模式会覆盖未处理的旧请求，默认1",
    )
    parser.add_argument(
        "--chunk-takeover-steps",
        type=int,
        choices=(0, 1, 2),
        default=2,
        help="新chunk前N步做关节短接管；默认2，0关闭，不改变5步时间尺度",
    )
    parser.add_argument(
        "--gripper-latest-action",
        action="store_true",
        help="时间集合时夹爪采用最新动作块，降低夹爪开合被平滑后的响应滞后",
    )
    noise = parser.add_mutually_exclusive_group()
    noise.add_argument(
        "--fixed-noise",
        dest="fixed_noise",
        action="store_true",
        default=False,
        help="重复使用固定Flow-Matching噪声（实验选项）",
    )
    noise.add_argument(
        "--random-noise",
        dest="fixed_noise",
        action="store_false",
        help="每次推理重新采样Flow-Matching噪声（deploy_new默认）",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=42,
        help="固定推理噪声种子",
    )
    parser.add_argument(
        "--noise-correlation",
        type=float,
        default=0.7,
        help="相邻随机推理噪声相关系数[0,1)，默认0.7；0为完全独立",
    )
    parser.add_argument(
        "--impedance-control",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="使用MIT关节阻抗+实时重力补偿（本脚本默认启用）",
    )
    parser.add_argument(
        "--mit-kp",
        type=float,
        nargs=6,
        default=None,
        help="MIT关节位置刚度，默认10.5 7.5 10.5 4.5 3 3",
    )
    parser.add_argument(
        "--mit-kd",
        type=float,
        nargs=6,
        default=None,
        help="MIT关节阻尼，默认1 2.5 1.5 1.5 0.8 0.6",
    )
    parser.add_argument(
        "--mit-vel-feedforward",
        action="store_true",
        help="启用MIT目标速度前馈；默认关闭以避免chunk跳变转成速度冲击",
    )
    parser.add_argument(
        "--mit-vel-limit",
        type=float,
        nargs=6,
        default=None,
        help="MIT速度前馈逐关节限幅",
    )
    parser.add_argument(
        "--mit-kp-ramp-sec",
        type=float,
        default=1.0,
        help="MIT刚度从起始比例渐入到目标值的时间，默认1秒",
    )
    parser.add_argument(
        "--mit-kp-ramp-start",
        type=float,
        default=0.7,
        help="MIT刚度渐入起始比例，默认0.7",
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
        help="重力前馈力矩逐关节绝对限幅",
    )
    parser.add_argument("--vlim", type=float, nargs=6, default=None)
    parser.add_argument("--max-step", type=float, nargs=6, default=None)
    parser.add_argument(
        "--max-accel",
        type=float,
        nargs=6,
        default=None,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="关节命令最大加速度rad/s^2；默认3 3 3 5 5 5",
    )
    accel_mode = parser.add_mutually_exclusive_group()
    accel_mode.add_argument(
        "--accel-limit",
        dest="no_accel_limit",
        action="store_false",
        help="实验性启用新增关节加速度连续限制",
    )
    accel_mode.add_argument(
        "--no-accel-limit",
        dest="no_accel_limit",
        action="store_true",
        default=True,
        help="使用原始vlim、max-step和SafetyGuard关节执行层（deploy_new默认）",
    )
    parser.add_argument("--max-tracking-error", type=float, default=1.0)
    parser.add_argument("--tracking-breach-samples", type=int, default=20)
    parser.add_argument("--dataset-action-margin", type=float, default=0.10)
    parser.add_argument("--no-dataset-action-clip", action="store_true")
    parser.add_argument("--gripper-max-step", type=float, default=0.15)  # 0.15
    parser.add_argument(
        "--gripper-filter-alpha",
        type=float,
        default=0.7,
        help="夹爪低通系数(0,1]，越小越平滑，默认0.7",
    )
    parser.add_argument(
        "--gripper-deadband",
        type=float,
        default=0.01,
        help="夹爪目标变化死区rad，默认0.01",
    )
    parser.add_argument(
        "--no-gripper-filter",
        action="store_true",
        help="关闭夹爪低通和死区，仅使用单步限幅",
    )
    parser.add_argument("--gripper-kp", type=float, default=1.25)  # 1
    parser.add_argument("--gripper-kd", type=float, default=0.055)  # 0.05
    parser.add_argument("--gripper-tau", type=float, default=0.0)
    parser.add_argument("--max-camera-age-ms", type=float, default=250.0)
    parser.add_argument(
        "--camera-ready-timeout-s",
        type=float,
        default=DEFAULT_CAMERA_READY_TIMEOUT_S,
        help="启动后等待相机输出有效帧的最长时间",
    )
    parser.add_argument(
        "--camera-retry-count",
        type=int,
        default=DEFAULT_CAMERA_RETRY_COUNT,
        help="每轮推理前的相机重试次数",
    )
    parser.add_argument(
        "--camera-retry-sleep-ms",
        type=float,
        default=DEFAULT_CAMERA_RETRY_SLEEP_S * 1000.0,
        help="相机读取失败后的重试等待时间",
    )
    parser.add_argument(
        "--visualize-history-sec",
        type=float,
        default=DEFAULT_VISUALIZE_HISTORY_SEC,
        help="可视化窗口保留最近多少秒的关节曲线历史",
    )
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--final-disable",
        action="store_true",
        help="退出时失能机械臂；默认保持使能以防下坠",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.rate <= 0:
        raise ValueError("--rate 必须大于 0")
    if args.temporal_ensemble_k < 0:
        raise ValueError("--temporal-ensemble-k 不能小于 0")
    if args.temporal_ensemble_history <= 0:
        raise ValueError("--temporal-ensemble-history 必须大于 0")
    if args.inference_every <= 0:
        raise ValueError("--inference-every 必须大于 0")
    if not 0.0 <= args.noise_correlation < 1.0:
        raise ValueError("--noise-correlation 必须在 [0, 1) 内")
    if args.mit_kp_ramp_sec < 0:
        raise ValueError("--mit-kp-ramp-sec 不能小于 0")
    if not 0.0 <= args.mit_kp_ramp_start <= 1.0:
        raise ValueError("--mit-kp-ramp-start 必须在 [0, 1] 内")
    if args.max_accel is not None and np.any(np.asarray(args.max_accel) <= 0):
        raise ValueError("--max-accel 的6个值必须大于 0")
    if not 0 < args.gripper_filter_alpha <= 1:
        raise ValueError("--gripper-filter-alpha 必须在 (0, 1] 内")
    if args.gripper_deadband < 0:
        raise ValueError("--gripper-deadband 不能小于 0")
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(f"SmolVLA checkpoint 不存在: {args.checkpoint}")
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"LeRobot 数据集不存在: {args.dataset_root}")
    if not args.cfg.is_file() or (
        not args.no_gripper and not args.gripper_cfg.is_file()
    ):
        raise FileNotFoundError("workflow/config 下的机械臂或夹爪配置不存在")
    if not args.instruction.strip():
        raise ValueError("--instruction 不能为空")

    stop = False
    interrupts = 0

    def handle_signal(_signum, _frame):
        nonlocal stop, interrupts
        interrupts += 1
        stop = True
        if interrupts > 1:
            os._exit(130)
        print("\n收到退出信号，正在停止策略循环。")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    device = _device(args.device)
    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root.resolve())
    policy = SmolVLAPolicy.from_pretrained(
        args.checkpoint.resolve(), dataset_stats=metadata.stats
    ).to(device).eval()
    policy.reset()
    chunk_size = int(policy.config.chunk_size)
    if (
        not args.async_inference
        and args.temporal_ensemble
        and args.inference_every > chunk_size
    ):
        raise ValueError(
            f"--inference-every ({args.inference_every}) 不能大于 checkpoint "
            f"chunk_size ({chunk_size})，否则控制流会出现无动作空窗"
        )
    fixed_noise = None
    if args.fixed_noise:
        noise_generator = torch.Generator(device=device)
        noise_generator.manual_seed(int(args.noise_seed))
        fixed_noise = torch.randn(
            (
                1,
                chunk_size,
                int(policy.config.max_action_dim),
            ),
            generator=noise_generator,
            device=device,
            dtype=torch.float32,
        )
    input_keys = set(policy.config.input_features)
    expected_inputs = {
        "observation.image",
        "observation.wrist_image",
        "observation.state",
    }
    if input_keys != expected_inputs:
        raise ValueError(
            f"checkpoint 输入字段不匹配: expected={sorted(expected_inputs)}, "
            f"actual={sorted(input_keys)}"
        )

    action_stats = metadata.stats["action"]
    action_min = np.asarray(action_stats["min"], dtype=np.float64)
    action_max = np.asarray(action_stats["max"], dtype=np.float64)
    margin = max(float(args.dataset_action_margin), 0.0)
    lower = action_min - margin
    upper = action_max + margin
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
        raise ValueError("--mit-kp 和 --mit-kd 不能包含负数")
    if np.any(mit_vel_limit <= 0) or np.any(torque_limits <= 0):
        raise ValueError("--mit-vel-limit 和 --torque-limits 必须全部大于0")

    execute = bool(args.execute)
    impedance_controller: PolicyJointImpedanceController | None = None
    if execute and args.impedance_control:
        # Preflight dynamics before starting cameras, connecting, or enabling
        # the arm. A native-library failure must not mutate hardware state.
        impedance_controller = PolicyJointImpedanceController(
            gravity_scales=gravity_scales,
            torque_limits=torque_limits,
        )

    cameras = {}
    async_runner: AsyncSmolVLAInference | None = None
    arm = gripper_motor = gripper_controller = None
    q_cmd: np.ndarray | None = None
    try:
        cameras["cam_high"] = camera_hw.ThreadedAstraSCamera(name="cam_high")
        cameras["cam_wrist"] = camera_hw.ThreadedRealSenseCamera(name="cam_wrist")
        RobotArm = robot_hw._load_robot_arm_class()
        arm = RobotArm(cfg_path=str(args.cfg))
        arm.connect()
        if execute:
            print("\n⚠️ 即将由 SmolVLA 控制真实机械臂。确认急停可用且工作空间安全。")
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

        period = 1.0 / args.rate
        impedance_start_time = time.perf_counter()
        step = 0
        mode_name = "EXECUTE" if execute else "SHADOW"
        history_maxlen = max(20, int(max(float(args.visualize_history_sec), 1.0) * max(float(args.rate), 1.0)))
        joint_history: deque[dict] = deque(maxlen=history_maxlen)
        temporal_ensembler = (
            TemporalActionEnsembler(
                decay=args.temporal_ensemble_k,
                max_history=args.temporal_ensemble_history,
            )
            if args.temporal_ensemble
            else None
        )
        latest_chunk: np.ndarray | None = None
        latest_chunk_origin = -1
        last_policy_action: np.ndarray | None = None
        sync_previous_noise: torch.Tensor | None = None
        ensemble_count = 0
        previous_loop_start: float | None = None
        loop_periods: deque[float] = deque(maxlen=max(int(args.rate), 10))
        overrun_count = 0
        inference_ms = 0.0
        inference_age = -1
        print(
            f"\n{mode_name}: instruction={args.instruction!r}, "
            f"rate={args.rate:g}Hz, device={device}, "
            f"joint_layer={'original' if args.no_accel_limit else 'accel-limited'}"
        )
        if temporal_ensembler is not None:
            print(
                "时间集合已启用: "
                f"k={args.temporal_ensemble_k:g}, "
                f"history={args.temporal_ensemble_history}, "
                f"inference_every={args.inference_every}, chunk_size={chunk_size}, "
                f"noise={'fixed' if fixed_noise is not None else f'correlated({args.noise_correlation:g})'}, "
                f"takeover_steps={args.chunk_takeover_steps}"
            )
        if execute and args.impedance_control:
            print(
                "MIT关节阻抗已启用: "
                f"kp={np.round(mit_kp, 3).tolist()}, "
                f"kd={np.round(mit_kd, 3).tolist()}, "
                f"vel_ff={args.mit_vel_feedforward}"
            )
            print(
                "实时重力补偿: "
                f"scales={np.round(gravity_scales, 3).tolist()}, "
                f"tau_limits={np.round(torque_limits, 3).tolist()}, "
                f"kp_ramp={args.mit_kp_ramp_sec:g}s"
            )
        print(
            f"等待相机稳定输出，timeout={args.camera_ready_timeout_s:.1f}s, "
            f"max_age={args.max_camera_age_ms:.0f}ms"
        )
        _wait_for_camera_ready(
            cameras,
            timeout_s=args.camera_ready_timeout_s,
            maximum_age_ms=args.max_camera_age_ms,
        )
        if args.async_inference:
            async_runner = AsyncSmolVLAInference(
                policy,
                device,
                fixed_noise,
                args.noise_correlation,
            )
            async_runner.start()
            print("异步推理已启动：只保留最新观测，控制循环不会等待GPU。")
        print("相机就绪，开始推理。")
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
                filter_dt = float(np.clip(loop_dt, period * 0.5, period * 2.0))
                samples = _read_stable_camera_samples(
                    cameras,
                    maximum_age_ms=args.max_camera_age_ms,
                    retry_count=args.camera_retry_count,
                    retry_sleep_s=args.camera_retry_sleep_ms / 1000.0,
                )
                q_feedback = np.asarray(
                    arm.get_positions(request=True)[:6], dtype=np.float64
                )
                if async_runner is not None:
                    if step % args.inference_every == 0:
                        async_runner.submit(
                            step,
                            q_feedback,
                            samples["cam_high"][0],
                            samples["cam_wrist"][0],
                            args.instruction,
                        )
                    if async_runner.error is not None:
                        raise RuntimeError(
                            f"异步推理线程异常: {async_runner.error}"
                        ) from async_runner.error
                    result = async_runner.pop_latest()
                    if result is not None:
                        latest_chunk = _apply_joint_chunk_takeover(
                            result["chunk"],
                            last_policy_action,
                            args.chunk_takeover_steps,
                        )
                        # The chunk becomes executable when inference finishes.
                        # Using submit_step here would make a 175ms result stale
                        # before its first action can be consumed.
                        latest_chunk_origin = step
                        inference_ms = float(result["inference_ms"])
                        inference_age = step - int(result["submit_step"])
                        if temporal_ensembler is not None:
                            temporal_ensembler.add(step, latest_chunk)

                    if latest_chunk is None:
                        raw_action = np.empty(7, dtype=np.float32)
                        raw_action[:6] = q_feedback.astype(np.float32)
                        raw_action[6] = float(gripper_cmd)
                        ensemble_count = 0
                    else:
                        raw_action = None
                        if temporal_ensembler is not None:
                            raw_action, ensemble_count = temporal_ensembler.action(
                                step
                            )
                        if raw_action is None:
                            chunk_index = int(
                                np.clip(
                                    step - latest_chunk_origin,
                                    0,
                                    len(latest_chunk) - 1,
                                )
                            )
                            raw_action = latest_chunk[chunk_index].copy()
                            ensemble_count = 1
                        if args.gripper_latest_action:
                            chunk_index = int(
                                np.clip(
                                    step - latest_chunk_origin,
                                    0,
                                    len(latest_chunk) - 1,
                                )
                            )
                            raw_action[6] = latest_chunk[chunk_index, 6]
                else:
                    batch = {
                        "observation.state": torch.as_tensor(
                            q_feedback, device=device, dtype=torch.float32
                        ).unsqueeze(0),
                        "observation.image": _image_tensor(
                            samples["cam_high"][0], device
                        ),
                        "observation.wrist_image": _image_tensor(
                            samples["cam_wrist"][0], device
                        ),
                        "task": [args.instruction],
                    }
                    infer_start = time.perf_counter()
                    if temporal_ensembler is None:
                        raw_action = (
                            policy.select_action(batch)[0, :7]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        ensemble_count = 1
                    else:
                        if latest_chunk is None or step % args.inference_every == 0:
                            inference_noise = fixed_noise
                            if inference_noise is None:
                                sync_previous_noise = _next_correlated_noise(
                                    sync_previous_noise,
                                    shape=(
                                        1,
                                        chunk_size,
                                        int(policy.config.max_action_dim),
                                    ),
                                    device=device,
                                    correlation=args.noise_correlation,
                                )
                                inference_noise = sync_previous_noise
                            latest_chunk = (
                                _predict_action_chunk(
                                    policy, batch, noise=inference_noise
                                )[0, :, :7]
                                .detach()
                                .cpu()
                                .numpy()
                            )
                            latest_chunk_origin = step
                            temporal_ensembler.add(step, latest_chunk)
                        raw_action, ensemble_count = temporal_ensembler.action(step)
                        if raw_action is None:
                            raise RuntimeError(
                                f"step={step} 没有可用的时间对齐动作"
                            )
                        if args.gripper_latest_action:
                            latest_index = step - latest_chunk_origin
                            if 0 <= latest_index < len(latest_chunk):
                                raw_action[6] = latest_chunk[latest_index, 6]
                    inference_ms = (time.perf_counter() - infer_start) * 1000.0
                    inference_age = 0
                if raw_action.shape != (7,) or not np.all(np.isfinite(raw_action)):
                    raise RuntimeError(f"策略输出非法: {raw_action}")
                last_policy_action = raw_action.copy()
                safe_action = raw_action.astype(np.float64, copy=True)
                if not args.no_dataset_action_clip:
                    safe_action = np.clip(safe_action, lower, upper)

                impedance_debug = None
                if execute:
                    q_feedback_unwrapped = robot_hw._unwrap_near(q_feedback, q_cmd)
                    policy_target_unwrapped = robot_hw._unwrap_near(
                        safe_action[:6], q_feedback_unwrapped
                    )
                    shaped_target = (
                        policy_target_unwrapped
                        if args.no_accel_limit
                        else command_filter.update(
                            policy_target_unwrapped, filter_dt
                        )
                    )
                    previous_q_cmd = q_cmd.copy()
                    q_cmd = guard.next_command(
                        shaped_target, q_feedback_unwrapped
                    )
                    if args.impedance_control:
                        if impedance_controller is None:
                            raise RuntimeError("MIT阻抗控制器未初始化")
                        mit_velocity = np.zeros(6, dtype=np.float64)
                        if args.mit_vel_feedforward:
                            mit_velocity = (q_cmd - previous_q_cmd) / max(
                                filter_dt, 1e-4
                            )
                            mit_velocity = np.clip(
                                mit_velocity, -mit_vel_limit, mit_vel_limit
                            )
                        kp_command = mit_kp
                        if args.mit_kp_ramp_sec > 0:
                            ramp = np.clip(
                                (
                                    time.perf_counter()
                                    - impedance_start_time
                                )
                                / args.mit_kp_ramp_sec,
                                0.0,
                                1.0,
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
                        if args.no_gripper_filter:
                            gripper_target = float(
                                np.clip(
                                    safe_action[6],
                                    gripper_cmd - args.gripper_max_step,
                                    gripper_cmd + args.gripper_max_step,
                                )
                            )
                        else:
                            gripper_target = gripper_filter.update(safe_action[6])
                        robot_hw.send_damiao_gripper_mit(
                            gripper_motor,
                            gripper_controller,
                            gripper_target,
                            args.gripper_kp,
                            args.gripper_kd,
                            args.gripper_tau,
                            request_feedback=True,
                        )
                        gripper_cmd = gripper_target
                        safe_action[6] = gripper_cmd

                elapsed = time.perf_counter() - loop_start
                if elapsed > period:
                    overrun_count += 1
                control_hz = (
                    1.0 / max(float(np.mean(loop_periods)), 1e-6)
                    if loop_periods
                    else args.rate
                )
                print(
                    f"[{mode_name} {step:06d}] inference={inference_ms:6.1f}ms "
                    f"age={inference_age:3d} "
                    f"hz={control_hz:5.1f} overrun={overrun_count} "
                    f"ensemble={ensemble_count} "
                    f"q1={q_feedback[0]:+.3f}->{safe_action[0]:+.3f} "
                    f"gripper={safe_action[6]:+.3f}"
                    + (
                        f" tau|max|={np.max(np.abs(impedance_debug['tau'])):.2f}"
                        if impedance_debug is not None
                        else ""
                    ),
                    end="\r",
                )
                if args.visualize:
                    joint_history.append(
                        {
                            "q_feedback": q_feedback.copy(),
                            "raw_action": raw_action.copy(),
                            "safe_action": safe_action.copy(),
                        }
                    )
                    panel = _compose_visual_panel(
                        samples["cam_high"][0],
                        samples["cam_wrist"][0],
                        q_feedback,
                        raw_action,
                        safe_action,
                        list(joint_history),
                        mode=mode_name,
                        step=step,
                        inference_ms=inference_ms,
                    )
                    cv2.imshow("reBot SmolVLA Deploy", panel)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                step += 1
                delay = period - (time.perf_counter() - loop_start)
                if delay > 0:
                    time.sleep(delay)
    finally:
        if async_runner is not None:
            async_runner.close()
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
                    q_cmd = q_hold
                    print(
                        "\n[安全保持] 已下发MIT当前位置+重力补偿保持命令。"
                    )
                except Exception as hold_exc:
                    print(f"\n⚠️ MIT退出保持命令失败: {hold_exc}")
            print(
                "\n[安全保持] 部署已停止，未调用 arm.disable()；"
                "机械臂保持使能，请由回零程序平稳接管。"
            )
        else:
            _close_connections_without_disabling(arm)


if __name__ == "__main__":
    main()

# python -m rebot_smolvla_real.workflow.deploy_new_impedance_control --checkpoint rebot_smolvla_real/ckpt/smolvla_rebot_real_banana_chunk16/checkpoints/last/pretrained_model --instruction "抓取香蕉并放置到盘子中" --execute --inference-every 1 --rate 50 --noise-correlation 0.9 --chunk-takeover-steps 2 --mit-kp 16 16 16 12 12 12 --mit-kd 0.1 0.2 0.2 0.3 0.1 0.1 --mit-kp-ramp-sec 0 --accel-limit --max-accel 3 3 3 4 4 4 --visualize --yes
