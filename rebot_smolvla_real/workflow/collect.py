#!/usr/bin/env python3
"""reBot 主从遥操作数据采集，异步写入 LeRobot Dataset。

设计目标
--------
1. 机器人控制主循环保持 50 Hz（由 ``--rate`` 指定）。
2. 数据采样保持 50 Hz（由 ``--dataset-fps`` 指定）。
3. ``to_lerobot_frame``、``writer.add_frame``、``writer.save_episode`` 和
   ``writer.discard_episode`` 全部在独立数据线程执行，不阻塞机器人控制线程。
4. 数据写入跟不上时不阻塞控制，而是自动丢弃当前 episode，避免产生不连续、
   时间戳失真的训练数据。
5. 已有 LeRobot 数据集继续追加时，自动读取下一条连续 episode 编号；通常无需
   手动指定 ``--episode_idx``。

启动示例
--------

    python -m rebot_smolvla_real.workflow.collect \
      --repo-id rebot_real/rebot_smolvla_multimodal \
      --root ./data_vla_real/rebot_smolvla_multimodal \
      --instruction "抓取香蕉并放置到盘子中" \
      --xml asset_rebot/reBot-DevArm_gripper.xml \
      --cfg rebot_smolvla_real/workflow/config/arm.yaml \
      --gripper-cfg rebot_smolvla_real/workflow/config/gripper.yaml \
      --port /dev/ttyUSB0 \
      --imu-port /dev/ttyUSB1 \
      --tactile-port /dev/ttyUSB2 \
      --rate 50 \
      --dataset-fps 50 \
      --episode_len 800 \
      --episodes 50 \
      --episode_idx 0

运行控制
--------
- Enter：开始录制一条 episode。
- s：录制完成且后台帧处理结束后，保存当前 episode。
- d：丢弃正在录制、后台整理或等待确认的 episode。
- q：安全退出。
- Ctrl+C：第一次安全退出；若底层驱动阻塞，第二次强制退出。
"""

from __future__ import annotations

import argparse
import itertools
import os
import queue
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from tqdm import tqdm

from rebot_scripts.Servo_control import record_rebot_episodes as hw
from rebot_smolvla_real.contracts import CameraSample, RobotFeedback, RobotTarget
from rebot_smolvla_real.dataset_writer import (
    RealSmolVLADatasetWriter,
    overwrite_lerobot_dataset,
)
from rebot_smolvla_real.sample import SynchronizedSample, to_lerobot_frame


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = Path(__file__).resolve().parent
DEFAULT_XML = PROJECT_ROOT / "asset_rebot" / "reBot-DevArm_gripper.xml"
DEFAULT_ARM_CFG = WORKFLOW_ROOT / "config" / "arm.yaml"
DEFAULT_GRIPPER_CFG = WORKFLOW_ROOT / "config" / "gripper.yaml"


# ---------------------------------------------------------------------------
# 异步数据线程任务
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _StartEpisode:
    episode_idx: int
    attempt_id: int
    expected_frames: int


@dataclass(frozen=True, slots=True)
class _FrameTask:
    episode_idx: int
    attempt_id: int
    sample: SynchronizedSample


@dataclass(frozen=True, slots=True)
class _SaveEpisode:
    episode_idx: int
    attempt_id: int


@dataclass(frozen=True, slots=True)
class _DiscardEpisode:
    episode_idx: int
    attempt_id: int
    reason: str


@dataclass(frozen=True, slots=True)
class _StopWorker:
    pass


class AsyncDatasetWorker:
    """让一个专用线程独占 LeRobot writer。

    主线程只负责抓取当前传感器快照并使用 ``submit_frame`` 非阻塞入队。
    所有可能产生磁盘 I/O、视频编码或较重 Python 处理的 writer 调用均在本线程执行。
    """

    _PRIORITY_STOP = -100
    _PRIORITY_DISCARD = -50
    _PRIORITY_CONTROL = 0
    _PRIORITY_FRAME = 10

    def __init__(
        self,
        writer: RealSmolVLADatasetWriter,
        *,
        maximum_camera_skew_ms: float,
        maximum_sample_age_ms: float,
        max_pending_frames: int,
    ) -> None:
        if max_pending_frames <= 0:
            raise ValueError("max_pending_frames 必须大于 0")

        self.writer = writer
        self.maximum_camera_skew_ms = float(maximum_camera_skew_ms)
        self.maximum_sample_age_ms = float(maximum_sample_age_ms)
        self.max_pending_frames = int(max_pending_frames)

        # 使用无界 PriorityQueue，保证停止/丢弃命令一定能进入队列。
        # 帧任务通过 submit_frame 中的阈值检查实现内存上限。
        self._tasks: queue.PriorityQueue[tuple[int, int, Any]] = queue.PriorityQueue()
        self._results: queue.Queue[dict[str, Any]] = queue.Queue()
        self._sequence = itertools.count()
        self._stop_requested = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="lerobot-dataset-writer",
            daemon=True,
        )

        self._state_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending_frames = 0
        self._max_observed_pending_frames = 0
        self._active_episode: int | None = None
        self._active_attempt: int | None = None
        self._expected_frames = 0
        self._processed_frames = 0
        self._mode = "idle"
        self._cancelled_attempts: set[int] = set()

        self._thread.start()

    def _put(self, priority: int, task: Any) -> None:
        self._tasks.put_nowait((priority, next(self._sequence), task))

    def request_start(
        self,
        episode_idx: int,
        attempt_id: int,
        expected_frames: int,
    ) -> None:
        self._put(
            self._PRIORITY_CONTROL,
            _StartEpisode(int(episode_idx), int(attempt_id), int(expected_frames)),
        )

    def submit_frame(
        self,
        episode_idx: int,
        attempt_id: int,
        sample: SynchronizedSample,
    ) -> bool:
        """非阻塞提交一帧。

        返回 False 表示后台写入已经积压。调用方应立即停止当前 episode 并请求丢弃，
        不能等待队列腾空，否则会直接破坏 50 Hz 控制周期。
        """
        if self._stop_requested.is_set():
            return False

        # 只统计尚未处理的帧任务，不把 start/save/discard 等控制命令算入容量。
        with self._pending_lock:
            if self._pending_frames >= self.max_pending_frames:
                return False
            self._pending_frames += 1
            self._max_observed_pending_frames = max(
                self._max_observed_pending_frames,
                self._pending_frames,
            )

        try:
            self._put(
                self._PRIORITY_FRAME,
                _FrameTask(int(episode_idx), int(attempt_id), sample),
            )
        except BaseException:
            with self._pending_lock:
                self._pending_frames = max(0, self._pending_frames - 1)
            raise
        return True

    def request_save(self, episode_idx: int, attempt_id: int) -> None:
        self._put(
            self._PRIORITY_CONTROL,
            _SaveEpisode(int(episode_idx), int(attempt_id)),
        )

    def request_discard(
        self,
        episode_idx: int,
        attempt_id: int,
        reason: str,
    ) -> None:
        # 高优先级丢弃会先标记该 episode 已取消；队列中尚未处理的旧帧随后会被跳过。
        self._put(
            self._PRIORITY_DISCARD,
            _DiscardEpisode(int(episode_idx), int(attempt_id), str(reason)),
        )

    def request_stop(self) -> None:
        if not self._stop_requested.is_set():
            self._stop_requested.set()
            self._put(self._PRIORITY_STOP, _StopWorker())

    def poll_results(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                return results

    def pending_frames(self) -> int:
        with self._pending_lock:
            return self._pending_frames

    def max_observed_pending_frames(self) -> int:
        with self._pending_lock:
            return self._max_observed_pending_frames

    def pending_tasks(self) -> int:
        # 保留总任务数接口用于退出/清理判断。
        return self._tasks.qsize()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)

    def _emit(self, kind: str, **payload: Any) -> None:
        self._results.put({"kind": kind, **payload})

    def _set_state(
        self,
        *,
        mode: str,
        active_episode: int | None,
        active_attempt: int | None = None,
        expected_frames: int = 0,
        processed_frames: int = 0,
    ) -> None:
        with self._state_lock:
            self._mode = mode
            self._active_episode = active_episode
            self._active_attempt = active_attempt
            self._expected_frames = expected_frames
            self._processed_frames = processed_frames

    def _safe_discard_writer_buffer(self) -> str | None:
        try:
            if bool(getattr(self.writer, "frames_in_buffer", False)):
                self.writer.discard_episode()
            return None
        except Exception as exc:  # noqa: BLE001 - 硬件采集程序需报告第三方库异常
            return repr(exc)

    def _handle_start(self, task: _StartEpisode) -> None:
        if task.expected_frames <= 0:
            self._emit(
                "fatal",
                episode_idx=task.episode_idx,
                error="expected_frames 必须大于 0",
            )
            return

        if task.attempt_id in self._cancelled_attempts:
            return

        if self._active_episode is not None or self._active_attempt is not None or self._mode != "idle":
            self._emit(
                "fatal",
                episode_idx=task.episode_idx,
                error=(
                    f"数据线程状态冲突：mode={self._mode}, "
                    f"active_episode={self._active_episode}, "
                    f"active_attempt={self._active_attempt}"
                ),
            )
            return

        cleanup_error = self._safe_discard_writer_buffer()
        if cleanup_error is not None:
            self._emit(
                "fatal",
                episode_idx=task.episode_idx,
                error=f"清理旧 writer 缓冲区失败：{cleanup_error}",
            )
            return

        self._set_state(
            mode="recording",
            active_episode=task.episode_idx,
            active_attempt=task.attempt_id,
            expected_frames=task.expected_frames,
            processed_frames=0,
        )
        self._emit(
            "started",
            episode_idx=task.episode_idx,
            attempt_id=task.attempt_id,
            expected_frames=task.expected_frames,
        )

    def _fail_episode(
        self,
        episode_idx: int,
        attempt_id: int,
        reason: str,
    ) -> None:
        self._cancelled_attempts.add(attempt_id)
        discard_error = self._safe_discard_writer_buffer()
        self._set_state(mode="idle", active_episode=None, active_attempt=None)
        if discard_error:
            reason = f"{reason}；同时清理 writer 缓冲区失败：{discard_error}"
        self._emit(
            "failed",
            episode_idx=episode_idx,
            attempt_id=attempt_id,
            reason=reason,
        )

    def _handle_frame(self, task: _FrameTask) -> None:
        if task.attempt_id in self._cancelled_attempts:
            return
        if (
            self._mode != "recording"
            or self._active_episode != task.episode_idx
            or self._active_attempt != task.attempt_id
        ):
            # 丢弃命令可能先于旧帧执行，这是正常情况；其他状态则报告异常。
            if self._mode == "idle":
                return
            self._emit(
                "fatal",
                episode_idx=task.episode_idx,
                error=(
                    f"收到不属于当前 episode 的帧：active={self._active_episode}, "
                    f"incoming={task.episode_idx}, mode={self._mode}, "
                    f"active_attempt={self._active_attempt}, "
                    f"incoming_attempt={task.attempt_id}"
                ),
            )
            return

        started = time.perf_counter()
        try:
            dataset_frame = to_lerobot_frame(
                task.sample,
                maximum_camera_skew_ms=self.maximum_camera_skew_ms,
                maximum_sample_age_ms=self.maximum_sample_age_ms,
            )
            packed_at = time.perf_counter()
            self.writer.add_frame(dataset_frame)
            finished = time.perf_counter()
        except RuntimeError as exc:
            self._fail_episode(
                task.episode_idx,
                task.attempt_id,
                f"相机/传感器同步失效：{exc}",
            )
            return
        except Exception as exc:  # noqa: BLE001
            self._fail_episode(
                task.episode_idx,
                task.attempt_id,
                f"add_frame 失败：{type(exc).__name__}: {exc}",
            )
            return

        self._processed_frames += 1
        if self._processed_frames >= self._expected_frames:
            episode_idx = task.episode_idx
            self._set_state(
                mode="review",
                active_episode=episode_idx,
                active_attempt=task.attempt_id,
                expected_frames=self._expected_frames,
                processed_frames=self._processed_frames,
            )
            self._emit(
                "ready",
                episode_idx=episode_idx,
                attempt_id=task.attempt_id,
                processed_frames=self._processed_frames,
                final_pack_ms=(packed_at - started) * 1000.0,
                final_add_ms=(finished - packed_at) * 1000.0,
            )

    def _handle_save(self, task: _SaveEpisode) -> None:
        if (
            self._mode != "review"
            or self._active_episode != task.episode_idx
            or self._active_attempt != task.attempt_id
        ):
            self._emit(
                "fatal",
                episode_idx=task.episode_idx,
                error=(
                    f"当前 episode 尚未进入可保存状态：mode={self._mode}, "
                    f"active={self._active_episode}, "
                    f"active_attempt={self._active_attempt}"
                ),
            )
            return

        self._set_state(
            mode="saving",
            active_episode=task.episode_idx,
            active_attempt=task.attempt_id,
            expected_frames=self._expected_frames,
            processed_frames=self._processed_frames,
        )
        started = time.perf_counter()
        try:
            self.writer.save_episode()
        except Exception as exc:  # noqa: BLE001
            self._emit(
                "fatal",
                episode_idx=task.episode_idx,
                attempt_id=task.attempt_id,
                error=f"save_episode 失败：{type(exc).__name__}: {exc}",
            )
            return

        elapsed = time.perf_counter() - started
        self._cancelled_attempts.discard(task.attempt_id)
        self._set_state(mode="idle", active_episode=None, active_attempt=None)
        self._emit(
            "saved",
            episode_idx=task.episode_idx,
            attempt_id=task.attempt_id,
            elapsed_s=elapsed,
        )

    def _handle_discard(self, task: _DiscardEpisode) -> None:
        self._cancelled_attempts.add(task.attempt_id)

        # 若 StartTask 还没有执行，active_episode 可能为 None。之后 StartTask 会看到
        # cancelled 标记并直接跳过，不会创建新的 writer 缓冲区。
        if (
            self._active_episode == task.episode_idx
            and self._active_attempt == task.attempt_id
        ):
            discard_error = self._safe_discard_writer_buffer()
            self._set_state(mode="idle", active_episode=None, active_attempt=None)
        else:
            discard_error = None

        reason = task.reason
        if discard_error:
            reason = f"{reason}；清理 writer 缓冲区失败：{discard_error}"
        self._emit(
            "discarded",
            episode_idx=task.episode_idx,
            attempt_id=task.attempt_id,
            reason=reason,
        )

    def _run(self) -> None:
        try:
            while True:
                _priority, _sequence, task = self._tasks.get()
                try:
                    if isinstance(task, _StopWorker):
                        cleanup_error = self._safe_discard_writer_buffer()
                        self._set_state(
                            mode="stopped",
                            active_episode=None,
                            active_attempt=None,
                        )
                        self._emit("stopped", cleanup_error=cleanup_error)
                        return
                    if isinstance(task, _DiscardEpisode):
                        self._handle_discard(task)
                    elif isinstance(task, _StartEpisode):
                        self._handle_start(task)
                    elif isinstance(task, _FrameTask):
                        self._handle_frame(task)
                    elif isinstance(task, _SaveEpisode):
                        self._handle_save(task)
                    else:
                        self._emit("fatal", error=f"未知数据线程任务：{type(task)!r}")
                finally:
                    if isinstance(task, _FrameTask):
                        with self._pending_lock:
                            self._pending_frames = max(0, self._pending_frames - 1)
                    self._tasks.task_done()
        except BaseException as exc:  # noqa: BLE001
            self._emit(
                "fatal",
                error=f"数据线程异常退出：{type(exc).__name__}: {exc}",
            )


# ---------------------------------------------------------------------------
# 参数与硬件辅助函数
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    parser = hw.build_argparser()
    parser.description = "reBot 主从遥操作 -> LeRobot 多模态语言数据集（异步写入）"
    parser.set_defaults(
        xml=DEFAULT_XML,
        cfg=DEFAULT_ARM_CFG,
        gripper_cfg=DEFAULT_GRIPPER_CFG,
        rate=50.0,
        episode_len=800,
    )
    parser.add_argument("--repo-id", default="rebot_real/rebot_smolvla_multimodal")
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "data_vla_real" / "rebot_smolvla_multimodal",
        help="LeRobotDataset 本地根目录",
    )
    parser.add_argument(
        "--instruction",
        default="抓取香蕉并放置到盘子中。",
        help="写入每一帧 task 字段的自然语言指令",
    )
    parser.add_argument("--dataset-fps", type=int, default=50)
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="本次成功保存数量；0 表示不限制，不是数据集最终总数",
    )
    parser.add_argument(
        "--overwrite-dataset",
        action="store_true",
        help="删除 --root 下已有 LeRobot 数据集并从 episode 0 重新采集",
    )
    parser.add_argument("--max-camera-skew-ms", type=float, default=80.0)
    parser.add_argument("--max-sample-age-ms", type=float, default=250.0)
    parser.add_argument(
        "--writer-queue-size",
        type=int,
        default=0,
        help=(
            "后台待处理帧上限；0 表示按当前 episode_len 自动设置为完整轨迹容量。"
            "达到上限时自动丢弃当前 episode，而不阻塞控制线程"
        ),
    )
    parser.add_argument(
        "--velocity-read-every",
        type=int,
        default=1,
        help="每 N 个控制周期主动请求一次关节速度；默认 1 保持原始 50 Hz 读取",
    )
    parser.add_argument(
        "--loop-warning-ms",
        type=float,
        default=22.0,
        help="单次控制循环超过该耗时时计入超时统计",
    )
    parser.add_argument(
        "--camera-copy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "入队前复制相机帧，防止相机线程覆盖底层缓冲区。"
            "仅当 camera.read() 已确认返回独立副本时才使用 --no-camera-copy"
        ),
    )

    for action in parser._actions:
        if action.dest == "episode_idx":
            action.help = (
                "手动指定本次第一条 episode 编号；必须等于数据集下一连续编号。"
                "通常建议省略，由程序自动读取"
            )
    return parser


def _stdin_worker(commands: queue.Queue[str]) -> None:
    while hw._running:
        try:
            commands.put(input().strip().lower())
        except (EOFError, KeyboardInterrupt):
            commands.put("q")
            return


def _read_camera(camera: Any, *, copy_frame: bool) -> CameraSample:
    rgb, frame_id, timestamp = camera.read()
    if copy_frame:
        rgb_array = np.array(rgb, dtype=np.uint8, copy=True, order="C")
    else:
        rgb_array = np.asarray(rgb, dtype=np.uint8)
    return CameraSample(
        rgb=rgb_array,
        frame_id=int(frame_id),
        timestamp=float(timestamp),
    )


def _camera_problem(
    cameras: dict[str, Any],
    *,
    maximum_age_ms: float,
) -> str | None:
    now = time.perf_counter()
    problems: list[str] = []
    for name, camera in cameras.items():
        # 就绪检查不进入异步队列，因此无需复制图像。
        sample = _read_camera(camera, copy_frame=False)
        valid = bool(getattr(camera, "valid", True))
        age_ms = (
            (now - sample.timestamp) * 1000.0
            if sample.timestamp > 0
            else float("inf")
        )
        if not valid or sample.frame_id <= 0 or sample.timestamp <= 0:
            problems.append(f"{name}: 无有效帧")
        elif age_ms > maximum_age_ms:
            problems.append(f"{name}: 最新帧已过期 {age_ms:.0f}ms")
    return "；".join(problems) if problems else None


def _zero_imu(now: float) -> dict[str, Any]:
    return {
        "imu_left": np.zeros(hw.IMU_LEFT_DIM, dtype=np.float32),
        "mag": np.zeros(3, dtype=np.float32),
        "euler": np.zeros(3, dtype=np.float32),
        "baro": np.zeros(4, dtype=np.float32),
        "frame_id": -1,
        "timestamp": now,
    }


def _print_controls() -> None:
    print(
        "\n操作命令：\n"
        "  [Enter] 开始一条 episode\n"
        "  s       完成后保存当前 episode\n"
        "  d       丢弃当前 episode\n"
        "  q       安全退出\n"
    )


def _copy_float32(values: Any) -> np.ndarray:
    return np.array(values, dtype=np.float32, copy=True, order="C")


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------


def main() -> None:
    args = build_argparser().parse_args()

    if args.rate <= 0:
        raise ValueError("--rate 必须大于 0")
    if args.dataset_fps <= 0 or args.dataset_fps > args.rate:
        raise ValueError("--dataset-fps 必须大于 0 且不能高于 --rate")
    if args.episodes < 0:
        raise ValueError("--episodes 不能为负数")
    if args.writer_queue_size < 0:
        raise ValueError("--writer-queue-size 不能为负数；0 表示自动设置")
    if args.velocity_read_every <= 0:
        raise ValueError("--velocity-read-every 必须大于 0")
    if not args.xml.is_file():
        raise FileNotFoundError(f"MuJoCo XML 不存在: {args.xml}")
    if args.cfg is None or not args.cfg.is_file():
        raise FileNotFoundError(f"RobotArm 配置不存在: {args.cfg}")
    if not args.no_gripper and not args.gripper_cfg.is_file():
        raise FileNotFoundError(f"夹爪配置不存在: {args.gripper_cfg}")
    if not args.instruction.strip():
        raise ValueError("--instruction 不能为空")
    if args.time is not None:
        if args.time <= 0:
            raise ValueError("--time 必须大于 0")
        args.episode_len = int(round(args.time * args.dataset_fps))
    if args.episode_len <= 0:
        raise ValueError("--episode_len 必须大于 0")

    # 50 Hz 双相机写入在部分机器上可能长期慢于采集。默认允许缓存完整
    # episode，使控制线程始终非阻塞；采样结束后再等待后台整理完成。
    if args.writer_queue_size == 0:
        args.writer_queue_size = max(args.episode_len + 8, 64)

    stop_event = threading.Event()
    interrupt_count = 0

    def handle_interrupt(_signum: int, _frame: Any) -> None:
        nonlocal interrupt_count
        interrupt_count += 1
        hw._running = False
        stop_event.set()
        if interrupt_count == 1:
            print(
                "\n[退出] 收到 Ctrl+C，正在停止控制并释放硬件；"
                "若驱动阻塞可再次按 Ctrl+C。"
            )
        else:
            print("\n[急停退出] 驱动未能及时返回，立即终止进程。")
            os._exit(130)

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)
    hw._running = True

    enable_gripper = not args.no_gripper
    servo_to_sim_sign = hw._parse_vector(
        args.servo_to_sim_signs,
        hw.DEFAULT_SERVO_TO_SIM_SIGN,
        "servo-to-sim-signs",
    )
    sim_home_rad = hw._parse_vector(
        args.sim_home,
        hw.DEFAULT_SIM_HOME_RAD,
        "sim-home",
    )
    signs = hw._parse_vector(args.signs, np.ones(6), "signs")
    offsets = hw._parse_vector(args.offsets, np.zeros(6), "offsets")
    vlim = hw._parse_vector(args.vlim, hw.DEFAULT_CMD_VLIM, "vlim")
    max_step = hw._parse_vector(args.max_step, hw.DEFAULT_MAX_STEP, "max-step")
    joint_names = tuple(x.strip() for x in args.joint_names.split(",") if x.strip())

    if args.overwrite_dataset:
        if args.episode_idx not in (None, 0):
            raise ValueError(
                "--overwrite-dataset 会创建空数据集，--episode_idx 只能省略或设为 0"
            )
        if overwrite_lerobot_dataset(args.root):
            print(f"🗑️ 已按 --overwrite-dataset 删除旧数据集: {args.root}")

    writer = RealSmolVLADatasetWriter(
        repo_id=args.repo_id,
        root=args.root,
        instruction=args.instruction,
        fps=args.dataset_fps,
    )
    next_dataset_episode = int(writer.dataset.meta.total_episodes)
    if args.episode_idx is None:
        current_episode_idx = next_dataset_episode
    else:
        if args.episode_idx < 0:
            raise ValueError("--episode_idx 不能为负数")
        if args.episode_idx != next_dataset_episode:
            raise ValueError(
                f"--episode_idx={args.episode_idx} 与数据集下一编号 "
                f"{next_dataset_episode} 不一致。LeRobot episode 必须从 0 连续编号，"
                "请修改参数或省略 --episode_idx。"
            )
        current_episode_idx = int(args.episode_idx)

    dataset_worker = AsyncDatasetWorker(
        writer,
        maximum_camera_skew_ms=args.max_camera_skew_ms,
        maximum_sample_age_ms=args.max_sample_age_ms,
        max_pending_frames=args.writer_queue_size,
    )

    cameras: dict[str, Any] = {
        "cam_high": hw.ThreadedAstraSCamera(name="cam_high"),
        "cam_wrist": hw.ThreadedRealSenseCamera(name="cam_wrist"),
    }
    imu_reader = None if args.no_imu else hw.ThreadedYbImuReader(
        port=args.imu_port,
        report_rate=args.imu_report_rate,
        alpha=args.imu_alpha,
        name="imu_left",
    )
    tactile_reader = None if args.no_tactile else hw.ThreadedFlexiTacReader(
        port=args.tactile_port,
        baud=args.tactile_baud,
        init_frames=args.tactile_init_frames,
        threshold=args.tactile_threshold,
        noise_scale=args.tactile_noise_scale,
        alpha=args.tactile_alpha,
        name="gripper_tactile",
    )

    port_handler = None
    arm = None
    progress_bar: tqdm | None = None
    commands: queue.Queue[str] = queue.Queue()

    saved_episodes = 0
    recording = False
    finalizing = False
    awaiting_decision = False
    # 用户可在后台尚未处理完全部帧时提前选择保存。
    # 此时只记录意图，待 worker 发出 ready 后再真正调用 save_episode()。
    save_after_finalize = False
    saving = False
    discarding = False
    episode_samples = 0
    attempt_ids = itertools.count(1)
    active_attempt_id: int | None = None

    # 上次有效反馈用于硬件读取偶发失败或可选降频读取。
    last_v_feedback = np.zeros(6, dtype=np.float64)
    last_gripper_fb_pos = 0.0
    last_gripper_fb_vel = 0.0

    # 控制循环性能统计。
    control_frames = 0
    deadline_misses = 0
    slow_loops = 0
    max_loop_ms = 0.0
    stats_window_start = time.perf_counter()

    try:
        model = mujoco.MjModel.from_xml_path(str(args.xml))
        port_handler = hw.PortHandler(args.port)
        scs = hw.sts(port_handler)
        if not port_handler.openPort():
            raise RuntimeError(f"主手串口打开失败: {args.port}")
        if not port_handler.setBaudRate(args.baudrate):
            raise RuntimeError(f"主手波特率设置失败: {args.baudrate}")

        time.sleep(2.5)
        if not args.keep_servo_torque:
            ids = hw.ARM_SERVO_IDS + (
                [hw.GRIPPER_SERVO_ID] if enable_gripper else []
            )
            hw.release_servo_torque(scs, ids)

        home_arm_deg = np.asarray(
            [hw.JOINT_LIMITS_DEG[i]["home_deg"] for i in hw.ARM_SERVO_IDS],
            dtype=np.float64,
        )
        home_q_sim = hw.servo_deg_array_to_sim_rad(
            home_arm_deg,
            servo_to_sim_sign,
            sim_home_rad,
        )
        home_gripper = hw.gripper_norm_to_real_rad(
            hw.gripper_servo_deg_to_norm(
                hw.JOINT_LIMITS_DEG[hw.GRIPPER_SERVO_ID]["home_deg"],
                args.invert_gripper,
            ),
            args.gripper_real_closed_rad,
            args.gripper_real_open_rad,
        )

        state_lock = threading.Lock()
        shared_state = {
            "arm_deg": home_arm_deg.copy(),
            "target_q_sim": home_q_sim.copy(),
            "gripper_deg": float(
                hw.JOINT_LIMITS_DEG[hw.GRIPPER_SERVO_ID]["home_deg"]
            ),
            "gripper_norm": 1.0,
            "gripper_target_rad": float(home_gripper),
            "success_count": 0,
            "failed_ids": [],
            "timestamp": time.perf_counter(),
            "read_frame": 0,
        }
        threading.Thread(
            target=hw.servo_reader_worker,
            args=(
                scs,
                state_lock,
                shared_state,
                args.read_rate,
                servo_to_sim_sign,
                sim_home_rad,
                enable_gripper,
                args.invert_gripper,
                args.gripper_real_closed_rad,
                args.gripper_real_open_rad,
            ),
            daemon=True,
            name="master-servo-reader",
        ).start()

        if not hw.wait_for_servo_ready(
            state_lock,
            shared_state,
            hw.ARM_DOF + (1 if enable_gripper else 0),
        ):
            raise RuntimeError("主手舵机数据未就绪")

        with state_lock:
            initial_q_sim = shared_state["target_q_sim"].copy()
            initial_gripper = float(shared_state["gripper_target_rad"])

        RobotArm = hw._load_robot_arm_class()
        arm = RobotArm(cfg_path=str(args.cfg) if args.cfg else None)
        arm.connect()
        arm.enable()
        arm.mode_pos_vel(vlim=vlim)

        gripper_motor = None
        gripper_controller = None
        if enable_gripper:
            gripper_motor, gripper_controller, _ = hw.setup_damiao_gripper(
                arm,
                args.gripper_cfg,
            )

        q_feedback = hw.read_stable_positions(
            arm,
            hw._sim_to_real_unclipped(initial_q_sim, signs, offsets),
            args.settle_samples,
            args.settle_interval,
        )
        if args.calibrate_current_as_master:
            offsets = initial_q_sim.copy() - signs * q_feedback[:6]

        mapper = hw.SimToRealMapper(
            model,
            joint_names,
            signs,
            offsets,
            args.soft_margin,
        )
        guard = hw.SafetyGuard(
            mapper,
            max_step,
            args.max_start_error,
            args.max_tracking_error,
            args.tracking_breach_samples,
        )
        q_cmd = guard.initialize(
            q_feedback,
            mapper.sim_to_real(initial_q_sim),
            args.allow_large_start,
        )
        arm.pos_vel(q_cmd, vlim=vlim)

        filtered_q_sim = initial_q_sim.copy()
        filtered_gripper = initial_gripper
        if enable_gripper:
            hw.send_damiao_gripper_mit(
                gripper_motor,
                gripper_controller,
                filtered_gripper,
                args.gripper_kp,
                args.gripper_kd,
                args.gripper_tau,
            )

        threading.Thread(
            target=_stdin_worker,
            args=(commands,),
            daemon=True,
            name="stdin-command-reader",
        ).start()

        print(f"\nLeRobot 数据集: {args.root}")
        print(f"语言任务: {args.instruction}")
        print(f"控制/采样频率: {args.rate:g}/{args.dataset_fps} Hz")
        print(
            f"每条轨迹: {args.episode_len} 帧 "
            f"({args.episode_len / args.dataset_fps:.1f}s)"
        )
        print(
            f"后台帧队列上限: {args.writer_queue_size} "
            f"({args.writer_queue_size / args.dataset_fps:.2f}s 原始采样容量)"
        )
        print(f"下一条 episode 编号: {current_episode_idx}")
        _print_controls()

        control_period = 1.0 / float(args.rate)
        sample_period = 1.0 / float(args.dataset_fps)
        next_control_deadline = time.perf_counter()
        next_sample_time = next_control_deadline
        frame = 0

        while hw._running:
            loop_start = time.perf_counter()

            # ---------------------------------------------------------------
            # 1. 处理数据线程结果。所有耗时 writer 操作均已在后台完成。
            # ---------------------------------------------------------------
            for result in dataset_worker.poll_results():
                kind = result["kind"]
                result_episode = result.get("episode_idx")
                result_attempt = result.get("attempt_id")

                if kind == "fatal":
                    raise RuntimeError(result.get("error", "数据线程未知错误"))

                if (
                    result_attempt is not None
                    and result_attempt != active_attempt_id
                ):
                    continue

                if kind == "started":
                    pass
                elif kind == "ready" and result_episode == current_episode_idx:
                    if not discarding:
                        finalizing = False
                        if save_after_finalize:
                            # 用户已在采样结束时选择保存。此时 worker 已处理完
                            # 全部帧，可以安全提交真正的 save_episode 任务。
                            save_after_finalize = False
                            awaiting_decision = False
                            saving = True
                            if active_attempt_id is None:
                                raise RuntimeError("缺少当前 episode 的 attempt_id")
                            dataset_worker.request_save(
                                current_episode_idx,
                                active_attempt_id,
                            )
                            print(
                                f"\n💾 episode_idx={current_episode_idx} 的全部帧"
                                "已进入 LeRobot 缓冲区，正在后台保存；"
                                "机器人控制保持运行。"
                            )
                        else:
                            awaiting_decision = True
                            print(
                                "\n🟡 后台处理完成：输入 s 保存，输入 d 丢弃"
                            )
                elif kind == "saved" and result_episode == current_episode_idx:
                    saving = False
                    save_after_finalize = False
                    saved_episodes += 1
                    saved_episode_idx = current_episode_idx
                    current_episode_idx += 1
                    episode_samples = 0
                    active_attempt_id = None
                    print(
                        f"\n✅ episode_idx={saved_episode_idx} 已保存；"
                        f"save_episode={result['elapsed_s']:.2f}s；"
                        f"下一编号={current_episode_idx}；"
                        f"本次成功轨迹数={saved_episodes}；"
                        f"队列峰值={dataset_worker.max_observed_pending_frames()}"
                    )
                    if args.episodes > 0 and saved_episodes >= args.episodes:
                        hw._running = False
                    else:
                        _print_controls()
                elif kind in {"discarded", "failed"}:
                    if result_episode == current_episode_idx:
                        recording = False
                        finalizing = False
                        awaiting_decision = False
                        save_after_finalize = False
                        saving = False
                        discarding = False
                        episode_samples = 0
                        active_attempt_id = None
                        print(
                            f"\n🗑️ episode_idx={current_episode_idx} 已丢弃："
                            f"{result.get('reason', '未说明原因')}"
                        )
                        _print_controls()
                elif kind == "stopped":
                    cleanup_error = result.get("cleanup_error")
                    if cleanup_error:
                        print(f"\n[警告] 数据线程退出清理失败：{cleanup_error}")

            # ---------------------------------------------------------------
            # 2. 处理键盘命令。命令只改变状态/入队，不执行磁盘操作。
            # ---------------------------------------------------------------
            while True:
                try:
                    command = commands.get_nowait()
                except queue.Empty:
                    break

                if command == "q":
                    hw._running = False
                    stop_event.set()
                    break

                if command == "s":
                    if finalizing and awaiting_decision and not saving:
                        # 后台还没有处理完所有帧，不能立刻执行 save_episode。
                        # 先记住用户选择，ready 事件到达后自动保存。
                        awaiting_decision = False
                        save_after_finalize = True
                        print(
                            f"\n✅ 已选择保存 episode_idx={current_episode_idx}；"
                            f"后台尚有 {dataset_worker.pending_frames()} 帧待处理，"
                            "处理完成后将自动保存。机器人控制保持 50 Hz。"
                        )
                    elif awaiting_decision and not saving:
                        awaiting_decision = False
                        saving = True
                        if active_attempt_id is None:
                            raise RuntimeError("缺少当前 episode 的 attempt_id")
                        dataset_worker.request_save(
                            current_episode_idx,
                            active_attempt_id,
                        )
                        print(
                            f"\n💾 episode_idx={current_episode_idx} 正在后台保存；"
                            "机器人控制保持运行。"
                        )
                    elif save_after_finalize:
                        print("\n已选择保存，正在等待后台处理剩余帧。")
                    elif saving:
                        print("\n当前 episode 正在后台保存，请勿重复提交。")
                    else:
                        print("\n当前没有可保存的已完成 episode。")
                    continue

                if command == "d":
                    if recording or finalizing or awaiting_decision:
                        if progress_bar is not None:
                            progress_bar.close()
                            progress_bar = None
                        recording = False
                        finalizing = False
                        awaiting_decision = False
                        save_after_finalize = False
                        discarding = True
                        if active_attempt_id is None:
                            raise RuntimeError("缺少当前 episode 的 attempt_id")
                        dataset_worker.request_discard(
                            current_episode_idx,
                            active_attempt_id,
                            "用户手动丢弃",
                        )
                        print(
                            f"\n🗑️ 正在后台清理 episode_idx={current_episode_idx}。"
                        )
                    elif saving:
                        print("\nepisode 已进入保存过程，不能再丢弃。")
                    else:
                        print("\n当前没有可丢弃的 episode。")
                    continue

                if command == "":
                    busy = recording or finalizing or awaiting_decision or saving or discarding
                    if busy:
                        print("\n当前 episode 尚未结束，不能开始下一条。")
                        continue

                    problem = _camera_problem(
                        cameras,
                        maximum_age_ms=args.max_sample_age_ms,
                    )
                    if problem:
                        print(
                            "\n⛔ 无法开始录制，两路有效相机是必需条件："
                            f"{problem}\n"
                            f"   {cameras['cam_high'].status()}\n"
                            f"   {cameras['cam_wrist'].status()}"
                        )
                        continue

                    if dataset_worker.pending_frames() > 0:
                        print(
                            "\n后台仍在清理旧帧，请在状态 queue=0 后再开始。"
                        )
                        continue

                    active_attempt_id = next(attempt_ids)
                    dataset_worker.request_start(
                        current_episode_idx,
                        active_attempt_id,
                        args.episode_len,
                    )
                    recording = True
                    save_after_finalize = False
                    episode_samples = 0
                    next_sample_time = loop_start
                    progress_bar = tqdm(
                        total=args.episode_len,
                        initial=0,
                        desc=f"🔴 episode_idx={current_episode_idx}",
                        unit="frame",
                        dynamic_ncols=False,
                        leave=True,
                        mininterval=0.5,
                        miniters=max(args.dataset_fps // 2, 1),
                        bar_format=(
                            "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                            "[{elapsed}<{remaining}, {rate_fmt}]"
                        ),
                    )

            if not hw._running:
                break

            # ---------------------------------------------------------------
            # 3. 50 Hz 主从控制。这里不允许出现 writer 或视频编码调用。
            # ---------------------------------------------------------------
            with state_lock:
                target_q_sim_raw = shared_state["target_q_sim"].copy()
                gripper_target_raw = float(shared_state["gripper_target_rad"])
                servo_age = loop_start - float(shared_state["timestamp"])

            if servo_age > args.max_servo_age:
                raise RuntimeError(f"主手数据超时: {servo_age * 1000:.0f} ms")

            filtered_q_sim = hw.smooth_update(
                filtered_q_sim,
                target_q_sim_raw,
                args.alpha_master,
            )
            q_target_real = mapper.sim_to_real(filtered_q_sim)

            q_feedback = np.asarray(
                arm.get_positions(request=True)[:6],
                dtype=np.float64,
            )
            if frame % args.velocity_read_every == 0:
                try:
                    last_v_feedback = np.asarray(
                        arm.get_velocities(request=True)[:6],
                        dtype=np.float64,
                    )
                except Exception:
                    # 沿用上一帧速度，避免一次读取失败中断实时控制。
                    pass
            v_feedback = last_v_feedback.copy()

            q_feedback_unwrapped = hw._unwrap_near(q_feedback, q_cmd)
            q_cmd = guard.next_command(q_target_real, q_feedback_unwrapped)
            arm.pos_vel(q_cmd, vlim=vlim)

            gripper_fb_pos = last_gripper_fb_pos
            gripper_fb_vel = last_gripper_fb_vel
            if enable_gripper:
                filtered_gripper = hw.smooth_update_scalar(
                    filtered_gripper,
                    gripper_target_raw,
                    args.alpha_gripper,
                )
                gripper_period = max(args.gripper_send_every, 1)
                if frame % gripper_period == 0:
                    hw.send_damiao_gripper_mit(
                        gripper_motor,
                        gripper_controller,
                        filtered_gripper,
                        args.gripper_kp,
                        args.gripper_kd,
                        args.gripper_tau,
                        request_feedback=True,
                    )
                    pos = hw.get_gripper_feedback_pos(gripper_motor)
                    vel = hw.get_gripper_feedback_vel(gripper_motor)
                    if pos is not None:
                        last_gripper_fb_pos = float(pos)
                    if vel is not None:
                        last_gripper_fb_vel = float(vel)
                    gripper_fb_pos = last_gripper_fb_pos
                    gripper_fb_vel = last_gripper_fb_vel

            # ---------------------------------------------------------------
            # 4. 50 Hz 数据快照。只复制快照并非阻塞入队，不做格式转换/写盘。
            # ---------------------------------------------------------------
            if recording and loop_start >= next_sample_time:
                agent = _read_camera(
                    cameras["cam_high"],
                    copy_frame=args.camera_copy,
                )
                wrist = _read_camera(
                    cameras["cam_wrist"],
                    copy_frame=args.camera_copy,
                )
                imu = (
                    imu_reader.read()
                    if imu_reader is not None
                    else _zero_imu(loop_start)
                )

                if tactile_reader is not None:
                    tactile, tactile_id, tactile_ts = tactile_reader.read()
                else:
                    tactile = np.zeros(
                        (hw.TACTILE_RAW_ROWS, hw.TACTILE_RAW_COLS),
                        dtype=np.float32,
                    )
                    tactile_id = -1
                    tactile_ts = loop_start

                sample = SynchronizedSample(
                    agent=agent,
                    wrist=wrist,
                    feedback=RobotFeedback(
                        _copy_float32(q_feedback),
                        _copy_float32(v_feedback),
                        float(gripper_fb_pos),
                        float(gripper_fb_vel),
                        loop_start,
                    ),
                    target=RobotTarget(
                        _copy_float32(q_cmd),
                        float(filtered_gripper),
                        loop_start,
                    ),
                    imu=_copy_float32(imu["imu_left"]),
                    imu_magnetometer=_copy_float32(imu["mag"]),
                    imu_euler=_copy_float32(imu["euler"]),
                    imu_barometer=_copy_float32(imu["baro"]),
                    tactile=_copy_float32(tactile),
                    imu_frame_id=int(imu["frame_id"]),
                    tactile_frame_id=int(tactile_id),
                    imu_timestamp=float(imu["timestamp"]),
                    tactile_timestamp=float(tactile_ts),
                    timestamp=loop_start,
                )

                if active_attempt_id is None:
                    raise RuntimeError("录制状态缺少 attempt_id")
                submitted = dataset_worker.submit_frame(
                    current_episode_idx,
                    active_attempt_id,
                    sample,
                )
                if not submitted:
                    if progress_bar is not None:
                        progress_bar.close()
                        progress_bar = None
                    recording = False
                    discarding = True
                    dataset_worker.request_discard(
                        current_episode_idx,
                        active_attempt_id,
                        (
                            "后台待处理帧达到上限 "
                            f"{args.writer_queue_size}，为保证 {args.rate:g} Hz "
                            "控制而自动丢弃。可增大 --writer-queue-size，"
                            "但会增加内存占用"
                        ),
                    )
                    print(
                        "\n⛔ 后台待处理帧已达到容量上限；当前 episode 自动丢弃，"
                        "机器人控制未被阻塞。"
                    )
                else:
                    episode_samples += 1
                    if progress_bar is not None:
                        progress_bar.update(1)

                    next_sample_time += sample_period
                    if next_sample_time <= loop_start:
                        # 不补采历史帧；一旦错过，只从当前时间继续，避免一个控制周期
                        # 连续抓取多帧导致更严重的实时性破坏。
                        next_sample_time = loop_start + sample_period

                    if episode_samples >= args.episode_len:
                        recording = False
                        finalizing = True
                        if progress_bar is not None:
                            progress_bar.close()
                            progress_bar = None
                        # 恢复旧版的即时 s/d 交互。用户可以现在选择：
                        # s 仅登记“处理完成后自动保存”，d 会高优先级丢弃。
                        awaiting_decision = True
                        print(
                            f"\n🟡 录制完成：输入 s 保存，输入 d 丢弃。"
                            f"后台尚有 {dataset_worker.pending_frames()} 帧待处理；"
                            "选择保存后会在处理完成时自动执行。"
                        )

            # ---------------------------------------------------------------
            # 5. 低频状态与性能报告。
            # ---------------------------------------------------------------
            if frame % max(args.print_every, 1) == 0 and not recording:
                if saving:
                    mode = "SAVING"
                elif save_after_finalize:
                    mode = "SAVE-WAIT"
                elif discarding:
                    mode = "DISCARD"
                elif finalizing:
                    mode = "FINALIZE"
                elif awaiting_decision:
                    mode = "REVIEW"
                else:
                    mode = "READY"
                print(
                    f"[{mode}] ctrl={frame:06d} "
                    f"sample={episode_samples:04d}/{args.episode_len} "
                    f"queue={dataset_worker.pending_frames():03d} "
                    f"q1={q_feedback[0]:+.3f} "
                    f"gripper={filtered_gripper:+.3f}",
                    end="\r",
                )

            frame += 1
            control_frames += 1

            loop_ms = (time.perf_counter() - loop_start) * 1000.0
            max_loop_ms = max(max_loop_ms, loop_ms)
            if loop_ms > args.loop_warning_ms:
                slow_loops += 1

            # 使用绝对 deadline，避免“本次耗时 + 固定 sleep”造成频率累计漂移。
            next_control_deadline += control_period
            now = time.perf_counter()
            sleep_time = next_control_deadline - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                deadline_misses += 1
                # 若落后超过一个完整周期，不追赶执行多次控制，直接重新对齐当前时间。
                if -sleep_time >= control_period:
                    next_control_deadline = now

            stats_now = time.perf_counter()
            if stats_now - stats_window_start >= 5.0:
                elapsed = stats_now - stats_window_start
                effective_hz = control_frames / elapsed
                pending = dataset_worker.pending_frames()
                queue_ratio = pending / max(args.writer_queue_size, 1)
                if slow_loops > 0 or deadline_misses > 0 or queue_ratio >= 0.25:
                    print(
                        f"\n[实时性] {effective_hz:.2f} Hz，"
                        f"deadline_miss={deadline_misses}，"
                        f"slow_loop={slow_loops}，"
                        f"max_loop={max_loop_ms:.2f} ms，"
                        f"writer_queue={pending}/{args.writer_queue_size}"
                    )
                control_frames = 0
                deadline_misses = 0
                slow_loops = 0
                max_loop_ms = 0.0
                stats_window_start = stats_now

    except KeyboardInterrupt:
        pass
    finally:
        hw._running = False
        stop_event.set()

        if progress_bar is not None:
            progress_bar.close()
            progress_bar = None

        # 先通知数据线程停止；Stop 为最高优先级，会清理未确认缓冲区。
        dataset_worker.request_stop()

        # 优先释放机器人硬件，不等待磁盘线程，确保退出安全。
        if arm is not None:
            try:
                hw.close_arm_fast(arm)
            except Exception as exc:  # noqa: BLE001
                print(f"\n[警告] 机械臂关闭失败：{exc}")
        if port_handler is not None:
            try:
                port_handler.closePort()
            except Exception as exc:  # noqa: BLE001
                print(f"\n[警告] 主手串口关闭失败：{exc}")

        for camera in cameras.values():
            try:
                camera.release()
            except Exception as exc:  # noqa: BLE001
                print(f"\n[警告] 相机释放失败：{exc}")
        if imu_reader is not None:
            try:
                imu_reader.release()
            except Exception as exc:  # noqa: BLE001
                print(f"\n[警告] IMU 释放失败：{exc}")
        if tactile_reader is not None:
            try:
                tactile_reader.release()
            except Exception as exc:  # noqa: BLE001
                print(f"\n[警告] 触觉读取器释放失败：{exc}")

        dataset_worker.join(timeout=10.0)
        if dataset_worker.is_alive():
            print(
                "\n[警告] 数据线程在 10 秒内未退出。硬件已经释放，"
                "可能仍有第三方编码器/文件系统调用未返回。"
            )

        print(f"\n硬件已安全释放，本次保存 {saved_episodes} 条 episode。")


if __name__ == "__main__":
    main()
