#!/usr/bin/env python3
"""重播LeRobot真机ACT episode，并同步显示全部多模态传感器。

仅可视化（不会连接或控制机械臂）::

    python -m rebot_act_real.workflow.replay \
      --root ./data_act_real/rebot_act_banana \
      --episode-idx 0 --visualize-only

真机动作重播（需要输入 ``REPLAY`` 二次确认）::

    python -m rebot_act_real.workflow.replay \
      --root ./data_act_real/rebot_act_banana \
      --episode-idx 0 \
      --cfg rebot_act_real/workflow/config/arm.yaml \
      --gripper-cfg rebot_act_real/workflow/config/gripper.yaml \
      --execute

窗口中按 ``q`` 或 ``Esc`` 可中止；终端 ``Ctrl+C`` 也会停止重播。真机默认重播
数据集 ``action``（六轴最终命令和夹爪目标），而不是反馈 ``observation.state``。
回播结束或中止后默认保持电机使能和最后目标，防止机械臂突然失能下坠；只有显式
传入 ``--final-disable`` 才会卸载电机力矩。
"""

from __future__ import annotations

import argparse
import io
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from rebot_scripts.Servo_control import replay_rebot_episodes as hw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = PROJECT_ROOT / "data_act_real" / "rebot_act_banana"
DEFAULT_ARM_CFG = WORKFLOW_ROOT / "config" / "arm.yaml"
DEFAULT_GRIPPER_CFG = WORKFLOW_ROOT / "config" / "gripper.yaml"


@dataclass
class EpisodeData:
    root: Path
    episode_idx: int
    fps: float
    task: str
    table: object
    state: np.ndarray
    action: np.ndarray
    joint_velocity: np.ndarray
    gripper_feedback: np.ndarray
    imu: np.ndarray
    imu_magnetometer: np.ndarray
    imu_euler: np.ndarray
    imu_barometer: np.ndarray
    tactile: np.ndarray
    frame_ids: np.ndarray
    sensor_timestamps: np.ndarray

    def __len__(self) -> int:
        return int(self.action.shape[0])

    def image_bgr(self, key: str, index: int) -> np.ndarray:
        value = self.table[key][index].as_py()
        encoded = value.get("bytes") if isinstance(value, dict) else None
        if encoded:
            with Image.open(io.BytesIO(encoded)) as image:
                rgb = np.asarray(image.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        raise RuntimeError(f"{key}[{index}] 没有内嵌图像字节")


def _column_array(table, key: str, dtype) -> np.ndarray:
    return np.asarray([value.as_py() for value in table[key]], dtype=dtype)


def load_episode(root: Path, episode_idx: int) -> EpisodeData:
    info_path = root / "meta" / "info.json"
    episodes_path = root / "meta" / "episodes.jsonl"
    tasks_path = root / "meta" / "tasks.jsonl"
    if not info_path.is_file():
        raise FileNotFoundError(f"找不到 LeRobot info.json: {info_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    episodes = [
        json.loads(line)
        for line in episodes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    episode_meta = next(
        (item for item in episodes if int(item["episode_index"]) == episode_idx), None
    )
    if episode_meta is None:
        available = [int(item["episode_index"]) for item in episodes]
        raise ValueError(f"episode_idx={episode_idx} 不存在，可用编号: {available}")

    chunk_size = int(info.get("chunks_size", 1000))
    relative = str(info["data_path"]).format(
        episode_chunk=episode_idx // chunk_size,
        episode_index=episode_idx,
    )
    parquet_path = root / relative
    if not parquet_path.is_file():
        raise FileNotFoundError(f"找不到 episode Parquet: {parquet_path}")
    table = pq.read_table(parquet_path)

    required = {
        "observation.image",
        "observation.wrist_image",
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
    }
    missing = sorted(required.difference(table.column_names))
    if missing:
        raise KeyError(f"episode 缺少传感器字段: {missing}")

    tasks = [
        json.loads(line)
        for line in tasks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    task = str(episode_meta.get("tasks", [""])[0])
    if not task and tasks:
        task = str(tasks[0].get("task", ""))

    data = EpisodeData(
        root=root,
        episode_idx=episode_idx,
        fps=float(info["fps"]),
        task=task,
        table=table,
        state=_column_array(table, "observation.state", np.float32),
        action=_column_array(table, "action", np.float32),
        joint_velocity=_column_array(table, "sensor.joint_velocity", np.float32),
        gripper_feedback=_column_array(table, "sensor.gripper_feedback", np.float32),
        imu=_column_array(table, "sensor.imu", np.float32),
        imu_magnetometer=_column_array(
            table, "sensor.imu_magnetometer", np.float32
        ),
        imu_euler=_column_array(table, "sensor.imu_euler", np.float32),
        imu_barometer=_column_array(table, "sensor.imu_barometer", np.float32),
        tactile=_column_array(table, "sensor.tactile", np.float32),
        frame_ids=_column_array(table, "sensor.frame_ids", np.int64),
        sensor_timestamps=_column_array(table, "sensor.timestamps", np.float64),
    )
    if len(data) == 0:
        raise ValueError("episode 为空")
    return data


def _put_lines(
    image: np.ndarray,
    lines: list[str],
    origin: tuple[int, int],
    *,
    scale: float = 0.48,
    spacing: int = 23,
    color: tuple[int, int, int] = (225, 225, 225),
) -> None:
    x, y = origin
    for line in lines:
        cv2.putText(
            image,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )
        y += spacing


def _draw_history(
    canvas: np.ndarray,
    values: np.ndarray,
    index: int,
    rect: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
    history: int = 120,
) -> None:
    x, y, w, h = rect
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (70, 70, 70), 1)
    cv2.putText(
        canvas, label, (x + 6, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1
    )
    start = max(0, index - history + 1)
    segment = np.asarray(values[start : index + 1], dtype=np.float32)
    if segment.size < 2:
        return
    finite = segment[np.isfinite(segment)]
    if finite.size == 0:
        return
    low, high = float(finite.min()), float(finite.max())
    if abs(high - low) < 1e-6:
        low -= 0.5
        high += 0.5
    points = []
    for i, value in enumerate(segment):
        px = x + int(i * (w - 1) / max(len(segment) - 1, 1))
        py = y + h - 2 - int((float(value) - low) / (high - low) * (h - 24))
        points.append((px, py))
    cv2.polylines(canvas, [np.asarray(points, np.int32)], False, color, 2)
    cv2.putText(
        canvas,
        f"{float(segment[-1]):+.3f}",
        (x + w - 82, y + 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        color,
        1,
    )


class SensorVisualizer:
    WINDOW = "reBot ACT Multimodal Episode Replay"

    def __init__(
        self, data: EpisodeData, scale: float = 1.0, *, create_window: bool = True
    ):
        self.data = data
        self.scale = max(float(scale), 0.25)
        self.window_created = bool(create_window)
        self.gyro_norm = np.linalg.norm(data.imu[:, 4:7], axis=1)
        self.accel_norm = np.linalg.norm(data.imu[:, 7:10], axis=1)
        self.tactile_max = data.tactile.reshape(len(data), -1).max(axis=1)
        self.gripper_target = data.action[:, 6]
        if self.window_created:
            cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)

    def render(self, index: int, stage: str, speed_scale: float) -> np.ndarray:
        data = self.data
        panel = np.full((820, 1400, 3), 24, dtype=np.uint8)
        high = cv2.resize(data.image_bgr("observation.image", index), (470, 350))
        wrist = cv2.resize(
            data.image_bgr("observation.wrist_image", index), (470, 350)
        )
        panel[40:390, 0:470] = high
        panel[40:390, 480:950] = wrist
        cv2.putText(
            panel,
            "cam_high / Gemini 336L",
            (8, 27),
            0,
            0.62,
            (80, 220, 255),
            2,
        )
        cv2.putText(panel, "cam_wrist / D405", (488, 27), 0, 0.62, (80, 220, 255), 2)

        tactile = np.clip(data.tactile[index], 0.0, 1.0)
        tactile_u8 = np.asarray(tactile * 255.0, dtype=np.uint8)
        tactile_color = cv2.applyColorMap(tactile_u8, cv2.COLORMAP_TURBO)
        tactile_color = cv2.resize(
            tactile_color, (590, 250), interpolation=cv2.INTER_NEAREST
        )
        panel[445:695, 0:590] = tactile_color
        cv2.putText(
            panel, "FlexiTac 12x30", (8, 430), 0, 0.62, (80, 220, 255), 2
        )

        imu = data.imu[index]
        info = [
            f"episode={data.episode_idx}  frame={index}/{len(data)-1}",
            f"stage={stage}  speed={speed_scale:.2f}x  fps={data.fps:.1f}",
            f"time={index/data.fps:.2f}s",
            "",
            "IMU quaternion (wxyz)",
            np.array2string(imu[:4], precision=3, suppress_small=True),
            "gyro xyz [rad/s]",
            np.array2string(imu[4:7], precision=3, suppress_small=True),
            "accel xyz [g]",
            np.array2string(imu[7:10], precision=3, suppress_small=True),
            "mag xyz",
            np.array2string(data.imu_magnetometer[index], precision=2),
            "euler xyz [deg]",
            np.array2string(data.imu_euler[index], precision=2),
            "barometer",
            np.array2string(data.imu_barometer[index], precision=2),
            "",
            f"tactile min/max/mean: {tactile.min():.3f} / "
            f"{tactile.max():.3f} / {tactile.mean():.3f}",
            f"sensor frame ids: {data.frame_ids[index].tolist()}",
            "sensor time offsets [ms]",
            np.array2string(
                (data.sensor_timestamps[index] - data.sensor_timestamps[index, 0])
                * 1000.0,
                precision=1,
            ),
        ]
        _put_lines(panel, info, (970, 28), spacing=22)

        state = np.array2string(data.state[index], precision=3, suppress_small=True)
        action = np.array2string(data.action[index], precision=3, suppress_small=True)
        velocity = np.array2string(
            data.joint_velocity[index], precision=3, suppress_small=True
        )
        gripper = data.gripper_feedback[index]
        _put_lines(
            panel,
            [
                "proprioception",
                f"state q: {state}",
                f"joint vel: {velocity}",
                f"action: {action}",
                f"gripper feedback pos/vel: {gripper[0]:+.3f}, {gripper[1]:+.3f}",
            ],
            (8, 724),
            scale=0.43,
            spacing=20,
        )

        _draw_history(
            panel, self.gyro_norm, index, (620, 500, 320, 95), "|gyro|", (70, 210, 255)
        )
        _draw_history(
            panel,
            self.accel_norm,
            index,
            (620, 620, 320, 95),
            "|accel|",
            (100, 255, 100),
        )
        _draw_history(
            panel,
            self.tactile_max,
            index,
            (960, 500, 420, 95),
            "tactile max",
            (255, 130, 80),
        )
        _draw_history(
            panel,
            self.gripper_target,
            index,
            (960, 620, 420, 95),
            "gripper target",
            (255, 100, 220),
        )
        cv2.putText(
            panel,
            "q / ESC: stop replay",
            (1170, 800),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (150, 150, 150),
            1,
        )
        if self.scale != 1.0:
            panel = cv2.resize(
                panel, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA
            )
        return panel

    def show(self, index: int, stage: str, speed_scale: float) -> bool:
        if not self.window_created:
            raise RuntimeError("SensorVisualizer 创建时禁用了窗口")
        cv2.imshow(self.WINDOW, self.render(index, stage, speed_scale))
        key = cv2.waitKey(1) & 0xFF
        visible = cv2.getWindowProperty(self.WINDOW, cv2.WND_PROP_VISIBLE) >= 1
        return visible and key not in (ord("q"), 27)

    def close(self) -> None:
        if self.window_created:
            try:
                cv2.destroyWindow(self.WINDOW)
            except cv2.error:
                pass


def replay_visual_only(
    data: EpisodeData, visualizer: SensorVisualizer, speed_scale: float
) -> None:
    hw._running = True
    period = 1.0 / (data.fps * speed_scale)
    with tqdm(total=len(data), desc=f"episode_idx={data.episode_idx}", unit="frame") as bar:
        for index in range(len(data)):
            if not hw._running:
                break
            start = time.perf_counter()
            if not visualizer.show(index, "visualize-only", speed_scale):
                break
            bar.update(1)
            delay = period - (time.perf_counter() - start)
            if delay > 0:
                time.sleep(delay)


def replay_hardware(args, data: EpisodeData, visualizer: SensorVisualizer) -> None:
    hw._running = True
    vlim = hw._parse_vector(args.vlim, hw.DEFAULT_CMD_VLIM, "--vlim")
    max_step = hw._parse_vector(args.max_step, hw.DEFAULT_MAX_STEP, "--max-step")
    trajectory = data.action if args.trajectory_key == "action" else data.state
    if trajectory.shape[1] < 6:
        raise ValueError(f"{args.trajectory_key} 少于六轴: {trajectory.shape}")

    print("\n⚠️ 即将真实驱动机械臂和夹爪。确认工作空间安全、急停可用。")
    if not args.yes and input("输入 REPLAY 后回车开始：").strip() != "REPLAY":
        print("已取消真机重播。")
        return

    arm = gripper_motor = gripper_controller = None
    try:
        RobotArm = hw._load_robot_arm_class()
        arm = RobotArm(cfg_path=str(args.cfg) if args.cfg else None)
        arm.connect()
        arm.enable()
        arm.mode_pos_vel(vlim=vlim)
        time.sleep(0.5)
        if not args.no_gripper:
            gripper_motor, gripper_controller = hw.setup_damiao_gripper(
                arm, args.gripper_cfg
            )

        q_raw = np.asarray(arm.get_positions(request=True)[:6], dtype=np.float64)
        first_arm = hw._unwrap_near(trajectory[0, :6], q_raw)
        q_feedback = hw._unwrap_near(q_raw, first_arm)
        gripper_fb = (
            hw.get_gripper_feedback_pos(gripper_motor)
            if gripper_motor is not None
            else None
        )
        gripper_start = (
            float(gripper_fb) if gripper_fb is not None else float(args.gripper_default)
        )
        first_gripper = (
            float(data.action[0, 6])
            if data.action.shape[1] >= 7
            else float(args.gripper_default)
        )
        guard = hw.SafetyGuard(
            max_step=max_step,
            max_tracking_error=args.max_tracking_error,
            breach_samples=args.breach_samples,
        )
        q_cmd = guard.initialize(q_feedback)
        gripper_cmd = gripper_start

        prepare = max(float(args.prepare_duration), 0.0)
        replay_duration = len(data) / data.fps / args.speed_scale
        post_hold = max(float(args.post_hold), 0.0)
        control_period = 1.0 / args.rate
        start_time = time.perf_counter()
        control_frame = 0
        last_index = -1
        last_arm = first_arm.copy()
        last_gripper = first_gripper
        progress = tqdm(total=len(data), desc=f"episode_idx={data.episode_idx}", unit="frame")

        while hw._running:
            loop_start = time.perf_counter()
            elapsed = loop_start - start_time
            q_raw = np.asarray(arm.get_positions(request=True)[:6], dtype=np.float64)
            q_feedback = hw._unwrap_near(q_raw, q_cmd)

            if elapsed < prepare:
                alpha = hw._cosine_smooth(elapsed / max(prepare, 1e-6))
                target_arm = q_feedback + (first_arm - q_feedback) * alpha
                target_gripper = gripper_start + (first_gripper - gripper_start) * alpha
                index, stage = 0, "prepare"
            elif elapsed < prepare + replay_duration:
                replay_time = elapsed - prepare
                index = min(int(replay_time * data.fps * args.speed_scale), len(data) - 1)
                target_arm = trajectory[index, :6]
                target_gripper = (
                    float(data.action[index, 6])
                    if data.action.shape[1] >= 7
                    else float(args.gripper_default)
                )
                stage = "replay"
            elif elapsed < prepare + replay_duration + post_hold:
                index, stage = len(data) - 1, "post-hold"
                target_arm, target_gripper = last_arm, last_gripper
            else:
                break

            last_arm = np.asarray(target_arm, dtype=np.float64).copy()
            last_gripper = float(target_gripper)
            q_cmd = guard.next_command(target_arm, q_feedback)
            arm.pos_vel(q_cmd, vlim=vlim)
            if (
                gripper_motor is not None
                and control_frame % max(int(args.gripper_send_every), 1) == 0
            ):
                gripper_cmd = float(target_gripper)
                hw.send_damiao_gripper_mit(
                    gripper_motor,
                    gripper_controller,
                    gripper_cmd,
                    args.gripper_kp,
                    args.gripper_kd,
                    args.gripper_tau,
                    request_feedback=True,
                )

            if index != last_index:
                progress.update(max(index - last_index, 0))
                last_index = index
                if not visualizer.show(index, stage, args.speed_scale):
                    hw._running = False
                    break
            control_frame += 1
            delay = control_period - (time.perf_counter() - loop_start)
            if delay > 0:
                time.sleep(delay)
        progress.close()
    finally:
        if args.final_disable:
            print("⚠️ 已按 --final-disable 要求卸载电机力矩。")
            hw.close_arm_fast(arm)
        else:
            print(
                "\n[安全保持] 回放已停止，但机械臂和夹爪继续保持使能及最后目标。"
            )
            print(
                "[安全保持] 未调用 arm.disable()，请使用回零/接管程序平稳接管；"
                "不要直接切断电源。"
            )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="reBot ACT LeRobot多模态轨迹重播")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--episode-idx", "--episode_idx", type=int, default=0)
    parser.add_argument("--visualize-only", action="store_true", help="只显示数据，不连接真机")
    parser.add_argument("--execute", action="store_true", help="允许真实机械臂动作重播")
    parser.add_argument("--yes", action="store_true", help="跳过 REPLAY 文本确认")
    parser.add_argument(
        "--trajectory-key",
        choices=("action", "observation.state"),
        default="action",
        help="真机六轴目标来源；默认使用采集时最终下发的 action",
    )
    parser.add_argument("--cfg", type=Path, default=DEFAULT_ARM_CFG)
    parser.add_argument("--gripper-cfg", type=Path, default=DEFAULT_GRIPPER_CFG)
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--prepare-duration", type=float, default=4.0)
    parser.add_argument("--post-hold", type=float, default=0.5)
    parser.add_argument("--max-tracking-error", type=float, default=1.0)
    parser.add_argument("--breach-samples", type=int, default=20)
    parser.add_argument("--vlim", type=float, nargs=6, default=None)
    parser.add_argument("--max-step", type=float, nargs=6, default=None)
    parser.add_argument("--gripper-kp", type=float, default=1.0)
    parser.add_argument("--gripper-kd", type=float, default=0.05)
    parser.add_argument("--gripper-tau", type=float, default=0.0)
    parser.add_argument("--gripper-send-every", type=int, default=1)
    parser.add_argument("--gripper-default", type=float, default=-5.8)
    final_group = parser.add_mutually_exclusive_group()
    final_group.add_argument(
        "--final-disable",
        action="store_true",
        help="回放结束后主动失能；机械臂可能下坠，默认不启用",
    )
    final_group.add_argument(
        "--no-final-disable",
        dest="final_disable",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(final_disable=False)
    parser.add_argument("--visualize-scale", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.episode_idx < 0:
        raise ValueError("--episode-idx 不能为负数")
    if args.speed_scale <= 0 or args.rate <= 0:
        raise ValueError("--speed-scale 和 --rate 必须大于 0")
    if args.execute and args.visualize_only:
        raise ValueError("--execute 与 --visualize-only 不能同时使用")
    if not args.execute and not args.visualize_only:
        raise ValueError("必须明确选择 --visualize-only 或 --execute")
    if args.execute:
        if not args.cfg.is_file():
            raise FileNotFoundError(f"机械臂配置不存在: {args.cfg}")
        if not args.no_gripper and not args.gripper_cfg.is_file():
            raise FileNotFoundError(f"夹爪配置不存在: {args.gripper_cfg}")

    data = load_episode(args.root.resolve(), args.episode_idx)
    print(
        f"已加载 episode_idx={data.episode_idx}: {len(data)} 帧, "
        f"{data.fps:g}Hz, {len(data)/data.fps:.2f}s"
    )
    print(f"ACT任务标签: {data.task}")
    visualizer = SensorVisualizer(data, args.visualize_scale)
    try:
        if args.visualize_only:
            replay_visual_only(data, visualizer, args.speed_scale)
        else:
            replay_hardware(args, data, visualizer)
    finally:
        visualizer.close()


if __name__ == "__main__":
    main()
