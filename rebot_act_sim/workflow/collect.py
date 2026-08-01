"""Collect synchronized visual, proprioceptive, IMU and tactile demonstrations."""

from __future__ import annotations

import argparse
import shutil

import numpy as np

from rebot_act_sim.config import (
    DEFAULT_CONFIG,
    load_config,
    object_randomization_kwargs,
    resolve_project_path,
    teleop_kwargs,
    tactile_processing_kwargs,
)
from rebot_act_sim.dataset_writer import SimACTDatasetWriter, overwrite_lerobot_dataset
from rebot_act_sim.environment import SimACTEnvironment
from rebot_act_sim.schema import FPS
from rebot_act_sim.timing import WallClockRate
from rebot_act_sim.visualization import ReplaySensorVisualizer


def _frame(observation, target: np.ndarray, object_initial_position: np.ndarray) -> dict:
    return {
        "observation.image": observation.image,
        "observation.wrist_image": observation.wrist_image,
        "observation.state": observation.joint_position.astype(np.float32),
        "action": np.asarray(target, dtype=np.float32),
        "sensor.joint_velocity": observation.joint_velocity.astype(np.float32),
        "sensor.gripper_feedback": np.asarray(
            [observation.gripper_position, observation.gripper_velocity], dtype=np.float32
        ),
        "sensor.imu": observation.imu.astype(np.float32),
        "sensor.tactile_left": observation.tactile_left.astype(np.float32),
        "sensor.tactile_right": observation.tactile_right.astype(np.float32),
        "sensor.tactile_left_raw": observation.tactile_left_raw.astype(np.float32),
        "sensor.tactile_right_raw": observation.tactile_right_raw.astype(np.float32),
        "sensor.sim_time": np.asarray([observation.sim_time], dtype=np.float64),
        "episode.object_initial_position": object_initial_position.astype(np.float32),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None, help="base reset seed; -1 randomizes")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-frames", type=int, default=800)
    parser.add_argument(
        "--render-every",
        type=int,
        default=1,
        help="Render MuJoCo viewer every N control steps (1 = every step).",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    raw = load_config(args.config)
    dataset_cfg = raw["dataset"]
    environment_cfg = raw["environment"]
    root = resolve_project_path(dataset_cfg["root"])
    if args.overwrite:
        overwrite_lerobot_dataset(root)
    writer = SimACTDatasetWriter(
        repo_id=str(dataset_cfg["repo_id"]),
        root=root,
        task=str(dataset_cfg["task"]),
        fps=int(dataset_cfg.get("fps", FPS)),
        tactile_processing=environment_cfg.get("tactile_processing"),
    )
    configured_seed = int(environment_cfg.get("seed", 0))
    base_seed = configured_seed if args.seed is None else args.seed
    env = SimACTEnvironment(
        resolve_project_path(environment_cfg["xml"]),
        seed=None,
        success_hold_seconds=float(environment_cfg.get("success_hold_seconds", 0.5)),
        control_hz=int(dataset_cfg["fps"]),
        **object_randomization_kwargs(raw),
        **tactile_processing_kwargs(raw),
        **teleop_kwargs(raw),
    )
    env.reset(None if base_seed < 0 else base_seed)
    episode = 0
    recording = False
    frame_count = 0
    rate = WallClockRate(float(dataset_cfg["fps"]))
    sensor_visualizer = ReplaySensorVisualizer(
        history_frames=int(dataset_cfg["fps"]) * 2,
        tactile_color_max=float(
            environment_cfg.get("tactile_processing", {}).get(
                "visualization_color_max", 15.0
            )
        ),
    )
    control_frame = 0
    render_every = max(1, int(args.render_every))
    try:
        while env.is_alive() and episode < args.episodes:
            env.advance_physics()
            if not env.is_control_tick(int(dataset_cfg.get("fps", FPS))):
                continue
            observation = env.observe()
            target, reset_requested = env.teleop_target()
            if reset_requested:
                writer.discard_episode()
                reset_seed = None if base_seed < 0 else base_seed + episode
                env.reset(reset_seed)
                sensor_visualizer.reset()
                recording = False
                frame_count = 0
                print("Discarded current episode")
                continue

            movement = np.linalg.norm(target[:6] - observation.joint_position) > 1e-3
            gripper_changed = target[-1] != float(observation.gripper_position < 0.025)
            if not recording and (movement or gripper_changed):
                recording = True
                print(f"Recording episode {episode}")
            if recording:
                writer.add_frame(_frame(observation, target, env.object_initial_position))
                frame_count += 1
            if control_frame % render_every == 0:
                sensor_panel = sensor_visualizer.render(
                    observation.imu,
                    observation.tactile_left,
                    observation.tactile_right,
                    frame_index=control_frame,
                    timestamp=observation.sim_time,
                )
                env.render(teleop=True, sensor_panel=sensor_panel)
            control_frame += 1

            if recording and env.check_success():
                writer.save_episode()
                episode += 1
                print(f"Saved episode {episode}/{args.episodes}, frames={frame_count}")
                reset_seed = None if base_seed < 0 else base_seed + episode
                env.reset(reset_seed)
                sensor_visualizer.reset()
                recording = False
                frame_count = 0
            elif recording and frame_count >= args.max_frames:
                writer.discard_episode()
                reset_seed = None if base_seed < 0 else base_seed + episode
                env.reset(reset_seed)
                sensor_visualizer.reset()
                recording = False
                frame_count = 0
                print("Discarded episode after reaching --max-frames")
            rate.wait()
    finally:
        if writer.frames_in_buffer:
            writer.discard_episode()
        env.close()
    images = writer.dataset.root / "images"
    if images.exists():
        shutil.rmtree(images)


if __name__ == "__main__":
    main()
