"""Collect synchronized visual, proprioceptive, IMU and hand contact demonstrations."""

from __future__ import annotations

import argparse
import shutil

import numpy as np

from rebot_aerohand_right_act_sim.config import (
    DEFAULT_CONFIG,
    camera_overlay_kwargs,
    hand_contact_processing_kwargs,
    load_config,
    object_randomization_kwargs,
    resolve_project_path,
    teleop_kwargs,
)
from rebot_aerohand_right_act_sim.dataset_writer import (
    SimAeroHandACTDatasetWriter,
    overwrite_lerobot_dataset,
)
from rebot_aerohand_right_act_sim.environment import (
    HAND_FINGER_NAMES,
    SimAeroHandACTEnvironment,
)
from rebot_aerohand_right_act_sim.schema import FPS, STATE_DIM
from rebot_aerohand_right_act_sim.timing import WallClockRate

_STATUS_PERIOD_FRAMES = 100  # one terminal status line every ~2 s


def _frame(observation, target: np.ndarray, object_initial_position: np.ndarray) -> dict:
    return {
        "observation.image": observation.image,
        "observation.wrist_image": observation.wrist_image,
        "observation.state": observation.joint_position.astype(np.float32),
        "action": np.asarray(target, dtype=np.float32),
        "sensor.joint_velocity": observation.joint_velocity.astype(np.float32),
        "sensor.hand_feedback": observation.hand_feedback.astype(np.float32),
        "sensor.hand_joint_position": observation.hand_joint_position.astype(
            np.float32
        ),
        "sensor.imu": observation.imu.astype(np.float32),
        "sensor.hand_contact": observation.hand_contact.astype(np.float32),
        "sensor.sim_time": np.asarray([observation.sim_time], dtype=np.float64),
        "episode.object_initial_position": object_initial_position.astype(
            np.float32
        ),
    }


def _finger_state(env) -> str:
    return " ".join(
        f"{name}:{'closed' if closed else 'open'}"
        for name, closed in zip(HAND_FINGER_NAMES, env.hand_mapper.closed)
    )


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
    writer = SimAeroHandACTDatasetWriter(
        repo_id=str(dataset_cfg["repo_id"]),
        root=root,
        task=str(dataset_cfg["task"]),
        fps=int(dataset_cfg.get("fps", FPS)),
        hand_contact_processing=environment_cfg.get("hand_contact_processing"),
    )
    configured_seed = int(environment_cfg.get("seed", 0))
    base_seed = configured_seed if args.seed is None else args.seed
    env = SimAeroHandACTEnvironment(
        resolve_project_path(environment_cfg["xml"]),
        seed=None,
        success_hold_seconds=float(environment_cfg.get("success_hold_seconds", 0.5)),
        control_hz=int(dataset_cfg["fps"]),
        **object_randomization_kwargs(raw),
        **hand_contact_processing_kwargs(raw),
        **teleop_kwargs(raw),
        **camera_overlay_kwargs(raw),
    )
    env.reset(None if base_seed < 0 else base_seed)
    episode = 0
    episode_seed = None if base_seed < 0 else base_seed
    recording = False
    frame_count = 0
    rate = WallClockRate(float(dataset_cfg["fps"]))
    control_frame = 0
    render_every = max(1, int(args.render_every))
    last_hand_ctrl = env.hand_mapper.command_ctrl().copy()
    last_arm_target = env.arm_controller.command_q.copy()

    print(f"[Scene]   {environment_cfg['xml']}")
    print(f"[Dataset] repo_id={dataset_cfg['repo_id']}  root={root}")
    print(
        f"[Task]    {dataset_cfg['task']}  fps={dataset_cfg['fps']}  "
        f"episodes={args.episodes}  seed={'random' if base_seed < 0 else base_seed}"
    )
    print("[Controls] arm: W/A/S/D/R/F move | arrows pitch/roll | Q/E yaw")
    print("[Controls] hand: 1-5 toggle fingers | SPACE grasp | O/C open/close | H arm home")
    print("[Controls] Z discard episode | close the window to quit")
    print(f"[Status]  recording starts on arm/hand motion; max {args.max_frames} frames per episode")

    try:
        while env.is_alive() and episode < args.episodes:
            env.advance_physics()
            if not env.is_control_tick(int(dataset_cfg.get("fps", FPS))):
                continue
            observation = env.observe()
            target, reset_requested = env.teleop_target()
            if reset_requested:
                writer.discard_episode()
                if frame_count > 0:
                    print(f"Discarded current episode (frames={frame_count})")
                else:
                    print("Reset environment (nothing recorded)")
                # Discarded resets get a fresh random object layout instead of
                # retrying the same seed.
                episode_seed = None
                env.reset(None)
                recording = False
                frame_count = 0
                last_arm_target = env.arm_controller.command_q.copy()
                last_hand_ctrl = env.hand_mapper.command_ctrl().copy()
                continue

            hand_changed = (
                last_hand_ctrl is not None
                and np.linalg.norm(target[STATE_DIM:] - last_hand_ctrl) > 1e-4
            )
            if hand_changed:
                print(f"[Hand] {_finger_state(env)}")
            last_hand_ctrl = target[STATE_DIM:].copy()
            # Recording starts when the human commands something. Comparing
            # the arm target against the measured joint feedback would trigger
            # immediately at rest: the hand payload makes the position servo
            # sag ~0.03 rad below the IK target.
            movement = (
                last_arm_target is not None
                and np.linalg.norm(target[:STATE_DIM] - last_arm_target) > 1e-4
            )
            last_arm_target = target[:STATE_DIM].copy()
            if not recording and (movement or hand_changed):
                recording = True
                object_pose = env.object_initial_position
                print(
                    f"Recording episode {episode} "
                    f"(seed={episode_seed if episode_seed is not None else 'random'}, "
                    f"object=({object_pose[0]:.3f}, {object_pose[1]:.3f}, {object_pose[2]:.3f}))"
                )
            if recording:
                writer.add_frame(_frame(observation, target, env.object_initial_position))
                frame_count += 1
            if control_frame % render_every == 0:
                status_lines = (
                    f"COLLECT  episode={episode}  {'REC' if recording else 'IDLE'}",
                    f"frames={frame_count}  t={observation.sim_time:.2f}s  "
                    f"dist_to_obj={env.grasp_surface_distance():.3f}m  "
                    f"hand[{_finger_state(env)}]",
                )
                # No sensor panel during collection; the camera PIPs and the
                # status text are the only in-window extras.
                env.render(teleop=True, status_lines=status_lines)
            control_frame += 1

            if recording and env.check_success():
                writer.save_episode()
                episode += 1
                print(
                    f"Saved episode {episode}/{args.episodes}, "
                    f"frames={frame_count}, duration={observation.sim_time:.1f}s"
                )
                episode_seed = None if base_seed < 0 else base_seed + episode
                env.reset(episode_seed)
                recording = False
                frame_count = 0
                last_arm_target = env.arm_controller.command_q.copy()
                last_hand_ctrl = env.hand_mapper.command_ctrl().copy()
            elif recording and frame_count >= args.max_frames:
                writer.discard_episode()
                # Discarded resets get a fresh random object layout instead of
                # retrying the same seed.
                episode_seed = None
                env.reset(None)
                recording = False
                frame_count = 0
                last_arm_target = env.arm_controller.command_q.copy()
                last_hand_ctrl = env.hand_mapper.command_ctrl().copy()
                print(
                    f"Discarded episode after reaching --max-frames "
                    f"(frames={args.max_frames}); reset with a new random "
                    f"object position"
                )
            elif recording and frame_count > 0 and frame_count % 200 == 0:
                status = env.success_status()
                print(
                    f"[HINT] placed={'yes' if status['placed'] else 'no'} "
                    f"released={'yes' if status['released'] else 'no'} "
                    f"retreated={'yes' if status['retreated'] else 'no'} "
                    f"hold={status['hold']}/{status['hold_steps']}"
                )
            elif control_frame % _STATUS_PERIOD_FRAMES == 0:
                print(
                    f"[{observation.sim_time:7.2f}s] episode={episode} "
                    f"{'REC' if recording else 'IDLE'} frames={frame_count} "
                    f"dist_to_obj={env.grasp_surface_distance():.3f}m",
                    end="\r",
                )
            rate.wait()
    finally:
        if writer.frames_in_buffer:
            writer.discard_episode()
        env.close()
    images = writer.dataset.root / "images"
    if images.exists():
        shutil.rmtree(images)
    print(
        f"\n[Summary] saved {writer.dataset.num_episodes} episodes "
        f"({len(writer.dataset)} frames) at {root}"
    )


if __name__ == "__main__":
    main()
