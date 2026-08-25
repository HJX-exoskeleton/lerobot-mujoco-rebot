"""Collect synchronized visual, proprioceptive, IMU and hand contact demonstrations."""

from __future__ import annotations

import argparse
import shutil
from collections import deque
from dataclasses import dataclass

import mujoco
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
_OPEN_PREROLL_SECONDS = 0.20


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


@dataclass
class _AutoStage:
    name: str
    duration: float
    tool_target: np.ndarray
    hand_target: np.ndarray


class _AutoCollector:
    """Scripted cylinder pick-and-place demonstrator for dataset collection."""

    def __init__(self, env: SimAeroHandACTEnvironment, speed: float):
        self.env = env
        self.speed = max(float(speed), 0.1)
        data, model = env.data, env.model
        goal_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "target_box"
        )
        if goal_body < 0:
            raise RuntimeError("auto collection requires body 'target_box'")

        obj = data.xpos[env.object_body].copy()
        goal = data.xpos[goal_body].copy()
        cylinder_half_height = float(model.geom_size[env.object_geom, 1])
        disk_half_height = float(model.geom_size[env.target_geom, 1])
        mount_to_grasp_x = 0.100
        grasp = obj + np.asarray([-mount_to_grasp_x, -0.050, 0.042])
        contact = grasp + np.asarray([-0.015, 0.0, 0.0])
        behind = contact + np.asarray([-0.10, -0.020, 0.0])
        above_behind = behind + np.asarray([0.0, 0.0, 0.16])
        lift = grasp + np.asarray([0.0, 0.0, 0.20])
        object_place_z = goal[2] + disk_half_height + cylinder_half_height + 0.008
        place = np.asarray(
            [goal[0] - mount_to_grasp_x, goal[1] - 0.050, object_place_z + 0.042]
        )
        above_goal = place + np.asarray([0.0, 0.0, 0.20])
        retreat = place + np.asarray([0.02, -0.03, 0.18])

        opened = env.hand_mapper.open_ctrl
        thumb_opposed = opened.copy()
        thumb_opposed[4] = min(1.35, env.hand_mapper.ctrl_max[4])
        closed = opened.copy()
        closed[:4] += 0.58 * (
            env.hand_mapper.closed_ctrl[:4] - opened[:4]
        )
        closed[5:7] += 0.55 * (
            env.hand_mapper.closed_ctrl[5:7] - opened[5:7]
        )
        closed[4] = thumb_opposed[4]
        release_shaped = opened.copy()
        release_shaped[4] = thumb_opposed[4]
        loosened = closed + 0.55 * (release_shaped - closed)

        def duration(seconds: float) -> float:
            return seconds / self.speed

        self.stages = [
            _AutoStage("settle open", duration(0.8), env.arm_controller.target_pos.copy(), opened),
            _AutoStage("move above pregrasp", duration(2.8), above_behind, opened),
            _AutoStage("lower behind object", duration(2.0), behind, opened),
            _AutoStage("oppose thumb", duration(1.2), behind, thumb_opposed),
            _AutoStage("approach with tiger mouth", duration(2.0), contact, thumb_opposed),
            _AutoStage("envelop object", duration(2.4), contact, closed),
            _AutoStage("settle grasp", duration(1.2), grasp, closed),
            _AutoStage("lift", duration(2.4), lift, closed),
            _AutoStage("move above target", duration(3.0), above_goal, closed),
            _AutoStage("stabilize", duration(1.0), above_goal, closed),
            _AutoStage("lower", duration(3.5), place, closed),
            _AutoStage("loosen grasp", duration(0.8), place, loosened),
            _AutoStage("release", duration(1.8), place, release_shaped),
            _AutoStage("retreat", duration(1.6), retreat, release_shaped),
            _AutoStage("return thumb", duration(1.0), retreat, opened),
            _AutoStage("verify", duration(1.5), retreat, opened),
        ]
        self.index = 0
        self.stage_start = float(data.time)
        self.start_tool = env.arm_controller.target_pos.copy()
        self.start_hand = opened.copy()
        self.finished = False
        self._last_target = np.concatenate(
            [env.arm_controller.command_q, opened]
        ).astype(np.float32)
        self._announce()

    @property
    def phase(self) -> str:
        return "finished" if self.finished else self.stages[self.index].name

    def _announce(self) -> None:
        stage = self.stages[self.index]
        print(
            f"[AUTO {self.index + 1}/{len(self.stages)}] {stage.name} "
            f"({stage.duration:.2f}s)"
        )

    def _compensate_transport_targets(self) -> None:
        object_pos = self.env.data.xpos[self.env.object_body].copy()
        carry_offset = object_pos - self.env.arm_controller.target_pos
        goal_body = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, "target_box"
        )
        goal = self.env.data.xpos[goal_body].copy()
        half_height = float(self.env.model.geom_size[self.env.object_geom, 1])
        disk_half_height = float(self.env.model.geom_size[self.env.target_geom, 1])
        desired_object = np.asarray(
            [goal[0], goal[1], goal[2] + disk_half_height + half_height + 0.008]
        )
        place = desired_object - carry_offset
        above = place + np.asarray([0.0, 0.0, 0.20])
        retreat = place + np.asarray([0.02, -0.03, 0.18])
        for stage in self.stages:
            if stage.name in ("move above target", "stabilize"):
                stage.tool_target[:] = above
            elif stage.name in ("lower", "loosen grasp", "release"):
                stage.tool_target[:] = place
            elif stage.name in ("retreat", "return thumb", "verify"):
                stage.tool_target[:] = retreat

    def update(self) -> np.ndarray:
        if self.finished:
            return self._last_target.copy()
        stage = self.stages[self.index]
        elapsed = float(self.env.data.time) - self.stage_start
        progress = float(np.clip(elapsed / max(stage.duration, 1e-6), 0.0, 1.0))
        # A smoothstep per waypoint forces zero velocity at every stage boundary
        # and creates visible stop-and-go motion. Keep Cartesian travel linear
        # across adjacent waypoints; stages whose target is unchanged remain
        # true holds. Hand shaping stays eased to avoid tendon command shocks.
        tool = self.start_tool + progress * (stage.tool_target - self.start_tool)
        hand_blend = _smoothstep(progress)
        hand = self.start_hand + hand_blend * (
            stage.hand_target - self.start_hand
        )
        target = self.env.scripted_target(tool, hand)
        self._last_target = target.copy()
        if elapsed < stage.duration:
            return target

        self.index += 1
        if self.index >= len(self.stages):
            self.finished = True
            return target
        if self.stages[self.index].name == "move above target":
            self._compensate_transport_targets()
        self.stage_start = float(self.env.data.time)
        self.start_tool = stage.tool_target.copy()
        self.start_hand = stage.hand_target.copy()
        self._announce()
        return target


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
        "--auto_collect",
        action="store_true",
        help="Automatically execute scripted cylinder pick-and-place episodes.",
    )
    parser.add_argument(
        "--auto-speed",
        type=float,
        default=2.5,
        help="Automatic trajectory speed multiplier (default: 2.5).",
    )
    parser.add_argument(
        "--camera-render-hz",
        type=float,
        default=None,
        help="Camera rendering rate; defaults to environment.camera_render_hz.",
    )
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
        camera_render_hz=(
            args.camera_render_hz
            if args.camera_render_hz is not None
            else float(environment_cfg.get("camera_render_hz", dataset_cfg["fps"]))
        ),
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
    preroll_frames = max(
        1,
        int(round(_OPEN_PREROLL_SECONDS * float(dataset_cfg["fps"]))),
    )
    idle_preroll: deque[dict] = deque(maxlen=preroll_frames)
    last_hand_ctrl = env.hand_mapper.command_ctrl().copy()
    last_arm_target = env.arm_controller.command_q.copy()
    auto = _AutoCollector(env, args.auto_speed) if args.auto_collect else None
    if auto is not None:
        recording = True

    print(f"[Scene]   {environment_cfg['xml']}")
    print(f"[Dataset] repo_id={dataset_cfg['repo_id']}  root={root}")
    print(
        f"[Task]    {dataset_cfg['task']}  fps={dataset_cfg['fps']}  "
        f"episodes={args.episodes}  seed={'random' if base_seed < 0 else base_seed}"
    )
    if auto is None:
        print("[Controls] arm: W/A/S/D/R/F move | arrows pitch/roll | Q/E yaw")
        print("[Controls] hand: 1-5 toggle fingers | SPACE grasp | O/C open/close | H arm home")
        print("[Controls] Z discard episode | close the window to quit")
        print(
            f"[Status]  recording starts on arm/hand motion; "
            f"max {args.max_frames} frames per episode"
        )
    else:
        print(
            f"[AUTO]    scripted collection enabled; speed={args.auto_speed:g}x; "
            f"max {args.max_frames} frames per episode"
        )

    try:
        while env.is_alive() and episode < args.episodes:
            env.advance_physics()
            if not env.is_control_tick(int(dataset_cfg.get("fps", FPS))):
                continue
            observation = env.observe()
            if auto is None:
                target, reset_requested = env.teleop_target()
            else:
                target = auto.update()
                reset_requested = False
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
                auto = (
                    _AutoCollector(env, args.auto_speed)
                    if args.auto_collect
                    else None
                )
                recording = auto is not None
                frame_count = 0
                idle_preroll.clear()
                last_arm_target = env.arm_controller.command_q.copy()
                last_hand_ctrl = env.hand_mapper.command_ctrl().copy()
                continue

            hand_changed = (
                last_hand_ctrl is not None
                and np.linalg.norm(target[STATE_DIM:] - last_hand_ctrl) > 1e-4
            )
            if hand_changed and auto is None:
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
            current_frame = _frame(observation, target, env.object_initial_position)
            if auto is None and not recording and (movement or hand_changed):
                recording = True
                for buffered_frame in idle_preroll:
                    writer.add_frame(buffered_frame)
                frame_count = len(idle_preroll)
                idle_preroll.clear()
                object_pose = env.object_initial_position
                print(
                    f"Recording episode {episode} "
                    f"(seed={episode_seed if episode_seed is not None else 'random'}, "
                    f"object=({object_pose[0]:.3f}, {object_pose[1]:.3f}, {object_pose[2]:.3f}))"
                )
            if recording:
                writer.add_frame(current_frame)
                frame_count += 1
            else:
                # Keep a short rolling window whose action is the explicit
                # open-hand target. It becomes the start of the episode once
                # the operator first moves the arm or commands a grasp.
                idle_preroll.append(current_frame)
            if control_frame % render_every == 0:
                collected_cameras = {
                    "top": observation.image,
                    "cam_wrist": observation.wrist_image,
                }
                overlay_names = list(environment_cfg.get("camera_overlays", []))
                camera_frames = (
                    [(name, collected_cameras[name]) for name in overlay_names]
                    if all(name in collected_cameras for name in overlay_names)
                    else None
                )
                status_lines = (
                    f"COLLECT  episode={episode}  "
                    f"{'AUTO' if auto is not None else ('REC' if recording else 'IDLE')}",
                    f"frames={frame_count}  t={observation.sim_time:.2f}s  "
                    f"dist_to_obj={env.grasp_surface_distance():.3f}m  "
                    + (
                        f"phase={auto.phase}"
                        if auto is not None
                        else f"hand[{_finger_state(env)}]"
                    ),
                )
                # No sensor panel during collection; the camera PIPs and the
                # status text are the only in-window extras.
                env.render(
                    teleop=auto is None,
                    status_lines=status_lines,
                    camera_frames=camera_frames,
                )
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
                auto = (
                    _AutoCollector(env, args.auto_speed)
                    if args.auto_collect and episode < args.episodes
                    else None
                )
                recording = auto is not None
                frame_count = 0
                idle_preroll.clear()
                last_arm_target = env.arm_controller.command_q.copy()
                last_hand_ctrl = env.hand_mapper.command_ctrl().copy()
            elif recording and frame_count >= args.max_frames:
                writer.discard_episode()
                # Discarded resets get a fresh random object layout instead of
                # retrying the same seed.
                episode_seed = None
                env.reset(None)
                auto = (
                    _AutoCollector(env, args.auto_speed)
                    if args.auto_collect
                    else None
                )
                recording = auto is not None
                frame_count = 0
                idle_preroll.clear()
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
