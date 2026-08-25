"""Collect synchronized visual, proprioceptive, IMU and tactile demonstrations."""

from __future__ import annotations

import argparse
import faulthandler
import shutil
from dataclasses import dataclass

import mujoco
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
from rebot_act_sim.environment import JOINT_NAMES, SimACTEnvironment
from rebot_act_sim.schema import FPS
from rebot_act_sim.timing import WallClockRate
from rebot_act_sim.visualization import AsyncSensorVisualizer, ReplaySensorVisualizer


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    # Quintic minimum-jerk blend: position, velocity and acceleration are all
    # continuous at stage boundaries.
    return value**3 * (value * (value * 6.0 - 15.0) + 10.0)


@dataclass
class _AutoStage:
    name: str
    duration: float
    tool_target: np.ndarray
    gripper_target: float


class _AutoCollector:
    """Scripted randomized red-cube pick-and-place demonstrator."""

    def __init__(self, env: SimACTEnvironment, speed: float):
        self.env = env
        self.speed = max(float(speed), 0.1)
        model, data = env.parser.model, env.parser.data
        self.object_body = int(model.body("body_obj_mug_5").id)
        self.object_geom = int(model.geom("geom_obj_red_cube").id)
        self.plate_body = int(model.body("body_obj_plate_11").id)
        self.plate_top_site = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "top_site_plate_11"
        )
        if self.plate_top_site < 0:
            raise RuntimeError("auto collection requires site 'top_site_plate_11'")

        tcp = np.asarray(env.task.p0, dtype=np.float64).copy()
        left_pad = data.geom_xpos[model.geom("touch_base_left").id]
        right_pad = data.geom_xpos[model.geom("touch_base_right").id]
        pad_center = 0.5 * (left_pad + right_pad)
        self.pad_offset = pad_center - tcp
        object_position = data.xpos[self.object_body].copy()
        grasp = object_position - self.pad_offset
        # Keep the lower finger surfaces clear of the tabletop.  The pads are
        # slightly above the cube centre and still overlap enough of the cube
        # height to produce a stable pinch.
        grasp[2] += 0.016
        above_object = grasp + np.asarray([0.0, 0.0, 0.16])
        lift = grasp + np.asarray([0.0, 0.0, 0.18])

        def seconds(value: float) -> float:
            return value / self.speed

        # Target-side waypoints are corrected from the actual carry offset
        # after the lift begins, as in the AeroHand automatic collector.
        placeholder = lift.copy()
        self.stages = [
            _AutoStage("settle open", seconds(0.6), tcp, 0.0),
            _AutoStage("move above object", seconds(1.8), above_object, 0.0),
            _AutoStage("lower to grasp", seconds(1.4), grasp, 0.0),
            _AutoStage("close gripper", seconds(1.0), grasp, 1.0),
            _AutoStage("settle grasp", seconds(0.6), grasp, 1.0),
            _AutoStage("lift", seconds(1.5), lift, 1.0),
            _AutoStage("move above plate", seconds(2.0), placeholder.copy(), 1.0),
            _AutoStage("settle above plate", seconds(0.5), placeholder.copy(), 1.0),
            _AutoStage("lower onto plate", seconds(1.4), placeholder.copy(), 1.0),
            _AutoStage("center on plate", seconds(0.8), placeholder.copy(), 1.0),
            _AutoStage("release", seconds(1.0), placeholder.copy(), 0.0),
            _AutoStage("settle released", seconds(0.5), placeholder.copy(), 0.0),
            _AutoStage("retreat", seconds(1.2), placeholder.copy(), 0.0),
            _AutoStage("verify", seconds(0.8), placeholder.copy(), 0.0),
        ]
        self.index = 0
        self.stage_start = float(data.time)
        self.start_tool = tcp.copy()
        self.start_gripper = 0.0
        self.finished = False
        self._transport_compensated = False
        self._last_target = np.concatenate(
            [
                np.asarray(
                    env.parser.get_qpos_joints(joint_names=list(JOINT_NAMES)),
                    dtype=np.float32,
                ),
                [0.0],
            ]
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

    def _desired_object_on_plate(self) -> np.ndarray:
        data, model = self.env.parser.data, self.env.parser.model
        plate = data.xpos[self.plate_body].copy()
        top_z = float(data.site_xpos[self.plate_top_site, 2])
        cube_half_height = self.env._geom_vertical_half_extent(
            model.geom(self.object_geom)
        )
        return np.asarray(
            [plate[0], plate[1], top_z + cube_half_height + 0.004]
        )

    def _compensate_transport_targets(self, *, include_above: bool) -> None:
        """Correct targets from the cube's current measured position.

        Repeating this correction before descent and before release removes
        residual tracking/carry-offset error instead of accumulating it into
        the final placement position.
        """
        data = self.env.parser.data
        desired_object = self._desired_object_on_plate()
        object_position = data.xpos[self.object_body].copy()
        commanded_tcp = np.asarray(self.env.task.p0, dtype=np.float64).copy()
        place = commanded_tcp + desired_object - object_position
        above = place + np.asarray([0.0, 0.0, 0.16])
        retreat = place + np.asarray([0.0, 0.0, 0.18])
        for stage in self.stages:
            if include_above and stage.name in (
                "move above plate",
                "settle above plate",
            ):
                stage.tool_target[:] = above
            elif stage.name in (
                "lower onto plate",
                "center on plate",
                "release",
                "settle released",
            ):
                stage.tool_target[:] = place
            elif stage.name in ("retreat", "verify"):
                stage.tool_target[:] = retreat
        self._transport_compensated = True

    def update(self) -> np.ndarray:
        if self.finished:
            return self._last_target.copy()
        stage = self.stages[self.index]
        elapsed = float(self.env.parser.data.time) - self.stage_start
        progress = float(np.clip(elapsed / max(stage.duration, 1e-6), 0.0, 1.0))
        blend = _smoothstep(progress)
        tool = self.start_tool + blend * (stage.tool_target - self.start_tool)
        gripper = self.start_gripper + blend * (
            stage.gripper_target - self.start_gripper
        )
        target = self.env.scripted_target(tool, gripper)
        self._last_target = target.copy()
        if elapsed < stage.duration:
            return target

        self.index += 1
        if self.index >= len(self.stages):
            self.finished = True
            return target
        if (
            self.stages[self.index].name == "move above plate"
            and not self._transport_compensated
        ):
            self._compensate_transport_targets(include_above=True)
        elif self.stages[self.index].name in ("lower onto plate", "center on plate"):
            self._compensate_transport_targets(include_above=False)
        self.stage_start = float(self.env.parser.data.time)
        self.start_tool = stage.tool_target.copy()
        self.start_gripper = float(stage.gripper_target)
        self._announce()
        return target


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
    parser.add_argument(
        "--xml",
        help="Override environment XML (for example the diagnostic cylinder scene).",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None, help="base reset seed; -1 randomizes")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-frames", type=int, default=1200)
    parser.add_argument(
        "--auto_collect",
        action="store_true",
        help="Automatically execute randomized red-cube pick-and-place episodes.",
    )
    parser.add_argument(
        "--auto-speed",
        type=float,
        default=1.0,
        help="Automatic trajectory speed multiplier (default: 1.0).",
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=2,
        help="Render MuJoCo viewer every N control steps (default: 2 = 25 Hz).",
    )
    parser.add_argument(
        "--camera-render-hz",
        type=float,
        help="Effective dataset camera rate; defaults to environment.camera_render_hz.",
    )
    parser.add_argument(
        "--async-sensor-visualization",
        action="store_true",
        help=(
            "Render the IMU/tactile panel on a background thread. Disabled by "
            "default because some OpenCV/OpenGL driver combinations can crash."
        ),
    )
    return parser


def main() -> None:
    # Native GLFW/OpenGL/OpenCV failures otherwise exit with code 139 and no
    # useful Python context. This prints all Python thread stacks on SIGSEGV.
    faulthandler.enable(all_threads=True)
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
        resolve_project_path(args.xml or environment_cfg["xml"]),
        seed=None,
        success_hold_seconds=float(environment_cfg.get("success_hold_seconds", 0.5)),
        control_hz=int(dataset_cfg["fps"]),
        camera_render_hz=(
            args.camera_render_hz
            if args.camera_render_hz is not None
            else float(environment_cfg.get("camera_render_hz", dataset_cfg["fps"]))
        ),
        # The collection viewer uses the sensor panel, so the legacy sideview
        # was rendered but never displayed or stored.
        grab_sideview=False,
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
    async_sensor_visualizer = (
        AsyncSensorVisualizer(
            history_frames=int(dataset_cfg["fps"]) * 2,
            tactile_color_max=float(
                environment_cfg.get("tactile_processing", {}).get(
                    "visualization_color_max", 15.0
                )
            ),
        )
        if args.async_sensor_visualization
        else None
    )
    control_frame = 0
    reset_count = 0
    render_every = max(1, int(args.render_every))
    auto = _AutoCollector(env, args.auto_speed) if args.auto_collect else None
    if auto is not None:
        recording = True
        print(
            f"[AUTO] scripted collection enabled; speed={args.auto_speed:g}x; "
            f"episodes={args.episodes}"
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
                reset_count += 1
                reset_seed = None if base_seed < 0 else base_seed + reset_count
                env.reset(reset_seed)
                auto = (
                    _AutoCollector(env, args.auto_speed)
                    if args.auto_collect
                    else None
                )
                sensor_visualizer.reset()
                if async_sensor_visualizer is not None:
                    async_sensor_visualizer.reset()
                recording = auto is not None
                frame_count = 0
                print("Discarded current episode")
                continue

            movement = np.linalg.norm(target[:6] - observation.joint_position) > 1e-3
            gripper_changed = target[-1] != float(observation.gripper_position < 0.025)
            if auto is None and not recording and (movement or gripper_changed):
                recording = True
                print(f"Recording episode {episode}")
            if recording:
                writer.add_frame(_frame(observation, target, env.object_initial_position))
                frame_count += 1
            if async_sensor_visualizer is not None:
                async_sensor_visualizer.push(
                    observation.imu,
                    observation.tactile_left,
                    observation.tactile_right,
                    frame_index=control_frame,
                    timestamp=observation.sim_time,
                )
            if control_frame % render_every == 0:
                sensor_panel = (
                    async_sensor_visualizer.get_panel()
                    if async_sensor_visualizer is not None
                    else None
                )
                if sensor_panel is None:
                    sensor_panel = sensor_visualizer.render(
                        observation.imu,
                        observation.tactile_left,
                        observation.tactile_right,
                        frame_index=control_frame,
                        timestamp=observation.sim_time,
                    )
                env.render(teleop=auto is None, sensor_panel=sensor_panel)
            control_frame += 1

            if recording and env.check_success():
                writer.save_episode()
                episode += 1
                print(f"Saved episode {episode}/{args.episodes}, frames={frame_count}")
                reset_count += 1
                reset_seed = None if base_seed < 0 else base_seed + reset_count
                env.reset(reset_seed)
                auto = (
                    _AutoCollector(env, args.auto_speed)
                    if args.auto_collect and episode < args.episodes
                    else None
                )
                sensor_visualizer.reset()
                if async_sensor_visualizer is not None:
                    async_sensor_visualizer.reset()
                recording = auto is not None
                frame_count = 0
            elif recording and frame_count >= args.max_frames:
                writer.discard_episode()
                reset_count += 1
                reset_seed = None if base_seed < 0 else base_seed + reset_count
                env.reset(reset_seed)
                auto = (
                    _AutoCollector(env, args.auto_speed)
                    if args.auto_collect
                    else None
                )
                sensor_visualizer.reset()
                if async_sensor_visualizer is not None:
                    async_sensor_visualizer.reset()
                recording = auto is not None
                frame_count = 0
                print("Discarded episode after reaching --max-frames")
            rate.wait()
    finally:
        if async_sensor_visualizer is not None:
            async_sensor_visualizer.stop()
        if writer.frames_in_buffer:
            writer.discard_episode()
        env.close()
    images = writer.dataset.root / "images"
    if images.exists():
        shutil.rmtree(images)


if __name__ == "__main__":
    main()
