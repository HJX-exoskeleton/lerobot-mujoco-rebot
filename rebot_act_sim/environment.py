"""Task-level adapter with explicit observation, target and sensor semantics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import glfw
import mujoco
import numpy as np
from PIL import Image

from .schema import FPS, IMAGE_HEIGHT, IMAGE_WIDTH, TACTILE_SHAPE
from .sensors import (
    TactileProcessingConfig,
    TactileSignalProcessor,
    read_tactile_contact_projection,
    read_imu,
    read_tactile_normal,
    read_tactile_proximity,
    tactile_proximity_geom_ids,
)

JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
TRANSLATION_STEP = 0.003
ROTATION_STEP = 0.02


def _rotation_to_rpy(rotation: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to XYZ roll/pitch/yaw without GUI imports."""
    return np.asarray(
        [
            np.arctan2(rotation[2, 1], rotation[2, 2]),
            np.arctan2(
                -rotation[2, 0],
                np.hypot(rotation[2, 1], rotation[2, 2]),
            ),
            np.arctan2(rotation[1, 0], rotation[0, 0]),
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class SimObservation:
    image: np.ndarray
    wrist_image: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    gripper_position: float
    gripper_velocity: float
    imu: np.ndarray
    tactile_left: np.ndarray
    tactile_right: np.ndarray
    tactile_left_raw: np.ndarray
    tactile_right_raw: np.ndarray
    sim_time: float


def _resize_rgb(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.uint8)
    if value.shape[:2] == (IMAGE_HEIGHT, IMAGE_WIDTH):
        return value.copy()
    return np.asarray(
        Image.fromarray(value).resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.BILINEAR)
    )


def success_hold_steps(seconds: float, control_hz: int) -> int:
    if seconds <= 0:
        raise ValueError("success hold seconds must be positive")
    if control_hz <= 0:
        raise ValueError("control_hz must be positive")
    return max(1, int(np.ceil(seconds * control_hz)))


def teleop_delta_from_keys(
    repeated_keys: set[int],
    *,
    gripper_state: bool,
    toggle_gripper: bool = False,
) -> tuple[np.ndarray, bool]:
    """Map keys to a tool-local Cartesian delta and updated gripper state.

    Arrow keys follow screen-like semantics:
    up/down pitch the gripper forward/backward; left/right roll it
    counter-clockwise/clockwise. Q/E retain local yaw control.
    """

    dpos = np.zeros(3, dtype=np.float32)
    drpy = np.zeros(3, dtype=np.float32)
    if glfw.KEY_S in repeated_keys:
        dpos[0] += TRANSLATION_STEP
    if glfw.KEY_W in repeated_keys:
        dpos[0] -= TRANSLATION_STEP
    if glfw.KEY_A in repeated_keys:
        dpos[1] -= TRANSLATION_STEP
    if glfw.KEY_D in repeated_keys:
        dpos[1] += TRANSLATION_STEP
    if glfw.KEY_R in repeated_keys:
        dpos[2] += TRANSLATION_STEP
    if glfw.KEY_F in repeated_keys:
        dpos[2] -= TRANSLATION_STEP

    # SimpleEnv post-multiplies the rotation, so these are tool-body-local axes.
    if glfw.KEY_UP in repeated_keys:
        drpy[1] -= ROTATION_STEP       # pitch forward
    if glfw.KEY_DOWN in repeated_keys:
        drpy[1] += ROTATION_STEP       # pitch backward
    if glfw.KEY_LEFT in repeated_keys:
        drpy[0] -= ROTATION_STEP       # counter-clockwise roll
    if glfw.KEY_RIGHT in repeated_keys:
        drpy[0] += ROTATION_STEP       # clockwise roll
    if glfw.KEY_Q in repeated_keys:
        drpy[2] += ROTATION_STEP
    if glfw.KEY_E in repeated_keys:
        drpy[2] -= ROTATION_STEP

    next_gripper_state = not gripper_state if toggle_gripper else gripper_state
    action = np.concatenate(
        [dpos, drpy, np.asarray([next_gripper_state], dtype=np.float32)]
    )
    return action.astype(np.float32, copy=False), next_gripper_state


class SimACTEnvironment:
    """Thin compatibility layer over ``SimpleEnv``.

    ``observation.state`` is measured joint position. ``action`` is always the
    target sent for the next control interval: six radians plus binary gripper.
    """

    def __init__(
        self,
        xml_path: str | Path,
        *,
        seed: int | None = None,
        success_hold_seconds: float = 0.5,
        control_hz: int = FPS,
        camera_render_hz: float | None = None,
        plate_position: tuple[float, float, float] | list[float] | None = None,
        target_object_position_center: tuple[float, float, float]
        | list[float]
        | None = None,
        target_object_xy_half_range: tuple[float, float] | list[float] | None = None,
        tactile_processing: dict | None = None,
        teleop_motion_alpha: float = 0.25,
        grab_sideview: bool = True,
        enable_tactile_processing: bool = True,
    ):
        from mujoco_env.y_env import SimpleEnv

        self.task = SimpleEnv(
            str(xml_path),
            action_type="eef_pose",
            state_type="joint_angle",
            seed=seed,
            gripper_joint_name="finger_left",
            tcp_body_name="end_link",
            egocentric_camera_name="cam_wrist",
            tcp_marker_radius=0.01,
            tcp_marker_site_name="grasp_center",
        )
        self._scripted_joint_target: np.ndarray | None = None
        self._scripted_joint_velocity = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        # SimpleEnv historically initializes its viewer with black_sky=True.
        # This task ships an actual desert skybox, so explicitly turn skybox
        # rendering back on for collection/replay/deployment viewers.
        self.parser.viewer.scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = True
        self.success_hold_steps = success_hold_steps(success_hold_seconds, control_hz)
        self.control_hz = int(control_hz)
        requested_camera_hz = (
            float(self.control_hz)
            if camera_render_hz is None
            else float(camera_render_hz)
        )
        if requested_camera_hz <= 0:
            raise ValueError("camera_render_hz must be positive")
        self.camera_render_hz = min(requested_camera_hz, float(self.control_hz))
        self._camera_render_stride = max(
            1, int(round(float(self.control_hz) / self.camera_render_hz))
        )
        self._camera_cache: tuple[np.ndarray, np.ndarray] | None = None
        self._observation_count = 0
        self._grab_sideview = bool(grab_sideview)
        self._enable_tactile_proc = bool(enable_tactile_processing)
        self.tactile_config = TactileProcessingConfig.from_mapping(
            tactile_processing
        )
        self.tactile_processor = TactileSignalProcessor(self.tactile_config)
        self.teleop_motion_alpha = float(teleop_motion_alpha)
        if not 0 < self.teleop_motion_alpha <= 1:
            raise ValueError("teleop_motion_alpha must be in (0, 1]")
        self._teleop_motion = np.zeros(6, dtype=np.float32)
        if self._enable_tactile_proc:
            self._configure_tactile_contact_dynamics()
        self._proximity_object_geom = -1
        self._proximity_taxel_geoms: dict[str, np.ndarray] = {}
        if (
            self._enable_tactile_proc
            and self.tactile_config.signal_source == "distance_proximity"
        ):
            self._proximity_object_geom = int(
                self.parser.model.geom("geom_obj_red_cube").id
            )
            self._proximity_taxel_geoms = {
                side: tactile_proximity_geom_ids(self.parser.model, side)
                for side in ("left", "right")
            }
        self.plate_position = self._position3(plate_position, "plate_position")
        self.target_object_position_center = self._position3(
            target_object_position_center, "target_object_position_center"
        )
        if target_object_xy_half_range is None:
            self.target_object_xy_half_range = None
        else:
            self.target_object_xy_half_range = np.asarray(
                target_object_xy_half_range, dtype=np.float64
            )
            if self.target_object_xy_half_range.shape != (2,) or np.any(
                self.target_object_xy_half_range < 0
            ):
                raise ValueError(
                    "target_object_xy_half_range must contain two nonnegative values"
                )
        if (self.plate_position is None) != (
            self.target_object_position_center is None
        ):
            raise ValueError(
                "plate_position and target_object_position_center must be "
                "configured together"
            )
        if (
            self.target_object_position_center is not None
            and self.target_object_xy_half_range is None
        ):
            raise ValueError(
                "target_object_xy_half_range is required for object randomization"
            )
        self._last_gripper_position = self._gripper_position()
        self._last_time = float(self.task.env.get_sim_time())
        self._success_count = 0

    @staticmethod
    def _position3(value, name: str) -> np.ndarray | None:
        if value is None:
            return None
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (3,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain three finite values")
        return result

    def _configure_tactile_contact_dynamics(self) -> None:
        """Use one continuous collision surface instead of 128 coplanar geoms."""

        for side in ("left", "right"):
            pad = self.parser.model.geom(f"touch_base_{side}")
            if self.tactile_config.signal_source == "continuous_contact_projection":
                # The legacy base surface is 0.4 mm behind the taxel faces.
                # Move it flush with them and disable the competing cell contacts.
                pad.pos[2] = 0.012
                pad.contype[0] = 1
                pad.conaffinity[0] = 1
                pad.solref[0] = self.tactile_config.contact_time_constant
            for index in range(TACTILE_SHAPE[0] * TACTILE_SHAPE[1]):
                body = self.parser.model.body(
                    f"touch_cell_{side}_{index:03d}"
                )
                for offset in range(int(body.geomnum[0])):
                    geom_index = int(body.geomadr[0]) + offset
                    if (
                        self.tactile_config.signal_source
                        == "continuous_contact_projection"
                    ):
                        self.parser.model.geom_contype[geom_index] = 0
                        self.parser.model.geom_conaffinity[geom_index] = 0
                    elif self.tactile_config.signal_source == "distance_proximity":
                        # Distance-sensing taxels must never enter MuJoCo's
                        # collision/contact pipeline.
                        self.parser.model.geom_contype[geom_index] = 0
                        self.parser.model.geom_conaffinity[geom_index] = 0
                    self.parser.model.geom_solref[geom_index, 0] = (
                        self.tactile_config.contact_time_constant
                    )
            if self.tactile_config.signal_source == "distance_proximity":
                pad.contype[0] = 0
                pad.conaffinity[0] = 0

    def _set_controlled_object_positions(self, seed: int | None) -> None:
        if self.plate_position is None:
            return
        rng = np.random.default_rng(seed)
        target = self.target_object_position_center.copy()
        target[:2] += rng.uniform(
            -self.target_object_xy_half_range, self.target_object_xy_half_range
        )
        if np.linalg.norm(target[:2] - self.plate_position[:2]) < 0.12:
            raise ValueError(
                "configured target object randomization can place it too close "
                "to the plate"
            )
        self._set_object_pose_without_step(target, self.plate_position)
        actual_target, actual_plate = self.task.get_obj_pose()
        self.task.obj_init_pose = np.concatenate(
            [actual_target, actual_plate], dtype=np.float32
        )

    def _table_top_z(self) -> float:
        table = self.parser.model.geom("front_object_table")
        return float(table.pos[2] + table.size[2])

    @staticmethod
    def _geom_vertical_half_extent(geom) -> float:
        if int(geom.type[0]) in (
            int(mujoco.mjtGeom.mjGEOM_CYLINDER),
            int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        ):
            return float(geom.size[1])
        return float(geom.size[2])

    def _set_object_pose_without_step(
        self, target: np.ndarray, plate: np.ndarray
    ) -> None:
        """Place free bodies without generating a teleportation impulse."""

        target = np.asarray(target, dtype=np.float64).copy()
        # Keep the cube just above the tabletop. The following normal physics
        # step lets gravity establish contact gradually instead of correcting a
        # potentially penetrating pose with a large impulse.
        cube_geom = self.parser.model.geom("geom_obj_red_cube")
        minimum_z = (
            self._table_top_z()
            + self._geom_vertical_half_extent(cube_geom)
            + 0.0005
        )
        target[2] = max(float(target[2]), minimum_z)
        self.parser.set_p_base_body(body_name="body_obj_mug_5", p=target)
        self.parser.set_R_base_body(
            body_name="body_obj_mug_5", R=np.eye(3, dtype=np.float64)
        )
        self.parser.set_p_base_body(body_name="body_obj_plate_11", p=plate)
        self.parser.set_R_base_body(
            body_name="body_obj_plate_11", R=np.eye(3, dtype=np.float64)
        )
        self._zero_free_body_velocity("body_obj_mug_5")
        self._zero_free_body_velocity("body_obj_plate_11")
        self._enforce_plate_lock()
        self.parser.forward(increase_tick=False)

    def _free_joint_addresses(self, body_name: str) -> tuple[int, int]:
        body = self.parser.model.body(body_name)
        if int(body.jntnum[0]) != 1:
            raise RuntimeError(f"{body_name} must have exactly one free joint")
        joint_index = int(body.jntadr[0])
        qpos_address = int(self.parser.model.jnt_qposadr[joint_index])
        dof_address = int(self.parser.model.jnt_dofadr[joint_index])
        return qpos_address, dof_address

    def _zero_free_body_velocity(self, body_name: str) -> None:
        _, dof_address = self._free_joint_addresses(body_name)
        self.parser.data.qvel[dof_address : dof_address + 6] = 0.0

    def _enforce_plate_lock(self) -> None:
        if self.plate_position is None:
            return
        qpos_address, dof_address = self._free_joint_addresses(
            "body_obj_plate_11"
        )
        self.parser.data.qpos[qpos_address : qpos_address + 3] = self.plate_position
        self.parser.data.qpos[qpos_address + 3 : qpos_address + 7] = (
            1.0,
            0.0,
            0.0,
            0.0,
        )
        self.parser.data.qvel[dof_address : dof_address + 6] = 0.0
        self.parser.forward(increase_tick=False)

    @property
    def parser(self):
        return self.task.env

    def reset(self, seed: int | None = None) -> None:
        # Work around the legacy SimpleEnv reset bug that ignored nonzero seeds.
        if seed is None:
            self.task.reset(seed=None)
        else:
            rng_state = np.random.get_state()
            np.random.seed(seed)
            try:
                self.task.reset(seed=None)
            finally:
                np.random.set_state(rng_state)
        self._set_controlled_object_positions(seed)
        self._last_gripper_position = self._gripper_position()
        self._last_time = float(self.parser.get_sim_time())
        self._success_count = 0
        self.tactile_processor.reset()
        self._teleop_motion.fill(0)
        self._camera_cache = None
        self._observation_count = 0
        self._scripted_joint_target = np.asarray(
            self.parser.get_qpos_joints(joint_names=list(JOINT_NAMES)),
            dtype=np.float64,
        )
        self._scripted_joint_velocity.fill(0.0)

    def poll_events(self) -> None:
        """Poll GLFW events without rendering (headless mode).

        Keeps the viewer window responsive so the user can close it.
        """
        glfw.poll_events()

    def render_scene(self) -> None:
        """Render the MuJoCo 3D viewer scene only — no PIP camera overlays,
        no TCP sphere/capsule markers, no sensor panel.  The free camera
        (orbit/pan/zoom) works normally.
        """
        self.parser.render()

    def close(self) -> None:
        self.parser.close_viewer()

    def is_alive(self) -> bool:
        return self.parser.is_viewer_alive()

    def advance_physics(self) -> None:
        self.task.step_env()
        self._enforce_plate_lock()
        if not self._enable_tactile_proc:
            return
        if self.tactile_config.signal_source == "distance_proximity":
            return
        if self.tactile_config.signal_source == "continuous_contact_projection":
            tactile_left, tactile_right = read_tactile_contact_projection(
                self.parser,
                projection_sigma=self.tactile_config.projection_sigma,
            )
        else:
            tactile_left, tactile_right = read_tactile_normal(
                self.parser, normal_axis=self.tactile_config.normal_axis
            )
        self.tactile_processor.update(tactile_left, tactile_right)

    def is_control_tick(self, hz: int) -> bool:
        return self.parser.loop_every(HZ=hz)

    def _gripper_position(self) -> float:
        return float(self.parser.get_qpos_joint(self.task.gripper_joint_name)[0])

    def observe(self) -> SimObservation:
        if (
            self._camera_cache is None
            or self._observation_count % self._camera_render_stride == 0
        ):
            self._camera_cache = self.task.grab_image(
                grab_sideview=self._grab_sideview
            )
        image, wrist = self._camera_cache
        self._observation_count += 1
        now = float(self.parser.get_sim_time())
        dt = max(now - self._last_time, 1e-6)
        gripper = self._gripper_position()
        if self._enable_tactile_proc:
            if self.tactile_config.signal_source == "distance_proximity":
                proximity = {
                    side: read_tactile_proximity(
                        self.parser.model,
                        self.parser.data,
                        self._proximity_object_geom,
                        self._proximity_taxel_geoms[side],
                        sigma=self.tactile_config.proximity_sigma,
                        reach=self.tactile_config.proximity_reach,
                        threshold=self.tactile_config.proximity_threshold,
                    )
                    for side in ("left", "right")
                }
                self.tactile_processor.update(
                    proximity["left"], proximity["right"]
                )
            tactile_left, tactile_right = self.tactile_processor.consume()
        else:
            tactile_left = np.zeros(TACTILE_SHAPE, dtype=np.float32)
            tactile_right = np.zeros(TACTILE_SHAPE, dtype=np.float32)
        observation = SimObservation(
            image=_resize_rgb(image),
            wrist_image=_resize_rgb(wrist),
            joint_position=np.asarray(
                self.parser.get_qpos_joints(joint_names=list(JOINT_NAMES)), dtype=np.float32
            ),
            joint_velocity=np.asarray(
                self.parser.get_qvel_joints(joint_names=list(JOINT_NAMES)), dtype=np.float32
            ),
            gripper_position=gripper,
            gripper_velocity=(gripper - self._last_gripper_position) / dt,
            imu=read_imu(self.parser),
            tactile_left=tactile_left,
            tactile_right=tactile_right,
            tactile_left_raw=self.tactile_processor.latest_raw_left.copy(),
            tactile_right_raw=self.tactile_processor.latest_raw_right.copy(),
            sim_time=now,
        )
        self._last_gripper_position = gripper
        self._last_time = now
        return observation

    def teleop_target(self) -> tuple[np.ndarray, bool]:
        if self.parser.is_key_pressed_once(key=glfw.KEY_Z):
            return np.zeros(7, dtype=np.float32), True
        repeated_keys = {
            key
            for key in (
                glfw.KEY_W,
                glfw.KEY_A,
                glfw.KEY_S,
                glfw.KEY_D,
                glfw.KEY_R,
                glfw.KEY_F,
                glfw.KEY_UP,
                glfw.KEY_DOWN,
                glfw.KEY_LEFT,
                glfw.KEY_RIGHT,
                glfw.KEY_Q,
                glfw.KEY_E,
            )
            if self.parser.is_key_pressed_repeat(key=key)
        }
        delta_pose, self.task.gripper_state = teleop_delta_from_keys(
            repeated_keys,
            gripper_state=self.task.gripper_state,
            toggle_gripper=self.parser.is_key_pressed_once(key=glfw.KEY_SPACE),
        )
        alpha = self.teleop_motion_alpha
        self._teleop_motion = (
            alpha * delta_pose[:6] + (1.0 - alpha) * self._teleop_motion
        )
        delta_pose[:6] = self._teleop_motion
        self.task.action_type = "eef_pose"
        self.task.step(delta_pose)
        target = np.concatenate(
            [
                np.asarray(self.task.compute_q, dtype=np.float32),
                np.asarray([float(self.task.gripper_state)], dtype=np.float32),
            ]
        )
        return target, False

    def command(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise ValueError(f"invalid ACT action: {action}")
        action = action.copy()
        action[-1] = np.clip(action[-1], 0.0, 1.0)
        self.task.action_type = "joint_angle"
        self.task.step(action)

    def scripted_target(
        self,
        tool_position: np.ndarray,
        gripper_closed: float,
        tool_rotation: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convert an absolute Cartesian waypoint into the normal 7-D action.

        ``tool_rotation`` is an optional absolute world-frame rotation. The
        returned action is exactly the joint/gripper command sent to the
        simulator and can therefore be written directly to an ACT
        demonstration.
        """
        tool_position = np.asarray(tool_position, dtype=np.float64)
        if tool_position.shape != (3,) or not np.all(np.isfinite(tool_position)):
            raise ValueError("tool_position must contain three finite values")
        gripper_closed = float(np.clip(gripper_closed, 0.0, 1.0))
        delta = tool_position - np.asarray(self.task.p0, dtype=np.float64)
        rotation_delta = np.zeros(3, dtype=np.float64)
        if tool_rotation is not None:
            tool_rotation = np.asarray(tool_rotation, dtype=np.float64)
            if tool_rotation.shape != (3, 3) or not np.all(
                np.isfinite(tool_rotation)
            ):
                raise ValueError("tool_rotation must be a finite 3x3 matrix")
            rotation_delta = _rotation_to_rpy(self.task.R0.T @ tool_rotation)
        self.task.action_type = "eef_pose"
        self.task.step(
            np.concatenate(
                [delta, rotation_delta, [gripper_closed]]
            )
        )
        raw_joint_target = np.asarray(self.task.compute_q, dtype=np.float64)
        if self._scripted_joint_target is None:
            self._scripted_joint_target = np.asarray(
                self.parser.get_qpos_joints(joint_names=list(JOINT_NAMES)),
                dtype=np.float64,
            )

        # Numerical IK can occasionally select a noticeably different joint
        # solution for two adjacent Cartesian targets. Limit both velocity and
        # acceleration at the 50 Hz command level so this cannot become a
        # visible one-frame arm jerk.
        remaining = raw_joint_target - self._scripted_joint_target
        desired_velocity = np.clip(remaining, -0.03, 0.03)
        velocity_change = np.clip(
            desired_velocity - self._scripted_joint_velocity, -0.003, 0.003
        )
        velocity = self._scripted_joint_velocity + velocity_change
        step = velocity.copy()
        # Decelerate to zero before reversing direction, and do not overshoot
        # a nearby target.
        step[step * remaining <= 0.0] = 0.0
        overshoot = np.abs(step) > np.abs(remaining)
        step[overshoot] = remaining[overshoot]
        self._scripted_joint_target = self._scripted_joint_target + step
        self._scripted_joint_velocity = velocity
        self._scripted_joint_velocity[overshoot] = step[overshoot]

        # SimpleEnv.step() has already converted the normalized gripper command
        # to its physical actuator target. Preserve it while replacing the raw
        # IK jump with the smoothed six-joint command.
        self.task.compute_q = self._scripted_joint_target.copy()
        self.task.q[: len(JOINT_NAMES)] = self._scripted_joint_target
        return np.concatenate(
            [
                self._scripted_joint_target.astype(np.float32),
                np.asarray([gripper_closed], dtype=np.float32),
            ]
        )

    def render(
        self, *, teleop: bool = False, sensor_panel: np.ndarray | None = None
    ) -> None:
        if sensor_panel is None:
            self.task.render(teleop=teleop)
            return
        self.parser.viewer_rgb_overlay(sensor_panel, loc="top left")
        if teleop:
            self.parser.viewer_text_overlay(
                text1="Key Pressed",
                text2=f"{self.parser.get_key_pressed_list()}",
            )
            self.parser.viewer_text_overlay(
                text1="Key Repeated",
                text2=f"{self.parser.get_key_repeated_list()}",
            )
        # The sensor panel intentionally replaces the collection-only side view.
        self.task.render(teleop=False)

    def check_success(self) -> bool:
        target, plate = self.task.get_obj_pose()
        tcp, _ = self.parser.get_pR_body(body_name=self.task.tcp_body_name)
        placed = (
            np.linalg.norm(target[:2] - plate[:2]) < 0.08
            and -0.02 < float(target[2] - plate[2]) < 0.15
        )
        released = self._gripper_position() > 0.04
        retreated = (
            np.linalg.norm(tcp[:2] - target[:2]) < 0.15
            and float(tcp[2] - target[2]) > 0.08
        )
        self._success_count = self._success_count + 1 if placed and released and retreated else 0
        return self._success_count >= self.success_hold_steps

    @property
    def object_initial_position(self) -> np.ndarray:
        return np.asarray(self.task.obj_init_pose, dtype=np.float32)

    def set_object_initial_position(self, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=np.float32)
        if value.shape != (6,) or not np.all(np.isfinite(value)):
            raise ValueError("object_initial_position must contain six finite values")
        self._set_object_pose_without_step(value[:3], value[3:])
