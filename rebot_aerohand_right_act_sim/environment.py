"""Task-level adapter for the 6-DoF reBot arm + right AeroHand.

Self-contained MuJoCo environment: it does not depend on the legacy
``mujoco_env.SimpleEnv`` nor on the MediaPipe CV control scripts.  Keyboard
teleoperation is split into a Cartesian arm IK controller and a binary
per-finger hand closure mapper; the saved action is always the commanded
target (six arm joint angles plus seven hand actuator targets).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import glfw
import mujoco
import numpy as np

from .schema import (
    ACTION_DIM,
    FPS,
    HAND_ACTUATOR_DIM,
    HAND_CONTACT_DIM,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    STATE_DIM,
)
from .sensors import (
    HandContactProcessingConfig,
    HandContactSignalProcessor,
    classify_hand_geom_regions,
    read_hand_feedback,
    read_imu,
)

ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
HAND_JOINT_NAMES = (
    "right_index_mcp_flex",
    "right_index_pip",
    "right_index_dip",
    "right_middle_mcp_flex",
    "right_middle_pip",
    "right_middle_dip",
    "right_ring_mcp_flex",
    "right_ring_pip",
    "right_ring_dip",
    "right_pinky_mcp_flex",
    "right_pinky_pip",
    "right_pinky_dip",
    "right_thumb_cmc_abd",
    "right_thumb_cmc_flex",
    "right_thumb_mcp",
    "right_thumb_ip",
)
HAND_ACTUATOR_NAMES = (
    "right_index_A_tendon",
    "right_middle_A_tendon",
    "right_ring_A_tendon",
    "right_pinky_A_tendon",
    "right_thumb_A_cmc_abd",
    "right_th1_A_tendon",
    "right_th2_A_tendon",
)
HAND_FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

TOOL_BODY_NAME = "tetheria_mount"
OBJECT_BODY_NAME = "box"
OBJECT_GEOM_NAME = "red_box"
TARGET_BODY_NAME = "target_box"
AGENT_CAMERA = "top"
WRIST_CAMERA = "cam_wrist"
GRASP_SITE_NAME = "hand_grasp_site"

# Runtime grasp-site marker (visual aid only, never part of the data).
# Distances are measured from the site to the cylinder surface, so the color
# bands answer "how far is the palm anchor from touching the object".
GRASP_MARKER_RADIUS = 0.012
GRASP_MARKER_LINE_WIDTH = 3.0  # mjGEOM_LINE width is denominated in pixels
GRASP_MARKER_CLOSE_DISTANCE = 0.06   # <= : green, hand is at grasp distance
GRASP_MARKER_MID_DISTANCE = 0.12     # <= : orange, approaching
GRASP_MARKER_CLOSE_COLOR = (0.2, 0.9, 0.35, 0.2)
GRASP_MARKER_MID_COLOR = (1.0, 0.65, 0.2, 0.2)
GRASP_MARKER_FAR_COLOR = (1.0, 0.25, 0.25, 0.2)

# Picture-in-picture camera overlays drawn in the top-right corner of the
# viewer window, mirroring the agent/egocentric PIP views of rebot_act_sim.
PIP_WIDTH = 320
PIP_HEIGHT = 240
PIP_MARGIN = 10
PIP_TOP = 60  # leave room for the status text overlay above the first PIP


def camera_overlay_rects(
    window_width: int, window_height: int, count: int
) -> list[tuple[int, int]]:
    """Top-left-origin (x, y_top) positions of stacked top-right PIP views.

    ``mjr_drawPixels`` uses bottom-left origin; callers convert with
    ``bottom = window_height - y_top - PIP_HEIGHT``.
    """
    return [
        (
            max(0, window_width - PIP_WIDTH - PIP_MARGIN),
            PIP_TOP + index * (PIP_HEIGHT + PIP_MARGIN),
        )
        for index in range(count)
    ]


KEY_DISPLAY_NAMES = {
    glfw.KEY_W: "W", glfw.KEY_A: "A", glfw.KEY_S: "S", glfw.KEY_D: "D",
    glfw.KEY_R: "R", glfw.KEY_F: "F",
    glfw.KEY_UP: "UP", glfw.KEY_DOWN: "DOWN",
    glfw.KEY_LEFT: "LEFT", glfw.KEY_RIGHT: "RIGHT",
    glfw.KEY_Q: "Q", glfw.KEY_E: "E",
    glfw.KEY_1: "1", glfw.KEY_2: "2", glfw.KEY_3: "3",
    glfw.KEY_4: "4", glfw.KEY_5: "5",
    glfw.KEY_SPACE: "SPACE", glfw.KEY_O: "O", glfw.KEY_C: "C",
    glfw.KEY_H: "H", glfw.KEY_Z: "Z",
}


def grasp_marker_color(surface_distance: float) -> tuple[float, float, float, float]:
    """Color bands for the grasp-site marker based on surface distance."""
    if surface_distance <= GRASP_MARKER_CLOSE_DISTANCE:
        return GRASP_MARKER_CLOSE_COLOR
    if surface_distance <= GRASP_MARKER_MID_DISTANCE:
        return GRASP_MARKER_MID_COLOR
    return GRASP_MARKER_FAR_COLOR


def snap_hand_target(
    hand: np.ndarray,
    open_ctrl: np.ndarray,
    closed_ctrl: np.ndarray,
    previous: np.ndarray | None = None,
    *,
    band: float = 0.35,
) -> np.ndarray:
    """Snap a 7-dim hand target to the open/closed extremes with hysteresis.

    Tendon actuators in the demonstration data sit at their open/closed
    extremes with fast transitions.  A regression policy on sparse data tends
    to wobble between the two states (close -> open -> close), which slaps the
    cylinder during grasping.  Values within ``band`` of the open (or closed)
    extreme snap to it; everything else keeps the previous value, so a single
    noisy frame cannot flip the state.
    """
    hand = np.asarray(hand, dtype=np.float64).reshape(7)
    result = (
        np.asarray(previous, dtype=np.float64).reshape(7)
        if previous is not None
        else hand.copy()
    )
    open_ctrl = np.asarray(open_ctrl, dtype=np.float64).reshape(7)
    closed_ctrl = np.asarray(closed_ctrl, dtype=np.float64).reshape(7)
    span = np.abs(closed_ctrl - open_ctrl)
    for index in range(7):
        value = hand[index]
        if abs(value - open_ctrl[index]) < band * span[index]:
            result[index] = open_ctrl[index]
        elif abs(value - closed_ctrl[index]) < band * span[index]:
            result[index] = closed_ctrl[index]
        # else: keep the previous snapped value (hysteresis)
    return result.astype(np.float32, copy=False)

ARM_HOME = np.asarray([0.0, -1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float64)
TRANSLATION_STEP = 0.003
ROTATION_STEP = 0.02
RELEASED_TENDON_LENGTH = 0.105


def success_hold_steps(seconds: float, control_hz: int) -> int:
    if seconds <= 0:
        raise ValueError("success hold seconds must be positive")
    if control_hz <= 0:
        raise ValueError("control_hz must be positive")
    return max(1, int(np.ceil(seconds * control_hz)))


def teleop_arm_delta_from_keys(repeated_keys: set[int]) -> np.ndarray:
    """Map keys to a tool-local Cartesian delta for the arm IK target.

    Arrow keys follow screen-like semantics: up/down pitch the tool
    forward/backward; left/right roll it counter-clockwise/clockwise.
    Q/E retain local yaw control.  The IK controller post-multiplies the
    rotation, so these are ``tetheria_mount``-local axes.
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
    return np.concatenate([dpos, drpy]).astype(np.float32, copy=False)


@dataclass(frozen=True)
class SimObservation:
    image: np.ndarray
    wrist_image: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    hand_feedback: np.ndarray
    hand_joint_position: np.ndarray
    imu: np.ndarray
    hand_contact: np.ndarray
    sim_time: float


def _rotation_matrix(axis: int, angle: float) -> np.ndarray:
    """Return a 3x3 right-handed rotation about local x/y/z."""
    c, s = float(np.cos(angle)), float(np.sin(angle))
    if axis == 0:
        return np.asarray([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == 1:
        return np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    return np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _orientation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Small-angle world-frame orientation error compatible with mj_jacBody."""
    return 0.5 * sum(
        np.cross(current[:, axis], target[:, axis]) for axis in range(3)
    )


class CartesianArmIKController:
    """Incremental Cartesian target and damped-least-squares arm IK.

    Solves on a kinematic shadow state so ``command_q`` converges to a fixed
    IK solution instead of integrating errors from gravity and the position
    actuators' transient tracking lag.
    """

    def __init__(
        self,
        model,
        data,
        *,
        damping: float,
        max_joint_step: float,
        gravcomp: float = 1.0,
    ):
        self.model = model
        self.damping = max(float(damping), 1e-6)
        self.max_joint_step = max(float(max_joint_step), 1e-5)
        self.pending_pos = np.zeros(3, dtype=np.float64)
        self.pending_rpy = np.zeros(3, dtype=np.float64)

        joint_ids = np.asarray(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in ARM_JOINT_NAMES
            ],
            dtype=np.int32,
        )
        if np.any(joint_ids < 0):
            raise RuntimeError("The model must contain joint1 through joint6")
        self.qpos_adr = model.jnt_qposadr[joint_ids].astype(np.int32)
        self.dof_adr = model.jnt_dofadr[joint_ids].astype(np.int32)
        self.q_min = model.jnt_range[joint_ids, 0].copy()
        self.q_max = model.jnt_range[joint_ids, 1].copy()

        # Compensate the complete payload below joint1, including the hand,
        # to avoid the large steady-state Cartesian sag of a position servo.
        arm_root_body = int(model.jnt_bodyid[joint_ids[0]])
        for body_id in range(1, model.nbody):
            ancestor = body_id
            while ancestor > 0 and ancestor != arm_root_body:
                ancestor = int(model.body_parentid[ancestor])
            if ancestor == arm_root_body:
                model.body_gravcomp[body_id] = float(np.clip(gravcomp, 0.0, 1.0))

        actuator_ids = []
        for joint_id in joint_ids:
            # Filter by transmission type so a hand tendon whose numeric id
            # equals an arm joint id cannot be mistaken for an arm actuator.
            matches = np.flatnonzero(
                (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                & (model.actuator_trnid[:, 0] == joint_id)
            )
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one actuator transmitted by joint {joint_id}, "
                    f"found {len(matches)}"
                )
            actuator_ids.append(int(matches[0]))
        self.actuator_ids = np.asarray(actuator_ids, dtype=np.int32)

        self.body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, TOOL_BODY_NAME
        )
        if self.body_id < 0:
            raise RuntimeError(f"Model is missing tool body {TOOL_BODY_NAME!r}")

        self.jac_pos = np.zeros((3, model.nv), dtype=np.float64)
        self.jac_rot = np.zeros((3, model.nv), dtype=np.float64)
        self.ik_data = mujoco.MjData(model)
        self.target_pos = np.zeros(3, dtype=np.float64)
        self.target_rot = np.eye(3, dtype=np.float64)
        self.command_q = data.qpos[self.qpos_adr].copy()
        self.set_home(data)

    def _tool_pose(self, data) -> tuple[np.ndarray, np.ndarray]:
        return (
            data.xpos[self.body_id].copy(),
            data.xmat[self.body_id].reshape(3, 3).copy(),
        )

    def set_home(self, data) -> None:
        home = np.clip(ARM_HOME, self.q_min, self.q_max)
        data.qpos[self.qpos_adr] = home
        data.qvel[self.dof_adr] = 0.0
        data.ctrl[self.actuator_ids] = home
        mujoco.mj_forward(self.model, data)
        self.command_q = home.copy()
        self.target_pos, self.target_rot = self._tool_pose(data)

    def apply_delta(self, delta: np.ndarray) -> None:
        """Accumulate a Cartesian delta already expressed in metres/radians."""
        delta = np.asarray(delta, dtype=np.float64).reshape(6)
        self.pending_pos += delta[:3]
        self.pending_rpy += delta[3:]

    def step(self, data) -> None:
        self.target_pos += self.pending_pos
        # Match the reference controller: post-multiply local rotations.
        for axis in range(3):
            if self.pending_rpy[axis] != 0.0:
                self.target_rot = self.target_rot @ _rotation_matrix(
                    axis, self.pending_rpy[axis]
                )
        self.pending_pos.fill(0.0)
        self.pending_rpy.fill(0.0)

        self.ik_data.qpos[:] = data.qpos
        self.ik_data.qpos[self.qpos_adr] = self.command_q
        self.ik_data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.ik_data)
        current_pos, current_rot = self._tool_pose(self.ik_data)
        error = np.concatenate(
            [
                self.target_pos - current_pos,
                _orientation_error(current_rot, self.target_rot),
            ]
        )
        mujoco.mj_jacBody(
            self.model, self.ik_data, self.jac_pos, self.jac_rot, self.body_id
        )
        jac = np.vstack([self.jac_pos, self.jac_rot])[:, self.dof_adr]
        regularized = jac @ jac.T + (self.damping**2) * np.eye(6)
        try:
            dq = jac.T @ np.linalg.solve(regularized, error)
        except np.linalg.LinAlgError:
            dq = jac.T @ np.linalg.lstsq(regularized, error, rcond=None)[0]
        dq = np.clip(dq, -self.max_joint_step, self.max_joint_step)
        self.command_q = np.clip(self.command_q + dq, self.q_min, self.q_max)
        data.ctrl[self.actuator_ids] = self.command_q


class HandClosureMapper:
    """Map five binary finger closures to seven hand actuator targets.

    Tendons follow the reference simulation: high length = open and low
    length = closed.  Thumb abduction instead increases while closing.
    """

    def __init__(self, model):
        self.model = model
        ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in HAND_ACTUATOR_NAMES
        ]
        if any(index < 0 for index in ids):
            missing = [
                name
                for name, index in zip(HAND_ACTUATOR_NAMES, ids)
                if index < 0
            ]
            raise RuntimeError(f"MuJoCo model is missing hand actuators: {missing}")
        self.ids = np.asarray(ids, dtype=np.int32)
        self.ctrl_min = model.actuator_ctrlrange[self.ids, 0].copy()
        self.ctrl_max = model.actuator_ctrlrange[self.ids, 1].copy()
        self._open_ctrl = self.ctrl_max.copy()
        self._closed_ctrl = self.ctrl_min.copy()
        self._open_ctrl[4] = 0.0
        self._closed_ctrl[4] = min(1.5, self.ctrl_max[4])
        self.closed = np.zeros(5, dtype=bool)  # thumb, index, middle, ring, pinky

    @property
    def open_ctrl(self) -> np.ndarray:
        return self._open_ctrl.copy()

    @property
    def closed_ctrl(self) -> np.ndarray:
        return self._closed_ctrl.copy()

    def command_ctrl(self) -> np.ndarray:
        closure = np.zeros(HAND_ACTUATOR_DIM, dtype=np.float64)
        closure[0:4] = self.closed[1:5]  # four finger tendons
        closure[4:7] = self.closed[0]    # thumb abduction + two thumb tendons
        ctrl = self._open_ctrl + closure * (self._closed_ctrl - self._open_ctrl)
        return ctrl.astype(np.float32, copy=False)

    def set_all_open(self) -> None:
        self.closed.fill(False)

    def set_all_closed(self) -> None:
        self.closed.fill(True)

    def toggle_finger(self, index: int) -> None:
        if not 0 <= index < 5:
            raise ValueError("finger index must be in [0, 5)")
        self.closed[index] = not self.closed[index]

    def toggle_grasp(self) -> None:
        self.closed = ~self.closed

    def write(self, data, ctrl: np.ndarray) -> None:
        data.ctrl[self.ids] = np.clip(ctrl, self.ctrl_min, self.ctrl_max)


class AeroHandViewer:
    """Minimal GLFW MuJoCo viewer with keyboard-state tracking.

    Control keys are only consumed by the environment; the free camera
    (orbit/pan/zoom) works normally.  An optional RGB sensor panel is drawn
    into the top-left corner of the window on every sync.
    """

    def __init__(self, model, data, *, title: str = "reBot + AeroHand ACT"):
        self.model = model
        self.data = data
        self._pressed: set[int] = set()
        self._once: list[int] = []

        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")
        glfw.default_window_hints()
        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor) if monitor is not None else None
        width = min(1400, mode.size.width) if mode is not None else 1280
        height = min(900, mode.size.height) if mode is not None else 800
        self.window = glfw.create_window(width, height, title, None, None)
        if self.window is None:
            glfw.terminate()
            raise RuntimeError("Could not create the MuJoCo GLFW window")
        pos_x = max(0, (mode.size.width - width) // 2) if mode is not None else 60
        pos_y = max(0, (mode.size.height - height) // 2) if mode is not None else 60
        glfw.set_window_pos(self.window, pos_x, pos_y)
        glfw.focus_window(self.window)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.camera = mujoco.MjvCamera()
        self.option = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.context = mujoco.MjrContext(
            model, mujoco.mjtFontScale.mjFONTSCALE_150
        )
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.azimuth = 135.0
        self.camera.elevation = -25.0
        self.camera.distance = 1.25
        self.camera.lookat[:] = np.asarray([0.0, 0.0, 0.28])

        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_scroll_callback(self.window, self._on_scroll)

        self._left_pressed = False
        self._right_pressed = False
        self._last_x = 0.0
        self._last_y = 0.0

    # ------------------------------------------------------------------ keys
    def _on_key(self, _window, key, _scancode, action, _mods):
        if action == glfw.PRESS:
            self._once.append(key)
            self._pressed.add(key)
        elif action == glfw.REPEAT:
            self._pressed.add(key)
        elif action == glfw.RELEASE:
            self._pressed.discard(key)

    def is_key_repeat(self, key: int) -> bool:
        return key in self._pressed

    def is_key_pressed_once(self, key: int) -> bool:
        if key not in self._once:
            return False
        while key in self._once:
            self._once.remove(key)
        return True

    @property
    def pressed_keys(self) -> frozenset[int]:
        return frozenset(self._pressed)

    # ------------------------------------------------------------------ mouse
    def _on_mouse_button(self, window, _button, _action, _mods):
        self._left_pressed = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        )
        self._right_pressed = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        )
        self._last_x, self._last_y = glfw.get_cursor_pos(window)

    def _on_cursor(self, window, xpos, ypos):
        if not (self._left_pressed or self._right_pressed):
            return
        width, height = glfw.get_window_size(window)
        dx = (xpos - self._last_x) / max(1, height)
        dy = (ypos - self._last_y) / max(1, height)
        self._last_x, self._last_y = xpos, ypos
        shift = any(
            glfw.get_key(window, key) == glfw.PRESS
            for key in (glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT)
        )
        if self._right_pressed:
            action = (
                mujoco.mjtMouse.mjMOUSE_MOVE_H
                if shift
                else mujoco.mjtMouse.mjMOUSE_MOVE_V
            )
        else:
            action = (
                mujoco.mjtMouse.mjMOUSE_ROTATE_H
                if shift
                else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            )
        mujoco.mjv_moveCamera(self.model, action, dx, dy, self.camera)

    def _on_scroll(self, _window, _xoffset, yoffset):
        mujoco.mjv_moveCamera(
            self.model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0.0,
            -0.05 * yoffset,
            self.camera,
        )

    # ------------------------------------------------------------------ render
    def is_running(self) -> bool:
        return self.window is not None and not glfw.window_should_close(self.window)

    def _next_runtime_geom(self) -> mujoco.MjvGeom:
        """Return a mutable view of the next free scene geom slot.

        ``MjvScene.geoms`` is an immutable sequence in mujoco 3.x, but its
        elements alias the underlying C array: mutating the returned element
        writes through to the scene slot.  Callers must increment
        ``scene.ngeom`` after filling the slot, matching the legacy
        mujoco_parser marker pattern.
        """
        if self.scene.ngeom >= self.scene.maxgeom:
            raise RuntimeError(
                f"ran out of scene geoms: maxgeom={self.scene.maxgeom}"
            )
        return self.scene.geoms[self.scene.ngeom]

    def _add_runtime_sphere(
        self,
        pos: np.ndarray,
        rgba: tuple[float, float, float, float],
        *,
        radius: float,
    ) -> None:
        """Draw one transient sphere into the viewer scene for this frame."""
        geom = self._next_runtime_geom()
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.full(3, radius, dtype=np.float64),
            np.asarray(pos, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            np.asarray(rgba, dtype=np.float32),
        )
        geom.segid = -1
        self.scene.ngeom += 1

    def _add_runtime_line(
        self,
        start: np.ndarray,
        end: np.ndarray,
        rgba: tuple[float, float, float, float],
        *,
        width: float,
    ) -> None:
        """Draw one transient line into the viewer scene for this frame."""
        geom = self._next_runtime_geom()
        # mjv_connector (mujoco >= 3.11) expects mjv_initGeom to have been
        # called first with the same type; it then fills size/pos/mat.
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            np.full(3, width, dtype=np.float64),
            np.asarray(start, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            np.asarray(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            width,
            np.asarray(start, dtype=np.float64),
            np.asarray(end, dtype=np.float64),
        )
        self.scene.ngeom += 1

    def sync(
        self,
        sensor_panel: np.ndarray | None = None,
        grasp_marker: tuple[np.ndarray, np.ndarray, tuple] | None = None,
        overlays: list[tuple[mujoco.mjtGridPos, str, str]] | None = None,
        camera_overlays: list[tuple[str, np.ndarray]] | None = None,
    ) -> None:
        """Render one frame.

        ``grasp_marker`` is an optional ``(site_position, object_position,
        rgba)`` triplet drawn as a transient sphere plus a line to the object,
        matching the runtime sphere/connector markers of the legacy
        mujoco_parser viewer.  ``overlays`` is an optional list of
        ``(gridpos, text1, text2)`` text overlays drawn via ``mjr_overlay``,
        the same mechanism the legacy viewer used for key/status hints.
        ``camera_overlays`` is an optional list of ``(camera_name, rgb)``
        picture-in-picture views stacked in the top-right corner, mirroring
        the agent/egocentric PIP views of rebot_act_sim.
        """
        glfw.make_context_current(self.window)
        width, height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.option,
            None,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )
        if grasp_marker is not None:
            site_pos, object_pos, rgba = grasp_marker
            self._add_runtime_sphere(
                site_pos, rgba, radius=GRASP_MARKER_RADIUS
            )
            self._add_runtime_line(
                site_pos, object_pos, rgba, width=GRASP_MARKER_LINE_WIDTH
            )
        mujoco.mjr_render(viewport, self.scene, self.context)
        # IMPORTANT: draw the text overlays before any pixel overlays. In
        # mujoco 3.11, mjr_drawPixels silently renders nothing when called
        # right after mjr_render; one mjr_overlay call resets the offending
        # GL state, which is why the legacy mujoco_parser viewer (whose order
        # is render -> overlay -> drawPixels) works.
        if overlays:
            for gridpos, text1, text2 in overlays:
                mujoco.mjr_overlay(
                    mujoco.mjtFont.mjFONT_NORMAL,
                    gridpos,
                    viewport,
                    text1,
                    text2,
                    self.context,
                )
        if sensor_panel is not None:
            # mjr_drawPixels draws buffer row 0 at the BOTTOM of the viewport
            # (bottom-left origin), so flipud first — the same convention the
            # legacy mujoco_parser viewer uses.  Draw the panel top-left.
            panel = np.ascontiguousarray(sensor_panel, dtype=np.uint8)
            panel_viewport = mujoco.MjrRect(
                0,
                max(0, height - panel.shape[0]),
                panel.shape[1],
                panel.shape[0],
            )
            mujoco.mjr_drawPixels(
                np.flipud(panel).reshape(-1), None, panel_viewport, self.context
            )
        if camera_overlays:
            for index, (camera_name, frame) in enumerate(camera_overlays):
                frame = np.ascontiguousarray(frame, dtype=np.uint8)
                pip_x, pip_y_top = camera_overlay_rects(
                    width, height, len(camera_overlays)
                )[index]
                pip_viewport = mujoco.MjrRect(
                    pip_x,
                    max(0, height - pip_y_top - frame.shape[0]),
                    frame.shape[1],
                    frame.shape[0],
                )
                mujoco.mjr_drawPixels(
                    np.flipud(frame).reshape(-1), None, pip_viewport,
                    self.context,
                )
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self) -> None:
        if self.window is not None:
            glfw.make_context_current(self.window)
            self.context.free()
            # Drop the reference while GLFW is still initialized so the
            # mujoco MjrContext destructor's second free() cannot hit the
            # "library is not initialized" path after glfw.terminate().
            self.context = None
            glfw.destroy_window(self.window)
            self.window = None
            glfw.terminate()


class SimAeroHandACTEnvironment:
    """MuJoCo adapter for arm + hand ACT collection and deployment.

    ``observation.state`` is measured arm joint position.  ``action`` is
    always the target sent for the next control interval: six arm joint
    angles (rad) plus seven hand actuator targets.
    """

    def __init__(
        self,
        xml_path: str | Path,
        *,
        seed: int | None = None,
        success_hold_seconds: float = 0.5,
        control_hz: int = FPS,
        target_object_position_center: tuple[float, float, float]
        | list[float]
        | None = None,
        target_object_xy_half_range: tuple[float, float] | list[float] | None = None,
        hand_contact_processing: dict | None = None,
        teleop_motion_alpha: float = 0.25,
        hand_command_alpha: float = 0.25,
        enable_hand_contact_processing: bool = True,
        show_grasp_marker: bool = True,
        camera_overlays: list[str] | None = None,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.success_hold_steps = success_hold_steps(success_hold_seconds, control_hz)
        self._enable_hand_contact = bool(enable_hand_contact_processing)
        self.contact_config = HandContactProcessingConfig.from_mapping(
            hand_contact_processing
        )
        self.contact_processor = HandContactSignalProcessor(self.contact_config)
        self.teleop_motion_alpha = float(teleop_motion_alpha)
        if not 0 < self.teleop_motion_alpha <= 1:
            raise ValueError("teleop_motion_alpha must be in (0, 1]")
        self.hand_command_alpha = float(hand_command_alpha)
        if not 0 < self.hand_command_alpha <= 1:
            raise ValueError("hand_command_alpha must be in (0, 1]")
        self._teleop_motion = np.zeros(6, dtype=np.float32)

        timestep = float(self.model.opt.timestep)
        self._steps_per_tick = max(1, int(round(1.0 / (timestep * control_hz))))
        self._physics_tick = 0

        # Task object placement.
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
        if (
            self.target_object_position_center is not None
            and self.target_object_xy_half_range is None
        ):
            raise ValueError(
                "target_object_xy_half_range is required for object randomization"
            )

        # Names and ids.
        self.object_body = self._body_id(OBJECT_BODY_NAME)
        self.object_geom = self._geom_id(OBJECT_GEOM_NAME)
        self.target_geom = self._geom_id(TARGET_BODY_NAME)
        # The goal disk collision is enabled in the task scene XML itself.
        # mujoco 3.11 ignores runtime geom_contype/conaffinity edits, so the
        # redundant assignment below only guards against scene files that
        # forgot it; it does not make a phantom disk collide.
        if (
            int(self.model.geom_contype[self.target_geom]) == 0
            or int(self.model.geom_conaffinity[self.target_geom]) == 0
        ):
            raise RuntimeError(
                "target disk geom must have contype=1 conaffinity=1 in the "
                "scene XML; runtime edits are ignored by mujoco 3.11"
            )
        self.object_dof_adr = self._free_joint_dof_address(OBJECT_BODY_NAME)
        self._show_grasp_marker = bool(show_grasp_marker)
        self._grasp_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, GRASP_SITE_NAME
        )
        if self._grasp_site_id < 0:
            raise RuntimeError(
                f"model is missing grasp visualization site {GRASP_SITE_NAME!r} "
                "(add it to aerohand_right_body.xml)"
            )

        arm_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ARM_JOINT_NAMES
        ]
        hand_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in HAND_JOINT_NAMES
        ]
        if any(index < 0 for index in arm_joint_ids) or any(
            index < 0 for index in hand_joint_ids
        ):
            raise RuntimeError("model is missing arm or hand joints")
        self.arm_qpos_adr = self.model.jnt_qposadr[
            np.asarray(arm_joint_ids, dtype=np.int32)
        ].astype(np.int32)
        self.arm_dof_adr = self.model.jnt_dofadr[
            np.asarray(arm_joint_ids, dtype=np.int32)
        ].astype(np.int32)
        self.hand_qpos_adr = self.model.jnt_qposadr[
            np.asarray(hand_joint_ids, dtype=np.int32)
        ].astype(np.int32)
        self.hand_dof_adr = self.model.jnt_dofadr[
            np.asarray(hand_joint_ids, dtype=np.int32)
        ].astype(np.int32)

        self.geom_regions = classify_hand_geom_regions(self.model)
        self.hand_geoms = set(self.geom_regions)

        self.hand_mapper = HandClosureMapper(self.model)
        self.arm_controller = CartesianArmIKController(
            self.model,
            self.data,
            damping=0.03,
            max_joint_step=0.025,
            gravcomp=1.0,
        )
        self._hand_target = self.hand_mapper.command_ctrl()
        self._hand_filtered = self._hand_target.copy()

        self._renderers = {
            AGENT_CAMERA: mujoco.Renderer(
                self.model, IMAGE_HEIGHT, IMAGE_WIDTH
            ),
            WRIST_CAMERA: mujoco.Renderer(
                self.model, IMAGE_HEIGHT, IMAGE_WIDTH
            ),
        }
        # Optional picture-in-picture camera views shown in the viewer window.
        self._overlay_cameras = list(camera_overlays or [])
        for name in self._overlay_cameras:
            if (
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_CAMERA, name
                )
                < 0
            ):
                raise RuntimeError(f"model is missing overlay camera: {name}")
        self._overlay_renderers = {
            name: mujoco.Renderer(self.model, PIP_HEIGHT, PIP_WIDTH)
            for name in self._overlay_cameras
        }
        self._viewer = None
        self.obj_init_pose = np.zeros(7, dtype=np.float32)
        self._success_count = 0

    @staticmethod
    def _position3(value, name: str) -> np.ndarray | None:
        if value is None:
            return None
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (3,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain three finite values")
        return result

    def _body_id(self, name: str) -> int:
        index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if index < 0:
            raise RuntimeError(f"model is missing body: {name}")
        return index

    def _geom_id(self, name: str) -> int:
        index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if index < 0:
            raise RuntimeError(f"model is missing geom: {name}")
        return index

    def _free_joint_dof_address(self, body_name: str) -> int:
        body = self.model.body(body_name)
        if int(body.jntnum[0]) != 1:
            raise RuntimeError(f"{body_name} must have exactly one free joint")
        joint_index = int(body.jntadr[0])
        return int(self.model.jnt_dofadr[joint_index])

    # ------------------------------------------------------------- object pose
    def _place_object(self, pose: np.ndarray) -> None:
        """Place the cylinder free joint without a teleportation impulse.

        Accepts either a 3-element xyz center or a full 7-element qpos
        (xyz + unit quaternion).  Orientation is always reset to identity.
        """

        pose = np.asarray(pose, dtype=np.float64).reshape(-1)
        if pose.shape == (3,):
            qpos = np.zeros(7, dtype=np.float64)
            qpos[:3] = pose
        elif pose.shape == (7,):
            qpos = pose.copy()
        else:
            raise ValueError(
                f"object pose must have 3 or 7 elements, got {pose.shape}"
            )
        half_height = float(self.model.geom_size[self.object_geom, 1])
        qpos[2] = max(float(qpos[2]), half_height + 0.0005)
        qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        start = self.object_qpos_adr
        self.data.qpos[start : start + 7] = qpos
        self.data.qvel[self.object_dof_adr : self.object_dof_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    @property
    def object_qpos_adr(self) -> int:
        body = self.model.body(OBJECT_BODY_NAME)
        joint_index = int(body.jntadr[0])
        return int(self.model.jnt_qposadr[joint_index])

    def _set_controlled_object_positions(self, seed: int | None) -> None:
        if self.target_object_position_center is None:
            return
        rng = np.random.default_rng(seed)
        target = self.target_object_position_center.copy()
        target[:2] += rng.uniform(
            -self.target_object_xy_half_range, self.target_object_xy_half_range
        )
        self._place_object(target)
        start = self.object_qpos_adr
        self.obj_init_pose = self.data.qpos[start : start + 7].copy().astype(
            np.float32
        )

    # --------------------------------------------------------------- lifecycle
    @property
    def viewer(self) -> AeroHandViewer:
        if self._viewer is None:
            self._viewer = AeroHandViewer(self.model, self.data)
        return self._viewer

    def reset(self, seed: int | None = None) -> None:
        # Arm home pose, open hand, all velocities zeroed.
        self.arm_controller.set_home(self.data)
        self.data.qpos[self.hand_qpos_adr] = 0.0
        self.data.qvel[self.hand_dof_adr] = 0.0
        self.hand_mapper.set_all_open()
        self._hand_target = self.hand_mapper.command_ctrl()
        self._hand_filtered = self._hand_target.copy()
        self.hand_mapper.write(self.data, self._hand_target)
        self._set_controlled_object_positions(seed)
        self._physics_tick = 0
        self._success_count = 0
        self.contact_processor.reset()
        self._teleop_motion.fill(0)
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------- simulation
    def advance_physics(self) -> None:
        self._hand_filtered += self.hand_command_alpha * (
            self._hand_target - self._hand_filtered
        )
        self.hand_mapper.write(self.data, self._hand_filtered)
        mujoco.mj_step(self.model, self.data)
        self._physics_tick += 1
        if self._enable_hand_contact:
            self.contact_processor.update(self._read_region_forces())

    def is_control_tick(self, hz: int) -> bool:
        expected = max(1, int(round(1.0 / (float(self.model.opt.timestep) * hz))))
        return self._physics_tick % expected == 0

    def _read_region_forces(self) -> np.ndarray:
        forces = np.zeros(HAND_CONTACT_DIM, dtype=np.float64)
        wrench = np.zeros(6, dtype=np.float64)
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 in self.hand_geoms and geom2 not in self.hand_geoms:
                region = self.geom_regions[geom1]
            elif geom2 in self.hand_geoms and geom1 not in self.hand_geoms:
                region = self.geom_regions[geom2]
            else:
                continue
            mujoco.mj_contactForce(self.model, self.data, contact_index, wrench)
            forces[region] += max(0.0, float(wrench[0]))
        return forces

    # ------------------------------------------------------------- observation
    def _render_camera(self, name: str) -> np.ndarray:
        renderer = self._renderers[name]
        renderer.update_scene(self.data, camera=name)
        image = renderer.render()
        if image.shape[:2] != (IMAGE_HEIGHT, IMAGE_WIDTH):
            raise RuntimeError(
                f"camera {name} rendered {image.shape[:2]}, "
                f"expected {(IMAGE_HEIGHT, IMAGE_WIDTH)}"
            )
        return np.asarray(image, dtype=np.uint8)

    def get_sensor_value(self, name: str) -> np.ndarray:
        return self.data.sensor(name).data.copy()

    def observe(self) -> SimObservation:
        if self._enable_hand_contact:
            hand_contact = self.contact_processor.consume()
        else:
            hand_contact = np.zeros(HAND_CONTACT_DIM, dtype=np.float32)
        observation = SimObservation(
            image=self._render_camera(AGENT_CAMERA),
            wrist_image=self._render_camera(WRIST_CAMERA),
            joint_position=np.asarray(
                self.data.qpos[self.arm_qpos_adr], dtype=np.float32
            ),
            joint_velocity=np.asarray(
                self.data.qvel[self.arm_dof_adr], dtype=np.float32
            ),
            hand_feedback=read_hand_feedback(self.model, self.data),
            hand_joint_position=np.asarray(
                self.data.qpos[self.hand_qpos_adr], dtype=np.float32
            ),
            imu=read_imu(self),
            hand_contact=hand_contact,
            sim_time=float(self.data.time),
        )
        return observation

    # -------------------------------------------------------------- teleop
    _ARM_REPEAT_KEYS = (
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
    _FINGER_KEYS = (
        glfw.KEY_1,
        glfw.KEY_2,
        glfw.KEY_3,
        glfw.KEY_4,
        glfw.KEY_5,
    )

    def teleop_target(self) -> tuple[np.ndarray, bool]:
        viewer = self.viewer
        if viewer.is_key_pressed_once(glfw.KEY_Z):
            return np.zeros(ACTION_DIM, dtype=np.float32), True
        if viewer.is_key_pressed_once(glfw.KEY_H):
            self.arm_controller.set_home(self.data)
            self.hand_mapper.set_all_open()
        if viewer.is_key_pressed_once(glfw.KEY_SPACE):
            self.hand_mapper.toggle_grasp()
        if viewer.is_key_pressed_once(glfw.KEY_O):
            self.hand_mapper.set_all_open()
        if viewer.is_key_pressed_once(glfw.KEY_C):
            self.hand_mapper.set_all_closed()
        for index, key in enumerate(self._FINGER_KEYS):
            if viewer.is_key_pressed_once(key):
                self.hand_mapper.toggle_finger(index)

        repeated_keys = {
            key for key in self._ARM_REPEAT_KEYS if viewer.is_key_repeat(key)
        }
        delta = teleop_arm_delta_from_keys(repeated_keys)
        alpha = self.teleop_motion_alpha
        self._teleop_motion = (
            alpha * delta + (1.0 - alpha) * self._teleop_motion
        )
        self.arm_controller.apply_delta(self._teleop_motion)
        self.arm_controller.step(self.data)
        self._hand_target = self.hand_mapper.command_ctrl()
        target = np.concatenate(
            [
                np.asarray(self.arm_controller.command_q, dtype=np.float32),
                self._hand_target,
            ]
        )
        return target.astype(np.float32, copy=False), False

    # --------------------------------------------------------------- command
    def command(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (ACTION_DIM,) or not np.all(np.isfinite(action)):
            raise ValueError(f"invalid ACT action: {action}")
        arm_target = np.clip(
            action[:STATE_DIM],
            self.arm_controller.q_min,
            self.arm_controller.q_max,
        )
        self.arm_controller.command_q = arm_target.copy()
        self.data.ctrl[self.arm_controller.actuator_ids] = arm_target
        self._hand_target = np.clip(
            action[STATE_DIM:],
            self.hand_mapper.ctrl_min,
            self.hand_mapper.ctrl_max,
        ).copy()

    # --------------------------------------------------------------- render
    def grasp_surface_distance(self) -> float:
        """Distance from the grasp site to the cylinder surface, in metres."""
        site_pos = np.asarray(self.data.site_xpos[self._grasp_site_id])
        object_pos = np.asarray(self.data.xpos[self.object_body])
        center_distance = float(np.linalg.norm(site_pos - object_pos))
        object_radius = float(self.model.geom_size[self.object_geom, 0])
        return max(0.0, center_distance - object_radius)

    def _grasp_marker(self) -> tuple[np.ndarray, np.ndarray, tuple] | None:
        """Build the transient grasp-site marker for the current frame.

        Returns ``(site_position, object_position, rgba)`` where rgba is a
        distance band (green/orange/red) based on the gap between the grasp
        site and the cylinder surface.  Purely visual; never recorded.
        """
        if not self._show_grasp_marker:
            return None
        site_pos = np.asarray(self.data.site_xpos[self._grasp_site_id])
        object_pos = np.asarray(self.data.xpos[self.object_body])
        return (
            site_pos,
            object_pos,
            grasp_marker_color(self.grasp_surface_distance()),
        )

    def _render_camera_overlays(self) -> list[tuple[str, np.ndarray]]:
        """Render each configured PIP camera at 320x240.

        The frames are drawn flipped by ``viewer.sync`` (see there), so no
        embedded title bar is added here; the camera names are appended to
        the status text instead.
        """
        frames: list[tuple[str, np.ndarray]] = []
        for name in self._overlay_cameras:
            renderer = self._overlay_renderers[name]
            renderer.update_scene(self.data, camera=name)
            frames.append(
                (name, np.asarray(renderer.render(), dtype=np.uint8))
            )
        return frames

    def render(
        self,
        *,
        teleop: bool = False,
        sensor_panel: np.ndarray | None = None,
        status_lines: tuple[str, str] | None = None,
        show_camera_overlays: bool = True,
    ) -> None:
        overlays: list[tuple[mujoco.mjtGridPos, str, str]] = []
        if teleop:
            pressed = ", ".join(
                sorted(
                    KEY_DISPLAY_NAMES.get(key, f"key{key}")
                    for key in self.viewer.pressed_keys
                )
            )
            overlays.append(
                (
                    mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                    "Key Pressed/Repeated",
                    pressed or "-",
                )
            )
        if status_lines is not None:
            text2 = str(status_lines[1])
            if self._overlay_cameras:
                text2 += f"  PIP[{', '.join(self._overlay_cameras)}]"
            overlays.append(
                (
                    mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                    str(status_lines[0]),
                    text2,
                )
            )
        camera_frames = (
            self._render_camera_overlays()
            if show_camera_overlays and self._overlay_cameras
            else None
        )
        self.viewer.sync(
            sensor_panel=sensor_panel,
            grasp_marker=self._grasp_marker(),
            overlays=overlays or None,
            camera_overlays=camera_frames,
        )

    def is_alive(self) -> bool:
        return self.viewer.is_running()

    def close(self) -> None:
        # Release the offscreen camera renderers while GLFW is still alive:
        # their internal GL contexts are destroyed by the Renderer destructor,
        # which would otherwise run at interpreter exit after glfw.terminate().
        self._renderers = None
        self._overlay_renderers = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # --------------------------------------------------------------- success
    def _has_object_target_contact(self) -> bool:
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if {int(contact.geom1), int(contact.geom2)} == {
                self.object_geom,
                self.target_geom,
            }:
                return True
        return False

    def success_status(self) -> dict:
        """Expose the three success sub-conditions for diagnostics."""
        feedback = read_hand_feedback(self.model, self.data)
        placed = self._has_object_target_contact()
        released = bool(np.all(feedback[:4] > RELEASED_TENDON_LENGTH))
        tcp = self.data.xpos[self.arm_controller.body_id]
        target = self.data.xpos[self.object_body]
        retreated = (
            np.linalg.norm(tcp[:2] - target[:2]) < 0.15
            and float(tcp[2] - target[2]) > 0.08
        )
        return {
            "placed": placed,
            "released": released,
            "retreated": retreated,
            "hold": self._success_count,
            "hold_steps": self.success_hold_steps,
        }

    def check_success(self) -> bool:
        status = self.success_status()
        all_ok = status["placed"] and status["released"] and status["retreated"]
        self._success_count = self._success_count + 1 if all_ok else 0
        return self._success_count >= self.success_hold_steps

    # ------------------------------------------------------------- object api
    @property
    def object_initial_position(self) -> np.ndarray:
        return self.obj_init_pose.copy()

    def set_object_initial_position(self, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=np.float32)
        if value.shape != (7,) or not np.all(np.isfinite(value)):
            raise ValueError(
                "object_initial_position must contain seven finite values"
            )
        self._place_object(value)
        self.obj_init_pose = value.copy()
