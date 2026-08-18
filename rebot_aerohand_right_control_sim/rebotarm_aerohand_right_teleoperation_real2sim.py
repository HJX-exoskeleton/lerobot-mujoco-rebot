#!/usr/bin/env python3
"""ST/SMS_STS servo-master real2sim teleoperation + CV-controlled right AeroHand.

The ID1..ID6 ST/SMS_STS servo master arm drives the simulated reBot arm in
MuJoCo through data.ctrl (real2sim).  The mapping and serial-reader logic
follow master_slave_control/Servo_control/servo_arm_teleoperation_real2sim.py.
The OpenCV camera window controls the simulated AeroHand through MediaPipe;
while CV hand control is off, the ID7 master servo opens and closes the
AeroHand.

Control chain
-------------
ID1..ID6 servo angle deg
    -> MuJoCo arm target q_sim (rad)
    -> alpha-master low-pass filter
    -> data.ctrl on the six arm position actuators (never data.qpos)
ID7 servo angle deg
    -> normalized opening 0..1 (90 deg closed, 180 deg open)
    -> staged AeroHand target: opening moves the fingers first and the thumb
       web last; closing moves the thumb web first and the fingers last
    (used only while CV hand control is off)

MuJoCo viewer
-------------
Mouse orbit/pan/zoom (shift for pan) and ESC quit.

OpenCV window keys
------------------
SPACE           : toggle CV / ID7 hand control
O / C           : open/close the simulated hand (manual latch)
V               : show/hide diagnostics
Q or ESC        : quit

Pass --no-cv to disable CV hand control and the camera window entirely; the
ID7 servo then drives the hand and ESC in the MuJoCo viewer quits.
Pass --extra-cameras to overlay the top/cam_wrist/angle camera panels in the
MuJoCo viewer.

运行方式
python3 rebotarm_aerohand_right_teleoperation_real2sim.py --port /dev/ttyUSB0 --baudrate 115200 --no-cv
# 额外增加相机画面
python3 rebotarm_aerohand_right_teleoperation_real2sim.py --port /dev/ttyUSB0 --no-cv --extra-cameras
# 无舵机主手、仅 CV 手部控制(机械臂保持初始姿态):
python3 rebotarm_aerohand_right_teleoperation_real2sim.py --no-servo

"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import glfw
import numpy as np

import aerohand_right_cv_control_sim as hand_cv


# ---------------------------------------------------------------------------
# Model / servo constants
# ---------------------------------------------------------------------------

ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
DEFAULT_TASK_MODEL = (
    hand_cv.WORKSPACE_DIR
    / "asset_rebot_aerohand_right"
    / "mujoco_xml"
    / "rebotarm_aerohand_sim_transfer_cube.xml"
)

ARM_SERVO_IDS = [1, 2, 3, 4, 5, 6]
ID7_SERVO_ID = 7
ARM_DOF = len(ARM_SERVO_IDS)

SERVO_DIGITAL_RANGE = 4095.0
SERVO_ANGLE_RANGE = 360.0

# Software limits and home angles of the ST/SMS_STS master arm (deg).
# ID7 defaults to 90 deg (closed); 180 deg opens it fully.
JOINT_LIMITS_DEG = {
    1: {"min_deg": 50.0, "max_deg": 300.0, "home_deg": 180.0},
    2: {"min_deg": 10.0, "max_deg": 180.0, "home_deg": 180.0},
    3: {"min_deg": 22.0, "max_deg": 180.0, "home_deg": 180.0},
    4: {"min_deg": 100.0, "max_deg": 270.0, "home_deg": 180.0},
    5: {"min_deg": 90.0, "max_deg": 270.0, "home_deg": 180.0},
    6: {"min_deg": 90.0, "max_deg": 270.0, "home_deg": 180.0},
    7: {"min_deg": 90.0, "max_deg": 180.0, "home_deg": 90.0},
}

SAFETY_MARGIN_DEG = 0.0

# Master angle -> sim joint direction and home offset (same as the real2sim
# reference: ID1/4/5/6 rotate opposite to the MuJoCo joints).
DEFAULT_SERVO_TO_SIM_SIGN = np.array(
    [-1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
    dtype=np.float64,
)
DEFAULT_SIM_HOME_RAD = np.zeros(ARM_DOF, dtype=np.float64)

ID7_CLOSED_DEG = 90.0
ID7_OPEN_DEG = 180.0

DEFAULT_SERVO_PORT = "COM6" if os.name == "nt" else "/dev/ttyUSB0"

# Named environment cameras rendered as small panels inside the MuJoCo viewer.
# Missing cameras are skipped silently.
EXTRA_CAMERA_NAMES = ("top", "cam_wrist", "angle")


# ---------------------------------------------------------------------------
# Servo SDK discovery (lazy: only imported when the master arm is used)
# ---------------------------------------------------------------------------

def _load_servo_sdk():
    """Locate and import STservo_sdk relative to this workspace."""
    relative_paths = (
        "rebot_scripts/STservo_sdk",
        "STservo_sdk",
        "Python/STservo_sdk",
    )
    roots = [hand_cv.SCRIPT_DIR, hand_cv.WORKSPACE_DIR, Path.cwd()]
    roots.extend(hand_cv.WORKSPACE_DIR.parents)

    seen = set()
    for root in roots:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        for rel in relative_paths:
            sdk_dir = root / rel
            if sdk_dir.is_dir():
                for add in (sdk_dir, sdk_dir.parent):
                    if str(add) not in sys.path:
                        sys.path.insert(0, str(add))
                import STservo_sdk
                return STservo_sdk

    raise ImportError(
        "STservo_sdk not found. Expected under rebot_scripts/STservo_sdk."
    )


# ---------------------------------------------------------------------------
# Servo math (ported from servo_arm_teleoperation_real2sim.py)
# ---------------------------------------------------------------------------

def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(float(min_value), min(float(value), float(max_value)))


def servo_pos_to_deg(pos: int | float) -> float:
    return float(pos) / SERVO_DIGITAL_RANGE * SERVO_ANGLE_RANGE


def deg_to_rad(deg: float) -> float:
    return float(deg) * np.pi / 180.0


def limit_servo_deg(servo_id: int, angle_deg: float) -> float:
    cfg = JOINT_LIMITS_DEG[servo_id]
    return clamp(
        float(angle_deg),
        cfg["min_deg"] + SAFETY_MARGIN_DEG,
        cfg["max_deg"] - SAFETY_MARGIN_DEG,
    )


def servo_deg_to_sim_rad(
    servo_id: int,
    angle_deg: float,
    arm_index: int,
    servo_to_sim_sign: np.ndarray,
    sim_home_rad: np.ndarray,
) -> float:
    cfg = JOINT_LIMITS_DEG[servo_id]
    home_deg = cfg["home_deg"]

    safe_deg = limit_servo_deg(servo_id, angle_deg)
    delta_deg = safe_deg - home_deg

    q_sim = sim_home_rad[arm_index] + servo_to_sim_sign[arm_index] * deg_to_rad(delta_deg)

    return float(q_sim)


def servo_deg_array_to_sim_rad(
    arm_deg_array: np.ndarray,
    servo_to_sim_sign: np.ndarray,
    sim_home_rad: np.ndarray,
) -> np.ndarray:
    q_sim = np.zeros(ARM_DOF, dtype=np.float64)

    for i, servo_id in enumerate(ARM_SERVO_IDS):
        q_sim[i] = servo_deg_to_sim_rad(
            servo_id=servo_id,
            angle_deg=float(arm_deg_array[i]),
            arm_index=i,
            servo_to_sim_sign=servo_to_sim_sign,
            sim_home_rad=sim_home_rad,
        )

    return q_sim


def id7_deg_to_norm(angle_deg: float, invert_id7: bool) -> float:
    """ID7 master angle -> normalized AeroHand opening (0 closed, 1 open).

    Follows the ID7 calibration of the real2sim reference: 90 deg -> 0.0
    (closed) and 180 deg -> 1.0 (open).
    """
    angle_deg = limit_servo_deg(ID7_SERVO_ID, angle_deg)

    denom = ID7_OPEN_DEG - ID7_CLOSED_DEG
    if abs(denom) < 1e-9:
        norm = 0.0
    else:
        norm = (angle_deg - ID7_CLOSED_DEG) / denom

    norm = clamp(norm, 0.0, 1.0)

    if invert_id7:
        norm = 1.0 - norm

    return float(norm)


def id7_norm_to_hand_ctrl(
    norm: float,
    open_ctrl: np.ndarray,
    closed_ctrl: np.ndarray,
    web_threshold: float,
) -> np.ndarray:
    """Staged AeroHand target from the ID7 opening amount.

    Opening (norm 0 -> 1): the fingers and the thumb flexion open first while
    the thumb web (right_thumb_A_cmc_abd) stays closed; the web then opens
    over the last segment.  Closing is the exact reverse: the web rotates
    first and the fingers close afterwards, so the web wraps around an object
    before the fingers curl in for a more stable grasp.

    web_threshold is the opening amount where the finger phase ends and the
    web phase begins (clamped to 0.05..0.95 so both phases always exist).
    """
    norm = clamp(float(norm), 0.0, 1.0)
    t = clamp(float(web_threshold), 0.05, 0.95)

    # Per-group closure (0 = open, 1 = closed).
    finger_closure = 1.0 - clamp(norm / t, 0.0, 1.0)
    web_closure = clamp((1.0 - norm) / (1.0 - t), 0.0, 1.0)

    closure = np.ones(7, dtype=np.float64)
    closure[:4] = finger_closure  # index/middle/ring/pinky tendons
    closure[4] = web_closure      # thumb abduction (the web)
    closure[5:] = finger_closure  # thumb flexion tendons
    return open_ctrl + closure * (closed_ctrl - open_ctrl)


def smooth_update(prev: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    alpha = clamp(float(alpha), 0.0, 1.0)
    return alpha * target + (1.0 - alpha) * prev


def _parse_vector(values, default: np.ndarray, name: str) -> np.ndarray:
    arr = default.astype(np.float64) if values is None else np.asarray(values, dtype=np.float64)

    if arr.shape != default.shape:
        raise ValueError(f"{name} 必须提供 {default.size} 个数，当前为 {arr.size} 个。")

    return arr


def read_servo_angle(sdk, scs, servo_id: int, last_angle: float) -> tuple[float, bool]:
    try:
        pos, speed, result, error = scs.ReadPosSpeed(servo_id)

        if result == sdk.COMM_SUCCESS:
            angle_deg = servo_pos_to_deg(pos)
            angle_deg = limit_servo_deg(servo_id, angle_deg)
            return float(angle_deg), True

        return float(last_angle), False

    except Exception:
        return float(last_angle), False


def release_servo_torque(sdk, scs, servo_ids: list[int]) -> None:
    print("\nReleasing master servo torque for drag teleoperation...")

    for servo_id in servo_ids:
        try:
            result, error = scs.write1ByteTxRx(
                servo_id,
                sdk.STS_TORQUE_ENABLE,
                0,
            )

            if result == sdk.COMM_SUCCESS:
                print(f"  ID={servo_id} torque released")
            else:
                print(f"  ID={servo_id} torque release failed: {scs.getTxRxResult(result)}")

        except Exception as e:
            print(f"  ID={servo_id} torque release error: {e}")

        time.sleep(0.02)


# ---------------------------------------------------------------------------
# MuJoCo helpers
# ---------------------------------------------------------------------------

def find_arm_actuators(model, joint_names: tuple[str, ...]) -> np.ndarray:
    """Locate the position actuators transmitted by the six arm joints.

    Tendon actuators also store an id in actuator_trnid[:, 0], so filter by
    transmission type to avoid mistaking an AeroHand tendon whose numeric id
    equals an arm joint id for an arm actuator.
    """
    joint_ids = np.asarray(
        [
            hand_cv.mujoco.mj_name2id(
                model, hand_cv.mujoco.mjtObj.mjOBJ_JOINT, name
            )
            for name in joint_names
        ],
        dtype=np.int32,
    )
    if np.any(joint_ids < 0):
        raise RuntimeError("The model must contain joint1 through joint6")

    actuator_ids = []
    for joint_id, name in zip(joint_ids, joint_names):
        matches = np.flatnonzero(
            (model.actuator_trntype == hand_cv.mujoco.mjtTrn.mjTRN_JOINT)
            & (model.actuator_trnid[:, 0] == joint_id)
        )
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one actuator transmitted by {name}, found {len(matches)}"
            )
        actuator_ids.append(int(matches[0]))
    return np.asarray(actuator_ids, dtype=np.int32)


def set_arm_ctrl(model, data, actuator_ids: np.ndarray, q_sim: np.ndarray) -> None:
    """Write the six arm position-actuator targets, clamped to ctrlrange."""
    q_sim = np.asarray(q_sim, dtype=np.float64)[:ARM_DOF]

    for index, act_id in enumerate(actuator_ids):
        cmd = float(q_sim[index])

        if model.actuator_ctrllimited[act_id]:
            ctrl_min = float(model.actuator_ctrlrange[act_id, 0])
            ctrl_max = float(model.actuator_ctrlrange[act_id, 1])
            cmd = clamp(cmd, ctrl_min, ctrl_max)

        data.ctrl[act_id] = cmd


def apply_arm_gravcomp(model, joint_names: tuple[str, ...], gravcomp: float) -> None:
    """Compensate the complete payload below joint1, including the hand.

    This is MuJoCo's built-in body gravity compensation and avoids the large
    steady-state sag of the position actuators under the AeroHand payload.
    """
    joint_ids = np.asarray(
        [
            hand_cv.mujoco.mj_name2id(
                model, hand_cv.mujoco.mjtObj.mjOBJ_JOINT, name
            )
            for name in joint_names
        ],
        dtype=np.int32,
    )
    if np.any(joint_ids < 0):
        return

    arm_root_body = int(model.jnt_bodyid[joint_ids[0]])

    for body_id in range(1, model.nbody):
        ancestor = body_id
        while ancestor > 0 and ancestor != arm_root_body:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor == arm_root_body:
            model.body_gravcomp[body_id] = float(np.clip(gravcomp, 0.0, 1.0))


# ---------------------------------------------------------------------------
# GLFW MuJoCo viewer (no keyboard arm control, mouse orbit/pan/zoom only)
# ---------------------------------------------------------------------------

class ArmControlViewer:
    """Minimal GLFW MuJoCo viewer with no built-in keyboard shortcuts.

    This follows the viewer design used by lerobot-mujoco-rebot: GLFW events
    are registered directly, so only ESC is consumed by this window and no
    MuJoCo shortcut can interfere.  Mouse orbit/pan/zoom remains available.
    """

    def __init__(self, model, data, show_extra_cams: bool = True):
        self.model = model
        self.data = data
        self.window = None
        self.left_pressed = False
        self.right_pressed = False
        self.last_x = 0.0
        self.last_y = 0.0

        # Environment cameras rendered as stacked panels on the right edge.
        self.extra_cams = []
        if show_extra_cams:
            for name in EXTRA_CAMERA_NAMES:
                cam_id = hand_cv.mujoco.mj_name2id(
                    model, hand_cv.mujoco.mjtObj.mjOBJ_CAMERA, name
                )
                if cam_id >= 0:
                    self.extra_cams.append((name, int(cam_id)))

        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")
        # GLFW window hints persist across create_window calls. An offscreen
        # mujoco.Renderer (--save-videos) may have set VISIBLE=0 beforehand,
        # which would otherwise make this window invisible.
        glfw.default_window_hints()
        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor) if monitor is not None else None
        width = min(1400, mode.size.width) if mode is not None else 1280
        height = min(900, mode.size.height) if mode is not None else 800
        self.window = glfw.create_window(
            width, height, "reBot Servo Teleop + AeroHand CV", None, None
        )
        if self.window is None:
            glfw.terminate()
            raise RuntimeError("Could not create the MuJoCo GLFW window")
        # Center the window and request focus so it is not left hidden behind
        # a maximized terminal window.
        pos_x = max(0, (mode.size.width - width) // 2) if mode is not None else 60
        pos_y = max(0, (mode.size.height - height) // 2) if mode is not None else 60
        glfw.set_window_pos(self.window, pos_x, pos_y)
        glfw.focus_window(self.window)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.camera = hand_cv.mujoco.MjvCamera()
        self.option = hand_cv.mujoco.MjvOption()
        self.scene = hand_cv.mujoco.MjvScene(model, maxgeom=10000)
        self.context = hand_cv.mujoco.MjrContext(
            model, hand_cv.mujoco.mjtFontScale.mjFONTSCALE_150
        )
        self.camera.type = hand_cv.mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.azimuth = 135.0
        self.camera.elevation = -25.0
        self.camera.distance = 1.25
        self.camera.lookat[:] = np.asarray([0.0, 0.0, 0.28])

        self.fixed_camera = hand_cv.mujoco.MjvCamera()
        self.fixed_camera.type = hand_cv.mujoco.mjtCamera.mjCAMERA_FIXED

        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_scroll_callback(self.window, self._on_scroll)

    def _on_key(self, _window, key, _scancode, action, _mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(self.window, True)

    def _on_mouse_button(self, window, _button, _action, _mods):
        self.left_pressed = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self.right_pressed = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        self.last_x, self.last_y = glfw.get_cursor_pos(window)

    def _on_cursor(self, window, xpos, ypos):
        if not (self.left_pressed or self.right_pressed):
            return
        width, height = glfw.get_window_size(window)
        dx = (xpos - self.last_x) / max(1, height)
        dy = (ypos - self.last_y) / max(1, height)
        self.last_x, self.last_y = xpos, ypos
        shift = any(
            glfw.get_key(window, key) == glfw.PRESS
            for key in (glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT)
        )
        if self.right_pressed:
            action = hand_cv.mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else hand_cv.mujoco.mjtMouse.mjMOUSE_MOVE_V
        else:
            action = hand_cv.mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else hand_cv.mujoco.mjtMouse.mjMOUSE_ROTATE_V
        hand_cv.mujoco.mjv_moveCamera(
            # self.model, action, dx, dy, self.scene, self.camera  # 旧版 MuJoCo API mujoco 3.3.0
            self.model, action, dx, dy, self.camera  # 新版 MuJoCo API mujoco 3.11.0
        )

    def _on_scroll(self, _window, _xoffset, yoffset):
        hand_cv.mujoco.mjv_moveCamera(
            self.model,
            hand_cv.mujoco.mjtMouse.mjMOUSE_ZOOM,
            0.0,
            -0.05 * yoffset,
            # self.scene,
            self.camera,
        )

    def is_running(self) -> bool:
        return self.window is not None and not glfw.window_should_close(self.window)

    def sync(self) -> None:
        glfw.make_context_current(self.window)
        width, height = glfw.get_framebuffer_size(self.window)
        viewport = hand_cv.mujoco.MjrRect(0, 0, width, height)
        hand_cv.mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.option,
            None,
            self.camera,
            hand_cv.mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )
        hand_cv.mujoco.mjr_render(viewport, self.scene, self.context)

        # Stacked environment camera panels on the right edge.  The scene is
        # re-updated with each fixed camera; the main camera re-updates it at
        # the top of the next sync call.
        if self.extra_cams:
            margin = max(8, int(0.012 * height))
            panel_w = int(0.26 * width)
            panel_h = int(0.24 * height)
            x0 = width - margin - panel_w
            y_bottom = margin

            for name, cam_id in self.extra_cams:
                panel = hand_cv.mujoco.MjrRect(x0, y_bottom, panel_w, panel_h)
                self.fixed_camera.fixedcamid = cam_id
                hand_cv.mujoco.mjv_updateScene(
                    self.model,
                    self.data,
                    self.option,
                    None,
                    self.fixed_camera,
                    hand_cv.mujoco.mjtCatBit.mjCAT_ALL,
                    self.scene,
                )
                hand_cv.mujoco.mjr_render(panel, self.scene, self.context)

                # Camera name drawn directly on the image with a black
                # outline, so it stays readable on bright scenes without any
                # dark band behind it.  mjr_text coordinates are fractions of
                # the most recent viewport (the panel), calibrated against
                # mjr_readPixels:
                #   glyph left   = left + 2 + x_frac * width
                #   glyph bottom = bottom + 7 + y_frac * height (FONTSCALE_100)
                text_x = 6.0 / panel.width
                text_y = (panel.height - 29.0) / panel.height
                dx = 2.0 / panel.width
                dy = 2.0 / panel.height
                for ox, oy in (
                    (-dx, 0.0), (dx, 0.0), (0.0, -dy), (0.0, dy),
                    (-dx, -dy), (dx, -dy), (-dx, dy), (dx, dy),
                ):
                    hand_cv.mujoco.mjr_text(
                        hand_cv.mujoco.mjtFontScale.mjFONTSCALE_100,
                        name,
                        self.context,
                        text_x + ox,
                        text_y + oy,
                        0.0, 0.0, 0.0,
                    )
                hand_cv.mujoco.mjr_text(
                    hand_cv.mujoco.mjtFontScale.mjFONTSCALE_100,
                    name,
                    self.context,
                    text_x,
                    text_y,
                    1.0, 1.0, 1.0,
                )

                # 2 px border strips (bottom/left/right only, so the panel
                # top stays clean).
                hand_cv.mujoco.mjr_rectangle(
                    hand_cv.mujoco.MjrRect(x0, y_bottom, panel_w, 2), 0.0, 0.0, 0.0, 1.0
                )
                hand_cv.mujoco.mjr_rectangle(
                    hand_cv.mujoco.MjrRect(x0, y_bottom, 2, panel_h), 0.0, 0.0, 0.0, 1.0
                )
                hand_cv.mujoco.mjr_rectangle(
                    hand_cv.mujoco.MjrRect(x0 + panel_w - 2, y_bottom, 2, panel_h), 0.0, 0.0, 0.0, 1.0
                )

                y_bottom += panel_h + margin

        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self) -> None:
        if self.window is not None:
            glfw.make_context_current(self.window)
            self.context.free()
            glfw.destroy_window(self.window)
            self.window = None
            glfw.terminate()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()


# ---------------------------------------------------------------------------
# ST/SMS_STS master arm serial link and reader thread
# ---------------------------------------------------------------------------

class ServoMasterLink:
    """ST/SMS_STS master arm serial link.

    ID1..ID6 angles are converted to MuJoCo arm targets (q_sim) inside the
    reader thread, so the viewer loop only copies the latest snapshot.  ID7
    is read as the AeroHand open/close master while CV hand control is off.
    """

    def __init__(
        self,
        *,
        port: str,
        baudrate: int,
        read_rate: float,
        servo_to_sim_sign: np.ndarray,
        sim_home_rad: np.ndarray,
        enable_id7: bool,
        invert_id7: bool,
        keep_torque: bool,
    ):
        self.port = port
        self.baudrate = int(baudrate)
        self.read_rate = max(float(read_rate), 1e-6)
        self.servo_to_sim_sign = np.asarray(servo_to_sim_sign, dtype=np.float64)
        self.sim_home_rad = np.asarray(sim_home_rad, dtype=np.float64)
        self.enable_id7 = bool(enable_id7)
        self.invert_id7 = bool(invert_id7)
        self.keep_torque = bool(keep_torque)
        self.expected_success = ARM_DOF + (1 if self.enable_id7 else 0)

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.sdk = None
        self.port_handler = None
        self.scs = None
        self.sync_read = None

        home_arm_deg = np.array(
            [JOINT_LIMITS_DEG[servo_id]["home_deg"] for servo_id in ARM_SERVO_IDS],
            dtype=np.float64,
        )
        self._arm_deg = home_arm_deg.copy()
        self._target_q_sim = servo_deg_array_to_sim_rad(
            home_arm_deg,
            servo_to_sim_sign=self.servo_to_sim_sign,
            sim_home_rad=self.sim_home_rad,
        )
        self._id7_deg = float(JOINT_LIMITS_DEG[ID7_SERVO_ID]["home_deg"])
        self._id7_norm = 0.0
        self._id7_fresh = False
        self._success_count = 0
        self._failed_ids: list[int] = []
        self._timestamp = time.perf_counter()
        self._read_frame = 0

    def open(self) -> None:
        try:
            sdk = _load_servo_sdk()
        except ImportError as exc:
            raise RuntimeError(f"STservo_sdk not found: {exc}") from exc

        self.sdk = sdk
        self.port_handler = sdk.PortHandler(self.port)

        if not self.port_handler.openPort():
            raise RuntimeError(f"Could not open the servo master port: {self.port}")

        if not self.port_handler.setBaudRate(self.baudrate):
            self.port_handler.closePort()
            raise RuntimeError(f"Could not set the servo baudrate: {self.baudrate}")

        self.scs = sdk.sts(self.port_handler)

        print(f"Servo master port opened: {self.port} @ {self.baudrate}")
        print("Waiting for the ESP32 passthrough link...")
        time.sleep(2.5)

        try:
            self.port_handler.ser.reset_input_buffer()
            self.port_handler.ser.reset_output_buffer()
        except Exception as exc:
            print(f"Serial buffer reset failed (ignorable): {exc}")

        # Batch sync read: one broadcast transaction for all master servos
        # instead of one round trip each (much lower read latency).  The
        # worker falls back to per-servo reads for any servo that is missing
        # from a batch response.
        try:
            sync_read = sdk.GroupSyncRead(
                self.port_handler, sdk.STS_PRESENT_POSITION_L, 2
            )
            read_ids = list(ARM_SERVO_IDS)
            if self.enable_id7:
                read_ids.append(ID7_SERVO_ID)
            for servo_id in read_ids:
                sync_read.addParam(servo_id)
            self.sync_read = sync_read
            print("Batch sync read enabled for all master servos")
        except Exception as exc:
            print(f"Batch sync read unavailable ({exc}); using per-servo reads")

        if not self.keep_torque:
            release_ids = list(ARM_SERVO_IDS)
            if self.enable_id7:
                release_ids.append(ID7_SERVO_ID)
            release_servo_torque(self.sdk, self.scs, release_ids)
        else:
            print("--keep-servo-torque: master torque left enabled")

    def start(self) -> None:
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def wait_ready(self, timeout: float = 5.0) -> bool:
        t0 = time.perf_counter()

        while time.perf_counter() - t0 < timeout:
            with self.lock:
                read_frame = self._read_frame
                success_count = self._success_count

            if read_frame > 0 and success_count >= self.expected_success:
                return True

            time.sleep(0.02)

        return False

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "arm_deg": self._arm_deg.copy(),
                "target_q_sim": self._target_q_sim.copy(),
                "id7_deg": float(self._id7_deg),
                "id7_norm": float(self._id7_norm),
                "id7_fresh": bool(self._id7_fresh),
                "success_count": int(self._success_count),
                "failed_ids": list(self._failed_ids),
                "timestamp": float(self._timestamp),
                "read_frame": int(self._read_frame),
            }

    def close(self) -> None:
        self.stop_event.set()

        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None

        if self.port_handler is not None:
            try:
                self.port_handler.closePort()
            except Exception:
                pass
            self.port_handler = None

    def _worker(self) -> None:
        read_period = 1.0 / self.read_rate

        last_arm_deg = np.array(
            [JOINT_LIMITS_DEG[servo_id]["home_deg"] for servo_id in ARM_SERVO_IDS],
            dtype=np.float64,
        )
        last_id7_deg = JOINT_LIMITS_DEG[ID7_SERVO_ID]["home_deg"]

        print(f"Servo master reader thread started, target rate {self.read_rate:.1f} Hz")

        while not self.stop_event.is_set():
            loop_start = time.perf_counter()

            arm_deg = last_arm_deg.copy()
            success_count = 0
            failed_ids: list[int] = []

            # One broadcast read for all servos, then a per-servo fallback
            # read for any ID missing from the batch response.
            sync_ok = False
            if self.sync_read is not None:
                try:
                    sync_ok = self.sync_read.txRxPacket() == self.sdk.COMM_SUCCESS
                except Exception:
                    sync_ok = False

            for i, servo_id in enumerate(ARM_SERVO_IDS):
                ok = False

                if sync_ok:
                    try:
                        avail, _err = self.sync_read.isAvailable(
                            servo_id, self.sdk.STS_PRESENT_POSITION_L, 2
                        )
                        if avail:
                            pos = self.sync_read.getData(
                                servo_id, self.sdk.STS_PRESENT_POSITION_L, 2
                            )
                            arm_deg[i] = limit_servo_deg(servo_id, servo_pos_to_deg(pos))
                            ok = True
                    except Exception:
                        ok = False

                if not ok:
                    arm_deg[i], ok = read_servo_angle(
                        self.sdk, self.scs, servo_id, arm_deg[i]
                    )

                if ok:
                    success_count += 1
                else:
                    failed_ids.append(servo_id)

            last_arm_deg = arm_deg.copy()

            target_q_sim = servo_deg_array_to_sim_rad(
                arm_deg,
                servo_to_sim_sign=self.servo_to_sim_sign,
                sim_home_rad=self.sim_home_rad,
            )

            id7_deg = float(last_id7_deg)
            id7_norm = 0.0
            id7_fresh = False

            if self.enable_id7:
                ok = False

                if sync_ok:
                    try:
                        avail, _err = self.sync_read.isAvailable(
                            ID7_SERVO_ID, self.sdk.STS_PRESENT_POSITION_L, 2
                        )
                        if avail:
                            pos = self.sync_read.getData(
                                ID7_SERVO_ID, self.sdk.STS_PRESENT_POSITION_L, 2
                            )
                            id7_deg = limit_servo_deg(ID7_SERVO_ID, servo_pos_to_deg(pos))
                            ok = True
                    except Exception:
                        ok = False

                if not ok:
                    id7_deg, ok = read_servo_angle(
                        self.sdk, self.scs, ID7_SERVO_ID, last_id7_deg
                    )

                id7_fresh = ok

                if ok:
                    success_count += 1
                else:
                    failed_ids.append(ID7_SERVO_ID)

                last_id7_deg = float(id7_deg)
                id7_norm = id7_deg_to_norm(id7_deg, invert_id7=self.invert_id7)

            with self.lock:
                self._arm_deg = arm_deg.copy()
                self._target_q_sim = target_q_sim.copy()
                self._id7_deg = float(id7_deg)
                self._id7_norm = float(id7_norm)
                self._id7_fresh = bool(id7_fresh)
                self._success_count = int(success_count)
                self._failed_ids = list(failed_ids)
                self._timestamp = time.perf_counter()
                self._read_frame += 1

            elapsed = time.perf_counter() - loop_start

            if elapsed < read_period:
                time.sleep(read_period - elapsed)


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_TASK_MODEL)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--process-width", type=int, default=384)
    parser.add_argument("--vision-hz", type=float, default=20.0)
    parser.add_argument("--model-complexity", type=int, choices=(0, 1), default=0)
    parser.add_argument("--track-hand", choices=("right", "left", "any"), default="right")
    parser.add_argument("--landmark-alpha", type=float, default=0.70)
    parser.add_argument("--command-alpha", type=float, default=0.25)
    parser.add_argument("--open-deadband", type=float, default=0.10)
    parser.add_argument("--response-gamma", type=float, default=1.35)
    parser.add_argument("--lost-hand-open-delay", type=float, default=0.35)
    parser.add_argument("--realtime-factor", type=float, default=1.0)
    parser.add_argument("--gl-mode", choices=("software", "hardware"), default=hand_cv.EARLY_GL_MODE)
    parser.add_argument("--start-enabled", action="store_true")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Disable CV hand control and the camera window; ID7 drives the hand.",
    )
    parser.add_argument(
        "--extra-cameras",
        action="store_true",
        help="Show the top/cam_wrist/angle camera panels inside the MuJoCo viewer.",
    )

    # Servo master arm (ID1..ID6 -> MuJoCo data.ctrl)
    parser.add_argument("--port", type=str, default=DEFAULT_SERVO_PORT)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--read-rate", type=float, default=100.0)
    parser.add_argument("--keep-servo-torque", action="store_true")
    parser.add_argument("--servo-to-sim-signs", type=float, nargs=6, default=None)
    parser.add_argument("--sim-home", type=float, nargs=6, default=None)
    parser.add_argument("--alpha-master", type=float, default=0.85)
    parser.add_argument("--max-servo-age", type=float, default=0.5)
    parser.add_argument(
        "--no-servo",
        action="store_true",
        help="Run without the servo master: the sim arm holds its initial pose.",
    )
    parser.add_argument("--arm-gravcomp", type=float, default=1.0)
    parser.add_argument(
        "--arm-kp-gain",
        type=float,
        default=1.0,
        help="Scale factor (0.1..5.0) for the arm position-actuator kp; "
        ">1 makes the sim arm track the master more crisply.",
    )

    # ID7 servo -> AeroHand open/close (used while CV hand control is off)
    parser.add_argument("--no-id7", action="store_true")
    parser.add_argument("--invert-id7", action="store_true")
    parser.add_argument(
        "--id7-web-threshold",
        type=float,
        default=0.5,
        help=(
            "ID7 opening amount where the finger phase ends and the thumb-web "
            "phase begins (clamped to 0.05..0.95)."
        ),
    )
    parser.add_argument(
        "--id7-reengage-deg",
        type=float,
        default=3.0,
        help="Move ID7 by this many degrees to retake the hand from the O/C manual latch.",
    )
    parser.add_argument("--print-every", type=int, default=25)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    model_path = hand_cv.resolve_model_path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"MuJoCo model not found: {model_path}")

    model = hand_cv.mujoco.MjModel.from_xml_path(str(model_path))
    data = hand_cv.mujoco.MjData(model)
    hand_mapper = hand_cv.SimHandMapper(model)
    arm_actuator_ids = find_arm_actuators(model, ARM_JOINT_NAMES)
    apply_arm_gravcomp(model, ARM_JOINT_NAMES, args.arm_gravcomp)

    # Optional crispness knob for the arm position actuators: higher kp
    # reduces the sim arm's tracking lag and steady-state sag.
    if args.arm_kp_gain != 1.0:
        kp_gain = float(np.clip(args.arm_kp_gain, 0.1, 5.0))
        for act_id in arm_actuator_ids:
            model.actuator_gainprm[act_id, 0] *= kp_gain
        print(f"[Arm] position-actuator kp scaled by {kp_gain:.2f}")

    servo_to_sim_sign = _parse_vector(
        args.servo_to_sim_signs,
        DEFAULT_SERVO_TO_SIM_SIGN,
        "--servo-to-sim-signs",
    )
    sim_home_rad = _parse_vector(
        args.sim_home,
        DEFAULT_SIM_HOME_RAD,
        "--sim-home",
    )
    web_threshold = float(np.clip(args.id7_web_threshold, 0.05, 0.95))

    # Start from the master home pose: every servo at home_deg maps to
    # sim_home_rad, which is also the model's default arm qpos.
    set_arm_ctrl(model, data, arm_actuator_ids, sim_home_rad)
    hand_target = hand_mapper.open_ctrl
    hand_filtered = hand_target.copy()
    hand_mapper.write(data, hand_target)
    hand_cv.mujoco.mj_forward(model, data)

    grabber = None
    if not args.no_cv:
        grabber = hand_cv.LatestFrameGrabber(
            args.camera, args.width, args.height, args.camera_fps
        )
        grabber.start()

    # ---------- servo master setup ----------
    servo = None
    id7_active = False

    if not args.no_servo:
        print("[Arm] servo -> sim mapping:")
        for i, servo_id in enumerate(ARM_SERVO_IDS):
            cfg = JOINT_LIMITS_DEG[servo_id]
            print(
                f"  ID{servo_id}: [{cfg['min_deg']:.0f}, {cfg['max_deg']:.0f}] deg, "
                f"home={cfg['home_deg']:.0f} deg -> joint{i + 1}, "
                f"sign={servo_to_sim_sign[i]:+.0f}, sim_home={sim_home_rad[i]:+.3f} rad"
            )
        if not args.no_id7:
            print(
                f"  ID7: {ID7_CLOSED_DEG:.0f} deg closed -> {ID7_OPEN_DEG:.0f} deg open, "
                f"invert={args.invert_id7}, web_threshold={web_threshold:.2f}"
            )

        servo = ServoMasterLink(
            port=args.port,
            baudrate=args.baudrate,
            read_rate=args.read_rate,
            servo_to_sim_sign=servo_to_sim_sign,
            sim_home_rad=sim_home_rad,
            enable_id7=not args.no_id7,
            invert_id7=args.invert_id7,
            keep_torque=args.keep_servo_torque,
        )

        try:
            servo.open()
            servo.start()

            if not servo.wait_ready(timeout=5.0):
                raise RuntimeError("timed out waiting for the servo master reads")

            first = servo.snapshot()
            print(f"[Servo] master ready: arm_deg = {np.round(first['arm_deg'], 1).tolist()}")
            print(f"[Servo] master ready: q_sim   = {np.round(first['target_q_sim'], 3).tolist()}")
        except Exception as exc:
            print(f"[Servo] initialization failed: {exc}")
            servo.close()
            if grabber is not None:
                grabber.close()
            return 1

        id7_active = not args.no_id7

    vision = None
    cv_enabled = bool(args.start_enabled) and not args.no_cv
    show_details = True
    manual_target = None
    id7_manual_ref = None
    last_sequence = -1
    last_frame = None
    last_frame_time = 0.0
    last_label = "NONE"
    last_ratios = np.zeros(16, dtype=np.float64)
    last_detection_time = time.perf_counter()
    frame_times: list[float] = []
    command_alpha = float(np.clip(args.command_alpha, 0.01, 1.0))
    alpha_master = float(np.clip(args.alpha_master, 0.01, 1.0))
    reengage_deg = max(0.0, float(args.id7_reengage_deg))
    sim_period = float(model.opt.timestep) / max(0.05, args.realtime_factor)
    next_step = time.perf_counter()

    filtered_q_sim = sim_home_rad.copy()
    servo_primed = False
    servo_stale = False
    last_read_frame = -1
    missing_count = 0
    frame = 0
    last_print_time = 0.0

    print(f"[Model] {model_path}")
    print(f"[Model] nq={model.nq} nv={model.nv} nu={model.nu}")
    if grabber is not None:
        print(f"[Camera] {grabber.negotiated()}")
    else:
        print("[Camera] disabled (--no-cv)")
    print("[Arm] servo master ID1..ID6 -> MuJoCo data.ctrl (real2sim)")
    if args.no_cv:
        print("[Hand] CV disabled (--no-cv): ID7 drives the hand; ESC in the viewer quits")
    else:
        print("[Hand keys/camera] SPACE CV/ID7 | O open | C close | V details | Q/ESC quit")

    try:
        with ArmControlViewer(
            model, data, show_extra_cams=args.extra_cameras
        ) as viewer:
            if args.extra_cameras:
                extra_names = [name for name, _ in viewer.extra_cams]
                if extra_names:
                    print(f"[Viewer] extra camera panels: {', '.join(extra_names)}")
                else:
                    print("[Viewer] no extra cameras found in the model")
            if not args.no_cv:
                vision = hand_cv.VisionProcessor(grabber, args)
                vision.start()
            while viewer.is_running():
                if grabber is not None and grabber.error is not None:
                    raise RuntimeError(f"Camera thread failed: {grabber.error}")
                if vision is not None and vision.error is not None:
                    raise RuntimeError(f"Vision thread failed: {vision.error}")

                # ---- servo master -> sim arm ----
                snap = None
                servo_age = 0.0
                id7_deg = None
                id7_norm = 0.0
                id7_fresh = False
                arm_status = "OFF"

                if servo is not None:
                    snap = servo.snapshot()
                    servo_age = time.perf_counter() - snap["timestamp"]
                    arm_status = "SERVO" if servo_age <= args.max_servo_age else "STALE"

                    if snap["read_frame"] != last_read_frame:
                        last_read_frame = snap["read_frame"]

                        if servo_age > args.max_servo_age:
                            if not servo_stale:
                                print(
                                    f"[Arm] servo master data stale "
                                    f"({servo_age * 1000:.0f} ms): arm target frozen"
                                )
                                servo_stale = True
                        else:
                            servo_stale = False

                            if not servo_primed:
                                # Snap to the master pose on the first good
                                # frame instead of slowly easing into it.
                                filtered_q_sim = snap["target_q_sim"].copy()
                                servo_primed = True
                            else:
                                filtered_q_sim = smooth_update(
                                    filtered_q_sim,
                                    snap["target_q_sim"],
                                    alpha_master,
                                )

                    if snap["success_count"] < servo.expected_success:
                        missing_count += 1

                    if id7_active:
                        id7_deg = float(snap["id7_deg"])
                        id7_norm = float(snap["id7_norm"])
                        id7_fresh = bool(snap["id7_fresh"]) and servo_age <= args.max_servo_age

                # ---- CV hand tracking ----
                new_frame = False
                if vision is not None:
                    packet = vision.latest(last_sequence)
                    if packet is not None:
                        last_frame, capture_time, last_sequence, label, ratios, detected = packet
                        last_frame_time = capture_time
                        last_label = label
                        new_frame = True
                        if detected:
                            last_ratios = ratios
                            last_detection_time = time.perf_counter()
                            if cv_enabled:
                                hand_target = hand_mapper.ratios_to_ctrl(ratios)
                        frame_times.append(time.perf_counter())
                        frame_times = frame_times[-30:]

                    if (
                        cv_enabled
                        and time.perf_counter() - last_detection_time
                        >= max(0.0, args.lost_hand_open_delay)
                    ):
                        hand_target = hand_mapper.open_ctrl
                        last_ratios.fill(0.0)

                # ---- hand command source while CV is off ----
                hand_mode = "CV" if cv_enabled else "HOLD"
                if not cv_enabled:
                    if manual_target is not None:
                        hand_mode = "MAN"
                        hand_target = manual_target
                        if (
                            reengage_deg > 0.0
                            and id7_fresh
                            and id7_manual_ref is not None
                            and abs(float(id7_deg) - id7_manual_ref) >= reengage_deg
                        ):
                            manual_target = None
                            id7_manual_ref = None
                            print("[Hand] ID7 servo re-engaged")
                    elif id7_active and id7_fresh:
                        hand_mode = "ID7"
                        hand_target = id7_norm_to_hand_ctrl(
                            id7_norm,
                            hand_mapper.open_ctrl,
                            hand_mapper.closed_ctrl,
                            web_threshold,
                        )

                # ---- fixed-step simulation ----
                now = time.perf_counter()
                steps = 0
                while now >= next_step and steps < 5:
                    set_arm_ctrl(model, data, arm_actuator_ids, filtered_q_sim)
                    hand_filtered += command_alpha * (hand_target - hand_filtered)
                    hand_mapper.write(data, hand_filtered)
                    hand_cv.mujoco.mj_step(model, data)
                    next_step += sim_period
                    steps += 1
                if now - next_step > 0.05:
                    next_step = now
                viewer.sync()

                # ---- camera window ----
                if vision is not None and last_frame is not None and new_frame:
                    display = last_frame.copy()
                    fps = 0.0
                    if len(frame_times) >= 2:
                        fps = (len(frame_times) - 1) / max(
                            frame_times[-1] - frame_times[0], 1e-6
                        )
                    age_ms = (time.perf_counter() - last_frame_time) * 1000.0
                    status = "ON" if cv_enabled else "OFF"
                    cv2.putText(
                        display,
                        f"HAND {hand_mode} CV {status} | {last_label} | "
                        f"{fps:.1f} FPS | {age_ms:.0f} ms | ARM {arm_status}",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        (40, 230, 60) if cv_enabled else (80, 190, 255),
                        2, cv2.LINE_AA,
                    )
                    if show_details:
                        curls = [
                            last_ratios[0],
                            np.mean(last_ratios[4:7]),
                            np.mean(last_ratios[7:10]),
                            np.mean(last_ratios[10:13]),
                            np.mean(last_ratios[13:16]),
                        ]
                        cv2.putText(
                            display,
                            "curl T/I/M/R/P " + " / ".join(f"{v:.2f}" for v in curls),
                            (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                            (240, 240, 240), 1, cv2.LINE_AA,
                        )
                    cv2.imshow("reBot Servo Teleop + AeroHand CV", display)

                if vision is not None:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q"), ord("Q")):
                        break
                    if key == ord(" "):
                        cv_enabled = not cv_enabled
                        manual_target = None
                        id7_manual_ref = None
                        print(f"[Hand CV] {'ON' if cv_enabled else 'OFF'}")
                    elif key in (ord("o"), ord("O")):
                        cv_enabled = False
                        manual_target = hand_mapper.open_ctrl
                        id7_manual_ref = float(id7_deg) if id7_fresh else None
                        print("[Hand] OPEN (manual)")
                    elif key in (ord("c"), ord("C")):
                        cv_enabled = False
                        manual_target = hand_mapper.closed_ctrl
                        id7_manual_ref = float(id7_deg) if id7_fresh else None
                        print("[Hand] CLOSED (manual)")
                    elif key in (ord("v"), ord("V")):
                        show_details = not show_details

                # ---- status print ----
                frame += 1
                if args.print_every > 0 and frame % args.print_every == 0:
                    t_print = time.perf_counter()
                    if t_print - last_print_time >= 0.2:
                        if snap is not None:
                            deg_str = " ".join(f"{v:6.1f}" for v in snap["arm_deg"])
                            qsim_str = " ".join(f"{v:+6.2f}" for v in filtered_q_sim)
                            extra = ""
                            if id7_active:
                                extra = (
                                    f" | ID7={id7_deg:6.1f}deg norm={id7_norm:.2f}"
                                )
                            print(
                                f"[{frame:06d}] servo_deg=[{deg_str}] | "
                                f"q_sim_ctrl=[{qsim_str}] | "
                                f"age={servo_age * 1000:.1f}ms | miss={missing_count} | "
                                f"failed={snap['failed_ids']} | hand={hand_mode}{extra}"
                            )
                        else:
                            print(f"[{frame:06d}] servo=OFF | hand={hand_mode}")
                        last_print_time = t_print

                time.sleep(0.001)
    finally:
        if vision is not None:
            vision.close()
        if grabber is not None:
            grabber.close()
        if servo is not None:
            servo.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# 运行示例：
# python3 rebotarm_aerohand_right_teleoperation_real2sim.py --port /dev/ttyUSB0 --baudrate 115200
# 不带舵机主手、只跑 CV 手部控制（机械臂保持初始姿态）：
# python3 rebotarm_aerohand_right_teleoperation_real2sim.py --no-servo
# 不用 CV 摄像头、只用 ID7 控制灵巧手开合：
# python3 rebotarm_aerohand_right_teleoperation_real2sim.py --port /dev/ttyUSB0 --no-cv
# 额外增加相机画面
# python3 rebotarm_aerohand_right_teleoperation_real2sim.py --port /dev/ttyUSB0 --no-cv --extra-cameras
