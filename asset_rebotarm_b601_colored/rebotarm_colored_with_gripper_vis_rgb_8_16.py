#!/usr/bin/env python3
import argparse
import logging
import multiprocessing as mp
import os
import queue
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import glfw
import mujoco  # pip install mujoco==3.6.0
import mujoco.viewer
import numpy as np

# ============================================================
# 0. 动态环境变量注入 & SDK 路径修复
# ============================================================
current_dir = Path(__file__).resolve().parent
python_root = current_dir.parent  # 仓库根目录
sdk_path = python_root / "rebot_scripts" / "STservo_sdk"
sdk_parent = sdk_path.parent

print(f"[路径检查] 当前脚本目录: {current_dir}")
print(f"[路径检查] Python根目录: {python_root}")
print(f"[路径检查] SDK目录: {sdk_path}")

if not sdk_path.exists():
    print(f"❌ 未找到 STservo_sdk 目录: {sdk_path}")
    sys.exit(1)

# 导入 STservo_sdk 包需要把包的父目录 rebot_scripts 加入 sys.path；
# 同时保留 SDK 目录，以兼容 SDK 内部的非相对导入。
for p in [str(sdk_parent), str(sdk_path), str(python_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from STservo_sdk import *

logging.getLogger().setLevel(logging.ERROR)

# ============================================================
# 1. 硬件 & 映射配置 (Real2Sim 舵机参数)
# ============================================================
ARM_SERVO_IDS = [1, 2, 3, 4, 5, 6]
GRIPPER_SERVO_ID = 7
ARM_DOF = len(ARM_SERVO_IDS)

SERVO_DIGITAL_RANGE = 4095.0
SERVO_ANGLE_RANGE = 360.0

JOINT_LIMITS_DEG = {
    1: {"min_deg": 50.0, "max_deg": 300.0, "home_deg": 180.0},
    2: {"min_deg": 10.0, "max_deg": 180.0, "home_deg": 180.0},
    3: {"min_deg": 22.0, "max_deg": 180.0, "home_deg": 180.0},
    4: {"min_deg": 100.0, "max_deg": 270.0, "home_deg": 180.0},
    5: {"min_deg": 90.0, "max_deg": 270.0, "home_deg": 180.0},
    6: {"min_deg": 90.0, "max_deg": 270.0, "home_deg": 180.0},
    7: {"min_deg": 90.0, "max_deg": 180.0, "home_deg": 180.0},
}
SAFETY_MARGIN_DEG = 0.0

REAL_TO_SIM_SIGN = np.array([-1.0, 1.0, 1.0, -1.0, -1.0, -1.0], dtype=np.float32)
SIM_HOME_RAD = np.zeros(6, dtype=np.float32)

GRIPPER_ACTUATOR_NAME = "gripper_position"
GRIPPER_SIM_CLOSED_METER = 0.001
GRIPPER_SIM_OPEN_METER = 0.05
GRIPPER_REAL_OPEN_DEG = 180.0
GRIPPER_REAL_CLOSED_DEG = 90.0
INVERT_GRIPPER = False

# ============================================================
# 2. 仿真与可视化基础配置 (8x16 专属设置)
# ============================================================
XML_PATH_DEFAULT = "sim_rebotarm_colored_grasp.xml"

GRID_H, GRID_W = 8, 16
FORCE_MAX = 0.05
SENSOR_ORDER = "column_major_xy"
MODEL_ARENA_MEMORY = "64M"
AUTO_SCALE_ATTACK = 0.50
AUTO_SCALE_RELEASE = 0.05

HEATMAP_RESIZE_W = 480
HEATMAP_RESIZE_H = 240
ROTATE_HEATMAP_CLOCKWISE = True

ENABLE_MUJOCO_VIEWER = True
ENABLE_TOP_CAMERA = True
CAMERA_NAME = "top"
ENABLE_WRIST_CAMERA = True
WRIST_CAMERA_NAME = "cam_wrist"
CAMERA_WIDTH, CAMERA_HEIGHT = 320, 240

QUEUE_MAXSIZE = 1


# ============================================================
# 3. 工具函数 (数学截断、单位转换与队列)
# ============================================================
def clamp(val, min_val, max_val): return max(min_val, min(val, max_val))


def limit_real_deg(servo_id, angle_deg):
    cfg = JOINT_LIMITS_DEG[servo_id]
    return clamp(float(angle_deg), cfg["min_deg"] + SAFETY_MARGIN_DEG, cfg["max_deg"] - SAFETY_MARGIN_DEG)


def real_arm_deg_to_sim_rad(servo_id, angle_deg, arm_index):
    delta_deg = limit_real_deg(servo_id, angle_deg) - JOINT_LIMITS_DEG[servo_id]["home_deg"]
    return float(SIM_HOME_RAD[arm_index] + REAL_TO_SIM_SIGN[arm_index] * (delta_deg * np.pi / 180.0))


def real_gripper_deg_to_sim_meter(angle_deg, ctrl_min, ctrl_max):
    angle_deg = limit_real_deg(GRIPPER_SERVO_ID, angle_deg)
    denom = GRIPPER_REAL_OPEN_DEG - GRIPPER_REAL_CLOSED_DEG
    norm = 0.0 if abs(denom) < 1e-6 else (angle_deg - GRIPPER_REAL_CLOSED_DEG) / denom
    norm = clamp(norm, 0.0, 1.0)
    if INVERT_GRIPPER: norm = 1.0 - norm
    return float(clamp(ctrl_min + norm * (ctrl_max - ctrl_min), ctrl_min, ctrl_max)), float(norm)


def smooth_update(prev, target, alpha):
    return clamp(float(alpha), 0.0, 1.0) * target + (1.0 - clamp(float(alpha), 0.0, 1.0)) * prev


def put_latest(q: mp.Queue, item):
    try:
        while q.full():
            try:
                q.get_nowait()
            except queue.Empty:
                break
        q.put_nowait(item)
    except queue.Full:
        pass


def get_latest(q: mp.Queue, timeout=0.05):
    packet = q.get(timeout=timeout)
    while True:
        try:
            packet = q.get_nowait()
        except queue.Empty:
            return packet


# ============================================================
# 4. MuJoCo 传感器 & 触觉提取
# ============================================================
def find_joint_actuators(model, joint_names):
    actuator_ids = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        found = False
        for i in range(model.nu):
            if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT and model.actuator_trnid[i, 0] == joint_id:
                actuator_ids.append(i);
                found = True;
                break
        if not found: actuator_ids.append(-1)
    return actuator_ids


def find_sensors_by_keyword(model, keyword):
    return [i for i in range(model.nsensor) if
            keyword in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) or "")]


def load_model_with_extra_arena(xml_path, arena_memory=MODEL_ARENA_MEMORY):
    """Load an unmodified XML through a short-lived wrapper with more arena memory."""
    xml_path = Path(xml_path).resolve()
    wrapper = f'<mujoco model="runtime_wrapper"><size memory="{arena_memory}"/><include file="{xml_path.name}"/></mujoco>'
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", prefix=".runtime_",
                                         dir=xml_path.parent, delete=False) as temp_file:
            temp_file.write(wrapper)
            temp_path = Path(temp_file.name)
        return mujoco.MjModel.from_xml_path(str(temp_path))
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def enable_taxel_collisions(model):
    """Enable object-taxel contact while filtering left-pad/right-pad self contact."""
    counts = {"left": 0, "right": 0}
    backing_count = 0
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        body_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) or ""
        if geom_name in ("touch_base_left", "touch_base_right"):
            model.geom_contype[geom_id] = 1
            model.geom_conaffinity[geom_id] = 1
            backing_count += 1
        elif body_name.startswith("touch_cell_left_"):
            model.geom_contype[geom_id] = 2
            model.geom_conaffinity[geom_id] = 0
            counts["left"] += 1
        elif body_name.startswith("touch_cell_right_"):
            model.geom_contype[geom_id] = 4
            model.geom_conaffinity[geom_id] = 0
            counts["right"] += 1

    # Only free-joint objects receive the two taxel collision bits.  This keeps
    # taxels from contacting the finger's own large collision boxes, which would
    # otherwise create thousands of redundant constraints.
    tactile_bits = 2 | 4
    object_geom_count = 0
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        while body_id > 0:
            joint_adr = int(model.body_jntadr[body_id])
            joint_num = int(model.body_jntnum[body_id])
            if joint_num and np.any(model.jnt_type[joint_adr:joint_adr + joint_num] == mujoco.mjtJoint.mjJNT_FREE):
                model.geom_conaffinity[geom_id] |= tactile_bits
                object_geom_count += 1
                break
            body_id = int(model.body_parentid[body_id])
    if counts != {"left": GRID_H * GRID_W, "right": GRID_H * GRID_W}:
        raise RuntimeError(f"触觉 taxel geom 数量异常: {counts}")
    if object_geom_count == 0:
        raise RuntimeError("模型中没有找到带 free joint 的可抓取物体 geom")
    if backing_count != 2:
        raise RuntimeError(f"触觉背板数量异常: {backing_count}")
    return counts, object_geom_count, backing_count


def disable_taxel_collisions(model):
    """Keep taxels as display-only cells (physical fingers carry the load)."""
    counts = {"left": 0, "right": 0}
    backing = 0
    for gid in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                  int(model.geom_bodyid[gid])) or ""
        if gname in ("touch_base_left", "touch_base_right"):
            model.geom_contype[gid] = 0; model.geom_conaffinity[gid] = 0
            backing += 1
        elif bname.startswith("touch_cell_left_"):
            model.geom_contype[gid] = 0; model.geom_conaffinity[gid] = 0
            counts["left"] += 1
        elif bname.startswith("touch_cell_right_"):
            model.geom_contype[gid] = 0; model.geom_conaffinity[gid] = 0
            counts["right"] += 1
    if counts != {"left": GRID_H * GRID_W, "right": GRID_H * GRID_W}:
        raise RuntimeError(f"触觉 taxel geom 数量异常: {counts}")
    return counts, backing


def find_free_object_geom(model, data, grasp_site):
    """Select the free-joint object nearest the gripper at startup."""
    candidates = []
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        root = bid
        has_free = False
        while root > 0:
            ja = int(model.body_jntadr[root]); jn = int(model.body_jntnum[root])
            if jn and np.any(model.jnt_type[ja:ja + jn] == mujoco.mjtJoint.mjJNT_FREE):
                has_free = True; break
            root = int(model.body_parentid[root])
        if has_free:
            candidates.append((float(np.linalg.norm(data.geom_xpos[gid] - data.site_xpos[grasp_site])), gid))
    if not candidates:
        raise RuntimeError("模型中没有找到 free-joint 抓取物体")
    return min(candidates, key=lambda x: x[0])[1]


def tactile_from_pad_proximity(model, data, obj_geom, side):
    """8x16 display-only tactile map from exact taxel/object surface distance."""
    prefix = f"touch_cell_{side}_"
    ids = []
    for gid in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                 int(model.geom_bodyid[gid])) or ""
        if body.startswith(prefix): ids.append(gid)
    ids.sort(key=lambda gid: int((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                                     int(model.geom_bodyid[gid])) or "_000")[-3:]))
    if len(ids) != GRID_H * GRID_W:
        return np.zeros((GRID_H, GRID_W), dtype=np.float32)
    d = np.empty(len(ids), dtype=np.float32); fromto = np.empty(6, dtype=np.float64)
    for k, gid in enumerate(ids):
        d[k] = float(mujoco.mj_geomDistance(model, data, int(gid), int(obj_geom), 0.02, fromto))
    sigma, reach = 0.00030, 0.00125
    values = np.exp(-np.square(np.maximum(d, 0.0) / sigma)).astype(np.float32)
    values[d > reach] = 0.0; values[values < 0.05] = 0.0
    return values.reshape(GRID_W, GRID_H).T


def tactile_window(left, right):
    """Tall, center-symmetric left/right display with academic labels."""
    left = np.nan_to_num(left).clip(0, 1)
    right_in_left = np.fliplr(np.nan_to_num(right).clip(0, 1))
    symmetric = 0.5 * (left + right_in_left)
    def panel(a, label, mirror=False):
        im = cv2.applyColorMap((a * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        im = cv2.resize(im, (160, 320), interpolation=cv2.INTER_NEAREST)
        if mirror: im = cv2.flip(im, 1)
        cv2.putText(im, label, (8, 24), cv2.FONT_HERSHEY_TRIPLEX, .62,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return im
    canvas = np.hstack((panel(symmetric, "left"), panel(symmetric, "right", True)))
    cv2.line(canvas, (canvas.shape[1] // 2, 0),
             (canvas.shape[1] // 2, canvas.shape[0]), (255, 255, 255), 2)
    return canvas


def values_to_grid(values, grid_shape=(8, 16), order="column_major_xy"):
    h, w = grid_shape
    if order == "column_major_xy": return values.reshape(w, h).T
    if order == "row_major_xy": return values.reshape(h, w)
    raise ValueError(f"未知 SENSOR_ORDER: {order}")


def build_tactile_reader(model, keyword, grid_shape=(8, 16)):
    h, w = grid_shape
    sensor_ids = find_sensors_by_keyword(model, keyword)
    if len(sensor_ids) == h * w:
        dims = np.array([model.sensor_dim[sid] for sid in sensor_ids], dtype=np.int32)
        adrs = np.array([model.sensor_adr[sid] for sid in sensor_ids], dtype=np.int32)
        return {"mode": "grid_force", "keyword": keyword, "sensor_ids": sensor_ids, "adrs": adrs, "dims": dims,
                "grid_shape": grid_shape}
    if len(sensor_ids) == 1:
        sid = sensor_ids[0]
        return {"mode": "single_force", "keyword": keyword, "sensor_ids": sensor_ids, "adr": int(model.sensor_adr[sid]),
                "dim": int(model.sensor_dim[sid]), "grid_shape": grid_shape}
    raise RuntimeError(f"触觉 sensor 数量异常: keyword='{keyword}', found={len(sensor_ids)} (期望 {h * w})")


def read_tactile(reader, data, force_max=None, sensor_order="column_major_xy"):
    mode, h, w = reader["mode"], reader["grid_shape"][0], reader["grid_shape"][1]
    if mode == "grid_force":
        adrs, dims = reader["adrs"], reader["dims"]
        if np.all(dims == 3):
            raw = data.sensordata[adrs[:, None] + np.array([0, 1, 2], dtype=np.int32)]
            values = np.maximum(np.linalg.norm(raw, axis=1), 0.0)
            if force_max is not None:
                values = np.clip(values, 0.0, force_max)
            return values_to_grid(values, (h, w), sensor_order).astype(np.float32)
        values = np.zeros(h * w, dtype=np.float32)
        for k, adr in enumerate(adrs):
            dim = int(dims[k])
            raw = data.sensordata[adr:adr + dim]
            values[k] = max(float(raw[0] if dim == 1 else np.linalg.norm(raw)), 0.0)
            if force_max is not None:
                values[k] = min(values[k], force_max)
        return values_to_grid(values, (h, w), sensor_order).astype(np.float32)
    elif mode == "single_force":
        raw = data.sensordata[reader["adr"]:reader["adr"] + reader["dim"]]
        val = max(float(raw[0] if reader["dim"] == 1 else np.linalg.norm(raw)), 0.0)
        if force_max is not None:
            val = min(val, force_max)
        return np.full((h, w), val, dtype=np.float32)
    return np.zeros((h, w), dtype=np.float32)


# ============================================================
# 5. 图像渲染与初始化 (Heatmap & Offscreen)
# ============================================================
def tactile_to_colormap(tactile, force_max=1.0, resize_w=480, resize_h=240, rotate_clockwise=True,
                        label=None):
    tactile_normalized = np.clip(tactile, 0.0, force_max) / force_max * 255.0
    colored = cv2.applyColorMap(tactile_normalized.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    colored_resized = cv2.resize(colored, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)
    if rotate_clockwise:
        colored_resized = cv2.rotate(colored_resized, cv2.ROTATE_90_CLOCKWISE)
    if label:
        cv2.putText(colored_resized, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return colored_resized


def init_offscreen_renderer(model, camera_name="top", width=320, height=240):
    if not glfw.init(): raise RuntimeError("GLFW 初始化失败")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(width, height, "offscreen", None, None)
    glfw.make_context_current(window)
    cam = mujoco.MjvCamera()
    cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)
    return window, cam, scene, context, mujoco.MjrRect(0, 0, width, height)


def make_fixed_camera(model, camera_name):
    """Create another camera descriptor sharing the existing GL renderer."""
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        return None
    cam = mujoco.MjvCamera()
    cam.fixedcamid = camera_id
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    return cam


def render_camera_view(model, data, window, cam, scene, context, viewport, width=320, height=240):
    glfw.make_context_current(window)
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(rgb, None, viewport, context)
    return cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)


# ============================================================
# 6. 真实舵机读取线程 (异步)
# ============================================================
def servo_reader_worker(scs, state_lock, shared_state, read_rate, no_gripper, g_min, g_max, stop_event):
    read_period = 1.0 / max(read_rate, 1e-6)
    arm_deg = np.array([JOINT_LIMITS_DEG[i]["home_deg"] for i in ARM_SERVO_IDS], dtype=np.float32)
    gripper_deg = JOINT_LIMITS_DEG[GRIPPER_SERVO_ID]["home_deg"]

    while not stop_event.is_set():
        loop_start = time.perf_counter()

        for i, servo_id in enumerate(ARM_SERVO_IDS):
            try:
                pos, _, res, _ = scs.ReadPosSpeed(servo_id)
                if res == COMM_SUCCESS: arm_deg[i] = limit_real_deg(servo_id, (
                            float(pos) / SERVO_DIGITAL_RANGE) * SERVO_ANGLE_RANGE)
            except:
                pass

        if not no_gripper:
            try:
                pos, _, res, _ = scs.ReadPosSpeed(GRIPPER_SERVO_ID)
                if res == COMM_SUCCESS: gripper_deg = limit_real_deg(GRIPPER_SERVO_ID, (
                            float(pos) / SERVO_DIGITAL_RANGE) * SERVO_ANGLE_RANGE)
            except:
                pass

        target_arm_q = np.array([real_arm_deg_to_sim_rad(s, arm_deg[i], i) for i, s in enumerate(ARM_SERVO_IDS)],
                                dtype=np.float32)
        target_gripper_cmd, _ = real_gripper_deg_to_sim_meter(gripper_deg, g_min, g_max)

        with state_lock:
            shared_state["target_arm_q"] = target_arm_q
            shared_state["target_gripper_cmd"] = target_gripper_cmd

        sleep_time = read_period - (time.perf_counter() - loop_start)
        if sleep_time > 0: time.sleep(sleep_time)


# ============================================================
# 7. 仿真主进程 (接管物理与控制流)
# ============================================================
def simulation_process(args, frame_queue: mp.Queue, stop_event: mp.Event):
    print("[SIM] 加载模型与初始化...")
    model = load_model_with_extra_arena(args.xml)
    taxel_counts, backing_count = disable_taxel_collisions(model)
    model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_MULTICCD)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    grasp_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_center")
    if grasp_site < 0:
        raise RuntimeError("模型中缺少 grasp_center site")
    tactile_object_geom = find_free_object_geom(model, data, grasp_site)
    tactile_object_count = 1

    right_reader = build_tactile_reader(model, keyword="touch_point_right", grid_shape=(GRID_H, GRID_W))
    left_reader = build_tactile_reader(model, keyword="touch_point_left", grid_shape=(GRID_H, GRID_W))

    joint_names = [x.strip() for x in args.joint_names.split(",")]
    arm_actuator_ids = find_joint_actuators(model, joint_names)
    gripper_act_id = -1 if args.no_gripper else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                                                  GRIPPER_ACTUATOR_NAME)

    gripper_ctrl_min = float(
        model.actuator_ctrlrange[gripper_act_id, 0]) if gripper_act_id >= 0 else GRIPPER_SIM_CLOSED_METER
    gripper_ctrl_max = float(
        model.actuator_ctrlrange[gripper_act_id, 1]) if gripper_act_id >= 0 else GRIPPER_SIM_OPEN_METER

    # 初始化串口
    portHandler = PortHandler(args.port)
    scs = sts(portHandler)
    if portHandler.openPort() and portHandler.setBaudRate(args.baudrate):
        print(f"[SIM] ✅ 串口通信已建立: {args.port}")
        time.sleep(2.0)
        if not args.keep_torque:
            for sid in ARM_SERVO_IDS + ([] if args.no_gripper else [GRIPPER_SERVO_ID]):
                scs.write1ByteTxRx(sid, STS_TORQUE_ENABLE, 0)
    else:
        print("[SIM] ❌ 串口打开失败！");
        stop_event.set();
        return

    # 启动异步舵机读取
    state_lock = threading.Lock()
    shared_state = {
        "target_arm_q": np.zeros(ARM_DOF, dtype=np.float32),
        "target_gripper_cmd": float(gripper_ctrl_min)
    }
    threading.Thread(target=servo_reader_worker, args=(
        scs, state_lock, shared_state, args.read_rate, args.no_gripper, gripper_ctrl_min, gripper_ctrl_max, stop_event),
                     daemon=True).start()

    viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False,
                                          show_right_ui=False) if ENABLE_MUJOCO_VIEWER else None
    offscreen = init_offscreen_renderer(model, CAMERA_NAME, CAMERA_WIDTH, CAMERA_HEIGHT) if ENABLE_TOP_CAMERA else None
    wrist_cam = make_fixed_camera(model, WRIST_CAMERA_NAME) if (offscreen and ENABLE_WRIST_CAMERA) else None
    if ENABLE_WRIST_CAMERA and wrist_cam is None:
        print(f"[SIM] 警告：未找到相机 {WRIST_CAMERA_NAME}，腕部相机窗口将禁用。", flush=True)

    # 时间轴管理
    sim_period = 1.0 / max(args.rate, 1e-6)
    steps_per_loop = max(1, int(round(sim_period / float(model.opt.timestep))))

    vis_interval = max(1, int(round(args.rate / args.vis_hz)))
    viewer_interval = max(1, int(round(args.rate / args.viewer_hz)))
    camera_interval = max(1, int(round(args.rate / args.camera_hz)))

    filtered_arm_q = np.zeros(ARM_DOF, dtype=np.float32)
    filtered_gripper_cmd = gripper_ctrl_min
    loop_count = 0

    print(f"[SIM] 触觉模式: distance-proximity，taxel={taxel_counts}，"
          f"背板={backing_count}，object_geom={tactile_object_geom}，CCD=ON")
    print(f"[SIM] 仿真循环启动。频率:{args.rate}Hz，mj_step:{steps_per_loop}")
    try:
        while not stop_event.is_set():
            loop_start = time.perf_counter()
            if viewer and not viewer.is_running():
                stop_event.set();
                break

            # 1. 提取指令与滤波
            with state_lock:
                target_q = shared_state["target_arm_q"].copy()
                target_g = shared_state["target_gripper_cmd"]

            filtered_arm_q = smooth_update(filtered_arm_q, target_q, args.alpha_arm)
            filtered_gripper_cmd = smooth_update(filtered_gripper_cmd, target_g, args.alpha_gripper)

            # 2. 映射写入 data.ctrl
            for i, act_id in enumerate(arm_actuator_ids):
                if act_id >= 0: data.ctrl[act_id] = filtered_arm_q[i]
            if gripper_act_id >= 0: data.ctrl[gripper_act_id] = filtered_gripper_cmd

            # 3. 推进物理引擎
            for _ in range(steps_per_loop):
                mujoco.mj_step(model, data)

            # 4. 触觉 & 视觉派发
            frame_packet = None
            if loop_count % vis_interval == 0:
                touch_right = tactile_from_pad_proximity(model, data, tactile_object_geom, "right")
                touch_left = tactile_from_pad_proximity(model, data, tactile_object_geom, "left")
                frame_packet = {
                    "touch_right": touch_right,
                    "touch_left": touch_left
                }

            if ENABLE_TOP_CAMERA and offscreen and loop_count % camera_interval == 0:
                win, cam, scn, ctx, vp = offscreen
                if frame_packet is None:
                    frame_packet = {}
                frame_packet["top_view"] = render_camera_view(
                    model, data, win, cam, scn, ctx, vp, CAMERA_WIDTH, CAMERA_HEIGHT)
                if wrist_cam is not None:
                    frame_packet["wrist_view"] = render_camera_view(
                        model, data, win, wrist_cam, scn, ctx, vp, CAMERA_WIDTH, CAMERA_HEIGHT)
            if frame_packet is not None:
                put_latest(frame_queue, frame_packet)

            if viewer and loop_count % viewer_interval == 0: viewer.sync()

            if loop_count % max(1, int(args.rate)) == 0:
                right_peak = float(np.max(frame_packet["touch_right"])) if frame_packet and "touch_right" in frame_packet else 0.0
                left_peak = float(np.max(frame_packet["touch_left"])) if frame_packet and "touch_left" in frame_packet else 0.0
                print(f"[TACTILE] ncon={data.ncon}, left_peak={left_peak:.4g}, right_peak={right_peak:.4g}")

            # 5. 精确锁频 (替代 loop_rate_limiters)
            loop_count += 1
            sleep_time = sim_period - (time.perf_counter() - loop_start)
            if sleep_time > 0: time.sleep(sleep_time)

    finally:
        if viewer: viewer.close()
        if offscreen: glfw.destroy_window(offscreen[0]); glfw.terminate()
        portHandler.closePort()
        print("[SIM] 仿真进程完全停止。")


# ============================================================
# 8. 可视化进程 (支持旋转 Heatmap)
# ============================================================
def visualization_process(frame_queue: mp.Queue, stop_event: mp.Event):
    print("[VIS] 可视化进程启动。")
    cv2.namedWindow("Touch Heatmap - Left | Right", cv2.WINDOW_NORMAL)
    if ENABLE_TOP_CAMERA: cv2.namedWindow("Top Camera View", cv2.WINDOW_NORMAL)
    if ENABLE_WRIST_CAMERA: cv2.namedWindow("Wrist Camera View", cv2.WINDOW_NORMAL)

    last_tr = None
    last_tl = None
    last_top = None
    last_wrist = None
    display_force_max = FORCE_MAX

    while not stop_event.is_set():
        try:
            packet = get_latest(frame_queue, timeout=0.02)
        except queue.Empty:
            packet = {}

        tr = packet.get("touch_right")
        tl = packet.get("touch_left")
        top = packet.get("top_view")
        wrist = packet.get("wrist_view")
        if tr is not None: last_tr = tr
        if tl is not None: last_tl = tl
        tr, tl = last_tr, last_tl

        tactile_arrays = [x for x in (tr, tl) if x is not None]
        if tactile_arrays:
            observed_max = max(float(np.percentile(x, 99.0)) for x in tactile_arrays)
            target_scale = max(FORCE_MAX, observed_max)
            alpha = AUTO_SCALE_ATTACK if target_scale >= display_force_max else AUTO_SCALE_RELEASE
            display_force_max = alpha * target_scale + (1.0 - alpha) * display_force_max

        if tr is not None and tl is not None:
            cv2.imshow("Touch Heatmap - Left | Right", tactile_window(tl, tr))
        if top is not None: last_top = top
        if ENABLE_TOP_CAMERA and last_top is not None: cv2.imshow("Top Camera View", last_top)
        if wrist is not None: last_wrist = wrist
        if ENABLE_WRIST_CAMERA and last_wrist is not None: cv2.imshow("Wrist Camera View", last_wrist)

        if cv2.waitKey(1) in [27, ord('q')]:
            stop_event.set();
            break

    cv2.destroyAllWindows()
    print("[VIS] 界面已关闭。")


# ============================================================
# 9. Main 入口与命令行参数
# ============================================================
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default=XML_PATH_DEFAULT)
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=115200)

    parser.add_argument("--read-rate", type=float, default=60.0, help="串口读取速率")
    parser.add_argument("--rate", type=float, default=100.0, help="仿真物理刷新率(SIM_HZ)")
    parser.add_argument("--vis-hz", type=float, default=50.0, help="触觉渲染率")
    parser.add_argument("--viewer-hz", type=float, default=30.0, help="MuJoCo Viewer 刷新率")
    parser.add_argument("--camera-hz", type=float, default=5.0, help="Top 相机离屏渲染率")

    parser.add_argument("--alpha-arm", type=float, default=0.90)
    parser.add_argument("--alpha-gripper", type=float, default=0.90)
    parser.add_argument("--joint-names", type=str, default="joint1,joint2,joint3,joint4,joint5,joint6")
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--keep-torque", action="store_true")
    args = parser.parse_args()

    frame_queue = mp.Queue(maxsize=QUEUE_MAXSIZE)
    stop_event = mp.Event()

    sim_proc = mp.Process(target=simulation_process, args=(args, frame_queue, stop_event))
    vis_proc = mp.Process(target=visualization_process, args=(frame_queue, stop_event))

    sim_proc.start()
    vis_proc.start()

    try:
        while sim_proc.is_alive() and vis_proc.is_alive(): time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[MAIN] 手动中断，通知子进程退出...")
    finally:
        stop_event.set()
        sim_proc.join(timeout=3.0)
        vis_proc.join(timeout=3.0)
        if sim_proc.is_alive(): sim_proc.terminate()
        if vis_proc.is_alive(): vis_proc.terminate()
        print("[MAIN] 系统安全退出。")
