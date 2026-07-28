#!/usr/bin/env python3
import argparse
import logging
import multiprocessing as mp
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import glfw
import mujoco
import mujoco.viewer
import numpy as np

# ============================================================
# 0. 动态环境变量注入 (修复 SDK 路径)
# ============================================================
current_dir = Path(__file__).resolve().parent
python_root = current_dir.parent.parent  # 向上两级：当前目录 -> Servo_control -> rebot_scripts
sdk_path = python_root / "STservo_sdk"

print(f"[路径检查] 当前脚本目录: {current_dir}")
print(f"[路径检查] Python根目录: {python_root}")
print(f"[路径检查] SDK目录: {sdk_path}")

if not sdk_path.exists():
    print(f"❌ 未找到 STservo_sdk 目录: {sdk_path}")
    sys.exit(1)

for p in [str(python_root), str(sdk_path)]:
    if p not in sys.path:
        sys.path.append(p)

from STservo_sdk import *

logging.getLogger().setLevel(logging.ERROR)

# ============================================================
# 1. 硬件 & 映射配置 (Real2Sim)
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

GRIPPER_ACTUATOR_NAME = "gripper"
GRIPPER_SIM_CLOSED_METER = 0.001
GRIPPER_SIM_OPEN_METER = 0.05
GRIPPER_REAL_OPEN_DEG = 180.0
GRIPPER_REAL_CLOSED_DEG = 90.0
INVERT_GRIPPER = False

# ============================================================
# 2. 仿真 & 可视化基础参数
# ============================================================
XML_PATH_DEFAULT = "/home/hjx/hjx_file/rebot_devarm_ws/rebotArm_policy_learning/act_tactile/assets/rebotarm_sim_transfer_cube.xml"

# 触觉数据形状：C, H, W
TACTILE_C, GRID_H, GRID_W = 3, 16, 16
TOUCH_SHAPE = (TACTILE_C, GRID_H, GRID_W)

MAX_SHEAR, MAX_PRESSURE = 0.01, 0.02
WINDOW_SIZE = (480, 480)
HEATMAP_FORCE_MAX = 0.05
HEATMAP_DISPLAY_SIZE = 480

ENABLE_MUJOCO_VIEWER = True
ENABLE_TOP_CAMERA = False

CAMERA_NAME = "top"
CAMERA_WIDTH, CAMERA_HEIGHT = 320, 240

CAMERA_HZ = 5.0      # top camera 渲染频率
VIEWER_HZ = 30.0     # MuJoCo viewer 同步频率

QUEUE_MAXSIZE = 1
SENSOR_ORDER = "row_major_xy"


# ============================================================
# 3. 工具函数 (队列 & 硬件映射)
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
    alpha = clamp(float(alpha), 0.0, 1.0)
    return alpha * target + (1.0 - alpha) * prev


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
# 4. MuJoCo 传感器 & 执行器工具
# ============================================================
def find_joint_actuators(model, joint_names):
    actuator_ids = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        found = False
        for i in range(model.nu):
            if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT and model.actuator_trnid[i, 0] == joint_id:
                actuator_ids.append(i)
                found = True;
                break
        if not found: actuator_ids.append(-1)
    return actuator_ids


def find_exact_sensor(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)


def find_sensors_by_keyword(model, keyword):
    return [i for i in range(model.nsensor) if
            keyword in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) or "")]


def values_to_grid(values, grid_shape=(16, 16), order="row_major_xy"):
    h, w = grid_shape
    if order == "row_major_xy": return values.reshape(h, w)
    return values.reshape(w, h).T


def build_tactile3_reader(model, side):
    exact_name = f"touch_{side}"
    point_keyword = f"touch_point_{side}"
    sid = find_exact_sensor(model, exact_name)

    if sid >= 0:
        return {"mode": "grid3_exact", "side": side, "sensor_id": sid, "adr": int(model.sensor_adr[sid]),
                "dim": int(model.sensor_dim[sid])}

    sensor_ids = find_sensors_by_keyword(model, point_keyword)
    if len(sensor_ids) == GRID_H * GRID_W:
        dims = np.array([model.sensor_dim[i] for i in sensor_ids], dtype=np.int32)
        adrs = np.array([model.sensor_adr[i] for i in sensor_ids], dtype=np.int32)
        return {"mode": "points_force", "side": side, "sensor_ids": sensor_ids, "dims": dims, "adrs": adrs}

    raise RuntimeError(f"无法构建 {side} 侧触觉读取器。")


def read_tactile3(reader, data, sensor_order="row_major_xy"):
    mode = reader["mode"]
    if mode == "grid3_exact":
        return data.sensordata[reader["adr"]:reader["adr"] + reader["dim"]].reshape(TOUCH_SHAPE).astype(np.float32)
    elif mode == "points_force":
        adrs = reader["adrs"]
        index = adrs[:, None] + np.array([0, 1, 2], dtype=np.int32)
        raw = data.sensordata[index]
        return np.stack([values_to_grid(raw[:, i], (GRID_H, GRID_W), sensor_order) for i in range(3)], axis=0).astype(
            np.float32)
    return np.zeros(TOUCH_SHAPE, dtype=np.float32)


# ============================================================
# 5. 图像渲染与渲染器初始化 (触觉、热力图、相机)
# ============================================================
def tactile_to_arrow_image(tactile, size=(480, 480), max_shear=0.05, max_pressure=0.1, arrow_scale=20.0,
                           rotate_180=True):
    channels, ny, nx = tactile.shape
    loc_x, loc_y = np.linspace(0, size[1], nx), np.linspace(size[0], 0, ny)
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)

    for i in range(nx):
        for j in range(ny):
            dir_x = np.clip(tactile[0, j, i] / max_shear, -1.0, 1.0) * arrow_scale
            dir_y = np.clip(tactile[1, j, i] / max_shear, -1.0, 1.0) * arrow_scale
            pressure = np.clip(tactile[2, j, i] / max_pressure, 0.0, 1.0)
            cv2.arrowedLine(img, (int(loc_y[i]), int(loc_x[j])), (int(loc_y[i] + dir_y), int(loc_x[j] - dir_x)),
                            (0, int(255 * (1.0 - pressure)), int(255 * pressure)), 2, tipLength=0.5)
    return cv2.rotate(img, cv2.ROTATE_180) if rotate_180 else img


def tactile_to_rgb_heatmap(tactile, force_max=0.05, display_size=480):
    force_mag = np.linalg.norm(tactile, axis=0)
    normalized = np.clip(force_mag, 0.0, force_max) / force_max * 255.0
    heatmap = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    return cv2.resize(heatmap, (display_size, display_size), interpolation=cv2.INTER_NEAREST)


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


def render_camera_view(model, data, window, cam, scene, context, viewport, width, height):
    glfw.make_context_current(window)
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(rgb, None, viewport, context)
    return cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)


# ============================================================
# 6. 真实舵机读取线程
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
                if res == COMM_SUCCESS: arm_deg[i] = limit_real_deg(servo_id,
                                                                    (float(pos) / SERVO_DIGITAL_RANGE) * 360.0)
            except:
                pass

        if not no_gripper:
            try:
                pos, _, res, _ = scs.ReadPosSpeed(GRIPPER_SERVO_ID)
                if res == COMM_SUCCESS: gripper_deg = limit_real_deg(GRIPPER_SERVO_ID,
                                                                     (float(pos) / SERVO_DIGITAL_RANGE) * 360.0)
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
# 7. 独立进程：仿真系统 (MuJoCo + 串口)
# ============================================================
def simulation_process(args, frame_queue: mp.Queue, stop_event: mp.Event):
    print("[SIM] 启动 MuJoCo 仿真...")
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    # 传感器 & 执行器准备
    left_reader = build_tactile3_reader(model, "left")
    right_reader = build_tactile3_reader(model, "right")

    joint_names = [x.strip() for x in args.joint_names.split(",")]
    arm_actuator_ids = find_joint_actuators(model, joint_names)
    gripper_act_id = -1 if args.no_gripper else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                                                  GRIPPER_ACTUATOR_NAME)

    gripper_ctrl_min = float(
        model.actuator_ctrlrange[gripper_act_id, 0]) if gripper_act_id >= 0 else GRIPPER_SIM_CLOSED_METER
    gripper_ctrl_max = float(
        model.actuator_ctrlrange[gripper_act_id, 1]) if gripper_act_id >= 0 else GRIPPER_SIM_OPEN_METER

    # 串口初始化
    portHandler = PortHandler(args.port)
    scs = sts(portHandler)
    if portHandler.openPort() and portHandler.setBaudRate(args.baudrate):
        print(f"[SIM] ✅ 串口已打开: {args.port}")
        time.sleep(2.0)
        if not args.keep_torque:
            for sid in ARM_SERVO_IDS + ([] if args.no_gripper else [GRIPPER_SERVO_ID]):
                scs.write1ByteTxRx(sid, STS_TORQUE_ENABLE, 0)
    else:
        print("[SIM] ❌ 串口打开失败！");
        stop_event.set();
        return

    # 共享内存与读取线程
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

    # 控制与渲染频率规划
    sim_period = 1.0 / max(args.rate, 1e-6)
    # 动态匹配真实时间：每轮循环补足对应的物理时间 (sim_period / timestep)
    steps_per_loop = max(1, int(round(sim_period / float(model.opt.timestep))))

    vis_interval = max(1, int(round(args.rate / args.vis_hz)))
    camera_interval = max(1, int(round(args.rate / CAMERA_HZ)))
    viewer_interval = max(1, int(round(args.rate / VIEWER_HZ)))

    filtered_arm_q = np.zeros(ARM_DOF, dtype=np.float32)
    filtered_gripper_cmd = gripper_ctrl_min
    loop_count = 0

    print("[SIM] 仿真循环开始，控制映射已接管 data.ctrl。")
    try:
        while not stop_event.is_set():
            loop_start = time.perf_counter()
            if viewer and not viewer.is_running():
                stop_event.set();
                break

            # 1. 提取指令并滤波
            with state_lock:
                target_q = shared_state["target_arm_q"].copy()
                target_g = shared_state["target_gripper_cmd"]

            filtered_arm_q = smooth_update(filtered_arm_q, target_q, args.alpha_arm)
            filtered_gripper_cmd = smooth_update(filtered_gripper_cmd, target_g, args.alpha_gripper)

            # 2. 写入 data.ctrl 驱动物理特性
            for i, act_id in enumerate(arm_actuator_ids):
                if act_id >= 0: data.ctrl[act_id] = filtered_arm_q[i]
            if gripper_act_id >= 0: data.ctrl[gripper_act_id] = filtered_gripper_cmd

            # 3. 推进物理引擎
            for _ in range(steps_per_loop):
                mujoco.mj_step(model, data)

            # 4. 触觉 & 视觉派发
            if loop_count % vis_interval == 0:
                put_latest(frame_queue, {
                    "tactile_left": read_tactile3(left_reader, data, SENSOR_ORDER),
                    "tactile_right": read_tactile3(right_reader, data, SENSOR_ORDER)
                })

            if ENABLE_TOP_CAMERA and offscreen and loop_count % camera_interval == 0:
                win, cam, scn, ctx, vp = offscreen
                put_latest(frame_queue, {
                    "top_view": render_camera_view(model, data, win, cam, scn, ctx, vp, CAMERA_WIDTH, CAMERA_HEIGHT)})

            if viewer and loop_count % viewer_interval == 0: viewer.sync()

            loop_count += 1
            sleep_time = sim_period - (time.perf_counter() - loop_start)
            if sleep_time > 0: time.sleep(sleep_time)

    finally:
        if viewer: viewer.close()
        if offscreen: glfw.destroy_window(offscreen[0]); glfw.terminate()
        portHandler.closePort()
        print("[SIM] 仿真进程退出")


# ============================================================
# 8. 独立进程：OpenCV 可视化 (Heatmap & Camera)
# ============================================================
def visualization_process(frame_queue: mp.Queue, stop_event: mp.Event):
    print("[VIS] 视觉与触觉渲染进程启动...")
    cv2.namedWindow("Touch Left", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Touch Right", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Touch RGB Heatmap Left", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Touch RGB Heatmap Right", cv2.WINDOW_NORMAL)
    if ENABLE_TOP_CAMERA: cv2.namedWindow("Top Camera View", cv2.WINDOW_NORMAL)

    last_top = None

    while not stop_event.is_set():
        try:
            packet = get_latest(frame_queue, timeout=0.02)
        except queue.Empty:
            continue

        tl = packet.get("tactile_left")
        tr = packet.get("tactile_right")
        top = packet.get("top_view")

        if tl is not None:
            cv2.imshow("Touch Left", tactile_to_arrow_image(tl, WINDOW_SIZE, MAX_SHEAR, MAX_PRESSURE))
            cv2.imshow("Touch RGB Heatmap Left", tactile_to_rgb_heatmap(tl, HEATMAP_FORCE_MAX, HEATMAP_DISPLAY_SIZE))
        if tr is not None:
            cv2.imshow("Touch Right", tactile_to_arrow_image(tr, WINDOW_SIZE, MAX_SHEAR, MAX_PRESSURE))
            cv2.imshow("Touch RGB Heatmap Right", tactile_to_rgb_heatmap(tr, HEATMAP_FORCE_MAX, HEATMAP_DISPLAY_SIZE))
        if top is not None: last_top = top
        if ENABLE_TOP_CAMERA and last_top is not None: cv2.imshow("Top Camera View", last_top)

        if cv2.waitKey(1) in [27, ord('q')]:
            stop_event.set();
            break

    cv2.destroyAllWindows()
    print("[VIS] 可视化进程退出")


# ============================================================
# 9. Main 启动入口 (附加 CLI 参数控制)
# ============================================================
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default=XML_PATH_DEFAULT)
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--read-rate", type=float, default=60.0)
    parser.add_argument("--rate", type=float, default=100.0, help="仿真控制端的主循环频率")
    parser.add_argument("--vis-hz", type=float, default=30.0)
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
        print("\n[MAIN] 接收到退出指令...")
    finally:
        stop_event.set()
        sim_proc.join(timeout=3.0)
        vis_proc.join(timeout=3.0)
        if sim_proc.is_alive(): sim_proc.terminate()
        if vis_proc.is_alive(): vis_proc.terminate()
        print("[MAIN] 所有进程已安全结束！")
