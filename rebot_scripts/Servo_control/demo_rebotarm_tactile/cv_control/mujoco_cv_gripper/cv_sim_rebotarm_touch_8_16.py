#!/usr/bin/env python3
import argparse
import logging
import multiprocessing as mp
import queue
import sys
import threading
import time
import math
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np

# ============================================================
# 1. 动态环境变量注入 & SDK 路径修复
# ============================================================
current_dir = Path(__file__).resolve().parent
# 向上退 4 级：mujoco_cv_gripper -> cv_control -> demo_rebotarm_tactile -> Servo_control -> rebot_scripts
python_root = current_dir.parents[3]
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
# 2. 硬件 & 映射配置 (Real2Sim 舵机参数)
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
# 3. 仿真与可视化基础配置
# ============================================================
XML_PATH_DEFAULT = "/home/hjx/hjx_file/rebot_devarm_ws/rebotArm_policy_learning/act_tactile/assets/rebotarm_sim_transfer_cube.xml"

GRID_H, GRID_W = 8, 16
SENSOR_ORDER = "column_major_xy"

# RGB 热力图触觉参数
FORCE_MAX = 0.05
HEATMAP_RESIZE_W = 480
HEATMAP_RESIZE_H = 240
ROTATE_HEATMAP_CLOCKWISE = True

# MuJoCo Viewer 开关
ENABLE_MUJOCO_VIEWER = True

# ✅ 仿真相机组合配置 (大尺寸、三相机并行)
ENABLE_SIM_CAMERAS = True
TOP_CAMERA_NAME = "top"
WRIST_CAMERA_NAME = "arm_wrist"
ANGLE_CAMERA_NAME = "angle"
CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480

# 手势控制相机
ENABLE_HAND_CAMERA = True
HAND_HZ = 30.0
HAND_CAMERA_ID = 0
HAND_CAMERA_WIDTH, HAND_CAMERA_HEIGHT = 640, 480

QUEUE_MAXSIZE = 1


# ============================================================
# 4. 工具函数
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
# 5. MuJoCo 传感器 & RGB 触觉提取
# ============================================================
def find_joint_actuators(model, joint_names):
    actuator_ids = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        found = False
        for i in range(model.nu):
            if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT and model.actuator_trnid[i, 0] == joint_id:
                actuator_ids.append(i)
                found = True
                break
        if not found: actuator_ids.append(-1)
    return actuator_ids


def find_sensors_by_keyword(model, keyword):
    return [i for i in range(model.nsensor) if
            keyword in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) or "")]


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


def read_tactile(reader, data, force_max=1.0, sensor_order="column_major_xy"):
    mode, h, w = reader["mode"], reader["grid_shape"][0], reader["grid_shape"][1]
    if mode == "grid_force":
        adrs, dims = reader["adrs"], reader["dims"]
        if np.all(dims == 3):
            # 提取三维力向量并计算模长作为总受力大小
            raw = data.sensordata[adrs[:, None] + np.array([0, 1, 2], dtype=np.int32)]
            values = np.clip(np.linalg.norm(raw, axis=1), 0.0, force_max)
            return values_to_grid(values, (h, w), sensor_order).astype(np.float32)

        values = np.zeros(h * w, dtype=np.float32)
        for k, adr in enumerate(adrs):
            dim = int(dims[k])
            raw = data.sensordata[adr:adr + dim]
            values[k] = np.clip(raw[0] if dim == 1 else np.linalg.norm(raw), 0.0, force_max)
        return values_to_grid(values, (h, w), sensor_order).astype(np.float32)

    elif mode == "single_force":
        raw = data.sensordata[reader["adr"]:reader["adr"] + reader["dim"]]
        val = float(np.clip(raw[0] if reader["dim"] == 1 else np.linalg.norm(raw), 0.0, force_max))
        return np.full((h, w), val, dtype=np.float32)

    return np.zeros((h, w), dtype=np.float32)


# ============================================================
# 6. RGB 图像渲染与相机离屏渲染
# ============================================================
def tactile_to_colormap(tactile, force_max=1.0, resize_w=480, resize_h=240, rotate_clockwise=True):
    """使用 OpenCV 渲染 RGB 热力图，数值越大越偏向黄绿色"""
    tactile_normalized = np.clip(tactile, 0.0, force_max) / force_max * 255.0
    colored = cv2.applyColorMap(tactile_normalized.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    colored_resized = cv2.resize(colored, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)
    if rotate_clockwise:
        colored_resized = cv2.rotate(colored_resized, cv2.ROTATE_90_CLOCKWISE)
    return colored_resized


def init_offscreen_renderer(model, width=320, height=240):
    try:
        return mujoco.Renderer(model, height=height, width=width)
    except Exception as e:
        print(f"[SIM] ⚠️ 离屏渲染器初始化失败: {e}")
        return None


def render_camera_view(renderer, data, camera_name="top", width=640, height=480):
    """带防崩溃保护的三相机渲染函数"""
    try:
        renderer.update_scene(data, camera=camera_name)
        rgb = renderer.render()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        # 如果模型中缺少此相机，返回带提示的黑屏
        img = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(img, f"Camera Error:", (10, height // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(img, f"'{camera_name}'", (10, height // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return img


# ============================================================
# 7. 异步子线程 1: 真实舵机读取 (I/O 解耦)
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
                if res == COMM_SUCCESS:
                    arm_deg[i] = limit_real_deg(servo_id, (float(pos) / SERVO_DIGITAL_RANGE) * SERVO_ANGLE_RANGE)
            except:
                pass

        if not no_gripper:
            try:
                pos, _, res, _ = scs.ReadPosSpeed(GRIPPER_SERVO_ID)
                if res == COMM_SUCCESS:
                    gripper_deg = limit_real_deg(GRIPPER_SERVO_ID,
                                                 (float(pos) / SERVO_DIGITAL_RANGE) * SERVO_ANGLE_RANGE)
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
# 8. 异步子线程 2: 手势识别 CV 推理 (计算解耦)
# ============================================================
def hand_tracking_worker(hand_hz, g_min, g_max, state_lock, shared_state, stop_event):
    try:
        import HandTrackingModule as htm
    except ImportError as e:
        print(f"[CV-Worker] ❌ 导入失败: {e}")
        return

    cap = cv2.VideoCapture(HAND_CAMERA_ID)
    cap.set(3, HAND_CAMERA_WIDTH)
    cap.set(4, HAND_CAMERA_HEIGHT)

    if not cap.isOpened():
        print("[CV-Worker] ❌ 摄像头打开失败。")
        return

    print("[CV-Worker] ✅ 手势控制独立线程已启动。")
    detector = htm.handDetector(detectionCon=0.8)

    period = 1.0 / max(hand_hz, 1)
    fps_time = time.time()

    while not stop_event.is_set():
        t0 = time.perf_counter()
        success, img = cap.read()

        if success:
            img = detector.findHands(img, draw=True)
            lm_list = detector.findPosition(img, draw=False)

            cv_gripper_target = None
            if len(lm_list) != 0:
                x1, y1 = lm_list[4][1], lm_list[4][2]
                x2, y2 = lm_list[8][1], lm_list[8][2]
                xc, yc = (x1 + x2) // 2, (y1 + y2) // 2
                length = math.hypot(x2 - x1, y2 - y1)

                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (xc, yc), 8, (255, 0, 255), cv2.FILLED)
                cv2.putText(img, f"Dist: {int(length)}", (xc + 20, yc), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

                cv_gripper_target = np.interp(length, [30, 180], [g_min, g_max])
                if length < 30:
                    cv2.circle(img, (xc, yc), 12, (0, 255, 0), cv2.FILLED)

            c_time = time.time()
            fps = 1 / (c_time - fps_time) if (c_time - fps_time) > 0 else 0
            fps_time = c_time
            cv2.putText(img, f"FPS: {int(fps)}", (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

            with state_lock:
                shared_state["cv_gripper_target"] = cv_gripper_target
                shared_state["hand_view"] = img

        elapsed = time.perf_counter() - t0
        if elapsed < period:
            time.sleep(period - elapsed)

    cap.release()
    print("[CV-Worker] 线程安全退出。")


# ============================================================
# 9. 仿真主进程
# ============================================================
def simulation_process(args, frame_queue: mp.Queue, stop_event: mp.Event):
    print("[SIM] 加载模型与初始化...")
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

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

    portHandler = PortHandler(args.port)
    scs = sts(portHandler)
    if portHandler.openPort() and portHandler.setBaudRate(args.baudrate):
        print(f"[SIM] ✅ 串口通信已建立: {args.port}")
        time.sleep(2.0)
        if not args.keep_torque:
            for sid in ARM_SERVO_IDS + ([] if args.no_gripper else [GRIPPER_SERVO_ID]):
                scs.write1ByteTxRx(sid, STS_TORQUE_ENABLE, 0)
    else:
        print("[SIM] ❌ 串口打开失败！")
        stop_event.set()
        return

    state_lock = threading.Lock()
    shared_state = {
        "target_arm_q": np.zeros(ARM_DOF, dtype=np.float32),
        "target_gripper_cmd": float(gripper_ctrl_min),
        "cv_gripper_target": None,
        "hand_view": None
    }

    # 启动串口读取线程
    threading.Thread(
        target=servo_reader_worker,
        args=(
        scs, state_lock, shared_state, args.read_rate, args.no_gripper, gripper_ctrl_min, gripper_ctrl_max, stop_event),
        daemon=True
    ).start()

    # 启动手势识别线程
    if ENABLE_HAND_CAMERA:
        threading.Thread(
            target=hand_tracking_worker,
            args=(HAND_HZ, gripper_ctrl_min, gripper_ctrl_max, state_lock, shared_state, stop_event),
            daemon=True
        ).start()

    # ✅ 正确打开 MuJoCo 官方 3D Viewer
    viewer = None
    if ENABLE_MUJOCO_VIEWER:
        viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
        print("[SIM] ✅ MuJoCo 官方 3D 交互界面已开启。")

    # 离屏渲染器初始化
    offscreen = init_offscreen_renderer(model, CAMERA_WIDTH, CAMERA_HEIGHT) if ENABLE_SIM_CAMERAS else None

    sim_period = 1.0 / max(args.rate, 1e-6)
    steps_per_loop = max(1, int(round(sim_period / float(model.opt.timestep))))

    vis_interval = max(1, int(round(args.rate / args.vis_hz)))
    viewer_interval = max(1, int(round(args.rate / args.viewer_hz)))
    camera_interval = max(1, int(round(args.rate / args.camera_hz)))

    filtered_arm_q = np.zeros(ARM_DOF, dtype=np.float32)
    filtered_gripper_cmd = gripper_ctrl_min
    loop_count = 0

    print(f"[SIM] 仿真主循环无阻塞启动。频率:{args.rate}Hz，mj_step:{steps_per_loop}")
    try:
        while not stop_event.is_set():
            loop_start = time.perf_counter()

            if viewer is not None and not viewer.is_running():
                print("[SIM] 检测到 MuJoCo 窗口关闭，正在退出系统...")
                stop_event.set()
                break

            with state_lock:
                target_q = shared_state["target_arm_q"].copy()
                cv_target = shared_state["cv_gripper_target"]
                hw_target = shared_state["target_gripper_cmd"]
                current_hand_view = shared_state["hand_view"]

            target_g = cv_target if cv_target is not None else hw_target

            filtered_arm_q = smooth_update(filtered_arm_q, target_q, args.alpha_arm)
            filtered_gripper_cmd = smooth_update(filtered_gripper_cmd, target_g, args.alpha_gripper)

            for i, act_id in enumerate(arm_actuator_ids):
                if act_id >= 0: data.ctrl[act_id] = filtered_arm_q[i]
            if gripper_act_id >= 0: data.ctrl[gripper_act_id] = filtered_gripper_cmd

            for _ in range(steps_per_loop):
                mujoco.mj_step(model, data)

            if loop_count % vis_interval == 0:
                # 传入 FORCE_MAX 并在内部计算力向量范数，使其适配 RGB 渲染
                packet = {
                    "touch_right": read_tactile(right_reader, data, FORCE_MAX, SENSOR_ORDER),
                    "touch_left": read_tactile(left_reader, data, FORCE_MAX, SENSOR_ORDER),
                    "hand_view": current_hand_view
                }

                # ✅ 提取三个视角的相机画面
                if ENABLE_SIM_CAMERAS and offscreen and loop_count % camera_interval == 0:
                    packet["top_view"] = render_camera_view(offscreen, data, camera_name=TOP_CAMERA_NAME,
                                                            width=CAMERA_WIDTH, height=CAMERA_HEIGHT)
                    packet["wrist_view"] = render_camera_view(offscreen, data, camera_name=WRIST_CAMERA_NAME,
                                                              width=CAMERA_WIDTH, height=CAMERA_HEIGHT)
                    packet["angle_view"] = render_camera_view(offscreen, data, camera_name=ANGLE_CAMERA_NAME,
                                                              width=CAMERA_WIDTH, height=CAMERA_HEIGHT)

                put_latest(frame_queue, packet)

            if viewer is not None and loop_count % viewer_interval == 0:
                viewer.sync()

            loop_count += 1
            sleep_time = sim_period - (time.perf_counter() - loop_start)
            if sleep_time > 0: time.sleep(sleep_time)

    finally:
        if viewer is not None:
            viewer.close()
        portHandler.closePort()
        print("[SIM] 仿真进程完全停止。")


# ============================================================
# 10. 可视化独立进程 (并排显示大尺寸仿真相机)
# ============================================================
def visualization_process(frame_queue: mp.Queue, stop_event: mp.Event):
    print("[VIS] 可视化进程启动。")
    cv2.namedWindow("Touch Heatmap - Sensor Right", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Touch Heatmap - Sensor Left", cv2.WINDOW_NORMAL)

    if ENABLE_SIM_CAMERAS:
        cv2.namedWindow("Simulation Cameras (Top | Wrist | Angle)", cv2.WINDOW_NORMAL)

    if ENABLE_HAND_CAMERA:
        cv2.namedWindow("Hand Control View", cv2.WINDOW_NORMAL)

    last_sim_cameras = None
    last_hand = None

    while not stop_event.is_set():
        try:
            packet = get_latest(frame_queue, timeout=0.02)
        except queue.Empty:
            continue

        tr = packet.get("touch_right")
        tl = packet.get("touch_left")
        top = packet.get("top_view")
        wrist = packet.get("wrist_view")
        angle = packet.get("angle_view")
        hand = packet.get("hand_view")

        # 使用 RGB 热力图渲染触觉数据
        if tr is not None:
            cv2.imshow("Touch Heatmap - Sensor Right",
                       tactile_to_colormap(tr, FORCE_MAX, HEATMAP_RESIZE_W, HEATMAP_RESIZE_H, ROTATE_HEATMAP_CLOCKWISE))

        if tl is not None:
            cv2.imshow("Touch Heatmap - Sensor Left",
                       tactile_to_colormap(tl, FORCE_MAX, HEATMAP_RESIZE_W, HEATMAP_RESIZE_H, ROTATE_HEATMAP_CLOCKWISE))

        # ✅ 将 Top、Wrist、Angle 视角水平拼接
        if top is not None and wrist is not None and angle is not None:
            cv2.putText(top, "Top View", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(wrist, "Wrist View", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(angle, "Angle View", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # 水平拼接三个大图像 (640*3 = 1920 宽度)
            last_sim_cameras = cv2.hconcat([top, wrist, angle])

        if hand is not None: last_hand = hand

        if ENABLE_SIM_CAMERAS and last_sim_cameras is not None:
            cv2.imshow("Simulation Cameras (Top | Wrist | Angle)", last_sim_cameras)

        if ENABLE_HAND_CAMERA and last_hand is not None:
            cv2.imshow("Hand Control View", last_hand)

        if cv2.waitKey(1) in [27, ord('q')]:
            stop_event.set()
            break

    cv2.destroyAllWindows()
    print("[VIS] 界面已关闭。")


# ============================================================
# 11. Main 入口与命令行参数
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
