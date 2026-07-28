import numpy as np
import serial
import threading
import cv2
import time

# =====================================================
# 极速低延迟触觉可视化版本 (终极优化版)
# Arduino 格式：
# AA 55 + 16×32 bytes
# =====================================================

# =========================
# Array settings
# =========================
# 硬件原始矩阵尺寸保持不变，用于串口解析
ROWS, COLS = 16, 32
FRAME_BYTES = ROWS * COLS
MAGIC = b"\xAA\x55"

# 实际要可视化的裁切尺寸 (倒数12行，倒数32列)
VIS_ROWS, VIS_COLS = 12, 32

# =========================
# Serial settings
# =========================
# PORT = "COM12"
PORT = "/dev/ttyUSB2"  # 触觉 tactile 串口
BAUD = 2_000_000

# =========================
# Processing settings
# =========================
THRESHOLD = 20  # 15
NOISE_SCALE = 60

# 初始化帧数，用于计算 baseline (由于已经修复了初始化速度，设回 30 也没问题，这里保持 10)
INIT_FRAMES = 10

# 是否使用固定尺度归一化
# True  : 延迟更稳定，显示强度有绝对意义
# False : 每帧按最大值归一化，对比度更强，但显示强弱不绝对
USE_FIXED_SCALE = False

# 固定归一化尺度，只有 USE_FIXED_SCALE=True 时生效
FIXED_SCALE = 100.0

# 是否显示完整 16×32
# 设为 False 即启用下方的 12x30 裁剪逻辑
SHOW_FULL_MATRIX = False

# 主循环空闲时是否 sleep
# 极速模式建议 0.0
# 如果 CPU 占用太高，可以改成 0.0005 或 0.001
IDLE_SLEEP = 0.0

# =========================
# Global shared variables
# =========================
contact_data_norm = np.zeros((ROWS, COLS), dtype=np.float32)
flag = False
latest_frame_id = 0
running = True

# =========================
# OpenCV window
# =========================
# 窗口大小根据裁切后的尺寸 (12x32) 进行动态调整
if SHOW_FULL_MATRIX:
    WINDOW_WIDTH = COLS * 30
    WINDOW_HEIGHT = ROWS * 30
else:
    WINDOW_WIDTH = VIS_COLS * 30
    WINDOW_HEIGHT = VIS_ROWS * 30

cv2.namedWindow("Contact Data_left", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Contact Data_left", WINDOW_WIDTH, WINDOW_HEIGHT)


def extract_next_frame(ring):
    """
    专门用于初始化阶段：按顺序提取下一帧，绝不丢弃任何积压数据。
    """
    idx = ring.find(MAGIC)
    if idx < 0:
        if len(ring) > 1:
            return None, bytearray(ring[-1:])
        return None, ring

    end = idx + 2 + FRAME_BYTES
    if len(ring) >= end:
        frame_bytes = bytes(ring[idx + 2:end])
        new_ring = bytearray(ring[end:])
        return frame_bytes, new_ring

    return None, bytearray(ring[idx:])


def extract_latest_complete_frame(ring):
    """
    专门用于实时阶段：从 ring 缓冲区中提取最新的一帧完整数据。
    低延迟策略：如果缓冲区里有多帧，只取最后一帧完整帧，前面的旧帧全部丢弃。
    """
    positions = []
    start = 0

    while True:
        idx = ring.find(MAGIC, start)
        if idx < 0:
            break
        positions.append(idx)
        start = idx + 2

    if not positions:
        # 保留最后 1 个字节，防止帧头 AA55 被拆开
        if len(ring) > 1:
            return None, bytearray(ring[-1:])
        return None, ring

    latest_frame = None
    latest_end = None

    # 从后往前找最新的完整帧
    for idx in reversed(positions):
        end = idx + 2 + FRAME_BYTES
        if len(ring) >= end:
            latest_frame = bytes(ring[idx + 2:end])
            latest_end = end
            break

    if latest_frame is None:
        # 找到了帧头，但后面的 512 字节还没收完整
        # 保留最后一个帧头之后的数据
        last_idx = positions[-1]
        return None, bytearray(ring[last_idx:])

    # 丢弃最新完整帧之前的所有旧数据
    new_ring = bytearray(ring[latest_end:])

    # 防止残余 buffer 过大
    if len(new_ring) > 50000:
        new_ring = new_ring[-50000:]

    return latest_frame, new_ring


def readThread(serDev):
    """
    串口读取线程：
    1. 初始化阶段使用 extract_next_frame 快速收集帧计算 median baseline
    2. 实时阶段使用 extract_latest_complete_frame 丢弃旧帧保证零延迟
    """
    global contact_data_norm, flag, latest_frame_id, running

    ring = bytearray()
    data_tac = []

    serDev.timeout = 0.001

    # =====================================================
    # 1. Initialization (闪电初始化)
    # =====================================================
    print("开始初始化 baseline，请保持触觉阵列无接触...")

    while running and len(data_tac) < INIT_FRAMES:
        chunk = serDev.read(65536)

        if not chunk:
            continue

        ring.extend(chunk)

        if len(ring) > 50000:
            ring = ring[-50000:]

        # 在初始化阶段，循环提取缓冲区里的每一帧，瞬间凑够 INIT_FRAMES
        while running and len(data_tac) < INIT_FRAMES:
            frame_bytes, new_ring = extract_next_frame(ring)
            if frame_bytes is None:
                break  # 缓冲区里的完整帧被取完了，跳出内层循环继续读串口

            ring = new_ring
            frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((ROWS, COLS)).astype(np.float32)
            data_tac.append(frame)

            print(f"init frame {len(data_tac)}/{INIT_FRAMES}")

    if len(data_tac) == 0:
        print("初始化失败：没有收到有效帧。")
        return

    data_tac = np.stack(data_tac, axis=0)
    median = np.median(data_tac, axis=0).astype(np.float32)

    flag = True
    print("初始化完成！开始极速实时显示。")

    # =====================================================
    # 2. Streaming low-latency loop (极速实时流)
    # =====================================================
    while running:
        # 尽可能一次性读取当前串口缓冲区中的全部数据
        waiting = serDev.in_waiting
        if waiting > 0:
            chunk = serDev.read(waiting)
        else:
            chunk = serDev.read(4096)

        if not chunk:
            continue

        ring.extend(chunk)

        if len(ring) > 50000:
            ring = ring[-50000:]

        frame_bytes, ring = extract_latest_complete_frame(ring)

        if frame_bytes is None:
            continue

        # 原始帧：16×32, uint8, 0~255
        raw_frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((ROWS, COLS))

        # 转 float32 并做 baseline + threshold
        contact_data = raw_frame.astype(np.float32) - median - THRESHOLD
        np.clip(contact_data, 0, 100, out=contact_data)

        max_val = float(np.max(contact_data))

        if USE_FIXED_SCALE:
            norm = contact_data / FIXED_SCALE
            np.clip(norm, 0.0, 1.0, out=norm)
        else:
            if max_val < THRESHOLD:
                norm = contact_data / NOISE_SCALE
            else:
                norm = contact_data / (max_val + 1e-6)

        # 直接替换全局引用，避免 in-place 写入导致显示线程读到半帧
        contact_data_norm = norm.astype(np.float32, copy=False)

        latest_frame_id += 1


# =====================================================
# Start serial
# =====================================================
serDev = serial.Serial(PORT, BAUD)
serDev.flush()
serDev.reset_input_buffer()

# 尽量扩大串口缓冲区，部分平台支持，部分平台不支持
try:
    serDev.set_buffer_size(rx_size=262144, tx_size=262144)
except Exception:
    pass

serialThread = threading.Thread(target=readThread, args=(serDev,))
serialThread.daemon = True
serialThread.start()

# =====================================================
# Main visualization loop
# =====================================================
if __name__ == "__main__":
    print("开始接收数据测试，按 ESC 退出。")

    last_displayed_frame_id = -1

    try:
        while True:
            if flag and latest_frame_id != last_displayed_frame_id:
                last_displayed_frame_id = latest_frame_id

                # 极速模式：不做 temporal_filter
                frame_to_show = contact_data_norm

                # 转成 OpenCV 图像
                img = np.clip(frame_to_show * 255.0, 0, 255).astype(np.uint8)

                if SHOW_FULL_MATRIX:
                    img_display = img
                else:
                    # 核心修改：利用 numpy 切片提取倒数 12 行，倒数 32 列
                    # [-VIS_ROWS:] 取最后12行；[-VIS_COLS:] 取最后32列
                    img_display = img[-VIS_ROWS:, -VIS_COLS:]

                colormap = cv2.applyColorMap(img_display, cv2.COLORMAP_VIRIDIS)
                cv2.imshow("Contact Data_left", colormap)

            key = cv2.waitKey(1)

            if key == 27:  # ESC
                break

            if IDLE_SLEEP > 0:
                time.sleep(IDLE_SLEEP)

    except KeyboardInterrupt:
        pass

    finally:
        running = False
        try:
            serDev.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
