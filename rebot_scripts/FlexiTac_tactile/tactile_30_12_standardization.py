import numpy as np
import serial
import threading
import cv2
import time
from scipy.ndimage import gaussian_filter

# =========================
# Visualization settings
# =========================
# 硬件原始矩阵尺寸保持不变
ROWS, COLS = 16, 32

# 新增：实际要可视化的裁切尺寸 (倒数12行，中间30列)
VIS_ROWS, VIS_COLS = 12, 30

contact_data_norm = np.zeros((ROWS, COLS), dtype=np.float32)

# 窗口大小根据裁切后的尺寸进行调整
WINDOW_WIDTH = VIS_COLS * 30
WINDOW_HEIGHT = VIS_ROWS * 30

cv2.namedWindow("Contact Data_left", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Contact Data_left", WINDOW_WIDTH, WINDOW_HEIGHT)

# =========================
# Processing parameters
# =========================
THRESHOLD = 20  # 15
NOISE_SCALE = 60
flag = False

# =========================
# Serial binary format
# =========================
MAGIC = b"\xAA\x55"
FRAME_BYTES = ROWS * COLS
INIT_FRAMES = 30

# PORT = "COM12"
PORT = "/dev/ttyUSB2"  # 触觉 tactile 串口
BAUD = 2_000_000

# =========================
# Matrix printing settings
# =========================

# 是否打印矩阵
PRINT_MATRIX = True
# PRINT_MATRIX = False

# 打印类型：
# "raw"     : 打印 Arduino 原始矩阵，0~255，适合做 POINT_OFFSETS 标定
# "contact" : 打印扣除 median 和 THRESHOLD 后的接触矩阵
# "norm"    : 打印归一化后的矩阵，0~1，不适合直接复制到 Arduino
PRINT_MATRIX_TYPE = "raw"

# 每隔多少帧打印一次
# 1 表示每帧都打印，会非常卡
# 建议 10、20、30
PRINT_EVERY_N_FRAMES = 20

# 是否打印成 Arduino 数组格式
# True  : { 1, 2, 3, ... },
# False : numpy 普通矩阵格式
PRINT_AS_ARDUINO_ARRAY = True
# PRINT_AS_ARDUINO_ARRAY = False


def temporal_filter(new_frame, prev_frame, alpha=0.2):
    return alpha * new_frame + (1 - alpha) * prev_frame


def print_matrix_arduino_style(matrix, title="current_array"):
    """
    按 Arduino 数组格式打印 16×32 矩阵。
    可直接复制到：
    int POINT_OFFSETS[16][32] = { ... };
    """
    matrix_int = np.asarray(matrix).astype(np.int32)

    print()
    print(f"{title}:")
    for i in range(matrix_int.shape[0]):
        print("{", end=" ")
        print(*matrix_int[i], sep=", ", end=" },\n")
    print()


def print_matrix_numpy_style(matrix, title="current_array", as_float=False):
    """
    普通 numpy 格式打印矩阵。
    """
    print()
    print("=" * 100)
    print(title)
    print("shape:", matrix.shape, "max:", np.max(matrix), "min:", np.min(matrix))
    print("=" * 100)

    if as_float:
        print(
            np.array2string(
                matrix,
                precision=2,
                suppress_small=True,
                max_line_width=220
            )
        )
    else:
        print(
            np.array2string(
                matrix.astype(np.int32),
                max_line_width=220
            )
        )

    print()


def print_tactile_matrix(matrix, title="current_array", as_float=False):
    """
    根据设置选择打印格式。
    """
    if PRINT_AS_ARDUINO_ARRAY and not as_float:
        print_matrix_arduino_style(matrix, title=title)
    else:
        print_matrix_numpy_style(matrix, title=title, as_float=as_float)


def readThread(serDev):
    """
    Reads frames:
        0xAA 0x55 + 16×32 bytes

    Then:
        1. collect INIT_FRAMES for median baseline
        2. stream frames
        3. update contact_data_norm
        4. print tactile matrix in Arduino-style format
    """
    global contact_data_norm, flag

    ring = bytearray()
    frame_buf = bytearray(FRAME_BYTES)

    data_tac = []
    flag = False
    t1 = time.time()

    frame_count = 0

    serDev.timeout = 0.01

    def read_exact(n):
        """
        Read exactly n bytes from serial.
        Return None if timeout happens before enough bytes are read.
        """
        buf = bytearray(n)
        mv = memoryview(buf)
        got = 0

        while got < n:
            r = serDev.readinto(mv[got:])
            if r is None:
                r = 0

            if r == 0:
                return None

            got += r

        return buf

    # =====================================================
    # 1. Initialization
    # =====================================================
    while True:
        chunk = serDev.read(4096)

        if not chunk:
            continue

        ring.extend(chunk)

        if len(ring) > 50000:
            ring = ring[-50000:]

        idx = ring.find(MAGIC)

        if idx < 0:
            if len(ring) > 1:
                ring = ring[-1:]
            continue

        if idx > 0:
            del ring[:idx]

        if len(ring) < 2:
            continue

        del ring[:2]

        if len(ring) >= FRAME_BYTES:
            frame_buf[:] = ring[:FRAME_BYTES]
            del ring[:FRAME_BYTES]
        else:
            have = len(ring)
            frame_buf[:have] = ring[:have]
            del ring[:have]

            rest = read_exact(FRAME_BYTES - have)

            if rest is None:
                ring.clear()
                continue

            frame_buf[have:] = rest

        frame = np.frombuffer(frame_buf, dtype=np.uint8).reshape((ROWS, COLS)).astype(np.float32)
        data_tac.append(frame)

        now = time.time()
        print("init fps", 1.0 / (now - t1 + 1e-9))
        t1 = now

        if len(data_tac) >= INIT_FRAMES:
            break

    data_tac = np.stack(data_tac, axis=0)
    median = np.median(data_tac, axis=0)

    flag = True
    print("初始化完成！")
    print("median baseline shape:", median.shape)

    # =====================================================
    # 2. Streaming
    # =====================================================
    while True:
        chunk = serDev.read(8192)

        if not chunk:
            continue

        ring.extend(chunk)

        if len(ring) > 50000:
            ring = ring[-50000:]

        while True:
            idx = ring.find(MAGIC)

            if idx < 0:
                if len(ring) > 1:
                    ring = ring[-1:]
                break

            if idx > 0:
                del ring[:idx]

            if len(ring) < 2 + FRAME_BYTES:
                break

            del ring[:2]

            frame_bytes = ring[:FRAME_BYTES]
            del ring[:FRAME_BYTES]

            # 原始触觉矩阵，16×32，0~255
            backup = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((ROWS, COLS)).astype(np.float32)

            # 接触矩阵：去 baseline + 阈值
            contact_data = backup - median - THRESHOLD
            contact_data = np.clip(contact_data, 0, 100)

            # 归一化矩阵：用于热力图显示
            if np.max(contact_data) < THRESHOLD:
                contact_data_norm = contact_data / NOISE_SCALE
            else:
                contact_data_norm = contact_data / (np.max(contact_data) + 1e-6)

            frame_count += 1

            # =====================================================
            # 3. Real-time matrix printing
            # =====================================================
            if PRINT_MATRIX and frame_count % PRINT_EVERY_N_FRAMES == 0:
                if PRINT_MATRIX_TYPE == "raw":
                    print_tactile_matrix(backup, title="current_array_raw_16x32", as_float=False)
                elif PRINT_MATRIX_TYPE == "contact":
                    print_tactile_matrix(contact_data, title="current_array_contact_16x32", as_float=False)
                elif PRINT_MATRIX_TYPE == "norm":
                    print_tactile_matrix(contact_data_norm, title="current_array_norm_16x32", as_float=True)
                else:
                    print("Unknown PRINT_MATRIX_TYPE:", PRINT_MATRIX_TYPE)


# =========================
# Start serial + thread
# =========================
serDev = serial.Serial(PORT, BAUD)
serDev.flush()
serDev.reset_input_buffer()

serialThread = threading.Thread(target=readThread, args=(serDev,))
serialThread.daemon = True
serialThread.start()


# =========================
# Optional Gaussian blur
# =========================
def apply_gaussian_blur(contact_map, sigma=0.1):
    return gaussian_filter(contact_map, sigma=sigma)


# =========================
# Main visualization loop
# =========================

# 修改：基于裁切后的新尺寸 (12, 30) 初始化上一帧缓存
prev_frame = np.zeros((VIS_ROWS, VIS_COLS), dtype=np.float32)

if __name__ == "__main__":
    print("开始接收数据测试")

    while True:
        if flag:
            # 核心修改：利用 numpy 切片提取 12 行，30 列
            # [-VIS_ROWS:] 表示取最后 12 行；[1:-1] 表示取中间 30 列; 中间 30 列 = 去掉第 1 列和最后 1 列，即原始列索引 1~30
            # contact_data_cropped = contact_data_norm[-VIS_ROWS:, -VIS_COLS:]
            contact_data_cropped = contact_data_norm[-VIS_ROWS:, 1:-1]

            # alpha 调高，减少残影延迟，并且现在的平滑滤波只在 12x32 的矩阵上运行，速度更快
            temp_filtered_data = temporal_filter(contact_data_cropped, prev_frame, alpha=1)
            prev_frame = temp_filtered_data

            temp_filtered_data_scaled = np.clip(temp_filtered_data * 255.0, 0, 255).astype(np.uint8)
            colormap = cv2.applyColorMap(temp_filtered_data_scaled, cv2.COLORMAP_VIRIDIS)

            cv2.imshow("Contact Data_left", colormap)

            # 仅靠 waitKey 控制渲染节奏，并允许按 ESC 键优雅退出
            if cv2.waitKey(1) & 0xFF == 27:
                break
