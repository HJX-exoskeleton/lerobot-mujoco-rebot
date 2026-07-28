import numpy as np
import serial
import threading
import cv2
import time

# =====================================================
# 高级热力云图 + 低延迟触觉可视化版本
# Arduino 数据格式：
# AA 55 + 16×32 bytes
#
# 显示逻辑：
# 16×32 原始矩阵
# -> baseline 扣除
# -> 阈值过滤
# -> 裁切倒数12行、中间30列
# -> 归一化
# -> 自适应时间平滑
# -> 高分辨率插值
# -> 软阈值去背景
# -> 双层高斯模糊
# -> bloom 光晕
# -> gamma 增强
# -> 黑背景融合
# -> 暗角增强
# -> 高级热力图显示
# =====================================================

# 尽量开启 OpenCV 优化
cv2.setUseOptimized(True)

# =========================
# Hardware array settings
# =========================
ROWS, COLS = 16, 32
FRAME_BYTES = ROWS * COLS
MAGIC = b"\xAA\x55"

# =========================
# Visualization crop settings
# =========================
# 保持你的原逻辑：倒数12行，中间30列
VIS_ROWS, VIS_COLS = 12, 30

ROW_SLICE = slice(-VIS_ROWS, None)   # 最后12行
COL_SLICE = slice(1, -1)             # 中间30列，去掉左右边缘列

# =========================
# Serial settings
# =========================
# PORT = "COM3"
PORT = "/dev/ttyUSB2"  # 触觉 tactile 串口
BAUD = 2_000_000

# =========================
# Signal processing settings
# =========================
THRESHOLD = 20
NOISE_SCALE = 60

# baseline 初始化帧数
# 30 稳定；10 启动更快
INIT_FRAMES = 30

# 是否使用固定尺度归一化
# False：每帧动态归一化，视觉对比更强
# True ：固定归一化，颜色强度更有绝对意义
USE_FIXED_SCALE = False
FIXED_SCALE = 100.0

# =========================
# Advanced rendering settings
# =========================
# 每个原始触觉点放大多少像素
# 28 比较细腻；如果电脑卡，可以改 22 或 20
DISPLAY_SCALE = 28

WINDOW_WIDTH = VIS_COLS * DISPLAY_SCALE
WINDOW_HEIGHT = VIS_ROWS * DISPLAY_SCALE

# 是否开启高级渲染
ENABLE_PREMIUM_RENDER = True

# 插值方式
# INTER_CUBIC 更高级更平滑，INTER_LINEAR 更快
UPSAMPLE_INTERP = cv2.INTER_CUBIC

# 自适应时间平滑
ENABLE_ADAPTIVE_TEMPORAL_SMOOTH = True

# 快速变化时使用这个 alpha，更跟手
TEMPORAL_ALPHA_FAST = 0.78

# 慢速变化时使用这个 alpha，更稳定
TEMPORAL_ALPHA_SLOW = 0.45

# 判断是否快速变化的阈值
MOTION_THRESHOLD = 0.055

# 低值软裁剪，让背景干净，但不要硬切得太突兀
LOW_CUT = 0.025

# 软裁剪后整体增益
RENDER_GAIN = 1.12

# 小范围模糊：消除格子边缘
BLUR_SIGMA = 1.65

# 大范围模糊：制造 bloom 光晕
GLOW_SIGMA = 6.5
GLOW_STRENGTH = 0.58

# 二级远光晕，增强高级感
ENABLE_SECOND_GLOW = True
SECOND_GLOW_SIGMA = 14.0
SECOND_GLOW_STRENGTH = 0.18

# gamma < 1：提亮中低强度区域
# gamma > 1：压暗中低强度区域
GAMMA = 0.82

# 黑背景融合强度
# 越大背景越黑，热点越突出
DARK_BLEND_POWER = 0.72

# 是否开启暗角
ENABLE_VIGNETTE = True
VIGNETTE_STRENGTH = 0.22

# 颜色风格
# TURBO：更鲜艳高级
# VIRIDIS：更克制科研
# INFERNO：黑红黄风格
COLORMAP_STYLE = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_VIRIDIS)
# COLORMAP_STYLE = cv2.COLORMAP_VIRIDIS
# COLORMAP_STYLE = cv2.COLORMAP_INFERNO
# COLORMAP_STYLE = cv2.COLORMAP_JET

# 是否显示 FPS
SHOW_FPS = False

# 是否显示网格
# 高级云图一般建议 False
SHOW_GRID = False
GRID_COLOR = (35, 35, 35)

# =========================
# Performance settings
# =========================
IDLE_SLEEP = 0.0

# 实时打印矩阵会明显降低刷新速度，默认关闭
PRINT_MATRIX = False
PRINT_MATRIX_TYPE = "raw"      # "raw" / "contact" / "norm"
PRINT_EVERY_N_FRAMES = 100
PRINT_AS_ARDUINO_ARRAY = True

# =========================
# Global shared variables
# =========================
latest_vis_norm = np.zeros((VIS_ROWS, VIS_COLS), dtype=np.float32)

flag = False
running = True
latest_frame_id = 0

# 显示平滑缓存
display_prev = np.zeros((VIS_ROWS, VIS_COLS), dtype=np.float32)

# FPS 统计
fps_last_time = time.time()
fps_counter = 0
fps_value = 0.0


# =========================
# Precompute rendering LUT and masks
# =========================
def build_gamma_lut(gamma):
    lut = np.arange(256, dtype=np.float32) / 255.0
    lut = np.power(lut, gamma)
    lut = np.clip(lut * 255.0, 0, 255).astype(np.uint8)
    return lut


GAMMA_LUT = build_gamma_lut(GAMMA)


def build_vignette_mask(width, height, strength=0.22):
    """
    生成暗角 mask，中心亮，边缘略暗。
    """
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    radius = np.sqrt(xx * xx + yy * yy)
    radius = np.clip(radius, 0.0, 1.0)

    mask = 1.0 - strength * np.power(radius, 1.7)
    mask = np.clip(mask, 0.0, 1.0)

    return mask.astype(np.float32)


VIGNETTE_MASK = build_vignette_mask(
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    VIGNETTE_STRENGTH
)


# =========================
# Printing helpers
# =========================
def print_matrix_arduino_style(matrix, title="current_array"):
    matrix_int = np.asarray(matrix).astype(np.int32)

    print()
    print(f"{title}:")
    for i in range(matrix_int.shape[0]):
        print("{", end=" ")
        print(*matrix_int[i], sep=", ", end=" },\n")
    print()


def print_matrix_numpy_style(matrix, title="current_array", as_float=False):
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
    if PRINT_AS_ARDUINO_ARRAY and not as_float:
        print_matrix_arduino_style(matrix, title=title)
    else:
        print_matrix_numpy_style(matrix, title=title, as_float=as_float)


# =========================
# Serial frame extraction
# =========================
def extract_next_frame(ring):
    """
    初始化阶段使用：
    按顺序提取下一帧，不主动丢弃旧帧。
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
    实时阶段使用：
    只提取缓冲区里最新的一帧完整数据。
    如果缓冲区里积压多帧，旧帧全部丢弃。
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
        if len(ring) > 1:
            return None, bytearray(ring[-1:])
        return None, ring

    latest_frame = None
    latest_end = None

    for idx in reversed(positions):
        end = idx + 2 + FRAME_BYTES
        if len(ring) >= end:
            latest_frame = bytes(ring[idx + 2:end])
            latest_end = end
            break

    if latest_frame is None:
        last_idx = positions[-1]
        return None, bytearray(ring[last_idx:])

    new_ring = bytearray(ring[latest_end:])

    if len(new_ring) > 50000:
        new_ring = new_ring[-50000:]

    return latest_frame, new_ring


# =========================
# Signal processing
# =========================
def normalize_contact_map(contact_crop):
    """
    将裁切后的接触矩阵归一化到 0~1。
    """
    contact_crop = contact_crop.astype(np.float32, copy=False)

    if USE_FIXED_SCALE:
        norm = contact_crop / FIXED_SCALE
        np.clip(norm, 0.0, 1.0, out=norm)
        return norm.astype(np.float32, copy=False)

    max_val = float(np.max(contact_crop))

    if max_val < THRESHOLD:
        norm = contact_crop / NOISE_SCALE
    else:
        norm = contact_crop / (max_val + 1e-6)

    np.clip(norm, 0.0, 1.0, out=norm)
    return norm.astype(np.float32, copy=False)


def adaptive_temporal_filter(new_frame, prev_frame):
    """
    自适应时间平滑：
    - 变化大：alpha 大，响应快
    - 变化小：alpha 小，画面稳
    """
    delta = float(np.mean(np.abs(new_frame - prev_frame)))

    if delta > MOTION_THRESHOLD:
        alpha = TEMPORAL_ALPHA_FAST
    else:
        alpha = TEMPORAL_ALPHA_SLOW

    out = alpha * new_frame + (1.0 - alpha) * prev_frame
    return out.astype(np.float32, copy=False)


# =========================
# Premium rendering
# =========================
def soft_low_cut(frame):
    """
    平滑低值裁剪。
    相比直接 base[base < LOW_CUT] = 0，这种方式过渡更自然。
    """
    out = (frame - LOW_CUT) / max(1e-6, 1.0 - LOW_CUT)
    np.clip(out, 0.0, 1.0, out=out)
    return out.astype(np.float32, copy=False)


def render_premium_heatmap(frame_norm):
    """
    输入：
        frame_norm: 12×30, float32, 0~1

    输出：
        高级热力云图 BGR 图像
    """
    # 1. 复制，避免修改共享数据
    base = frame_norm.astype(np.float32, copy=True)

    # 2. 平滑低值裁剪，让背景干净
    base = soft_low_cut(base)

    # 3. 轻微增益
    base *= RENDER_GAIN
    np.clip(base, 0.0, 1.0, out=base)

    # 4. 高分辨率插值放大
    up = cv2.resize(
        base,
        (WINDOW_WIDTH, WINDOW_HEIGHT),
        interpolation=UPSAMPLE_INTERP
    )
    up = np.clip(up, 0.0, 1.0).astype(np.float32, copy=False)

    if ENABLE_PREMIUM_RENDER:
        # 5. 主体小范围模糊，削弱块状边缘
        smooth = cv2.GaussianBlur(up, (0, 0), BLUR_SIGMA)

        # 6. 第一层 glow
        glow = cv2.GaussianBlur(smooth, (0, 0), GLOW_SIGMA)

        mixed = smooth + GLOW_STRENGTH * glow

        # 7. 第二层远距离 glow，更有高级柔光感
        if ENABLE_SECOND_GLOW:
            second_glow = cv2.GaussianBlur(smooth, (0, 0), SECOND_GLOW_SIGMA)
            mixed = mixed + SECOND_GLOW_STRENGTH * second_glow

        np.clip(mixed, 0.0, 1.0, out=mixed)
    else:
        mixed = up

    # 8. 转灰度 0~255
    gray = np.clip(mixed * 255.0, 0, 255).astype(np.uint8)

    # 9. gamma LUT，速度比 np.power 更快
    gray = cv2.LUT(gray, GAMMA_LUT)

    # 10. 伪彩色
    color = cv2.applyColorMap(gray, COLORMAP_STYLE)

    # 11. 黑背景融合，让低值区域真正变暗
    intensity = gray.astype(np.float32) / 255.0
    alpha = np.power(intensity, DARK_BLEND_POWER)
    alpha = np.clip(alpha, 0.0, 1.0)

    color_float = color.astype(np.float32)
    color_float *= alpha[..., None]

    # 12. 暗角，让画面更有层次
    if ENABLE_VIGNETTE:
        color_float *= VIGNETTE_MASK[..., None]

    color_out = np.clip(color_float, 0, 255).astype(np.uint8)

    # 13. 可选网格
    if SHOW_GRID:
        for x in range(0, WINDOW_WIDTH, DISPLAY_SCALE):
            cv2.line(color_out, (x, 0), (x, WINDOW_HEIGHT), GRID_COLOR, 1)

        for y in range(0, WINDOW_HEIGHT, DISPLAY_SCALE):
            cv2.line(color_out, (0, y), (WINDOW_WIDTH, y), GRID_COLOR, 1)

    return color_out


# =========================
# Serial reading thread
# =========================
def readThread(serDev):
    """
    串口读取线程：
    1. 初始化阶段：连续采集 INIT_FRAMES 帧，计算 median baseline
    2. 实时阶段：只取最新完整帧，旧帧丢弃
    3. 完成 baseline、threshold、裁切、归一化
    """
    global latest_vis_norm, flag, running, latest_frame_id

    ring = bytearray()
    data_tac = []
    frame_count = 0

    serDev.timeout = 0.001

    print("开始初始化 baseline，请保持触觉阵列无接触...")

    # =====================================================
    # 1. Initialization
    # =====================================================
    while running and len(data_tac) < INIT_FRAMES:
        chunk = serDev.read(65536)

        if not chunk:
            continue

        ring.extend(chunk)

        if len(ring) > 50000:
            ring = ring[-50000:]

        while running and len(data_tac) < INIT_FRAMES:
            frame_bytes, ring_new = extract_next_frame(ring)

            if frame_bytes is None:
                break

            ring = ring_new

            frame = np.frombuffer(
                frame_bytes,
                dtype=np.uint8
            ).reshape((ROWS, COLS)).astype(np.float32)

            data_tac.append(frame)
            print(f"init frame {len(data_tac)}/{INIT_FRAMES}")

    if len(data_tac) == 0:
        print("初始化失败：没有收到有效帧。")
        return

    data_tac = np.stack(data_tac, axis=0)
    median = np.median(data_tac, axis=0).astype(np.float32)

    flag = True
    print("初始化完成！开始高级低延迟实时显示。")
    print("median baseline shape:", median.shape)

    # =====================================================
    # 2. Streaming
    # =====================================================
    while running:
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

        raw_frame = np.frombuffer(
            frame_bytes,
            dtype=np.uint8
        ).reshape((ROWS, COLS)).astype(np.float32)

        # baseline + threshold
        contact_data = raw_frame - median - THRESHOLD
        np.clip(contact_data, 0, 100, out=contact_data)

        # 裁切：倒数12行，中间30列
        contact_crop = contact_data[ROW_SLICE, COL_SLICE]

        # 归一化
        norm_crop = normalize_contact_map(contact_crop)

        # 替换全局引用，避免主线程读到半帧
        latest_vis_norm = norm_crop.astype(np.float32, copy=False)

        latest_frame_id += 1
        frame_count += 1

        if PRINT_MATRIX and frame_count % PRINT_EVERY_N_FRAMES == 0:
            if PRINT_MATRIX_TYPE == "raw":
                print_tactile_matrix(
                    raw_frame,
                    title="current_array_raw_16x32",
                    as_float=False
                )
            elif PRINT_MATRIX_TYPE == "contact":
                print_tactile_matrix(
                    contact_data,
                    title="current_array_contact_16x32",
                    as_float=False
                )
            elif PRINT_MATRIX_TYPE == "norm":
                print_tactile_matrix(
                    latest_vis_norm,
                    title="current_array_norm_12x30",
                    as_float=True
                )
            else:
                print("Unknown PRINT_MATRIX_TYPE:", PRINT_MATRIX_TYPE)


# =====================================================
# Start serial + thread
# =====================================================
serDev = serial.Serial(PORT, BAUD)
serDev.flush()
serDev.reset_input_buffer()

try:
    serDev.set_buffer_size(rx_size=262144, tx_size=262144)
except Exception:
    pass

serialThread = threading.Thread(target=readThread, args=(serDev,))
serialThread.daemon = True
serialThread.start()


# =====================================================
# OpenCV window
# =====================================================
cv2.namedWindow("Contact Data_left", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Contact Data_left", WINDOW_WIDTH, WINDOW_HEIGHT)


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

                frame_to_show = latest_vis_norm.copy()

                # 自适应时间平滑
                if ENABLE_ADAPTIVE_TEMPORAL_SMOOTH:
                    frame_to_show = adaptive_temporal_filter(
                        frame_to_show,
                        display_prev
                    )
                    display_prev = frame_to_show.copy()
                else:
                    display_prev = frame_to_show.copy()

                # 高级渲染
                rendered = render_premium_heatmap(frame_to_show)

                # FPS 显示
                if SHOW_FPS:
                    now = time.time()
                    fps_counter += 1

                    if now - fps_last_time >= 0.5:
                        fps_value = fps_counter / (now - fps_last_time)
                        fps_counter = 0
                        fps_last_time = now

                    cv2.putText(
                        rendered,
                        f"FPS: {fps_value:.1f}",
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (230, 230, 230),
                        2,
                        cv2.LINE_AA
                    )

                cv2.imshow("Contact Data_left", rendered)

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
