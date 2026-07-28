import numpy as np
import serial
import threading
import cv2
import time

# =====================================================
# 专业可调视角 3D 触觉曲面可视化 + 低延迟串口刷新版本
#
# Arduino 数据格式：
# AA 55 + 16×32 bytes
#
# 核心功能：
# 1. 保留原始 16×32 触觉矩阵读取
# 2. baseline 扣除 + 阈值过滤
# 3. 裁切倒数12行、中间30列
# 4. 压力值作为 Z 轴高度
# 5. 正交 3D 相机投影
# 6. 支持键盘实时旋转、缩放、高度调节
# 7. 白色背景 + 专业曲面 + 光照 + 阴影 + 坐标轴
# =====================================================

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
VIS_ROWS, VIS_COLS = 12, 30

ROW_SLICE = slice(-VIS_ROWS, None)
COL_SLICE = slice(1, -1)

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
INIT_FRAMES = 30

USE_FIXED_SCALE = False
FIXED_SCALE = 100.0

# =========================
# Canvas settings
# =========================
CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 820

WINDOW_WIDTH = CANVAS_WIDTH
WINDOW_HEIGHT = CANVAS_HEIGHT

BACKGROUND_COLOR = (255, 255, 255)

# =========================
# Mesh settings
# =========================
# 2 更快，3 更平滑，4 更细腻但更慢
MESH_UPSAMPLE = 3

MESH_ROWS = VIS_ROWS * MESH_UPSAMPLE
MESH_COLS = VIS_COLS * MESH_UPSAMPLE

SURFACE_X_SIZE = 3.2
SURFACE_Y_SIZE = 1.35

UPSAMPLE_INTERP = cv2.INTER_CUBIC

# =========================
# 3D camera settings
# =========================
DEFAULT_YAW_DEG = -45.0
DEFAULT_ELEV_DEG = 34.0
DEFAULT_Z_SCALE = 0.85
DEFAULT_CAMERA_ZOOM = 250.0

view_yaw_deg = DEFAULT_YAW_DEG
view_elev_deg = DEFAULT_ELEV_DEG
view_z_scale = DEFAULT_Z_SCALE
camera_zoom = DEFAULT_CAMERA_ZOOM

CAMERA_CENTER_X = CANVAS_WIDTH // 2
CAMERA_CENTER_Y = CANVAS_HEIGHT // 2 + 70

# 视角限制
MIN_ELEV_DEG = 12.0
MAX_ELEV_DEG = 75.0

MIN_Z_SCALE = 0.25
MAX_Z_SCALE = 2.20

MIN_CAMERA_ZOOM = 150.0
MAX_CAMERA_ZOOM = 420.0

# =========================
# Height processing
# =========================
LOW_CUT = 0.045
EMPTY_FRAME_MAX = 0.010

HEIGHT_GAIN = 1.45
HEIGHT_GAMMA = 0.82
HEIGHT_BLUR_SIGMA = 1.15

# 低于这个值的区域不绘制曲面
SURFACE_ALPHA_CUT = 0.045

# 越大，弱信号越透明
WHITE_BLEND_POWER = 0.82

# =========================
# Lighting settings
# =========================
NORMAL_STRENGTH = 5.5

AMBIENT_LIGHT = 0.42
DIFFUSE_STRENGTH = 0.76
SPECULAR_STRENGTH = 0.30
SPECULAR_POWER = 30.0

# 世界坐标光源方向
LIGHT_DIR_WORLD = np.array([-0.35, -0.65, 1.00], dtype=np.float32)

# =========================
# Shadow settings
# =========================
ENABLE_DROP_SHADOW = True
SHADOW_STRENGTH = 0.22
SHADOW_BLUR_SIGMA = 13.0
SHADOW_OFFSET_X = 18
SHADOW_OFFSET_Y = 24

# =========================
# Surface style
# =========================
COLORMAP_STYLE = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_VIRIDIS)
# COLORMAP_STYLE = cv2.COLORMAP_VIRIDIS
# COLORMAP_STYLE = cv2.COLORMAP_INFERNO
# COLORMAP_STYLE = cv2.COLORMAP_JET

GAMMA = 0.82

DRAW_BASE_GRID = True
BASE_GRID_STEP = 6
BASE_GRID_COLOR = (225, 225, 225)

DRAW_SURFACE_WIREFRAME = False
SURFACE_GRID_STEP = 6
SURFACE_GRID_COLOR = (70, 70, 70)

DRAW_AXIS = True
AXIS_COLOR_X = (70, 70, 70)
AXIS_COLOR_Y = (100, 100, 100)
AXIS_COLOR_Z = (50, 50, 50)

DRAW_OUTLINE = True
OUTLINE_COLOR = (45, 45, 45)

SHOW_VIEW_INFO = True
SHOW_FPS = False

# =========================
# Temporal smoothing settings
# =========================
ENABLE_ADAPTIVE_TEMPORAL_SMOOTH = True

TEMPORAL_ALPHA_FAST = 0.82
TEMPORAL_ALPHA_SLOW = 0.50
MOTION_THRESHOLD = 0.050

# =========================
# Performance settings
# =========================
IDLE_SLEEP = 0.0

PRINT_MATRIX = False
PRINT_MATRIX_TYPE = "raw"
PRINT_EVERY_N_FRAMES = 100
PRINT_AS_ARDUINO_ARRAY = True

# =========================
# Global shared variables
# =========================
latest_vis_norm = np.zeros((VIS_ROWS, VIS_COLS), dtype=np.float32)

flag = False
running = True
latest_frame_id = 0

display_prev = np.zeros((VIS_ROWS, VIS_COLS), dtype=np.float32)

fps_last_time = time.time()
fps_counter = 0
fps_value = 0.0


# =====================================================
# Precompute LUT / light / canvas
# =====================================================
def build_gamma_lut(gamma):
    lut = np.arange(256, dtype=np.float32) / 255.0
    lut = np.power(lut, gamma)
    lut = np.clip(lut * 255.0, 0, 255).astype(np.uint8)
    return lut


GAMMA_LUT = build_gamma_lut(GAMMA)


def build_colormap_lut(colormap_style):
    values = np.arange(256, dtype=np.uint8).reshape(256, 1)
    colors = cv2.applyColorMap(values, colormap_style)
    colors = colors.reshape(256, 3)
    return colors.astype(np.float32)


COLORMAP_LUT = build_colormap_lut(COLORMAP_STYLE)


def make_white_canvas():
    return np.full(
        (CANVAS_HEIGHT, CANVAS_WIDTH, 3),
        BACKGROUND_COLOR,
        dtype=np.uint8
    )


WHITE_CANVAS = make_white_canvas()

LIGHT_DIR_WORLD = LIGHT_DIR_WORLD / (np.linalg.norm(LIGHT_DIR_WORLD) + 1e-6)

VIEW_DIR_WORLD = np.array([0.0, 0.0, 1.0], dtype=np.float32)
HALF_DIR_WORLD = LIGHT_DIR_WORLD + VIEW_DIR_WORLD
HALF_DIR_WORLD = HALF_DIR_WORLD / (np.linalg.norm(HALF_DIR_WORLD) + 1e-6)


# =====================================================
# Mesh coordinates
# =====================================================
def build_mesh_coordinates():
    x = np.linspace(
        -SURFACE_X_SIZE / 2.0,
        SURFACE_X_SIZE / 2.0,
        MESH_COLS,
        dtype=np.float32
    )

    y = np.linspace(
        -SURFACE_Y_SIZE / 2.0,
        SURFACE_Y_SIZE / 2.0,
        MESH_ROWS,
        dtype=np.float32
    )

    xx, yy = np.meshgrid(x, y)
    return xx, yy


MESH_X, MESH_Y = build_mesh_coordinates()
MESH_Z_ZERO = np.zeros_like(MESH_X, dtype=np.float32)


# =====================================================
# Camera projection
# =====================================================
def get_camera_basis():
    """
    构建正交相机坐标系。
    yaw 控制水平旋转，elevation 控制俯仰。
    """
    yaw = np.deg2rad(view_yaw_deg)
    elev = np.deg2rad(view_elev_deg)

    # 从物体指向相机的方向
    camera_dir = np.array([
        np.cos(elev) * np.cos(yaw),
        np.cos(elev) * np.sin(yaw),
        np.sin(elev)
    ], dtype=np.float32)

    camera_dir = camera_dir / (np.linalg.norm(camera_dir) + 1e-6)

    # 屏幕右方向
    right = np.array([
        -np.sin(yaw),
        np.cos(yaw),
        0.0
    ], dtype=np.float32)

    right = right / (np.linalg.norm(right) + 1e-6)

    # 屏幕上方向
    up = np.cross(camera_dir, right)
    up = up / (np.linalg.norm(up) + 1e-6)

    return right, up, camera_dir


def project_arrays(x, y, z):
    """
    3D 正交投影到屏幕。
    """
    right, up, camera_dir = get_camera_basis()

    sx = CAMERA_CENTER_X + camera_zoom * (
        x * right[0] + y * right[1] + z * right[2]
    )

    sy = CAMERA_CENTER_Y - camera_zoom * (
        x * up[0] + y * up[1] + z * up[2]
    )

    depth = (
        x * camera_dir[0]
        + y * camera_dir[1]
        + z * camera_dir[2]
    )

    return sx, sy, depth


def project_point(x, y, z):
    right, up, camera_dir = get_camera_basis()

    sx = CAMERA_CENTER_X + camera_zoom * (
        x * right[0] + y * right[1] + z * right[2]
    )

    sy = CAMERA_CENTER_Y - camera_zoom * (
        x * up[0] + y * up[1] + z * up[2]
    )

    depth = (
        x * camera_dir[0]
        + y * camera_dir[1]
        + z * camera_dir[2]
    )

    return float(sx), float(sy), float(depth)


def point_int(x, y):
    return int(round(float(x))), int(round(float(y)))


def poly_from_points(points):
    return np.array(points, dtype=np.int32)


# =====================================================
# Printing helpers
# =====================================================
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


# =====================================================
# Serial frame extraction
# =====================================================
def extract_next_frame(ring):
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


# =====================================================
# Signal processing
# =====================================================
def normalize_contact_map(contact_crop):
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
    delta = float(np.mean(np.abs(new_frame - prev_frame)))

    if delta > MOTION_THRESHOLD:
        alpha = TEMPORAL_ALPHA_FAST
    else:
        alpha = TEMPORAL_ALPHA_SLOW

    out = alpha * new_frame + (1.0 - alpha) * prev_frame
    return out.astype(np.float32, copy=False)


# =====================================================
# 3D rendering helpers
# =====================================================
def soft_low_cut(frame):
    out = (frame - LOW_CUT) / max(1e-6, 1.0 - LOW_CUT)
    np.clip(out, 0.0, 1.0, out=out)
    return out.astype(np.float32, copy=False)


def smoothstep(edge0, edge1, x):
    t = (x - edge0) / max(1e-6, edge1 - edge0)
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def translate_float_map(src, dx, dy):
    h, w = src.shape[:2]

    mat = np.float32([
        [1, 0, dx],
        [0, 1, dy]
    ])

    shifted = cv2.warpAffine(
        src,
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return shifted.astype(np.float32, copy=False)


def color_blend_white(color_bgr, alpha):
    alpha = float(np.clip(alpha, 0.0, 1.0))

    color = np.asarray(color_bgr, dtype=np.float32)
    white = np.asarray(BACKGROUND_COLOR, dtype=np.float32)

    out = white * (1.0 - alpha) + color * alpha
    return tuple(np.clip(out, 0, 255).astype(np.uint8).tolist())


def draw_base_grid(canvas, base_sx, base_sy):
    if not DRAW_BASE_GRID:
        return

    for i in range(0, MESH_ROWS, BASE_GRID_STEP):
        pts = []
        for j in range(MESH_COLS):
            pts.append(point_int(base_sx[i, j], base_sy[i, j]))
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, BASE_GRID_COLOR, 1, cv2.LINE_AA)

    for j in range(0, MESH_COLS, BASE_GRID_STEP):
        pts = []
        for i in range(MESH_ROWS):
            pts.append(point_int(base_sx[i, j], base_sy[i, j]))
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, BASE_GRID_COLOR, 1, cv2.LINE_AA)


def draw_axis(canvas):
    if not DRAW_AXIS:
        return

    x_min = -SURFACE_X_SIZE / 2.0
    x_max = SURFACE_X_SIZE / 2.0

    y_min = -SURFACE_Y_SIZE / 2.0
    y_max = SURFACE_Y_SIZE / 2.0

    ox, oy, _ = project_point(x_min, y_max, 0.0)
    x2, y2, _ = project_point(x_max, y_max, 0.0)
    y2x, y2y, _ = project_point(x_min, y_min, 0.0)
    z2x, z2y, _ = project_point(x_min, y_max, view_z_scale * 0.85)

    cv2.arrowedLine(canvas, point_int(ox, oy), point_int(x2, y2), AXIS_COLOR_X, 2, cv2.LINE_AA, tipLength=0.05)
    cv2.arrowedLine(canvas, point_int(ox, oy), point_int(y2x, y2y), AXIS_COLOR_Y, 2, cv2.LINE_AA, tipLength=0.05)
    cv2.arrowedLine(canvas, point_int(ox, oy), point_int(z2x, z2y), AXIS_COLOR_Z, 2, cv2.LINE_AA, tipLength=0.08)

    cv2.putText(canvas, "X", point_int(x2 + 8, y2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, AXIS_COLOR_X, 2, cv2.LINE_AA)
    cv2.putText(canvas, "Y", point_int(y2x - 26, y2y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, AXIS_COLOR_Y, 2, cv2.LINE_AA)
    cv2.putText(canvas, "Z", point_int(z2x + 8, z2y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, AXIS_COLOR_Z, 2, cv2.LINE_AA)


def draw_shadow(canvas, base_sx, base_sy, alpha_map):
    if not ENABLE_DROP_SHADOW:
        return canvas

    mask = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=np.uint8)

    for i in range(MESH_ROWS - 1):
        for j in range(MESH_COLS - 1):
            a = float(np.mean(alpha_map[i:i + 2, j:j + 2]))

            if a < 0.02:
                continue

            pts = poly_from_points([
                point_int(base_sx[i, j], base_sy[i, j]),
                point_int(base_sx[i, j + 1], base_sy[i, j + 1]),
                point_int(base_sx[i + 1, j + 1], base_sy[i + 1, j + 1]),
                point_int(base_sx[i + 1, j], base_sy[i + 1, j]),
            ])

            cv2.fillConvexPoly(mask, pts, int(np.clip(a * 255, 0, 255)), cv2.LINE_AA)

    shadow = cv2.GaussianBlur(mask, (0, 0), SHADOW_BLUR_SIGMA)

    shadow = translate_float_map(
        shadow.astype(np.float32) / 255.0,
        SHADOW_OFFSET_X,
        SHADOW_OFFSET_Y
    )

    shadow = np.clip(shadow * SHADOW_STRENGTH, 0.0, 0.45)

    canvas_float = canvas.astype(np.float32)
    canvas_float *= (1.0 - shadow[..., None])

    return np.clip(canvas_float, 0, 255).astype(np.uint8)


def draw_surface_wireframe(canvas, sx, sy):
    if not DRAW_SURFACE_WIREFRAME:
        return

    for i in range(0, MESH_ROWS, SURFACE_GRID_STEP):
        pts = []
        for j in range(MESH_COLS):
            pts.append(point_int(sx[i, j], sy[i, j]))

        cv2.polylines(
            canvas,
            [np.array(pts, dtype=np.int32)],
            False,
            SURFACE_GRID_COLOR,
            1,
            cv2.LINE_AA
        )

    for j in range(0, MESH_COLS, SURFACE_GRID_STEP):
        pts = []
        for i in range(MESH_ROWS):
            pts.append(point_int(sx[i, j], sy[i, j]))

        cv2.polylines(
            canvas,
            [np.array(pts, dtype=np.int32)],
            False,
            SURFACE_GRID_COLOR,
            1,
            cv2.LINE_AA
        )


def draw_surface_outline(canvas, sx, sy):
    if not DRAW_OUTLINE:
        return

    # 前边界
    pts_front = []
    i = MESH_ROWS - 1
    for j in range(MESH_COLS):
        pts_front.append(point_int(sx[i, j], sy[i, j]))

    cv2.polylines(canvas, [np.array(pts_front, dtype=np.int32)], False, OUTLINE_COLOR, 2, cv2.LINE_AA)

    # 后边界
    pts_back = []
    i = 0
    for j in range(MESH_COLS):
        pts_back.append(point_int(sx[i, j], sy[i, j]))

    cv2.polylines(canvas, [np.array(pts_back, dtype=np.int32)], False, OUTLINE_COLOR, 1, cv2.LINE_AA)

    # 左边界
    pts_left = []
    j = 0
    for i in range(MESH_ROWS):
        pts_left.append(point_int(sx[i, j], sy[i, j]))

    cv2.polylines(canvas, [np.array(pts_left, dtype=np.int32)], False, OUTLINE_COLOR, 2, cv2.LINE_AA)

    # 右边界
    pts_right = []
    j = MESH_COLS - 1
    for i in range(MESH_ROWS):
        pts_right.append(point_int(sx[i, j], sy[i, j]))

    cv2.polylines(canvas, [np.array(pts_right, dtype=np.int32)], False, OUTLINE_COLOR, 2, cv2.LINE_AA)


def draw_view_info(canvas):
    if not SHOW_VIEW_INFO:
        return

    text1 = f"Yaw: {view_yaw_deg:.0f} deg   Elev: {view_elev_deg:.0f} deg   Z: {view_z_scale:.2f}   Zoom: {camera_zoom:.0f}"
    text2 = "A/D rotate | W/S pitch | Q/E height | +/- zoom | G grid | M mesh | F FPS | R reset | ESC quit"

    cv2.putText(
        canvas,
        text1,
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (70, 70, 70),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        canvas,
        text2,
        (18, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (110, 110, 110),
        1,
        cv2.LINE_AA
    )


# =====================================================
# Professional 3D surface renderer
# =====================================================
def render_professional_3d_surface(frame_norm):
    """
    真正的可调相机 3D 曲面渲染。
    """
    base = frame_norm.astype(np.float32, copy=True)
    base = soft_low_cut(base)

    if float(np.max(base)) < EMPTY_FRAME_MAX:
        canvas = WHITE_CANVAS.copy()

        base_sx, base_sy, _ = project_arrays(
            MESH_X,
            MESH_Y,
            MESH_Z_ZERO
        )

        draw_base_grid(canvas, base_sx, base_sy)
        draw_axis(canvas)
        draw_view_info(canvas)

        return canvas

    # 高度增强
    base *= HEIGHT_GAIN
    np.clip(base, 0.0, 1.0, out=base)

    # 插值成高分辨率曲面
    height = cv2.resize(
        base,
        (MESH_COLS, MESH_ROWS),
        interpolation=UPSAMPLE_INTERP
    )

    height = np.clip(height, 0.0, 1.0).astype(np.float32, copy=False)

    # 平滑高度场
    height = cv2.GaussianBlur(height, (0, 0), HEIGHT_BLUR_SIGMA)

    # 高度 gamma
    height = np.power(np.clip(height, 0.0, 1.0), HEIGHT_GAMMA)
    height = np.clip(height, 0.0, 1.0).astype(np.float32, copy=False)

    # 实际 Z 高度
    z = height * view_z_scale

    # 投影底面和曲面
    base_sx, base_sy, base_depth = project_arrays(
        MESH_X,
        MESH_Y,
        MESH_Z_ZERO
    )

    sx, sy, depth = project_arrays(
        MESH_X,
        MESH_Y,
        z
    )

    # =====================================================
    # A. 法线与光照
    # =====================================================
    dx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)

    nx = -dx * NORMAL_STRENGTH
    ny = -dy * NORMAL_STRENGTH
    nz = np.ones_like(height, dtype=np.float32)

    n_len = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
    nx /= n_len
    ny /= n_len
    nz /= n_len

    diffuse = (
        nx * LIGHT_DIR_WORLD[0]
        + ny * LIGHT_DIR_WORLD[1]
        + nz * LIGHT_DIR_WORLD[2]
    )

    diffuse = np.clip(diffuse, 0.0, 1.0)

    spec = (
        nx * HALF_DIR_WORLD[0]
        + ny * HALF_DIR_WORLD[1]
        + nz * HALF_DIR_WORLD[2]
    )

    spec = np.clip(spec, 0.0, 1.0)
    spec = np.power(spec, SPECULAR_POWER)

    shade = AMBIENT_LIGHT + DIFFUSE_STRENGTH * diffuse + SPECULAR_STRENGTH * spec
    shade = np.clip(shade, 0.0, 1.45).astype(np.float32, copy=False)

    # =====================================================
    # B. 颜色和透明度
    # =====================================================
    gray = np.clip(height * 255.0, 0, 255).astype(np.uint8)
    gray_gamma = cv2.LUT(gray, GAMMA_LUT)

    color_index_map = gray_gamma.astype(np.int32)

    alpha_map = smoothstep(SURFACE_ALPHA_CUT, 1.0, height)
    alpha_map = np.power(alpha_map, WHITE_BLEND_POWER)
    alpha_map = np.clip(alpha_map, 0.0, 1.0)
    alpha_map[height < SURFACE_ALPHA_CUT] = 0.0

    # =====================================================
    # C. 画布、底面和阴影
    # =====================================================
    canvas = WHITE_CANVAS.copy()

    draw_base_grid(canvas, base_sx, base_sy)

    canvas = draw_shadow(canvas, base_sx, base_sy, alpha_map)

    # =====================================================
    # D. 绘制曲面单元
    # painter's algorithm：远处先画，近处后画
    # =====================================================
    cells = []

    for i in range(MESH_ROWS - 1):
        for j in range(MESH_COLS - 1):
            a = float(np.mean(alpha_map[i:i + 2, j:j + 2]))

            if a < 0.01:
                continue

            d = float(np.mean(depth[i:i + 2, j:j + 2]))
            cells.append((d, i, j, a))

    # 正交相机中 depth 越小越远，先画远处
    cells.sort(key=lambda item: item[0])

    for _, i, j, a in cells:
        pts = poly_from_points([
            point_int(sx[i, j], sy[i, j]),
            point_int(sx[i, j + 1], sy[i, j + 1]),
            point_int(sx[i + 1, j + 1], sy[i + 1, j + 1]),
            point_int(sx[i + 1, j], sy[i + 1, j]),
        ])

        idx = int(np.clip(np.mean(color_index_map[i:i + 2, j:j + 2]), 0, 255))

        base_color = COLORMAP_LUT[idx]
        shade_mean = float(np.mean(shade[i:i + 2, j:j + 2]))

        surface_color = base_color * shade_mean
        surface_color = np.clip(surface_color, 0, 255)

        alpha = float(np.clip(a, 0.0, 1.0))
        final_color = color_blend_white(surface_color, alpha)

        cv2.fillConvexPoly(canvas, pts, final_color, cv2.LINE_AA)

    draw_surface_wireframe(canvas, sx, sy)
    draw_surface_outline(canvas, sx, sy)
    draw_axis(canvas)
    draw_view_info(canvas)

    return canvas


# =====================================================
# Serial reading thread
# =====================================================
def readThread(serDev):
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
    print("初始化完成！开始专业 3D 触觉曲面显示。")
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
# Start serial
# =====================================================
serDev = serial.Serial(PORT, BAUD)
serDev.flush()
serDev.reset_input_buffer()

try:
    serDev.set_buffer_size(rx_size=262144, tx_size=262144)
except Exception:
    pass


# =====================================================
# OpenCV window
# =====================================================
cv2.namedWindow("Contact Data_left_Professional_3D", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Contact Data_left_Professional_3D", WINDOW_WIDTH, WINDOW_HEIGHT)

initial_canvas = WHITE_CANVAS.copy()
base_sx0, base_sy0, _ = project_arrays(MESH_X, MESH_Y, MESH_Z_ZERO)
draw_base_grid(initial_canvas, base_sx0, base_sy0)
draw_axis(initial_canvas)
draw_view_info(initial_canvas)

cv2.imshow("Contact Data_left_Professional_3D", initial_canvas)
cv2.waitKey(1)


# =====================================================
# Start thread
# =====================================================
serialThread = threading.Thread(target=readThread, args=(serDev,))
serialThread.daemon = True
serialThread.start()


# =====================================================
# Keyboard control
# =====================================================
def clamp_view_params():
    global view_elev_deg, view_z_scale, camera_zoom

    view_elev_deg = float(np.clip(view_elev_deg, MIN_ELEV_DEG, MAX_ELEV_DEG))
    view_z_scale = float(np.clip(view_z_scale, MIN_Z_SCALE, MAX_Z_SCALE))
    camera_zoom = float(np.clip(camera_zoom, MIN_CAMERA_ZOOM, MAX_CAMERA_ZOOM))


def reset_view():
    global view_yaw_deg, view_elev_deg, view_z_scale, camera_zoom

    view_yaw_deg = DEFAULT_YAW_DEG
    view_elev_deg = DEFAULT_ELEV_DEG
    view_z_scale = DEFAULT_Z_SCALE
    camera_zoom = DEFAULT_CAMERA_ZOOM


# =====================================================
# Main visualization loop
# =====================================================
if __name__ == "__main__":
    print("开始接收数据测试，按 ESC 退出。")
    print("A/D 旋转 | W/S 俯仰 | Q/E 高度 | +/- 缩放 | G 网格 | M 曲面线框 | F FPS | R 重置")

    last_displayed_frame_id = -1
    view_dirty = True

    try:
        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

            # =========================
            # Keyboard controls
            # =========================
            if key == ord("a") or key == ord("A"):
                view_yaw_deg -= 5.0
                view_dirty = True

            elif key == ord("d") or key == ord("D"):
                view_yaw_deg += 5.0
                view_dirty = True

            elif key == ord("w") or key == ord("W"):
                view_elev_deg += 3.0
                clamp_view_params()
                view_dirty = True

            elif key == ord("s") or key == ord("S"):
                view_elev_deg -= 3.0
                clamp_view_params()
                view_dirty = True

            elif key == ord("q") or key == ord("Q"):
                view_z_scale -= 0.08
                clamp_view_params()
                view_dirty = True

            elif key == ord("e") or key == ord("E"):
                view_z_scale += 0.08
                clamp_view_params()
                view_dirty = True

            elif key == ord("+") or key == ord("="):
                camera_zoom += 15.0
                clamp_view_params()
                view_dirty = True

            elif key == ord("-") or key == ord("_"):
                camera_zoom -= 15.0
                clamp_view_params()
                view_dirty = True

            elif key == ord("r") or key == ord("R"):
                reset_view()
                view_dirty = True

            elif key == ord("g") or key == ord("G"):
                DRAW_BASE_GRID = not DRAW_BASE_GRID
                view_dirty = True

            elif key == ord("m") or key == ord("M"):
                DRAW_SURFACE_WIREFRAME = not DRAW_SURFACE_WIREFRAME
                view_dirty = True

            elif key == ord("f") or key == ord("F"):
                SHOW_FPS = not SHOW_FPS
                view_dirty = True

            # 初始化未完成前，显示白底 3D 坐标网格
            if not flag:
                if view_dirty:
                    initial_canvas = WHITE_CANVAS.copy()
                    base_sx0, base_sy0, _ = project_arrays(MESH_X, MESH_Y, MESH_Z_ZERO)
                    draw_base_grid(initial_canvas, base_sx0, base_sy0)
                    draw_axis(initial_canvas)
                    draw_view_info(initial_canvas)
                    view_dirty = False

                cv2.imshow("Contact Data_left_Professional_3D", initial_canvas)

                if IDLE_SLEEP > 0:
                    time.sleep(IDLE_SLEEP)

                continue

            # 有新帧或者视角发生变化时才重新渲染
            if latest_frame_id != last_displayed_frame_id or view_dirty:
                last_displayed_frame_id = latest_frame_id
                view_dirty = False

                frame_to_show = latest_vis_norm.copy()

                if ENABLE_ADAPTIVE_TEMPORAL_SMOOTH:
                    frame_to_show = adaptive_temporal_filter(
                        frame_to_show,
                        display_prev
                    )

                    display_prev = frame_to_show.copy()
                else:
                    display_prev = frame_to_show.copy()

                rendered = render_professional_3d_surface(frame_to_show)

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
                        (18, 88),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.70,
                        (60, 60, 60),
                        2,
                        cv2.LINE_AA
                    )

                cv2.imshow("Contact Data_left_Professional_3D", rendered)

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
