import os

# 必须放在 import cv2 前面，减少 OpenCV Qt 字体警告
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu"

import cv2

# cv2 导入后再设置一次，防止 opencv-python 覆盖字体路径
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu"

import time
import numpy as np
from openni import openni2


# =========================
# OpenNI / Astra-S 参数
# =========================
OPENNI2_REDIST = "/home/hjx/orbbec_openni_redist"

WINDOW_NAME = "Astra-S"  # 注意默认要开启左右镜像翻转

WIDTH = 640
HEIGHT = 480
FPS = 30

DISPLAY_SIZE = (WIDTH, HEIGHT)


# =========================
# 显示参数
# =========================
SHOW_FPS = True

# 解决左右颠倒：True 表示左右翻转
FLIP_HORIZONTAL = True

# 如果上下也颠倒，可改成 True
FLIP_VERTICAL = False


def init_openni():
    """初始化 OpenNI，并打开 Astra-S 设备"""
    openni2.initialize(OPENNI2_REDIST)

    devices = openni2.Device.enumerate_uris()
    if not devices:
        raise RuntimeError("未发现 Astra-S 设备")

    print(f"发现设备: {devices[0]}")

    device = openni2.Device.open_any()
    return device


def set_color_stream_mode(color_stream):
    """设置 RGB 彩色流模式：640x480 @ 30 FPS"""
    try:
        video_mode = openni2.c_api.OniVideoMode(
            pixelFormat=openni2.PIXEL_FORMAT_RGB888,
            resolutionX=WIDTH,
            resolutionY=HEIGHT,
            fps=FPS,
        )

        color_stream.set_video_mode(video_mode)
        print(f"设置 RGB 模式: {WIDTH}x{HEIGHT} @ {FPS} FPS")

    except Exception as e:
        print("设置 RGB 模式失败，将使用默认模式。")
        print("错误信息:", e)


def open_astra_color_stream():
    """打开 Astra-S RGB 彩色流"""
    device = init_openni()

    color_stream = device.create_color_stream()

    # 必须在 start() 前设置视频模式
    set_color_stream_mode(color_stream)

    color_stream.start()
    print("RGB 彩色流已启动")

    # 返回 device，避免设备对象被提前释放
    return device, color_stream


def read_rgb_frame(color_stream):
    """读取一帧 RGB 图像，并转换为 OpenCV 可显示的 BGR 图像"""
    frame = color_stream.read_frame()

    width = frame.width
    height = frame.height

    data = frame.get_buffer_as_uint8()
    rgb = np.frombuffer(data, dtype=np.uint8)

    expected_size = width * height * 3
    if rgb.size != expected_size:
        raise RuntimeError(f"RGB 数据尺寸异常: 当前 {rgb.size}, 期望 {expected_size}")

    rgb = rgb.reshape((height, width, 3))

    # OpenNI 输出 RGB，OpenCV 显示需要 BGR
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # 统一显示尺寸为 640x480
    if (width, height) != DISPLAY_SIZE:
        bgr = cv2.resize(bgr, DISPLAY_SIZE, interpolation=cv2.INTER_LINEAR)

    return bgr


def apply_flip(image, flip_horizontal=True, flip_vertical=False):
    """根据参数对图像进行翻转"""
    if flip_horizontal and flip_vertical:
        # -1 表示上下 + 左右同时翻转
        image = cv2.flip(image, -1)
    elif flip_horizontal:
        # 1 表示左右翻转
        image = cv2.flip(image, 1)
    elif flip_vertical:
        # 0 表示上下翻转
        image = cv2.flip(image, 0)

    return image


def draw_info(image, fps_value, mirror_enabled):
    """在图像左上角显示 FPS 和镜像状态"""
    cv2.putText(
        image,
        f"FPS: {fps_value:.1f}",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    mirror_text = "Mirror: ON" if mirror_enabled else "Mirror: OFF"
    cv2.putText(
        image,
        mirror_text,
        (15, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return image


def main():
    print("正在启动 Astra-S RGB 画面...")
    print(f"目标参数: {WIDTH}x{HEIGHT} @ {FPS} FPS")
    print(f"左右翻转 FLIP_HORIZONTAL = {FLIP_HORIZONTAL}")

    device = None
    color_stream = None

    mirror_enabled = FLIP_HORIZONTAL

    try:
        device, color_stream = open_astra_color_stream()

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WIDTH, HEIGHT)

        print("RGB 画面已启动")
        print("按 q 或 ESC 退出")
        print("按 m 切换左右镜像显示")

        frame_count = 0
        fps_value = 0.0
        last_time = time.perf_counter()

        while True:
            image = read_rgb_frame(color_stream)

            # 先翻转画面，再画 FPS 文字，保证文字不会被镜像
            image = apply_flip(
                image,
                flip_horizontal=mirror_enabled,
                flip_vertical=FLIP_VERTICAL,
            )

            if SHOW_FPS:
                frame_count += 1
                now = time.perf_counter()

                if now - last_time >= 1.0:
                    fps_value = frame_count / (now - last_time)
                    frame_count = 0
                    last_time = now

                image = draw_info(image, fps_value, mirror_enabled)

            cv2.imshow(WINDOW_NAME, image)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

            elif key == ord("m"):
                mirror_enabled = not mirror_enabled
                print(f"左右镜像显示: {'开启' if mirror_enabled else '关闭'}")

    finally:
        print("正在关闭 RGB 流...")

        if color_stream is not None:
            try:
                color_stream.stop()
            except Exception as e:
                print("关闭 RGB 流时出现异常:", e)

        cv2.destroyAllWindows()

        # 不调用 openni2.unload()
        # 你之前遇到过 corrupted size vs. prev_size，
        # 当前 Orbbec OpenNI beta 驱动在 Python 退出时可能释放内存异常。

        print("程序已退出")


if __name__ == "__main__":
    main()
