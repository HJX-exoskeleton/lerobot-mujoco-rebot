"""Orbbec Gemini 336L RGB-only preview.

Only the color stream is enabled; depth and infrared streams are not started.
"""

import os
import time

# Set before importing cv2 to reduce Qt font warnings on Linux.
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import numpy as np
from pyorbbecsdk import Config, OBError, OBFormat, OBSensorType, Pipeline


WINDOW_NAME = "Orbbec Gemini 336L - RGB"
WIDTH = 640
HEIGHT = 480
FPS = 30
FRAME_TIMEOUT_MS = 1000

SHOW_FPS = True
def select_color_profile(pipeline: Pipeline):
    """Select 640x480 RGB@30 when supported, otherwise use device default."""
    profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    if profiles is None or profiles.get_count() == 0:
        raise RuntimeError("设备没有可用的 RGB 彩色流")

    try:
        profile = profiles.get_video_stream_profile(
            WIDTH, HEIGHT, OBFormat.RGB, FPS
        )
        mode_source = "目标"
    except OBError:
        profile = profiles.get_default_video_stream_profile()
        mode_source = "设备默认"

    print(
        f"启用{mode_source}彩色流: {profile.get_width()}x{profile.get_height()} "
        f"@ {profile.get_fps()} FPS, 格式={profile.get_format()}"
    )
    return profile


def frame_to_bgr(frame):
    """Convert a pyorbbecsdk color frame to an OpenCV BGR image."""
    width = frame.get_width()
    height = frame.get_height()
    frame_format = frame.get_format()
    data = np.asanyarray(frame.get_data())

    if frame_format == OBFormat.RGB:
        image = data.reshape((height, width, 3))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if frame_format == OBFormat.BGR:
        return data.reshape((height, width, 3)).copy()
    if frame_format == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame_format == OBFormat.YUYV:
        image = data.reshape((height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    if frame_format == OBFormat.UYVY:
        image = data.reshape((height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)

    raise RuntimeError(f"暂不支持的彩色格式: {frame_format}")


def draw_info(image, fps_value: float):
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
    cv2.putText(
        image,
        "Native mirror",
        (15, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def main():
    pipeline = None
    started = False

    try:
        pipeline = Pipeline()
        config = Config()

        # Explicitly enable only Color. Calling pipeline.start() without this
        # config would load the default RGB-D configuration and start Depth.
        config.disable_all_stream()
        config.enable_stream(select_color_profile(pipeline))
        pipeline.start(config)
        started = True

        print("RGB 彩色流已启动（未启用深度/红外流）")
        print("使用相机原生镜像方向；按 Q 或 ESC 退出")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WIDTH, HEIGHT)

        frame_count = 0
        fps_value = 0.0
        fps_started_at = time.perf_counter()

        while True:
            frames = pipeline.wait_for_frames(FRAME_TIMEOUT_MS)
            if frames is None:
                print("等待 RGB 帧超时，继续重试...")
                continue

            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue

            image = frame_to_bgr(color_frame)
            if image is None or image.size == 0:
                print("RGB 帧解码失败，跳过当前帧")
                continue

            if SHOW_FPS:
                frame_count += 1
                now = time.perf_counter()
                elapsed = now - fps_started_at
                if elapsed >= 1.0:
                    fps_value = frame_count / elapsed
                    frame_count = 0
                    fps_started_at = now
                draw_info(image, fps_value)

            cv2.imshow(WINDOW_NAME, image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    except KeyboardInterrupt:
        print("收到中断，正在退出...")
    except (OBError, RuntimeError) as exc:
        print(f"相机启动或取流失败: {exc}")
        print("请检查 Gemini 336L 连接、USB 权限及 pyorbbecsdk2 安装。")
    finally:
        cv2.destroyAllWindows()
        if started and pipeline is not None:
            try:
                pipeline.stop()
            except OBError as exc:
                print(f"停止 Pipeline 时出现异常: {exc}")
        print("程序已退出")


if __name__ == "__main__":
    main()
