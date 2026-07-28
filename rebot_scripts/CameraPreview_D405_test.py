import os

# 先设置一次
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu"

import cv2

# cv2 导入后再设置一次，防止 opencv-python 覆盖字体路径
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu"

import numpy as np
import pyrealsense2 as rs


WINDOW_NAME = "RealSense D405 - RGB Only"
WIDTH = 640
HEIGHT = 480
FPS = 30


def start_d405_rgb_pipeline():
    pipeline = rs.pipeline()
    config = rs.config()

    # 优先使用 640x480 BGR8，OpenCV 可直接显示
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    print("正在连接 RealSense D405，仅 RGB 画面...")

    try:
        pipeline.start(config)
        print(f"成功启动彩色流：{WIDTH}x{HEIGHT} @ {FPS}fps")
        return pipeline

    except RuntimeError as e:
        print("标准配置启动失败，尝试自动兼容模式...")
        print("错误信息:", e)

        pipeline = rs.pipeline()
        config_fallback = rs.config()
        config_fallback.enable_stream(rs.stream.color)

        pipeline.start(config_fallback)
        print("已在自动兼容模式下启动彩色流")
        return pipeline


def get_color_image(frames):
    color_frame = frames.get_color_frame()

    if not color_frame:
        return None

    color_image = np.asanyarray(color_frame.get_data())

    # 如果当前流格式是 RGB8，则转为 OpenCV 使用的 BGR
    if color_frame.profile.format() == rs.format.rgb8:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

    # 统一显示为 640x480
    if color_image.shape[1] != WIDTH or color_image.shape[0] != HEIGHT:
        color_image = cv2.resize(color_image, (WIDTH, HEIGHT))

    return color_image


def main():
    pipeline = start_d405_rgb_pipeline()

    # 注意：窗口只创建一次，不要放在 while 循环里面
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WIDTH, HEIGHT)

    print("RGB 画面已启动，按 q 或 ESC 退出")

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                print("等待图像超时，请检查 USB 接口、相机连接或是否被其他程序占用")
                continue

            color_image = get_color_image(frames)

            if color_image is None:
                print("没有获取到 RGB 帧")
                continue

            cv2.imshow(WINDOW_NAME, color_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("摄像头已安全关闭")


if __name__ == "__main__":
    main()
