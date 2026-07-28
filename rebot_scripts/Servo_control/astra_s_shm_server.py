#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Astra-S RGB shared-memory capture helper.

This script is intentionally minimal and should be launched as a completely
separate Python process by record_rebot_episodes.py.  It avoids importing
MuJoCo, RealSense, robot drivers, tqdm, etc., because Orbbec OpenNI beta can
segfault when mixed with a complex robotics process.

Shared memory protocol:
    frame shm: uint8[height, width, 3], RGB
    meta  shm: struct '<qqdii'
        seq:      int64  seqlock counter; odd while writer is writing, even when stable
        frame_id: int64  monotonically increasing camera frame id
        timestamp: double time.perf_counter() when frame was written
        valid:    int32  1 after first valid frame, else 0
        error:    int32  0 normal, nonzero on handled error
"""

from __future__ import annotations

import argparse
import os
import struct
import time
from multiprocessing import resource_tracker, shared_memory

# Must be set before importing cv2.
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import numpy as np
from openni import openni2

META_FORMAT = "<qqdii"
META_SIZE = struct.calcsize(META_FORMAT)


def write_meta(meta_buf, seq: int, frame_id: int, timestamp: float, valid: int, error: int) -> None:
    struct.pack_into(META_FORMAT, meta_buf, 0, int(seq), int(frame_id), float(timestamp), int(valid), int(error))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Astra-S RGB shared-memory capture server")
    parser.add_argument("--frame-shm", required=True)
    parser.add_argument("--meta-shm", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--openni2-redist", type=str, default="/home/hjx/orbbec_openni_redist")
    parser.add_argument("--flip-horizontal", type=int, default=1)
    parser.add_argument("--flip-vertical", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0005)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frame_shm = shared_memory.SharedMemory(name=args.frame_shm)
    meta_shm = shared_memory.SharedMemory(name=args.meta_shm)
    # 共享内存由父采集进程创建和销毁。Python 3.10 会错误地让附加到共享内存的
    # helper 也注册清理，helper 异常退出时便会产生 leaked shared_memory 警告，
    # 甚至抢先 unlink 父进程仍在使用的段。
    resource_tracker.unregister(frame_shm._name, "shared_memory")
    resource_tracker.unregister(meta_shm._name, "shared_memory")

    frame_buf = np.ndarray((args.height, args.width, 3), dtype=np.uint8, buffer=frame_shm.buf)
    meta_buf = meta_shm.buf

    seq = 0
    frame_id = 0
    write_meta(meta_buf, seq, frame_id, 0.0, 0, 0)

    color_stream = None

    try:
        openni2.initialize(args.openni2_redist)

        devices = openni2.Device.enumerate_uris()
        if not devices:
            print("❌ [Astra-S external] OpenNI 未发现 Astra-S 设备", flush=True)
            write_meta(meta_buf, seq, frame_id, 0.0, 0, 1)
            return

        print(f"📷 [Astra-S external] 发现设备: {devices[0]}", flush=True)

        device = openni2.Device.open_any()
        color_stream = device.create_color_stream()

        # 与 CameraPreview_Astra-s_test.py 保持一致。该 Astra-S 能枚举设备，
        # 但驱动默认流在 read_frame() 时会返回 ONI_STATUS_ERROR；显式指定
        # RGB888 640x480@30 后才能稳定采集。
        video_mode = openni2.c_api.OniVideoMode(
            pixelFormat=openni2.PIXEL_FORMAT_RGB888,
            resolutionX=args.width,
            resolutionY=args.height,
            fps=30,
        )
        color_stream.set_video_mode(video_mode)
        print(
            f"📷 [Astra-S external] RGB 模式: {args.width}x{args.height} @ 30 FPS",
            flush=True,
        )

        color_stream.start()

        while True:
            frame = color_stream.read_frame()
            src_w = int(frame.width)
            src_h = int(frame.height)

            raw = np.frombuffer(frame.get_buffer_as_uint8(), dtype=np.uint8)
            expected = src_w * src_h * 3
            if raw.size != expected:
                continue

            rgb = raw.reshape((src_h, src_w, 3))

            if src_w != args.width or src_h != args.height:
                rgb = cv2.resize(rgb, (args.width, args.height), interpolation=cv2.INTER_LINEAR)

            if args.flip_horizontal and args.flip_vertical:
                rgb = cv2.flip(rgb, -1)
            elif args.flip_horizontal:
                rgb = cv2.flip(rgb, 1)
            elif args.flip_vertical:
                rgb = cv2.flip(rgb, 0)

            # Seqlock write: odd seq while writing, even seq after stable.
            seq += 1
            write_meta(meta_buf, seq, frame_id, time.perf_counter(), 1, 0)
            frame_buf[:] = rgb
            frame_id += 1
            seq += 1
            write_meta(meta_buf, seq, frame_id, time.perf_counter(), 1, 0)

            if args.sleep > 0:
                time.sleep(args.sleep)

    except KeyboardInterrupt:
        pass
    except BaseException as e:
        print(f"❌ [Astra-S external] 采集进程异常: {repr(e)}", flush=True)
        try:
            write_meta(meta_buf, seq, frame_id, time.perf_counter(), 0, 2)
        except Exception:
            pass
    finally:
        try:
            if color_stream is not None:
                color_stream.stop()
        except Exception:
            pass

        # Do not call openni2.unload(); Orbbec beta driver may crash at unload.
        try:
            frame_shm.close()
            meta_shm.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
