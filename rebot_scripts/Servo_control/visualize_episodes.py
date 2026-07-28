#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
具身智能 HDF5 数据集全面分析与可视化工具（优化版）
功能：
1. 提取并渲染指定相机视角的视频（MP4）
2. 绘制 6轴机械臂关节 目标指令(Action) vs 实际反馈(qpos) 追踪曲线
3. 绘制夹爪 (Gripper) 追踪曲线
4. 绘制 IMU 10维数据曲线：四元数、角速度、线加速度
5. 绘制并导出触觉热力图，兼容真机右侧单片 (T, 12, 30) 与仿真双通道 (T, 2, 8, 16)

用法示例：
    python visualize_hdf5.py --path /media/hjx/PSSD/hjx_ws/data/rebot/data_real/rebot_real_grasp_banana/episode_0.hdf5 --cams cam_high cam_wrist
    python visualize_hdf5.py --path /media/hjx/PSSD/hjx_ws/data/rebot/data_real/rebot_real_grasp_banana/episode_0.hdf5 --cams cam_high --output ./results
"""

import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import numpy as np
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="HDF5 数据集可视化工具")
    parser.add_argument("--path", type=str, required=True,
                        help="输入的 HDF5 文件路径")
    parser.add_argument("--cams", nargs="+", default=["cam_high"],
                        help="需要导出的相机视角名称，例如 cam_high cam_wrist（默认: cam_high）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录（默认与输入文件同目录）")
    parser.add_argument("--fps", type=int, default=None,
                        help="视频帧率（默认从 HDF5 属性读取，若不存在则使用 50）")
    parser.add_argument("--skip_tactile_video", action="store_true",
                        help="只保存触觉统计图，跳过触觉 MP4")
    parser.add_argument("--tactile_force_max", type=float, default=None,
                        help="触觉热力图上限；默认用 99.5 分位自动估计")
    parser.add_argument("--tactile_video_width", type=int, default=480,
                        help="单通道触觉视频宽度")
    parser.add_argument("--tactile_video_height", type=int, default=240,
                        help="单通道触觉视频高度")
    return parser.parse_args()


def read_action(root):
    """兼容 /action/target_pos 和标准 /action dataset。"""
    if "/action/target_pos" in root:
        return np.asarray(root["/action/target_pos"][()], dtype=np.float32)
    if "/action" in root:
        node = root["/action"]
        if isinstance(node, h5py.Dataset):
            return np.asarray(node[()], dtype=np.float32)
        if "target_pos" in node:
            return np.asarray(node["target_pos"][()], dtype=np.float32)
    raise KeyError("HDF5 中找不到 /action 或 /action/target_pos")


def read_imu(root):
    for key in ("/observations/imu_left", "observations/imu_left", "imu_left"):
        if key in root:
            imu = np.asarray(root[key][()], dtype=np.float32)
            if imu.ndim == 1:
                imu = imu[None, :]
            if imu.ndim != 2 or imu.shape[1] != 10:
                raise ValueError(f"/observations/imu_left 应为 (T, 10)，当前 shape={imu.shape}")
            return imu
    return None


def normalize_tactile_for_visualization(tactile):
    """
    返回统一可视化格式：
        tactile_vis: (T, C, H, W)
        channel_names: list[str]

    支持：
        真机右侧单片: (T, 12, 30)
        仿真双通道:   (T, 2, 8, 16)
        通道在最后:   (T, H, W, C)
    """
    tactile = np.asarray(tactile, dtype=np.float32)
    if tactile.ndim == 3:
        return tactile[:, None, :, :], ["right_gripper"]
    if tactile.ndim == 4 and tactile.shape[1] in (1, 2, 3):
        names = ["left", "right"] if tactile.shape[1] == 2 else [f"ch{i}" for i in range(tactile.shape[1])]
        if tactile.shape[1] == 1:
            names = ["right_gripper"]
        return tactile, names
    if tactile.ndim == 4 and tactile.shape[-1] in (1, 2, 3):
        moved = np.moveaxis(tactile, -1, 1)
        names = ["left", "right"] if moved.shape[1] == 2 else [f"ch{i}" for i in range(moved.shape[1])]
        if moved.shape[1] == 1:
            names = ["right_gripper"]
        return moved, names
    raise ValueError(f"/observations/tactile shape 不支持: {tactile.shape}")


def read_tactile(root):
    if "/observations/tactile" in root:
        tactile = np.asarray(root["/observations/tactile"][()], dtype=np.float32)
        return normalize_tactile_for_visualization(tactile)
    return None, []


def load_data(h5_path):
    """从 HDF5 文件中安全加载所有必要数据"""
    with h5py.File(h5_path, "r") as f:
        # 时间戳
        if "time" in f:
            time_steps = f["time"][:]
        else:
            fps_fallback = float(f.attrs.get("hz_rate", 50))
            action_len = read_action(f).shape[0]
            time_steps = np.arange(action_len, dtype=np.float32) / max(fps_fallback, 1e-6)
        # 帧率
        fps = f.attrs.get("hz_rate", 50)

        # 动作与状态
        action = read_action(f)
        qpos = f["observations"]["qpos"][:]
        qvel = f["observations"]["qvel"][:]  # 虽未使用，仍保留读取
        imu = read_imu(f)
        tactile, tactile_channel_names = read_tactile(f)

        # 相机图像数据字典: {camera_name: ndarray}
        images_dict = {}
        if "images" in f["observations"]:
            img_group = f["observations"]["images"]
            for cam_name in img_group.keys():
                images_dict[cam_name] = img_group[cam_name][:]

        print(f"\n=== 数据概览 ===")
        print(f"  时间步数: {len(time_steps)}")
        print(f"  动作维度: {action.shape}")
        print(f"  Qpos 维度: {qpos.shape}")
        print(f"  Qvel 维度: {qvel.shape}")
        print(f"  IMU 维度: {None if imu is None else imu.shape}")
        print(f"  触觉维度: {None if tactile is None else tactile.shape}, channels={tactile_channel_names}")
        print(f"  可用相机: {list(images_dict.keys())}")
        print(f"  帧率: {fps} Hz")

    return time_steps, action, qpos, qvel, images_dict, imu, tactile, tactile_channel_names, fps


def align_to_min_length(*arrays):
    valid = [arr for arr in arrays if arr is not None]
    min_len = min(len(arr) for arr in valid)
    return tuple(None if arr is None else arr[:min_len] for arr in arrays)


def plot_arm_tracking(time_steps, action, qpos, save_path):
    """绘制 6 轴机械臂关节跟踪曲线（Action vs Qpos）"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("Arm Joints Tracking: Command vs Feedback", fontsize=16)

    for i in range(6):
        row, col = divmod(i, 2)
        ax = axes[row, col]
        ax.plot(time_steps, action[:, i], "--", color="red", linewidth=2, label="Command (Action)")
        ax.plot(time_steps, qpos[:, i], "-", color="blue", alpha=0.7, linewidth=2, label="Real (qpos)")
        ax.set_title(f"Joint {i+1}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (rad)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"✅ 机械臂曲线已保存: {save_path}")


def plot_gripper_tracking(time_steps, action, qpos, save_path):
    """绘制夹爪跟踪曲线（第7个关节）"""
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("Gripper Tracking", fontsize=14)

    ax.plot(time_steps, action[:, 6], "--", color="red", linewidth=2, label="Command (Action)")
    ax.plot(time_steps, qpos[:, 6], "-", color="green", alpha=0.8, linewidth=2, label="Real (qpos)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"✅ 夹爪曲线已保存: {save_path}")


def plot_imu_summary(time_steps, imu, save_path):
    if imu is None:
        print("⚠️ 未找到 /observations/imu_left，跳过 IMU 可视化。")
        return
    time_steps, imu = align_to_min_length(time_steps, imu)

    groups = [
        ("Quaternion (w, x, y, z)", imu[:, 0:4], ["w", "x", "y", "z"], ""),
        ("Angular Velocity (rad/s)", imu[:, 4:7], ["x", "y", "z"], "rad/s"),
        ("Linear Acceleration (g)", imu[:, 7:10], ["x", "y", "z"], "g"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("IMU Data Summary", fontsize=16)
    for ax, (title, values, labels, ylabel) in zip(axes, groups):
        for idx, label in enumerate(labels):
            ax.plot(time_steps, values[:, idx], label=label, linewidth=1.6, drawstyle="steps-pre")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.55)
        ax.legend(loc="upper right", ncol=min(len(labels), 4))
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"✅ IMU 曲线已保存: {save_path}")


def choose_tactile_force_max(tactile, configured_force_max=None):
    if configured_force_max is not None and configured_force_max > 0:
        return float(configured_force_max)
    finite = tactile[np.isfinite(tactile)]
    if finite.size == 0:
        return 1.0
    value = float(np.percentile(finite, 99.5))
    return max(value, 1e-6)


def build_gamma_lut(gamma):
    lut = np.arange(256, dtype=np.float32) / 255.0
    lut = np.power(lut, gamma)
    return np.clip(lut * 255.0, 0, 255).astype(np.uint8)


def build_vignette_mask(width, height, strength=0.22):
    cache_key = (int(width), int(height))
    cached = _VIGNETTE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    radius = np.sqrt(xx * xx + yy * yy)
    radius = np.clip(radius, 0.0, 1.0)
    mask = 1.0 - strength * np.power(radius, 1.7)
    mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
    _VIGNETTE_CACHE[cache_key] = mask
    return mask


def soft_low_cut(frame, low_cut=0.025):
    out = (frame - low_cut) / max(1e-6, 1.0 - low_cut)
    np.clip(out, 0.0, 1.0, out=out)
    return out.astype(np.float32, copy=False)


_TACTILE_GAMMA_LUT = build_gamma_lut(0.82)
_VIGNETTE_CACHE = {}


def tactile_to_colormap(tactile_frame, force_max, width=480, height=240):
    tactile_frame = np.nan_to_num(np.asarray(tactile_frame, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    base = np.clip(tactile_frame, 0.0, force_max) / max(float(force_max), 1e-8)
    base = soft_low_cut(base, low_cut=0.025)
    base *= 1.12
    np.clip(base, 0.0, 1.0, out=base)

    up = cv2.resize(base, (width, height), interpolation=cv2.INTER_CUBIC)
    up = np.clip(up, 0.0, 1.0).astype(np.float32, copy=False)

    smooth = cv2.GaussianBlur(up, (0, 0), 1.15)
    glow = cv2.GaussianBlur(smooth, (0, 0), 4.6)
    mixed = smooth + 0.48 * glow
    np.clip(mixed, 0.0, 1.0, out=mixed)

    gray = np.clip(mixed * 255.0, 0, 255).astype(np.uint8)
    gray = cv2.LUT(gray, _TACTILE_GAMMA_LUT)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)

    intensity = gray.astype(np.float32) / 255.0
    alpha = np.power(np.clip(intensity, 0.0, 1.0), 0.72)
    color_float = color.astype(np.float32) * alpha[..., None]
    color_float *= build_vignette_mask(width, height, strength=0.22)[..., None]
    return np.clip(color_float, 0, 255).astype(np.uint8)


def save_tactile_video(tactile, fps, save_path, channel_names, force_max, width=480, height=240):
    if tactile is None:
        print("⚠️ 未找到 /observations/tactile，跳过触觉视频。")
        return

    tactile = np.asarray(tactile, dtype=np.float32)
    num_frames, num_channels = tactile.shape[:2]
    frame_w = width * num_channels
    frame_h = height
    writer = cv2.VideoWriter(
        save_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1, int(round(float(fps)))),
        (frame_w, frame_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建触觉视频: {save_path}")

    for idx in tqdm(range(num_frames), desc="🖐️ 渲染触觉云图视频", unit="帧"):
        panels = []
        for ch in range(num_channels):
            panel = tactile_to_colormap(tactile[idx, ch], force_max, width=width, height=height)
            label = channel_names[ch] if ch < len(channel_names) else f"ch{ch}"
            if num_channels == 1 and label == "right_gripper":
                label = "right tactile 12x30"
            cv2.putText(panel, label, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2, cv2.LINE_AA)
            cv2.putText(
                panel,
                f"max={np.max(tactile[idx, ch]):.3f} sum={np.sum(tactile[idx, ch]):.2f}",
                (14, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
            panels.append(panel)
        writer.write(np.hstack(panels))
    writer.release()
    print(f"🎉 触觉视频导出完成: {save_path}")


def plot_tactile_summary(time_steps, tactile, save_path, channel_names, force_max):
    if tactile is None:
        print("⚠️ 未找到 /observations/tactile，跳过触觉统计图。")
        return
    time_steps, tactile = align_to_min_length(time_steps, tactile)
    num_channels = tactile.shape[1]

    total_force = tactile.sum(axis=(2, 3))
    max_force = tactile.max(axis=(2, 3))
    active_cells = (tactile > (0.05 * force_max)).sum(axis=(2, 3))
    mean_map = tactile.mean(axis=0)
    max_map = tactile.max(axis=0)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, max(num_channels, 2))
    fig.suptitle("Tactile Summary", fontsize=16)

    ax_total = fig.add_subplot(gs[0, 0])
    ax_max = fig.add_subplot(gs[0, 1])
    ax_active = fig.add_subplot(gs[1, :])

    for ch in range(num_channels):
        name = channel_names[ch] if ch < len(channel_names) else f"ch{ch}"
        ax_total.plot(time_steps, total_force[:, ch], label=f"{name} total")
        ax_max.plot(time_steps, max_force[:, ch], label=f"{name} max")
        ax_active.plot(time_steps, active_cells[:, ch], label=f"{name} active cells")

    ax_total.set_title("Total Contact")
    ax_total.set_xlabel("Time (s)")
    ax_total.grid(True, linestyle=":", alpha=0.55)
    ax_total.legend()

    ax_max.set_title("Max Cell Value")
    ax_max.set_xlabel("Time (s)")
    ax_max.grid(True, linestyle=":", alpha=0.55)
    ax_max.legend()

    ax_active.set_title("Active Cells")
    ax_active.set_xlabel("Time (s)")
    ax_active.grid(True, linestyle=":", alpha=0.55)
    ax_active.legend()

    for ch in range(num_channels):
        ax = fig.add_subplot(gs[2, ch])
        name = channel_names[ch] if ch < len(channel_names) else f"ch{ch}"
        im = ax.imshow(max_map[ch], cmap="turbo", vmin=0.0, vmax=force_max, aspect="auto")
        ax.set_title(f"{name} max heatmap")
        ax.set_xlabel("col")
        ax.set_ylabel("row")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"✅ 触觉统计图已保存: {save_path}")


def render_camera_video(images, fps, save_path, scale=1.0):
    """
    将图像序列渲染为 MP4 视频
    :param images: ndarray of shape (T, H, W, C), RGB uint8
    :param fps: 视频帧率
    :param save_path: 输出视频路径
    :param scale: 缩放比例（用于降低分辨率，节省空间）
    """
    T, H, W, C = images.shape
    if scale != 1.0:
        new_w, new_h = int(W * scale), int(H * scale)
        out_size = (new_w, new_h)
    else:
        out_size = (W, H)

    # 尝试多种编码器
    codecs = ["mp4v", "avc1", "X264"]
    writer = None
    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        temp_writer = cv2.VideoWriter(save_path, fourcc, fps, out_size)
        if temp_writer.isOpened():
            writer = temp_writer
            break
        temp_writer.release()

    if writer is None:
        raise RuntimeError(f"无法创建视频写入器，请检查 OpenCV 编码器支持。尝试保存到: {save_path}")

    for i in tqdm(range(T), desc=f"🎞️ 渲染视频", unit="帧"):
        frame = images[i]
        if scale != 1.0:
            frame = cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    print(f"🎉 视频导出完成: {save_path}")


def main():
    args = parse_args()

    # 确定输出目录
    base_dir = os.path.dirname(args.path) if args.output is None else args.output
    os.makedirs(base_dir, exist_ok=True)
    file_stem = os.path.splitext(os.path.basename(args.path))[0]

    # 加载数据
    try:
        time_steps, action, qpos, qvel, images_dict, imu, tactile, tactile_channel_names, fps = load_data(args.path)
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        sys.exit(1)

    # 覆盖帧率（如果用户指定）
    if args.fps is not None:
        fps = args.fps

    time_steps, action, qpos, qvel = align_to_min_length(time_steps, action, qpos, qvel)
    if imu is not None:
        _, imu = align_to_min_length(time_steps, imu)
    if tactile is not None:
        _, tactile = align_to_min_length(time_steps, tactile)

    # ---------- 绘制曲线 ----------
    arm_save = os.path.join(base_dir, f"{file_stem}_arm_tracking.png")
    gripper_save = os.path.join(base_dir, f"{file_stem}_gripper_tracking.png")
    imu_save = os.path.join(base_dir, f"{file_stem}_imu_summary.png")
    tactile_save = os.path.join(base_dir, f"{file_stem}_tactile_summary.png")
    tactile_video_save = os.path.join(base_dir, f"{file_stem}_tactile.mp4")
    print("\n📈 正在生成关节跟踪曲线图...")
    plot_arm_tracking(time_steps, action, qpos, arm_save)
    if action.shape[1] > 6 and qpos.shape[1] > 6:
        plot_gripper_tracking(time_steps, action, qpos, gripper_save)
    else:
        print("⚠️ action/qpos 不包含第7维夹爪数据，跳过夹爪曲线。")

    print("\n🧭 正在生成 IMU 曲线图...")
    plot_imu_summary(time_steps, imu, imu_save)

    print("\n🖐️ 正在生成触觉可视化...")
    if tactile is not None:
        tactile_force_max = choose_tactile_force_max(tactile, args.tactile_force_max)
        print(f"  触觉热力图上限: {tactile_force_max:.6g}")
        plot_tactile_summary(time_steps, tactile, tactile_save, tactile_channel_names, tactile_force_max)
        if not args.skip_tactile_video:
            try:
                save_tactile_video(
                    tactile,
                    fps,
                    tactile_video_save,
                    tactile_channel_names,
                    tactile_force_max,
                    width=args.tactile_video_width,
                    height=args.tactile_video_height,
                )
            except Exception as e:
                print(f"❌ 触觉视频渲染失败: {e}")
    else:
        print("⚠️ 未找到触觉数据，跳过触觉可视化。")

    # ---------- 渲染相机视频 ----------
    print("\n🎥 开始导出相机视频...")
    for cam_name in args.cams:
        if cam_name not in images_dict:
            print(f"⚠️ 警告: 数据集不包含相机 '{cam_name}'，跳过。")
            continue

        cam_images = images_dict[cam_name]
        if cam_images.ndim != 4 or cam_images.shape[-1] not in [1, 3]:
            print(f"⚠️ 相机 '{cam_name}' 图像格式异常 (shape={cam_images.shape})，跳过。")
            continue

        # 如果是单通道灰度图，转为三通道以便合成彩色视频
        if cam_images.shape[-1] == 1:
            cam_images = np.repeat(cam_images, 3, axis=-1)

        video_path = os.path.join(base_dir, f"{file_stem}_{cam_name}.mp4")
        print(f"\n--- 处理相机: {cam_name} ({cam_images.shape[0]} 帧) ---")
        try:
            render_camera_video(cam_images, fps, video_path, scale=1.0)
        except Exception as e:
            print(f"❌ 视频渲染失败 ({cam_name}): {e}")

    print("\n✨ 所有分析任务完成！")


if __name__ == "__main__":
    main()

# python visualize_episodes.py --path /media/hjx/PSSD/hjx_ws/data/rebot/data_real_tactile/rebot_real_grasp_banana/episode_0.hdf5 --cams cam_high cam_wrist --tactile_video_width 840 --tactile_video_height 336
