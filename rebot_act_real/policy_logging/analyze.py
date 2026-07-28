"""Generate publication-ready diagnostics from one recorded policy run."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

# Analysis is a batch job. Never initialize Qt/Tk even when the environment or
# OpenCV installation advertises an interactive Matplotlib backend.
os.environ["MPLBACKEND"] = "Agg"
_MPL_CACHE = Path(tempfile.gettempdir()) / f"rebot_act_real_mpl_{os.getuid()}"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_MPL_CACHE)


def load_run(run_dir: Path) -> tuple[dict, dict[str, np.ndarray]]:
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    chunks = sorted((run_dir / "data").glob("chunk_*.npz"))
    if not chunks:
        raise FileNotFoundError(f"没有找到日志分块: {run_dir / 'data'}")
    columns: dict[str, list[np.ndarray]] = {}
    for path in chunks:
        with np.load(path, allow_pickle=False) as chunk:
            for key in chunk.files:
                columns.setdefault(key, []).append(chunk[key])
    return metadata, {key: np.concatenate(value) for key, value in columns.items()}


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.15,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _select_keyframes(data: dict[str, np.ndarray], count: int = 6) -> list[dict]:
    """Select distinct, interpretable events and return them in time order."""
    size = len(data["time_s"])
    if size == 0:
        return []
    count = max(2, min(int(count), size))
    candidates: list[tuple[int, str, float | None]] = [
        (0, "Start", None),
        (size - 1, "End", None),
    ]
    if size > 1:
        action_delta = np.linalg.norm(
            np.diff(data["safe_action"][:, :6], axis=0), axis=1
        )
        index = int(np.argmax(action_delta)) + 1
        candidates.append((index, "Max arm action change", float(action_delta[index - 1])))
        gripper_delta = np.abs(np.diff(data["safe_action"][:, 6]))
        index = int(np.argmax(gripper_delta)) + 1
        candidates.append((index, "Gripper transition", float(gripper_delta[index - 1])))
    tracking = np.max(
        np.abs(data["safe_action"][:, :6] - data["joint_position"]), axis=1
    )
    index = int(np.argmax(tracking))
    candidates.append((index, "Max tracking error", float(tracking[index])))
    if "tactile" in data:
        tactile_peak = np.max(data["tactile"], axis=(1, 2))
        index = int(np.argmax(tactile_peak))
        candidates.append((index, "Peak tactile response", float(tactile_peak[index])))
    else:
        latency = data["inference_ms"]
        index = int(np.argmax(latency))
        candidates.append((index, "Max inference latency", float(latency[index])))

    # Keep event frames distinct. Uniform frames fill gaps when two metrics peak
    # at nearly the same instant.
    minimum_gap = max(size // 40, 1)
    selected: list[tuple[int, str, float | None]] = []
    for candidate in candidates:
        if all(abs(candidate[0] - existing[0]) >= minimum_gap for existing in selected):
            selected.append(candidate)
        if len(selected) == count:
            break
    for index in np.linspace(0, size - 1, count, dtype=int):
        if len(selected) == count:
            break
        if all(abs(int(index) - existing[0]) >= minimum_gap for existing in selected):
            selected.append((int(index), "Task progression", None))
    if len(selected) < count:
        for index in range(size):
            if len(selected) == count:
                break
            if index not in {item[0] for item in selected}:
                selected.append((index, "Task progression", None))

    events = []
    for index, label, value in sorted(selected, key=lambda item: item[0]):
        events.append(
            {
                "sample_index": int(index),
                "step": int(data["step"][index]) if "step" in data else int(index),
                "time_s": float(data["time_s"][index]),
                "event": label,
                "event_value": value,
            }
        )
    return events


def _chunk_lengths(run_dir: Path) -> list[int]:
    lengths = []
    for path in sorted((run_dir / "data").glob("chunk_*.npz")):
        with np.load(path, allow_pickle=False) as chunk:
            first = chunk[chunk.files[0]]
            lengths.append(int(first.shape[0]))
    return lengths


def _read_segment_frame(
    run_dir: Path,
    camera_name: str,
    global_index: int,
    chunk_lengths: list[int],
) -> np.ndarray | None:
    import cv2

    offset = 0
    for chunk_index, length in enumerate(chunk_lengths):
        if global_index < offset + length:
            local_index = global_index - offset
            path = run_dir / "videos" / f"{camera_name}_{chunk_index:06d}.mp4"
            if not path.is_file():
                return None
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                return None
            capture.set(cv2.CAP_PROP_POS_FRAMES, local_index)
            ok, bgr = capture.read()
            capture.release()
            if not ok:
                return None
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        offset += length
    return None


def _generate_keyframe_montage(
    run_dir: Path,
    output: Path,
    data: dict[str, np.ndarray],
    plt,
) -> list[dict]:
    video_dir = run_dir / "videos"
    if not video_dir.is_dir():
        return []
    events = _select_keyframes(data)
    lengths = _chunk_lengths(run_dir)
    frames = {
        camera: [
            _read_segment_frame(
                run_dir, camera, event["sample_index"], lengths
            )
            for event in events
        ]
        for camera in ("cam_high", "cam_wrist")
    }
    if not events or not any(
        frame is not None for camera_frames in frames.values() for frame in camera_frames
    ):
        return []

    columns = len(events)
    fig, axes = plt.subplots(
        2, columns, figsize=(7.1, 3.2), squeeze=False,
        gridspec_kw={"wspace": 0.025, "hspace": 0.08},
    )
    for column, event in enumerate(events):
        for row, camera in enumerate(("cam_high", "cam_wrist")):
            ax = axes[row, column]
            image = frames[camera][column]
            if image is None:
                ax.set_facecolor("0.9")
                ax.text(0.5, 0.5, "Frame unavailable", ha="center", va="center")
            else:
                ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("0.25")
                spine.set_linewidth(0.45)
        axes[0, column].set_title(
            f"Step {event['step']} | {event['time_s']:.2f} s",
            fontsize=6.5,
            pad=3,
        )
    fig.savefig(output / "keyframes.png")
    plt.close(fig)
    (output / "keyframes.json").write_text(
        json.dumps(events, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return events


def _generate_overview(
    run_dir: Path,
    output: Path,
    metadata: dict,
    data: dict[str, np.ndarray],
    events: list[dict],
    plt,
) -> None:
    """Build one adaptive, camera-to-control multimodal summary figure."""
    from matplotlib.lines import Line2D

    t = data["time_s"]
    colors = plt.get_cmap("tab10").colors
    has_video = bool(events) and (run_dir / "videos").is_dir()
    has_imu = "imu" in data
    has_tactile = "tactile" in data
    sensor_rows = int(has_imu or has_tactile)
    image_rows = 1 if has_video else 0
    total_rows = image_rows + 2 + sensor_rows
    height_ratios = ([2.15] if has_video else []) + [1.35, 1.15]
    if sensor_rows:
        height_ratios.append(1.25)
    fig = plt.figure(figsize=(12.0, 2.75 * total_rows))
    fig.subplots_adjust(
        left=0.065,
        right=0.965,
        bottom=0.055,
        top=0.965,
        hspace=0.52,
        wspace=0.62,
    )
    grid = fig.add_gridspec(
        total_rows,
        12,
        height_ratios=height_ratios,
        hspace=0.52,
        wspace=0.62,
    )
    row = 0

    if has_video:
        overview_events = events
        if len(events) > 5:
            positions = np.linspace(0, len(events) - 1, 5, dtype=int)
            overview_events = [events[int(index)] for index in positions]
        lengths = _chunk_lengths(run_dir)
        image_grid = grid[0, :].subgridspec(
            2, len(overview_events), wspace=0.025, hspace=0.035
        )
        for column, event in enumerate(overview_events):
            for camera_row, camera in enumerate(("cam_high", "cam_wrist")):
                ax = fig.add_subplot(image_grid[camera_row, column])
                image = _read_segment_frame(
                    run_dir, camera, event["sample_index"], lengths
                )
                if image is not None:
                    ax.imshow(image)
                else:
                    ax.set_facecolor("0.92")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(0.55)
                    spine.set_color("0.25")
                if camera_row == 0:
                    ax.set_title(
                        f"Step {event['step']} | {event['time_s']:.2f} s",
                        fontsize=7.2,
                        pad=2.5,
                    )
        row = 1

    ax_joint = fig.add_subplot(grid[row, :7])
    for joint in range(6):
        ax_joint.plot(
            t, data["joint_position"][:, joint],
            color=colors[joint], linewidth=1.15, label=f"J{joint + 1}",
        )
        ax_joint.plot(
            t, data["safe_action"][:, joint],
            color=colors[joint], linewidth=0.8, linestyle="--", alpha=0.75,
        )
    ax_joint.set_title("Joint-space policy execution", loc="left", fontweight="semibold")
    ax_joint.set_ylabel("Position (rad)")
    joint_legend = ax_joint.legend(
        ncol=6, frameon=False, loc="upper center", columnspacing=1.0,
        handlelength=1.6,
    )
    ax_joint.add_artist(joint_legend)
    ax_joint.legend(
        handles=[
            Line2D([0], [0], color="0.25", label="Feedback"),
            Line2D([0], [0], color="0.25", linestyle="--", label="Command"),
        ],
        frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.82),
    )

    ax_error = fig.add_subplot(grid[row, 7:])
    tracking = data["safe_action"][:, :6] - data["joint_position"]
    for joint in range(6):
        ax_error.plot(t, tracking[:, joint], color=colors[joint])
    ax_error.axhline(0, color="0.2", linewidth=0.65)
    ax_error.set_title("Closed-loop tracking error", loc="left", fontweight="semibold")
    ax_error.set_ylabel("Error (rad)")
    row += 1

    ax_timing = fig.add_subplot(grid[row, :5])
    nominal_ms = 1000.0 / float(metadata["nominal_rate_hz"])
    ax_timing.plot(t, data["inference_ms"], color="#0072B2", label="Inference")
    ax_timing.plot(
        t, data["loop_dt_s"] * 1000.0, color="#009E73", label="Control period"
    )
    ax_timing.axhline(
        nominal_ms, color="#D55E00", linestyle="--", linewidth=1.0,
        label="Nominal budget",
    )
    ax_timing.set_title("Real-time performance", loc="left", fontweight="semibold")
    ax_timing.set_ylabel("Latency (ms)")
    ax_timing.set_xlabel("Time (s)")
    ax_timing.legend(frameon=False, ncol=3)

    ax_gripper = fig.add_subplot(grid[row, 5:8])
    ax_gripper.plot(t, data["raw_action"][:, 6], color="#0072B2", label="Policy")
    ax_gripper.plot(t, data["safe_action"][:, 6], color="#D55E00", label="Command")
    ax_gripper.set_title("Gripper action", loc="left", fontweight="semibold")
    ax_gripper.set_ylabel("Position (rad)")
    ax_gripper.set_xlabel("Time (s)")
    ax_gripper.legend(frameon=False)

    ax_age = fig.add_subplot(grid[row, 8:])
    ax_age.plot(t, data["action_age_ms"], color="#CC79A7", label="Action age")
    if "control_hz" in data:
        ax_rate = ax_age.twinx()
        ax_rate.plot(t, data["control_hz"], color="#56B4E9", alpha=0.75)
        ax_rate.set_ylabel("Control rate (Hz)", color="#0072B2")
        ax_rate.tick_params(axis="y", colors="#0072B2")
    ax_age.set_title("Observation-to-action freshness", loc="left", fontweight="semibold")
    ax_age.set_ylabel("Action age (ms)")
    ax_age.set_xlabel("Time (s)")
    row += 1

    if sensor_rows:
        if has_imu and has_tactile:
            imu_slice, tactile_slice = slice(0, 6), slice(6, 12)
        elif has_imu:
            imu_slice, tactile_slice = slice(0, 12), None
        else:
            imu_slice, tactile_slice = None, slice(0, 12)
        if has_imu and imu_slice is not None:
            ax_imu = fig.add_subplot(grid[row, imu_slice])
            imu = data["imu"]
            gyro_norm = np.linalg.norm(imu[:, 4:7], axis=1)
            accel_norm = np.linalg.norm(imu[:, 7:10], axis=1)
            ax_imu.plot(t, gyro_norm, color="#0072B2", label=r"$\|\omega\|$")
            ax_imu.plot(t, accel_norm, color="#E69F00", label=r"$\|a\|$")
            ax_imu.set_title("IMU dynamics", loc="left", fontweight="semibold")
            ax_imu.set_ylabel("Sensor magnitude")
            ax_imu.set_xlabel("Time (s)")
            ax_imu.legend(frameon=False, ncol=2)
        if has_tactile and tactile_slice is not None:
            tactile_grid = grid[row, tactile_slice].subgridspec(
                1, 5, wspace=0.35
            )
            ax_touch = fig.add_subplot(tactile_grid[0, :3])
            tactile = data["tactile"]
            touch_mean = tactile.mean(axis=(1, 2))
            touch_max = tactile.max(axis=(1, 2))
            ax_touch.plot(t, touch_mean, color="#009E73", label="Mean")
            ax_touch.plot(t, touch_max, color="#D55E00", label="Maximum")
            ax_touch.set_title("Tactile response", loc="left", fontweight="semibold")
            ax_touch.set_ylabel("Response")
            ax_touch.set_xlabel("Time (s)")
            ax_touch.legend(frameon=False, ncol=2)
            ax_heat = fig.add_subplot(tactile_grid[0, 3:])
            peak_index = int(np.argmax(touch_max))
            image = ax_heat.imshow(
                tactile[peak_index], cmap="magma", aspect="auto", interpolation="nearest"
            )
            ax_heat.set_title(
                f"Peak map | {t[peak_index]:.2f} s", loc="left",
                fontweight="semibold",
            )
            ax_heat.set_xlabel("Taxel column")
            ax_heat.set_ylabel("Taxel row")
            fig.colorbar(image, ax=ax_heat, fraction=0.055, pad=0.03)

    for ax in fig.axes:
        if hasattr(ax, "grid") and ax.get_images() == []:
            ax.grid(True, color="0.91", linewidth=0.55)
        if hasattr(ax, "spines") and not ax.get_images():
            ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "overview.png", dpi=300, facecolor="white")
    plt.close(fig)


def _tactile_statistics(
    tactile: np.ndarray, time_s: np.ndarray
) -> tuple[dict[str, np.ndarray | float | int], dict[str, float | int | list]]:
    tactile = np.asarray(tactile, dtype=np.float64)
    peak = tactile.max(axis=(1, 2))
    mean = tactile.mean(axis=(1, 2))
    total = tactile.sum(axis=(1, 2))
    global_peak = float(np.max(peak))
    positive = tactile[tactile > 0]
    noise_floor = float(np.percentile(positive, 20)) if positive.size else 0.0
    threshold = min(
        max(0.08 * global_peak, 2.0 * noise_floor, 1e-6),
        max(0.5 * global_peak, 1e-6),
    )
    active = tactile >= threshold
    area = active.mean(axis=(1, 2))
    rows = np.arange(tactile.shape[1], dtype=np.float64)[None, :, None]
    cols = np.arange(tactile.shape[2], dtype=np.float64)[None, None, :]
    valid = total > max(threshold, 1e-12)
    cop_row = np.full(tactile.shape[0], np.nan)
    cop_col = np.full(tactile.shape[0], np.nan)
    cop_row[valid] = (tactile * rows).sum(axis=(1, 2))[valid] / total[valid]
    cop_col[valid] = (tactile * cols).sum(axis=(1, 2))[valid] / total[valid]
    contact_frames = np.flatnonzero(peak >= threshold)
    peak_index = int(np.argmax(peak))
    if contact_frames.size:
        onset_index = int(contact_frames[0])
        release_index = int(contact_frames[-1])
    else:
        onset_index = release_index = peak_index
    path_length = 0.0
    valid_points = np.column_stack([cop_col[valid], cop_row[valid]])
    if len(valid_points) > 1:
        path_length = float(np.linalg.norm(np.diff(valid_points, axis=0), axis=1).sum())
    dt = float(np.median(np.diff(time_s))) if len(time_s) > 1 else 0.0
    summary = {
        "threshold": float(threshold),
        "peak": global_peak,
        "mean": float(np.mean(mean)),
        "peak_time_s": float(time_s[peak_index]),
        "peak_contact_area_fraction": float(area[peak_index]),
        "maximum_contact_area_fraction": float(np.max(area)),
        "active_duration_s": float(contact_frames.size * dt),
        "contact_center_path_taxels": path_length,
        "peak_center_row_col": [
            float(cop_row[peak_index]) if np.isfinite(cop_row[peak_index]) else None,
            float(cop_col[peak_index]) if np.isfinite(cop_col[peak_index]) else None,
        ],
    }
    arrays: dict[str, np.ndarray | float | int] = {
        "peak": peak,
        "mean": mean,
        "total": total,
        "area": area,
        "cop_row": cop_row,
        "cop_col": cop_col,
        "threshold": threshold,
        "onset_index": onset_index,
        "peak_index": peak_index,
        "release_index": release_index,
    }
    return arrays, summary


def _generate_tactile_analysis(
    run_dir: Path,
    output: Path,
    data: dict[str, np.ndarray],
    plt,
) -> dict[str, float | int | list]:
    """Create a spatial, temporal, and 3-D tactile analysis plate."""
    tactile = np.asarray(data["tactile"], dtype=np.float64)
    t = np.asarray(data["time_s"], dtype=np.float64)
    stats, summary = _tactile_statistics(tactile, t)
    peak = np.asarray(stats["peak"])
    mean = np.asarray(stats["mean"])
    area = np.asarray(stats["area"])
    cop_row = np.asarray(stats["cop_row"])
    cop_col = np.asarray(stats["cop_col"])
    threshold = float(stats["threshold"])
    onset_index = int(stats["onset_index"])
    peak_index = int(stats["peak_index"])
    release_index = int(stats["release_index"])

    snapshot_candidates = [0, onset_index, peak_index, release_index]
    snapshots: list[int] = []
    for index in snapshot_candidates:
        if index not in snapshots:
            snapshots.append(index)
    for index in np.linspace(0, len(t) - 1, 4, dtype=int):
        if len(snapshots) >= 4:
            break
        if int(index) not in snapshots:
            snapshots.append(int(index))
    snapshots = sorted(snapshots[:4])

    has_video = (run_dir / "videos").is_dir()
    rows = 4 if has_video else 3
    height_ratios = [1.05, 1.0, 1.3] + ([1.0] if has_video else [])
    fig = plt.figure(figsize=(12.0, 3.0 * rows))
    fig.subplots_adjust(
        left=0.07, right=0.965, bottom=0.055, top=0.96,
        hspace=0.62, wspace=0.62,
    )
    grid = fig.add_gridspec(
        rows, 12, height_ratios=height_ratios, hspace=0.62, wspace=0.62
    )

    ax_signal = fig.add_subplot(grid[0, :8])
    ax_signal.plot(t, peak, color="#D55E00", label="Maximum", linewidth=1.35)
    ax_signal.plot(t, mean, color="#009E73", label="Mean", linewidth=1.15)
    ax_signal.fill_between(
        t, 0, peak, where=peak >= threshold, color="#E69F00", alpha=0.12,
        label="Detected contact",
    )
    ax_signal.axhline(
        threshold, color="0.35", linestyle="--", linewidth=0.8,
        label=f"Threshold ({threshold:.3g})",
    )
    ax_signal.axvline(t[peak_index], color="#CC79A7", linestyle=":", linewidth=1.0)
    ax_signal.set_title("Contact intensity over time", loc="left", fontweight="semibold")
    ax_signal.set_ylabel("Tactile response")
    ax_signal.set_xlabel("Time (s)")
    ax_signal.legend(frameon=False, ncol=4, loc="upper right")

    ax_contact = fig.add_subplot(grid[0, 8:])
    ax_contact.plot(t, area * 100.0, color="#0072B2", label="Active taxels")
    ax_contact.set_title("Contact geometry", loc="left", fontweight="semibold")
    ax_contact.set_ylabel("Active area (%)")
    ax_contact.set_xlabel("Time (s)")
    ax_cop = ax_contact.twinx()
    ax_cop.plot(t, cop_col, color="#CC79A7", alpha=0.8, label="CoP column")
    ax_cop.plot(t, cop_row, color="#56B4E9", alpha=0.8, label="CoP row")
    ax_cop.set_ylabel("Center of pressure (taxel)")
    handles_a, labels_a = ax_contact.get_legend_handles_labels()
    handles_b, labels_b = ax_cop.get_legend_handles_labels()
    ax_contact.legend(
        handles_a + handles_b, labels_a + labels_b, frameon=False, ncol=3,
        loc="upper right",
    )

    snapshots_grid = grid[1, :].subgridspec(1, 4, wspace=0.25)
    vmax = max(float(np.max(tactile)), 1e-6)
    for column, index in enumerate(snapshots):
        ax = fig.add_subplot(snapshots_grid[0, column])
        image = ax.imshow(
            tactile[index], cmap="magma", vmin=0, vmax=vmax,
            aspect="auto", interpolation="nearest",
        )
        if np.isfinite(cop_col[index]) and np.isfinite(cop_row[index]):
            ax.scatter(
                cop_col[index], cop_row[index], marker="+", s=65,
                linewidths=1.2, color="white",
            )
        step = int(data["step"][index]) if "step" in data else index
        ax.set_title(f"Step {step} | {t[index]:.2f} s", fontsize=8)
        ax.set_xlabel("Taxel column")
        if column == 0:
            ax.set_ylabel("Taxel row")
        else:
            ax.set_yticklabels([])
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)

    ax_surface = fig.add_subplot(grid[2, :5], projection="3d")
    columns, row_values = np.meshgrid(
        np.arange(tactile.shape[2]), np.arange(tactile.shape[1])
    )
    surface = ax_surface.plot_surface(
        columns, row_values, tactile[peak_index],
        cmap="magma", vmin=0, vmax=vmax, linewidth=0,
        antialiased=True, rcount=tactile.shape[1], ccount=tactile.shape[2],
    )
    ax_surface.set_title(
        f"Peak contact surface | {t[peak_index]:.2f} s",
        loc="left", fontweight="semibold",
    )
    ax_surface.set_xlabel("Column")
    ax_surface.set_ylabel("Row")
    ax_surface.set_zlabel("Response")
    ax_surface.view_init(elev=32, azim=-58)
    fig.colorbar(surface, ax=ax_surface, shrink=0.62, pad=0.08)

    ax_space_time = fig.add_subplot(grid[2, 5:9])
    column_activity = tactile.mean(axis=1).T
    extent = [float(t[0]), float(t[-1]), 0, tactile.shape[2] - 1]
    temporal_image = ax_space_time.imshow(
        column_activity, origin="lower", aspect="auto", cmap="viridis",
        extent=extent, interpolation="nearest",
    )
    ax_space_time.set_title(
        "Spatiotemporal contact evolution", loc="left", fontweight="semibold"
    )
    ax_space_time.set_xlabel("Time (s)")
    ax_space_time.set_ylabel("Taxel column")
    fig.colorbar(temporal_image, ax=ax_space_time, fraction=0.05, pad=0.03)

    ax_path = fig.add_subplot(grid[2, 9:])
    occupancy = (tactile >= threshold).mean(axis=0)
    ax_path.imshow(
        occupancy, cmap="Greys", origin="upper", aspect="auto",
        vmin=0, vmax=max(float(np.max(occupancy)), 1e-6),
    )
    valid = np.isfinite(cop_col) & np.isfinite(cop_row)
    if np.any(valid):
        path = ax_path.scatter(
            cop_col[valid], cop_row[valid], c=t[valid], cmap="plasma",
            s=9, alpha=0.8, edgecolors="none",
        )
        ax_path.plot(cop_col[valid], cop_row[valid], color="white", alpha=0.3, linewidth=0.5)
        fig.colorbar(path, ax=ax_path, fraction=0.055, pad=0.035, label="Time (s)")
    ax_path.set_title("Contact-center trajectory", loc="left", fontweight="semibold")
    ax_path.set_xlabel("Taxel column")
    ax_path.set_ylabel("Taxel row")

    if has_video:
        lengths = _chunk_lengths(run_dir)
        camera_grid = grid[3, :].subgridspec(1, 2, wspace=0.06)
        for column, camera in enumerate(("cam_high", "cam_wrist")):
            ax = fig.add_subplot(camera_grid[0, column])
            frame = _read_segment_frame(
                run_dir, camera, peak_index, lengths
            )
            if frame is not None:
                ax.imshow(frame)
            else:
                ax.set_facecolor("0.92")
            ax.set_xticks([])
            ax.set_yticks([])
            step = int(data["step"][peak_index]) if "step" in data else peak_index
            ax.set_title(
                f"Peak tactile synchronized RGB | Step {step} | {t[peak_index]:.2f} s",
                fontsize=8,
            )
            for spine in ax.spines.values():
                spine.set_linewidth(0.55)
                spine.set_color("0.25")

    for ax in fig.axes:
        if not getattr(ax, "name", "") == "3d" and not ax.get_images():
            ax.grid(True, color="0.91", linewidth=0.55)
        if hasattr(ax, "spines") and not ax.get_images():
            ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "tactile_analysis.png", dpi=300, facecolor="white")
    plt.close(fig)
    return summary


def _quaternion_kinematics(
    quaternion_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(quaternion_wxyz, dtype=np.float64).copy()
    norms = np.linalg.norm(q, axis=1)
    q /= np.maximum(norms[:, None], 1e-12)
    w, x, y, z = q.T
    rotation = np.empty((len(q), 3, 3), dtype=np.float64)
    rotation[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotation[:, 0, 1] = 2 * (x * y - z * w)
    rotation[:, 0, 2] = 2 * (x * z + y * w)
    rotation[:, 1, 0] = 2 * (x * y + z * w)
    rotation[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotation[:, 1, 2] = 2 * (y * z - x * w)
    rotation[:, 2, 0] = 2 * (x * z - y * w)
    rotation[:, 2, 1] = 2 * (y * z + x * w)
    rotation[:, 2, 2] = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(rotation[:, 2, 1], rotation[:, 2, 2])
    pitch = np.arcsin(np.clip(-rotation[:, 2, 0], -1.0, 1.0))
    yaw = np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0])
    euler_deg = np.rad2deg(np.column_stack([roll, pitch, yaw]))
    return rotation, euler_deg, norms


def _generate_imu_analysis(
    run_dir: Path,
    output: Path,
    data: dict[str, np.ndarray],
    plt,
) -> dict[str, float | list]:
    """Create a 3-D attitude, dynamics, and frequency-domain IMU plate."""
    imu = np.asarray(data["imu"], dtype=np.float64)
    t = np.asarray(data["time_s"], dtype=np.float64)
    rotation, euler_deg, quaternion_norm = _quaternion_kinematics(imu[:, :4])
    gyro = imu[:, 4:7]
    accel = imu[:, 7:10]
    gyro_norm = np.linalg.norm(gyro, axis=1)
    accel_norm = np.linalg.norm(accel, axis=1)
    peak_index = int(np.argmax(gyro_norm))
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    sample_rate = 1.0 / max(dt, 1e-9)
    frequencies = np.fft.rfftfreq(len(t), d=dt)
    gyro_spectrum = np.abs(
        np.fft.rfft(gyro - gyro.mean(axis=0, keepdims=True), axis=0)
    ) ** 2
    if len(frequencies) > 1:
        dominant_index = int(np.argmax(gyro_spectrum[1:].sum(axis=1))) + 1
        dominant_frequency = float(frequencies[dominant_index])
    else:
        dominant_frequency = 0.0
    relative_rotation = np.einsum("ij,njk->nik", rotation[0].T, rotation)
    trace = np.trace(relative_rotation, axis1=1, axis2=2)
    orientation_change = np.rad2deg(
        np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    )
    summary: dict[str, float | list] = {
        "quaternion_norm_mean_abs_error": float(np.mean(np.abs(quaternion_norm - 1.0))),
        "peak_angular_rate_rad_s": float(np.max(gyro_norm)),
        "peak_angular_rate_time_s": float(t[peak_index]),
        "rms_angular_rate_rad_s": float(np.sqrt(np.mean(gyro_norm**2))),
        "peak_acceleration_g": float(np.max(accel_norm)),
        "maximum_acceleration_deviation_from_1g": float(
            np.max(np.abs(accel_norm - 1.0))
        ),
        "maximum_orientation_change_deg": float(np.max(orientation_change)),
        "dominant_angular_rate_frequency_hz": dominant_frequency,
        "initial_euler_deg": euler_deg[0].tolist(),
        "final_euler_deg": euler_deg[-1].tolist(),
    }

    has_video = (run_dir / "videos").is_dir()
    rows = 4 if has_video else 3
    fig = plt.figure(figsize=(12.0, 3.05 * rows))
    fig.subplots_adjust(
        left=0.07, right=0.965, bottom=0.055, top=0.96,
        hspace=0.62, wspace=0.68,
    )
    grid = fig.add_gridspec(
        rows, 12, height_ratios=[1.2, 1.0, 1.15] + ([1.0] if has_video else []),
        hspace=0.62, wspace=0.68,
    )

    ax_sphere = fig.add_subplot(grid[0, :5], projection="3d")
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 16)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones_like(u), np.cos(v))
    ax_sphere.plot_wireframe(
        sphere_x, sphere_y, sphere_z, color="0.82", linewidth=0.35, alpha=0.55
    )
    body_z = rotation[:, :, 2]
    stride = max(len(t) // 180, 1)
    path = ax_sphere.scatter(
        body_z[::stride, 0], body_z[::stride, 1], body_z[::stride, 2],
        c=t[::stride], cmap="plasma", s=10, depthshade=False,
    )
    ax_sphere.plot(
        body_z[:, 0], body_z[:, 1], body_z[:, 2],
        color="#0072B2", linewidth=0.8, alpha=0.65,
    )
    ax_sphere.scatter(*body_z[0], color="#009E73", s=35, marker="o", label="Start")
    ax_sphere.scatter(*body_z[-1], color="#D55E00", s=40, marker="^", label="End")
    ax_sphere.set_title(
        "3-D orientation trajectory on unit sphere", loc="left",
        fontweight="semibold",
    )
    ax_sphere.set_xlabel("World X")
    ax_sphere.set_ylabel("World Y")
    ax_sphere.set_zlabel("World Z")
    ax_sphere.set_box_aspect((1, 1, 1))
    ax_sphere.view_init(elev=24, azim=-48)
    ax_sphere.legend(frameon=False, loc="upper left")
    fig.colorbar(path, ax=ax_sphere, shrink=0.62, pad=0.08, label="Time (s)")

    ax_euler = fig.add_subplot(grid[0, 5:])
    for axis, label, color in zip(
        range(4), ("$q_w$", "$q_x$", "$q_y$", "$q_z$"),
        ("#0072B2", "#D55E00", "#009E73", "#E69F00"),
    ):
        ax_euler.plot(t, imu[:, axis], label=label, color=color)
    ax_rotation = ax_euler.twinx()
    ax_rotation.plot(
        t, orientation_change, color="#CC79A7", linestyle="--",
        label="Relative rotation",
    )
    ax_euler.set_title(
        "Quaternion and relative orientation", loc="left", fontweight="semibold"
    )
    ax_euler.set_ylabel("Quaternion component")
    ax_rotation.set_ylabel("Rotation from initial (deg)", color="#A64D79")
    ax_rotation.tick_params(axis="y", colors="#A64D79")
    ax_euler.set_xlabel("Time (s)")
    handles_a, labels_a = ax_euler.get_legend_handles_labels()
    handles_b, labels_b = ax_rotation.get_legend_handles_labels()
    ax_euler.legend(
        handles_a + handles_b, labels_a + labels_b, frameon=False, ncol=5
    )

    ax_gyro = fig.add_subplot(grid[1, :6])
    for axis, label, color in zip(
        range(3), (r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"),
        ("#0072B2", "#D55E00", "#009E73"),
    ):
        ax_gyro.plot(t, gyro[:, axis], label=label, color=color, alpha=0.9)
    ax_gyro.plot(t, gyro_norm, color="0.15", linewidth=1.15, label=r"$\|\omega\|$")
    ax_gyro.axvline(t[peak_index], color="#CC79A7", linestyle=":", linewidth=1.0)
    ax_gyro.set_title("Angular-rate dynamics", loc="left", fontweight="semibold")
    ax_gyro.set_ylabel("Angular rate (rad/s)")
    ax_gyro.set_xlabel("Time (s)")
    ax_gyro.legend(frameon=False, ncol=4)

    ax_accel = fig.add_subplot(grid[1, 6:])
    for axis, label, color in zip(
        range(3), ("$a_x$", "$a_y$", "$a_z$"),
        ("#0072B2", "#D55E00", "#009E73"),
    ):
        ax_accel.plot(t, accel[:, axis], label=label, color=color, alpha=0.9)
    ax_accel.plot(t, accel_norm, color="0.15", linewidth=1.15, label=r"$\|a\|$")
    ax_accel.axhline(1.0, color="0.4", linestyle="--", linewidth=0.8, label="1 g")
    ax_accel.set_title("Linear-acceleration dynamics", loc="left", fontweight="semibold")
    ax_accel.set_ylabel("Acceleration (g)")
    ax_accel.set_xlabel("Time (s)")
    ax_accel.legend(frameon=False, ncol=5)

    ax_frames = fig.add_subplot(grid[2, :5], projection="3d")
    pose_indices = np.linspace(0, len(t) - 1, 6, dtype=int)
    axis_colors = ("#D55E00", "#009E73", "#0072B2")
    for position, index in enumerate(pose_indices):
        origin = np.array([float(position), 0.0, 0.0])
        for axis in range(3):
            direction = rotation[index, :, axis] * 0.42
            ax_frames.quiver(
                *origin, *direction, color=axis_colors[axis],
                arrow_length_ratio=0.18, linewidth=1.2,
            )
        ax_frames.text(position, -0.62, -0.55, f"{t[index]:.1f}s", fontsize=6)
    ax_frames.set_title("Attitude-frame sequence", loc="left", fontweight="semibold")
    ax_frames.set_xlabel("Task progression")
    ax_frames.set_ylabel("Orientation Y")
    ax_frames.set_zlabel("Orientation Z")
    ax_frames.set_xlim(-0.3, len(pose_indices) - 0.7)
    ax_frames.set_ylim(-0.8, 0.8)
    ax_frames.set_zlim(-0.8, 0.8)
    ax_frames.view_init(elev=24, azim=-62)

    ax_spectrum = fig.add_subplot(grid[2, 5:9])
    frequency_mask = frequencies <= min(20.0, sample_rate / 2.0)
    for axis, label, color in zip(
        range(3), ("X", "Y", "Z"), ("#0072B2", "#D55E00", "#009E73")
    ):
        spectrum = gyro_spectrum[:, axis]
        normalized = spectrum / max(float(np.max(spectrum)), 1e-12)
        ax_spectrum.semilogy(
            frequencies[frequency_mask],
            np.maximum(normalized[frequency_mask], 1e-8),
            label=label, color=color,
        )
    ax_spectrum.axvline(
        dominant_frequency, color="#CC79A7", linestyle="--",
        label=f"Dominant {dominant_frequency:.2f} Hz",
    )
    ax_spectrum.set_title("Angular-rate spectrum", loc="left", fontweight="semibold")
    ax_spectrum.set_xlabel("Frequency (Hz)")
    ax_spectrum.set_ylabel("Normalized power")
    ax_spectrum.legend(frameon=False, ncol=2)

    ax_activity = fig.add_subplot(grid[2, 9:])
    angular_accel = np.gradient(gyro, t, axis=0)
    jerk = np.gradient(accel, t, axis=0)
    angular_activity = np.linalg.norm(angular_accel, axis=1)
    jerk_activity = np.linalg.norm(jerk, axis=1)
    angular_scale = max(float(np.percentile(angular_activity, 99)), 1e-12)
    jerk_scale = max(float(np.percentile(jerk_activity, 99)), 1e-12)
    ax_activity.plot(
        t, np.clip(angular_activity / angular_scale, 0, 1.5),
        color="#0072B2", label="Angular acceleration",
    )
    ax_activity.plot(
        t, np.clip(jerk_activity / jerk_scale, 0, 1.5),
        color="#E69F00", label="Linear jerk",
    )
    ax_activity.set_title("Normalized motion intensity", loc="left", fontweight="semibold")
    ax_activity.set_xlabel("Time (s)")
    ax_activity.set_ylabel("Robust normalized magnitude")
    ax_activity.legend(frameon=False)

    if has_video:
        lengths = _chunk_lengths(run_dir)
        camera_grid = grid[3, :].subgridspec(1, 2, wspace=0.06)
        for column, camera in enumerate(("cam_high", "cam_wrist")):
            ax = fig.add_subplot(camera_grid[0, column])
            frame = _read_segment_frame(run_dir, camera, peak_index, lengths)
            if frame is not None:
                ax.imshow(frame)
            else:
                ax.set_facecolor("0.92")
            ax.set_xticks([])
            ax.set_yticks([])
            step = int(data["step"][peak_index]) if "step" in data else peak_index
            ax.set_title(
                f"Peak angular-rate synchronized RGB | Step {step} | "
                f"{t[peak_index]:.2f} s",
                fontsize=8,
            )
            for spine in ax.spines.values():
                spine.set_linewidth(0.55)
                spine.set_color("0.25")

    for ax in fig.axes:
        if getattr(ax, "name", "") != "3d" and not ax.get_images():
            ax.grid(True, color="0.91", linewidth=0.55)
        if hasattr(ax, "spines") and not ax.get_images():
            ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "imu_analysis.png", dpi=300, facecolor="white")
    plt.close(fig)
    return summary


def _remove_legacy_pdfs(output: Path) -> None:
    for name in ("trajectory.pdf", "timing.pdf", "multimodal.pdf", "keyframes.pdf"):
        path = output / name
        if path.is_file():
            path.unlink()


def generate_report(run_dir: Path, output_dir: Path | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    metadata, data = load_run(run_dir)
    _style()
    output = Path(output_dir) if output_dir else Path(run_dir) / "figures"
    output.mkdir(parents=True, exist_ok=True)
    _remove_legacy_pdfs(output)
    t = data["time_s"]
    colors = plt.get_cmap("tab10").colors
    tracking_error = data["safe_action"][:, :6] - data["joint_position"]
    metrics = {
        "samples": int(t.size),
        "duration_s": float(t[-1] - t[0]) if t.size > 1 else 0.0,
        "inference_ms": {
            "mean": float(np.mean(data["inference_ms"])),
            "p50": float(np.percentile(data["inference_ms"], 50)),
            "p95": float(np.percentile(data["inference_ms"], 95)),
            "p99": float(np.percentile(data["inference_ms"], 99)),
            "max": float(np.max(data["inference_ms"])),
        },
        "control_period_ms": {
            "mean": float(np.mean(data["loop_dt_s"]) * 1000.0),
            "std": float(np.std(data["loop_dt_s"]) * 1000.0),
            "p99": float(np.percentile(data["loop_dt_s"], 99) * 1000.0),
        },
        "tracking_rmse_rad_per_joint": np.sqrt(
            np.mean(np.square(tracking_error), axis=0)
        ).tolist(),
        "tracking_max_abs_rad_per_joint": np.max(
            np.abs(tracking_error), axis=0
        ).tolist(),
        "overrun_frames": int(np.sum(data["loop_dt_s"] > 1.0 / metadata["nominal_rate_hz"])),
        "writer_dropped_frames": int(metadata.get("summary", {}).get("dropped", 0)),
    }

    fig, axes = plt.subplots(3, 1, figsize=(7.1, 6.1), sharex=True)
    state = data["joint_position"]
    command = data["safe_action"][:, :6]
    for joint in range(6):
        axes[0].plot(t, state[:, joint], color=colors[joint], label=f"J{joint + 1}")
        axes[1].plot(t, command[:, joint] - state[:, joint], color=colors[joint])
    axes[0].set_ylabel("Joint position (rad)")
    axes[0].legend(ncol=6, frameon=False, loc="upper center")
    axes[1].axhline(0, color="0.2", linewidth=0.7)
    axes[1].set_ylabel("Tracking error (rad)")
    axes[2].plot(t, data["raw_action"][:, 6], color="#0072B2", label="Policy")
    axes[2].plot(t, data["safe_action"][:, 6], color="#D55E00", label="Command")
    axes[2].set_ylabel("Gripper (rad)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(frameon=False)
    for ax in axes:
        ax.grid(True, color="0.9", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "trajectory.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.25))
    axes[0].plot(t, data["inference_ms"], color="#0072B2")
    nominal_ms = 1000.0 / float(metadata["nominal_rate_hz"])
    axes[0].axhline(nominal_ms, color="#D55E00", linestyle="--", label="Control budget")
    axes[0].set(xlabel="Time (s)", ylabel="Inference latency (ms)")
    axes[0].legend(frameon=False)
    axes[1].hist(data["loop_dt_s"] * 1000.0, bins=40, color="#009E73", alpha=0.85)
    axes[1].axvline(nominal_ms, color="#D55E00", linestyle="--")
    axes[1].set(xlabel="Control period (ms)", ylabel="Count")
    axes[2].plot(t, data["action_age_ms"], color="#CC79A7")
    axes[2].set(xlabel="Time (s)", ylabel="Action age (ms)")
    for ax in axes:
        ax.grid(True, color="0.92", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "timing.png")
    plt.close(fig)

    if "imu" in data or "tactile" in data:
        rows = int("imu" in data) + int("tactile" in data)
        fig, axes = plt.subplots(rows, 1, figsize=(7.1, 2.1 * rows), squeeze=False)
        row = 0
        if "imu" in data:
            imu = data["imu"]
            for axis, label in zip(range(4, 7), ("x", "y", "z")):
                axes[row, 0].plot(t, imu[:, axis], label=label)
            axes[row, 0].set_ylabel("Angular velocity")
            axes[row, 0].legend(frameon=False, ncol=3)
            row += 1
        if "tactile" in data:
            pressure = data["tactile"]
            axes[row, 0].plot(t, pressure.mean((1, 2)), label="Mean")
            axes[row, 0].plot(t, pressure.max((1, 2)), label="Maximum")
            axes[row, 0].set_ylabel("Tactile response")
            axes[row, 0].legend(frameon=False)
        axes[-1, 0].set_xlabel("Time (s)")
        for ax in axes[:, 0]:
            ax.grid(True, color="0.92", linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(output / "multimodal.png")
        plt.close(fig)
    keyframes = _generate_keyframe_montage(run_dir, output, data, plt)
    if keyframes:
        metrics["keyframes"] = keyframes
    _generate_overview(run_dir, output, metadata, data, keyframes, plt)
    if "tactile" in data:
        metrics["tactile"] = _generate_tactile_analysis(
            run_dir, output, data, plt
        )
    else:
        stale_tactile = output / "tactile_analysis.png"
        if stale_tactile.is_file():
            stale_tactile.unlink()
    if "imu" in data:
        metrics["imu"] = _generate_imu_analysis(run_dir, output, data, plt)
    else:
        stale_imu = output / "imu_analysis.png"
        if stale_imu.is_file():
            stale_imu.unlink()
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="生成ACT真机推理论文级诊断图")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = generate_report(args.run, args.output)
    print(f"图表已保存: {output}")


if __name__ == "__main__":
    main()
