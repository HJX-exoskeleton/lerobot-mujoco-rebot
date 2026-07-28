"""Shared IMU/tactile visualization for ACT real-world deployment."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

import cv2
import numpy as np

_AXIS_COLORS = ((80, 80, 255), (80, 220, 80), (255, 160, 60))


class AsyncPanelRenderer:
    """Render panels off the control thread without accumulating stale frames."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="act_visualization"
        )
        self._future: Future[np.ndarray] | None = None
        self._latest: np.ndarray | None = None
        self._fresh = False

    def poll(self) -> np.ndarray | None:
        if self._future is not None and self._future.done():
            # Propagate rendering failures instead of silently freezing the window.
            self._latest = self._future.result()
            self._future = None
            self._fresh = True
        return self._latest

    def take_latest(self) -> np.ndarray | None:
        self.poll()
        if not self._fresh:
            return None
        self._fresh = False
        return self._latest

    @property
    def idle(self) -> bool:
        self.poll()
        return self._future is None

    def submit_latest(self, render: Callable[[], np.ndarray]) -> bool:
        """Submit only when idle; callers can try again with fresher data next step."""
        self.poll()
        if self._future is not None:
            return False
        self._future = self._executor.submit(render)
        return True

    def close(self) -> None:
        self._executor.shutdown(wait=True)


def _put_lines(
    image: np.ndarray,
    lines: list[str],
    *,
    x: int,
    y: int,
    color: tuple[int, int, int] = (230, 230, 230),
    step: int = 25,
) -> None:
    for line in lines:
        cv2.putText(
            image,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            color,
            1,
            cv2.LINE_AA,
        )
        y += step


def _tactile_metrics(tactile: np.ndarray) -> tuple[float, float, float, float]:
    tactile = np.maximum(np.asarray(tactile, dtype=np.float32), 0.0)
    total = float(tactile.sum())
    if total <= 1e-8:
        return float(tactile.max()), float(tactile.mean()), -1.0, -1.0
    rows, columns = np.indices(tactile.shape, dtype=np.float32)
    center_row = float((rows * tactile).sum() / total)
    center_col = float((columns * tactile).sum() / total)
    return float(tactile.max()), float(tactile.mean()), center_row, center_col


def _draw_vector_history(
    image: np.ndarray,
    values: np.ndarray,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
) -> None:
    """Draw an auto-scaled, zero-centred rolling three-axis plot."""
    cv2.rectangle(image, (x, y), (x + width, y + height), (60, 60, 60), 1)
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.all(np.isfinite(finite), axis=1)]
    if not len(finite):
        _put_lines(image, [f"{title}: waiting"], x=x + 5, y=y + 17, step=18)
        return

    limit = max(float(np.max(np.abs(finite))), 1e-3)
    zero_y = y + height // 2
    cv2.line(image, (x + 1, zero_y), (x + width - 1, zero_y), (75, 75, 75), 1)
    if len(finite) == 1:
        xs = np.array([x + width - 2], dtype=np.int32)
    else:
        xs = np.linspace(x + 2, x + width - 2, len(finite)).astype(np.int32)
    ys = (
        zero_y
        - finite / limit * max(height // 2 - 4, 1)
    ).astype(np.int32)
    for axis, color in enumerate(_AXIS_COLORS):
        points = np.column_stack((xs, ys[:, axis]))
        if len(points) == 1:
            cv2.circle(image, tuple(points[0]), 2, color, -1, cv2.LINE_AA)
        else:
            cv2.polylines(image, [points], False, color, 1, cv2.LINE_AA)
    cv2.putText(
        image,
        f"{title}  +/-{limit:.2f}",
        (x + 5, y + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "x  y  z",
        (x + width - 58, y + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )


def add_multimodal_panel(
    base_panel: np.ndarray,
    sensor_values: dict[str, np.ndarray],
    *,
    use_imu: bool,
    use_tactile: bool,
    sensor_history: Iterable[Mapping[str, object]] | None = None,
) -> np.ndarray:
    """Append a fixed-width sensor panel without changing the existing ACT view."""
    height = max(int(base_panel.shape[0]), 700)
    result = np.full((height, int(base_panel.shape[1]) + 340, 3), 24, dtype=np.uint8)
    result[: base_panel.shape[0], : base_panel.shape[1]] = base_panel
    x0 = int(base_panel.shape[1])
    cv2.line(result, (x0, 0), (x0, height), (85, 85, 85), 1)
    cv2.putText(
        result,
        "MULTIMODAL SENSORS",
        (x0 + 14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (80, 220, 255),
        1,
        cv2.LINE_AA,
    )

    imu = sensor_values.get("sensor.imu")
    if use_imu and imu is not None:
        imu = np.asarray(imu, dtype=np.float32).reshape(10)
        quat, gyro, accel = imu[:4], imu[4:7], imu[7:10]
        imu_lines = [
            "IMU: ACTIVE",
            f"quat  {quat[0]:+.2f} {quat[1]:+.2f}",
            f"      {quat[2]:+.2f} {quat[3]:+.2f}",
            f"gyro  {gyro[0]:+.2f} {gyro[1]:+.2f} {gyro[2]:+.2f}",
            f"accel {accel[0]:+.2f} {accel[1]:+.2f} {accel[2]:+.2f}",
        ]
        _put_lines(result, imu_lines, x=x0 + 14, y=55, step=20)
        history_values = []
        for item in sensor_history or ():
            value = item.get("sensor.imu")
            if value is not None:
                array = np.asarray(value, dtype=np.float32)
                if array.size == 10:
                    history_values.append(array.reshape(10))
        # Always show the current point, including for callers without a history.
        if not history_values or not np.array_equal(history_values[-1], imu):
            history_values.append(imu)
        imu_history = np.asarray(history_values, dtype=np.float32)
        _draw_vector_history(
            result,
            imu_history[:, 4:7],
            x=x0 + 14,
            y=150,
            width=312,
            height=72,
            title="GYRO",
        )
        _draw_vector_history(
            result,
            imu_history[:, 7:10],
            x=x0 + 14,
            y=230,
            width=312,
            height=72,
            title="ACCEL",
        )
    else:
        status = "WAITING" if use_imu else "DISABLED"
        _put_lines(result, [f"IMU: {status}"], x=x0 + 14, y=62, color=(150, 150, 150))

    tactile = sensor_values.get("sensor.tactile")
    tactile_top = 330
    if use_tactile and tactile is not None:
        tactile = np.asarray(tactile, dtype=np.float32).reshape(12, 30)
        display = np.clip(tactile, 0.0, 1.0)
        heatmap = cv2.applyColorMap(
            np.asarray(display * 255.0, dtype=np.uint8), cv2.COLORMAP_TURBO
        )
        heatmap = cv2.resize(heatmap, (312, 180), interpolation=cv2.INTER_NEAREST)
        result[tactile_top : tactile_top + 180, x0 + 14 : x0 + 326] = heatmap
        maximum, mean, center_row, center_col = _tactile_metrics(tactile)
        if center_row >= 0:
            px = x0 + 14 + int((center_col + 0.5) / 30.0 * 312)
            py = tactile_top + int((center_row + 0.5) / 12.0 * 180)
            cv2.drawMarker(
                result, (px, py), (255, 255, 255), cv2.MARKER_CROSS, 18, 2
            )
            center_text = f"center=({center_row:.1f},{center_col:.1f})"
        else:
            center_text = "center=no contact"
        _put_lines(
            result,
            [
                "TACTILE: ACTIVE  12x30",
                f"max={maximum:.3f} mean={mean:.3f}",
                center_text,
            ],
            x=x0 + 14,
            y=tactile_top + 210,
        )
    else:
        status = "WAITING" if use_tactile else "DISABLED"
        _put_lines(
            result,
            [f"TACTILE: {status}"],
            x=x0 + 14,
            y=tactile_top,
            color=(150, 150, 150),
        )
    return result
