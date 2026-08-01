"""Synchronous and asynchronous IMU and tactile replay visualization."""

from __future__ import annotations

import threading
import time
from collections import deque

import cv2
import numpy as np

AXIS_COLORS_RGB = ((255, 90, 90), (90, 230, 90), (80, 170, 255))


def tactile_metrics(value: np.ndarray) -> tuple[float, float, float, float]:
    value = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    total = float(value.sum())
    if total <= 1e-8:
        return float(value.max()), float(value.mean()), -1.0, -1.0
    rows, columns = np.indices(value.shape, dtype=np.float32)
    return (
        float(value.max()),
        float(value.mean()),
        float((rows * value).sum() / total),
        float((columns * value).sum() / total),
    )


def _text(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    *,
    scale: float = 0.45,
    color: tuple[int, int, int] = (225, 225, 225),
) -> None:
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def _history_plot(
    panel: np.ndarray,
    history: np.ndarray,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
) -> None:
    cv2.rectangle(panel, (x, y), (x + width, y + height), (75, 75, 75), 1)
    if history.size == 0:
        return
    limit = max(float(np.max(np.abs(history))), 1e-3)
    zero_y = y + height // 2
    cv2.line(panel, (x + 1, zero_y), (x + width - 1, zero_y), (65, 65, 65), 1)
    xs = np.linspace(x + 2, x + width - 2, len(history)).astype(np.int32)
    ys = (zero_y - history / limit * max(height // 2 - 4, 1)).astype(np.int32)
    for axis, color in enumerate(AXIS_COLORS_RGB):
        points = np.column_stack((xs, ys[:, axis]))
        if len(points) == 1:
            cv2.circle(panel, tuple(points[0]), 2, color, -1, cv2.LINE_AA)
        else:
            cv2.polylines(panel, [points], False, color, 1, cv2.LINE_AA)
    _text(panel, f"{title}  range +/-{limit:.2f}", (x + 5, y + 16), scale=0.38)


def _tactile_heatmap(
    value: np.ndarray,
    *,
    width: int,
    height: int,
    scale: float,
) -> np.ndarray:
    # Pressure projection intentionally spreads force over the contact patch;
    # sqrt contrast keeps low but meaningful taxel loads visible without
    # changing the stored values or policy inputs.
    normalized = np.sqrt(
        np.clip(np.maximum(value, 0.0) / max(scale, 1e-6), 0.0, 1.0)
    )
    bgr = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_NEAREST)


class ReplaySensorVisualizer:
    """Render the sensor sample belonging to the current replay frame."""

    def __init__(
        self, *, history_frames: int = 100, tactile_color_max: float | None = None
    ):
        if history_frames <= 0:
            raise ValueError("history_frames must be positive")
        if tactile_color_max is not None and tactile_color_max <= 0:
            raise ValueError("tactile_color_max must be positive")
        self.imu_history: deque[np.ndarray] = deque(maxlen=history_frames)
        self.fixed_tactile_scale = (
            None if tactile_color_max is None else float(tactile_color_max)
        )
        self.tactile_scale = self.fixed_tactile_scale or 1e-3

    def reset(self) -> None:
        self.imu_history.clear()
        self.tactile_scale = self.fixed_tactile_scale or 1e-3

    def render(
        self,
        imu: np.ndarray,
        tactile_left: np.ndarray,
        tactile_right: np.ndarray,
        *,
        frame_index: int,
        timestamp: float,
    ) -> np.ndarray:
        imu = np.asarray(imu, dtype=np.float32).reshape(10)
        left = np.asarray(tactile_left, dtype=np.float32).reshape(8, 16)
        right = np.asarray(tactile_right, dtype=np.float32).reshape(8, 16)
        self.imu_history.append(imu.copy())
        if self.fixed_tactile_scale is None:
            current_max = max(float(np.max(left)), float(np.max(right)), 1e-6)
            self.tactile_scale = max(
                current_max, self.tactile_scale * 0.98, 1e-3
            )

        panel = np.full((480, 640, 3), 22, dtype=np.uint8)
        _text(
            panel,
            f"SYNCHRONIZED SENSORS  frame={frame_index}  t={timestamp:.3f}s",
            (14, 24),
            scale=0.52,
            color=(90, 220, 255),
        )
        quat, gyro, accel = imu[:4], imu[4:7], imu[7:10]
        _text(panel, "IMU quaternion (wxyz)", (14, 51))
        _text(panel, " ".join(f"{item:+.3f}" for item in quat), (14, 72))
        history = np.asarray(self.imu_history, dtype=np.float32)
        _history_plot(
            panel, history[:, 4:7], x=14, y=84, width=298, height=82, title="GYRO xyz"
        )
        _history_plot(
            panel, history[:, 7:10], x=328, y=84, width=298, height=82, title="ACCEL xyz"
        )
        _text(
            panel,
            f"gyro now:  {gyro[0]:+.2f}  {gyro[1]:+.2f}  {gyro[2]:+.2f}",
            (14, 184),
        )
        _text(
            panel,
            f"accel now: {accel[0]:+.2f}  {accel[1]:+.2f}  {accel[2]:+.2f}",
            (328, 184),
        )

        heatmap_y, heatmap_h, heatmap_w = 220, 180, 298
        panel[heatmap_y : heatmap_y + heatmap_h, 14 : 14 + heatmap_w] = _tactile_heatmap(
            left, width=heatmap_w, height=heatmap_h, scale=self.tactile_scale
        )
        panel[heatmap_y : heatmap_y + heatmap_h, 328 : 328 + heatmap_w] = _tactile_heatmap(
            right, width=heatmap_w, height=heatmap_h, scale=self.tactile_scale
        )
        for x, name, value in ((14, "LEFT 8x16", left), (328, "RIGHT 8x16", right)):
            maximum, mean, row, column = tactile_metrics(value)
            center = "no contact" if row < 0 else f"center=({row:.1f},{column:.1f})"
            _text(panel, name, (x, 214), color=(90, 220, 255))
            _text(panel, f"max={maximum:.3f} mean={mean:.3f}", (x, 421))
            _text(panel, center, (x, 443))
        _text(panel, f"shared tactile color max={self.tactile_scale:.3f}", (14, 470))
        return panel


class AsyncSensorVisualizer:
    """Background-thread sensor panel renderer for latency-sensitive control loops.

    The main control thread calls :meth:`push` with the latest sensor readings
    (non-blocking) and later retrieves a fully-rendered panel via
    :meth:`get_panel` (also non-blocking).  A daemon thread runs the
    ``ReplaySensorVisualizer.render`` workload in the background so that
    sensor-panel construction overlaps with the next policy inference or
    physics step.
    """

    def __init__(
        self, *, history_frames: int = 100, tactile_color_max: float | None = None
    ):
        self._visualizer = ReplaySensorVisualizer(
            history_frames=history_frames, tactile_color_max=tactile_color_max
        )
        self._lock = threading.Lock()
        self._data: tuple | None = None
        self._version: int = 0
        self._panel: np.ndarray | None = None
        self._running = True
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    def push(
        self,
        imu: np.ndarray,
        tactile_left: np.ndarray,
        tactile_right: np.ndarray,
        *,
        frame_index: int,
        timestamp: float,
    ) -> None:
        """Submit the latest sensor sample for background rendering.

        This copies the arrays so the caller may reuse the originals without
        data races.
        """
        with self._lock:
            self._version += 1
            self._data = (
                imu.copy(),
                tactile_left.copy(),
                tactile_right.copy(),
                frame_index,
                timestamp,
                self._version,
            )

    def get_panel(self) -> np.ndarray | None:
        """Return the most recently rendered panel, or *None*."""
        with self._lock:
            return self._panel

    def reset(self) -> None:
        """Clear accumulated history and discard any pending render."""
        with self._lock:
            self._data = None
            self._version += 1
            self._panel = None
        self._visualizer.reset()

    def stop(self) -> None:
        """Signal the background thread to exit (best-effort, daemon thread)."""
        self._running = False

    # ------------------------------------------------------------------
    def _render_loop(self) -> None:
        last_version = -1
        while self._running:
            with self._lock:
                data = self._data
            if data is not None and data[-1] != last_version:
                imu, left, right, idx, ts, version = data
                panel = self._visualizer.render(
                    imu, left, right, frame_index=idx, timestamp=ts
                )
                with self._lock:
                    self._panel = panel
                last_version = version
            else:
                time.sleep(0.001)  # idle to avoid busy-waiting
