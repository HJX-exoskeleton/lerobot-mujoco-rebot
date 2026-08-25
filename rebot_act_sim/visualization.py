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
    # Match the official B601 tactile viewer: linear VIRIDIS mapping with
    # nearest-neighbour taxel enlargement (no nonlinear contrast transform).
    normalized = np.clip(np.maximum(value, 0.0) / max(scale, 1e-6), 0.0, 1.0)
    bgr = cv2.applyColorMap(
        (normalized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS
    )
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_NEAREST)


def _reference_tactile_image(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Independent B601 pad images in the new XML's physical orientation.

    The sensor tensor remains reference-compatible [8, 16].  Only the display
    is transformed: the 16-point finger-length axis becomes image height, and
    the opposing pads are oriented so their fingertip ends are both at the
    bottom of the final MuJoCo overlay.  Do not average the opposing pads:
    doing so duplicates two displaced cylinder generatrices into both panels.
    """
    left_display = np.flipud(np.nan_to_num(left).clip(0, 1).T)
    right_display = np.nan_to_num(right).clip(0, 1).T

    def panel(value: np.ndarray, label: str, mirror: bool = False) -> np.ndarray:
        image = cv2.applyColorMap(
            (value * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS
        )
        image = cv2.resize(image, (160, 320), interpolation=cv2.INTER_NEAREST)
        if mirror:
            image = cv2.flip(image, 1)
        cv2.putText(
            image,
            label,
            (8, 24),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return image

    bgr = np.hstack(
        (panel(left_display, "left"), panel(right_display, "right", True))
    )
    cv2.line(
        bgr,
        (bgr.shape[1] // 2, 0),
        (bgr.shape[1] // 2, bgr.shape[0]),
        (255, 255, 255),
        2,
    )
    # cv2.imshow in the reference consumes BGR; MuJoCo mjr_drawPixels consumes RGB.
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _stabilize_distance_map(
    value: np.ndarray, previous: np.ndarray | None
) -> np.ndarray:
    """Exact per-taxel display EMA used by the B601 auto-grasp demo."""
    value = np.nan_to_num(value).clip(0, 1).astype(np.float32, copy=False)
    if previous is None:
        previous = np.zeros_like(value)
    return (0.25 * value + 0.75 * previous).astype(np.float32, copy=False)


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
        self._display_left: np.ndarray | None = None
        self._display_right: np.ndarray | None = None

    def reset(self) -> None:
        self.imu_history.clear()
        self.tactile_scale = self.fixed_tactile_scale or 1e-3
        self._display_left = None
        self._display_right = None

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
        self._display_left = _stabilize_distance_map(left, self._display_left)
        self._display_right = _stabilize_distance_map(right, self._display_right)
        self.imu_history.append(imu.copy())
        if self.fixed_tactile_scale is None:
            current_max = max(float(np.max(left)), float(np.max(right)), 1e-6)
            self.tactile_scale = max(
                current_max, self.tactile_scale * 0.98, 1e-3
            )

        # Square panel leaves enough vertical space for stacked IMU plots and
        # the reference-size 320x320 tactile image below them.
        panel = np.full((640, 640, 3), 22, dtype=np.uint8)
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
            panel, history[:, 4:7], x=14, y=84, width=612, height=92,
            title="GYRO xyz"
        )
        _history_plot(
            panel, history[:, 7:10], x=14, y=190, width=612, height=92,
            title="ACCEL xyz"
        )

        # Use the official B601 visualization verbatim. No EMA, morphology,
        # nonlinear contrast or extra thresholding is applied here.
        panel[320:640, 160:480] = _reference_tactile_image(
            self._display_left, self._display_right
        )
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
