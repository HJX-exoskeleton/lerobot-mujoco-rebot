"""Synchronous and asynchronous IMU and hand contact replay visualization."""

from __future__ import annotations

import threading
import time
from collections import deque

import cv2
import numpy as np

from .schema import HAND_CONTACT_DIM, HAND_CONTACT_REGIONS

AXIS_COLORS_RGB = ((255, 90, 90), (90, 230, 90), (80, 170, 255))


def hand_contact_metrics(
    value: np.ndarray,
) -> tuple[float, float, int]:
    value = np.maximum(np.asarray(value, dtype=np.float32).reshape(-1), 0.0)
    total = float(value.sum())
    if total <= 1e-8:
        return float(value.max()), float(value.mean()), -1
    return (
        float(value.max()),
        float(value.mean()),
        int(np.argmax(value)),
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


def _contact_bar_chart(
    panel: np.ndarray,
    value: np.ndarray,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    scale: float,
) -> None:
    cv2.rectangle(panel, (x, y), (x + width, y + height), (75, 75, 75), 1)
    # sqrt contrast keeps low but meaningful forces visible without changing
    # the stored values or policy inputs.
    normalized = np.sqrt(
        np.clip(np.maximum(value, 0.0) / max(scale, 1e-6), 0.0, 1.0)
    )
    slot = width // HAND_CONTACT_DIM
    bar_width = max(6, int(slot * 0.55))
    baseline = y + height - 8
    for index, ratio in enumerate(normalized):
        bar_height = int(ratio * (height - 24))
        left = x + 4 + index * slot
        top = baseline - bar_height
        color = (
            (80, 200, 255)
            if index != HAND_CONTACT_DIM - 1
            else (255, 200, 90)
        )
        cv2.rectangle(
            panel, (left, top), (left + bar_width, baseline), color, -1
        )
        _text(
            panel,
            f"{value[index]:.1f}",
            (left, max(y + 4, top - 6)),
            scale=0.30,
        )
    for index, name in enumerate(HAND_CONTACT_REGIONS):
        short = {"thumb": "T", "index": "I", "middle": "M", "ring": "R",
                 "pinky": "P", "palm": "PA"}[name]
        _text(panel, short, (x + 8 + index * slot, y + height - 6), scale=0.38)


class ReplaySensorVisualizer:
    """Render the sensor sample belonging to the current replay frame."""

    def __init__(
        self, *, history_frames: int = 100, contact_color_max: float | None = None
    ):
        if history_frames <= 0:
            raise ValueError("history_frames must be positive")
        if contact_color_max is not None and contact_color_max <= 0:
            raise ValueError("contact_color_max must be positive")
        self.imu_history: deque[np.ndarray] = deque(maxlen=history_frames)
        self.fixed_contact_scale = (
            None if contact_color_max is None else float(contact_color_max)
        )
        self.contact_scale = self.fixed_contact_scale or 1e-3

    def reset(self) -> None:
        self.imu_history.clear()
        self.contact_scale = self.fixed_contact_scale or 1e-3

    def render(
        self,
        imu: np.ndarray,
        hand_contact: np.ndarray,
        hand_feedback: np.ndarray,
        *,
        frame_index: int,
        timestamp: float,
    ) -> np.ndarray:
        imu = np.asarray(imu, dtype=np.float32).reshape(10)
        contact = np.asarray(hand_contact, dtype=np.float32).reshape(
            HAND_CONTACT_DIM
        )
        feedback = np.asarray(hand_feedback, dtype=np.float32).reshape(7)
        self.imu_history.append(imu.copy())
        if self.fixed_contact_scale is None:
            current_max = max(float(np.max(contact)), 1e-6)
            self.contact_scale = max(
                current_max, self.contact_scale * 0.98, 1e-3
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

        _contact_bar_chart(
            panel,
            contact,
            x=14,
            y=220,
            width=612,
            height=170,
            scale=self.contact_scale,
        )
        maximum, mean, dominant = hand_contact_metrics(contact)
        region = (
            "no contact"
            if dominant < 0
            else f"dominant={HAND_CONTACT_REGIONS[dominant]}"
        )
        _text(
            panel,
            "HAND CONTACT (N)  thumb / index / middle / ring / pinky / palm",
            (14, 214),
            color=(90, 220, 255),
        )
        _text(panel, f"max={maximum:.3f} mean={mean:.3f}", (14, 407))
        _text(panel, region, (14, 429))
        _text(
            panel,
            f"tendons(IF/MF/RF/PF)={feedback[0]:.4f}/{feedback[1]:.4f}/"
            f"{feedback[2]:.4f}/{feedback[3]:.4f}  thumb_abd={feedback[4]:.2f}",
            (14, 451),
        )
        _text(
            panel, f"shared contact color max={self.contact_scale:.3f}", (14, 470)
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
        self, *, history_frames: int = 100, contact_color_max: float | None = None
    ):
        self._visualizer = ReplaySensorVisualizer(
            history_frames=history_frames, contact_color_max=contact_color_max
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
        hand_contact: np.ndarray,
        hand_feedback: np.ndarray,
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
                hand_contact.copy(),
                hand_feedback.copy(),
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
                imu, contact, feedback, idx, ts, version = data
                panel = self._visualizer.render(
                    imu, contact, feedback, frame_index=idx, timestamp=ts
                )
                with self._lock:
                    self._panel = panel
                last_version = version
            else:
                time.sleep(0.001)  # idle to avoid busy-waiting
