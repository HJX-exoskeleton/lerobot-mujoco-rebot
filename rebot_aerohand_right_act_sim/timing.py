"""Wall-clock pacing utilities for real-time simulation workflows."""

from __future__ import annotations

import time
from collections.abc import Callable


class WallClockRate:
    """Absolute-deadline rate limiter that never runs catch-up bursts."""

    def __init__(
        self,
        hz: float,
        *,
        clock: Callable[[], float] = time.perf_counter,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        if hz <= 0:
            raise ValueError("hz must be positive")
        self.hz = float(hz)
        self.period = 1.0 / self.hz
        self._clock = clock
        self._sleep = sleeper
        self._next_deadline = self._clock() + self.period
        self.deadline_misses = 0

    def reset(self) -> None:
        self._next_deadline = self._clock() + self.period
        self.deadline_misses = 0

    def wait(self) -> float:
        """Wait for the next frame and return nonnegative lateness in seconds."""

        now = self._clock()
        remaining = self._next_deadline - now
        if remaining > 0:
            self._sleep(remaining)
            now = self._clock()
            lateness = max(0.0, now - self._next_deadline)
            self._next_deadline += self.period
            return lateness

        lateness = -remaining
        self.deadline_misses += 1
        # Do not execute several frames immediately after a slow frame.
        if lateness >= self.period:
            self._next_deadline = now + self.period
        else:
            self._next_deadline += self.period
        return lateness
