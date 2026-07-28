"""Latest-observation asynchronous inference for real-time deployment."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InferenceResult:
    action: np.ndarray
    inference_ms: float
    completed_at: float
    sequence: int


class LatestInferenceWorker:
    """Run one inference at a time and replace queued stale observations."""

    def __init__(self, initial: InferenceResult) -> None:
        self._condition = threading.Condition()
        self._pending: Callable[[], tuple[np.ndarray, float]] | None = None
        self._result = initial
        self._error: BaseException | None = None
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run, name="act_inference", daemon=True
        )
        self._thread.start()

    def submit_latest(self, inference: Callable[[], tuple[np.ndarray, float]]) -> None:
        with self._condition:
            if self._stopping:
                return
            # Replacing rather than appending bounds observation-to-action latency.
            self._pending = inference
            self._condition.notify()

    def latest(self) -> InferenceResult:
        with self._condition:
            if self._error is not None:
                raise RuntimeError("异步ACT推理失败") from self._error
            return self._result

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                inference = self._pending
                self._pending = None
            try:
                action, inference_ms = inference()
                completed_at = time.perf_counter()
                with self._condition:
                    self._result = InferenceResult(
                        action=np.asarray(action).copy(),
                        inference_ms=float(inference_ms),
                        completed_at=completed_at,
                        sequence=self._result.sequence + 1,
                    )
            except BaseException as exc:
                with self._condition:
                    self._error = exc
                    self._stopping = True
                    self._condition.notify_all()
                return

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            self._pending = None
            self._condition.notify_all()
        self._thread.join()
