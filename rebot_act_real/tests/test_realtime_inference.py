from __future__ import annotations

import threading
import time

import numpy as np

from rebot_act_real.realtime_inference import (
    InferenceResult,
    LatestInferenceWorker,
)


def test_latest_inference_worker_replaces_stale_pending_observation():
    release = threading.Event()
    started = threading.Event()
    initial = InferenceResult(np.zeros(1), 1.0, time.perf_counter(), 0)
    worker = LatestInferenceWorker(initial)

    def slow():
        started.set()
        release.wait()
        return np.asarray([1.0]), 10.0

    worker.submit_latest(slow)
    assert started.wait(timeout=1.0)
    worker.submit_latest(lambda: (np.asarray([2.0]), 2.0))
    worker.submit_latest(lambda: (np.asarray([3.0]), 3.0))
    release.set()

    deadline = time.perf_counter() + 1.0
    while worker.latest().sequence < 2 and time.perf_counter() < deadline:
        time.sleep(0.001)
    result = worker.latest()
    worker.close()

    assert result.sequence == 2
    assert result.action.tolist() == [3.0]
    assert result.inference_ms == 3.0
