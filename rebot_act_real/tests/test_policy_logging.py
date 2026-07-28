from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from rebot_act_real.policy_logging.analyze import generate_report, load_run
from rebot_act_real.policy_logging.recorder import PolicyRunRecorder


def test_recorder_writes_lossless_chunks_and_metadata(tmp_path):
    args = argparse.Namespace(imu=True, tactile=True)
    recorder = PolicyRunRecorder(
        root=tmp_path,
        run_name="unit-test",
        entrypoint="test",
        args=args,
        project_root=tmp_path,
        record_images=False,
        chunk_size=2,
        queue_size=8,
        rate_hz=50,
    )
    for step in range(3):
        recorder.record(
            {
                "step": step,
                "joint_position": np.full(6, step, dtype=np.float64),
                "raw_action": np.full(7, step + 0.5, dtype=np.float32),
            }
        )
    run_dir = recorder.close()

    metadata, data = load_run(run_dir)
    assert metadata["format"] == "rebot-policy-run"
    assert metadata["summary"]["written"] == 3
    assert metadata["summary"]["dropped"] == 0
    assert data["step"].tolist() == [0, 1, 2]
    np.testing.assert_allclose(data["joint_position"][:, 0], [0, 1, 2])
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["chunks"] == 2


def test_recorder_marks_optional_missing_values_as_nan(tmp_path):
    recorder = PolicyRunRecorder(
        root=tmp_path,
        run_name="missing",
        entrypoint="test",
        args=argparse.Namespace(imu=False, tactile=False),
        project_root=tmp_path,
        record_images=False,
        chunk_size=10,
        queue_size=8,
        rate_hz=10,
    )
    recorder.record({"step": 0, "optional": np.ones(2)})
    recorder.record({"step": 1})
    run_dir = recorder.close()
    _, data = load_run(run_dir)
    assert np.isnan(data["optional"][1]).all()


def test_publication_report_contains_vector_figures_and_metrics(tmp_path):
    recorder = PolicyRunRecorder(
        root=tmp_path,
        run_name="report",
        entrypoint="test",
        args=argparse.Namespace(imu=False, tactile=False),
        project_root=tmp_path,
        record_images=False,
        chunk_size=10,
        queue_size=8,
        rate_hz=50,
    )
    for step in range(4):
        tactile = np.zeros((12, 30), dtype=np.float32)
        if step in (1, 2):
            tactile[3:5, 12:16] = 0.4 + 0.2 * step
        angle = step * 0.1
        imu = np.asarray(
            [
                np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2),
                0.01 * step, 0.02 * step, 0.03 * step,
                0.0, 0.0, 1.0,
            ],
            dtype=np.float32,
        )
        recorder.record(
            {
                "time_s": step / 50,
                "loop_dt_s": 1 / 50,
                "joint_position": np.zeros(6),
                "raw_action": np.full(7, 0.1 * step),
                "safe_action": np.full(7, 0.1 * step),
                "inference_ms": 5.0 + step,
                "action_age_ms": 2.0 + step,
                "tactile": tactile,
                "imu": imu,
            }
        )
    run_dir = recorder.close()
    output = generate_report(run_dir)
    assert (output / "trajectory.png").is_file()
    assert (output / "timing.png").is_file()
    assert (output / "overview.png").is_file()
    assert (output / "tactile_analysis.png").is_file()
    assert (output / "imu_analysis.png").is_file()
    assert not list(output.glob("*.pdf"))
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["samples"] == 4
    assert len(metrics["tracking_rmse_rad_per_joint"]) == 6
    assert metrics["tactile"]["peak"] > 0
    assert metrics["tactile"]["active_duration_s"] > 0
    assert metrics["imu"]["peak_angular_rate_rad_s"] > 0


def test_video_is_finalized_as_independently_playable_segments(tmp_path):
    recorder = PolicyRunRecorder(
        root=tmp_path,
        run_name="video",
        entrypoint="test",
        args=argparse.Namespace(imu=False, tactile=False),
        project_root=tmp_path,
        record_images=True,
        chunk_size=2,
        queue_size=8,
        rate_hz=10,
    )
    for step in range(3):
        image = np.full((48, 64, 3), step * 60, dtype=np.uint8)
        recorder.record(
            {
                "step": step,
                "time_s": step / 10,
                "loop_dt_s": 0.1,
                "joint_position": np.zeros(6),
                "raw_action": np.full(7, step * 0.1),
                "safe_action": np.full(7, step * 0.1),
                "inference_ms": 5.0 + step,
                "action_age_ms": 1.0 + step,
                "image_high_rgb": image,
                "image_wrist_rgb": image,
            }
        )
    run_dir = recorder.close()

    high_segments = sorted((run_dir / "videos").glob("cam_high_*.mp4"))
    wrist_segments = sorted((run_dir / "videos").glob("cam_wrist_*.mp4"))
    assert len(high_segments) == len(wrist_segments) == 2
    assert not list((run_dir / "videos").glob(".*.partial.mp4"))
    expected_frames = [2, 1]
    for path, count in zip(high_segments, expected_frames):
        capture = cv2.VideoCapture(str(path))
        assert capture.isOpened()
        decoded = 0
        while capture.read()[0]:
            decoded += 1
        capture.release()
        assert decoded == count
    output = generate_report(run_dir)
    assert (output / "keyframes.png").is_file()
    assert (output / "overview.png").is_file()
    assert not list(output.glob("*.pdf"))
    events = json.loads((output / "keyframes.json").read_text())
    assert events[0]["event"] == "Start"
    assert events[-1]["event"] == "End"
