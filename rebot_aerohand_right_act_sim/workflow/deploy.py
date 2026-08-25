"""Run a visual or multimodal ACT checkpoint in closed-loop MuJoCo."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

from rebot_aerohand_right_act_sim.config import (
    DEFAULT_CONFIG,
    build_act_config,
    camera_overlay_kwargs,
    hand_contact_processing_kwargs,
    load_config,
    object_randomization_kwargs,
    resolve_project_path,
)
from rebot_aerohand_right_act_sim.environment import (
    SimAeroHandACTEnvironment,
    snap_hand_target,
)
from rebot_aerohand_right_act_sim.policy import load_policy
from rebot_aerohand_right_act_sim.multimodal_policy import (
    is_multimodal_checkpoint,
    load_multimodal_spec,
)
from rebot_aerohand_right_act_sim.sensors import validate_processing_spec
from rebot_aerohand_right_act_sim.timing import WallClockRate
from rebot_aerohand_right_act_sim.visualization import AsyncSensorVisualizer, ReplaySensorVisualizer

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def _image(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(np.ascontiguousarray(value))
        .permute(2, 0, 1)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
        .unsqueeze(0)
    )


def _open_video(path: Path, image: np.ndarray, fps: float) -> cv2.VideoWriter:
    """Open an MP4 writer for side-by-side top and wrist RGB frames."""
    height, width = image.shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def _write_camera_frame(
    writer: cv2.VideoWriter, top: np.ndarray, wrist: np.ndarray
) -> None:
    frame = np.concatenate([top, wrist], axis=1).copy()
    cv2.putText(
        frame, "TOP", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    cv2.putText(
        frame,
        "WRIST",
        (top.shape[1] + 12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def _plot_channels(
    path: Path,
    time_axis: np.ndarray,
    groups: list[tuple[str, np.ndarray, list[str]]],
    *,
    y_label: str,
) -> None:
    fig, axes = plt.subplots(
        len(groups), 1, figsize=(12, max(3.2, 2.6 * len(groups))), sharex=True
    )
    axes = np.atleast_1d(axes)
    for axis, (title, values, labels) in zip(axes, groups, strict=True):
        values = np.asarray(values)
        if values.ndim == 1:
            values = values[:, None]
        for channel, label in enumerate(labels):
            axis.plot(time_axis, values[:, channel], linewidth=1.2, label=label)
        axis.set_title(title)
        axis.set_ylabel(y_label)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best", ncol=min(4, len(labels)), fontsize=8)
    axes[-1].set_xlabel("Control time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_rollout_plots(output_dir: Path, frames: dict[str, list]) -> list[str]:
    """Save category-specific diagnostic plots and return their filenames."""
    values = {name: np.asarray(items) for name, items in frames.items()}
    time_axis = values["control_time"]
    plots: list[str] = []

    arm_path = output_dir / "arm_joints.png"
    actual_arm = values["joint_position"]
    target_arm = values["action"][:, :6]
    _plot_channels(
        arm_path,
        time_axis,
        [
            (
                f"Arm joint {index + 1}",
                np.column_stack([actual_arm[:, index], target_arm[:, index]]),
                ["measured", "policy target"],
            )
            for index in range(actual_arm.shape[1])
        ],
        y_label="rad",
    )
    plots.append(arm_path.name)

    hand_actuator_path = output_dir / "hand_actuators.png"
    hand_groups = [
        (
            "Policy hand actuator targets",
            values["action"][:, 6:],
            [f"actuator_{i + 1}" for i in range(values["action"].shape[1] - 6)],
        ),
        (
            "Hand actuator feedback",
            values["hand_feedback"],
            [f"feedback_{i + 1}" for i in range(values["hand_feedback"].shape[1])],
        ),
    ]
    _plot_channels(
        hand_actuator_path, time_axis, hand_groups, y_label="actuator value"
    )
    plots.append(hand_actuator_path.name)

    hand_joint_path = output_dir / "hand_joints.png"
    hand_joint_values = values["hand_joint_position"]
    _plot_channels(
        hand_joint_path,
        time_axis,
        [
            (
                "Hand joint positions",
                hand_joint_values,
                [f"hand_joint_{i + 1}" for i in range(hand_joint_values.shape[1])],
            )
        ],
        y_label="rad",
    )
    plots.append(hand_joint_path.name)

    imu_path = output_dir / "imu.png"
    imu = values["imu"]
    imu_groups = [
        ("IMU quaternion", imu[:, :4], ["qw", "qx", "qy", "qz"]),
        ("IMU gyroscope", imu[:, 4:7], ["gx", "gy", "gz"]),
        ("IMU acceleration", imu[:, 7:10], ["ax", "ay", "az"]),
    ]
    _plot_channels(imu_path, time_axis, imu_groups, y_label="sensor value")
    plots.append(imu_path.name)

    contact_path = output_dir / "hand_contact_force.png"
    contact = values["hand_contact"]
    _plot_channels(
        contact_path,
        time_axis,
        [
            (
                "Hand contact force",
                contact,
                [f"region_{i + 1}" for i in range(contact.shape[1])],
            )
        ],
        y_label="force",
    )
    plots.append(contact_path.name)

    object_path = output_dir / "object_motion.png"
    _plot_channels(
        object_path,
        time_axis,
        [
            ("Object position", values["object_pose"][:, :3], ["x", "y", "z"]),
            (
                "Object linear velocity",
                values["object_velocity"][:, :3],
                ["vx", "vy", "vz"],
            ),
            (
                "Object angular velocity",
                values["object_velocity"][:, 3:6],
                ["wx", "wy", "wz"],
            ),
        ],
        y_label="SI units",
    )
    plots.append(object_path.name)

    tool_path = output_dir / "tool_motion.png"
    _plot_channels(
        tool_path,
        time_axis,
        [
            ("Tool position", values["tool_position"], ["x", "y", "z"]),
            (
                "Tool quaternion",
                values["tool_quaternion"],
                ["qw", "qx", "qy", "qz"],
            ),
        ],
        y_label="pose value",
    )
    plots.append(tool_path.name)

    distance_path = output_dir / "grasp_distance.png"
    _plot_channels(
        distance_path,
        time_axis,
        [("Grasp surface distance", values["grasp_surface_distance"], ["distance"])],
        y_label="m",
    )
    plots.append(distance_path.name)
    return plots


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument(
        "--num-rollouts",
        "--inference-count",
        dest="num_rollouts",
        type=int,
        default=50,
        help="Number of randomized evaluation rollouts (default: 50).",
    )
    parser.add_argument(
        "--output-dir",
        help="Evaluation output directory. Defaults to a timestamped directory "
             "under <checkpoint parent>/deploy_evaluations/.",
    )
    parser.add_argument(
        "--no-save-video",
        action="store_true",
        help="Save diagnostic plots and summaries without encoding camera video.",
    )
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--temporal-ensemble", type=float, default=0.9)
    parser.add_argument(
        "--no-sensors",
        action="store_true",
        help="Clean 3D view: skip the top-left IMU/hand-contact sensor panel "
             "and the top-right camera PIP views.",
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=1,
        help="Render MuJoCo viewer every N control steps (1 = every step).",
    )
    parser.add_argument(
        "--no-hand-contact",
        action="store_true",
        help="Skip per-physics-step hand contact accumulation (saves ~1-3 ms per cycle).",
    )
    parser.add_argument(
        "--hand-smooth",
        type=float,
        default=0.0,
        help="EMA smoothing coefficient for the policy's 7-dim hand target "
             "across control steps (0 = off; 0.4-0.6 suppresses open/close "
             "wobble on sparse datasets).",
    )
    parser.add_argument(
        "--snap-hand",
        action="store_true",
        help="Snap the hand target to the open/closed extremes with "
             "hysteresis; prevents close->open->close flapping during grasp.",
    )
    args = parser.parse_args()
    if args.num_rollouts <= 0:
        raise ValueError("--num-rollouts must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if not 0.0 <= args.hand_smooth < 1.0:
        raise ValueError("--hand-smooth must be in [0, 1)")
    raw = load_config(args.config)
    if args.device:
        raw["policy"]["device"] = args.device
    raw["policy"]["temporal_ensemble_coeff"] = args.temporal_ensemble
    dataset_cfg = raw["dataset"]
    metadata = LeRobotDatasetMetadata(
        str(dataset_cfg["repo_id"]), root=resolve_project_path(dataset_cfg["root"])
    )
    if int(metadata.fps) != int(dataset_cfg["fps"]):
        raise ValueError(
            f"dataset fps={metadata.fps} does not match configured "
            f"control frequency {dataset_cfg['fps']} Hz"
        )
    config = build_act_config(raw, metadata, n_action_steps=1)
    device = torch.device(str(raw["policy"]["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint = resolve_project_path(
        args.checkpoint
        or resolve_project_path(raw["training"]["output_dir"]) / "pretrained_model"
    )
    if args.output_dir:
        evaluation_dir = resolve_project_path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        evaluation_dir = checkpoint.parent / "deploy_evaluations" / timestamp
    evaluation_dir.mkdir(parents=True, exist_ok=False)
    print(f"Evaluation outputs: {evaluation_dir}")
    hand_contact_processing = raw["environment"].get("hand_contact_processing")
    multimodal_spec = load_multimodal_spec(checkpoint) if is_multimodal_checkpoint(checkpoint) else None
    use_hand_contact_policy = bool(multimodal_spec and multimodal_spec.get("use_hand_contact"))
    validate_processing_spec(
        resolve_project_path(dataset_cfg["root"]), hand_contact_processing
    )
    if use_hand_contact_policy:
        validate_processing_spec(checkpoint, hand_contact_processing)
    # Standard ACT never consumes hand contact — skip the per-physics-step
    # contact classification unless the checkpoint actually encodes a hand
    # contact sensor branch.
    enable_hand_contact = use_hand_contact_policy and not args.no_hand_contact
    if use_hand_contact_policy and args.no_hand_contact:
        print("Warning: checkpoint uses hand contact but --no-hand-contact was "
              "passed; contact processing disabled, policy may produce degraded actions.")

    policy = load_policy(checkpoint, config, metadata.stats).to(device).eval()
    seed = int(raw["environment"].get("seed", 0) if args.seed is None else args.seed)
    environment_cfg = raw["environment"]
    env = SimAeroHandACTEnvironment(
        resolve_project_path(environment_cfg["xml"]),
        seed=None,
        success_hold_seconds=float(
            environment_cfg.get("success_hold_seconds", 0.5)
        ),
        control_hz=int(dataset_cfg["fps"]),
        camera_render_hz=float(
            environment_cfg.get("camera_render_hz", dataset_cfg["fps"])
        ),
        enable_hand_contact_processing=enable_hand_contact,
        **object_randomization_kwargs(raw),
        **hand_contact_processing_kwargs(raw),
        **camera_overlay_kwargs(raw),
    )
    headless = args.no_sensors
    _sync_viz = None
    sensor_visualizer = None
    if not headless:
        _sync_viz = ReplaySensorVisualizer(
            history_frames=int(dataset_cfg["fps"]) * 2,
            contact_color_max=float(
                environment_cfg.get("hand_contact_processing", {}).get(
                    "visualization_color_max", 15.0
                )
            ),
        )
        sensor_visualizer = AsyncSensorVisualizer(
            history_frames=int(dataset_cfg["fps"]) * 2,
            contact_color_max=float(
                environment_cfg.get("hand_contact_processing", {}).get(
                    "visualization_color_max", 15.0
                )
            ),
        )
    rate = WallClockRate(float(dataset_cfg["fps"]))
    render_every = max(1, int(args.render_every))
    results: list[dict[str, int | float | bool | None]] = []
    evaluation_cancelled = False
    active_video_writer: cv2.VideoWriter | None = None
    try:
        with torch.inference_mode():
            for rollout_index in range(args.num_rollouts):
                rollout_seed = None if seed < 0 else seed + rollout_index
                env.reset(rollout_seed)
                policy.reset()
                rate.reset()
                if sensor_visualizer is not None:
                    sensor_visualizer.reset()
                if _sync_viz is not None:
                    _sync_viz.reset()
                step = 0
                success = False
                hand_prev: np.ndarray | None = None
                hand_snapped: np.ndarray | None = None
                rollout_started = time.perf_counter()
                object_pose = env.object_initial_position
                rollout_dir = evaluation_dir / f"rollout_{rollout_index + 1:03d}"
                rollout_dir.mkdir()
                frame_data: dict[str, list] = {
                    "control_time": [],
                    "sim_time": [],
                    "joint_position": [],
                    "joint_velocity": [],
                    "hand_feedback": [],
                    "hand_joint_position": [],
                    "imu": [],
                    "hand_contact": [],
                    "action": [],
                    "object_pose": [],
                    "object_velocity": [],
                    "tool_position": [],
                    "tool_quaternion": [],
                    "grasp_surface_distance": [],
                }
                seed_label = rollout_seed if rollout_seed is not None else "random"
                print(
                    f"Rollout {rollout_index + 1}/{args.num_rollouts}: "
                    f"seed={seed_label}, object=({object_pose[0]:.3f}, "
                    f"{object_pose[1]:.3f}, {object_pose[2]:.3f})"
                )

                while env.is_alive() and step < args.max_steps:
                    env.advance_physics()
                    if not env.is_control_tick(int(dataset_cfg["fps"])):
                        continue
                    observation = env.observe()

                    if not headless:
                        sensor_visualizer.push(
                            observation.imu,
                            observation.hand_contact,
                            observation.hand_feedback,
                            frame_index=step,
                            timestamp=observation.sim_time,
                        )

                    batch = {
                        "observation.image": _image(observation.image, device),
                        "observation.wrist_image": _image(observation.wrist_image, device),
                        "observation.state": torch.as_tensor(
                            observation.joint_position, device=device
                        ).float().unsqueeze(0),
                        "sensor.imu": torch.as_tensor(
                            observation.imu, device=device
                        ).float().unsqueeze(0),
                        "sensor.hand_contact": torch.as_tensor(
                            observation.hand_contact, device=device
                        ).float().unsqueeze(0),
                    }
                    action = policy.select_action(batch)[0].detach().cpu().numpy()
                    # Stabilize the hand dims: the arm trajectory is continuous,
                    # but the hand target is binary (open/closed) in the data and
                    # a sparse-data policy can wobble between the two states.
                    if args.hand_smooth > 0 or args.snap_hand:
                        hand = action[6:].astype(np.float64)
                        if args.hand_smooth > 0:
                            if hand_prev is None:
                                hand_prev = hand.copy()
                            hand = (
                                args.hand_smooth * hand
                                + (1.0 - args.hand_smooth) * hand_prev
                            )
                            hand_prev = hand.copy()
                        if args.snap_hand:
                            hand = snap_hand_target(
                                hand,
                                env.hand_mapper.open_ctrl,
                                env.hand_mapper.closed_ctrl,
                                previous=hand_snapped,
                            )
                            hand_snapped = hand.copy()
                        action = np.concatenate(
                            [action[:6], hand]
                        ).astype(np.float32)
                    env.command(action)

                    frame_data["control_time"].append(step / float(dataset_cfg["fps"]))
                    frame_data["sim_time"].append(observation.sim_time)
                    frame_data["joint_position"].append(observation.joint_position.copy())
                    frame_data["joint_velocity"].append(observation.joint_velocity.copy())
                    frame_data["hand_feedback"].append(observation.hand_feedback.copy())
                    frame_data["hand_joint_position"].append(
                        observation.hand_joint_position.copy()
                    )
                    frame_data["imu"].append(observation.imu.copy())
                    frame_data["hand_contact"].append(observation.hand_contact.copy())
                    frame_data["action"].append(action.copy())
                    object_qpos_adr = env.object_qpos_adr
                    frame_data["object_pose"].append(
                        env.data.qpos[object_qpos_adr : object_qpos_adr + 7].copy()
                    )
                    frame_data["object_velocity"].append(
                        env.data.qvel[
                            env.object_dof_adr : env.object_dof_adr + 6
                        ].copy()
                    )
                    tool_body_id = env.arm_controller.body_id
                    frame_data["tool_position"].append(
                        env.data.xpos[tool_body_id].copy()
                    )
                    frame_data["tool_quaternion"].append(
                        env.data.xquat[tool_body_id].copy()
                    )
                    frame_data["grasp_surface_distance"].append(
                        env.grasp_surface_distance()
                    )
                    if not args.no_save_video:
                        if active_video_writer is None:
                            active_video_writer = _open_video(
                                rollout_dir / "cameras.mp4",
                                observation.image,
                                float(dataset_cfg["fps"]),
                            )
                        _write_camera_frame(
                            active_video_writer,
                            observation.image,
                            observation.wrist_image,
                        )

                    if step % render_every == 0:
                        sensor_panel = None
                        if not headless:
                            sensor_panel = sensor_visualizer.get_panel()
                            if sensor_panel is None:
                                sensor_panel = _sync_viz.render(
                                    observation.imu,
                                    observation.hand_contact,
                                    observation.hand_feedback,
                                    frame_index=step,
                                    timestamp=observation.sim_time,
                                )
                        collected_cameras = {
                            "top": observation.image,
                            "cam_wrist": observation.wrist_image,
                        }
                        overlay_names = list(environment_cfg.get("camera_overlays", []))
                        camera_frames = (
                            [(name, collected_cameras[name]) for name in overlay_names]
                            if all(name in collected_cameras for name in overlay_names)
                            else None
                        )
                        status_lines = (
                            f"DEPLOY  rollout={rollout_index + 1}/{args.num_rollouts}",
                            f"step={step}  dist_to_obj="
                            f"{env.grasp_surface_distance():.3f}m",
                        )
                        env.render(
                            sensor_panel=sensor_panel,
                            status_lines=status_lines,
                            show_camera_overlays=not headless,
                            camera_frames=camera_frames,
                        )

                    step += 1
                    success = env.check_success()
                    if success:
                        break
                    rate.wait()

                if not env.is_alive():
                    if active_video_writer is not None:
                        active_video_writer.release()
                        active_video_writer = None
                    evaluation_cancelled = True
                    print("Viewer closed; stopping evaluation.")
                    break
                if active_video_writer is not None:
                    active_video_writer.release()
                    active_video_writer = None
                elapsed = time.perf_counter() - rollout_started
                final_object_pose = env.data.qpos[
                    env.object_qpos_adr : env.object_qpos_adr + 7
                ].copy()
                rollout_result = {
                    "rollout": rollout_index + 1,
                    "success": success,
                    "steps": step,
                    "elapsed_seconds": elapsed,
                    "seed": rollout_seed,
                    "object_initial_pose": object_pose.tolist(),
                    "object_final_pose": final_object_pose.tolist(),
                    "final_grasp_surface_distance": env.grasp_surface_distance(),
                    "video": None if args.no_save_video else "cameras.mp4",
                }
                rollout_result["plots"] = _save_rollout_plots(
                    rollout_dir, frame_data
                )
                _write_json(rollout_dir / "result.json", rollout_result)
                results.append(rollout_result)
                outcome = "SUCCESS" if success else "FAILURE"
                print(
                    f"Rollout {rollout_index + 1}/{args.num_rollouts}: {outcome} "
                    f"after {step} policy steps ({elapsed:.1f}s)"
                )
    finally:
        if active_video_writer is not None:
            active_video_writer.release()
        if sensor_visualizer is not None:
            sensor_visualizer.stop()
        env.close()

    completed = len(results)
    successes = sum(bool(result["success"]) for result in results)
    failures = completed - successes
    success_rate = 100.0 * successes / completed if completed else 0.0
    print("\nEvaluation summary")
    print(f"  requested rollouts : {args.num_rollouts}")
    print(f"  completed rollouts : {completed}")
    print(f"  successes / failures: {successes} / {failures}")
    print(f"  success rate       : {success_rate:.2f}%")
    successful_steps = [
        int(result["steps"]) for result in results if bool(result["success"])
    ]
    if successful_steps:
        print(f"  mean success steps : {np.mean(successful_steps):.1f}")
    if evaluation_cancelled:
        print("  status             : interrupted by viewer close")

    summary = {
        "checkpoint": str(checkpoint),
        "config": str(resolve_project_path(args.config)),
        "requested_rollouts": args.num_rollouts,
        "completed_rollouts": completed,
        "successes": successes,
        "failures": failures,
        "success_rate": successes / completed if completed else 0.0,
        "mean_success_steps": (
            float(np.mean(successful_steps)) if successful_steps else None
        ),
        "interrupted": evaluation_cancelled,
        "fps": int(dataset_cfg["fps"]),
        "max_steps": args.max_steps,
        "rollouts": results,
    }
    _write_json(evaluation_dir / "summary.json", summary)
    with (evaluation_dir / "rollouts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        fieldnames = [
            "rollout", "success", "steps", "elapsed_seconds", "seed",
            "object_initial_x", "object_initial_y", "object_initial_z",
            "object_final_x", "object_final_y", "object_final_z",
            "final_grasp_surface_distance",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            initial_pose = result["object_initial_pose"]
            final_pose = result["object_final_pose"]
            writer.writerow(
                {
                    "rollout": result["rollout"],
                    "success": result["success"],
                    "steps": result["steps"],
                    "elapsed_seconds": f"{float(result['elapsed_seconds']):.6f}",
                    "seed": "random" if result["seed"] is None else result["seed"],
                    "object_initial_x": initial_pose[0],
                    "object_initial_y": initial_pose[1],
                    "object_initial_z": initial_pose[2],
                    "object_final_x": final_pose[0],
                    "object_final_y": final_pose[1],
                    "object_final_z": final_pose[2],
                    "final_grasp_surface_distance": result[
                        "final_grasp_surface_distance"
                    ],
                }
            )
    print(f"  output directory   : {evaluation_dir}")


if __name__ == "__main__":
    main()
