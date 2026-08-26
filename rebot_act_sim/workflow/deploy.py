"""Run a visual or multimodal ACT checkpoint in closed-loop MuJoCo."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CACHE_ROOT = _PROJECT_ROOT / "models"
os.environ.setdefault("HF_HOME", str(_CACHE_ROOT / ".hf_home"))
os.environ.setdefault("HF_HUB_CACHE", str(_CACHE_ROOT))
os.environ.setdefault("HF_DATASETS_CACHE", str(_CACHE_ROOT / "datasets"))
os.environ.setdefault("HF_XET_CACHE", str(_CACHE_ROOT / ".xet"))
os.environ.setdefault("HF_ASSETS_CACHE", str(_CACHE_ROOT / ".assets"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import cv2
import matplotlib
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.configs.policies import PreTrainedConfig

from rebot_act_sim.config import (
    DEFAULT_CONFIG,
    load_config,
    object_randomization_kwargs,
    resolve_project_path,
    tactile_processing_kwargs,
)
from rebot_act_sim.environment import SimACTEnvironment
from rebot_act_sim.policy import load_policy
from rebot_act_sim.multimodal_policy import (
    is_multimodal_checkpoint,
    load_multimodal_spec,
)
from rebot_act_sim.sensors import validate_processing_spec
from rebot_act_sim.timing import WallClockRate
from rebot_act_sim.visualization import AsyncSensorVisualizer, ReplaySensorVisualizer

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def _image(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(np.array(value, copy=True, order="C"))
        .permute(2, 0, 1)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
        .unsqueeze(0)
    )


def _object_contact_names(env: SimACTEnvironment) -> list[str]:
    model, data = env.parser.model, env.parser.data
    object_id = int(model.geom("geom_obj_red_cube").id)
    names: set[str] = set()
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        if object_id not in (int(contact.geom1), int(contact.geom2)):
            continue
        other = int(contact.geom2) if int(contact.geom1) == object_id else int(contact.geom1)
        name = model.geom(other).name
        names.add(str(name) if name else f"geom#{other}")
    return sorted(names)


def _deployment_seed(value: int | None) -> int | None:
    """Default deployment to entropy-backed randomization.

    A nonnegative CLI seed remains reproducible. Omitting ``--seed`` or using
    a negative value deliberately requests a fresh object pose on every run;
    the collection seed in the YAML must not silently fix deployment layouts.
    """
    return None if value is None or value < 0 else int(value)


def _open_video(path: Path, image: np.ndarray, fps: float) -> cv2.VideoWriter:
    height, width = image.shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer: {path}")
    return writer


def _write_camera_frame(
    writer: cv2.VideoWriter, top: np.ndarray, wrist: np.ndarray
) -> None:
    frame = np.concatenate([top, wrist], axis=1).copy()
    cv2.putText(frame, "TOP", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)
    cv2.putText(frame, "WRIST", (top.shape[1] + 12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def _plot_channels(
    path: Path,
    time_axis: np.ndarray,
    groups: list[tuple[str, np.ndarray, list[str]]],
    *,
    y_label: str,
) -> None:
    figure, axes = plt.subplots(
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
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _tactile_statistics(values: np.ndarray) -> np.ndarray:
    """Compress each 8x16 pad into stable plot diagnostics, not policy input."""
    values = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    rows, columns = np.indices(values.shape[1:], dtype=np.float32)
    total = values.sum(axis=(1, 2))
    safe_total = np.maximum(total, 1e-8)
    return np.column_stack(
        [
            values.max(axis=(1, 2)),
            values.mean(axis=(1, 2)),
            (values * rows).sum(axis=(1, 2)) / safe_total,
            (values * columns).sum(axis=(1, 2)) / safe_total,
        ]
    )


def _save_rollout_plots(output_dir: Path, frames: dict[str, list]) -> list[str]:
    values = {name: np.asarray(items) for name, items in frames.items()}
    time_axis = values["control_time"]
    plots: list[str] = []

    path = output_dir / "arm_joints.png"
    actual, target = values["joint_position"], values["action"][:, :6]
    _plot_channels(
        path, time_axis,
        [(f"Arm joint {i + 1}", np.column_stack([actual[:, i], target[:, i]]),
          ["measured", "policy target"]) for i in range(6)],
        y_label="rad",
    )
    plots.append(path.name)

    path = output_dir / "gripper.png"
    _plot_channels(
        path, time_axis,
        [("Gripper", np.column_stack([values["gripper_feedback"],
          values["action"][:, 6]]), ["position", "velocity", "policy target"])],
        y_label="position / normalized target",
    )
    plots.append(path.name)

    path = output_dir / "imu.png"
    imu = values["imu"]
    _plot_channels(
        path, time_axis,
        [("Quaternion", imu[:, :4], ["qw", "qx", "qy", "qz"]),
         ("Gyroscope", imu[:, 4:7], ["gx", "gy", "gz"]),
         ("Acceleration", imu[:, 7:10], ["ax", "ay", "az"])],
        y_label="sensor value",
    )
    plots.append(path.name)

    path = output_dir / "tactile.png"
    labels = ["max", "mean", "row centroid", "column centroid"]
    _plot_channels(
        path, time_axis,
        [("Left tactile", _tactile_statistics(values["tactile_left"]), labels),
         ("Right tactile", _tactile_statistics(values["tactile_right"]), labels)],
        y_label="proximity statistic",
    )
    plots.append(path.name)

    path = output_dir / "object_motion.png"
    _plot_channels(
        path, time_axis,
        [("Object position", values["object_position"], ["x", "y", "z"]),
         ("Object linear velocity", values["object_velocity"][:, :3], ["vx", "vy", "vz"]),
         ("Object angular velocity", values["object_velocity"][:, 3:], ["wx", "wy", "wz"])],
        y_label="SI units",
    )
    plots.append(path.name)

    path = output_dir / "tool_motion.png"
    _plot_channels(
        path, time_axis,
        [("Tool position", values["tool_position"], ["x", "y", "z"]),
         ("Tool quaternion", values["tool_quaternion"], ["qw", "qx", "qy", "qz"])],
        y_label="pose value",
    )
    plots.append(path.name)
    return plots


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_checkpoint_policy(
    checkpoint,
    metadata,
    device: torch.device,
    *,
    n_action_steps: int | None,
    temporal_ensemble_coeff: float | None,
):
    """Restore the checkpoint's own ACT architecture, as in real deployment."""
    config = PreTrainedConfig.from_pretrained(checkpoint)
    if not isinstance(config, ACTConfig):
        raise ValueError(f"checkpoint is not ACT: {type(config).__name__}")
    if n_action_steps is not None and n_action_steps <= 0:
        raise ValueError("--n-action-steps must be positive")
    config.device = str(device)
    if temporal_ensemble_coeff is not None:
        if not 0.0 < temporal_ensemble_coeff < 1.0:
            raise ValueError("--temporal-ensemble coefficient must be in (0, 1)")
        config.n_action_steps = 1
        config.temporal_ensemble_coeff = float(temporal_ensemble_coeff)
    else:
        config.temporal_ensemble_coeff = None
        if n_action_steps is not None:
            config.n_action_steps = int(n_action_steps)
    config.__post_init__()
    policy = load_policy(checkpoint, config, metadata.stats).to(device).eval()
    policy.reset()
    expected = {
        "observation.image",
        "observation.wrist_image",
        "observation.state",
    }
    if is_multimodal_checkpoint(checkpoint):
        expected.add("observation.environment_state")
    if set(policy.config.input_features) != expected:
        raise ValueError(
            "checkpoint input features mismatch: "
            f"expected={sorted(expected)}, actual={sorted(policy.config.input_features)}"
        )
    if set(policy.config.output_features) != {"action"}:
        raise ValueError(f"checkpoint output features are invalid: {policy.config.output_features}")
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--seed",
        type=int,
        help=(
            "Reproducible object-layout seed. Omit it (or use -1) for a fresh "
            "random object position on every deployment run."
        ),
    )
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
        help=(
            "Evaluation directory. Defaults to a timestamped directory under "
            "<checkpoint parent>/deploy_evaluations/."
        ),
    )
    parser.add_argument(
        "--no-save-video",
        action="store_true",
        help="Save plots and summaries without encoding top/wrist camera video.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1500,
        help="Maximum policy control steps (default: 1500, about 30 s at 50 Hz).",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=1,
        help="Actions executed from each prediction (AeroHand-aligned default: 1).",
    )
    parser.add_argument(
        "--temporal-ensemble",
        type=float,
        default=0.01,
        help=(
            "ACT temporal ensemble coefficient (default: 0.01, verified for this task)."
        ),
    )
    parser.add_argument(
        "--sensors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Show the top-left IMU and tactile panel (default: enabled; use "
            "--no-sensors for a clean MuJoCo view)."
        ),
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=1,
        help="Render MuJoCo viewer every N control steps (1 = every step).",
    )
    parser.add_argument(
        "--no-tactile",
        action="store_true",
        help="Skip tactile contact projection in physics steps (saves ~5 ms per cycle).",
    )
    args = parser.parse_args()
    if args.num_rollouts <= 0:
        raise ValueError("--num-rollouts must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    raw = load_config(args.config)
    if args.device:
        raw["policy"]["device"] = args.device
    dataset_cfg = raw["dataset"]
    metadata = LeRobotDatasetMetadata(
        str(dataset_cfg["repo_id"]), root=resolve_project_path(dataset_cfg["root"])
    )
    if int(metadata.fps) != int(dataset_cfg["fps"]):
        raise ValueError(
            f"dataset fps={metadata.fps} does not match configured "
            f"control frequency {dataset_cfg['fps']} Hz"
        )
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
    print(f"[DEPLOY] evaluation outputs={evaluation_dir}")
    tactile_processing = raw["environment"].get("tactile_processing")
    multimodal_spec = load_multimodal_spec(checkpoint) if is_multimodal_checkpoint(checkpoint) else None
    use_tactile_policy = bool(multimodal_spec and multimodal_spec.get("use_tactile"))
    use_imu_policy = bool(multimodal_spec and multimodal_spec.get("use_imu"))
    validate_processing_spec(
        resolve_project_path(dataset_cfg["root"]), tactile_processing
    )
    if use_tactile_policy:
        validate_processing_spec(checkpoint, tactile_processing)
    # Standard ACT never consumes tactile — skip the expensive per-physics-step
    # contact projection (~5-8 ms per control cycle) unless the checkpoint
    # actually encodes a tactile sensor branch.
    enable_tactile = use_tactile_policy and not args.no_tactile
    if use_tactile_policy and args.no_tactile:
        print("Warning: checkpoint uses tactile but --no-tactile was passed; "
              "tactile processing disabled, policy may produce degraded actions.")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    policy = _load_checkpoint_policy(
        checkpoint,
        metadata,
        device,
        n_action_steps=args.n_action_steps,
        temporal_ensemble_coeff=args.temporal_ensemble,
    )
    print(
        "[DEPLOY] checkpoint="
        f"{checkpoint} | IMU={use_imu_policy} | tactile={use_tactile_policy} | "
        f"n_action_steps={policy.config.n_action_steps} | "
        f"temporal_ensemble={policy.config.temporal_ensemble_coeff}"
    )
    base_seed = _deployment_seed(args.seed)
    env = SimACTEnvironment(
        resolve_project_path(raw["environment"]["xml"]),
        seed=None,
        success_hold_seconds=float(
            raw["environment"].get("success_hold_seconds", 0.5)
        ),
        control_hz=int(dataset_cfg["fps"]),
        camera_render_hz=float(
            raw["environment"].get("camera_render_hz", dataset_cfg["fps"])
        ),
        grab_sideview=False,               # deploy never uses sideview
        enable_tactile_processing=enable_tactile,
        **object_randomization_kwargs(raw),
        **tactile_processing_kwargs(raw),
    )
    clean_view = not args.sensors
    _sync_viz = None
    sensor_visualizer = None
    if args.sensors:
        _sync_viz = ReplaySensorVisualizer(
            history_frames=int(dataset_cfg["fps"]) * 2,
            tactile_color_max=float(
                raw["environment"].get("tactile_processing", {}).get(
                    "visualization_color_max", 2.0
                )
            ),
        )
        sensor_visualizer = AsyncSensorVisualizer(
            history_frames=int(dataset_cfg["fps"]) * 2,
            tactile_color_max=float(
                raw["environment"].get("tactile_processing", {}).get(
                    "visualization_color_max", 2.0
                )
            ),
        )
    rate = WallClockRate(float(dataset_cfg["fps"]))
    render_every = max(1, int(args.render_every))
    results: list[dict] = []
    evaluation_cancelled = False
    active_video_writer: cv2.VideoWriter | None = None
    try:
        with torch.inference_mode():
            for rollout_index in range(args.num_rollouts):
                rollout_seed = (
                    None if base_seed is None else base_seed + rollout_index
                )
                env.reset(rollout_seed)
                policy.reset()
                rate.reset()
                if sensor_visualizer is not None:
                    sensor_visualizer.reset()
                if _sync_viz is not None:
                    _sync_viz.reset()

                initial_object, initial_plate = env.task.get_obj_pose()
                rollout_dir = evaluation_dir / f"rollout_{rollout_index + 1:03d}"
                rollout_dir.mkdir()
                frame_data: dict[str, list] = {
                    "control_time": [], "sim_time": [], "joint_position": [],
                    "joint_velocity": [], "gripper_feedback": [], "imu": [],
                    "tactile_left": [], "tactile_right": [], "action": [],
                    "object_position": [], "object_velocity": [],
                    "tool_position": [], "tool_quaternion": [],
                }
                object_qpos_adr, object_dof_adr = env._free_joint_addresses(
                    "body_obj_mug_5"
                )
                tool_body = int(env.parser.model.body("end_link").id)
                step = 0
                success = False
                rollout_started = time.perf_counter()
                print(
                    f"[DEPLOY RESET] rollout={rollout_index + 1}/"
                    f"{args.num_rollouts} | "
                    f"seed={'random' if rollout_seed is None else rollout_seed} | "
                    f"object={np.round(initial_object, 4).tolist()} | "
                    f"plate={np.round(initial_plate, 4).tolist()}"
                )

                while env.is_alive() and step < args.max_steps:
                    env.advance_physics()
                    if not env.is_control_tick(int(dataset_cfg["fps"])):
                        continue
                    observation = env.observe()
                    if args.sensors:
                        sensor_visualizer.push(
                            observation.imu,
                            observation.tactile_left,
                            observation.tactile_right,
                            frame_index=step,
                            timestamp=observation.sim_time,
                        )

                    batch = {
                        "observation.image": _image(observation.image, device),
                        "observation.wrist_image": _image(
                            observation.wrist_image, device
                        ),
                        "observation.state": torch.as_tensor(
                            observation.joint_position, device=device
                        ).float().unsqueeze(0),
                        "sensor.imu": torch.as_tensor(
                            observation.imu, device=device
                        ).float().unsqueeze(0),
                        "sensor.tactile_left": torch.as_tensor(
                            observation.tactile_left, device=device
                        ).float().unsqueeze(0),
                        "sensor.tactile_right": torch.as_tensor(
                            observation.tactile_right, device=device
                        ).float().unsqueeze(0),
                    }
                    action = policy.select_action(batch)[0].detach().cpu().numpy()
                    if action.shape != (7,) or not np.all(np.isfinite(action)):
                        raise RuntimeError(f"policy produced invalid action: {action}")
                    env.command(action)

                    frame_data["control_time"].append(
                        step / float(dataset_cfg["fps"])
                    )
                    frame_data["sim_time"].append(observation.sim_time)
                    frame_data["joint_position"].append(
                        observation.joint_position.copy()
                    )
                    frame_data["joint_velocity"].append(
                        observation.joint_velocity.copy()
                    )
                    frame_data["gripper_feedback"].append(
                        [observation.gripper_position, observation.gripper_velocity]
                    )
                    frame_data["imu"].append(observation.imu.copy())
                    frame_data["tactile_left"].append(
                        observation.tactile_left.copy()
                    )
                    frame_data["tactile_right"].append(
                        observation.tactile_right.copy()
                    )
                    frame_data["action"].append(action.copy())
                    frame_data["object_position"].append(
                        env.parser.data.qpos[
                            object_qpos_adr : object_qpos_adr + 3
                        ].copy()
                    )
                    frame_data["object_velocity"].append(
                        env.parser.data.qvel[
                            object_dof_adr : object_dof_adr + 6
                        ].copy()
                    )
                    frame_data["tool_position"].append(
                        env.parser.data.xpos[tool_body].copy()
                    )
                    frame_data["tool_quaternion"].append(
                        env.parser.data.xquat[tool_body].copy()
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
                        if clean_view:
                            env.render(sensor_panel=None)
                        else:
                            sensor_panel = sensor_visualizer.get_panel()
                            if sensor_panel is None:
                                sensor_panel = _sync_viz.render(
                                    observation.imu,
                                    observation.tactile_left,
                                    observation.tactile_right,
                                    frame_index=step,
                                    timestamp=observation.sim_time,
                                )
                            env.render(sensor_panel=sensor_panel)

                    step += 1
                    success = env.check_success()
                    if success:
                        break
                    rate.wait()

                if not env.is_alive():
                    evaluation_cancelled = True
                    print("Viewer closed; stopping evaluation.")
                    break
                if active_video_writer is not None:
                    active_video_writer.release()
                    active_video_writer = None

                elapsed = time.perf_counter() - rollout_started
                final_object, final_plate = env.task.get_obj_pose()
                final_tcp = env.parser.data.xpos[tool_body].copy()
                final_left = env._gripper_position()
                final_right = float(env.parser.get_qpos_joint("finger_right")[0])
                result = {
                    "rollout": rollout_index + 1,
                    "success": bool(success),
                    "steps": step,
                    "elapsed_seconds": elapsed,
                    "seed": rollout_seed,
                    "object_initial_position": initial_object.tolist(),
                    "object_final_position": final_object.tolist(),
                    "plate_position": final_plate.tolist(),
                    "object_plate_xy": float(
                        np.linalg.norm(final_object[:2] - final_plate[:2])
                    ),
                    "tool_final_position": final_tcp.tolist(),
                    "gripper_left": float(final_left),
                    "gripper_right": final_right,
                    "object_contacts": _object_contact_names(env),
                    "video": None if args.no_save_video else "cameras.mp4",
                }
                result["plots"] = _save_rollout_plots(rollout_dir, frame_data)
                _write_json(rollout_dir / "result.json", result)
                results.append(result)
                print(
                    f"[DEPLOY RESULT] rollout={rollout_index + 1}/"
                    f"{args.num_rollouts} | "
                    f"{'SUCCESS' if success else 'FAILURE'} | steps={step} | "
                    f"object_plate_xy={result['object_plate_xy']:.4f} m | "
                    f"elapsed={elapsed:.1f}s"
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
    success_rate = successes / completed if completed else 0.0
    successful_steps = [
        int(result["steps"]) for result in results if bool(result["success"])
    ]
    summary = {
        "checkpoint": str(checkpoint),
        "config": str(resolve_project_path(args.config)),
        "requested_rollouts": args.num_rollouts,
        "completed_rollouts": completed,
        "successes": successes,
        "failures": failures,
        "success_rate": success_rate,
        "success_rate_percent": 100.0 * success_rate,
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
            "object_plate_xy", "gripper_left", "gripper_right",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            initial = result["object_initial_position"]
            final = result["object_final_position"]
            writer.writerow({
                "rollout": result["rollout"],
                "success": result["success"],
                "steps": result["steps"],
                "elapsed_seconds": f"{float(result['elapsed_seconds']):.6f}",
                "seed": "random" if result["seed"] is None else result["seed"],
                "object_initial_x": initial[0], "object_initial_y": initial[1],
                "object_initial_z": initial[2], "object_final_x": final[0],
                "object_final_y": final[1], "object_final_z": final[2],
                "object_plate_xy": result["object_plate_xy"],
                "gripper_left": result["gripper_left"],
                "gripper_right": result["gripper_right"],
            })
    print("\n[DEPLOY EVALUATION SUMMARY]")
    print(f"  requested rollouts  : {args.num_rollouts}")
    print(f"  completed rollouts  : {completed}")
    print(f"  successes / failures: {successes} / {failures}")
    print(f"  success rate        : {100.0 * success_rate:.2f}%")
    if successful_steps:
        print(f"  mean success steps  : {np.mean(successful_steps):.1f}")
    if evaluation_cancelled:
        print("  status              : interrupted by viewer close")
    print(f"  output directory    : {evaluation_dir}")


if __name__ == "__main__":
    main()
