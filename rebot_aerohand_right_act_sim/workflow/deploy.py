"""Run a visual or multimodal ACT checkpoint in closed-loop MuJoCo."""

from __future__ import annotations

import argparse

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


def _image(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(np.ascontiguousarray(value))
        .permute(2, 0, 1)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
        .unsqueeze(0)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
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
    policy.reset()
    seed = int(raw["environment"].get("seed", 0) if args.seed is None else args.seed)
    env = SimAeroHandACTEnvironment(
        resolve_project_path(raw["environment"]["xml"]),
        seed=None,
        success_hold_seconds=float(
            raw["environment"].get("success_hold_seconds", 0.5)
        ),
        control_hz=int(dataset_cfg["fps"]),
        enable_hand_contact_processing=enable_hand_contact,
        **object_randomization_kwargs(raw),
        **hand_contact_processing_kwargs(raw),
        **camera_overlay_kwargs(raw),
    )
    env.reset(None if seed < 0 else seed)

    headless = args.no_sensors
    _sync_viz = None
    sensor_visualizer = None
    if not headless:
        _sync_viz = ReplaySensorVisualizer(
            history_frames=int(dataset_cfg["fps"]) * 2,
            contact_color_max=float(
                raw["environment"].get("hand_contact_processing", {}).get(
                    "visualization_color_max", 15.0
                )
            ),
        )
        sensor_visualizer = AsyncSensorVisualizer(
            history_frames=int(dataset_cfg["fps"]) * 2,
            contact_color_max=float(
                raw["environment"].get("hand_contact_processing", {}).get(
                    "visualization_color_max", 15.0
                )
            ),
        )
    step = 0
    rate = WallClockRate(float(dataset_cfg["fps"]))
    render_every = max(1, int(args.render_every))
    hand_prev: np.ndarray | None = None
    hand_snapped: np.ndarray | None = None
    try:
        with torch.inference_mode():
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
                    status_lines = (
                        "DEPLOY",
                        f"step={step}  t={observation.sim_time:.2f}s  "
                        f"dist_to_obj={env.grasp_surface_distance():.3f}m",
                    )
                    env.render(
                        sensor_panel=sensor_panel,
                        status_lines=status_lines,
                        show_camera_overlays=not headless,
                    )

                step += 1
                if env.check_success():
                    print(f"Success after {step} policy steps")
                    return
                rate.wait()
        print(f"Rollout stopped without success after {step} policy steps")
    finally:
        if sensor_visualizer is not None:
            sensor_visualizer.stop()
        env.close()


if __name__ == "__main__":
    main()
