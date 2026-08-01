"""Run a visual or multimodal ACT checkpoint in closed-loop MuJoCo."""

from __future__ import annotations

import argparse

import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

from rebot_act_sim.config import (
    DEFAULT_CONFIG,
    build_act_config,
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
        help="Clean 3D view: render the MuJoCo scene with free camera but skip "
             "PIP camera overlays and the IMU/tactile sensor panel.",
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
    tactile_processing = raw["environment"].get("tactile_processing")
    multimodal_spec = load_multimodal_spec(checkpoint) if is_multimodal_checkpoint(checkpoint) else None
    use_tactile_policy = bool(multimodal_spec and multimodal_spec.get("use_tactile"))
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

    policy = load_policy(checkpoint, config, metadata.stats).to(device).eval()
    policy.reset()
    seed = int(raw["environment"].get("seed", 0) if args.seed is None else args.seed)
    env = SimACTEnvironment(
        resolve_project_path(raw["environment"]["xml"]),
        seed=None,
        success_hold_seconds=float(
            raw["environment"].get("success_hold_seconds", 0.5)
        ),
        control_hz=int(dataset_cfg["fps"]),
        grab_sideview=False,               # deploy never uses sideview
        enable_tactile_processing=enable_tactile,
        **object_randomization_kwargs(raw),
        **tactile_processing_kwargs(raw),
    )
    env.reset(None if seed < 0 else seed)

    headless = args.no_sensors
    _sync_viz = None
    sensor_visualizer = None
    if not headless:
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
    step = 0
    rate = WallClockRate(float(dataset_cfg["fps"]))
    render_every = max(1, int(args.render_every))
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
                        observation.tactile_left,
                        observation.tactile_right,
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
                    "sensor.tactile_left": torch.as_tensor(
                        observation.tactile_left, device=device
                    ).float().unsqueeze(0),
                    "sensor.tactile_right": torch.as_tensor(
                        observation.tactile_right, device=device
                    ).float().unsqueeze(0),
                }
                action = policy.select_action(batch)[0].detach().cpu().numpy()
                env.command(action)

                if step % render_every == 0:
                    if headless:
                        # 3D scene with free camera only — no PIP overlays,
                        # no sensor panel.
                        env.render_scene()
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
