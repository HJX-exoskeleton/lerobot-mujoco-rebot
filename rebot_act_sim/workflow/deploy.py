"""Run a visual or multimodal ACT checkpoint in closed-loop MuJoCo."""

from __future__ import annotations

import argparse
import os
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


class ReleaseSettleGuard:
    """Preserve the demonstration's pause between opening and retreating.

    Only the six arm targets are held. The policy's gripper output is passed
    through unchanged, so this is not gripper compensation or hysteresis.
    """

    def __init__(self, hold_steps: int):
        self.hold_steps = max(int(hold_steps), 0)
        self.remaining = 0
        self.saw_closed = False
        self.finished = False
        self.arm_hold: np.ndarray | None = None

    def update(
        self, action: np.ndarray, measured_arm: np.ndarray
    ) -> tuple[np.ndarray, bool]:
        action = np.asarray(action, dtype=np.float32).copy()
        if float(action[6]) >= 0.8:
            self.saw_closed = True
        if (
            self.saw_closed
            and not self.finished
            and self.remaining == 0
            and float(action[6]) <= 0.1
            and self.hold_steps > 0
        ):
            self.arm_hold = np.asarray(measured_arm, dtype=np.float32).copy()
            self.remaining = self.hold_steps
        holding = self.remaining > 0
        if holding:
            action[:6] = self.arm_hold
            self.remaining -= 1
            if self.remaining == 0:
                self.finished = True
        return action, holding


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
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--max-steps", type=int, default=800)
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
        default=False,
        help="Show camera/IMU/tactile overlays; disabled by default.",
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
    parser.add_argument(
        "--release-settle-seconds",
        type=float,
        default=0.5,
        help="Hold only the arm after the first open command before retreat (default: 0.5 s).",
    )
    args = parser.parse_args()
    if args.release_settle_seconds < 0.0:
        raise ValueError("--release-settle-seconds must be nonnegative")
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
    seed = int(raw["environment"].get("seed", 0) if args.seed is None else args.seed)
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
    env.reset(None if seed < 0 else seed)

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
    step = 0
    initial_tcp = env.parser.get_pR_body(body_name=env.task.tcp_body_name)[0]
    initial_object, _ = env.task.get_obj_pose()
    action_min = np.full(7, np.inf, dtype=np.float64)
    action_max = np.full(7, -np.inf, dtype=np.float64)
    last_action = np.zeros(7, dtype=np.float64)
    rate = WallClockRate(float(dataset_cfg["fps"]))
    render_every = max(1, int(args.render_every))
    release_guard = ReleaseSettleGuard(
        round(args.release_settle_seconds * float(dataset_cfg["fps"]))
    )
    release_announced = False
    try:
        with torch.inference_mode():
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
                raw_action = policy.select_action(batch)[0].detach().cpu().numpy()
                if raw_action.shape != (7,) or not np.all(np.isfinite(raw_action)):
                    raise RuntimeError(f"policy produced invalid action: {raw_action}")
                # Do not reshape or compensate the learned command. The final
                # component is clipped by environment.command only to the
                # normalized gripper range used during collection/replay.
                action, release_holding = release_guard.update(
                    raw_action, observation.joint_position
                )
                if release_holding and not release_announced:
                    release_announced = True
                    left_opening = env._gripper_position()
                    right_opening = float(
                        env.parser.get_qpos_joint("finger_right")[0]
                    )
                    print(
                        "[RELEASE] holding arm for object detachment; "
                        f"left={left_opening:.4f} right={right_opening:.4f} "
                        f"policy_gripper={float(raw_action[6]):.4f}"
                    )
                action_min = np.minimum(action_min, action)
                action_max = np.maximum(action_max, action)
                last_action = action.copy()
                env.command(action)

                if step % render_every == 0:
                    if clean_view:
                        # Follow replay/AeroHand's normal interactive viewer
                        # path while omitting only the sensor panel.
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
                if env.check_success():
                    print(f"Success after {step} policy steps")
                    return
                rate.wait()
        print(f"Rollout stopped without success after {step} policy steps")
    finally:
        final_tcp = env.parser.get_pR_body(body_name=env.task.tcp_body_name)[0]
        final_object, final_plate = env.task.get_obj_pose()
        final_left = env._gripper_position()
        final_right = float(env.parser.get_qpos_joint("finger_right")[0])
        final_contacts = _object_contact_names(env)
        print(
            "[DEPLOY SUMMARY] "
            f"tcp_delta={np.round(final_tcp - initial_tcp, 4).tolist()} | "
            f"object_delta={np.round(final_object - initial_object, 4).tolist()} | "
            f"object_plate_xy={np.linalg.norm(final_object[:2] - final_plate[:2]):.4f} | "
            f"gripper_left={final_left:.4f} gripper_right={final_right:.4f} | "
            f"object_contacts={final_contacts}\n"
            f"[DEPLOY ACTION] min={np.round(action_min, 4).tolist()} | "
            f"max={np.round(action_max, 4).tolist()} | "
            f"last={np.round(last_action, 4).tolist()}"
        )
        if sensor_visualizer is not None:
            sensor_visualizer.stop()
        env.close()


if __name__ == "__main__":
    main()
