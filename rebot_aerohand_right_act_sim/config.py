"""Configuration loading and ACT feature construction."""

from __future__ import annotations

from pathlib import Path

import yaml

from .schema import FPS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "aerohand_act_sim.yaml"


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    path = resolve_project_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"configuration does not exist: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("dataset", "environment", "policy", "training"):
        if key not in value:
            raise KeyError(f"configuration is missing section: {key}")
    dataset_fps = int(value["dataset"].get("fps", FPS))
    if dataset_fps != FPS:
        raise ValueError(
            f"simulation collection/control frequency must be {FPS} Hz, got {dataset_fps}"
        )
    policy = value["policy"]
    chunk = int(policy["chunk_size"])
    steps = int(policy["n_action_steps"])
    if chunk <= 0 or not 0 < steps <= chunk:
        raise ValueError(f"invalid action chunk: chunk_size={chunk}, n_action_steps={steps}")
    return value


def object_randomization_kwargs(raw: dict) -> dict:
    value = raw["environment"].get("object_randomization")
    if value is None:
        return {}
    required = {
        "target_object_position_center",
        "target_object_xy_half_range",
    }
    missing = required.difference(value)
    if missing:
        raise KeyError(f"object_randomization is missing: {sorted(missing)}")
    return {key: value[key] for key in required}


def hand_contact_processing_kwargs(raw: dict) -> dict:
    value = raw["environment"].get("hand_contact_processing")
    return {} if value is None else {"hand_contact_processing": dict(value)}


def teleop_kwargs(raw: dict) -> dict:
    value = raw["environment"].get("teleop")
    return {} if value is None else {
        "teleop_motion_alpha": float(value.get("motion_alpha", 0.25)),
        "hand_command_alpha": float(value.get("hand_command_alpha", 0.25)),
    }


def camera_overlay_kwargs(raw: dict) -> dict:
    value = raw["environment"].get("camera_overlays")
    return {} if value is None else {"camera_overlays": list(value)}


def build_act_config(raw: dict, metadata, *, n_action_steps: int | None = None):
    from lerobot.common.datasets.utils import dataset_to_policy_features
    from lerobot.common.policies.act.configuration_act import ACTConfig
    from lerobot.configs.types import FeatureType

    features = dataset_to_policy_features(metadata.features)
    allowed_inputs = {
        "observation.image",
        "observation.wrist_image",
        "observation.state",
    }
    inputs = {key: value for key, value in features.items() if key in allowed_inputs}
    missing = allowed_inputs.difference(inputs)
    if missing:
        raise ValueError(f"dataset is missing ACT inputs: {sorted(missing)}")
    outputs = {key: value for key, value in features.items() if value.type is FeatureType.ACTION}
    if set(outputs) != {"action"}:
        raise ValueError(f"expected one action output, got {sorted(outputs)}")
    policy = raw["policy"]
    return ACTConfig(
        input_features=inputs,
        output_features=outputs,
        chunk_size=int(policy["chunk_size"]),
        n_action_steps=int(n_action_steps or policy["n_action_steps"]),
        vision_backbone=str(policy.get("vision_backbone", "resnet18")),
        pretrained_backbone_weights=policy.get(
            "pretrained_backbone_weights", "ResNet18_Weights.IMAGENET1K_V1"
        ),
        use_vae=bool(policy.get("use_vae", True)),
        latent_dim=int(policy.get("latent_dim", 32)),
        kl_weight=float(policy.get("kl_weight", 10.0)),
        temporal_ensemble_coeff=policy.get("temporal_ensemble_coeff"),
        device=str(policy.get("device", "cuda")),
    )
