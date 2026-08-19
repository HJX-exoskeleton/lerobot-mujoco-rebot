"""ACT sensor fusion shared by simulation training and deployment."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType, PolicyFeature

MULTIMODAL_CONFIG_NAME = "rebot_aerohand_right_multimodal.json"
SENSOR_FEATURE_KEY = "observation.environment_state"


class IMUEncoder(nn.Module):
    def __init__(self, output_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, output_dim)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class HandContactEncoder(nn.Module):
    """Encode the six per-region hand contact normal forces."""

    def __init__(self, output_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, output_dim)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class MultimodalACTPolicy(ACTPolicy):
    """Add one learned sensor token to the normal visual ACT encoder."""

    def __init__(
        self,
        config: ACTConfig,
        dataset_stats: dict | None = None,
        *,
        use_imu: bool,
        use_hand_contact: bool,
        sensor_embed_dim: int = 64,
    ):
        if not use_imu and not use_hand_contact:
            raise ValueError("multimodal ACT requires IMU and/or hand contact input")
        self.use_imu = bool(use_imu)
        self.use_hand_contact = bool(use_hand_contact)
        self.sensor_embed_dim = int(sensor_embed_dim)
        fusion_dim = self.sensor_embed_dim * (self.use_imu + self.use_hand_contact)
        config.input_features = dict(config.input_features)
        config.input_features[SENSOR_FEATURE_KEY] = PolicyFeature(
            type=FeatureType.ENV, shape=(fusion_dim,)
        )
        stats = dict(dataset_stats or {})
        stats[SENSOR_FEATURE_KEY] = {
            "mean": torch.zeros(fusion_dim),
            "std": torch.ones(fusion_dim),
            "min": -torch.ones(fusion_dim),
            "max": torch.ones(fusion_dim),
        }
        super().__init__(config, dataset_stats=stats)
        if self.use_imu:
            self.imu_encoder = IMUEncoder(self.sensor_embed_dim)
            self._register_stats("imu", stats["sensor.imu"])
        if self.use_hand_contact:
            self.hand_contact_encoder = HandContactEncoder(self.sensor_embed_dim)
            self._register_stats("hand_contact", stats["sensor.hand_contact"])

    def _register_stats(self, name: str, stats: dict) -> None:
        self.register_buffer(
            f"{name}_mean", torch.as_tensor(stats["mean"], dtype=torch.float32)
        )
        self.register_buffer(
            f"{name}_std",
            torch.as_tensor(stats["std"], dtype=torch.float32).clamp_min(1e-6),
        )

    def _augment(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        embeddings = []
        if self.use_imu:
            value = (batch["sensor.imu"].float() - self.imu_mean) / self.imu_std
            embeddings.append(self.imu_encoder(value))
        if self.use_hand_contact:
            value = (
                batch["sensor.hand_contact"].float() - self.hand_contact_mean
            ) / self.hand_contact_std
            embeddings.append(self.hand_contact_encoder(value))
        result = dict(batch)
        result[SENSOR_FEATURE_KEY] = torch.cat(embeddings, dim=-1)
        return result

    def forward(self, batch):
        return super().forward(self._augment(batch))

    @torch.no_grad()
    def select_action(self, batch):
        return super().select_action(self._augment(batch))

    def save_pretrained(self, save_directory, **kwargs):
        result = super().save_pretrained(save_directory, **kwargs)
        Path(save_directory, MULTIMODAL_CONFIG_NAME).write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "use_imu": self.use_imu,
                    "use_hand_contact": self.use_hand_contact,
                    "sensor_embed_dim": self.sensor_embed_dim,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return result


def load_multimodal_spec(checkpoint: str | Path) -> dict:
    path = Path(checkpoint) / MULTIMODAL_CONFIG_NAME
    if not path.is_file():
        raise FileNotFoundError(f"multimodal checkpoint metadata is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    return {
        "use_imu": bool(value["use_imu"]),
        "use_hand_contact": bool(value["use_hand_contact"]),
        "sensor_embed_dim": int(value.get("sensor_embed_dim", 64)),
    }


def is_multimodal_checkpoint(checkpoint: str | Path) -> bool:
    return (Path(checkpoint) / MULTIMODAL_CONFIG_NAME).is_file()
