"""Optional IMU/tactile encoders for the real reBot ACT policy."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType, PolicyFeature

MULTIMODAL_CONFIG_NAME = "rebot_multimodal.json"
SENSOR_FEATURE_KEY = "observation.environment_state"


class IMUEncoder(nn.Module):
    def __init__(self, output_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class TactileEncoder(nn.Module):
    """Preserve the 12x30 tactile layout instead of flattening it at input."""

    def __init__(self, output_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((3, 5)),
            nn.Flatten(),
            nn.Linear(32 * 3 * 5, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value.unsqueeze(1))


class MultimodalACTPolicy(ACTPolicy):
    """ACT with learned sensor encoders and an additional sensor token."""

    def __init__(
        self,
        config: ACTConfig,
        dataset_stats: dict | None = None,
        *,
        use_imu: bool,
        use_tactile: bool,
        sensor_embed_dim: int = 64,
    ):
        if not use_imu and not use_tactile:
            raise ValueError("MultimodalACTPolicy至少需要启用IMU或触觉")
        self.use_imu = bool(use_imu)
        self.use_tactile = bool(use_tactile)
        self.sensor_embed_dim = int(sensor_embed_dim)
        fusion_dim = self.sensor_embed_dim * (self.use_imu + self.use_tactile)
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
            self._register_sensor_stats("imu", stats["sensor.imu"])
        if self.use_tactile:
            self.tactile_encoder = TactileEncoder(self.sensor_embed_dim)
            self._register_sensor_stats("tactile", stats["sensor.tactile"])

    def _register_sensor_stats(self, name: str, stats: dict) -> None:
        mean = torch.as_tensor(stats["mean"], dtype=torch.float32)
        std = torch.as_tensor(stats["std"], dtype=torch.float32).clamp_min(1e-6)
        self.register_buffer(f"{name}_mean", mean)
        self.register_buffer(f"{name}_std", std)

    def _augment_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        embeddings: list[torch.Tensor] = []
        if self.use_imu:
            if "sensor.imu" not in batch:
                raise KeyError("多模态ACT checkpoint需要sensor.imu")
            imu = (batch["sensor.imu"].float() - self.imu_mean) / self.imu_std
            embeddings.append(self.imu_encoder(imu))
        if self.use_tactile:
            if "sensor.tactile" not in batch:
                raise KeyError("多模态ACT checkpoint需要sensor.tactile")
            tactile = (
                batch["sensor.tactile"].float() - self.tactile_mean
            ) / self.tactile_std
            embeddings.append(self.tactile_encoder(tactile))
        result = dict(batch)
        result[SENSOR_FEATURE_KEY] = torch.cat(embeddings, dim=-1)
        return result

    def forward(self, batch):
        return super().forward(self._augment_batch(batch))

    @torch.no_grad()
    def select_action(self, batch):
        return super().select_action(self._augment_batch(batch))

    def save_pretrained(self, save_directory, **kwargs):
        result = super().save_pretrained(save_directory, **kwargs)
        payload = {
            "format_version": 1,
            "use_imu": self.use_imu,
            "use_tactile": self.use_tactile,
            "sensor_embed_dim": self.sensor_embed_dim,
        }
        Path(save_directory, MULTIMODAL_CONFIG_NAME).write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        return result

    @classmethod
    def from_multimodal_pretrained(
        cls, checkpoint: str | Path, *, config: ACTConfig, dataset_stats: dict
    ) -> "MultimodalACTPolicy":
        payload = load_multimodal_spec(checkpoint)
        return cls.from_pretrained(
            checkpoint,
            config=config,
            dataset_stats=dataset_stats,
            use_imu=payload["use_imu"],
            use_tactile=payload["use_tactile"],
            sensor_embed_dim=payload["sensor_embed_dim"],
            local_files_only=True,
        )


def load_multimodal_spec(checkpoint: str | Path) -> dict:
    path = Path(checkpoint) / MULTIMODAL_CONFIG_NAME
    if not path.is_file():
        raise FileNotFoundError(f"多模态checkpoint缺少{path.name}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "use_imu": bool(payload["use_imu"]),
        "use_tactile": bool(payload["use_tactile"]),
        "sensor_embed_dim": int(payload.get("sensor_embed_dim", 64)),
    }


def is_multimodal_checkpoint(checkpoint: str | Path) -> bool:
    return (Path(checkpoint) / MULTIMODAL_CONFIG_NAME).is_file()
