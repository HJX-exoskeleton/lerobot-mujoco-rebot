"""ACT sensor fusion shared by simulation training and deployment."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType, PolicyFeature

MULTIMODAL_CONFIG_NAME = "rebot_sim_multimodal.json"
SENSOR_FEATURE_KEY = "observation.environment_state"


class IMUEncoder(nn.Module):
    def __init__(self, output_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, output_dim)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class TactileEncoder(nn.Module):
    """Encode native left/right tactile maps as two spatial channels."""

    def __init__(self, output_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((3, 5)),
            nn.Flatten(),
            nn.Linear(32 * 3 * 5, output_dim),
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
        use_tactile: bool,
        sensor_embed_dim: int = 64,
        sensor_dropout: float = 0.0,
        tactile_fusion_gain: float = 1.0,
    ):
        if not use_imu and not use_tactile:
            raise ValueError("multimodal ACT requires IMU and/or tactile input")
        self.use_imu = bool(use_imu)
        self.use_tactile = bool(use_tactile)
        self.sensor_embed_dim = int(sensor_embed_dim)
        self.sensor_dropout = float(sensor_dropout)
        self.tactile_fusion_gain = float(tactile_fusion_gain)
        if not 0.0 <= self.sensor_dropout < 1.0:
            raise ValueError("sensor_dropout must be in [0, 1)")
        if self.tactile_fusion_gain <= 0.0:
            raise ValueError("tactile_fusion_gain must be positive")
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
            self._register_stats("imu", stats["sensor.imu"])
        if self.use_tactile:
            self.tactile_encoder = TactileEncoder(self.sensor_embed_dim)
            self._register_stats("tactile_left", stats["sensor.tactile_left"])
            self._register_stats("tactile_right", stats["sensor.tactile_right"])

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
        if self.use_tactile:
            left = (
                batch["sensor.tactile_left"].float() - self.tactile_left_mean
            ) / self.tactile_left_std
            right = (
                batch["sensor.tactile_right"].float() - self.tactile_right_mean
            ) / self.tactile_right_std
            value = torch.stack([left, right], dim=1)
            embeddings.append(self.tactile_encoder(value) * self.tactile_fusion_gain)
        # Drop complete modality tokens, rather than individual taxels. In the
        # scripted simulator tactile contact is strongly correlated with task
        # phase; modality dropout prevents ACT from learning a tactile-only
        # open/contact lookup table and forces vision/state to remain useful.
        if self.training and self.sensor_dropout > 0.0:
            embeddings = [
                embedding
                * (
                    torch.rand(
                        (embedding.shape[0], 1),
                        device=embedding.device,
                        dtype=embedding.dtype,
                    )
                    >= self.sensor_dropout
                )
                / (1.0 - self.sensor_dropout)
                for embedding in embeddings
            ]
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
                    "format_version": 2,
                    "use_imu": self.use_imu,
                    "use_tactile": self.use_tactile,
                    "sensor_embed_dim": self.sensor_embed_dim,
                    "sensor_dropout": self.sensor_dropout,
                    "tactile_fusion_gain": self.tactile_fusion_gain,
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
        "use_tactile": bool(value["use_tactile"]),
        "sensor_embed_dim": int(value.get("sensor_embed_dim", 64)),
        "sensor_dropout": float(value.get("sensor_dropout", 0.0)),
        "tactile_fusion_gain": float(value.get("tactile_fusion_gain", 1.0)),
    }


def is_multimodal_checkpoint(checkpoint: str | Path) -> bool:
    return (Path(checkpoint) / MULTIMODAL_CONFIG_NAME).is_file()
