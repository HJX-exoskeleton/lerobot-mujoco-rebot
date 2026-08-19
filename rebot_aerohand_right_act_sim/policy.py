"""Policy construction helpers that keep train/deploy feature sets identical."""

from __future__ import annotations

from pathlib import Path

from lerobot.common.policies.act.modeling_act import ACTPolicy

from .multimodal_policy import (
    MultimodalACTPolicy,
    is_multimodal_checkpoint,
    load_multimodal_spec,
)


def make_policy(
    config,
    dataset_stats: dict,
    *,
    use_imu: bool = False,
    use_hand_contact: bool = False,
    sensor_embed_dim: int = 64,
):
    if use_imu or use_hand_contact:
        return MultimodalACTPolicy(
            config,
            dataset_stats=dataset_stats,
            use_imu=use_imu,
            use_hand_contact=use_hand_contact,
            sensor_embed_dim=sensor_embed_dim,
        )
    return ACTPolicy(config, dataset_stats=dataset_stats)


def load_policy(checkpoint: str | Path, config, dataset_stats: dict):
    checkpoint = Path(checkpoint)
    if is_multimodal_checkpoint(checkpoint):
        spec = load_multimodal_spec(checkpoint)
        return MultimodalACTPolicy.from_pretrained(
            checkpoint,
            config=config,
            dataset_stats=dataset_stats,
            local_files_only=True,
            **spec,
        )
    return ACTPolicy.from_pretrained(
        checkpoint,
        config=config,
        dataset_stats=dataset_stats,
        local_files_only=True,
    )
