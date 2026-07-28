"""Adapter that lets the existing LeRobot trainer build MultimodalACTPolicy."""

from __future__ import annotations

import os

from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType

import train_model
from rebot_act_real.multimodal_policy import MultimodalACTPolicy


def _make_multimodal_policy(cfg, ds_meta=None, env_cfg=None):
    if ds_meta is None or env_cfg is not None:
        raise ValueError("reBot多模态ACT训练只支持离线真机数据集")
    features = dataset_to_policy_features(ds_meta.features)
    cfg.output_features = {
        key: value for key, value in features.items() if value.type is FeatureType.ACTION
    }
    cfg.input_features = {
        key: value for key, value in features.items() if key not in cfg.output_features
    }
    policy = MultimodalACTPolicy(
        cfg,
        dataset_stats=ds_meta.stats,
        use_imu=os.environ.get("REBOT_ACT_USE_IMU") == "1",
        use_tactile=os.environ.get("REBOT_ACT_USE_TACTILE") == "1",
        sensor_embed_dim=int(os.environ.get("REBOT_ACT_SENSOR_EMBED_DIM", "64")),
    )
    return policy.to(cfg.device)


def main() -> None:
    train_model.make_policy = _make_multimodal_policy
    train_model.train()


if __name__ == "__main__":
    main()
