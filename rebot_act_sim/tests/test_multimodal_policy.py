import json

import pytest
import torch

from rebot_act_sim.multimodal_policy import (
    IMUEncoder,
    MULTIMODAL_CONFIG_NAME,
    MultimodalACTPolicy,
    TactileEncoder,
    load_multimodal_spec,
)


def test_sensor_encoder_shapes():
    assert IMUEncoder(32)(torch.zeros(4, 10)).shape == (4, 32)
    assert TactileEncoder(32)(torch.zeros(4, 2, 8, 16)).shape == (4, 32)


def test_multimodal_spec_is_backward_compatible(tmp_path):
    (tmp_path / MULTIMODAL_CONFIG_NAME).write_text(
        json.dumps({"use_imu": True, "use_tactile": True, "sensor_embed_dim": 32})
    )
    assert load_multimodal_spec(tmp_path) == {
        "use_imu": True,
        "use_tactile": True,
        "sensor_embed_dim": 32,
        "sensor_dropout": 0.0,
        "tactile_fusion_gain": 1.0,
    }


def test_multimodal_regularization_parameters_are_validated():
    policy = MultimodalACTPolicy.__new__(MultimodalACTPolicy)
    # Validation occurs before ACT's heavy model is constructed.
    with pytest.raises(ValueError, match="sensor_dropout"):
        MultimodalACTPolicy.__init__(
            policy, None, use_imu=True, use_tactile=False, sensor_dropout=1.0
        )
    with pytest.raises(ValueError, match="tactile_fusion_gain"):
        MultimodalACTPolicy.__init__(
            policy, None, use_imu=True, use_tactile=False, tactile_fusion_gain=0.0
        )
