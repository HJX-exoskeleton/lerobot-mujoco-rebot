import torch

from rebot_aerohand_right_act_sim.multimodal_policy import (
    HandContactEncoder,
    IMUEncoder,
)


def test_sensor_encoder_shapes():
    assert IMUEncoder(32)(torch.zeros(4, 10)).shape == (4, 32)
    assert HandContactEncoder(32)(torch.zeros(4, 6)).shape == (4, 32)
