import torch

from rebot_act_sim.multimodal_policy import IMUEncoder, TactileEncoder


def test_sensor_encoder_shapes():
    assert IMUEncoder(32)(torch.zeros(4, 10)).shape == (4, 32)
    assert TactileEncoder(32)(torch.zeros(4, 2, 8, 16)).shape == (4, 32)
