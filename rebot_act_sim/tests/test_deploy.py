from rebot_act_sim.workflow.deploy import _deployment_seed


def test_deployment_seed_defaults_to_random() -> None:
    assert _deployment_seed(None) is None
    assert _deployment_seed(-1) is None


def test_deployment_seed_preserves_reproducible_values() -> None:
    assert _deployment_seed(0) == 0
    assert _deployment_seed(17) == 17
