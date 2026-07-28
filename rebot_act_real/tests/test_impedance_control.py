from __future__ import annotations

import numpy as np

from rebot_act_real.workflow.deploy_impedance_control import (
    PolicyJointImpedanceController,
    build_argparser,
)


def test_impedance_is_default_for_dedicated_entrypoint():
    args = build_argparser().parse_args(["--shadow"])
    assert args.impedance_control is True
    assert args.gravity_compensation is True
    assert args.mit_vel_feedforward is False
    assert args.n_action_steps == 1


def test_zero_gravity_impedance_returns_only_position_error():
    controller = PolicyJointImpedanceController(
        gravity_compensation=False,
        gravity_scales=np.ones(6),
        torque_limits=np.full(6, 5.0),
    )
    result = controller.compute(np.zeros(6), np.ones(6))
    np.testing.assert_allclose(result["tau"], 0.0)
    np.testing.assert_allclose(result["tau_g"], 0.0)
    np.testing.assert_allclose(result["q_err"], 1.0)


def test_torque_limits_are_stored_per_joint():
    limits = np.asarray([10, 9, 8, 5, 4, 3], dtype=np.float64)
    controller = PolicyJointImpedanceController(
        gravity_compensation=False,
        gravity_scales=np.ones(6),
        torque_limits=limits,
    )
    np.testing.assert_allclose(controller.torque_limits, limits)
