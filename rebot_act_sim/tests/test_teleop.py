import glfw
import pytest

from rebot_act_sim.environment import (
    ROTATION_STEP,
    success_hold_steps,
    teleop_delta_from_keys,
)


@pytest.mark.parametrize(
    ("key", "axis", "expected"),
    [
        (glfw.KEY_UP, 4, -ROTATION_STEP),
        (glfw.KEY_DOWN, 4, ROTATION_STEP),
        (glfw.KEY_LEFT, 3, -ROTATION_STEP),
        (glfw.KEY_RIGHT, 3, ROTATION_STEP),
    ],
)
def test_arrow_key_rotation_semantics(key, axis, expected):
    action, _ = teleop_delta_from_keys({key}, gripper_state=False)
    assert action[axis] == pytest.approx(expected)
    other_rotation_axes = {3, 4, 5} - {axis}
    assert all(action[index] == 0 for index in other_rotation_axes)


def test_space_toggle_is_independent_of_motion():
    action, state = teleop_delta_from_keys(
        set(), gripper_state=False, toggle_gripper=True
    )
    assert state is True
    assert action[-1] == 1


def test_half_second_success_hold_at_50_hz():
    assert success_hold_steps(0.5, 50) == 25
