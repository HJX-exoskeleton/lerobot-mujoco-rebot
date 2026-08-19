from pathlib import Path

import glfw
import numpy as np
import pytest

from rebot_aerohand_right_act_sim.environment import (
    GRASP_MARKER_CLOSE_COLOR,
    GRASP_MARKER_FAR_COLOR,
    GRASP_MARKER_MID_COLOR,
    HAND_FINGER_NAMES,
    PIP_HEIGHT,
    PIP_WIDTH,
    ROTATION_STEP,
    HandClosureMapper,
    camera_overlay_rects,
    grasp_marker_color,
    snap_hand_target,
    success_hold_steps,
    teleop_arm_delta_from_keys,
)

SCENE_XML = (
    Path(__file__).resolve().parents[2]
    / "asset_rebot_aerohand_right"
    / "mujoco_xml"
    / "rebotarm_aerohand_act_cylinder.xml"
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
    action = teleop_arm_delta_from_keys({key})
    assert action[axis] == pytest.approx(expected)
    other_rotation_axes = {3, 4, 5} - {axis}
    assert all(action[index] == 0 for index in other_rotation_axes)


def test_translation_keys_are_screen_aligned():
    action = teleop_arm_delta_from_keys({glfw.KEY_R})
    assert action[2] > 0
    assert np.count_nonzero(action) == 1


def test_half_second_success_hold_at_50_hz():
    assert success_hold_steps(0.5, 50) == 25


def test_grasp_marker_color_bands():
    assert grasp_marker_color(0.0) == GRASP_MARKER_CLOSE_COLOR
    assert grasp_marker_color(0.06) == GRASP_MARKER_CLOSE_COLOR
    assert grasp_marker_color(0.061) == GRASP_MARKER_MID_COLOR
    assert grasp_marker_color(0.12) == GRASP_MARKER_MID_COLOR
    assert grasp_marker_color(0.121) == GRASP_MARKER_FAR_COLOR


def test_camera_overlay_rects_stack_within_window():
    rects = camera_overlay_rects(1400, 900, 2)
    assert len(rects) == 2
    # Both PIPs hug the right edge and stay inside the window.
    for x, y in rects:
        assert x + PIP_WIDTH <= 1400
        assert y + PIP_HEIGHT <= 900
    # The second PIP stacks below the first without overlapping it.
    assert rects[1][1] >= rects[0][1] + PIP_HEIGHT


def test_snap_hand_target_with_hysteresis(hand_mapper):
    open_c = hand_mapper.open_ctrl
    closed_c = hand_mapper.closed_ctrl
    # Values near an extreme snap to it.
    near_open = open_c + 0.1 * (closed_c - open_c)
    assert np.allclose(
        snap_hand_target(near_open, open_c, closed_c, previous=None), open_c
    )
    near_closed = closed_c - 0.1 * (closed_c - open_c)
    assert np.allclose(
        snap_hand_target(near_closed, open_c, closed_c, previous=None),
        closed_c,
    )
    # A mid value keeps the previous state (hysteresis): a single noisy frame
    # cannot flip the hand between open and closed.
    middle = (open_c + closed_c) / 2
    assert np.allclose(
        snap_hand_target(middle, open_c, closed_c, previous=open_c), open_c
    )
    assert np.allclose(
        snap_hand_target(middle, open_c, closed_c, previous=closed_c),
        closed_c,
    )


@pytest.fixture(scope="module")
def hand_mapper():
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    return HandClosureMapper(model)


def test_hand_open_close_mapping(hand_mapper):
    open_ctrl = hand_mapper.command_ctrl()
    assert open_ctrl[4] == pytest.approx(0.0)          # thumb abduction open
    assert np.allclose(open_ctrl[:4], hand_mapper.ctrl_max[:4])
    hand_mapper.set_all_closed()
    closed_ctrl = hand_mapper.command_ctrl()
    assert closed_ctrl[4] == pytest.approx(1.5)
    assert np.allclose(closed_ctrl[:4], hand_mapper.ctrl_min[:4])
    # Tendons: high length = open.
    assert np.all(open_ctrl[:4] > closed_ctrl[:4])
    hand_mapper.set_all_open()


def test_per_finger_toggle(hand_mapper):
    hand_mapper.set_all_open()
    hand_mapper.toggle_finger(2)  # middle finger -> actuator index 1
    ctrl = hand_mapper.command_ctrl()
    assert ctrl[1] == pytest.approx(hand_mapper.ctrl_min[1])
    assert ctrl[0] == pytest.approx(hand_mapper.ctrl_max[0])  # index untouched
    hand_mapper.toggle_finger(2)
    assert hand_mapper.command_ctrl()[1] == pytest.approx(
        hand_mapper.ctrl_max[1]
    )


def test_toggle_grasp_flips_all_fingers(hand_mapper):
    hand_mapper.set_all_open()
    hand_mapper.toggle_grasp()
    assert hand_mapper.closed.tolist() == [True] * len(HAND_FINGER_NAMES)
    hand_mapper.toggle_grasp()
    assert not np.any(hand_mapper.closed)
