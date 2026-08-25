import pytest

from rebot_act_sim.workflow.collect import _AutoCollector, _smoothstep, build_argparser


def test_auto_collect_arguments() -> None:
    args = build_argparser().parse_args(
        ["--auto_collect", "--auto-speed", "1.2", "--episodes", "3"]
    )
    assert args.auto_collect is True
    assert args.auto_speed == pytest.approx(1.2)
    assert args.warmup_seconds == pytest.approx(1.0)
    assert args.episodes == 3


def test_auto_collect_uses_elevated_suspended_release() -> None:
    assert _AutoCollector.RELEASE_CLEARANCE == pytest.approx(0.045)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-1.0, 0.0), (0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (2.0, 1.0)],
)
def test_smoothstep_is_clamped(value: float, expected: float) -> None:
    assert _smoothstep(value) == pytest.approx(expected)
