import pytest

from rebot_aerohand_right_act_sim.timing import WallClockRate


class FakeTime:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def test_rate_limiter_paces_50_hz_without_drift():
    fake = FakeTime()
    rate = WallClockRate(50, clock=fake.clock, sleeper=fake.sleep)
    for _ in range(10):
        fake.now += 0.005
        rate.wait()
    assert fake.now == pytest.approx(0.2)
    assert rate.deadline_misses == 0


def test_rate_limiter_does_not_catch_up_after_slow_frame():
    fake = FakeTime()
    rate = WallClockRate(50, clock=fake.clock, sleeper=fake.sleep)
    fake.now = 0.08
    lateness = rate.wait()
    assert lateness == pytest.approx(0.06)
    assert rate.deadline_misses == 1
    assert rate._next_deadline == pytest.approx(0.10)
