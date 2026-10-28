"""Pieces of the UK feeder run (run_uk_feeders) that are not orchestration.
Written before the code.

The turn-on count is the power-side report's (section 2.3): on one-minute means, the mean of the
next W minutes minus the mean of the previous W minutes; every rise of at least `step` is a
false alarm for an island switching on, and exceedances within W minutes of each other are one event.

Origin: verification@776d1246
"""
import numpy as np
import pytest

from island_detectability_map import run_uk_feeders as U


def test_a_step_that_stays_up_is_one_jump():
    x = np.r_[np.zeros(60), np.full(60, 30.0)]
    assert U.jumps(x, w=15, step=25.0) == 1


def test_a_step_below_the_threshold_is_not_counted():
    x = np.r_[np.zeros(60), np.full(60, 20.0)]
    assert U.jumps(x, w=15, step=25.0) == 0


def test_a_five_minute_spike_does_not_lift_a_fifteen_minute_mean_enough():
    x = np.zeros(120); x[60:65] = 30.0
    assert U.jumps(x, w=15, step=25.0) == 0


def test_two_steps_far_apart_are_two_jumps():
    x = np.r_[np.zeros(60), np.full(60, 30.0), np.zeros(60), np.full(60, 30.0)]
    assert U.jumps(x, w=15, step=25.0) == 2


def test_minutes_with_no_data_do_not_make_a_jump():
    x = np.r_[np.full(60, 10.0), np.full(30, np.nan), np.full(60, 10.0)]
    assert U.jumps(x, w=15, step=5.0) == 0


def test_hours_touching_an_empty_second_are_not_usable():
    x = np.ones((3, 4 * 3600)); x[1, 3600 + 5] = np.nan
    assert U.usable_hours(x) == [0, 2, 3]


def test_rate_at_a_scale_is_read_off_the_log_scale_grid():
    scales = [0.25, 0.5, 1.0, 2.0]
    rates = [0.0, 0.2, 0.6, 1.0]
    assert U.rate_at(rates, scales, 0.5) == pytest.approx(0.2)
    assert U.rate_at(rates, scales, 2 ** -0.5) == pytest.approx(0.4)      # half way between 0.5 and 1 in log scale
