"""Held-out thresholds and false-alarm counts.
Written before the code.

Until now every threshold was the worst no-signal hour of the same hours it was scored on,
so false alarms were zero by construction. Here a threshold is set on one set of hours
("set") and false alarms and detections are counted on a different set ("score").

Origin: verification@23d3a497
"""
import numpy as np
import pytest

from island_detectability_map import holdout


def test_threshold_is_the_worst_set_hour_and_does_not_depend_on_the_score_hours():
    null = np.array([1.0, 4.0, 2.0, 9.0, 3.0, 50.0])
    assert holdout.threshold(null, [0, 1, 2]) == 4.0
    null2 = null.copy(); null2[3:] = 1e6
    assert holdout.threshold(null2, [0, 1, 2]) == 4.0


def test_threshold_can_be_a_quantile_of_the_set_hours():
    null = np.arange(101, dtype=float)
    assert holdout.threshold(null, range(101), q=0.9) == pytest.approx(90.0)


def test_false_alarms_are_score_hours_above_the_threshold():
    null = np.array([1.0, 4.0, 2.0, 9.0, 3.0, 50.0, 4.0])
    fa = holdout.false_alarms(null, thr=4.0, score_idx=[3, 4, 5, 6])
    assert fa == [3, 5]                       # 4.0 itself is not above the threshold


def test_family_threshold_is_the_worst_over_the_family_series():
    nulls = {"phase1": np.array([1.0, 2.0, 9.0]), "phase2": np.array([5.0, 1.0, 9.0]), "phase3": np.array([2.0, 3.0, 9.0])}
    assert holdout.family_threshold(nulls, [0, 1]) == 5.0


def test_family_false_alarm_is_an_hour_where_any_series_exceeds():
    nulls = {"a": np.array([0, 0, 6.0, 0, 0]), "b": np.array([0, 0, 6.0, 7.0, 0]), "c": np.zeros(5)}
    assert holdout.family_false_alarms(nulls, thr=5.0, score_idx=[1, 2, 3, 4]) == [2, 3]   # hour 2 counted once


def test_detection_rate_uses_score_hours_only():
    inj = np.array([10.0, 10.0, 1.0, 1.0, 10.0, 1.0])
    assert holdout.detection_rate(inj, thr=5.0, score_idx=[2, 3, 4, 5]) == 0.25


def test_rate_interval_for_zero_events_is_the_rule_of_three():
    lo, hi = holdout.rate_interval(0, exposure_hours=100.0, conf=0.95)
    assert lo == 0.0
    assert hi == pytest.approx(-np.log(0.05) / 100.0, rel=1e-6)          # 2.996 / T, one-sided


def test_rate_interval_for_a_few_events_is_the_exact_poisson_interval():
    lo, hi = holdout.rate_interval(3, exposure_hours=10.0, conf=0.90)
    assert lo == pytest.approx(0.8177 / 10.0, rel=1e-3)                  # standard table values for k = 3
    assert hi == pytest.approx(7.754 / 10.0, rel=1e-3)


def test_swing_at_heldout_threshold_reads_the_score_hours():
    swings = [1.0, 2.0, 4.0]
    inj = np.array([[0, 0, 0, 0, 9, 9],          # swing 1: only set hours exceed
                    [0, 0, 9, 0, 9, 9],          # swing 2: one of four score hours
                    [0, 0, 9, 9, 9, 9]], float)  # swing 4: all... score hours are 0-3 here
    v = holdout.swing_for(inj, swings, thr=5.0, score_idx=[0, 1, 2, 3], level=0.5)
    assert v == pytest.approx(4.0)               # rates 0, 0.25, 0.5 -> reaches 0.5 exactly at swing 4


def test_contiguous_split_gives_each_half_every_class():
    labels = ["work"] * 10 + ["weeknight"] * 10 + ["weekend"] * 10 + ["work"] * 10 + ["weeknight"] * 10 + ["weekend"] * 10
    a, b = holdout.split_at(len(labels), 30)
    assert a == list(range(30)) and b == list(range(30, 60))
    by = holdout.by_class(labels, b)
    assert {k: len(v) for k, v in by.items()} == {"work": 10, "weeknight": 10, "weekend": 10}
