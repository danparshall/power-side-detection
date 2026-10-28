"""Tests for island_detectability_map.turn_on: a change-in-level (turn-on / turn-off) detector on one-second
data, keeping the DC component.

Behaviour under test:
  * S_k(tau) = mean(x[tau+1 : tau+1+k]) - mean(x[tau-k : tau]) on a grid tau_j = j k/2; a step on a grid
    point is measured at its height, an off-grid step loses at most a quarter of it;
  * the formula is run_sync.upsteps' at the grid times; up and down directions are separated;
  * a zero-mean pulse train barely moves S_k, a step of the same size moves it fully;
  * a side with fewer than 90% finite samples gives NaN; otherwise the mean of the finite samples;
  * injecting a placed job equals computing S_k from scratch on (x + coef job); points whose spans miss
    the run keep their no-signal value exactly;
  * spread and thresholds come only from the learning points;
  * a run's turn-on (turn-off) is credited only to an up (down) alarm within 60 s + k of that edge and
    nearer to it than to the other edge; chance is the same rule on the no-signal alarms;
  * false alarms are merged into events (one per hour, keeping the first) and counted per day.

Origin: verification@4019f777
"""
import numpy as np
import pytest

from island_detectability_map import run_sync
from island_detectability_map import strided as SW
from island_detectability_map import turn_on as TO


def _step(n, t0, h):
    """h from sample t0 on (x[t0] is the first high sample)."""
    return np.where(np.arange(n) >= t0, float(h), 0.0)


def _at(taus, S, tau):
    return S[np.flatnonzero(taus == tau)[0]]


# --- the statistic ---------------------------------------------------------------------------------

def test_step_on_a_grid_point_is_measured_at_its_height():
    k, h = 120, 5.0
    rng = np.random.default_rng(3)
    x = rng.normal(size=20 * k)
    tau = 10 * (k // 2)                              # a grid point; the step's first high sample is tau + 1
    x = x + _step(x.size, tau + 1, h)
    taus, S = TO.steps(x, k)
    assert abs(_at(taus, S, tau) - h) < 4 * np.sqrt(2 / k)


def test_off_grid_step_loses_at_most_a_quarter():
    k, h, n = 600, 7.0, 40 * 600
    rng = np.random.default_rng(5)
    for t0 in rng.integers(2 * k, n - 2 * k, size=25):
        taus, S = TO.steps(_step(n, int(t0), h), k)
        best = np.nanmax(S)
        assert 0.75 * h - 1e-9 <= best <= h + 1e-9


def test_grid_is_every_half_window_with_both_spans_inside_the_record():
    k, n = 120, 1000
    taus, S = TO.steps(np.zeros(n), k)
    assert np.all(np.diff(taus) == k // 2)
    assert np.all(taus % (k // 2) == 0)
    assert taus.min() >= k and taus.max() + 1 + k <= n
    assert taus.min() - k // 2 < k and taus.max() + k // 2 + 1 + k > n


def test_agrees_with_run_sync_upsteps_at_the_grid_times():
    k = 300
    x = np.random.default_rng(7).normal(size=30 * k).cumsum()
    taus, S = TO.steps(x, k)
    t_ref, s_ref = run_sync.upsteps(x, k)
    ref = dict(zip(t_ref.tolist(), s_ref.tolist()))
    common = [i for i, t in enumerate(taus) if t in ref]
    assert len(common) >= len(taus) - 1
    np.testing.assert_allclose(S[common], [ref[taus[i]] for i in common], rtol=1e-9, atol=1e-9)


def test_down_step_raises_down_not_up_and_up_step_the_reverse():
    k, n = 300, 40 * 300
    base = np.random.default_rng(11).normal(size=n)
    _, S0 = TO.steps(base, k)
    _, Sd = TO.steps(base - _step(n, 6001, 4.0), k)
    _, Su = TO.steps(base + _step(n, 6001, 4.0), k)
    assert np.nanmax(Sd) <= np.nanmax(S0) + 1e-9           # up statistic not raised by a down-step
    assert np.nanmax(-Sd) > np.nanmax(-S0) + 2.0            # down statistic raised
    assert np.nanmax(-Su) <= np.nanmax(-S0) + 1e-9
    assert np.nanmax(Su) > np.nanmax(S0) + 2.0


def test_pulse_train_barely_moves_the_statistic_but_a_step_does():
    k, n = 120, 60 * 120
    period, swing = 10, 5.0
    sq = np.where(np.arange(n) % period < period // 2, swing / 2, -swing / 2)
    on = np.arange(n) >= 3000
    _, S_pulse = TO.steps(np.where(on, sq, 0.0), k)
    _, S_step = TO.steps(_step(n, 3000, swing), k)
    assert np.nanmax(np.abs(S_pulse)) < 0.5
    assert np.nanmax(S_step) >= 0.75 * swing


# --- missing samples -------------------------------------------------------------------------------

def test_side_with_less_than_ninety_percent_finite_is_nan():
    k = 120
    x = np.random.default_rng(2).normal(size=20 * k)
    tau = 8 * (k // 2)
    y = x.copy()
    y[tau - k: tau - k + 13] = np.nan                       # 13 of 120 missing on the before side (89%)
    taus, S = TO.steps(y, k)
    assert np.isnan(_at(taus, S, tau))


def test_side_with_five_percent_missing_uses_the_finite_mean():
    k = 120
    x = np.random.default_rng(4).normal(size=20 * k)
    tau = 8 * (k // 2)
    y = x.copy()
    gone = np.arange(tau + 1, tau + 1 + 6)                  # 6 of 120 missing after (5%)
    y[gone] = np.nan
    after = np.delete(x[tau + 1: tau + 1 + k], np.arange(6))
    expect = after.mean() - x[tau - k: tau].mean()
    taus, S = TO.steps(y, k)
    assert _at(taus, S, tau) == pytest.approx(expect, rel=1e-9, abs=1e-12)


# --- injection -------------------------------------------------------------------------------------

def test_injection_equals_computing_from_scratch():
    k = 300
    rng = np.random.default_rng(9)
    n = 12 * 3600
    x = rng.normal(size=n) * 3 + 50
    x[rng.integers(0, n, 400)] = np.nan
    trace = rng.normal(size=777) + 10
    job = SW.placed(trace, t0=4 * 3600 + 17, dur=3600, phase=123, add=2.5)
    coef = 0.37
    taus_all, _ = TO.steps(x, k)
    near = taus_all[np.abs(taus_all - (4 * 3600 + 17)) <= 60 + k]
    got = TO.injected_steps(x, k, near, job, coef)
    y = x + coef * job(np.arange(n))
    taus_y, S_y = TO.steps(y, k)
    want = np.array([_at(taus_y, S_y, t) for t in near])
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)


def test_points_whose_spans_miss_the_run_keep_their_null_exactly():
    k = 600
    rng = np.random.default_rng(10)
    n = 12 * 3600
    x = rng.normal(size=n)
    t0, dur = 5 * 3600 + 101, 3600
    job = SW.placed(rng.normal(size=500) + 8, t0=t0, dur=dur, phase=0)
    taus, S0 = TO.steps(x, k)
    far = taus[(taus + 1 + k <= t0) | (taus - k >= t0 + dur)]
    got = TO.injected_steps(x, k, far, job, 1.0)
    want = np.array([_at(taus, S0, t) for t in far])
    assert np.array_equal(got, want)


# --- learning ---------------------------------------------------------------------------------------

def test_spread_and_thresholds_come_only_from_the_learning_points():
    k = 120
    rng = np.random.default_rng(12)
    n = 48 * 3600
    x = rng.normal(size=n)
    taus, S = TO.steps(x, k)
    learn = TO.span_ok(taus, k, set(range(24)))
    cal = TO.calibrate([S], learn)
    y = x.copy()
    y[30 * 3600:] += rng.normal(size=n - 30 * 3600) * 50 + _step(n - 30 * 3600, 3600, 1e3)
    _, Sy = TO.steps(y, k)
    cal_y = TO.calibrate([Sy], learn)
    assert cal_y["sigma"] == cal["sigma"]
    assert cal_y["up"] == cal["up"] and cal_y["down"] == cal["down"]


def test_thresholds_are_the_worst_learning_z_per_direction():
    S = np.array([1.0, -2.0, 0.5, 3.0, -9.0, 40.0])
    learn = np.array([True, True, True, True, True, False])
    cal = TO.calibrate([S], learn)
    sig = 1.4826 * np.median(np.abs(S[learn] - np.median(S[learn])))
    assert cal["sigma"][0] == pytest.approx(sig)
    assert cal["up"] == pytest.approx(3.0 / sig)
    assert cal["down"] == pytest.approx(9.0 / sig)


def test_family_threshold_is_the_worst_learning_max_over_channels():
    a = np.array([1.0, -1.0, 2.0, -2.0, 0.0, 5.0])
    b = np.array([0.0, 4.0, -1.0, 1.0, -6.0, 0.0])
    learn = np.array([True, True, True, True, True, False])
    cal = TO.calibrate([a, b], learn)
    up, down = TO.family([a, b], cal["sigma"])
    assert cal["up"] == pytest.approx(np.max(up[learn]))
    assert cal["down"] == pytest.approx(np.max(down[learn]))
    assert up[1] == pytest.approx(max(-1.0 / cal["sigma"][0], 4.0 / cal["sigma"][1]))


def test_span_ok_needs_both_spans_wholly_in_the_hours():
    k = 600
    taus = np.array([3600 - 300, 3600 + 600, 7200 - 601, 7200 - 600, 7200 - 599])
    ok = TO.span_ok(taus, k, {1})
    # spans [tau-k, tau) and [tau+1, tau+1+k) must lie in [3600, 7200)
    assert ok.tolist() == [False, True, True, False, False]


# --- scoring -------------------------------------------------------------------------------------

def test_turn_on_is_credited_only_to_an_up_alarm_near_the_start():
    t0, dur = 100_000, 6 * 3600
    tau = np.array([t0 - 50, t0 + 3 * 3600, t0 + 10, t0 + dur + 20], float)
    k = np.array([120, 120, 300, 300])
    up = np.array([False, True, False, False])
    down = np.array([False, False, True, True])
    out = TO.edge_outcome(tau, k, up, down, t0, dur)
    assert not out["on"]                                    # mid-run up alarm and a down alarm near t0
    assert out["off"]
    up2 = np.array([True, False, False, False])
    assert TO.edge_outcome(tau, k, up2, down, t0, dur)["on"]


def test_tolerance_is_sixty_seconds_plus_k():
    t0, dur = 50_000, 3 * 3600
    for k, d, expect in ((120, 180, True), (120, 181, False), (3600, 3660, True), (3600, 3661, False)):
        r = TO.edge_outcome(np.array([t0 - d], float), np.array([k]), np.array([True]), np.array([False]), t0, dur)
        assert r["on"] is expect


def test_an_alarm_nearer_the_other_edge_is_not_credited():
    t0, dur, k = 20_000, 3600, 3600
    tau = np.array([t0 + 3000.0])                            # within 60 + k of both edges, nearer the stop
    r = TO.edge_outcome(tau, np.array([k]), np.array([True]), np.array([True]), t0, dur)
    assert not r["on"] and r["off"]


def test_either_and_the_first_raised_hit():
    t0, dur = 10_000, 3600
    tau = np.array([t0 + 30.0, t0 - 100.0])
    k = np.array([600, 120])
    r = TO.edge_outcome(tau, k, np.array([True, True]), np.array([False, False]), t0, dur)
    assert r["on"] and not r["off"] and r["either"]
    assert r["on_hit"] == (t0 - 100.0, 120)                 # raised at t0 - 100 + 121, before t0 + 30 + 601
    assert r["on_delay_s"] == pytest.approx(-100 + 1 + 120)


def test_events_merge_alarms_within_an_hour_and_count_per_day():
    times = np.array([0.0, 20 * 60, 50 * 60, 3 * 3600])
    assert TO.events(times) == 2
    assert TO.events_per_day(times, days=0.5) == pytest.approx(4.0)
    assert TO.events(np.array([])) == 0


# --- hardening after the test-quality review ------------------------------------------------------

def test_injection_is_right_over_the_whole_run_not_just_near_the_edges():
    k = 120
    rng = np.random.default_rng(21)
    n = 6 * 3600
    x = rng.normal(size=n) + 20
    t0, dur = 2 * 3600 + 7, 3600
    job = SW.placed(rng.normal(size=97) * 3 + 6, t0=t0, dur=dur, phase=5, add=1.5)
    taus, _ = TO.steps(x, k)
    got = TO.injected_steps(x, k, taus, job, 0.8)
    _, want = TO.steps(x + 0.8 * job(np.arange(n)), k)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)


def test_turn_on_tolerance_after_the_edge_is_also_sixty_plus_k():
    t0, dur, k = 50_000, 6 * 3600, 600
    for d, expect in ((660, True), (661, False)):
        r = TO.edge_outcome(np.array([t0 + d], float), np.array([k]), np.array([True]), np.array([False]), t0, dur)
        assert r["on"] is expect


def test_calibration_ignores_nan_points():
    S = np.array([1.0, np.nan, -2.0, 0.5, 3.0, -9.0])
    learn = np.ones(6, bool)
    cal = TO.calibrate([S], learn)
    f = S[np.isfinite(S)]
    sig = 1.4826 * np.median(np.abs(f - np.median(f)))
    assert cal["sigma"][0] == pytest.approx(sig)
    assert cal["up"] == pytest.approx(3.0 / sig) and cal["down"] == pytest.approx(9.0 / sig)


def test_ninety_percent_rule_on_both_sides():
    k = 120
    x = np.random.default_rng(22).normal(size=20 * k)
    tau = 8 * (k // 2)
    y = x.copy()
    y[tau - k: tau - k + 12] = np.nan                       # 108 of 120 finite before: allowed
    taus, S = TO.steps(y, k)
    assert np.isfinite(_at(taus, S, tau))
    y = x.copy()
    y[tau + 1: tau + 1 + 13] = np.nan                       # 107 of 120 finite after: NaN
    taus, S = TO.steps(y, k)
    assert np.isnan(_at(taus, S, tau))


def test_turn_off_hit_and_delay_are_measured_from_the_stop():
    t0, dur = 100_000, 6 * 3600
    stop = t0 + dur
    r = TO.edge_outcome(np.array([stop - 40.0]), np.array([300]), np.array([False]), np.array([True]), t0, dur)
    assert r["off"] and not r["on"]
    assert r["off_hit"] == (stop - 40.0, 300)
    assert r["off_delay_s"] == pytest.approx(-40 + 1 + 300)
    assert r["off_loc_s"] == pytest.approx(40)


def test_first_hit_is_the_first_raised_not_the_earliest_tau():
    t0, dur = 30_000, 6 * 3600
    tau = np.array([t0 - 200.0, t0 + 30.0])
    k = np.array([3600, 120])
    r = TO.edge_outcome(tau, k, np.array([True, True]), np.array([False, False]), t0, dur)
    assert r["on_hit"] == (t0 + 30.0, 120)                  # raised t0 + 151, before t0 - 200 + 3601


def test_an_alarm_equidistant_from_both_edges_counts_for_both():
    t0, dur, k = 10_000, 2 * 3600, 3600
    tau = np.array([t0 + 3600.0])
    r = TO.edge_outcome(tau, np.array([k]), np.array([True]), np.array([True]), t0, dur)
    assert r["on"] and r["off"]


def test_events_sort_their_input_and_measure_from_the_events_first_alarm():
    assert TO.events(np.array([3 * 3600, 0.0, 50 * 60, 20 * 60])) == 2
    assert TO.events(np.array([0.0, 3000, 6000])) == 2       # 6000 is an hour after 0, though 3000 is nearer


def test_span_ok_across_clock_hours():
    k = 600
    tau = np.array([7200 - 100])                            # spans [6500, 7100) and [7101, 7701)
    assert TO.span_ok(tau, k, {1, 2}).tolist() == [True]
    k = 3600
    tau = np.array([3 * 3600])                              # spans [7200, 10800) and [10801, 14401)
    assert TO.span_ok(tau, k, {1, 3}).tolist() == [False]
    assert TO.span_ok(tau, k, {2, 3, 4}).tolist() == [True]


def test_family_normalizes_each_channel_by_its_own_spread():
    a = np.array([1.0, -1.0, 2.0, -2.0, 0.0])
    b = 10 * np.array([0.5, 1.0, -1.5, 1.0, -3.0])
    learn = np.ones(5, bool)
    cal = TO.calibrate([a, b], learn)
    assert cal["sigma"][0] != pytest.approx(cal["sigma"][1])
    up, down = TO.family([a, b], cal["sigma"])
    np.testing.assert_allclose(up, np.maximum(a / cal["sigma"][0], b / cal["sigma"][1]))
    np.testing.assert_allclose(down, np.maximum(-a / cal["sigma"][0], -b / cal["sigma"][1]))


def test_a_nan_channel_does_not_mute_the_family():
    up, down = TO.family([np.array([np.nan, 1.0]), np.array([2.0, np.nan])], [1.0, 1.0])
    assert up.tolist() == [2.0, 1.0] and down.tolist() == [-2.0, -1.0]
    up, _ = TO.family([np.array([np.nan]), np.array([np.nan])], [1.0, 1.0])
    assert np.isnan(up[0])


def test_alarms_apply_the_margin_strictly_and_never_on_nan():
    cal = {"up": 2.0, "down": 3.0}
    up = np.array([4.0, 4.1, np.nan, 1.0])
    down = np.array([6.0, 6.5, 9.0, np.nan])
    au, ad = TO.alarms(up, down, cal, 2.0)
    assert au.tolist() == [False, True, False, False]      # threshold 4.0, strict
    assert ad.tolist() == [False, True, True, False]       # threshold 6.0, strict
    au1, _ = TO.alarms(up, down, cal, 1.0)
    assert au1.tolist() == [True, True, False, False]
