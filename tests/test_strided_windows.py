"""Tests for island_detectability_map.strided: the report's spectral detector on one-hour windows at any
stride, with a job placed at an arbitrary start for a finite run (or filling the record), plus the
per-window, per-hour and per-run counting used to score overlapping windows.

Behaviour under test:
  * at stride = window, with the job looped from the report's random starts, the statistic is
    welch_engine.experiment's (null and injected), including segments skipped for non-finite samples;
  * a placed job's statistic equals the plain Welch-ratio statistic of (background + job) in that window;
  * a window the run does not touch scores exactly its no-signal statistic;
  * the job's mean (a step) raises a window that straddles the run's start, and changes nothing in a
    window wholly inside the run;
  * windows are classed by how they meet the run (straddle start, inside, straddle stop, both, outside);
  * false alarms over overlapping windows are counted per window and per distinct clock hour;
  * a run is found if any window it touches is found; its chance is the same windows' no-signal alarms;
    the delay runs from the start to the end of the first found window.

Origin: verification@5941aa10
"""
import numpy as np
import pytest

from island_detectability_map import pulse, welch_engine as WE
from island_detectability_map import strided as SW

FS = 1.0
W = 3600
FAST = dict(band=(0.08, 0.5), seg_s=100.0)


def _noise(hours, seed=1, line_hz=0.13):
    rng = np.random.default_rng(seed)
    t = np.arange(hours * W)
    return rng.normal(size=t.size) + 0.5 * np.sin(2 * np.pi * line_hz * t)


def _square(period, n_periods, swing=1.0, offset=0.0):
    one = np.where(np.arange(period) < period // 2, swing, 0.0)
    return np.tile(one, n_periods) + offset


# --- equivalence with the existing engine ----------------------------------------------------------

def _with_intermittent_line(x, hour, hz=0.21, amp=3.0):
    """A site line present in one hour only, so the learning windows notch it."""
    x = x.copy()
    t = np.arange(hour * W, (hour + 1) * W)
    x[t] += amp * np.sin(2 * np.pi * hz * t)
    return x


def test_stride_equal_to_window_reproduces_welch_engine_experiment():
    x = _with_intermittent_line(_noise(10), hour=2)
    x[5 * W + 1234: 5 * W + 1300] = np.nan             # a gap: segments that touch it are skipped
    tr = _square(11, 70, 0.8)
    tr = tr - tr.mean()
    learn = list(range(5))
    swings = [0.5, 1.0, 2.0]
    seed = 7
    e = WE.experiment(x, FS, W, tr, seed, swings, learn, char_hours=0, f0_range=(0.1, 0.4), **FAST)
    assert e["notched_bins"] > 0

    st = SW.prepare(x, FS, W, W, learn, **FAST)
    np.testing.assert_array_equal(st["keep"], e["keep"])
    np.testing.assert_allclose(st["null"], e["null_stats"], rtol=1e-10)
    starts = SW.report_draws(seed, len(swings), len(st["starts"]), len(tr), (0.1, 0.4))
    for j, s in enumerate(swings):
        for i, ws in enumerate(st["starts"]):
            job = SW.placed(tr, phase=int(starts[j, i]) - int(ws))
            assert SW.statistic(st, i, job, s) == pytest.approx(e["inj_stats"][j, i], rel=1e-10)


def test_report_draws_are_welch_engine_starts():
    x = _noise(6)
    tr = _square(11, 70, 0.8)
    ref = WE.prepare(x, FS, W, tr, 11, 4, list(range(3)), 0, f0_range=(0.02, 0.1), **FAST)["starts"]
    np.testing.assert_array_equal(SW.report_draws(11, 4, 6, len(tr), (0.02, 0.1)), ref)


# --- the strided baseline and null against a direct Welch per window --------------------------------

def test_strided_baseline_and_null_equal_direct_welch_per_window():
    x = _with_intermittent_line(_noise(8, seed=5), hour=1)
    x[2 * W + 700: 2 * W + 760] = np.nan
    learn = list(range(20))                            # 7 of the 20 overlap the line's hour
    st = SW.prepare(x, FS, W, 900, learn, **FAST)
    assert st["notch"].any()
    spectra = np.array([pulse.welch_psd(x[s:s + W], FS, FAST["seg_s"])[1] for s in st["starts"]])
    f = np.fft.rfftfreq(int(FAST["seg_s"]), 1.0)
    sel = (f >= FAST["band"][0]) & (f <= FAST["band"][1]) & (f < 0.5)   # welch_engine's scale doubles the
    base = np.median(spectra[learn], axis=0)                                # Nyquist bin too; ratios are unaffected
    np.testing.assert_allclose(st["baseline"][sel], base[sel], rtol=1e-10)
    np.testing.assert_allclose(st["null"], (spectra[:, st["keep"]] / base[st["keep"]]).max(axis=1), rtol=1e-10)


# --- the placed job ------------------------------------------------------------------------------------

def test_placed_job_is_the_looped_trace_from_its_start_until_its_stop():
    job = SW.placed(np.array([1.0, 2.0, 3.0]), t0=10, dur=5, phase=0)
    np.testing.assert_array_equal(job(np.arange(8, 17)), [0, 0, 1, 2, 3, 1, 2, 0, 0])
    job = SW.placed(np.array([1.0, 2.0, 3.0]), t0=10, dur=5, phase=2, add=10.0)
    np.testing.assert_array_equal(job(np.arange(8, 17)), [0, 0, 13, 11, 12, 13, 11, 0, 0])


def test_job_filling_the_record_is_the_looped_trace_on_the_record_clock():
    job = SW.placed(np.array([1.0, 2.0, 3.0]), phase=1)
    np.testing.assert_array_equal(job(np.arange(0, 7)), [2, 3, 1, 2, 3, 1, 2])


# --- injection is linear, and absent where the run is ----------------------------------------------

def test_placed_job_statistic_equals_direct_welch_of_background_plus_job():
    x = _noise(8, seed=2)
    x[6 * W + 300: 6 * W + 340] = np.nan
    tr = _square(13, 300, 3.0, offset=5.0)
    st = SW.prepare(x, FS, W, 900, list(range(12)), **FAST)
    job = SW.placed(tr, t0=5 * W + 1777, dur=W + 500, phase=40)    # stops at 7.63 h, inside the record
    checked = 0
    for i in np.flatnonzero((st["starts"] > 4 * W) & (st["starts"] <= 7 * W)):
        ws = int(st["starts"][i])
        y = x[ws:ws + W] + job(np.arange(ws, ws + W))
        _, p = pulse.welch_psd(y, FS, FAST["seg_s"])
        direct = float(np.max(p[st["keep"]] / st["baseline"][st["keep"]]))
        assert SW.statistic(st, i, job, 1.0) == pytest.approx(direct, rel=1e-9)
        checked += 1
    assert checked >= 10


def test_job_dfts_from_one_channel_score_another_with_different_notches():
    plain = _noise(8, seed=6)
    lined = _with_intermittent_line(plain, hour=1)
    learn = list(range(20))
    a, b = SW.prepare(lined, FS, W, 900, learn, **FAST), SW.prepare(plain, FS, W, 900, learn, **FAST)
    assert a["keep"].sum() != b["keep"].sum()
    job = SW.placed(_square(13, 300, 3.0, offset=5.0), t0=5 * W + 100, dur=W, phase=3)
    i = int(np.flatnonzero((a["starts"] < 5 * W + 100) & (a["starts"] + W > 5 * W + 100))[0])
    assert SW.stat_from(b, i, SW.job_dfts(a, i, job), 0.7) == pytest.approx(SW.statistic(b, i, job, 0.7), rel=1e-12)


def test_windows_the_run_does_not_touch_score_their_null():
    x = _noise(8, seed=3)
    st = SW.prepare(x, FS, W, 900, list(range(12)), **FAST)
    job = SW.placed(_square(13, 300, 3.0, offset=5.0), t0=4 * W, dur=W, phase=0)
    before = int(np.flatnonzero(st["starts"] + W <= 4 * W)[-1])     # ends exactly where the run starts
    after = int(np.flatnonzero(st["starts"] >= 5 * W)[0])          # starts exactly where the run stops
    for i in (before, after):
        assert SW.statistic(st, i, job, 1.0) == st["null"][i]
    assert SW.statistic(st, after - 1, job, 1.0) != st["null"][after - 1]


def test_job_mean_raises_window_straddling_the_start_and_cancels_inside():
    x = _noise(10, seed=4, line_hz=0.0)
    whole = _square(12, 1000, 0.3, offset=200.0)          # the step reaches only the ~2 of 71 segments holding it
    swing = whole - whole.mean()
    st = SW.prepare(x, FS, W, 900, list(range(16)), **FAST)
    t0 = 5 * W + 1111
    on = dict(t0=t0, dur=3 * W, phase=0)
    i_start = int(np.flatnonzero((st["starts"] < t0) & (st["starts"] + W > t0))[0])
    i_in = int(np.flatnonzero((st["starts"] >= t0) & (st["starts"] + W <= t0 + 3 * W))[0])
    s_whole = SW.statistic(st, i_start, SW.placed(whole, **on), 1.0)
    s_swing = SW.statistic(st, i_start, SW.placed(swing, **on), 1.0)
    assert s_whole > 2 * s_swing
    assert SW.statistic(st, i_in, SW.placed(whole, **on), 1.0) == pytest.approx(
        SW.statistic(st, i_in, SW.placed(swing, **on), 1.0), rel=1e-9)


# --- classing windows against a run -----------------------------------------------------------------

def test_windows_are_classed_by_how_they_meet_the_run():
    starts = np.array([0, 900, 1800, 4500, 4600, 9000])
    got = SW.window_class(starts, W, t0=1000, dur=3600)
    assert list(got) == ["start", "start", "stop", "stop", "outside", "outside"]
    assert list(SW.window_class(np.array([0, 1800]), W, t0=1000, dur=1800)) == ["both", "stop"]
    assert list(SW.window_class(np.array([1000, 2000, 5000]), W, t0=1000, dur=6 * W)) == ["inside"] * 3
    got = SW.window_class(np.array([0, 4 * W]), W, t0=W, dur=4 * W)       # ends at the start; ends at the stop
    assert list(got) == ["outside", "inside"]


# --- counting alarms over overlapping windows -------------------------------------------------------

def test_false_alarms_counted_per_window_and_per_distinct_hour():
    starts = np.arange(0, 6 * W - W + 1, 900)           # 21 windows over 6 hours
    alarm = np.zeros(len(starts), bool)
    alarm[[0, 1, 4, 20]] = True                          # [0,1h), [0.25h,1.25h), [1h,2h), [5h,6h)
    per_window, per_hour = SW.alarm_counts(alarm, starts, W, W)
    assert per_window == 4
    assert per_hour == 3                                 # hours 0, 1 and 5


def test_an_off_grid_alarm_counts_for_both_hours_it_overlaps():
    starts = np.arange(0, 4 * W - W + 1, 900)
    alarm = np.zeros(len(starts), bool)
    alarm[2] = True                                      # [0.5h, 1.5h): hours 0 and 1
    assert SW.alarm_counts(alarm, starts, W, W) == (1, 2)


# --- a run's outcome ---------------------------------------------------------------------------------

def test_run_found_if_any_touched_window_is_found_with_delay_to_that_window_end():
    starts = np.arange(0, 10 * W, 900)
    t0, dur = 2 * W + 100, W                             # run [7300, 10900)
    found = np.zeros(len(starts), bool)
    null_alarm = np.zeros(len(starts), bool)
    found[[3, 7, 10, 15]] = True                         # [2700, 6300), [13500, 17100) miss it; 7 and 10 touch
    null_alarm[0] = True                                 # [0, 3600) does not touch the run
    out = SW.run_outcome(found, null_alarm, starts, W, t0, dur)
    assert out["found"]
    assert not out["chance"]
    assert out["delay_s"] == 6300 + W - t0               # the first found window, not a later one


def test_window_ending_exactly_at_the_start_does_not_find_the_run():
    starts = np.arange(0, 10 * W, 900)
    found = starts + W == 2 * W                          # only [3600, 7200) is found; the run starts at 7200
    out = SW.run_outcome(found, np.zeros(len(starts), bool), starts, W, 2 * W, W)
    assert not out["found"]


def test_run_not_found_when_only_untouched_windows_clear_and_chance_uses_touched_windows():
    starts = np.arange(0, 10 * W, 900)
    t0, dur = 4 * W, 2 * W
    found = np.zeros(len(starts), bool)
    found[0] = True
    null_alarm = np.zeros(len(starts), bool)
    null_alarm[int(np.flatnonzero(starts == 5 * W)[0])] = True
    out = SW.run_outcome(found, null_alarm, starts, W, t0, dur)
    assert not out["found"]
    assert out["chance"]
    assert np.isnan(out["delay_s"])
