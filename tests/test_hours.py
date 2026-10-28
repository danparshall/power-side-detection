"""Hour-of-day split for the current-monitor detectors.
Written before the code.

Origin: verification@67698f42
"""
import numpy as np

from island_detectability_map import hours, run_pulse

T0_NS = 1443657600 * 10**9          # 2015-10-01 00:00 UTC = Wed 2015-09-30 17:00 PDT (start of the LBNL record)


def test_hour_labels_follow_local_time_not_utc():
    lab = hours.hour_labels(T0_NS, 75)
    assert lab[0] == "work"          # Wed 17:00 PDT
    assert lab[1] == "work"          # Wed 18:00
    assert lab[2] == "weeknight"     # Wed 19:00
    assert lab[13] == "weeknight"    # Thu 06:00
    assert lab[14] == "work"         # Thu 07:00
    assert lab[25] == "work"         # Thu 18:00
    assert lab[26] == "weeknight"    # Thu 19:00
    assert lab[36] == "weeknight"    # Fri 05:00
    assert lab[38] == "work"         # Fri 07:00
    assert lab[49] == "work"         # Fri 18:00
    assert lab[50] == "weeknight"    # Fri 19:00
    assert lab[55] == "weekend"      # Sat 00:00
    assert lab[74] == "weekend"      # Sat 19:00


def test_two_class_rule_is_the_plans_rule():
    lab = hours.hour_labels(T0_NS, 75)
    two = [hours.two_class(v) for v in lab]
    assert two[:36].count("work") == 14 and two[:36].count("quiet") == 22
    assert two[36:].count("work") == 12 and two[36:].count("quiet") == 27


def test_class_rate_uses_the_class_own_worst_null_window():
    null = np.array([1.0, 5.0, 2.0, 9.0])
    inj = np.array([6.0, 6.0, 6.0, 6.0])
    # class {0, 2}: threshold 2 -> both injected windows exceed it
    assert hours.class_rate(null, inj, [0, 2]) == 1.0
    # class {1, 3}: threshold 9 -> neither does
    assert hours.class_rate(null, inj, [1, 3]) == 0.0
    # shared threshold from elsewhere (unknown-phase case)
    assert hours.class_rate(null, inj, [0, 2], thr=7.0) == 0.0


def test_split_control_lowers_threshold_mechanically_even_when_hours_are_alike():
    """Exchangeable windows: a random split still lowers each part's threshold,
    so its detectable swing is never worse than the unsplit one."""
    rng = np.random.default_rng(0)
    null = rng.gumbel(size=39)
    swings = [1.0, 2.0, 4.0, 8.0]
    inj = np.array([null + s for s in swings])       # statistic grows with swing, same noise
    whole = hours.swing_for(null, inj, swings, list(range(39)), 0.8)
    ctl = hours.split_control(null, inj, swings, size=12, level=0.8, n_draws=200, rng=np.random.default_rng(1))
    assert len(ctl) == 200
    assert np.nanmedian(ctl) <= whole
    # deterministic given the rng
    ctl2 = hours.split_control(null, inj, swings, size=12, level=0.8, n_draws=200, rng=np.random.default_rng(1))
    assert np.array_equal(ctl, ctl2, equal_nan=True)


def test_experiment_defaults_unchanged_and_stats_reproduce_reported_rates():
    rng = np.random.default_rng(3)
    fs = 12.0
    x = 700 + rng.normal(0, 3, int(8 * 3600 * fs))
    kw = dict(char_hours=4.0, swings=[1.0, 4.0])
    a = run_pulse.experiment(x, fs, 3600, np.random.default_rng(7), **kw)
    b = run_pulse.experiment(x, fs, 3600, np.random.default_rng(7), char_idx=[0, 1, 2, 3], return_null=True, return_stats=True, **kw)
    assert a["rows"] == b["rows"] and a["max_null"] == b["max_null"]
    null, inj = np.array(b["null_stats"]), b["inj_stats"]
    assert inj.shape == (2, 4)
    for row, st in zip(b["rows"], inj):
        assert row["detect_rate_zero_fa"] == float(np.mean(st > null.max()))


def test_experiment_baseline_can_come_from_a_subset_of_characterisation_windows():
    rng = np.random.default_rng(4)
    fs = 12.0
    x = 700 + rng.normal(0, 3, int(8 * 3600 * fs))
    x[: int(2 * 3600 * fs)] += rng.normal(0, 30, int(2 * 3600 * fs))     # two loud characterisation hours
    quiet = run_pulse.experiment(x, fs, 3600, np.random.default_rng(7), char_hours=4.0, char_idx=[2, 3], swings=[1.0])
    loud = run_pulse.experiment(x, fs, 3600, np.random.default_rng(7), char_hours=4.0, char_idx=[0, 1], swings=[1.0])
    assert quiet["n_test_windows"] == loud["n_test_windows"] == 4       # test period is unchanged
    assert quiet["null_stat_median"] > 10 * loud["null_stat_median"]    # a loud baseline hides everything
