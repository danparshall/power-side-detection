"""Tests for island_detectability_map.

Fixtures are hand-derived numbers and the measured S3 tables: the ROC for a persistent added
load and the hourly residual sigma survey, on summed household and zone-substation records
(origin: verification@f7a9006e).

Origin: verification@f7a9006e
"""
import numpy as np
import pytest

from island_detectability_map import model, pulse


def test_dprime_from_auc_matches_hand_values():
    assert model.dprime_from_auc(0.5) == pytest.approx(0.0, abs=1e-9)
    assert model.dprime_from_auc(0.84) == pytest.approx(1.406, abs=0.005)
    assert model.dprime_from_auc(0.91) == pytest.approx(1.896, abs=0.005)


def test_sigma_curves_reproduce_measured_anchors():
    # residential-synthetic: power law fitted through the four GoiEner points
    for mean_kw, sigma_kw in [(2.6, 0.99), (13.7, 3.34), (57.5, 11.9), (278.2, 54.0)]:
        assert model.sigma_fixed(mean_kw, "residential_synthetic") == pytest.approx(sigma_kw, rel=0.15)
    # real mixed load: power law through the two Ausgrid medians, exact at both
    assert model.sigma_fixed(3200.0, "real_mixed") == pytest.approx(980.0, rel=1e-6)
    assert model.sigma_fixed(12100.0, "real_mixed") == pytest.approx(3360.0, rel=1e-6)


def test_s3_calibration_constant_is_order_one():
    # d' backed out of measured AUC, divided by (load / sigma_fixed), at the four
    # S3 points with AUC away from 0.5 and 1.0. Hand values: 1.39, 1.29, 1.02, 0.70.
    ks = model.s3_calibration_points()
    got = [round(p["k"], 2) for p in ks]
    assert got == pytest.approx([1.39, 1.29, 1.02, 0.70], abs=0.03)


def test_mean_shift_frontier_is_monotone_and_hits_hand_value():
    # 90% detection at 1% false alarm: d' = 2.326 + 1.282 = 3.608; k = 1.2
    # at the 278.2 kW synthetic feeder: 3.608 / 1.2 * sigma(278.2)
    p = model.mean_shift_min_load(278.2, "residential_synthetic", tpr=0.90, fpr=0.01, k=1.2)
    assert p == pytest.approx(3.608 / 1.2 * model.sigma_fixed(278.2, "residential_synthetic"), rel=1e-3)
    ms = np.logspace(1, 5, 30)
    ps = [model.mean_shift_min_load(m, "real_mixed") for m in ms]
    assert np.all(np.diff(ps) > 0)


def test_balance_floor_matches_floor_memo():
    # floor memo: ~20 kW at a 500 kVA transformer with a 1.5% monitor (3 * e * P)
    assert model.balance_min_load(500.0, 0.015) == pytest.approx(22.5)
    # and ~0.75 MW at a 50 MW substation with revenue-grade 0.5%
    assert model.balance_min_load(50_000.0, 0.005) == pytest.approx(750.0)


def test_rect_fundamental_rms():
    assert pulse.rect_fundamental_rms(1.0, duty=0.5) == pytest.approx(np.sqrt(2) / np.pi, rel=1e-9)
    assert pulse.rect_fundamental_rms(1.0, duty=0.8) == pytest.approx(np.sqrt(2) / np.pi * np.sin(0.8 * np.pi), rel=1e-9)


def test_welch_level_on_white_noise():
    rng = np.random.default_rng(0)
    fs, sigma = 12.0, 2.0
    x = rng.normal(0, sigma, 12 * 3600)
    f, s = pulse.welch_psd(x, fs, seg_seconds=100.0)
    band = (f > 0.2) & (f < 5.0)
    # one-sided PSD of white noise = sigma^2 / (fs/2)
    assert np.median(s[band]) == pytest.approx(sigma**2 / (fs / 2), rel=0.08)


def test_pulse_train_mean_and_swing():
    fs = 12.0
    y = pulse.pulse_train(n=12 * 600, fs=fs, f0=0.5, swing=10.0, duty=0.8, jitter=0.0, rng=np.random.default_rng(1))
    assert y.max() - y.min() == pytest.approx(10.0)
    assert abs(y.mean()) < 0.2  # returned mean-removed: a pulse adds no net energy here


def test_injected_line_is_found_and_absent_line_is_not():
    rng = np.random.default_rng(2)
    fs = 12.0
    n = int(fs * 3600)
    bg = rng.normal(0, 5.0, n)
    base_f, base_s = pulse.welch_psd(rng.normal(0, 5.0, n), fs, 100.0)
    stat0, f_hit0 = pulse.line_statistic(bg, fs, base_s, band=(0.1, 3.0))
    sig = pulse.pulse_train(n, fs, f0=0.7, swing=6.0, duty=0.8, jitter=0.01, rng=rng)
    stat1, f_hit1 = pulse.line_statistic(bg + sig, fs, base_s, band=(0.1, 3.0))
    assert stat1 > 3 * stat0
    assert f_hit1 == pytest.approx(0.7, abs=0.02)


def test_pulse_frontier_scaling():
    # reference: swing detected at M_ref in T_ref; island = swing / depth
    kw = dict(swing_ref_kva=7.0, m_ref_kw=709.0, t_ref_h=1.0)
    assert model.pulse_min_load(709.0, depth=0.3, beta=1.0, t_hours=1.0, **kw) == pytest.approx(7.0 / 0.3)
    # amplitude noise ~ M^beta
    assert model.pulse_min_load(4 * 709.0, depth=0.3, beta=0.5, t_hours=1.0, **kw) == pytest.approx(2 * 7.0 / 0.3)
    assert model.pulse_min_load(4 * 709.0, depth=0.3, beta=1.0, t_hours=1.0, **kw) == pytest.approx(4 * 7.0 / 0.3)
    # incoherent averaging: detectable amplitude falls as T^(-1/4)
    assert model.pulse_min_load(709.0, depth=0.3, beta=1.0, t_hours=16.0, **kw) == pytest.approx(7.0 / 0.3 / 2)


def test_max_aggregation_inverts_frontiers():
    m = model.max_aggregation(lambda M: model.balance_min_load(M, 0.015), 22.5)
    assert m == pytest.approx(500.0, rel=1e-3)


def test_ar1_log_periods_has_requested_correlation_and_spread():
    x = pulse.ar1_log_periods(200_000, sigma_log=0.3, rho=0.9, rng=np.random.default_rng(3))
    assert np.std(x) == pytest.approx(0.3, rel=0.03)
    assert np.corrcoef(x[:-1], x[1:])[0, 1] == pytest.approx(0.9, abs=0.01)


def test_training_trace_limits():
    fs, n = 100.0, 100 * 600
    rng = np.random.default_rng(4)
    # no straggling, no variation: a rectangle with duty 1/(1+sync_frac), peak-to-peak = swing
    y = pulse.training_trace(n, fs, f_med=0.5, swing=10.0, sigma_log=0.0, rho=0.0, straggle=0.0, sync_frac=0.25, rng=rng)
    assert y.max() - y.min() == pytest.approx(10.0, rel=1e-6)
    assert np.mean(y > y.min() + 5.0) == pytest.approx(0.8, abs=0.01)
    # full straggling with many chips: a sawtooth, so the compute phase averages half the swing
    z = pulse.training_trace(n, fs, f_med=0.5, swing=10.0, sigma_log=0.0, rho=0.0, n_chips=400, straggle=1.0, sync_frac=0.25, rng=rng)
    assert (z - z.min()).mean() == pytest.approx(10.0 * 0.5 * 0.8, rel=0.03)
    # steps far faster than the sample rate average out to (almost) nothing
    w = pulse.training_trace(12 * 600, 12.0, f_med=100.0, swing=10.0, sigma_log=0.0, rho=0.0, straggle=1.0, rng=rng)
    assert w.std() < 0.5


def test_training_trace_mean_cadence():
    fs, n = 50.0, 50 * 2000
    y = pulse.training_trace(n, fs, f_med=1.0, swing=1.0, sigma_log=0.1, rho=0.95, straggle=0.0, rng=np.random.default_rng(5))
    ups = np.sum(np.diff((y > 0).astype(int)) == 1)
    assert ups / 2000.0 == pytest.approx(1.0, rel=0.08)


# --- pulse reference read from a held-out result row ---

def _heldout_fixture():
    def row(series, rule, margin, fa, swing):
        return {"family": series, "rule": rule, "class": "all", "margin": margin,
                "false_alarm_hours": fa, "score_hours": 121, "swing80": swing}
    return {"families": {"maglp": {"edge": {"rows": [
        row("sum", "set on B, scored on A", 1.0, 1, [20.0, 20.0]),
        row("sum", "set on A, scored on B", 1.0, 0, [21.6, 21.6]),
        row("sum", "set on A, scored on B", 2.0, 0, [26.2, 26.2]),
        row("imbalance", "set on A, scored on B", 1.0, 0, [5.7, 6.5]),
    ]}}}}


def _ref_cfg(**over):
    cfg = {"family": "maglp", "detector": "edge", "series": "sum",
           "rule": "set on A, scored on B", "margin": 1.0, "m_ref_kva": 742.3}
    cfg.update(over)
    return cfg


def test_pulse_reference_takes_the_named_held_out_row():
    from island_detectability_map.run import pulse_reference
    ref = pulse_reference(_ref_cfg(), _heldout_fixture())
    assert ref["swing_ref_kva"] == pytest.approx(21.6)
    assert ref["m_ref_kw"] == pytest.approx(742.3)
    assert ref["t_ref_h"] == 1.0


def test_pulse_reference_uses_the_top_of_a_range():
    from island_detectability_map.run import pulse_reference
    ref = pulse_reference(_ref_cfg(series="imbalance"), _heldout_fixture())
    assert ref["swing_ref_kva"] == pytest.approx(6.5)


def test_pulse_reference_refuses_a_row_with_false_alarms():
    from island_detectability_map.run import pulse_reference
    with pytest.raises(ValueError, match="false alarm"):
        pulse_reference(_ref_cfg(rule="set on B, scored on A"), _heldout_fixture())


def test_pulse_reference_refuses_to_guess_when_no_row_matches():
    from island_detectability_map.run import pulse_reference
    with pytest.raises(ValueError, match="no row"):
        pulse_reference(_ref_cfg(margin=1.5), _heldout_fixture())
