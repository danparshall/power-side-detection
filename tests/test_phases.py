"""Per-phase analysis for the current-monitor pulse channel.
Written before the code.

Origin: verification@3af18129
"""
from pathlib import Path

import numpy as np
import pytest

from island_detectability_map import phases, run_pulse

FS = 12.0


def _write_npz(path: Path, n=2400, with_l23=False, seed=0):
    rng = np.random.default_rng(seed)
    z = dict(t0_ns=0, fs=120.0,
             L1MAG=(7200 + rng.normal(0, 20, n)).astype(np.float32),
             C1MAG=(29 + rng.normal(0, 1, n)).astype(np.float32),
             C2MAG=(34 + rng.normal(0, 1, n)).astype(np.float32),
             C3MAG=(33 + rng.normal(0, 1, n)).astype(np.float32))
    if with_l23:
        z["L2MAG"] = (7150 + rng.normal(0, 20, n)).astype(np.float32)
        z["L3MAG"] = (7250 + rng.normal(0, 20, n)).astype(np.float32)
    np.savez(path / "grizzly_head.npz", **z)
    return z


def test_per_phase_series_sum_to_existing_three_phase_proxy(tmp_path):
    _write_npz(tmp_path)
    s_sum, fs, _ = run_pulse.load(tmp_path)
    x, fs2 = phases.load_phases(tmp_path, voltage="L1")
    assert x.shape == (3, len(s_sum)) and fs2 == fs
    np.testing.assert_allclose(x.sum(axis=0), s_sum, rtol=1e-9)


def test_per_phase_series_use_own_phase_voltage_when_asked(tmp_path):
    z = _write_npz(tmp_path, with_l23=True)
    x, _ = phases.load_phases(tmp_path, voltage="own")
    for k in range(3):
        expect = z[f"L{k+1}MAG"].astype(np.float64) * z[f"C{k+1}MAG"].astype(np.float64) / 1e3
        np.testing.assert_allclose(x[k], expect, rtol=1e-9)


def test_current_only_series_are_the_current_magnitudes(tmp_path):
    z = _write_npz(tmp_path)
    x, _ = phases.load_phases(tmp_path, voltage=None)
    np.testing.assert_allclose(x[1], z["C2MAG"].astype(np.float64), rtol=1e-9)


def test_band_correlation_of_independent_noise_is_near_zero():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, int(FS * 20000)))
    c = phases.band_correlation(x, FS, (0.1, 1.0), seg_s=100.0)
    off = c[~np.eye(3, dtype=bool)]
    assert np.all(np.abs(off) < 0.05)


def test_band_correlation_of_common_mode_is_near_one():
    rng = np.random.default_rng(2)
    common = rng.normal(size=int(FS * 20000))
    x = np.vstack([common * a + 0.01 * rng.normal(size=common.size) for a in (1.0, 2.0, 0.5)])
    c = phases.band_correlation(x, FS, (0.1, 1.0), seg_s=100.0)
    assert np.all(c > 0.99)


def test_band_correlation_is_specific_to_the_band():
    rng = np.random.default_rng(3)
    n = int(FS * 20000)
    t = np.arange(n) / FS
    common = 5.0 * np.sin(2 * np.pi * 0.5 * t + rng.uniform(0, 6.28))
    x = np.vstack([common + rng.normal(size=n) for _ in range(3)])
    lo = phases.band_correlation(x, FS, (0.45, 0.55), seg_s=100.0)
    hi = phases.band_correlation(x, FS, (3.0, 5.0), seg_s=100.0)
    assert lo[0, 1] > 0.95 and abs(hi[0, 1]) < 0.05


def test_imbalance_residual_cancels_balanced_step_and_keeps_single_phase_step():
    n = 1000
    base = np.zeros((3, n))
    balanced = base.copy(); balanced[:, n // 2:] += 10.0
    single = base.copy(); single[2, n // 2:] += 10.0
    np.testing.assert_allclose(phases.imbalance(balanced, 2), 0.0, atol=1e-12)
    r = phases.imbalance(single, 2)
    assert r[-1] - r[0] == pytest.approx(10.0)


def test_predictive_residual_removes_what_other_phases_explain():
    rng = np.random.default_rng(4)
    n = 200_000
    a, b = rng.normal(size=n), rng.normal(size=n)
    x = np.vstack([a, b, 2.0 * a - 0.5 * b + 0.1 * rng.normal(size=n)])
    coef = phases.fit_predictor(x[:, : n // 2], k=2)
    r = phases.predictive_residual(x[:, n // 2:], k=2, coef=coef)
    assert np.std(r) == pytest.approx(0.1, rel=0.05)


def test_predictive_residual_passes_single_phase_injection_with_unit_gain():
    rng = np.random.default_rng(5)
    n = 100_000
    a, b = rng.normal(size=n), rng.normal(size=n)
    x = np.vstack([a, b, a + b + 0.1 * rng.normal(size=n)])
    coef = phases.fit_predictor(x[:, : n // 2], k=2)
    test = x[:, n // 2:].copy()
    inj = np.sin(np.arange(test.shape[1]) / 7.0) * 3.0
    with_inj = test.copy(); with_inj[2] += inj
    d = phases.predictive_residual(with_inj, 2, coef) - phases.predictive_residual(test, 2, coef)
    np.testing.assert_allclose(d, inj, atol=1e-9)


def test_step_split_reports_balanced_share_of_each_step():
    n = int(FS * 600)
    x = np.zeros((3, n))
    x[:, n // 3:] += 6.0           # balanced step (three-phase motor start)
    x[1, 2 * n // 3:] += 6.0       # single-phase step
    ev = phases.step_events(x, FS, win_s=2.0, min_step=3.0)
    assert len(ev) == 2
    shares = sorted(e["balanced_share"] for e in ev)
    assert shares[0] == pytest.approx(1.0 / 3.0, abs=0.02)
    assert shares[1] == pytest.approx(1.0, abs=0.02)


def test_single_phase_step_shows_with_opposite_sign_and_half_size_on_the_other_residuals():
    """Sign convention, pinned after a 2026-10-03 question: a load ADDED on one phase raises
    that phase's residual by its full size and lowers the other two by half each."""
    x = np.zeros((3, 100)); x[0, 50:] += 20.0
    steps = [phases.imbalance(x, k)[-1] - phases.imbalance(x, k)[0] for k in range(3)]
    assert steps == pytest.approx([20.0, -10.0, -10.0])
    assert sum(phases.imbalance(x, k) for k in range(3)) == pytest.approx(np.zeros(100))
