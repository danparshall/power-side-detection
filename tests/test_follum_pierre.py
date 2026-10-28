"""Tests for island_detectability_map.follum_pierre: the P_FA-calibrated (Follum-Pierre) threshold
for the Welch-ratio detector used in run_pulse.experiment.

Behaviour under test, all through the detector's own Welch estimator (pulse.welch_psd: Hann,
50% overlap, mean removed per segment):
  * effective degrees of freedom of the averaged periodogram (overlap-corrected);
  * the offline ambient-PSD estimate (median of learning-hour spectra) is unbiased after correction;
  * a threshold set for a target P_FA per hour gives that per-bin rate exactly and a family-wise rate
    at or below the target (Bonferroni), on white and on coloured (AR(1)) noise;
  * an injected sinusoid of known A^2/(2 phi_c) is detected at the rate the noncentral chi-square predicts;
  * GMSC (PNNL-24681 App. D) and its interpolated threshold behave at the independent/identical limits;
  * the fundamental-amplitude helper returns A for a sinusoid and (2S/pi) sin(pi D) for a square wave.
Tests use 1 sample/s and 100 s segments (K = 71 segments an hour) to keep run time short; the code
is rate-independent and the real runs use 12 samples/s.

Origin: verification@532a1237
"""
import numpy as np
import pytest
from scipy import signal as sps

from island_detectability_map import pulse
from island_detectability_map import follum_pierre as FP

FS, SEG, WIN = 1.0, 100.0, 3600.0
L = int(SEG * FS)
NWIN = int(WIN * FS)
K = (NWIN - L) // (L // 2) + 1


def hour_spectra(x, fs=FS, seg=SEG, win=WIN):
    n = int(win * fs)
    return np.array([pulse.welch_psd(x[i:i + n], fs, seg)[1] for i in range(0, len(x) - n + 1, n)])


def ar1(n, phi, rng, sigma=1.0):
    return sps.lfilter([1.0], [1.0, -phi], sigma * rng.standard_normal(n))


def ar1_psd_one_sided(f, phi, fs, sigma=1.0):
    return 2.0 * sigma ** 2 / fs / np.abs(1 - phi * np.exp(-2j * np.pi * f / fs)) ** 2


def test_effective_dof_matches_welch_variance_on_white_noise():
    rng = np.random.default_rng(1)
    sp = hour_spectra(rng.standard_normal(NWIN * 400))
    band = slice(5, 45)                                  # away from DC/Nyquist
    r = sp[:, band] / (2.0 / FS)                         # true one-sided PSD of unit white noise
    nu = FP.effective_dof(L, L // 2, K)
    assert np.var(r) == pytest.approx(2.0 / nu, rel=0.06)
    assert nu < 2 * K                                    # overlap costs dof
    assert FP.effective_dof(L, L, K) == pytest.approx(2 * K)   # no overlap: exactly 2K


def test_segment_counts_follow_the_estimator_skipping_nan_segments():
    x = np.zeros(NWIN * 3)
    x[NWIN + 10] = np.nan                                # hour 1: one NaN kills the first two segments
    counts = FP.segment_counts(x, FS, WIN, SEG)
    assert list(counts) == [K, K - 1, K]


def test_ambient_estimate_from_median_of_learning_hours_is_unbiased():
    rng = np.random.default_rng(2)
    phi = 0.8
    sp = hour_spectra(ar1(NWIN * 120, phi, rng))
    f = np.fft.rfftfreq(L, 1.0 / FS)
    nu = FP.effective_dof(L, L // 2, K)
    amb = FP.ambient_psd(sp, nu)
    true = ar1_psd_one_sided(f, phi, FS)
    band = slice(5, 45)
    assert np.mean(amb[band] / true[band]) == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize("kind", ["white", "ar1"])
def test_threshold_gives_target_false_alarm_rate(kind):
    rng = np.random.default_rng(3 if kind == "white" else 4)
    mk = (lambda n: rng.standard_normal(n)) if kind == "white" else (lambda n: ar1(n, 0.8, rng))
    nu = FP.effective_dof(L, L // 2, K)
    learn = hour_spectra(mk(NWIN * 120))
    amb = FP.ambient_psd(learn, nu)
    test = hour_spectra(mk(NWIN * 3000))
    band = slice(5, 45)
    ratio = test[:, band] / amb[band]
    B = ratio.shape[1]
    # per-bin calibration: a per-bin level of 1e-2 (threshold for P_FA = 1e-2 * B with B bins)
    thr_bin = FP.threshold(1e-2 * B, B, nu)
    per_bin = np.mean(ratio > thr_bin)
    assert per_bin == pytest.approx(1e-2, rel=0.25)
    # family-wise (max over the band, one test per hour): at or below target, not grossly below
    target = 0.05
    fw = np.mean(ratio.max(axis=1) > FP.threshold(target, B, nu))
    assert 0.4 * target <= fw <= 1.2 * target


def test_implied_pfa_inverts_threshold():
    nu, B = 135.0, 40
    for p in (1e-1, 1e-3, 1e-6):
        assert FP.implied_pfa(FP.threshold(p, B, nu), B, nu) == pytest.approx(p, rel=1e-6)
    assert FP.implied_pfa(0.5, B, nu) == 1.0             # capped at 1


def test_sinusoid_detected_at_predicted_rate():
    rng = np.random.default_rng(5)
    nu = FP.effective_dof(L, L // 2, K)
    k0 = 20                                              # bin-centred: f0 = k0 / SEG
    f0 = k0 / SEG
    sigma = 1.0
    phi_one = 2 * sigma ** 2 / FS                        # true one-sided PSD of the white noise
    thr = FP.threshold(1e-3, 40, nu)
    t = np.arange(NWIN) / FS
    for target_pd in (0.5, 0.8):
        snr = FP.snr_for_pd(target_pd, thr, nu, L, FS)   # A^2/(2 phi_c) giving that P_d at the f0 bin
        A = np.sqrt(2 * snr * FP.per_sample_psd(phi_one, FS))
        hits = []
        for _ in range(1500):
            x = sigma * rng.standard_normal(NWIN) + A * np.cos(2 * np.pi * f0 * t + rng.uniform(0, 2 * np.pi))
            p = pulse.welch_psd(x, FS, SEG)[1]
            hits.append(p[k0] / phi_one > thr)
        assert np.mean(hits) == pytest.approx(target_pd, abs=0.05)
        assert FP.detection_probability(snr, thr, nu, L, FS) == pytest.approx(target_pd, abs=1e-6)


def test_snr_metric_convention_is_per_sample_psd():
    # PNNL-24681 / Follum: A^2 / (2 phi_c) with phi_c the discrete-time (per-sample, two-sided) PSD.
    # For white noise of variance sigma^2 that is A^2 / (2 sigma^2), whatever the sampling rate.
    for fs in (1.0, 12.0, 30.0):
        sigma, A = 2.0, 0.5
        phi_one = 2 * sigma ** 2 / fs
        assert FP.snr_metric(A, phi_one, fs) == pytest.approx(A ** 2 / (2 * sigma ** 2))


def test_gmsc_limits():
    rng = np.random.default_rng(6)
    n = NWIN * 16                                        # GMSC of independent channels is biased up by ~sqrt(M / segments)
    a, b, c = (rng.standard_normal(n) for _ in range(3))
    band = slice(5, 45)
    g_ind = FP.gmsc(FP.cross_spectral_matrix(np.vstack([a, b, c]), FS, SEG))[band]
    g_same = FP.gmsc(FP.cross_spectral_matrix(np.vstack([a, a, a]), FS, SEG))[band]
    resid = np.vstack([a - (b + c) / 2, b - (a + c) / 2, c - (a + b) / 2])   # three phase-unbalance residuals
    g_res = FP.gmsc(FP.cross_spectral_matrix(resid, FS, SEG))[band]
    assert np.median(g_ind) < 0.05
    assert np.allclose(g_same, 1.0, atol=1e-9)
    assert np.median(g_res) == pytest.approx(0.25, abs=0.05)


def test_gmsc_threshold_interpolates_between_independent_and_identical():
    nu, B, M, p = 135.0, 40, 3, 1e-3
    t0 = FP.gmsc_threshold(p, B, nu, M, 0.0)
    t1 = FP.gmsc_threshold(p, B, nu, M, 1.0)
    # statistic T = sum_m nu * P_m / phi_m: chi2(M nu) if independent, M * chi2(nu) if identical
    from scipy import stats
    assert t0 == pytest.approx(stats.chi2.isf(p / B, M * nu))
    assert t1 == pytest.approx(M * stats.chi2.isf(p / B, nu))
    assert FP.gmsc_threshold(p, B, nu, M, 0.25) == pytest.approx(0.75 * t0 + 0.25 * t1)


def test_independent_channels_sum_statistic_is_calibrated():
    rng = np.random.default_rng(7)
    nu = FP.effective_dof(L, L // 2, K)
    M = 3
    learn = [hour_spectra(rng.standard_normal(NWIN * 60)) for _ in range(M)]
    amb = [FP.ambient_psd(s, nu) for s in learn]
    test = [hour_spectra(rng.standard_normal(NWIN * 2000)) for _ in range(M)]
    band = slice(5, 45)
    T = sum(nu * test[m][:, band] / amb[m][band] for m in range(M))
    B = T.shape[1]
    target = 0.05
    fw = np.mean(T.max(axis=1) > FP.gmsc_threshold(target, B, nu, M, 0.0))
    assert 0.4 * target <= fw <= 1.3 * target


def test_fundamental_amplitude():
    fs, f0 = 12.0, 0.1
    t = np.arange(int(3600 * fs)) / fs
    A = 1.7
    assert FP.fundamental_amplitude(A * np.sin(2 * np.pi * f0 * t), fs, f0) == pytest.approx(A, rel=0.01)
    S, D = 2.4, 0.57
    sq = np.where((t * f0) % 1.0 < D, S, 0.0)
    assert FP.fundamental_amplitude(sq, fs, f0) == pytest.approx(2 * S / np.pi * np.sin(np.pi * D), rel=0.01)


# ------------------------------------------------------------------ fast equivalent of run_pulse.experiment
def _toy_case():
    rng = np.random.default_rng(8)
    fs, win, seg = 2.0, 600.0, 50.0
    n_win = 20
    x = ar1(int(fs * win * n_win), 0.6, rng) + 0.3 * np.sin(2 * np.pi * 0.23 * np.arange(int(fs * win * n_win)) / fs)
    x[int(fs * win * 9.5)] = np.nan                       # one window loses a few segments
    x[int(fs * win * 14):int(fs * win * 14) + 400] += 4 * np.sin(2 * np.pi * 0.31 * np.arange(400) / fs)  # a site line in one test window
    t = np.arange(1500) / fs
    trace = np.where((t / 9.7) % 1.0 < 0.57, 1.0, 0.0) + 0.05 * rng.standard_normal(len(t))
    trace -= trace.mean()
    return x, fs, win, seg, trace


def test_fast_experiment_reproduces_run_pulse_experiment():
    from island_detectability_map import run_pulse as R, run_nlr_inject as NI
    from island_detectability_map import welch_engine as WE
    x, fs, win, seg, trace = _toy_case()
    kw = dict(band=(0.08, 0.5), f0_range=(0.1, 0.4), seg_s=seg, char_hours=win * 4 / 3600, whitelist=(0.2,))
    swings = [0.25, 1.0, 4.0]
    char_idx = [0, 1, 2, 3, 6, 7, 8]
    ref = R.experiment(x, fs, win, np.random.default_rng(42), return_null=True, return_stats=True,
                       make_signal=NI.make_signal_from(trace), char_idx=char_idx, swings=swings, **kw)
    got = WE.experiment(x, fs, win, trace, seed=42, swings=swings, char_idx=char_idx, **kw)
    assert got["notched_bins"] == ref["notched_bins"] and got["band_bins"] == ref["band_bins"]
    np.testing.assert_allclose(got["null_stats"], ref["null_stats"], rtol=1e-9)
    np.testing.assert_allclose(got["inj_stats"], ref["inj_stats"], rtol=1e-9)
    # the per-bin ratios it also returns are the ones the null maximum is taken over
    np.testing.assert_allclose(got["ratios"].max(axis=1), got["null_stats"], rtol=1e-12)
    assert list(got["segments"]) == list(FP.segment_counts(x, fs, win, seg))


def test_sum_statistic_with_one_channel_is_the_scaled_single_channel_statistic():
    # M = 1: T = nu_h * P / Phi = nu_h * m(nu_char) * (P / median baseline), i.e. experiment's ratio rescaled
    from island_detectability_map import welch_engine as WE
    x, fs, win, seg, trace = _toy_case()
    kw = dict(band=(0.08, 0.5), f0_range=(0.1, 0.4), seg_s=seg, char_hours=win * 4 / 3600, whitelist=(0.2,))
    swings = [0.25, 1.0, 4.0]
    char_idx = [0, 1, 2, 3, 6, 7, 8]
    one = WE.experiment(x, fs, win, trace, seed=42, swings=swings, char_idx=char_idx, **kw)
    s = WE.sum_statistic(x[None], fs, win, trace, seed=42, swings=swings, char_idx=char_idx, coefs=[[1.0]],
                         p_fas=[1e-2], **kw)
    nu = s["nu"][one["test"]]
    m = FP.median_factor(s["nu_char"])
    np.testing.assert_allclose(s["null_Tmax"], nu * m * np.array(one["null_stats"]), rtol=1e-9)
    np.testing.assert_allclose(s["inj_Tmax"][0], nu[None] * m * one["inj_stats"], rtol=1e-9)
