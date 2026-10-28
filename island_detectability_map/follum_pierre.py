"""P_FA-calibrated threshold for the Welch-ratio line detector (Follum & Pierre 2016, IEEE TPWRS,
"Detection of periodic forced oscillations in power systems"; multichannel: Follum, PNNL-24681, 2015).

The detector in run_pulse.experiment divides an hour's Welch PSD bin by bin by a baseline from the
learning hours and takes the maximum over the search band. Follum & Pierre's periodogram GLRT is the
same statistic with the baseline replaced by the ambient PSD and a threshold derived from a target
probability of false alarm (P_FA). This module supplies that threshold for the averaged-periodogram
(Welch) form used here:

  * under the null, P_hat(f) / Phi_c(f) ~ chi2(nu) / nu, with nu the EFFECTIVE degrees of freedom of a
    Welch average of K Hann segments at 50% overlap (Welch 1967; Percival & Walden 1993, eq. 292b),
    slightly under the nominal 2K;
  * offline ambient PSD = median of the learning-hour spectra divided by the median of chi2(nu)/nu
    (the run_pulse baseline is that median, uncorrected);
  * threshold per bin for a family-wise P_FA per hour with Bonferroni over the B bins searched
    (Follum & Pierre eq. 16-18), an upper bound on P_FA because neighbouring Hann bins correlate;
  * PNNL-24681's multichannel sum statistic T = sum_m nu P_m / Phi_m with thresholds for independent
    and identical channels and the GMSC interpolation between them (its eq. 4 and appendices B-D,
    generalised from 2 to nu degrees of freedom per channel);
  * Follum's SNR metric A^2 / (2 phi_c), phi_c the per-sample (discrete-time, two-sided) ambient PSD at
    the oscillation frequency, and the noncentral chi-square detection probability it implies for a
    bin-centred sinusoid.
Not modelled: error in the ambient estimate itself (adds a little to P_FA; an F rather than chi-square
null), scalloping for off-centre frequencies, and any non-sinusoidal or jittered signal.

Origin: verification@532a1237
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, stats


def _window(n: int) -> np.ndarray:
    return np.hanning(n)                       # same window as pulse.welch_psd


def effective_dof(seg_len: int, step: int, n_segments: int) -> float:
    """Equivalent degrees of freedom of a Welch average of n_segments Hann segments of seg_len samples
    taken every `step` samples: nu = 2K / (1 + 2 sum_j (1 - j/K) rho(j)^2), rho(j) the window's
    overlap correlation at lag j*step. No overlap gives exactly 2K."""
    K = int(n_segments)
    if K <= 0:
        return 0.0
    w = _window(seg_len)
    e = np.sum(w * w)
    acc = 0.0
    for j in range(1, K):
        lag = j * step
        if lag >= seg_len:
            break
        rho = np.sum(w[:seg_len - lag] * w[lag:]) / e
        acc += (1.0 - j / K) * rho * rho
    return 2.0 * K / (1.0 + 2.0 * acc)


def segment_counts(x: np.ndarray, fs: float, win_s: float, seg_s: float) -> np.ndarray:
    """Number of all-finite segments pulse.welch_psd averages in each window of win_s seconds."""
    n = int(win_s * fs)
    L = int(round(seg_s * fs))
    step = L // 2
    out = []
    for i in range(0, len(x) - n + 1, n):
        w = x[i:i + n]
        out.append(sum(bool(np.all(np.isfinite(w[s:s + L]))) for s in range(0, n - L + 1, step)))
    return np.array(out, dtype=int)


def median_factor(nu: float) -> float:
    """Median of chi2(nu)/nu: the bias of a median-of-hours baseline as an estimate of the PSD."""
    return float(stats.chi2.median(nu) / nu)


def ambient_psd(learning_spectra: np.ndarray, nu: float) -> np.ndarray:
    """Offline ambient PSD from the learning hours' Welch spectra (hours x bins)."""
    return np.median(learning_spectra, axis=0) / median_factor(nu)


def threshold(p_fa: float, n_bins: int, nu: float) -> float:
    """Per-bin threshold on P_hat/Phi_c giving family-wise P_FA <= p_fa over n_bins bins (Bonferroni)."""
    return float(stats.chi2.isf(p_fa / n_bins, nu) / nu)


def implied_pfa(thr: float, n_bins: int, nu: float) -> float:
    """Bonferroni P_FA bound implied by a threshold thr on P_hat/Phi_c (inverse of `threshold`)."""
    return float(min(1.0, n_bins * stats.chi2.sf(nu * thr, nu)))


def per_sample_psd(phi_one_sided: float, fs: float) -> float:
    """Discrete-time two-sided PSD (units^2 per sample) from a one-sided PSD in units^2/Hz."""
    return phi_one_sided * fs / 2.0


def snr_metric(A: float, phi_one_sided: float, fs: float) -> float:
    """Follum / PNNL-24681 SNR metric A^2 / (2 phi_c), phi_c the per-sample PSD at the frequency.
    Depends on the sampling rate for a given physical PSD: phi_c = fs * Phi_one_sided / 2."""
    return A * A / (2.0 * per_sample_psd(phi_one_sided, fs))


def _deflection(snr: float, seg_len: int) -> float:
    """Expected signal-to-noise ratio in one segment's periodogram bin for a bin-centred sinusoid:
    d = snr * G * L / 2, G = (sum w)^2 / (L sum w^2) (2/3 for Hann)."""
    w = _window(seg_len)
    G = np.sum(w) ** 2 / (seg_len * np.sum(w * w))
    return snr * G * seg_len / 2.0


def detection_probability(snr: float, thr: float, nu: float, seg_len: int, fs: float) -> float:
    """P(P_hat/Phi_c > thr) at the bin of a bin-centred sinusoid of SNR metric `snr`:
    nu * P_hat/Phi_c ~ noncentral chi2(nu, lambda = nu * d). (fs enters only through snr.)"""
    d = _deflection(snr, seg_len)
    return float(stats.ncx2.sf(nu * thr, nu, nu * d))


def snr_for_pd(p_d: float, thr: float, nu: float, seg_len: int, fs: float) -> float:
    """SNR metric A^2/(2 phi_c) at which a bin-centred sinusoid is detected with probability p_d."""
    f = lambda lg: detection_probability(np.exp(lg), thr, nu, seg_len, fs) - p_d
    return float(np.exp(optimize.brentq(f, np.log(1e-9), np.log(1e3), xtol=1e-12)))


def cross_spectral_matrix(X: np.ndarray, fs: float, seg_s: float) -> np.ndarray:
    """Welch cross-spectral matrix (bins x M x M), Hann, 50% overlap, mean removed per segment,
    one-sided scaling as pulse.welch_psd; segments with any non-finite value in any channel skipped."""
    X = np.asarray(X, dtype=float)
    L = int(round(seg_s * fs))
    step = L // 2
    w = _window(L)
    scale = fs * np.sum(w * w)
    acc, count = None, 0
    for s in range(0, X.shape[1] - L + 1, step):
        seg = X[:, s:s + L]
        if not np.all(np.isfinite(seg)):
            continue
        Z = np.fft.rfft((seg - seg.mean(axis=1, keepdims=True)) * w, axis=1)      # M x bins
        P = np.einsum("if,jf->fij", Z, Z.conj()) / scale
        P[1:-1] *= 2.0
        acc = P if acc is None else acc + P
        count += 1
    if count == 0:
        raise ValueError("no complete finite segment")
    return acc / count


def gmsc(S: np.ndarray) -> np.ndarray:
    """Generalised magnitude-squared coherence per bin (PNNL-24681 App. D; Ramirez et al. 2008):
    (lambda_max(C) - 1) / (M - 1), C the coherence matrix D^-1/2 S D^-1/2."""
    d = np.sqrt(np.real(np.einsum("fii->fi", S)))
    C = S / (d[:, :, None] * d[:, None, :])
    lam = np.linalg.eigvalsh(C)[:, -1]
    M = S.shape[-1]
    return np.clip((lam - 1.0) / (M - 1.0), 0.0, 1.0)


def gmsc_threshold(p_fa: float, n_bins: int, nu: float, M: int, G) -> np.ndarray:
    """Threshold on T = sum_m nu P_m/Phi_m: gamma_ind (1 - G) + gamma_same G, with
    gamma_ind = chi2^-1(1 - p/B; M nu) and gamma_same = M chi2^-1(1 - p/B; nu)."""
    q = p_fa / n_bins
    g_ind = stats.chi2.isf(q, M * nu)
    g_same = M * stats.chi2.isf(q, nu)
    return g_ind * (1.0 - np.asarray(G)) + g_same * np.asarray(G)


def fundamental_amplitude(x: np.ndarray, fs: float, f0: float, rel_width: float = 0.05) -> float:
    """Amplitude A of the sinusoidal component of x at its fundamental f0: A = sqrt(2 * power in the
    line), the power integrated over |f - f0| <= max(rel_width f0, 3 bins) of a Hann Welch PSD with
    segments of up to 60 periods (a jittered training step broadens the line)."""
    from island_detectability_map import pulse
    seg_s = min(len(x) / fs, 60.0 / f0)
    f, p = pulse.welch_psd(np.asarray(x, float), fs, seg_s)
    df = f[1] - f[0]
    sel = np.abs(f - f0) <= max(rel_width * f0, 3 * df)
    return float(np.sqrt(2.0 * np.sum(p[sel]) * df))
