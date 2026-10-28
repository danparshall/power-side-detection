"""Per-phase view of the current-monitor pulse channel.

A single-phase island (US home, small shop) adds all of its current to one phase,
while the earlier analysis used the three-phase sum. Each phase is a separate
conductor with its own current transformer at the substation, so per-phase series
are free. Two ways to use them:

* imbalance(x, k):  phase k minus the mean of the other two. Cancels balanced
  three-phase load (motors, compressors on three-phase service) exactly; passes a
  single-phase signal on phase k with unit gain.
* predictive_residual(x, k, coef):  phase k minus its least-squares prediction from
  the other two, fitted on a characterisation period only. The linear optimum of
  the two; equals imbalance() when the fitted weights are (0.5, 0.5).

The gain over the summed series depends on the covariance of the phases'
background in the band of interest (band_correlation). Correlation here is a
property of the load mix, not of the wiring.

Magnitude-only caveat: the LBNL streams used are RMS magnitudes. A change in phase
current adds as a phasor; with near-unity power-factor loads and a ~0.9 PF
background the magnitude change is ~cos(angle difference) of the true change.

Origin: verification@3af18129
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_phases(data_dir: Path, voltage: str | None = "L1", fname: str = "grizzly_head.npz"):
    """Return (x, fs): x is (3, n) per-phase series from a fetch_upmu cache (fname).

    voltage="L1"  -> S_k = V1 * I_k / 1e3 (kVA); sums to run_pulse.load()'s proxy.
    voltage="own" -> S_k = V_k * I_k / 1e3 (needs L2MAG, L3MAG).
    voltage=None  -> I_k in A (current magnitudes only; no voltage common mode).
    """
    z = np.load(Path(data_dir) / fname)
    i = np.vstack([z[f"C{k}MAG"].astype(np.float64) for k in (1, 2, 3)])
    n = min(len(z[k]) for k in z.files if k.endswith("MAG"))
    i = i[:, :n]
    if voltage is None:
        x = i
    elif voltage == "L1":
        x = z["L1MAG"].astype(np.float64)[:n] * i / 1e3
    elif voltage == "own":
        v = np.vstack([z[f"L{k}MAG"].astype(np.float64)[:n] for k in (1, 2, 3)])
        x = v * i / 1e3
    else:
        raise ValueError(f"voltage must be 'L1', 'own' or None, not {voltage!r}")
    return x, float(z["fs"])


def cross_spectra(x: np.ndarray, fs: float, seg_s: float = 100.0):
    """Welch cross-spectral matrix, shape (n_freq, m, m), Hann, 50% overlap."""
    n = int(round(seg_s * fs))
    step = n // 2
    w = np.hanning(n)
    scale = fs * np.sum(w * w)
    acc, count = None, 0
    for start in range(0, x.shape[1] - n + 1, step):
        seg = x[:, start:start + n]
        if not np.all(np.isfinite(seg)):
            continue
        sp = np.fft.rfft((seg - seg.mean(axis=1, keepdims=True)) * w, axis=1)   # (m, nf)
        p = np.einsum("if,jf->fij", sp, sp.conj()) / scale
        acc = p if acc is None else acc + p
        count += 1
    if count == 0:
        raise ValueError("no complete finite segment")
    return np.fft.rfftfreq(n, 1.0 / fs), acc / count


def band_covariance(x: np.ndarray, fs: float, band, seg_s: float = 100.0) -> np.ndarray:
    """Covariance of the series' fluctuations within band (lo, hi) Hz (real part of the
    band-integrated cross-spectrum; units^2)."""
    f, c = cross_spectra(x, fs, seg_s)
    sel = (f >= band[0]) & (f < band[1])
    return 2.0 * c[sel].real.sum(axis=0) * (f[1] - f[0])


def band_correlation(x: np.ndarray, fs: float, band, seg_s: float = 100.0) -> np.ndarray:
    c = band_covariance(x, fs, band, seg_s)
    d = np.sqrt(np.diag(c))
    return c / np.outer(d, d)


def imbalance(x: np.ndarray, k: int) -> np.ndarray:
    others = [j for j in range(x.shape[0]) if j != k]
    return x[k] - x[others].mean(axis=0)


def fit_predictor(x_char: np.ndarray, k: int) -> np.ndarray:
    """Least-squares weights predicting phase k's fluctuation from the others
    (means removed). Fit on characterisation data only."""
    others = [j for j in range(x_char.shape[0]) if j != k]
    a = x_char[others] - x_char[others].mean(axis=1, keepdims=True)
    y = x_char[k] - x_char[k].mean()
    coef, *_ = np.linalg.lstsq(a.T, y, rcond=None)
    return coef


def predictive_residual(x: np.ndarray, k: int, coef: np.ndarray) -> np.ndarray:
    others = [j for j in range(x.shape[0]) if j != k]
    return x[k] - coef @ x[others]


def step_events(x: np.ndarray, fs: float, win_s: float = 2.0, min_step: float = 1.0,
                dead_s: float | None = None):
    """Steps in the summed series: difference of the means of the win_s before and
    after each sample, local maxima of |step| >= min_step, at least dead_s apart.
    For each, the per-phase step vector and the balanced share
    3*mean(d)^2 / |d|^2 (1 for a balanced step, 1/3 for a single-phase step)."""
    w = int(round(win_s * fs))
    dead = int(round((dead_s if dead_s is not None else 2 * win_s) * fs))
    c = np.cumsum(np.concatenate([np.zeros((x.shape[0], 1)), x], axis=1), axis=1)
    idx = np.arange(w, x.shape[1] - w + 1)
    before = (c[:, idx] - c[:, idx - w]) / w
    after = (c[:, idx + w] - c[:, idx]) / w
    d = after - before                                  # (3, len(idx))
    s = np.abs(d.sum(axis=0))
    events = []
    order = np.argsort(-s)
    taken = np.zeros(len(idx), dtype=bool)
    for j in order:
        if s[j] < min_step:
            break
        if taken[max(0, j - dead):j + dead + 1].any():
            continue
        taken[j] = True
        dv = d[:, j]
        events.append({"i": int(idx[j]), "t_s": float(idx[j] / fs), "step": dv.tolist(),
                       "step_sum": float(dv.sum()),
                       "balanced_share": float(3 * dv.mean() ** 2 / np.sum(dv ** 2))})
    return sorted(events, key=lambda e: e["i"])
