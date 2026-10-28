"""Low-pass, then keep one sample in k (tests in tests/test_decimate.py).

The pipeline's 120-to-12 samples-a-second step was a plain ten-sample mean. Its gain at
frequency f is sin(pi f k / fs) / (k sin(pi f / fs)): 0.19 at 10 Hz and 0.17 at 20 Hz, which
land at 2 Hz and 4 Hz in the 12-a-second series, inside the band the spectral detector
searches; and 0.90 at 3 Hz, 0.69 at 5.5 Hz, so the top of that band was attenuated.

Here: a linear-phase FIR low-pass (Kaiser window), unit gain to `keep_hz`, at least
`stop_db` down from `stop_hz` (the lowest frequency that folds back to `keep_hz` when the
output rate is fs / k is fs / k - keep_hz). The filter has an even number of taps, so its
centre falls between two input samples, which is where the centre of a block of k = 10
samples is: output j sits at input position k j + (k - 1) / 2, the same instant as the old
block mean. Nothing is filled: an output whose filter span touches a missing (NaN) input
sample, or runs off either end of the record, is NaN.

Cost of the sharper filter: it is `design(k, fs).size / fs` seconds long (4.4 s at the
defaults), so each missing second blanks about 5.4 s, and a sharp step in the input shows
a few per cent of ringing near 6 Hz either side of it. The spectral detector does not
mind; a detector that counts sample-to-sample edges may.

Origin: verification@f68ea5ab
"""
from __future__ import annotations

import numpy as np
from scipy import signal


def design(k: int, fs: float, keep_hz: float = 5.5, stop_db: float = 70.0) -> np.ndarray:
    """FIR taps (even count, sum 1). Pass band to keep_hz; stop band from fs/k - keep_hz."""
    out_rate = fs / k
    stop_hz = out_rate - keep_hz
    if not 0 < keep_hz < stop_hz:
        raise ValueError("keep_hz must be below half the output rate")
    n, beta = signal.kaiserord(stop_db, (stop_hz - keep_hz) / (fs / 2))
    n += n % 2                                         # even: centre between samples
    if (n + k) % 2:                                    # centre must coincide with a block centre
        n += 1
    h = signal.firwin(n, (keep_hz + stop_hz) / 2, window=("kaiser", beta), fs=fs)
    return h / h.sum()


def lowpass_decimate(x: np.ndarray, k: int, fs: float, h: np.ndarray | None = None) -> np.ndarray:
    """len(x) // k output samples at the old block centres; NaN where the span is incomplete."""
    x = np.asarray(x, dtype=np.float64)
    h = design(k, fs) if h is None else h
    n_taps = len(h)
    n_out = len(x) // k
    out = np.full(n_out, np.nan)
    if len(x) < n_taps:
        return out
    bad = ~np.isfinite(x)
    y = signal.oaconvolve(np.where(bad, 0.0, x), h, mode="valid")          # y[i] spans x[i : i + n_taps]
    # output j is centred at k j + (k-1)/2 = i + (n_taps-1)/2  ->  i = k j + (k - n_taps)/2
    off = (k - n_taps) // 2
    j = np.arange(n_out)
    i = k * j + off
    ok = (i >= 0) & (i < len(y))
    out[ok] = y[i[ok]]
    if bad.any():
        c = np.concatenate([[0], np.cumsum(bad)])
        touched = np.zeros(n_out, dtype=bool)
        touched[ok] = (c[i[ok] + n_taps] - c[i[ok]]) > 0
        out[touched] = np.nan
    return out


def lowpass_decimate_chunked(x: np.ndarray, k: int, fs: float, chunk: int, h: np.ndarray | None = None) -> np.ndarray:
    """Same result as lowpass_decimate, working through x in pieces of `chunk` input samples
    (plus the filter's half-length of context either side), for records too long for memory."""
    if chunk % k:
        raise ValueError("chunk must be a multiple of k")
    h = design(k, fs) if h is None else h
    pad = int(np.ceil(len(h) / 2 / k)) * k
    n_out = len(x) // k
    out = np.full(n_out, np.nan)
    for a in range(0, n_out * k, chunk):
        b = min(n_out * k, a + chunk)
        lo, hi = max(0, a - pad), min(len(x), b + pad)
        y = lowpass_decimate(x[lo:hi], k, fs, h)
        out[a // k:b // k] = y[(a - lo) // k:(a - lo) // k + (b - a) // k]
    return out
