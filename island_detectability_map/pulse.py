"""Sub-second 'iteration pulse' channel: spectra, synthetic pulse, line detector.

The detector is the simplest defensible one: Welch spectrum of a window, divided
bin-by-bin by a baseline spectrum of the same point, maximised over a search band.
A new narrow line (a training cadence) shows up as a large ratio. Lines already
present in the baseline are divided out.

Origin: verification@f7a9006e
"""
from __future__ import annotations

import numpy as np


def rect_fundamental_rms(swing: float, duty: float = 0.5) -> float:
    """RMS of the fundamental of a rectangular wave with peak-to-peak `swing`."""
    return float(np.sqrt(2.0) / np.pi * swing * np.sin(np.pi * duty))


def welch_psd(x: np.ndarray, fs: float, seg_seconds: float = 100.0):
    """One-sided PSD (units^2/Hz), Hann window, 50% overlap, mean removed per segment."""
    n = int(round(seg_seconds * fs))
    step = n // 2
    w = np.hanning(n)
    scale = fs * np.sum(w * w)
    acc, count = None, 0
    for start in range(0, len(x) - n + 1, step):
        seg = x[start:start + n]
        if not np.all(np.isfinite(seg)):
            continue
        seg = (seg - seg.mean()) * w
        p = np.abs(np.fft.rfft(seg)) ** 2 / scale
        p[1:-1] *= 2.0
        acc = p if acc is None else acc + p
        count += 1
    if count == 0:
        raise ValueError("no complete finite segment")
    return np.fft.rfftfreq(n, 1.0 / fs), acc / count


def pulse_train(n: int, fs: float, f0: float, swing: float, duty: float = 0.8,
                jitter: float = 0.01, rng: np.random.Generator | None = None) -> np.ndarray:
    """Mean-removed rectangular power pulse: `duty` of each step at full power, the
    rest `swing` lower. Step durations jitter by a fractional `jitter` (1 sigma),
    so the cadence random-walks as a real training loop's does."""
    rng = rng or np.random.default_rng()
    t_step = 1.0 / f0
    n_steps = int(n / fs / t_step * 1.2) + 10
    durs = t_step * (1.0 + jitter * rng.standard_normal(n_steps))
    starts = np.concatenate([[0.0], np.cumsum(durs)[:-1]]) - rng.uniform(0, t_step)
    t = np.arange(n) / fs
    idx = np.clip(np.searchsorted(starts, t, side="right") - 1, 0, n_steps - 1)
    phase = (t - starts[idx]) / durs[idx]
    y = np.where(phase < duty, swing, 0.0)
    return y - y.mean()


def line_statistic(x: np.ndarray, fs: float, baseline_psd: np.ndarray, band=(0.1, 3.0),
                   seg_seconds: float = 100.0):
    """max over band of PSD(window)/PSD(baseline); returns (statistic, frequency of the max)."""
    f, s = welch_psd(x, fs, seg_seconds)
    sel = (f >= band[0]) & (f <= band[1])
    ratio = s[sel] / baseline_psd[sel]
    i = int(np.argmax(ratio))
    return float(ratio[i]), float(f[sel][i])


def ar1_log_periods(k: int, sigma_log: float, rho: float, rng: np.random.Generator) -> np.ndarray:
    """k draws of x with stationary N(0, sigma_log^2) marginal and lag-1 correlation rho."""
    eps = rng.standard_normal(k)
    x = np.empty(k)
    x[0] = eps[0]
    c = np.sqrt(1.0 - rho * rho)
    for i in range(1, k):
        x[i] = rho * x[i - 1] + c * eps[i]
    return sigma_log * x


def training_trace(n: int, fs: float, f_med: float, swing: float, sigma_log: float = 0.1, rho: float = 0.95,
                   n_chips: int = 16, straggle: float = 1.0, sync_frac: float = 0.2,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """Straggler model: each step's lag is set by the last chip to finish, which triggers the
    gradient exchange, so the cadence wanders. Each training step: all chips start
    together (power jumps up by `swing`), each chip drops out when it finishes, and when
    the LAST one finishes the gradients are exchanged (`sync_frac` of the step, all chips
    low) and the next step starts. Step duration is lognormal about the median and each
    draw depends on the one before (AR(1) in log-duration, correlation `rho`).

    straggle = spread of chip finish times as a fraction of the step: 1.0 -> finish times
    uniform over the whole step (sawtooth); ~0 -> all finish together (rectangle).
    Samples are exact averages over each sample interval (what an RMS meter reports), so
    steps much shorter than 1/fs average out instead of aliasing. Returned mean-removed.
    """
    rng = rng or np.random.default_rng()
    t_end = n / fs
    mu = np.log(1.0 / f_med / (1.0 + sync_frac))
    durs = []
    total, k = 0.0, int(t_end * f_med * 2) + 64
    while total < t_end * 1.05 + 5.0 / f_med:
        d = np.exp(mu + ar1_log_periods(k, sigma_log, rho, rng))
        durs.append(d)
        total += float(d.sum()) * (1.0 + sync_frac)
    d = np.concatenate(durs)
    starts = np.concatenate([[0.0], np.cumsum(d * (1.0 + sync_frac))[:-1]]) - rng.uniform(0, d[0])
    u = rng.uniform(0.0, 1.0, (len(d), n_chips))
    u[:, 0] = 0.0                                  # the last chip to finish defines the step
    fin = starts[:, None] + d[:, None] * (1.0 - straggle * u)
    times = np.concatenate([starts, fin.ravel()])
    delta = np.concatenate([np.full(len(d), swing), np.full(fin.size, -swing / n_chips)])
    o = np.argsort(times, kind="stable")
    times, delta = times[o], delta[o]
    c1, c2 = np.cumsum(delta), np.cumsum(delta * times)
    edges = np.arange(n + 1) / fs
    idx = np.searchsorted(times, edges, side="right") - 1
    integ = np.where(idx >= 0, edges * c1[np.maximum(idx, 0)] - c2[np.maximum(idx, 0)], 0.0)
    y = np.diff(integ) * fs
    return y - y.mean()
