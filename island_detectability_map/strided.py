"""The report's spectral detector (welch_engine / run_pulse.experiment) on one-hour windows at any stride,
with a job placed at a chosen start: looped to fill the record, or switched on for a finite run.

Statistic, per window: Welch PSD (Hann, 50% overlap, each segment's mean removed, segments with a
non-finite sample skipped), divided by the baseline (median over the learning windows), site lines notched
(a ratio of notch_ratio or more in any learning window, padded by notch_pad bins), maximum over kept bins.
Noise segment DFTs are computed once per window; a placed job's segment DFTs are computed directly from
its samples for the windows it touches, and added (the DFT is linear). A window the job does not touch
scores its no-signal statistic.

At stride = window and with the job looped from report_draws' starts, the statistic is welch_engine's.

Origin: verification@5941aa10
"""
from __future__ import annotations

import numpy as np

from island_detectability_map import welch_engine as WE


def window_starts(n_samples, win_n, stride_n):
    return np.arange(0, n_samples - win_n + 1, stride_n)


def prepare(x, fs, win_s, stride_s, learn, band, seg_s, notch_ratio=10.0, notch_pad=2):
    """Noise segment DFTs at the kept bins, baseline, notches and the no-signal statistic per window.
    learn: indices (into the window list) of the learning windows."""
    x = np.asarray(x, float)
    n = int(win_s * fs)
    L = int(round(seg_s * fs))
    starts = window_starts(len(x), n, int(stride_s * fs))
    f = np.fft.rfftfreq(L, 1.0 / fs)
    sel = (f >= band[0]) & (f <= band[1])
    sb = np.flatnonzero(sel)
    w = np.hanning(L)
    sc = WE._psd_scale(L, fs)
    offs = np.arange(0, n - L + 1, L // 2)
    segs = x[starts[:, None, None] + offs[None, :, None] + np.arange(L)[None, None, :]]   # windows x segs x L
    ok = np.all(np.isfinite(segs), axis=2)
    Z = np.fft.rfft((segs - segs.mean(axis=2, keepdims=True)) * w, axis=2)[:, :, sb]
    spectra = np.zeros((len(starts), len(f)))
    with np.errstate(invalid="ignore"):
        for i in range(len(starts)):
            spectra[i, sb] = np.mean(np.abs(Z[i][ok[i]]) ** 2, axis=0) * sc if ok[i].any() else np.nan
    learn = list(learn)
    baseline = np.median(spectra[learn], axis=0)
    if not np.all(np.isfinite(baseline[sel])):
        raise ValueError("a learning window has no finite segment; the baseline is undefined")
    with np.errstate(divide="ignore", invalid="ignore"):
        noisy = (spectra[learn].max(axis=0) / baseline >= notch_ratio) & sel
    notch = noisy.copy()
    for k in range(1, notch_pad + 1):
        notch[k:] |= noisy[:-k]; notch[:-k] |= noisy[k:]
    keep = sel & ~notch
    kb = np.flatnonzero(keep[sb])
    base_k = baseline[keep]
    null = np.array([np.max(spectra[i][keep] / base_k) if ok[i].any() else np.nan for i in range(len(starts))])
    return {"starts": starts, "win_n": n, "L": L, "w": w, "sc": sc, "band_bins": sb, "kb": kb, "freqs": f, "keep": keep,
            "notch": notch, "baseline": baseline, "base_k": base_k, "null": null, "learn": learn,
            "Z": [Z[i][ok[i]][:, kb] for i in range(len(starts))],
            "seg_abs": [starts[i] + offs[ok[i]] for i in range(len(starts))]}


def report_draws(seed, n_swings, n_windows, trace_len, f0_range):
    """welch_engine.experiment's random starts (per swing, per window: one unused uniform, then the start)."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_swings, n_windows), int)
    for j in range(n_swings):
        for t in range(n_windows):
            rng.uniform(np.log(f0_range[0]), np.log(f0_range[1]))
            out[j, t] = int(rng.integers(0, trace_len))
    return out


class Placed:
    """A job on the record's sample clock: trace[(t - t0 + phase) mod len(trace)] + add, looped. With dur
    None it fills the record (t0 taken as 0); otherwise it is on for t0 <= t < t0 + dur and zero outside."""

    def __init__(self, trace, t0=None, dur=None, phase=0, add=0.0):
        self.trace = np.asarray(trace, float)
        self.t0 = 0 if t0 is None else int(t0)
        self.dur = None if dur is None else int(dur)
        self.phase = int(phase)
        self.add = float(add)

    def __call__(self, t):
        t = np.asarray(t)
        v = self.trace[(t - self.t0 + self.phase) % len(self.trace)] + self.add
        if self.dur is not None:
            v = np.where((t >= self.t0) & (t < self.t0 + self.dur), v, 0.0)
        return v

    def touches(self, a, b):
        return self.dur is None or (self.t0 < b and self.t0 + self.dur > a)


def placed(trace, t0=None, dur=None, phase=0, add=0.0):
    return Placed(trace, t0, dur, phase, add)


def job_dfts(st, i, job):
    """The job's segment DFTs (segments x band bins) in window i, or None if the job does not touch it.
    Channels with the same segments and band share them; each keeps its own bins in stat_from."""
    ws = int(st["starts"][i])
    if not job.touches(ws, ws + st["win_n"]) or len(st["seg_abs"][i]) == 0:
        return None
    y = job(st["seg_abs"][i][:, None] + np.arange(st["L"]))
    return np.fft.rfft((y - y.mean(axis=1, keepdims=True)) * st["w"], axis=1)[:, st["band_bins"]]


def stat_from(st, i, Zj, coef):
    if Zj is None or coef == 0:
        return float(st["null"][i])
    p = np.mean(np.abs(st["Z"][i] + coef * Zj[:, st["kb"]]) ** 2, axis=0) * st["sc"]
    return float(np.max(p / st["base_k"]))


def statistic(st, i, job, coef):
    """Statistic of window i with coef x job added."""
    return stat_from(st, i, job_dfts(st, i, job), coef)


def window_class(starts, win_n, t0, dur):
    """How each window [s, s + win_n) meets the run [t0, t0 + dur): 'start' (straddles its start), 'stop',
    'both', 'inside', or 'outside'."""
    stop = t0 + dur
    out = []
    for s in np.asarray(starts):
        e = s + win_n
        a, b = s < t0 < e, s < stop < e
        out.append("both" if a and b else "start" if a else "stop" if b else
                   "inside" if (s >= t0 and e <= stop) else "outside")
    return np.array(out)


def alarm_counts(alarm, starts, win_n, hour_n):
    """(alarming windows, distinct clock hours [h hour_n, (h + 1) hour_n) overlapped by an alarming window)."""
    alarm = np.asarray(alarm, bool)
    hours = set()
    for s in np.asarray(starts)[alarm]:
        hours.update(range(int(s) // hour_n, (int(s) + win_n - 1) // hour_n + 1))
    return int(alarm.sum()), len(hours)


def run_outcome(found, null_alarm, starts, win_n, t0, dur):
    """A run is found if any window touching it is found; its chance is any no-signal alarm in the same
    windows; the delay is the end of the first found touching window minus t0 (NaN if not found)."""
    starts = np.asarray(starts)
    touch = (starts < t0 + dur) & (starts + win_n > t0)
    hit = np.asarray(found, bool) & touch
    return {"found": bool(hit.any()), "chance": bool((np.asarray(null_alarm, bool) & touch).any()),
            "delay_s": float(starts[hit][0] + win_n - t0) if hit.any() else float("nan"),
            "windows": int(touch.sum())}
