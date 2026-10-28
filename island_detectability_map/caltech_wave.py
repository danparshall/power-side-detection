"""Amplitude envelope of the Caltech point-on-wave captures, for background at 2-30 Hz.
Tests: tests/test_caltech_wave.py.

A capture is ~1 s of one channel at ~2.5 kHz with slightly uneven sample times. The envelope
is the RMS of the fundamental as a function of time: the wave is multiplied by a complex
exponential at the capture's own fundamental frequency and averaged over exactly one cycle
(which also nulls harmonics), then sampled at OUT_HZ. The one-cycle average attenuates fast
modulation (0.64 in amplitude at 30 Hz); capture_psd divides that response back out.

One capture gives a spectrum at ~1 Hz resolution. Its 1 Hz bin holds leakage from slower
drift, so bands start at 2 Hz. Meter noise is included in whatever this measures: band figures
are upper bounds on the load's own fluctuation.

Origin: verification@b2855af0
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

OUT_HZ = 240.0
RESAMPLE_HZ = 4800.0


def _fundamental(t, v, guess=60.0):
    """Fundamental frequency from the phase advance of the demodulated wave."""
    z = v * np.exp(-2j * np.pi * guess * t)
    n = len(t) // 2
    a, b = np.sum(z[:n]), np.sum(z[n:])
    dt = t[n:].mean() - t[:n].mean()
    return guess + float(np.angle(b * np.conj(a))) / (2 * np.pi * dt)


def envelope(t, v, f0: float | None = None):
    """(times, RMS envelope) of one capture. t in seconds, any origin."""
    t = np.asarray(t, float); v = np.asarray(v, float)
    f0 = f0 or _fundamental(t, v)
    tu = np.arange(t[0], t[-1], 1.0 / RESAMPLE_HZ)
    vu = CubicSpline(t, v)(tu)          # linear interpolation would shave 0.2% off a 60 Hz wave at this spacing
    z = vu * np.exp(-2j * np.pi * f0 * tu)
    w = int(round(RESAMPLE_HZ / f0))                       # one cycle
    c = np.cumsum(np.r_[0, z])
    avg = (c[w:] - c[:-w]) / w
    tc = tu[: len(avg)] + (w - 1) / 2.0 / RESAMPLE_HZ
    step = int(round(RESAMPLE_HZ / OUT_HZ))
    return tc[::step], np.sqrt(2.0) * np.abs(avg[::step])


def capture_psd(te, env, f0: float = 60.0):
    """One-sided PSD of one capture's envelope (units^2/Hz), Hann window, mean removed,
    corrected for the one-cycle average."""
    fs = 1.0 / np.median(np.diff(te))
    x = env - env.mean()
    w = np.hanning(len(x))
    p = np.abs(np.fft.rfft(x * w)) ** 2 / (fs * np.sum(w ** 2))
    p[1:] *= 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    resp = np.sinc(f / f0) ** 2
    ok = f < 0.75 * f0
    p = np.where(ok, p / np.maximum(resp, 1e-6), 0.0)
    return f, p


def phase_kva(caps: dict, current, voltage, line_to_line: bool = False):
    """Per-phase kVA envelopes on a common time base from one meter's captures.
    caps: channel -> (t seconds, values). voltage: three line-to-neutral channels, or two
    line-to-line channels with line_to_line=True (mean / sqrt(3) used for every phase)."""
    env = {k: envelope(*caps[k]) for k in list(current) + list(voltage)}
    t0 = max(e[0][0] for e in env.values()); t1 = min(e[0][-1] for e in env.values())
    te = np.arange(t0, t1, 1.0 / OUT_HZ)
    g = {k: np.interp(te, *env[k]) for k in env}
    if line_to_line:
        vln = np.mean([g[k] for k in voltage], axis=0) / np.sqrt(3.0)
        v = [vln, vln, vln]
    else:
        v = [g[k] for k in voltage]
    return te, np.vstack([g[c] * vk / 1e3 for c, vk in zip(current, v)])


def band_rms_kva(te, kva, lo: float, hi: float) -> dict:
    """Band RMS of the sum, each phase and each imbalance residual for one capture."""
    def b(x):
        f, p = capture_psd(te, x)
        sel = (f >= lo) & (f <= hi)
        return float(np.sqrt(np.sum(p[sel]) * (f[1] - f[0])))
    resid = [kva[k] - 0.5 * (kva[(k + 1) % 3] + kva[(k + 2) % 3]) for k in range(3)]
    return {"sum": b(kva.sum(axis=0)), "phase": [b(kva[k]) for k in range(3)], "resid": [b(r) for r in resid]}


def harmonics(t, v, orders=(1, 3, 5, 7, 9, 11, 13), f0: float | None = None) -> dict:
    """RMS size of each harmonic of one capture: the wave is correlated with a complex
    exponential at n times the capture's own fundamental over a whole number of cycles.
    Returns {order: rms}."""
    t = np.asarray(t, float); v = np.asarray(v, float)
    f0 = f0 or _fundamental(t, v)
    ncyc = int((t[-1] - t[0]) * f0)
    tu = np.arange(t[0], t[0] + ncyc / f0, 1.0 / (4 * RESAMPLE_HZ))
    vu = CubicSpline(t, v)(tu)
    return {n: float(np.sqrt(2.0) * np.abs(np.mean(vu * np.exp(-2j * np.pi * n * f0 * tu)))) for n in orders}

