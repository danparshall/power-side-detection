"""Amplitude envelope from the Caltech point-on-wave captures (background
at 1-30 Hz). Written before the code. Captures are ~1 s at ~2.5 kHz with sampling intervals of
400 +/- 4 microseconds (README), so the fixtures use jittered sample times.

Origin: verification@b2855af0
"""
import numpy as np

from island_detectability_map import caltech_wave as W

X = 100.0          # true RMS


def capture(f0=60.0, mod_hz=None, depth=0.0, harmonic=0.0, seed=0, n=2483):
    rng = np.random.default_rng(seed)
    t = np.cumsum(400e-6 + rng.uniform(-4e-6, 4e-6, n))
    amp = 1.0 + (depth * np.cos(2 * np.pi * mod_hz * t + 0.3) if mod_hz else 0.0)
    v = np.sqrt(2) * X * amp * np.cos(2 * np.pi * f0 * t + 1.0)
    v = v + harmonic * np.sqrt(2) * X * np.cos(2 * np.pi * 5 * f0 * t)
    return t, v


def band(te, env, lo, hi):
    f, p = W.capture_psd(te, env)
    sel = (f >= lo) & (f <= hi)
    return float(np.sqrt(np.sum(p[sel]) * (f[1] - f[0])))


def test_steady_wave_gives_a_flat_envelope_at_the_true_rms():
    te, env = W.envelope(*capture())
    assert abs(env.mean() / X - 1) < 1e-3
    assert band(te, env, 2, 30) < 5e-4 * X


def test_off_nominal_frequency_does_not_create_modulation():
    te, env = W.envelope(*capture(f0=59.95))
    assert abs(env.mean() / X - 1) < 1e-3 and band(te, env, 2, 30) < 1e-3 * X


def test_waveform_distortion_does_not_leak_into_the_band():
    te, env = W.envelope(*capture(harmonic=0.05))
    assert band(te, env, 2, 30) < 1e-3 * X


def test_modulation_at_5_hz_is_recovered():
    depth = 0.02
    te, env = W.envelope(*capture(mod_hz=5.0, depth=depth))
    assert abs(band(te, env, 2, 30) / (X * depth / np.sqrt(2)) - 1) < 0.05


def test_modulation_at_25_hz_is_recovered_after_correcting_the_one_cycle_average():
    depth = 0.02
    te, env = W.envelope(*capture(mod_hz=25.0, depth=depth))
    assert abs(band(te, env, 2, 30) / (X * depth / np.sqrt(2)) - 1) < 0.10


def test_phase_power_and_residual_from_three_phase_captures():
    caps = {}
    for k, (i_rms, depth) in enumerate(((10.0, 0.0), (10.0, 0.0), (10.0, 0.05))):
        rng = np.random.default_rng(k)
        t = np.cumsum(400e-6 + rng.uniform(-4e-6, 4e-6, 2483))
        caps[f"L{k+1}"] = (t, np.sqrt(2) * 1000.0 * np.cos(2 * np.pi * 60 * t - 2 * np.pi * k / 3))
        caps[f"S{k+1}"] = (t, np.sqrt(2) * i_rms * (1 + depth * np.cos(2 * np.pi * 7 * t)) * np.cos(2 * np.pi * 60 * t - 2 * np.pi * k / 3))
    te, kva = W.phase_kva(caps, current=("S1", "S2", "S3"), voltage=("L1", "L2", "L3"))
    assert kva.shape[0] == 3 and np.allclose(kva.mean(axis=1), 10.0, rtol=2e-3)
    b = W.band_rms_kva(te, kva, 2, 30)
    expect = 10.0 * 0.05 / np.sqrt(2)                       # only phase c is modulated
    assert abs(b["phase"][2] / expect - 1) < 0.05 and b["phase"][0] < 0.02 * expect
    assert abs(b["sum"] / expect - 1) < 0.05
    assert abs(b["resid"][2] / expect - 1) < 0.05           # the modulated phase's residual sees all of it
    assert abs(b["resid"][0] / (expect / 2) - 1) < 0.06     # the others see half


def test_harmonics_of_a_capture_are_recovered_in_size():
    """A 60 Hz wave of 100 A RMS with a 3% fifth harmonic and a 1% third (question:
    could a rectifier's harmonics be seen in the ~41-samples-per-cycle captures?)."""
    rng = np.random.default_rng(5)
    t = np.cumsum(400e-6 + rng.uniform(-4e-6, 4e-6, 2483))
    f0 = 59.98
    v = np.sqrt(2) * 100.0 * (np.cos(2 * np.pi * f0 * t + 0.4) + 0.03 * np.cos(2 * np.pi * 5 * f0 * t + 1.0) + 0.01 * np.cos(2 * np.pi * 3 * f0 * t))
    h = W.harmonics(t, v, orders=(1, 3, 5, 7))
    assert abs(h[1] - 100.0) < 0.1 and abs(h[5] - 3.0) < 0.05 and abs(h[3] - 1.0) < 0.05 and h[7] < 0.05


def test_square_wave_harmonics_fall_as_one_over_n():
    t = np.arange(0, 1.0, 1 / 20000.0)                       # finely sampled so the edges are clean
    v = np.sign(np.sin(2 * np.pi * 60.0 * t))
    h = W.harmonics(t, v, orders=(1, 3, 5, 7), f0=60.0)
    for n in (3, 5, 7):
        assert abs(h[n] / h[1] - 1.0 / n) < 0.01
