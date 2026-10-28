"""Synthetic one-hour envelopes for look-alike loads, and their spectrograms.

Each generator returns real power P (kW) and reactive power Q (kvar). Steady
loads get Q from their power factor; motor starts are modelled as a locked-rotor
kVA surge at a low power factor, decaying as the rotor comes up to speed. A
monitor on the service sees RMS current I = sqrt(P² + Q²) / V.

Origin: verification@81ea80da
"""

from pathlib import Path

import numpy as np
import yaml
from scipy.signal import stft

PARAMS = yaml.safe_load((Path(__file__).parent / "params.yaml").read_text())


def _grid(P):
    n = round(P["duration_s"] / P["dt_s"])
    return np.arange(n) * P["dt_s"]


def _surge(t, start, kva, pf, tau, run_kva):
    """Motor start at `start`: the excess of the locked-rotor kVA over the running kVA,
    at the locked-rotor power factor, decaying with `tau`. Returns (P kW, Q kvar)."""
    x = np.maximum(t - start, 0.0)
    s = (t >= start) * (kva - run_kva) * np.exp(-x / tau)
    return s * pf, s * np.sqrt(1 - pf**2)


def _q(p, pf):
    """Reactive power of a load at power factor pf drawing real power p."""
    return p * np.tan(np.arccos(pf))


def ev_charger(t, q, rng):
    p = np.zeros_like(t)
    on = (t >= q["start_s"]) & (t < q["stop_s"])
    ramp = np.clip((t - q["start_s"]) / q["ramp_s"], 0, 1)
    taper = np.where(
        t >= q["taper_start_s"],
        np.interp(t, [q["taper_start_s"], q["stop_s"]], [q["level_kw"], q["taper_end_kw"]]),
        q["level_kw"],
    )
    p[on] = (ramp * taper)[on]
    p += rng.normal(0, q["noise_kw"], t.size) * on
    return p, _q(p, q["pf"])


def heat_pump(t, q, rng):
    cycle = q["on_s"] + q["off_s"]
    phase = np.mod(t, cycle)
    on = phase < q["on_s"]
    p = on * (q["run_kw"] + q["fan_kw"]) + rng.normal(0, q["noise_kw"], t.size) * on
    qq = _q(p, q["pf"])
    run_kva = (q["run_kw"] + q["fan_kw"]) / q["pf"]
    for s in np.arange(0, t[-1], cycle):
        sp, sq = _surge(t, s, q["locked_rotor_kva"], q["locked_rotor_pf"], q["surge_tau_s"], run_kva)
        p, qq = p + sp, qq + sq
    return p, qq


def laser_fuser(t, q, rng):
    p = np.full_like(t, q["standby_kw"])
    motor = np.zeros_like(t)
    for s in np.arange(0, t[-1], q["job_every_s"]):
        job = (t >= s) & (t < s + q["job_len_s"])
        warm = (t - s) < 20
        duty = np.where(warm, q["duty_warm"], q["duty_hold"])
        pulse = np.mod(t - s, q["pulse_period_s"]) < duty * q["pulse_period_s"]
        p += job * q["heater_kw"] * pulse
        motor += job * q["motor_kw"]
    p += motor + rng.normal(0, q["noise_kw"], t.size)
    return p, _q(motor, q["motor_pf"])


def well_pump(t, q, rng):
    p = np.zeros_like(t)
    for s in np.arange(120, t[-1], q["every_s"]):
        p += ((t >= s) & (t < s + q["run_len_s"])) * q["run_kw"]
    p += rng.normal(0, q["noise_kw"], t.size) * (p > 0)
    qq = _q(p, q["pf"])
    run_kva = q["run_kw"] / q["pf"]
    for s in np.arange(120, t[-1], q["every_s"]):
        sp, sq = _surge(t, s, q["locked_rotor_kva"], q["locked_rotor_pf"], q["surge_tau_s"], run_kva)
        p, qq = p + sp, qq + sq
    return p, qq


def training_16gpu(t, q, rng):
    T = q["period_s"]
    low_len = q["low_fraction"] * T
    phase = np.mod(t, T)
    e = q["edge_s"]
    # 1 in the high phase, 0 in the low phase, linear edges of width e.
    high = np.clip((phase - low_len) / e, 0, 1) * np.clip((T - phase) / e, 0, 1)
    gpu = q["gpu_high_kw"] - q["swing_kw"] * (1 - high)
    ckpt = np.mod(t - q["checkpoint_offset_s"], q["checkpoint_every_s"]) < q["checkpoint_len_s"]
    gpu = np.where(ckpt, q["gpu_high_kw"] - q["swing_kw"], gpu)
    p = q["node_kw"] + gpu + rng.normal(0, q["noise_kw"], t.size)
    return p, _q(p, q["pf"])


def crypto_miner(t, q, rng):
    p = np.full_like(t, q["level_kw"])
    dip = (t >= q["dip_at_s"]) & (t < q["dip_at_s"] + q["dip_len_s"])
    p[dip] = q["dip_kw"]
    p += rng.normal(0, q["noise_kw"], t.size)
    return p, _q(p, q["pf"])


def inference_server(t, q, rng):
    dt = t[1] - t[0]
    n_req = rng.poisson(q["request_rate_hz"] * t[-1])
    starts = rng.uniform(0, t[-1], n_req)
    lengths = rng.exponential(q["burst_mean_s"], n_req)
    edges = np.zeros(t.size + 1)
    i0 = np.clip((starts / dt).astype(int), 0, t.size - 1)
    i1 = np.clip(((starts + lengths) / dt).astype(int), 0, t.size)
    np.add.at(edges, i0, q["burst_kw"])
    np.add.at(edges, i1, -q["burst_kw"])
    p = q["base_kw"] + np.cumsum(edges[:-1])
    p = np.minimum(p, q["max_kw"]) + rng.normal(0, q["noise_kw"], t.size)
    return p, _q(p, q["pf"])


GENERATORS = {
    "ev_charger": ev_charger,
    "heat_pump": heat_pump,
    "laser_fuser": laser_fuser,
    "well_pump": well_pump,
    "training_16gpu": training_16gpu,
    "crypto_miner": crypto_miner,
    "inference_server": inference_server,
}


def loads_pq(P=PARAMS):
    """{name: (P kW, Q kvar)} for every load."""
    t = _grid(P)
    rng = np.random.default_rng(P["seed"])
    return {name: g(t, P[name], rng) for name, g in GENERATORS.items()}


def loads(P=PARAMS):
    """{name: real power, kW}."""
    return {name: p for name, (p, _) in loads_pq(P).items()}


def currents(P=PARAMS):
    """{name: RMS current, A} as a monitor on a single-phase service at P['volts'] sees it."""
    return {name: 1e3 * np.hypot(p, q) / P["volts"] for name, (p, q) in loads_pq(P).items()}


def spectrogram(p, P=PARAMS):
    """Power spectral density per window (kW²/Hz), window mean removed.

    Returns (f, window_start_s, S) with S of shape (len(f), n_windows).
    """
    fs = 1 / P["dt_s"]
    nper = int(P["spectrogram"]["window_s"] * fs)
    hop = int(P["spectrogram"]["stride_s"] * fs)
    f, t, Z = stft(
        p, fs=fs, window="hann", nperseg=nper, noverlap=nper - hop,
        detrend="constant", boundary=None, padded=False, scaling="psd",
    )
    S = np.abs(Z) ** 2
    keep = f >= P["spectrogram"]["f_min_hz"]
    return f[keep], t, S[keep]
