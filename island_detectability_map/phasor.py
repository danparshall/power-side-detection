"""Phasor view of the per-phase current monitor.

The earlier per-phase work (phases.py) used current MAGNITUDES. Magnitudes do not add: a new
load's current adds to the background current as a phasor, so a real-power step of size dP
changes the magnitude series by about dP * cos(background power-factor angle), and a balanced
step added to phases whose backgrounds sit at different angles leaves a false trace in the
magnitude residual. With the angle streams each phase's current can be expressed in its own
voltage's frame, where loads add linearly:

    S_k = V_k * I_k * exp(j * (angle_V_k - angle_I_k)) / 1000        kW + j kvar

Real part: real power. Imaginary part: reactive power, positive when the current lags the
voltage (motor-like). The sensor reports both angles against a fixed 60 Hz clock, so they turn
together when the grid is off 60 Hz; only their difference is used.

* complex_residual(s, k): phase k minus the mean of the other two, on complex power. A balanced
  step (equal complex power added to all three phases) cancels exactly; a single-phase step on
  phase k passes with gain 1 and appears at -1/2 in the other two residuals.
* add_load(...): the same load expressed as the magnitude and angle the sensor would report,
  for planting a known signal in the raw streams.
* step_pq / balanced_real_only: real and reactive size of a step, and the test for the edge a
  three-phase island would make (balanced over the phases, almost no reactive part).

Origin: verification@cf2dc041
"""
from __future__ import annotations

import numpy as np


def complex_power(vmag, vang_deg, imag, iang_deg):
    """Per-phase complex power, kW + j kvar. Positive imaginary part = lagging current."""
    d = np.deg2rad(np.asarray(vang_deg, dtype=np.float64) - np.asarray(iang_deg, dtype=np.float64))
    return np.asarray(vmag, dtype=np.float64) * np.asarray(imag, dtype=np.float64) * np.exp(1j * d) / 1e3


def complex_residual(s: np.ndarray, k: int) -> np.ndarray:
    others = [j for j in range(s.shape[0]) if j != k]
    return s[k] - s[others].mean(axis=0)


def add_load(vmag, vang_deg, imag, iang_deg, p_kw, q_kvar=0.0):
    """Current magnitude (A) and angle (degrees) after adding a load of p_kw + j q_kvar on this
    phase. p_kw and q_kvar may be arrays (a time-varying load)."""
    vmag = np.asarray(vmag, dtype=np.float64)
    va = np.deg2rad(np.asarray(vang_deg, dtype=np.float64))
    i = np.asarray(imag, dtype=np.float64) * np.exp(1j * np.deg2rad(np.asarray(iang_deg, dtype=np.float64)))
    # S = V * conj(I) with V = vmag * exp(j va)  ->  added current = conj(dS / V)
    di = np.conj((np.asarray(p_kw, dtype=np.float64) + 1j * np.asarray(q_kvar, dtype=np.float64)) * 1e3
                 / (vmag * np.exp(1j * va)))
    i2 = i + di
    return np.abs(i2), np.rad2deg(np.angle(i2))


def step_pq(p: np.ndarray, q: np.ndarray, idx: np.ndarray, w: int = 3):
    """Change in real and reactive power across each event index t, with the convention of
    run_sync.upsteps: after = mean(x[t+1 : t+1+w]), before = mean(x[t-w : t]); sample t,
    which holds the transition, is skipped. p, q may be 1-D or (phases, n)."""
    idx = np.asarray(idx, dtype=int)

    def d(x):
        c = np.concatenate([np.zeros(x.shape[:-1] + (1,)), np.cumsum(x, axis=-1)], axis=-1)
        return (c[..., idx + 1 + w] - c[..., idx + 1]) / w - (c[..., idx] - c[..., idx - w]) / w
    return d(np.asarray(p, dtype=np.float64)), d(np.asarray(q, dtype=np.float64))


def balanced_real_only(dp3: np.ndarray, dq3: np.ndarray, share=(0.23, 0.43), q_ratio: float = 0.1) -> np.ndarray:
    """True where a step looks like a three-phase island's edge: each phase carries between
    share[0] and share[1] of the real-power change (one third each if perfectly balanced), and
    the total reactive change is at most q_ratio of the total real change. Same balance rule
    as run_caltech_pq_steps; the reactive rule is run_pq_steps'."""
    dp3 = np.asarray(dp3, dtype=np.float64); dq3 = np.asarray(dq3, dtype=np.float64)
    tot = dp3.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sh = dp3 / tot
        ok_bal = np.all((sh >= share[0]) & (sh <= share[1]), axis=0)
        ok_q = np.abs(dq3.sum(axis=0)) <= q_ratio * np.abs(tot)
    return ok_bal & ok_q & (tot != 0)
