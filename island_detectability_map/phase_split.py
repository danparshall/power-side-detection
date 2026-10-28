"""Three-phase split of a training island's swing by NVIDIA's DGX SuperPOD H100 PSU phase tables, the
mapping through a delta-wye (Dyn) service transformer, and the scoring helpers for the split runs.

Source of the tables: NVIDIA, "DGX SuperPOD Data Center Design (H100), Electrical", Tables 11-14
(https://docs.nvidia.com/dgx-superpod/design-guides/dgx-superpod-data-center-design-h100/latest/electrical.html,
read 2026-10-22 from the page's HTML). Each DGX H100 has six PSUs; each table gives, for two systems
in one rack, the rPDU phase (A, B or C) of every PSU. Rack-level rPDUs are 415 V wye, so each PSU sits
line-to-neutral (240 V). Assumed here, not stated as a number by NVIDIA: the six PSUs share their
system's load equally (current-sharing error is a parameter), and phase labels are the same on every rPDU.

Phasor convention: secondary phase k (A, B, C = 0, 1, 2) has voltage along a^-k, a = exp(2 pi j / 3).
A PSU load is in phase with its voltage, so its current is f_k a^-k (f = share of the island's swing).
A Dyn transformer passes each secondary winding's current to the delta winding on the primary; a primary
line current is the difference of two winding currents. Normalised so a balanced load maps to itself:
  Dyn1:  J_k = (I_k - I_{k-1}) e^{+j pi/6} / sqrt 3        Dyn11: J_k = (I_k - I_{k+1}) e^{-j pi/6} / sqrt 3
The zero-sequence part of the secondary currents circulates in the delta and never reaches the lines.

What a current-magnitude meter sees: a small added current changes |I| by its component along the
background current. With the background lagging its voltage by bg_lag_deg, that is Re(J_k a^k e^{j lag}).
bg_lag_deg = 0 is the report's convention (job added in phase, kW / 7.2 kV = A, kW taken as kVA).

Origin: verification@29bb8624
"""
from __future__ import annotations

import itertools

import numpy as np

A = np.exp(2j * np.pi / 3)
K3 = np.arange(3)

# Per system, PSU1..PSU6 phase, as printed (Tables 11-14).
_N1_R1 = [list("ABBCAB"), list("CACAAB")]
_N1_R2 = [list("CABCCA"), list("BCABBC")]
_ENH_R1 = [list("ABABAB"), list("CACACA")]
_ENH_R2 = [list("BCBCBC"), list("ABABAB")]
CONFIGS = {
    "n1_1dgx": _N1_R1[:1],
    "n1_2dgx": _N1_R1,
    "n1_4dgx": _N1_R1 + _N1_R2,
    "enh_1dgx": _ENH_R1[:1],
    "enh_2dgx": _ENH_R1,
    "enh_4dgx": _ENH_R1 + _ENH_R2,
}
GPUS_PER_DGX = 8


def _systems(config):
    return CONFIGS[config]


def psu_phases(config):
    """Phase letter of every PSU, system by system."""
    return [p for s in _systems(config) for p in s]


def phase_counts(config):
    ph = psu_phases(config)
    return tuple(ph.count(c) for c in "ABC")


def n_gpus(config):
    return GPUS_PER_DGX * len(_systems(config))


def shares(config, eps=None):
    """Share of the island's swing on phases A, B, C. eps: per-PSU current-sharing error (fraction, in
    psu_phases order); a PSU carries (1 + eps) / sum over its system of (1 + eps) of that system's load,
    and every system carries the same load."""
    systems = _systems(config)
    eps = np.zeros(6 * len(systems)) if eps is None else np.asarray(eps, float)
    f = np.zeros(3)
    for s, phs in enumerate(systems):
        w = 1.0 + eps[6 * s:6 * s + 6]
        for p, wi in zip(phs, w / w.sum()):
            f["ABC".index(p)] += wi / len(systems)
    return f


def dyn_line_currents(f, group="Dyn1"):
    """Primary line currents (per unit of the island's swing) for in-phase secondary loads with shares f."""
    I = np.asarray(f, float) * A ** (-K3)
    if group == "Dyn1":
        return (I - np.roll(I, 1)) * np.exp(1j * np.pi / 6) / np.sqrt(3)
    if group == "Dyn11":
        return (I - np.roll(I, -1)) * np.exp(-1j * np.pi / 6) / np.sqrt(3)
    raise ValueError(f"group must be Dyn1 or Dyn11, not {group!r}")


def _along_background(J, bg_lag_deg):
    return np.real(J * A ** K3 * np.exp(1j * np.radians(bg_lag_deg)))


def dyn_inphase(f, group="Dyn1", bg_lag_deg=0.0):
    """Share of the island's swing a magnitude meter sees on each primary line, through a Dyn transformer."""
    return _along_background(dyn_line_currents(f, group), bg_lag_deg)


def project_unmapped(f, bg_lag_deg=0.0):
    """Same, with the island's phases taken straight onto the bus phases (no transformer)."""
    return _along_background(np.asarray(f, float) * A ** (-K3), bg_lag_deg)


def unbalance_coefs(g):
    """Coefficient of the job on each phase's unbalance (phase k minus the mean of the other two)."""
    g = np.asarray(g, float)
    return g - (g.sum() - g) / 2


def assignments(v):
    """Every distinct ordering of v onto the three bus phases (which island phase lands on which bus
    phase is unknown)."""
    seen, out = set(), []
    for p in itertools.permutations(range(3)):
        w = np.asarray(v, float)[list(p)]
        key = tuple(np.round(w, 12))
        if key not in seen:
            seen.add(key)
            out.append(w)
    return out


def score_family(nulls, injs, learn, score, margin):
    """One threshold for a family of channels: worst learning-hour null over the family x margin.
    nulls: per channel, statistic per hour; injs: per channel, scales x hours. An hour is a false alarm
    if any channel's null is above the threshold; the job is found in it if any channel's injected
    statistic is. Found is the share of scored hours where every channel has a statistic."""
    thr = max(np.nanmax(np.asarray(n)[learn]) for n in nulls) * margin
    above = lambda a: np.nan_to_num(a, nan=-np.inf) > thr
    fa = sum(bool(any(above(np.asarray(n)[i]) for n in nulls)) for i in score)
    ok = [i for i in score if all(np.isfinite(np.asarray(n)[i]) for n in nulls)]
    n_sc = np.asarray(injs[0]).shape[0]
    found = [float(np.mean([any(above(np.asarray(m)[j, i]) for m in injs) for i in ok])) for j in range(n_sc)]
    return {"threshold": float(thr), "false_alarms": int(fa), "found": found, "scored_hours": len(ok),
            "score_hours": len(score)}


class HarmonicSum:
    """Harmonic-sum statistic on a row of ratios (spectrum / baseline) at the kept bins `freqs`: for each
    trial fundamental f0, the mean over harmonics n = 1..n_harm of the largest ratio within 1.5 bins of
    n f0 (harmonics with no kept bin there are left out of the mean); the statistic is the largest over f0."""

    def __init__(self, freqs, f0_grid, n_harm):
        freqs = np.asarray(freqs, float)
        df = float(np.min(np.diff(freqs)))
        cells = [[np.flatnonzero(np.abs(freqs - n * f0) <= 1.5 * df + 1e-12) for n in range(1, n_harm + 1)]
                 for f0 in f0_grid]
        width = max(1, max(len(c) for row in cells for c in row))
        self.idx = np.full((len(f0_grid), n_harm, width), len(freqs))      # len(freqs) points at a NaN pad
        for a, row in enumerate(cells):
            for b, c in enumerate(row):
                self.idx[a, b, :len(c)] = c

    def __call__(self, R):
        R = np.asarray(R, float)
        Rp = np.concatenate([R, np.full((R.shape[0], 1), np.nan)], axis=1)
        V = np.nan_to_num(Rp[:, self.idx], nan=-np.inf).max(axis=3)        # rows x f0 x harmonics
        have = np.isfinite(V)
        mean = np.where(have, V, 0.0).sum(axis=2) / np.maximum(have.sum(axis=2), 1)
        mean[~have.any(axis=2)] = -np.inf
        return mean.max(axis=1)
