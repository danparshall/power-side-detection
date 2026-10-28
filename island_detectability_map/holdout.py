"""Thresholds set on one set of hours, false alarms and detections counted on another.

Every detection figure before this used a threshold equal to the worst no-signal hour of the
very hours it was scored on, so false alarms were zero by construction. Here:

    thr = threshold(null, set_idx)                 # worst (or a quantile) of the SET hours
    false_alarms(null, thr, score_idx)             # SCORE hours whose no-signal statistic exceeds it
    detection_rate(injected, thr, score_idx)       # SCORE hours whose with-signal statistic exceeds it

"Family" versions cover the unknown-phase case: one threshold for the three phases (or the
three residuals), and an hour is a false alarm if any of the three exceeds it.
rate_interval gives the exact Poisson interval for events per hour.

Origin: verification@cf2dc041
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from island_detectability_map.run_pulse import swing_at


def threshold(null: np.ndarray, set_idx, q: float | None = None) -> float:
    v = np.asarray(null, dtype=float)[list(set_idx)]
    v = v[np.isfinite(v)]
    return float(np.max(v)) if q is None else float(np.quantile(v, q))


def false_alarms(null: np.ndarray, thr: float, score_idx) -> list[int]:
    null = np.asarray(null, dtype=float)
    return [int(i) for i in score_idx if np.isfinite(null[i]) and null[i] > thr]


def family_threshold(nulls: dict, set_idx, q: float | None = None) -> float:
    return max(threshold(v, set_idx, q) for v in nulls.values())


def family_false_alarms(nulls: dict, thr: float, score_idx) -> list[int]:
    hit = set()
    for v in nulls.values():
        hit.update(false_alarms(v, thr, score_idx))
    return sorted(hit)


def detection_rate(inj: np.ndarray, thr: float, score_idx) -> float:
    v = np.asarray(inj, dtype=float)[list(score_idx)]
    return float(np.mean(v > thr))


def swing_for(inj: np.ndarray, swings, thr: float, score_idx, level: float):
    """Swing at which the detection rate over the score hours first reaches level
    (log-interpolated, as run_pulse.swing_at). inj is (swings, hours)."""
    rows = [{"swing_kva": s, "r": detection_rate(inj[j], thr, score_idx)} for j, s in enumerate(swings)]
    return swing_at(rows, "r", level)


def rate_interval(k: int, exposure_hours: float, conf: float = 0.90):
    """Exact Poisson interval for a rate (events per hour) from k events in exposure_hours.
    k = 0: one-sided upper limit at conf (the 'rule of three' at 95%). k > 0: central interval."""
    if k == 0:
        return 0.0, float(-np.log(1.0 - conf) / exposure_hours)
    a = 1.0 - conf
    lo = stats.chi2.ppf(a / 2, 2 * k) / 2
    hi = stats.chi2.ppf(1 - a / 2, 2 * (k + 1)) / 2
    return float(lo / exposure_hours), float(hi / exposure_hours)


def split_at(n: int, cut: int):
    return list(range(cut)), list(range(cut, n))


def by_class(labels, idx) -> dict:
    out: dict = {}
    for i in idx:
        out.setdefault(labels[i], []).append(i)
    return out
