"""Hour-of-day classes and per-class thresholds for the current-monitor detectors.

Every detection figure so far uses one threshold: the worst no-signal statistic over all
test hours. This module labels each hourly window by local time and applies a threshold
per class. Two cautions are built in:

* A threshold that is the maximum over fewer windows is lower for that reason alone.
  split_control() measures how much: random splits of the same size, same statistics.
* The plan's "quiet" class mixes weeknights with the weekend. hour_labels() keeps the
  three apart ("work", "weeknight", "weekend"); two_class() folds them to the plan's rule.

Origin: verification@67698f42
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

from island_detectability_map.run_pulse import swing_at


def hour_labels(t0_ns: int, n_hours: int, tz: str = "America/Los_Angeles",
                work_hours=(7, 19)) -> list[str]:
    """Label for each whole hour after t0: 'work' (weekday, local work_hours),
    'weeknight' (weekday, other hours) or 'weekend'."""
    t0 = datetime.fromtimestamp(t0_ns / 1e9, tz=timezone.utc)
    z = ZoneInfo(tz)
    out = []
    for h in range(n_hours):
        t = (t0 + timedelta(hours=h)).astimezone(z)
        if t.weekday() >= 5:
            out.append("weekend")
        elif work_hours[0] <= t.hour < work_hours[1]:
            out.append("work")
        else:
            out.append("weeknight")
    return out


def two_class(label: str) -> str:
    """The plan's rule: working hours (weekday 07:00-19:00 local) or quiet."""
    return "work" if label == "work" else "quiet"


def class_rate(null: np.ndarray, inj: np.ndarray, idx, thr: float | None = None) -> float:
    """Detection rate over windows idx, at zero false alarms within idx: the threshold is
    the largest no-signal statistic among idx unless thr is given."""
    idx = list(idx)
    t = float(np.max(null[idx])) if thr is None else thr
    return float(np.mean(inj[idx] > t))


def swing_for(null, inj, swings, idx, level: float, thr: float | None = None):
    """Swing at which the per-class detection rate first reaches level (log-interpolated)."""
    rows = [{"swing_kva": s, "r": class_rate(null, inj[j], idx, thr)} for j, s in enumerate(swings)]
    return swing_at(rows, "r", level)


def split_control(null, inj, swings, size: int, level: float, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """Detectable swing for n_draws random subsets of `size` windows, each with its own
    worst-window threshold. NaN where the level is never reached."""
    n = len(null)
    out = np.full(n_draws, np.nan)
    for d in range(n_draws):
        idx = rng.choice(n, size=size, replace=False)
        v = swing_for(null, inj, swings, idx, level)
        if v is not None:
            out[d] = v
    return out
