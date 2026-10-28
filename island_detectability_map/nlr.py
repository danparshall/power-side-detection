"""Reader for the NLR generative-AI power profiles (Vercellino et al., arXiv 2604.07345;
data.nlr.gov/submissions/312). Tests in tests/test_nlr.py.

What the files are, and are not (read from the paper and from the logging tool's source,
2026-10-04):
* Software readings: GPU power from NVIDIA's management library, CPU package power from the
  processor's energy counters. Not AC-side: no fans, memory, storage, network cards or
  power-supply losses.
* The training runs were logged in September 2025, every 0.2 s. The logging tool (WattAMeter)
  read NVIDIA's *averaged* power until its commit 984f6f9 of 2026-03-24 ("The old
  implementation was reading averaged power"). On this GPU generation that average is over
  one second. So each GPU reading here is a one-second running mean: see running_mean_gain.
* One log per node, each with its own clock ticks; a node holds four H100s and two CPUs.

Origin: verification@f68ea5ab
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

_POWER = re.compile(r"^(gpu-\d+|cpu-\d+)\[(mW|W)\]$")       # device power only: no '-core', no energy, no temperature


def read_wattameter(path) -> tuple[np.ndarray, dict]:
    """(time-stamps in ns since the epoch, {device: power in W})."""
    with open(path) as f:
        header = f.readline().lstrip("#").split()
        df = pd.read_csv(f, sep=r"\s+", header=None, names=header)
    t = pd.to_datetime(df["timestamp"], format="%Y-%m-%d_%H:%M:%S.%f").to_numpy().astype("datetime64[ns]").astype(np.int64)
    cols = {}
    for c in header:
        m = _POWER.match(c)
        if m:
            cols[m.group(1)] = df[c].to_numpy(dtype=np.float64) * (1e-3 if m.group(2) == "mW" else 1.0)
    return t, cols


def to_grid(t_ns, v, t0_ns: int, dt_s: float, n: int):
    """Place readings on a grid of n slots of dt_s starting at t0_ns, each in its nearest slot.
    A slot with no reading is NaN. Two readings in one slot: the later is kept and it is counted."""
    idx = np.rint((np.asarray(t_ns, dtype=np.int64) - int(t0_ns)) / (dt_s * 1e9)).astype(np.int64)
    ok = (idx >= 0) & (idx < n)
    out = np.full(n, np.nan)
    out[idx[ok]] = np.asarray(v, dtype=np.float64)[ok]          # numpy assigns in order: the later reading wins
    placed = int(len(np.unique(idx[ok])))
    return out, {"placed": int(ok.sum()), "empty_slots": int(n - placed), "collisions": int(ok.sum() - placed),
                 "outside": int((~ok).sum())}


def island_total(paths, dt_s: float = 0.2):
    """Sum of every device in every file, on one grid from the latest start to the earliest end.
    Returns (total in W, t0 in ns, report). NaN wherever any device has no reading."""
    logs = [read_wattameter(p) for p in paths]
    t0 = max(int(t[0]) for t, _ in logs)
    t1 = min(int(t[-1]) for t, _ in logs)
    n = int(np.floor((t1 - t0) / (dt_s * 1e9) + 0.5)) + 1
    total = np.zeros(n)
    rep = {"devices": 0, "empty_slots": 0, "collisions": 0}
    for t, cols in logs:
        for v in cols.values():
            y, r = to_grid(t, v, t0, dt_s, n)
            total += y
            rep["devices"] += 1; rep["empty_slots"] += r["empty_slots"]; rep["collisions"] += r["collisions"]
    rep["nan_slots"] = int(np.isnan(total).sum())
    return total, t0, rep


def running_mean_gain(f, window_s: float):
    """Amplitude gain of a mean over the last window_s seconds for a sine at f Hz."""
    return np.abs(np.sinc(np.asarray(f, dtype=np.float64) * window_s))   # numpy's sinc is sin(pi x)/(pi x)


def swing(x, lo: float = 5, hi: float = 95) -> dict:
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    a, m, b = np.percentile(v, [lo, 50, hi])
    return {"low": float(a), "mid": float(m), "high": float(b), "swing": float(b - a), "mean": float(v.mean())}


def runs_below(x, level: float):
    """Start index and length of each run of consecutive samples below level (NaN ends a run)."""
    b = np.r_[False, np.asarray(x, dtype=np.float64) < level, False]     # NaN < level is False
    d = np.diff(b.astype(np.int8))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return starts, ends - starts
