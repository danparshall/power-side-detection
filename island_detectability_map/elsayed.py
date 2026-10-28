"""Reader for the Elsayed, Al-Obaidi & Farag node traces (Sci Data 2026,
doi:10.1038/s41597-026-07496-6; figshare 10.6084/m9.figshare.31654879). Tests in tests/test_elsayed.py.

What the files are (read from the paper, the README in the archive, and the data, 2026-10-10):
* One CSV per session: 8 x H100 SXM or 8 x B200 Lambda Cloud node, 15 minutes, one row every
  ~20 ms (median 20.0 ms, 1st-99th percentile 13-26 ms, no gaps over 0.1 s).
* Per-GPU power from NVIDIA's management library (pynvml), board power in W, in gpuN_power_W.
  gpuN_Power_TDP is a percentage of TDP, not a power; cpu_power_W is empty.
* Software readings, not AC-side: no host, fans, NVSwitch, network cards or supply losses.
* The 20 ms readings do not resolve 20 ms edges: the largest single-sample change is about a
  tenth of the swing and the big transitions take 1.7-2.4 s from 80% to 20% of the swing,
  as expected if the library's default averaged power (about one second) is what was read.

Origin: verification@b28fe07c
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

_POWER = re.compile(r"^(gpu\d+)_power_W$")


def read_node(path) -> tuple[np.ndarray, dict]:
    """(time-stamps in ns since the epoch, {device: power in W})."""
    df = pd.read_csv(path)
    t = pd.to_datetime(df["timestamp"]).to_numpy().astype("datetime64[ns]").astype(np.int64)
    cols = {m.group(1): df[c].to_numpy(dtype=np.float64) for c in df.columns if (m := _POWER.match(c))}
    return t, cols


def node_total(path, dt_s: float = 1.0):
    """Sum of every GPU's power, averaged within fixed slots of dt_s from the first time-stamp.
    Returns (total in W, t0 in ns, report). NaN where a slot has no reading."""
    t, cols = read_node(path)
    p = np.sum(list(cols.values()), axis=0)
    idx = np.floor((t - t[0]) / (dt_s * 1e9)).astype(np.int64)
    n = int(idx[-1]) + 1
    count = np.bincount(idx, minlength=n)
    total = np.bincount(idx, weights=p, minlength=n) / np.where(count > 0, count, 1)
    total[count == 0] = np.nan
    return total, int(t[0]), {"devices": len(cols), "rows": len(t), "empty_slots": int((count == 0).sum())}
