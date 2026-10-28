"""Sub-second notches at the LBNL bus: how many does the background make by itself?
(Extends run_dips' notch table from 5 s down to one 120/s sample.) Writes fast_dips.json.
EXPLORATORY, no tests of its own; same notch definition as run_dips.py.

Why: the evader's cheapest counter to a stall is to keep the chips busy ("burn through"). That leaves a gap between the crash and the filler load taking over. A gap of
d samples is a full-depth notch of d samples. Question: how short a notch of 15-20 kVA is
still rare in the background?

Notch of d samples at t: depth = mean of the w samples before and the w after, minus the mean
of the d samples between; w = max(d, 12) (0.1 s). Counted if pre and post agree within half the
depth (it recovers), at local maxima at least d + w apart. Series: three-phase sum, each phase,
each imbalance residual, kVA = current magnitude x L1 voltage magnitude (as run_phases).

Usage: python -m island_detectability_map.run_fast_dips ../data/upmu OUT_JSON

Origin: verification@60c6bbf5
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

FS = 120
DURATIONS = [1, 2, 3, 6, 12, 30, 60, 120]          # samples: 8 ms ... 1 s
DEPTHS = [10.0, 15.0, 20.0]
CHUNK = FS * 3600


def count(x, d, w):
    c = np.concatenate([[0.0], np.cumsum(x)])
    t = np.arange(w, len(x) - d - w)
    pre = (c[t] - c[t - w]) / w
    dur = (c[t + d] - c[t]) / d
    post = (c[t + d + w] - c[t + d]) / w
    depth = 0.5 * (pre + post) - dur
    depth = np.where(np.abs(pre - post) <= 0.5 * np.maximum(depth, 1e-9), depth, 0.0)
    pk, prop = find_peaks(depth, height=min(DEPTHS), distance=d + w)
    h = prop["peak_heights"]
    return [int((h >= a).sum()) for a in DEPTHS]


def main(data_dir, out_json):
    z = np.load(Path(data_dir) / "grizzly_head6.npz")
    n = min(len(z[k]) for k in ("C1MAG", "C2MAG", "C3MAG", "L1MAG"))
    v = z["L1MAG"][:n].astype(np.float64)
    kva = [z[f"C{k}MAG"][:n].astype(np.float64) * v / 1e3 for k in (1, 2, 3)]
    del v
    names = ["sum", "phase_a", "phase_b", "phase_c", "resid_a", "resid_b", "resid_c"]
    tot = {nm: {d: np.zeros(len(DEPTHS), int) for d in DURATIONS} for nm in names}
    rms = {nm: [] for nm in names}
    for a in range(0, n - CHUNK + 1, CHUNK):
        p = [k[a:a + CHUNK] for k in kva]
        ser = {"sum": p[0] + p[1] + p[2], "phase_a": p[0], "phase_b": p[1], "phase_c": p[2],
               "resid_a": p[0] - 0.5 * (p[1] + p[2]), "resid_b": p[1] - 0.5 * (p[0] + p[2]), "resid_c": p[2] - 0.5 * (p[0] + p[1])}
        for nm, x in ser.items():
            rms[nm].append(float(np.sqrt(np.mean(np.diff(x) ** 2))))
            for d in DURATIONS:
                tot[nm][d] += count(x, d, max(d, 12))
    days = (n // CHUNK) * CHUNK / FS / 86400.0
    res = {"source": "LBNL grizzly_bus1_2, 120 samples/s", "days": days, "depths_kva": DEPTHS,
           "rms_sample_to_sample_step_kva": {nm: float(np.median(v)) for nm, v in rms.items()},
           "notches_per_day": {nm: {f"{d / FS * 1000:.0f} ms": [float(c / days) for c in tot[nm][d]] for d in DURATIONS} for nm in names}}
    Path(out_json).write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    r = main(sys.argv[1], sys.argv[2])
    print("days", round(r["days"], 2), "| rms sample-to-sample step kVA:", {k: round(v, 2) for k, v in r["rms_sample_to_sample_step_kva"].items()})
    for nm, tab in r["notches_per_day"].items():
        print(f"{nm:<8}", {k: [round(x, 1) for x in v] for k, v in tab.items()})
