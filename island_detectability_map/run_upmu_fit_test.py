"""Is the LBNL uPMU's magnitude a fit at a fixed 60 Hz?
Writes upmu_fit_test.json. EXPLORATORY, no tests;
the script checks itself by planting the effect and recovering it.

A one-cycle fit forced to exactly 60 Hz, when the grid runs at 60 + d Hz, leaves a ripple in the
magnitude of relative size d/120 that follows twice the reported phase angle. At 120 reports a
second that ripple is slow (twice the frequency deviation, 0 to ~0.1 Hz). The test regresses the
magnitude (5-minute running mean removed, 10 samples/s) on d*cos(2*angle), d*sin(2*angle), with
d taken from the voltage angle's rate of change. Reference lines: the same fit with the angle
series reversed in time (noise floor), and the same fit after planting the forced-fit ripple.

Usage: python -m island_detectability_map.run_upmu_fit_test MAG_NPZ ANG_NPZ OUT_JSON [HOURS]

Origin: verification@b2e879e6
"""
from __future__ import annotations

import json, sys

import numpy as np

FS = 120


def fit(x, th, df, plant=False):
    q = 12; n = len(x) // q
    if plant:
        x = x * (1 + (df / 120.0) * np.cos(2 * th + 0.7))
    blk = lambda v: v[: n * q].reshape(n, q).mean(1)
    xs, c, s, d = blk(x), blk(np.cos(2 * th)), blk(np.sin(2 * th)), blk(df)
    k = 3000
    cs = np.cumsum(np.r_[0, xs]); sm = (cs[k:] - cs[:-k]) / k; sl = slice(k // 2, k // 2 + len(sm))
    xr = xs[sl] / sm - 1
    A = np.vstack([d[sl] * c[sl], d[sl] * s[sl]]).T
    amp = float(np.hypot(*np.linalg.lstsq(A, xr, rcond=None)[0]))
    floor = float(np.hypot(*np.linalg.lstsq(A[::-1], xr, rcond=None)[0]))
    return amp, floor, float(xr.std())


def main(mag_npz, ang_npz, out_json, hours=12):
    m, a = np.load(mag_npz), np.load(ang_npz)
    assert int(m["t0_ns"]) == int(a["t0_ns"])
    n = FS * 3600 * int(hours)
    thv = np.deg2rad(a["L1ANG"][:n].astype(np.float64))
    df = np.gradient(np.unwrap(thv)) * FS / (2 * np.pi)
    res = {"hours": int(hours), "frequency_deviation_mhz": {"median": float(np.median(df) * 1e3), "sd": float(df.std() * 1e3),
                                                             "p01_p99": [float(v * 1e3) for v in np.percentile(df, [1, 99])]},
           "forced_one_cycle_fit_predicts_per_hz": 1 / 120.0, "channels": {}}
    for mag, ang in (("L1MAG", "L1ANG"), ("C1MAG", "C1ANG")):
        th = np.deg2rad(a[ang][:n].astype(np.float64)); x = m[mag][:n].astype(np.float64)
        amp, floor, sd = fit(x, th, df); planted, _, _ = fit(x, th, df, plant=True)
        res["channels"][mag] = {"fitted_ripple_per_hz": amp, "noise_floor_per_hz": floor, "recovered_when_planted_per_hz": planted,
                                "slow_relative_fluctuation": sd}
    open(out_json, "w").write(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1], sys.argv[2], sys.argv[3], *(sys.argv[4:5])), indent=1))
