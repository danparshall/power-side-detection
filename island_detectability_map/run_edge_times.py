"""How fast are natural load-step edges, in real and in reactive power? (Question: the
island's drop arrives over about 16-50 ms; does anything else look like that, and could a faked
reactive step be told apart by its timing?) Writes edge_times.json. EXPLORATORY, no tests.

LBNL bus, phase 1, 120 samples a second. Events: sharp steps in 1-second real power (as
run_pq_steps). For each, on the 120/s series within +-2 s: level before (median of 1.5..0.5 s
before the second boundary) and after (median of 1.0..2.0 s after); edge time = the smallest gap
across which 90% of the change happens, between 2-sample means either side (edge_gap; reads about one sample short). Same for
reactive power where it changes by at least 2 kvar. Lag = Q edge centre minus P edge centre.

Usage: python -m island_detectability_map.run_edge_times MAG_NPZ ANG_NPZ OUT_JSON

Origin: verification@15bf610f
"""
from __future__ import annotations

import json, sys

import numpy as np

from island_detectability_map import run_sync as S

FS = 120


def edge_gap(x, pre, post, lo, hi, k=2, share=0.9, max_gap=120):
    """Edge time as a gap: the smallest number of samples g such that somewhere in [lo, hi) the
    mean of the k samples after a gap of g differs from the mean of the k samples before it by at
    least `share` of the whole change. Returns (g, position of the gap start) or (None, None).
    Calibrated on synthetic 6.7 kW ramps with the measured sample noise (0.6 kW): the estimate
    reads about one sample short (true ramp of 0/1/2/3/4/6/12 samples -> median 0/1/1/2/3/4/9), so
    an estimate of 0-1 means a true edge of at most two samples (17 ms) and 2-4 means about three
    to six samples (25-50 ms). With k=6 and share=0.8 it could not tell 0 from 4 samples.
    (A first version timed 10% and 90% threshold crossings on raw samples; noise crossed the
    thresholds anywhere in the search window, so it measured the window, not the edge.)"""
    c = np.concatenate([[0.0], np.cumsum(x[lo - k: hi + k + max_gap + 1])])
    full = post - pre
    idx = np.arange(k, hi - lo + k)                      # gap start, relative to lo - k
    before = (c[idx] - c[idx - k]) / k
    for g in range(0, max_gap + 1):
        after = (c[idx + g + k] - c[idx + g]) / k
        ch = (after - before) / full
        j = int(np.argmax(ch))
        if ch[j] >= share:
            return g, lo + int(idx[j]) - k
    return None, None


def main(mag_npz, ang_npz, out_json):
    m, a = np.load(mag_npz), np.load(ang_npz)
    n = (min(len(a["L1ANG"]), len(a["C1ANG"]), len(m["C1MAG"]), len(m["L1MAG"])) // FS) * FS
    d = np.deg2rad(a["L1ANG"][:n].astype(np.float64) - a["C1ANG"][:n].astype(np.float64))
    s = m["L1MAG"][:n].astype(np.float64) * m["C1MAG"][:n].astype(np.float64) / 1e3
    p, q = s * np.cos(d), s * np.sin(d)
    p1 = p.reshape(-1, FS).mean(axis=1)
    res = {"source": "LBNL grizzly_bus1_2, phase 1, 120 samples/s", "sample_ms": 1000.0 / FS, "classes": {}}
    for direction, series in (("down", -p1), ("up", p1)):
        t, st = S.upsteps(series)
        for size in (5.0, 10.0):
            ev = np.asarray(S.events(t, st, size), dtype=int)
            rows = []
            for e in ev:
                c = (e + 1) * FS                                  # the second boundary, in samples
                if c - 2 * FS < 0 or c + 3 * FS > n:
                    continue
                pre = np.median(p[c - int(1.5 * FS): c - int(0.5 * FS)]); post = np.median(p[c + FS: c + 2 * FS])
                if abs(post - pre) < 0.8 * size:
                    continue
                lo, hi = c - FS, c + FS
                g, pos = edge_gap(p, pre, post, lo, hi)
                if g is None:
                    continue
                qpre = np.median(q[c - int(1.5 * FS): c - int(0.5 * FS)]); qpost = np.median(q[c + FS: c + 2 * FS])
                row = {"p_edge_samples": g, "dp": post - pre, "dq": qpost - qpre}
                if abs(qpost - qpre) >= 2.0:
                    gq, posq = edge_gap(q, qpre, qpost, lo, hi)
                    if gq is not None:
                        row["q_edge_samples"] = gq; row["lag_samples"] = (posq + gq / 2.0) - (pos + g / 2.0)
                rows.append(row)
            pe = np.array([r["p_edge_samples"] for r in rows])
            qe = np.array([r["q_edge_samples"] for r in rows if "q_edge_samples" in r])
            lag = np.array([r["lag_samples"] for r in rows if "lag_samples" in r])
            ms = 1000.0 / FS
            share = lambda x, a, b: float(np.mean((x >= a) & (x <= b))) if len(x) else None
            res["classes"][f"{direction}, >= {size:g} kW"] = {
                "events": len(rows),
                "p_edge_ms_quartiles": [float(v * ms) for v in np.percentile(pe, [25, 50, 75])] if len(pe) else None,
                "p_edge_share_within_17ms": share(pe, 0, 1), "p_edge_share_25_to_50ms": share(pe, 2, 4), "p_edge_share_60_to_250ms": share(pe, 5, 28), "p_edge_share_over_250ms": share(pe, 29, 10 ** 6),
                "per_day": float(len(pe) / (n / FS / 86400.0)),
                "per_day_25_to_50ms": float(np.sum((pe >= 2) & (pe <= 4)) / (n / FS / 86400.0)) if len(pe) else 0.0,
                "per_day_25_to_50ms_with_almost_no_reactive_part": float(sum(1 for r in rows if 2 <= r["p_edge_samples"] <= 4 and abs(r["dq"]) <= 0.1 * abs(r["dp"])) / (n / FS / 86400.0)),
                "per_day_with_almost_no_reactive_part": float(sum(1 for r in rows if abs(r["dq"]) <= 0.1 * abs(r["dp"])) / (n / FS / 86400.0)),
                "q_edge_ms_quartiles": [float(v * ms) for v in np.percentile(qe, [25, 50, 75])] if len(qe) else None,
                "lag_ms_quartiles": [float(v * ms) for v in np.percentile(lag, [25, 50, 75])] if len(lag) else None,
                "share_with_lag_within_two_samples": share(np.abs(lag), 0, 2)}
    open(out_json, "w").write(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    r = main(*sys.argv[1:4])
    for k, v in r["classes"].items():
        print(k, {kk: (round(x, 2) if isinstance(x, float) else ([round(y, 1) for y in x] if isinstance(x, list) else x)) for kk, x in v.items()})
