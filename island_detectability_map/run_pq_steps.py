"""Real and reactive parts of natural load steps on one phase of the LBNL bus (Question:
a motor draws reactive power, the island's supplies almost none - how distinctive
is that, and how hard to fake?). Writes pq_steps.json. EXPLORATORY, no tests of its own; step
finder is run_sync.upsteps/events, applied to real power.

Phase 1 only (the angle streams fetched are L1ANG and C1ANG). Per 1-second mean:
P = V*I*cos(angle_V - angle_I), Q = V*I*sin(angle_V - angle_I), in kW and kvar (positive Q =
lagging, i.e. motor-like). For every sharp step in P (3-second means either side), the change in
Q over the same windows. Reported: steps per day by size, and how many have almost no reactive
part (|dQ| <= 0.1 |dP|), as a corrected-input supply would.

Usage: python -m island_detectability_map.run_pq_steps MAG_NPZ ANG_NPZ OUT_JSON

Origin: verification@b1f4541b
"""
from __future__ import annotations

import json, sys

import numpy as np

from island_detectability_map import run_sync as S

FS = 120
SIZES = [2.0, 5.0, 10.0, 15.0]


def main(mag_npz, ang_npz, out_json):
    m, a = np.load(mag_npz), np.load(ang_npz)
    assert int(m["t0_ns"]) == int(a["t0_ns"])
    n = (min(len(a["L1ANG"]), len(a["C1ANG"]), len(m["C1MAG"]), len(m["L1MAG"])) // FS) * FS
    d = np.deg2rad(a["L1ANG"][:n].astype(np.float64) - a["C1ANG"][:n].astype(np.float64))
    s = m["L1MAG"][:n].astype(np.float64) * m["C1MAG"][:n].astype(np.float64) / 1e3
    blk = lambda x: x.reshape(-1, FS).mean(axis=1)
    p, q = blk(s * np.cos(d)), blk(s * np.sin(d))
    days = len(p) / 86400.0
    res = {"source": "LBNL grizzly_bus1_2, phase 1, 1-s means of 120/s phasors", "days": days,
           "mean_p_kw": float(p.mean()), "mean_q_kvar": float(q.mean()), "power_factor": float(p.mean() / np.hypot(p.mean(), q.mean())),
           "steps": {}}
    w = 3
    cq = np.concatenate([[0.0], np.cumsum(q)])
    for direction, series in (("up", p), ("down", -p)):
        t, st = S.upsteps(series)
        for size in SIZES:
            ev = np.asarray(S.events(t, st, size), dtype=int)
            ev = ev[(ev > w + 1) & (ev < len(p) - w - 2)]
            cp = np.concatenate([[0.0], np.cumsum(p)])
            dp = (cp[ev + w + 1] - cp[ev + 1]) / w - (cp[ev] - cp[ev - w]) / w
            dq = (cq[ev + w + 1] - cq[ev + 1]) / w - (cq[ev] - cq[ev - w]) / w
            ratio = dq / dp
            res["steps"][f"{direction}, >= {size:g} kW"] = {
                "per_day": float(len(ev) / days),
                "ratio_dq_over_dp_quartiles": [float(v) for v in np.percentile(ratio, [25, 50, 75])] if len(ev) else None,
                "per_day_with_almost_no_reactive_part": float(np.sum(np.abs(ratio) <= 0.1) / days) if len(ev) else 0.0,
                "per_day_motor_like_0.3_to_1.5": float(np.sum((ratio >= 0.3) & (ratio <= 1.5)) / days) if len(ev) else 0.0}
    open(out_json, "w").write(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    r = main(*sys.argv[1:4])
    print(f"days {r['days']:.2f} | phase 1 mean P {r['mean_p_kw']:.0f} kW, Q {r['mean_q_kvar']:.0f} kvar, power factor {r['power_factor']:.3f}")
    for k, v in r["steps"].items():
        print(f"  {k:<16} {v['per_day']:8.1f}/day | dQ/dP quartiles {v['ratio_dq_over_dp_quartiles'] and [round(x, 2) for x in v['ratio_dq_over_dp_quartiles']]} | no reactive part: {v['per_day_with_almost_no_reactive_part']:.1f}/day | motor-like: {v['per_day_motor_like_0.3_to_1.5']:.1f}/day")
