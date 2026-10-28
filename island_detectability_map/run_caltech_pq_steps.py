"""Balanced, real-power-only steps at the Caltech building mains (Question: a three-phase
island's edge is ~20 kW, on all three phases at once, with no reactive part; a home water heater is
one phase and ~5 kW; how common is a 20 kW three-phase resistive step in a commercial building?).
Writes caltech_pq_steps.json. EXPLORATORY, no tests of its own.

One week of 1-second data per building main. P = the meter's real-power register (kW).
S = sum over phases of current magnitude x voltage magnitude (kVA). Reactive power is not
recorded; it is inferred as Q = sqrt(S^2 - P^2), taken as lagging. (S adds the phases'
magnitudes, so Q is slightly overstated when the phases' power factors differ.)

For every sharp step in P (3-second means either side; run_sync.upsteps/events), inside gap-free
stretches: the change in P, in inferred Q, and in each phase's kVA. A step is
  balanced   if every phase carries between 23% and 43% of the kVA change,
  real-only  if |dQ| <= 0.15 |dP|.
Reported per building and per day: all steps, balanced ones, balanced and real-only ones.

Usage: python -m island_detectability_map.run_caltech_pq_steps WEEK_ROOT NETWORK_JSON OUT_JSON START END

Origin: verification@bdaa3ae8
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

from island_detectability_map import caltech, run_caltech_levels as L, run_sync as S

METERS = [f"egauge_{k}" for k in (3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14)]
SIZES = [10.0, 15.0, 20.0]
W = 3


def main(root, net, out_json, start, end):
    mm = caltech.meter_map(net)
    res = {"time_range_utc": [start, end], "rule": "balanced: each phase 23-43% of the kVA change; real-only: |dQ| <= 0.15 |dP|; Q inferred as sqrt(S^2 - P^2)",
           "meters": {}}
    for name in METERS:
        d = caltech.load_element(Path(root) / name / "magnitudes", mm[name], start, end)
        if d.p_kw is None:
            continue
        p = d.p_kw; s = d.total
        ok = d.mask & np.isfinite(p)
        q = np.sqrt(np.maximum(s ** 2 - p ** 2, 0.0))
        days = ok.sum() / 86400.0
        out = {"mean_kw": float(np.nanmean(p[ok])), "power_factor": float(np.nanmean(p[ok]) / np.nanmean(s[ok])), "days": days, "steps": {}}
        for direction, sign in (("down", -1.0), ("up", 1.0)):
            rows = []
            for a, b in L.runs(ok):
                pp = p[a:b]
                t, st = S.upsteps(sign * pp)
                ev = np.asarray(S.events(t, st, min(SIZES)), dtype=int)
                ev = ev[(ev > W + 1) & (ev < len(pp) - W - 2)]
                mean = lambda x, e: x[a + e + 1: a + e + 1 + W].mean() - x[a + e - W: a + e].mean()
                for e in ev:
                    dp, dq = mean(p, e), mean(q, e)
                    dk = np.array([mean(d.kva[k], e) for k in range(3)])
                    share = dk / dk.sum() if abs(dk.sum()) > 1e-9 else np.array([np.nan] * 3)
                    rows.append((abs(dp), abs(dq) / abs(dp), bool(np.all((share >= 0.23) & (share <= 0.43)))))
            for size in SIZES:
                sel = [r for r in rows if r[0] >= size]
                out["steps"][f"{direction}, >= {size:g} kW"] = {
                    "per_day": len(sel) / days, "balanced_per_day": sum(r[2] for r in sel) / days,
                    "balanced_real_only_per_day": sum(r[2] and r[1] <= 0.15 for r in sel) / days,
                    "count_balanced_real_only": int(sum(r[2] and r[1] <= 0.15 for r in sel)),
                    "median_dq_over_dp_of_balanced": float(np.median([r[1] for r in sel if r[2]])) if any(r[2] for r in sel) else None}
        res["meters"][name] = out
        print(name, f"{out['mean_kw']:.0f} kW pf {out['power_factor']:.2f}", {k: (round(v['per_day'], 1), round(v['balanced_per_day'], 1), round(v['balanced_real_only_per_day'], 2), v['median_dq_over_dp_of_balanced'] and round(v['median_dq_over_dp_of_balanced'], 2)) for k, v in out["steps"].items() if "15" in k or "20" in k}, flush=True)
    Path(out_json).write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main(*sys.argv[1:6])
