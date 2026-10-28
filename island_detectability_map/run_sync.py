"""Blocking-sync pause: how often, and how obvious across a fleet?
Measures on the LBNL bus (1-s data): (1) the background rate of sharp up-steps by size, and
(2) how reliably an injected island restart edge is recovered. Then does the fleet
coincidence arithmetic. Usage: python -m island_detectability_map.run_sync [DATA_DIR] [OUT_DIR]

Origin: verification@2c06bb3e
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

from island_detectability_map import run_pulse as R

SIZES = [5.0, 10.0, 15.0, 20.0]
PAUSE_S = 176.0                         # corpus: pause ~ constant; = (1-u)/u * H * t_step at 100 Mbps
UTIL_100MBPS = {100: 0.36, 500: 0.74, 1000: 0.85}   # Koneputugodage & Long 2026 (as summarised in corpus)
T_STEP_S = 1.0                          # ASSUMED seconds per inner step
FEEDERS = 300_000                       # ASSUMED number of monitored feeders
ISLANDS = [34, 101, 625, 4706]          # Rahman 2026 node counts
WINDOW_S = 4.0                          # +-2 s coincidence window


def upsteps(x, k=3):
    """step[t] = mean(x[t+1 : t+1+k]) - mean(x[t-k : t]) on 1-s data (skips the transition second)."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    t = np.arange(k, len(x) - k - 1)
    return t, (c[t + 1 + k] - c[t + 1]) / k - (c[t] - c[t - k]) / k


def events(t, s, thr, sep=5):
    idx = np.where(s >= thr)[0]
    out, last = [], -10**9
    for i in idx:                       # one event per `sep` seconds, keep the first crossing
        if t[i] - last >= sep:
            out.append(t[i]); last = t[i]
    return np.array(out)


def main(data_dir: Path, out_dir: Path):
    rng = np.random.default_rng(R.SEED + 2)
    s, fs, _ = R.load(data_dir)
    x = R.block_mean(s, int(fs)); days = len(x) / 86400.0
    t, st = upsteps(x)
    res = {"days": days, "upsteps_per_day": {str(a): len(events(t, st, a)) / days for a in SIZES}}
    # inject a blocking-sync schedule for one island: compute 500 s, pause 176 s, +-10% jitter on compute
    eff = {}
    for depth in SIZES:
        y = x.copy(); restarts = []; pos = float(rng.uniform(0, 600))
        while pos + PAUSE_S + 10 < len(y):
            a, b = int(pos), int(pos + PAUSE_S)
            y[a:b] -= depth; restarts.append(b)
            pos = b + 500.0 * (1 + 0.1 * rng.standard_normal())
        t2, s2 = upsteps(y)
        ev = events(t2, s2, 0.75 * depth)
        hit = sum(np.any(np.abs(ev - r) <= 2) for r in restarts)
        eff[str(depth)] = {"restarts": len(restarts), "recovered": hit / len(restarts),
                           "background_per_day_at_0.75_depth": len(events(t, st, 0.75 * depth)) / days}
    res["restart_edge_recovery"] = eff
    res["syncs_per_day"] = {str(h): 86400.0 * u / (h * T_STEP_S) for h, u in UTIL_100MBPS.items()}
    rate = eff["20.0"]["background_per_day_at_0.75_depth"]
    p = rate * WINDOW_S / 86400.0
    res["fleet"] = {"p_chance_per_sync": p, "background_feeders_per_sync": FEEDERS * p,
                    "sigma": float(np.sqrt(FEEDERS * p)),
                    "islands": {str(n): {"z_single_sync": n * eff["20.0"]["recovered"] / np.sqrt(FEEDERS * p)} for n in ISLANDS}}
    (out_dir / "sync_pause.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    dd = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/upmu")
    od = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/20260930_island_detectability_map")
    r = main(dd, od)
    print("up-steps/day by size:", {k: round(v) for k, v in r["upsteps_per_day"].items()})
    print("restart-edge recovery:", {k: (round(v["recovered"], 2), round(v["background_per_day_at_0.75_depth"])) for k, v in r["restart_edge_recovery"].items()})
    print("syncs/day at 1 s/step:", {k: round(v) for k, v in r["syncs_per_day"].items()})
    f = r["fleet"]; print("chance p per sync %.4f%% | background feeders per sync %.0f +- %.0f | z by islands:" % (100 * f["p_chance_per_sync"], f["background_feeders_per_sync"], f["sigma"]),
                          {k: round(v["z_single_sync"], 1) for k, v in f["islands"].items()})
