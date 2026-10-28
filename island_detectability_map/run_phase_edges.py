"""Per-phase re-run of the two time-domain detectors.

EXPLORATORY orchestration, same conventions as run_phases.py (paired rng per series,
island on one phase, unknown-phase shared threshold). The detector logic is copied
from run_edges.py and run_sync.py unchanged except for one parameter: the edge-count
size bins start at 0.125 kVA instead of 0.5 kVA, because the imbalance residual's
per-edge noise is several times smaller than the sum's (the sum is re-run with the same
bins as a check against the 11.8 kVA reference).

1. Edge count (run_edges): hourly histogram of up-step sizes vs the first 36 h, robust z,
   max over size bands; hours where the site's own 1 Hz load trips the series (z > 100)
   are set aside per series, as in run_edges.
2. Restart edges (run_sync): background rate of sharp up-steps per day by size on 1-s data,
   and recovery of injected blocking-sync restart edges (500 s compute, 176 s pause).
Usage: python -m island_detectability_map.run_phase_edges ../data/upmu OUT_DIR

Origin: verification@ba559110
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

from island_detectability_map import phases, pulse, run_pulse as R, run_sync as S
from island_detectability_map.run_phases import series_set

SEED = 20261001
EDGES = np.geomspace(0.125, 128.0, 31)
SWINGS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
STEP_SIZES = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
DEPTHS = [1.0, 2.0, 5.0, 10.0, 20.0]
REF_MODEL = R.REALISTIC[1]          # sawtooth, 10% spread, rho 0.95 (the map reference)


def counts(x):
    d2 = x[2:] - x[:-2]
    pk = d2[1:-1][(d2[1:-1] > d2[:-2]) & (d2[1:-1] >= d2[2:])]
    return np.histogram(pk, EDGES)[0].astype(float)


def edge_null(x, fx):
    n = int(3600 * fx)
    wins = [x[i:i + n] for i in range(0, len(x) - n + 1, n)]
    c = np.array([counts(w) for w in wins])
    med = np.median(c[:36], axis=0)
    mad = 1.4826 * np.median(np.abs(c[:36] - med), axis=0) + 1.0
    z = lambda cc: float(np.max((cc - med) / mad))
    test = list(range(36, len(wins)))
    null_all = {i: z(c[i]) for i in test}
    clean = [i for i in test if null_all[i] <= 100.0]
    return wins, z, clean, {"site_hours_set_aside": len(test) - len(clean),
                            "null_z_max": max(null_all[i] for i in clean),
                            "diff2_rms_kva_median_hour": float(np.median([np.sqrt(np.mean((w[2:] - w[:-2]) ** 2)) for w in wins]))}


def edge_rates(wins, z, clean, fx, thr, model=REF_MODEL):
    name, f0r, kw = model
    rng = np.random.default_rng(SEED)
    n = len(wins[0])
    rows = []
    for swing in SWINGS:
        hits = 0
        for i in clean:
            f0 = float(np.exp(rng.uniform(np.log(f0r[0]), np.log(f0r[1]))))
            hits += z(counts(wins[i] + pulse.training_trace(n, fx, f0, swing, rng=rng, **kw))) > thr
        rows.append({"swing_kva": swing, "detect_rate_zero_fa": hits / len(clean)})
    return rows


def restart(x1, days):
    t, st = S.upsteps(x1)
    out = {"upsteps_per_day": {str(a): len(S.events(t, st, a)) / days for a in STEP_SIZES}, "recovery": {}}
    for depth in DEPTHS:
        rng = np.random.default_rng(SEED)
        y = x1.copy(); restarts = []; pos = float(rng.uniform(0, 600))
        while pos + S.PAUSE_S + 10 < len(y):
            a, b = int(pos), int(pos + S.PAUSE_S)
            y[a:b] -= depth; restarts.append(b)
            pos = b + 500.0 * (1 + 0.1 * rng.standard_normal())
        t2, s2 = S.upsteps(y)
        ev = S.events(t2, s2, 0.75 * depth)
        out["recovery"][str(depth)] = {"recovered": sum(np.any(np.abs(ev - r) <= 2) for r in restarts) / len(restarts),
                                       "background_per_day_at_0.75_depth": len(S.events(t, st, 0.75 * depth)) / days}
    return out


def main(data_dir: Path, out_dir: Path):
    x120, fs = phases.load_phases(data_dir, voltage="L1", fname="grizzly_head6.npz")
    x12 = np.vstack([R.block_mean(r, R.DEC) for r in x120]); fx = fs / R.DEC
    x1 = np.vstack([R.block_mean(r, int(fs)) for r in x120]); days = x1.shape[1] / 86400.0
    ser12, ser1 = series_set(x12), series_set(x1)
    res = {"edges_kva": [float(EDGES[0]), float(EDGES[-1]), len(EDGES)], "seed": SEED,
           "model": REF_MODEL[0], "edge_count": {}, "restart": {}}
    prep = {k: edge_null(v, fx) for k, v in ser12.items()}
    for k, (wins, z, clean, info) in prep.items():
        rows = edge_rates(wins, z, clean, fx, info["null_z_max"])
        res["edge_count"][k] = {**info, "clean_hours": len(clean), "rows": rows,
                                "swing50_kva": R.swing_at(rows, "detect_rate_zero_fa", 0.5),
                                "swing80_kva": R.swing_at(rows, "detect_rate_zero_fa", 0.8)}
    for fam in ("phase", "imbalance"):
        keys = [f"{fam}{k}" for k in (1, 2, 3)]
        thr = max(prep[k][3]["null_z_max"] for k in keys)
        for k in keys:
            wins, z, clean, _ = prep[k]
            rows = edge_rates(wins, z, clean, fx, thr)
            res["edge_count"][f"{k}|unknown"] = {"threshold": thr, "rows": rows,
                                                 "swing50_kva": R.swing_at(rows, "detect_rate_zero_fa", 0.5),
                                                 "swing80_kva": R.swing_at(rows, "detect_rate_zero_fa", 0.8)}
    for k, v in ser1.items():
        res["restart"][k] = restart(v, days)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase_edges.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    r = main(Path(sys.argv[1]), Path(sys.argv[2]))
    print("## edge count,", r["model"])
    for k, e in r["edge_count"].items():
        print(f"  {k:<20} {'set aside %d' % e['site_hours_set_aside'] if 'site_hours_set_aside' in e else '':<14}"
              f" diff2 {e.get('diff2_rms_kva_median_hour', float('nan')):5.2f}  swing50/80: "
              f"{e['swing50_kva'] and round(e['swing50_kva'], 2)} / {e['swing80_kva'] and round(e['swing80_kva'], 2)}")
    print("## restart edges: up-steps/day by size | recovery (rate, background/day at 0.75 depth) by depth")
    for k, e in r["restart"].items():
        print(f"  {k:<12}", {a: round(v) for a, v in e["upsteps_per_day"].items()}, "|",
              {d: (round(v["recovered"], 2), round(v["background_per_day_at_0.75_depth"])) for d, v in e["recovery"].items()})
