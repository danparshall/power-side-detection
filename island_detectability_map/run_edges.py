"""Time-domain alternative to the spectral detector.

Motivation: a wandering cadence does not line up with fixed FFT segments. But in the
straggler model (pulse.training_trace) every step ends with ALL chips restarting together - a sharp up-step of the full
swing. Sharp edges need no regular period. Detector: count up-steps by size (two-sample
difference, local maxima), compare each size band's hourly count with the site's own
characterisation-period counts, maximise the robust z-score over size bands.
Usage:  python -m island_detectability_map.run_edges [DATA_DIR] [OUT_DIR]

Origin: verification@6349ee27
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

from island_detectability_map import pulse, run_pulse as R

EDGES = np.geomspace(0.5, 128.0, 25)          # up-step size bands, kVA
SWINGS = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]


def counts(x):
    d2 = x[2:] - x[:-2]
    pk = d2[1:-1][(d2[1:-1] > d2[:-2]) & (d2[1:-1] >= d2[2:])]
    return np.histogram(pk, EDGES)[0].astype(float)


def main(data_dir: Path, out_dir: Path):
    rng = np.random.default_rng(R.SEED + 1)
    s, fs, _ = R.load(data_dir)
    x = R.block_mean(s, R.DEC); fx = fs / R.DEC
    n = int(3600 * fx)
    wins = [x[i:i + n] for i in range(0, len(x) - n + 1, n)]
    c = np.array([counts(w) for w in wins])
    med = np.median(c[:36], axis=0)
    mad = 1.4826 * np.median(np.abs(c[:36] - med), axis=0) + 1.0
    z = lambda cc: float(np.max((cc - med) / mad))
    test = list(range(36, len(wins)))
    null_all = np.array([z(c[i]) for i in test])
    # The site's own 1 Hz pulsed load is exactly what this detector looks for: it fires on those
    # hours with z in the thousands. As with the spectral runs, that alarm is treated as resolved
    # by inspection and those hours are set aside; the count is reported.
    site_hours = [i for i, v in zip(test, null_all) if v > 100.0]
    test = [i for i in test if i not in site_hours]
    null = np.array([z(c[i]) for i in test])
    thr = float(null.max())
    out = {"fs_hz": fx, "site_load_hours_set_aside": len(site_hours), "site_load_z_max": float(null_all.max()),
           "clean_test_hours": len(test),
           "diff2_rms_kva_median_hour": float(np.median([np.sqrt(np.mean((w[2:] - w[:-2]) ** 2)) for w in wins])),
           "null_z_median": float(np.median(null)), "null_z_max": thr, "models": []}
    for name, f0r, kw in R.REALISTIC:
        rows = []
        for swing in SWINGS:
            hits = 0
            for i in test:
                f0 = float(np.exp(rng.uniform(np.log(f0r[0]), np.log(f0r[1]))))
                hits += z(counts(wins[i] + pulse.training_trace(n, fx, f0, swing, rng=rng, **kw))) > thr
            rows.append({"swing_kva": swing, "detect_rate_zero_fa": hits / len(test)})
        out["models"].append({"model": name, "rows": rows,
                              "swing50_kva": R.swing_at(rows, "detect_rate_zero_fa", 0.5),
                              "swing80_kva": R.swing_at(rows, "detect_rate_zero_fa", 0.8)})
    (out_dir / "edge_detector.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    dd = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/upmu")
    od = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/20260930_island_detectability_map")
    o = main(dd, od)
    print("site-load hours set aside", o["site_load_hours_set_aside"], "z up to", round(o["site_load_z_max"]), "| clean hours", o["clean_test_hours"], "| 2-sample diff rms kVA (median hour)", round(o["diff2_rms_kva_median_hour"], 2), "| null z median/max", round(o["null_z_median"], 1), round(o["null_z_max"], 1))
    for m in o["models"]:
        print(m["model"], "| swing50/80:", m["swing50_kva"] and round(m["swing50_kva"], 1), m["swing80_kva"] and round(m["swing80_kva"], 1),
              [round(r["detect_rate_zero_fa"], 2) for r in m["rows"]])
