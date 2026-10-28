"""Per-phase re-run of the pulse channel.

EXPLORATORY orchestration: the building blocks (phases.load_phases, band_covariance,
imbalance, fit_predictor, step_events) are unit-tested in tests/test_phases.py; this
script only wires them to run_pulse.experiment and writes phase_measurement.json.

Question: a single-phase island (home, small shop) puts all of its swing on one phase.
How much smaller a swing is detected if the monitor looks at (a) that phase alone or
(b) the imbalance residual x_k - mean(others), instead of the three-phase sum?

Design:
* Series in kVA: S_k = V1 * I_k, so sum_k S_k is exactly run_pulse's reference series
  (voltage choice was checked: V1, V_k or constant voltage give the same correlations).
* Same 120 -> 12 Hz block-mean decimation, same 36 h characterisation / test split.
* Paired injections: every series gets a fresh rng with the same seed, so the same
  cadences and swings are injected into each.
* The signal is injected on ONE phase. In the sum it appears with gain 1; in phase k
  with gain 1; in imbalance_k with gain 1 (the others are untouched).
* Unknown phase: the monitor does not know which phase the island is on, so it must
  test all three. "unknown phase" rows use one threshold = the max over the three
  series' null maxima (zero false alarms over all three), and a hit on the right phase.

Usage: python -m island_detectability_map.run_phases ../data/upmu OUT_DIR
(needs grizzly_head6.npz: see fetch_upmu docstring)

Origin: verification@3af18129
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

from island_detectability_map import phases, pulse, run_pulse

SEED = 20261001
DEC = run_pulse.DEC
SWINGS = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
HARMONICS = [float(k) for k in range(1, 7)]       # site's own 1.00 Hz load + harmonics
COV_BANDS = [((0.02, 0.1), 500.0), ((0.1, 1.0), 100.0), ((1.0, 3.0), 100.0)]


def series_set(x3):
    s = {"sum": x3.sum(axis=0)}
    for k in range(3):
        s[f"phase{k+1}"] = x3[k]
        s[f"imbalance{k+1}"] = phases.imbalance(x3, k)
    return s


def run_config(series, fs, name, **kw):
    """Run one detector config on every series (paired rng); add unknown-phase rows."""
    out = {}
    for key, x in series.items():
        e = run_pulse.experiment(x, fs, 3600, np.random.default_rng(SEED), return_null=True, **kw)
        out[key] = e
    for fam in ("phase", "imbalance"):
        keys = [f"{fam}{k}" for k in (1, 2, 3)]
        thr = {"max_null": max(out[k]["max_null"] for k in keys),
               "p99_null": max(out[k]["p99_null"] for k in keys)}
        for k in keys:
            out[f"{k}|unknown"] = run_pulse.experiment(series[k], fs, 3600, np.random.default_rng(SEED),
                                                      thr_override=thr, return_null=True, **kw)
    rows = []
    for key, e in out.items():
        r = {"config": name, "series": key, "max_null": e["max_null"],
             "threshold": e["threshold_used"]["max_null"], "notched_bins": e["notched_bins"],
             "rates": [[q["swing_kva"], q["detect_rate_zero_fa"]] for q in e["rows"]]}
        for lvl in (0.5, 0.8):
            r[f"swing{int(lvl*100)}_kva"] = run_pulse.swing_at(e["rows"], "detect_rate_zero_fa", lvl)
        rows.append(r)
    return rows


def line_power(x, fs, f0, seg_s=100.0, half_width=0.03):
    f, p = pulse.welch_psd(x, fs, seg_s)
    sel = np.abs(f - f0) <= half_width
    return float(np.sum(p[sel]) * (f[1] - f[0]))


def main(data_dir: Path, out_dir: Path):
    x120, fs120 = phases.load_phases(data_dir, voltage="L1", fname="grizzly_head6.npz")
    x3 = np.vstack([run_pulse.block_mean(r, DEC) for r in x120])
    fs = fs120 / DEC
    nc = int(36 * 3600 * fs)
    res = {"source": "LBNL Open uPMU grizzly_bus1_2, first 200 MiB of C1-3MAG, L1-3MAG",
           "series_units": "kVA, S_k = V1*I_k (sum = run_pulse reference series)", "seed": SEED,
           "fs_hz": fs, "hours": x3.shape[1] / fs / 3600, "mean_kva_per_phase": x3.mean(axis=1).tolist()}
    # 1. covariance and noise ratios by band, characterisation vs test half
    cov = []
    for (band, seg) in COV_BANDS:
        for part, sl in (("char", slice(0, nc)), ("test", slice(nc, None))):
            C = phases.band_covariance(x3[:, sl], fs, band, seg)
            sd_sum = float(np.sqrt(np.ones(3) @ C @ np.ones(3)))
            row = {"band_hz": list(band), "part": part, "cov": C.tolist(),
                   "corr": (C / np.sqrt(np.outer(np.diag(C), np.diag(C)))).tolist(), "sd_sum_kva": sd_sum}
            for k in range(3):
                o = [j for j in range(3) if j != k]
                w = -0.5 * np.ones(3); w[k] = 1.0
                cond = C[k, k] - C[k, o] @ np.linalg.solve(C[np.ix_(o, o)], C[o, k])
                row[f"noise_ratio_phase{k+1}"] = sd_sum / float(np.sqrt(C[k, k]))
                row[f"noise_ratio_imbalance{k+1}"] = sd_sum / float(np.sqrt(w @ C @ w))
                row[f"noise_ratio_bandopt{k+1}"] = sd_sum / float(np.sqrt(cond))
            cov.append(row)
    res["band_covariance"] = cov
    # 2. the site's own 1.00 Hz pulsed load: which phases, and does the imbalance cancel it?
    ser = series_set(x3)
    res["site_1hz_line_kva2"] = {part: {k: line_power(v[sl], fs, 1.0) for k, v in ser.items()}
                                 for part, sl in (("char", slice(0, nc)), ("test", slice(nc, None)))}
    # 3. detection experiments
    rows = []
    rows += run_config(ser, fs, "rectangle, as flagged")
    rows += run_config(ser, fs, "rectangle, site load whitelisted", whitelist=run_pulse.RESOLVED_LINES_HZ)
    mk = lambda n, fs_, f0, swing, rng_: pulse.training_trace(n, fs_, f0, swing, rng=rng_, sigma_log=0.1, rho=0.95, straggle=1.0)
    rows += run_config(ser, fs, "sawtooth 10% spread rho 0.95 (map reference), seg 100 s",
                       whitelist=HARMONICS, band=(0.08, 5.5), f0_range=(0.1, 3.0), seg_s=100.0,
                       swings=SWINGS, make_signal=mk)
    for name, (lo, hi), seg in run_pulse.CADENCE_BANDS:
        rows += run_config(ser, fs, f"rectangle, cadence band {name}", whitelist=HARMONICS,
                           band=(lo / 1.25, min(3.0 * hi, 5.5)), f0_range=(lo, hi), seg_s=seg,
                           swings=SWINGS + [64.0, 128.0])
    res["detection"] = rows
    # 4. steps in the summed series (1 s windows): how balanced is the background's step load?
    ev = phases.step_events(x3, fs, win_s=1.0, min_step=5.0)
    shares = np.array([e["balanced_share"] for e in ev]); sizes = np.array([abs(e["step_sum"]) for e in ev])
    res["steps"] = {"min_step_kva": 5.0, "win_s": 1.0, "n_events": len(ev),
                    "balanced_share_quantiles": {q: float(np.quantile(shares, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)} if len(ev) else {},
                    "by_size": [{"min_kva": a, "n": int((sizes >= a).sum()),
                                 "median_share": float(np.median(shares[sizes >= a])) if (sizes >= a).any() else None}
                                for a in (5, 10, 20, 50)],
                    "share_lt_0p5": int((shares < 0.5).sum())}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase_measurement.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    r = main(Path(sys.argv[1]), Path(sys.argv[2]))
    for c in r["band_covariance"]:
        print(c["band_hz"], c["part"], "corr", np.round(np.array(c["corr"])[[0, 0, 1], [1, 2, 2]], 2),
              "| ratios phase/imb/opt:", [round(c[f"noise_ratio_{t}{k}"], 1) for k in (1, 2, 3) for t in ("phase", "imbalance", "bandopt")])
    print("1 Hz line kVA^2:", {p: {k: round(v, 2) for k, v in d.items()} for p, d in r["site_1hz_line_kva2"].items()})
    cur = None
    for row in r["detection"]:
        if row["config"] != cur:
            cur = row["config"]; print("\n##", cur)
        print(f"  {row['series']:<20} thr {row['threshold']:8.1f}  swing50/80 kVA: {row['swing50_kva'] and round(row['swing50_kva'],2)} / {row['swing80_kva'] and round(row['swing80_kva'],2)}")
    print("\nsteps:", r["steps"])
