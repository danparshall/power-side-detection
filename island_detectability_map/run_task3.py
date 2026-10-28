"""The measured 16-H100 job added to the Caltech
1-second data at three levels, learned in both directions (see run_task2 for why), with the LBNL bus put
through the same 1-second pipeline as a cross-check.

EXPLORATORY orchestration, no tests of its own. Tested pieces: caltech.* (loader),
run_pulse.experiment, phases.imbalance, nlr.*; run_nlr_inject.trace_kw / make_signal_from.

Signal: NLR Llama-2 fine-tune (GPU one-second means), resampled to 1 a second, looped from a
random start, times a scale (1 = as logged, 2.45 kW p5-p95 swing; kW taken as kVA). Three-phase
island -> added to the sum; single-phase island -> added to one phase, looked for in that phase's
residual (phase minus the mean of the other two, so the full signal lands in it).
Detector: spectral, one-hour windows, run_caltech_inject's two bands (slow: 500 s segments,
search 0.016-0.3 Hz; fast: 100 s segments, search 0.08-0.5 Hz). The job's 0.098 Hz is in both.
Folds: Caltech week 2025-03-10..17 (168 h): A = hours 0-83 (Mon 00:00 to Thu 12:00 UTC), B = 84-167.
LBNL (1-s block means of the 120/s data, mag1.npy):
A = hours 36-155, B = 156-276, the first 36 hours always among the learning hours.
Fold A learns (baseline spectrum, site-line list, threshold) on A and is scored on B (causal);
fold B the reverse. Threshold = worst no-signal hour of the learning half (one per family) x margin;
"scored_hours_max" = worst no-signal hour of the scored half (optimistic diagnostic only).

Usage: python -m island_detectability_map.run_task3 CALTECH_ROOT NETWORK_JSON UPMU_DIR NLR_DIR OUT_DIR [STAGE]
STAGE = <site>:<fold>:<band slow|fast>  (all four series of that site; cached)

Origin: verification@3bf8de0f
"""
from __future__ import annotations

import json, pickle, sys
from pathlib import Path

import numpy as np

from island_detectability_map import caltech, phases, run_pulse as R
from island_detectability_map import run_nlr_inject as NI
from island_detectability_map.run_phasor import cached

SEED = 20261005
START, END = "2025-03-10T00:00:00", "2025-03-17T00:00:00"
SITES = {"egauge_12": "Caltech building main, 0.25 MVA", "egauge_17": "Caltech 2.4 kV side, 1.2 MVA",
         "egauge_19": "Caltech 16.5 kV line, 5.5 MVA", "lbnl": "LBNL grizzly_bus1_2, 0.7 MVA (1-s means)"}
BANDS = {"slow": dict(f0_range=(0.02, 0.1), seg_s=500.0, band=(0.016, 0.3), notch_ratio=10.0),
         "fast": dict(f0_range=(0.1, 0.4), seg_s=100.0, band=(0.08, 0.5), notch_ratio=10.0)}
SERIES = ["sum", "resid_a", "resid_b", "resid_c"]
GROUPS = {"sum": ["sum"], "resid": ["resid_a", "resid_b", "resid_c"]}
SCALES = NI.SCALES
MARGINS = [1.0, 2.0, "scored_hours_max"]
WORKLOAD = "llama2_70b_lora"


def load(site, root, net, upmu):
    if site == "lbnl":
        x = np.load(Path(upmu) / "series" / "mag1.npy")
    else:
        x = caltech.load_element(Path(root) / site / "magnitudes", caltech.meter_map(net)[site], START, END).kva
    n_h = x.shape[1] // 3600
    return x[:, :n_h * 3600]


def folds(site, n_h):
    if site == "lbnl":
        a, b = list(range(36, 156)), list(range(156, n_h))
        return {"A": (list(range(36)) + a, b), "B": (list(range(36)) + b, a)}
    h = n_h // 2
    a, b = list(range(h)), list(range(h, n_h))
    return {"A": (a, b), "B": (b, a)}


def series_of(kva):
    s = {"sum": kva.sum(axis=0)}
    for k, p in enumerate("abc"):
        s[f"resid_{p}"] = phases.imbalance(kva, k)
    return s


def stage(site, fold, band, root, net, upmu, nlr_dir, trace=None):
    """trace: mean-removed job trace in kW at 1 sample/s; default is the NLR job read from nlr_dir."""
    kva = load(site, root, net, upmu)
    n_h = kva.shape[1] // 3600
    learn, score = folds(site, n_h)[fold]
    tr = NI.trace_kw(Path(nlr_dir), WORKLOAD, dict(NI.CASES)[WORKLOAD], 1.0) if trace is None else trace
    out = {"n_hours": n_h, "learn": learn, "score": score, "mean_kva": float(np.nanmean(kva.sum(axis=0))),
           "trace_swing_kw": float(np.percentile(tr, 95) - np.percentile(tr, 5)), "series": {}}
    for key, x in series_of(kva).items():
        e = R.experiment(x, 1.0, 3600, np.random.default_rng(SEED), char_hours=0, char_idx=learn, swings=SCALES,
                         make_signal=NI.make_signal_from(tr), return_null=True, return_stats=True, **BANDS[band])
        out["series"][key] = {"null": np.array(e["null_stats"]), "inj": np.array(e["inj_stats"]),
                              "set_aside": e["notched_bins"] / e["band_bins"]}
    return out


def scale_for(rates, level):
    for j in range(len(rates)):
        if rates[j] >= level:
            if j == 0:
                return SCALES[0]
            a = (level - rates[j - 1]) / (rates[j] - rates[j - 1])
            return float(np.exp(np.log(SCALES[j - 1]) + a * np.log(SCALES[j] / SCALES[j - 1])))
    return None


def score_stage(st):
    learn, score = st["learn"], st["score"]
    learn_t = [h for h in learn if h < st["n_hours"]]
    rows = []
    for g, keys in GROUPS.items():
        nul = {k: st["series"][k]["null"] for k in keys}
        ok = lambda k, idx: [i for i in idx if np.isfinite(nul[k][i])]
        for m in MARGINS:
            if m == "scored_hours_max":
                thr = max(np.nanmax(nul[k][score]) for k in keys)
            else:
                thr = max(np.nanmax(nul[k][learn_t]) for k in keys) * m
            fa = [i for i in score if any(np.nan_to_num(nul[k][i], nan=-np.inf) > thr for k in keys)]
            per = []
            for k in keys:
                idx = ok(k, score)
                inj = st["series"][k]["inj"]
                rates = [float(np.mean(inj[j, idx] > thr)) for j in range(len(SCALES))]
                per.append({"series": k, "scored_hours": len(idx), "rate_at_scale_1": float(np.interp(0.0, np.log(SCALES), rates)),
                            "scale_80": scale_for(rates, 0.8)})
            s80 = [p["scale_80"] for p in per if p["scale_80"] is not None]
            sw = st["trace_swing_kw"]
            rows.append({"group": g, "margin": m, "alarm_hours": len(fa), "score_hours": len(score),
                         "rate_at_scale_1": [p["rate_at_scale_1"] for p in per],
                         "swing_kw_80_min_max": [min(s80) * sw, max(s80) * sw] if len(s80) == len(keys) else None,
                         "found_on_n_of_series_at_80": len(s80),
                         "share_of_band_set_aside": [float(st["series"][k]["set_aside"]) for k in keys]})
    return rows


def main(root, net, upmu, nlr_dir, out_dir, only=None):
    cache = Path(upmu) / "task3_cache_v1"
    if only:
        site, fold, band = only.split(":")
        cached(cache, f"{site}__{fold}__{band}", lambda: stage(site, fold, band, root, net, upmu, nlr_dir))
        return None
    res = {"design": __doc__.split("Folds:")[1].split("Usage")[0].strip(), "sites": SITES, "scales": SCALES, "rows": []}
    for f in sorted(cache.glob("*.pkl")):
        site, fold, band = f.stem.split("__")
        st = pickle.loads(f.read_bytes())
        for r in score_stage(st):
            res["rows"].append({"site": site, "level": SITES[site], "mean_kva": st["mean_kva"], "fold": fold, "band": band,
                                "trace_swing_kw": st["trace_swing_kw"], **r})
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "task3_caltech_real_job.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    a = sys.argv
    r = main(a[1], a[2], a[3], a[4], a[5], a[6] if len(a) > 6 else None)
    if r:
        for row in r["rows"]:
            f = lambda v: "-" if v is None else f"{v[0]:.1f}-{v[1]:.1f}"
            print(f"{row['site']:<10}{row['band']:<5} fold {row['fold']} {row['group']:<6} x{str(row['margin'])[:6]:<6} "
                  f"| alarms {row['alarm_hours']:>3}/{row['score_hours']:<3} | rate@1 {[round(v, 2) for v in row['rate_at_scale_1']]} "
                  f"| 80% at {f(row['swing_kw_80_min_max'])} kW | set aside {[round(v, 2) for v in row['share_of_band_set_aside']]}")
