"""Add a measured 16-H100 power trace (NLR logs) to the LBNL background and ask the spectral
detector for it, with everything the detector learns taken from hours it is not scored on.
Writes nlr_injection.json.

EXPLORATORY orchestration. Tested pieces: nlr.*, run_pulse.experiment, holdout.*. No tests of its own.

* Signal: the 16-GPU total of one NLR run (software readings, one-second running mean, see
  nlr.py), busy stretch only, mean removed, resampled from 0.2 s to the series' 12 a second by
  linear interpolation, looped to fill each hour from a random starting point. Multiplied by a
  scale factor; scale 1 is the job as logged (GPU power only; kW taken as kVA).
* Background: the LBNL bus, 12 a second, block-mean ("mag") or low-passed ("maglp") series.
  Three-phase island -> added to the three-phase sum. Single-phase island -> added to one
  phase, looked for in that phase's imbalance residual.
* Detector: run_pulse.experiment as in run_heldout.spectral_fold. Baseline spectrum and the
  list of the site's own lines come from the 36 characterisation hours plus half B (hours
  156-276); the threshold is the worst no-signal hour of half B times a margin; detection is
  counted on half A (hours 36-155), which the detector has not seen.

Usage: python -m island_detectability_map.run_nlr_inject UPMU_DIR NLR_DATA_DIR OUT_DIR [STAGE]
STAGE = <family>:<series>:<workload>:<slurm id>  runs and caches one case.

Origin: verification@fa3c1e3c
"""
from __future__ import annotations

import glob, json, pickle, sys
from pathlib import Path

import numpy as np

from island_detectability_map import holdout as HO, nlr, run_hours as H, run_nlr as RN, run_pulse as R
from island_detectability_map import run_heldout as RH, run_phases as P
from island_detectability_map.run_phasor import cached

SCALES = [float(v) for v in np.geomspace(0.125, 16.0, 29)]       # steps of 2^(1/4)
MARGINS = [1.0, 2.0]
CASES = [("llama2_70b_lora", "10742829"), ("stable_diffusion", "10742982")]
SERIES = ["sum", "imbalance1", "imbalance2", "imbalance3"]


def trace_kw(nlr_dir: Path, workload: str, sid: str, fx: float, remove_mean: bool = True):
    """GPU total in kW at fx samples a second; mean removed (the swing alone) unless remove_mean=False
    (the whole job, its mean draw included)."""
    folder = RN.RUNS[workload][0]
    g = sorted(glob.glob(str(nlr_dir / "00_raw_datasets" / folder / "4node" / f"nvml*{sid}*")))
    G, _, _ = nlr.island_total(g, RN.DT)
    a, b = RN.busy_span(G)
    x = G[a:b] / 1e3
    t = np.arange(len(x)) * RN.DT
    ok = np.isfinite(x)
    tt = np.arange(0, t[-1], 1.0 / fx)
    y = np.interp(tt, t[ok], x[ok])                         # the few empty slots are bridged in the SIGNAL only
    return y - y.mean() if remove_mean else y


def make_signal_from(trace):
    def mk(n, fs_, f0, scale, rng):
        start = int(rng.integers(0, len(trace)))
        idx = (start + np.arange(n)) % len(trace)
        return scale * trace[idx]
    return mk


def one(upmu_dir: Path, nlr_dir: Path, fam: str, key: str, workload: str, sid: str):
    x = P.series_set(RH.load12(upmu_dir, fam))[key]
    tr = trace_kw(nlr_dir, workload, sid, RH.FX)
    n_hours = len(x) // int(3600 * RH.FX)
    set_hours = list(range(RH.CUT, n_hours))                # half B
    e = R.experiment(x, RH.FX, 3600, np.random.default_rng(H.SEED), return_null=True, return_stats=True,
                     make_signal=make_signal_from(tr), char_idx=list(range(RH.N_CHAR)) + set_hours,
                     **dict(H.REF_KW, swings=SCALES))
    return np.array(e["null_stats"]), e["inj_stats"], float(np.percentile(tr, 95) - np.percentile(tr, 5))


def main(upmu_dir: Path, nlr_dir: Path, out_dir: Path, stage: str | None = None):
    cache = upmu_dir / "nlr_inject_cache_v2"
    if stage:
        fam, key, workload, sid = stage.split(":")
        cached(cache, f"{fam}_{key}_{workload}_{sid}", lambda: one(upmu_dir, nlr_dir, fam, key, workload, sid))
        return None
    n_test = 277 - RH.N_CHAR
    a = list(range(0, RH.CUT - RH.N_CHAR)); b = list(range(RH.CUT - RH.N_CHAR, n_test))
    res = {"design": "learned on 36 characterisation hours + half B (hours 156-276); threshold = worst no-signal hour of half B x margin; scored on half A (hours 36-155)",
           "scales": SCALES, "rows": []}
    st = {}
    for f in sorted(cache.glob("*.pkl")):
        fam, key, rest = f.stem.split("_", 2)
        workload, sid = rest.rsplit("_", 1)
        st[(fam, workload, sid, key)] = pickle.loads(f.read_bytes())
    groups = {"sum (three-phase island)": ["sum"], "imbalance residual (single-phase island, phase unknown)": ["imbalance1", "imbalance2", "imbalance3"]}
    for (fam, workload, sid) in sorted({k[:3] for k in st}):
        for gname, keys in groups.items():
            if not all((fam, workload, sid, k) in st for k in keys):
                continue
            nulls = {k: st[(fam, workload, sid, k)][0] for k in keys}
            swing = st[(fam, workload, sid, keys[0])][2]
            for m in MARGINS:
                thr = HO.family_threshold(nulls, b) * m          # one threshold for the family, from half B
                fa = HO.family_false_alarms(nulls, thr, a)
                per = []
                for k in keys:
                    inj = st[(fam, workload, sid, k)][1]
                    rates = [HO.detection_rate(inj[j], thr, a) for j in range(len(SCALES))]
                    per.append({"series": k, "detection_rate_by_scale": rates,
                                "detection_rate_at_scale_1": float(np.interp(0.0, np.log(SCALES), rates)),
                                "scale_for_80pct": HO.swing_for(inj, SCALES, thr, a, 0.8), "scale_for_50pct": HO.swing_for(inj, SCALES, thr, a, 0.5)})
                s80 = [p["scale_for_80pct"] for p in per if p["scale_for_80pct"] is not None]
                res["rows"].append({"series_family": fam, "group": gname, "workload": workload, "slurm_id": sid, "margin": m,
                                    "trace_swing_kw_p5_p95": swing, "alarm_hours_in_half_A": len(fa), "score_hours": len(a),
                                    "detection_rate_at_scale_1": [p["detection_rate_at_scale_1"] for p in per],
                                    "scale_for_80pct_min_max": [min(s80), max(s80)] if s80 else None,
                                    "swing_kw_for_80pct_min_max": [min(s80) * swing, max(s80) * swing] if s80 else None,
                                    "per_series": per})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "nlr_injection.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    r = main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else None)
    if r:
        for row in r["rows"]:
            f = lambda v: "-" if v is None else f"{v[0]:.2f}-{v[1]:.2f}"
            print(f"{row['series_family']:<6}{row['group'][:22]:<23}{row['workload']:<17} x{row['margin']:<4} swing {row['trace_swing_kw_p5_p95']:.2f} kW | alarms {row['alarm_hours_in_half_A']}/{row['score_hours']} | "
                  f"rate at scale 1: {[round(v, 2) for v in row['detection_rate_at_scale_1']]} | scale for 80% {f(row['scale_for_80pct_min_max'])} = {f(row['swing_kw_for_80pct_min_max'])} kW")
