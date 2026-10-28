"""Can anything find a three-phase island's steady slow pulse? And the real-job test repeated in
the causal direction: thresholds learned on later hours and scored on earlier ones look better than a
deployed monitor (which learns only from the past) would, because a site load can change frequency
between the two halves.

EXPLORATORY orchestration, no tests of its own. Tested pieces: run_pulse.experiment, the edge
counter (run_phase_edges / run_hours), nlr.*, run_nlr_inject.trace_kw / make_signal_from.

Signal: the measured 16-H100 trace (NLR logs, one-second means; see run_nlr_inject), looped from a
random start, times a scale factor (scale 1 = as logged, GPU power only, kW taken as kVA).
Background: LBNL bus, low-passed 12-a-second series ("maglp", build_upmu_lowpass).

Folds (both run; see above for why direction matters):
  fold "A" = learn on the 36 characterisation hours + hours 36-155, score hours 156-276 (causal)
  fold "B" = learn on the 36 characterisation hours + hours 156-276, score hours 36-155 (run_nlr_inject's design)
Spectral detector: run_pulse.experiment, baseline spectrum and site-line list from the learning
hours, segment length seg_s. Edge-count detector: as run_heldout (median/MAD from the first 36
hours); only its threshold comes from the learning half.
Threshold = worst no-signal hour of the learning half (one threshold per family) x margin.

Usage: python -m island_detectability_map.run_task2 UPMU_DIR NLR_DIR OUT_DIR [STAGE]
STAGE = <detector spec|edge>:<series>:<fold A|B>:<seg_s>:<workload[@run id] | elsayed-<job>>[:low|:slow]   (cached; one per command;
        "elsayed-<job>" takes an ELSAYED_JOBS session from DATA_DIR/external, the data dir being UPMU_DIR's parent;
        ":low" narrows the spectral search to 0.08-0.5 Hz, the band of the 1-s pipeline)
Without STAGE: summarise every cached stage into OUT_DIR/task2_slow_pulse.json.

Origin: verification@15a36a19
"""
from __future__ import annotations

import json, pickle, sys
from pathlib import Path

import numpy as np

from island_detectability_map import run_elsayed_inject as EI
from island_detectability_map import run_heldout as RH, run_hours as H, run_nlr_inject as NI
from island_detectability_map import run_phases as P, run_pulse as R
from island_detectability_map import run_phase_edges as E
from island_detectability_map.run_phasor import cached

FAM = "maglp"
SCALES = NI.SCALES
MARGINS = [1.0, 2.0]
SIDS = dict(NI.CASES)
N_HOURS = 277


def fold_hours(fold: str):
    """Learning hours (absolute) and the test-axis index sets (hour 36 = 0)."""
    n_test = N_HOURS - RH.N_CHAR
    a = list(range(0, RH.CUT - RH.N_CHAR)); b = list(range(RH.CUT - RH.N_CHAR, n_test))
    learn_abs = list(range(RH.N_CHAR, RH.CUT)) if fold == "A" else list(range(RH.CUT, N_HOURS))
    return learn_abs, (a, b) if fold == "A" else (b, a)


BANDS = {"full": {}, "low": dict(band=(0.08, 0.5), f0_range=(0.1, 0.4)),   # "low" = the 1-s pipeline fast band
         "slow": dict(band=(0.016, 0.3), f0_range=(0.02, 0.1))}                # "slow" = the 1-s pipeline slow band (use seg 500)


def spec_stage(x, trace, fold, seg_s, band="full"):
    learn_abs, _ = fold_hours(fold)
    e = R.experiment(x, RH.FX, 3600, np.random.default_rng(H.SEED), return_null=True, return_stats=True,
                     make_signal=NI.make_signal_from(trace), char_idx=list(range(RH.N_CHAR)) + learn_abs,
                     **dict(H.REF_KW, seg_s=float(seg_s), swings=SCALES, **BANDS[band]))
    return {"null": np.array(e["null_stats"]), "inj": np.array(e["inj_stats"]),
            "notched_bins": e["notched_bins"], "band_bins": e["band_bins"], "notched_freqs_hz": e["notched_freqs_hz"]}


def edge_stage(x, trace):
    """run_hours.edge_stats with the measured trace as the signal; null on the full test axis
    (NaN where an hour was set aside as a site load, as run_heldout does)."""
    n = int(3600 * RH.FX)
    wins = [x[i:i + n] for i in range(0, len(x) - n + 1, n)]
    c = np.array([E.counts(w) for w in wins])
    cidx = list(range(RH.N_CHAR))
    med = np.median(c[cidx], axis=0)
    mad = 1.4826 * np.median(np.abs(c[cidx] - med), axis=0) + 1.0
    z = lambda cc: float(np.max((cc - med) / mad))
    test = list(range(RH.N_CHAR, len(wins)))
    null = np.array([z(c[i]) for i in test])
    clean = null <= 100.0
    null[~clean] = np.nan
    mk = NI.make_signal_from(trace)
    rng = np.random.default_rng(H.SEED)
    inj = np.full((len(SCALES), len(test)), np.nan)
    for j, s in enumerate(SCALES):
        for t, i in enumerate(test):
            if clean[t]:
                inj[j, t] = z(E.counts(wins[i] + mk(n, RH.FX, 0.0, s, rng)))
    return {"null": null, "inj": inj}


ELSAYED_JOBS = {   # Elsayed et al. (Sci Data 2026) 8 x H100 node sessions, relative to the archive root
    "llama8b": "Node_Dataset/Text Generation LLMs/H100/Model_Size/LLM_H100_8GPU_Node_Lamma8B_SeqLength2048_Deepspeedds_z3_BatchSize16.csv",
    "diffusion": "Node_Dataset/Image Generation Diffusion Models/H100/Model_size_impact/Image_generation_H100_8GPU_Node_ModelSize128_ImageSize32_BatchSize128.csv",
}
ELSAYED_ROOT = Path("external/elsayed_2026/High-resolution-AI-Data-Center-Training-Workloads-Dataset_FigShare")   # under the data dir


def load_trace(workload: str, nlr_dir, elsayed_root, fx: float):
    """Signal in kW, mean removed, fx samples a second. "elsayed-<job>": an ELSAYED_JOBS session;
    "<nlr workload>[@<run id>]": an NLR run."""
    if workload.startswith("elsayed-"):
        return EI.trace_kw(Path(elsayed_root) / ELSAYED_JOBS[workload.removeprefix("elsayed-")], fx)
    name, sid = workload.split("@") if "@" in workload else (workload, SIDS[workload])   # "llama2_70b_lora@10742831": another run
    return NI.trace_kw(nlr_dir, name, sid, fx)


def run_stage(upmu_dir: Path, nlr_dir: Path, stage: str):
    det, key, fold, seg, workload, *rest = stage.split(":")
    x = P.series_set(RH.load12(upmu_dir, FAM))[key]
    tr = load_trace(workload, nlr_dir, upmu_dir.parent / ELSAYED_ROOT, RH.FX)
    out = spec_stage(x, tr, fold, seg, rest[0] if rest else "full") if det == "spec" else edge_stage(x, tr)
    out["trace_swing_kw_p5_p95"] = float(np.percentile(tr, 95) - np.percentile(tr, 5))
    return out


def scale_for(rates, level):
    for j in range(len(rates)):
        if rates[j] >= level:
            if j == 0:
                return SCALES[0]
            a = (level - rates[j - 1]) / (rates[j] - rates[j - 1])
            return float(np.exp(np.log(SCALES[j - 1]) + a * np.log(SCALES[j] / SCALES[j - 1])))
    return None


def score(stats: dict, keys, fold):
    _, (set_idx, score_idx) = fold_hours(fold)
    rows = []
    # last entry: DIAGNOSTIC ONLY, threshold = worst no-signal hour of the SCORED hours (optimistic;
    # says whether the statistic separates the job at all, not whether its threshold carries over)
    for m in MARGINS + ["scored_hours_max"]:
        thr = (max(np.nanmax(stats[k]["null"][score_idx]) for k in keys) if m == "scored_hours_max"
               else max(np.nanmax(stats[k]["null"][set_idx]) for k in keys) * m)
        fa = [i for i in score_idx if any(np.nan_to_num(stats[k]["null"][i], nan=-np.inf) > thr for k in keys)]
        per = []
        for k in keys:
            inj = stats[k]["inj"]
            ok = [i for i in score_idx if np.isfinite(stats[k]["null"][i])]
            rates = [float(np.mean(inj[j, ok] > thr)) for j in range(len(SCALES))]
            per.append({"series": k, "scored_hours": len(ok),
                        "rate_at_scale_1": float(np.interp(0.0, np.log(SCALES), rates)),
                        "scale_80": scale_for(rates, 0.8), "scale_50": scale_for(rates, 0.5)})
        rows.append({"margin": m, "threshold": float(thr), "alarm_hours": len(fa), "score_hours": len(score_idx), "per_series": per})
    return rows


def summarise(cache: Path, out_dir: Path):
    st = {}
    for f in sorted(cache.glob("*.pkl")):
        det, key, fold, seg, workload, *rest = f.stem.split("__")
        det = det if not rest else f"{det}-{rest[0]}"     # e.g. "spec-low": search band 0.08-0.5 Hz
        st[(det, key, fold, seg, workload)] = pickle.loads(f.read_bytes())
    groups = {"sum": ["sum"], "imbalance": ["imbalance1", "imbalance2", "imbalance3"]}
    res = {"design": __doc__.split("Folds")[1].split("Usage")[0].strip(), "scales": SCALES, "rows": []}
    for (det, fold, seg, workload) in sorted({(k[0], k[2], k[3], k[4]) for k in st}):
        for g, keys in groups.items():
            if not all((det, k, fold, seg, workload) in st for k in keys):
                continue
            stats = {k: st[(det, k, fold, seg, workload)] for k in keys}
            sw = stats[keys[0]]["trace_swing_kw_p5_p95"]
            extra = {}
            if det.startswith("spec"):
                extra = {"share_of_band_set_aside": [stats[k]["notched_bins"] / stats[k]["band_bins"] for k in keys],
                         "site_lines_hz": stats[keys[0]]["notched_freqs_hz"]}
            for r in score(stats, keys, fold):
                s80 = [p["scale_80"] for p in r["per_series"] if p["scale_80"] is not None]
                res["rows"].append({"detector": det, "group": g, "fold": fold, "learned_on": "hours 36-155" if fold == "A" else "hours 156-276",
                                    "seg_s": float(seg) if det.startswith("spec") else None, "workload": workload,
                                    "trace_swing_kw": sw, **r,
                                    "swing_kw_80_min_max": [min(s80) * sw, max(s80) * sw] if s80 else None, **extra})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "task2_slow_pulse.json").write_text(json.dumps(res, indent=1))
    return res


def main(upmu_dir: Path, nlr_dir: Path, out_dir: Path, stage: str | None = None):
    cache = upmu_dir / "task2_cache_v1"
    if stage:
        cached(cache, stage.replace(":", "__"), lambda: run_stage(upmu_dir, nlr_dir, stage))
        return None
    return summarise(cache, out_dir)


if __name__ == "__main__":
    r = main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else None)
    if r:
        for row in r["rows"]:
            f = lambda v: "-" if v is None else f"{v[0]:.1f}-{v[1]:.1f}"
            print(f"{row['detector']:<5}{row['group']:<10} learn {row['fold']} seg {row['seg_s']} x{str(row['margin'])[:6]:<6}"
                  f"| alarms {row['alarm_hours']}/{row['score_hours']} | rate@1 {[round(p['rate_at_scale_1'], 2) for p in row['per_series']]}"
                  f" | 80% at {f(row['swing_kw_80_min_max'])} kW (job {row['trace_swing_kw']:.2f} kW)")
