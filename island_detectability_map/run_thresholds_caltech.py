"""Two thresholds on one statistic at the Caltech levels (the LBNL version is run_thresholds.py).

EXPLORATORY orchestration. Tested pieces: follum_pierre, welch_engine.experiment (exact equivalent of
run_pulse.experiment), welch_engine.sum_statistic, caltech.* (loader), phases.imbalance.

Design is run_task3's: Caltech week 2025-03-10..17, 1-second kVA, one-hour windows,
fold A = learn hours 0-83 (baseline, site lines, empirical threshold) and score 84-167, fold B the
reverse; no separate characterisation block (char_hours = 0); seed 20261005; bands slow (500 s segments,
0.016-0.3 Hz, f0 0.02-0.1) and fast (100 s, 0.08-0.5 Hz, f0 0.1-0.4), notch ratio 10, no whitelist.
Signal: NLR Llama-2 traces at 1 sample/s (run 10742829 as in run_task3, plus 10742831-34). Single-phase
island: added to one residual (max family) or +1/-1/2/-1/2 on the three residuals (PNNL sum).
Thresholds as in run_thresholds: empirical (worst learning hour x 1, x 2, max over the family) and
Follum-Pierre at P_FA 1e-2 and 1e-3 per hour (Bonferroni over all the family's searched bins, effective
dof per hour), plus the PNNL sum with GMSC / independent / identical thresholds.

Usage: python -m island_detectability_map.run_thresholds_caltech CALTECH_ROOT NETWORK_JSON CACHE_DIR NLR_DIR OUT_DIR [site:fold:band:workload[@run]]

Origin: verification@3e62e578
"""
from __future__ import annotations

import json, pickle, sys, time
from pathlib import Path

import numpy as np
from scipy import stats

from island_detectability_map import follum_pierre as FP, welch_engine as WE
from island_detectability_map import run_nlr_inject as NI, run_task3 as T3, run_thresholds as RT
from island_detectability_map.run_phasor import cached

FS = 1.0
SEG = {"slow": 500.0, "fast": 100.0}
P_FAS = RT.P_FAS
KEYS = {"sum": ["sum"], "resid": ["resid_a", "resid_b", "resid_c"]}
SITES = {k: v for k, v in T3.SITES.items() if k != "lbnl"}
CACHE = "thresholds_caltech_cache_v1"


def run_stage(root, net, nlr_dir, stage):
    site, fold, band, workload = stage.split(":")
    t0 = time.time()
    kva = T3.load(site, root, net, None)
    n_h = kva.shape[1] // 3600
    learn, score = T3.folds(site, n_h)[fold]
    name, sid = workload.split("@") if "@" in workload else (workload, dict(NI.CASES)[workload])
    tr = NI.trace_kw(Path(nlr_dir), name, sid, FS)
    kw = {k: v for k, v in T3.BANDS[band].items()}
    ser = T3.series_of(kva)
    out = {"stage": stage, "n_hours": n_h, "learn": learn, "score": score, "mean_kva": float(np.nanmean(kva.sum(axis=0))),
           "series": {}}
    for k in KEYS["sum"] + KEYS["resid"]:
        e = WE.experiment(ser[k], FS, 3600, tr, seed=T3.SEED, swings=T3.SCALES, char_idx=learn, char_hours=0, **kw)
        out["series"][k] = {x: e[x] for x in ("notched_bins", "band_bins", "notched_freqs_hz", "ratios", "segments", "keep",
                                              "baseline", "freqs", "test")}
        out["series"][k]["null"] = np.array(e["null_stats"]); out["series"][k]["inj"] = e["inj_stats"]
    X = np.vstack([ser[k] for k in KEYS["resid"]])
    s = WE.sum_statistic(X, FS, 3600, tr, seed=T3.SEED, swings=T3.SCALES, char_idx=learn, coefs=RT.COEFS, p_fas=P_FAS,
                         char_hours=0, **kw)
    out["pnnl_sum"] = {x: s[x] for x in ("null_Tmax", "G", "nu", "nu_char", "kept_bins", "band_bins", "norm")}
    from island_detectability_map import pulse
    f, p = pulse.welch_psd(tr, FS, min(len(tr) / FS, 1200.0))
    sel = (f >= 0.07) & (f <= 0.13)
    out["trace_f0_hz"] = float(f[sel][np.argmax(p[sel])])
    out["trace_A_kw"] = FP.fundamental_amplitude(tr, FS, out["trace_f0_hz"])
    out["trace_swing_kw"] = float(np.percentile(tr, 95) - np.percentile(tr, 5))
    out["seconds"] = time.time() - t0
    return out


def scale_for(rates):
    return T3.scale_for(rates, 0.8)


def compare(st, g):
    """Empirical and Follum-Pierre thresholds on the identical statistic (test axis = all hours here)."""
    keys = KEYS[g]
    band = st["stage"].split(":")[2]
    L = int(SEG[band] * FS)
    S = [st["series"][k] for k in keys]
    learn, score = st["learn"], st["score"]
    nu = np.array([FP.effective_dof(L, L // 2, n) if n > 0 else np.nan for n in S[0]["segments"]])
    nu_c = float(np.nanmedian(nu[learn]))
    m = FP.median_factor(nu_c)
    Btot = int(sum(int(s["keep"].sum()) for s in S))
    sw = st["trace_swing_kw"]
    fin = lambda s, idx: [i for i in idx if np.isfinite(s["null"][i])]
    row = {"group": g, "nu_char": nu_c, "bins_searched": Btot, "score_hours": len(score), "learn_hours": len(learn),
           "share_set_aside": [s["notched_bins"] / s["band_bins"] for s in S], "trace_swing_kw": sw,
           "hours_without_data": int(np.sum(~np.isfinite(S[0]["null"])))}

    def eval_thr(thr_h):                         # thr_h: per-hour threshold on the ratio statistic (test axis)
        nullmax = np.nanmax(np.vstack([s["null"] for s in S]), axis=0)
        alarms = int(np.sum(nullmax[score] > thr_h[score]))
        alarms_in = int(np.sum(nullmax[learn] > thr_h[learn]))
        s80 = []
        for s in S:
            idx = fin(s, score)
            rates = [float(np.mean(s["inj"][j, idx] > thr_h[idx])) for j in range(len(T3.SCALES))]
            s80.append(scale_for(rates))
        ok = [v for v in s80 if v is not None]
        return alarms, alarms_in, ([min(ok) * sw, max(ok) * sw] if ok else None), ([min(ok), max(ok)] if ok else None), len(ok)

    row["empirical"] = []
    for mg in (1.0, 2.0):
        thr = max(np.nanmax(s["null"][learn]) for s in S) * mg
        a, ai, swk, sc, n_ok = eval_thr(np.full(len(S[0]["null"]), thr))
        row["empirical"].append({"margin": mg, "threshold_ratio": thr, "alarm_hours": a, "swing_kw_80": swk, "scale_80": sc,
                                 "found_on_n_series": n_ok, "implied_pfa_per_hour": FP.implied_pfa(thr * m, Btot, nu_c)})
    row["fp"] = []
    for p in P_FAS:
        thr = np.array([FP.threshold(p, Btot, v) / m if np.isfinite(v) else np.inf for v in nu])
        a, ai, swk, sc, n_ok = eval_thr(thr)
        row["fp"].append({"p_fa_per_hour": p, "predicted_alarms": p * len(score), "alarm_hours": a,
                          "learning_half_alarm_hours": ai, "learning_half_predicted": p * len(learn),
                          "swing_kw_80": swk, "scale_80": sc, "found_on_n_series": n_ok})
    cal = {}
    for k, s in zip(keys, S):
        r = s["ratios"] * m
        lvl = np.array([stats.chi2.isf(RT.PER_BIN_LEVEL, v) / v if np.isfinite(v) else np.inf for v in nu])
        fl, fs_ = fin(s, learn), fin(s, score)
        cal[k] = {"learning_half": float(np.mean(r[fl] > lvl[fl, None])), "scored_half": float(np.mean(r[fs_] > lvl[fs_, None])),
                  "level_scatter_db_learning": RT.level_scatter_db(r[fl], nu_c), "level_scatter_db_scored": RT.level_scatter_db(r[fs_], nu_c)}
    row["per_bin_exceedance_at_1e-3"] = cal
    f = S[0]["freqs"]; b0 = int(np.argmin(np.abs(f - st["trace_f0_hz"])))
    row["f0_bin_searched"] = [bool(s["keep"][b0]) for s in S]
    snr1 = [FP.snr_metric(st["trace_A_kw"], s["baseline"][b0] / m, FS) for s in S]
    for r in row["empirical"] + row["fp"]:
        r["snr_db_at_80"] = (None if r["scale_80"] is None else
                             [min(10 * np.log10(v * c ** 2) for v in snr1 for c in r["scale_80"]),
                              max(10 * np.log10(v * c ** 2) for v in snr1 for c in r["scale_80"])])
    if g == "resid":
        q = st["pnnl_sum"]
        row["pnnl_sum"] = {"gmsc_median": float(np.median(q["G"])), "rows": []}
        for (mode, p), d in q["norm"].items():
            per = []
            for k in range(d["inj"].shape[0]):
                rates = [float(np.mean(d["inj"][k, j, score] > 1.0)) for j in range(len(T3.SCALES))]
                per.append(scale_for(rates))
            ok = [v for v in per if v is not None]
            row["pnnl_sum"]["rows"].append({"mode": mode, "p_fa_per_hour": p, "alarm_hours": int(np.sum(d["null"][score] > 1.0)),
                                            "learning_half_alarm_hours": int(np.sum(d["null"][learn] > 1.0)),
                                            "swing_kw_80": [min(ok) * sw, max(ok) * sw] if ok else None})
    return row


def summarise(cache: Path, out_dir: Path):
    res = {"design": __doc__.split("Design is")[1].split("Usage")[0].strip(), "sites": SITES, "p_fas": P_FAS, "rows": []}
    for fpk in sorted(cache.glob("*.pkl")):
        st = pickle.loads(fpk.read_bytes())
        site, fold, band, workload = st["stage"].split(":")
        for g in KEYS:
            r = compare(st, g)
            r.update(site=site, level=SITES[site], mean_kva=st["mean_kva"], fold=fold, band=band, seg_s=SEG[band],
                     workload=workload, trace_f0_hz=st["trace_f0_hz"], trace_A_kw=st["trace_A_kw"])
            res["rows"].append(r)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "thresholds_side_by_side_caltech.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def main(root, net, cache_dir, nlr_dir, out_dir, stage=None):
    cache = Path(cache_dir) / CACHE
    if stage:
        r = cached(cache, stage.replace(":", "__"), lambda: run_stage(root, net, nlr_dir, stage))
        print(stage, "done", round(r.get("seconds", 0)), "s")
        return None
    return summarise(cache, Path(out_dir))


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], a[3], a[4], a[5], a[6] if len(a) > 6 else None)
