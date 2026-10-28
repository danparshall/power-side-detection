"""Two thresholds on one statistic, side by side. Both are run because
their disagreement is itself a measure of how non-stationary the background is.

EXPLORATORY orchestration. Tested pieces: follum_pierre (theory), welch_engine.experiment (exact equivalent
of run_pulse.experiment, the detector behind run_task2), welch_engine.sum_statistic.

Data and design are run_task2's: LBNL bus, low-passed 12-a-second series, folds A (learn hours 36-155, score
156-276) and B (the reverse), one-hour windows, real NLR Llama-2 trace looped from a random start, the same
swings (scales) and the same random draws. Bands: "low" (0.08-0.5 Hz, 100 s segments) and "slow"
(0.016-0.3 Hz, 500 s segments).

Statistic: the hourly maximum over searched bins of Welch PSD / learning-hour median PSD (run_task2's).
Thresholds compared on it:
  empirical   worst no-signal hour of the learning half x margin (1, 2), max over the family's series;
  FP          Follum-Pierre: the same ratio rescaled to P/Phi_c (Phi_c = median / median of chi2(nu)/nu),
              compared with chi2^-1(1 - P_FA/B; nu_h)/nu_h, nu_h the effective dof of that hour, B the bins
              searched over the whole family (Bonferroni; for the three residuals that is the max-based
              correction: 3 series' bins).
  PNNL sum    residual family only: T = sum_m nu P_m/Phi_m with the GMSC-interpolated threshold, and its
              independent / identical limits; injection on all three residuals (+1, -1/2, -1/2).
Also: the P_FA the empirical threshold implies under the chi-square null; per-bin calibration (share of
(hour, bin) pairs over a per-bin level of 1e-3) and the hour-to-hour level scatter in dB; Follum's A^2/(2 phi_c) at the 80% point.

Usage: python -m island_detectability_map.run_thresholds UPMU_DIR NLR_DIR OUT_DIR [fold:band:workload[@run]]
Without a stage: summarise cached stages into OUT_DIR/thresholds_side_by_side.json and the figure.

Origin: verification@8f90ad3c
"""
from __future__ import annotations

import json, pickle, sys, time
from pathlib import Path

import numpy as np
from scipy import stats

from island_detectability_map import follum_pierre as FP, welch_engine as WE
from island_detectability_map import run_heldout as RH, run_hours as H, run_nlr_inject as NI
from island_detectability_map import run_phases as P, run_task2 as T2
from island_detectability_map.run_phasor import cached

FS = RH.FX
SEG = {"low": 100.0, "slow": 500.0}
P_FAS = [1e-2, 1e-3]
KEYS = {"sum": ["sum"], "imbalance": ["imbalance1", "imbalance2", "imbalance3"]}
COEFS = [[1.0, -0.5, -0.5], [-0.5, 1.0, -0.5], [-0.5, -0.5, 1.0]]
PER_BIN_LEVEL = 1e-3
CACHE = "thresholds_cache_v1"


def trace_fundamental(tr):
    """Fundamental frequency (peak of the trace's PSD in 0.07-0.13 Hz) and its amplitude A."""
    from island_detectability_map import pulse
    f, p = pulse.welch_psd(tr, FS, min(len(tr) / FS, 1200.0))
    sel = (f >= 0.07) & (f <= 0.13)
    f0 = float(f[sel][np.argmax(p[sel])])
    return f0, FP.fundamental_amplitude(tr, FS, f0)


def run_stage(upmu_dir: Path, nlr_dir: Path, stage: str):
    fold, band, workload = stage.split(":")
    t0 = time.time()
    ser = P.series_set(RH.load12(upmu_dir, T2.FAM))
    name, sid = workload.split("@") if "@" in workload else (workload, T2.SIDS[workload])
    tr = NI.trace_kw(nlr_dir, name, sid, FS)
    learn_abs, _ = T2.fold_hours(fold)
    char_idx = list(range(RH.N_CHAR)) + learn_abs
    kw = {k: v for k, v in H.REF_KW.items() if k != "swings"}
    kw.update(seg_s=SEG[band], **T2.BANDS[band])
    out = {"stage": stage, "series": {}}
    for k in KEYS["sum"] + KEYS["imbalance"]:
        e = WE.experiment(ser[k], FS, 3600, tr, seed=H.SEED, swings=T2.SCALES, char_idx=char_idx, **kw)
        out["series"][k] = {x: e[x] for x in ("null_stats", "inj_stats", "notched_bins", "band_bins", "notched_freqs_hz",
                                              "ratios", "segments", "keep", "baseline", "freqs", "test")}
        out["series"][k]["null"] = np.array(e["null_stats"]); out["series"][k]["inj"] = e["inj_stats"]
    X = np.vstack([ser[k] for k in KEYS["imbalance"]])
    s = WE.sum_statistic(X, FS, 3600, tr, seed=H.SEED, swings=T2.SCALES, char_idx=char_idx, coefs=COEFS, p_fas=P_FAS, **kw)
    out["pnnl_sum"] = {x: s[x] for x in ("null_Tmax", "G", "nu", "nu_char", "kept_bins", "band_bins", "norm")}
    f0, A = trace_fundamental(tr)
    out.update(trace_swing_kw=float(np.percentile(tr, 95) - np.percentile(tr, 5)), trace_f0_hz=f0, trace_A_kw=A,
               trace_len_s=len(tr) / FS, seconds=time.time() - t0)
    return out


def level_scatter_db(r, nu):
    """Hour-to-hour scatter of the spectrum level beyond chi-square sampling, in dB (1 sd): per bin, the
    variance over hours of ln(P/Phi) minus its chi-square value trigamma(nu/2), averaged over bins.
    0 dB = stationary as the null assumes."""
    from scipy.special import polygamma
    v = float(np.mean(np.var(np.log(r), axis=0))) - float(polygamma(1, nu / 2.0))
    return float(10 / np.log(10) * np.sqrt(max(v, 0.0)))


def rates_to_swing(rates, sw):
    s = T2.scale_for(rates, 0.8)
    return (None if s is None else s * sw), s


def summarise_group(st, g, fold):
    keys = KEYS[g]
    _, (set_idx, score_idx) = T2.fold_hours(fold)
    S = [st["series"][k] for k in keys]
    nu = np.array([FP.effective_dof(int(SEG_OF(st) * FS), int(SEG_OF(st) * FS) // 2, n) for n in S[0]["segments"]])
    test = S[0]["test"]
    nu_t = nu[test]
    char_idx = list(range(RH.N_CHAR)) + T2.fold_hours(fold)[0]
    nu_c = float(np.median(nu[char_idx]))
    m = FP.median_factor(nu_c)
    Btot = int(sum(int(s["keep"].sum()) for s in S))
    sw = st["trace_swing_kw"]
    row = {"group": g, "fold": fold, "nu_char": nu_c, "nu_nominal": 2 * int(np.median(S[0]["segments"])),
           "median_factor": m, "bins_searched": Btot, "band_bins_each": S[0]["band_bins"],
           "share_set_aside": [s["notched_bins"] / s["band_bins"] for s in S],
           "site_lines_hz": sorted({v for s in S for v in s["notched_freqs_hz"]}),
           "score_hours": len(score_idx), "learn_hours": len(set_idx), "trace_swing_kw": sw}
    # empirical (run_task2.score on identical statistics)
    emp = T2.score({k: {"null": st["series"][k]["null"], "inj": st["series"][k]["inj"]} for k in keys}, keys, fold)
    row["empirical"] = []
    for r in emp:
        s80 = [p["scale_80"] for p in r["per_series"] if p["scale_80"] is not None]
        thr_fp_units = r["threshold"] * m
        row["empirical"].append({"margin": r["margin"], "threshold_ratio": r["threshold"], "threshold_P_over_Phi": thr_fp_units,
                                 "alarm_hours": r["alarm_hours"],
                                 "implied_pfa_per_hour": FP.implied_pfa(thr_fp_units, Btot, nu_c),
                                 "swing_kw_80": [min(s80) * sw, max(s80) * sw] if s80 else None,
                                 "scale_80": [min(s80), max(s80)] if s80 else None})
    # Follum-Pierre (max over the family's series, Bonferroni over all their bins)
    nullm = np.max(np.vstack([s["null"] for s in S]), axis=0) * m          # test axis
    row["fp"] = []
    for p in P_FAS:
        thr = np.array([FP.threshold(p, Btot, v) for v in nu_t])
        alarms = int(np.sum(nullm[score_idx] > thr[score_idx]))
        alarms_in = int(np.sum(nullm[set_idx] > thr[set_idx]))
        per = []
        for s in S:
            rates = [float(np.mean(s["inj"][j, score_idx] * m > thr[score_idx])) for j in range(len(T2.SCALES))]
            per.append(T2.scale_for(rates, 0.8))
        s80 = [v for v in per if v is not None]
        row["fp"].append({"p_fa_per_hour": p, "threshold_P_over_Phi_median": float(np.median(thr)),
                          "predicted_alarms": p * len(score_idx), "alarm_hours": alarms,
                          "learning_half_alarm_hours": alarms_in, "learning_half_predicted": p * len(set_idx),
                          "swing_kw_80": [min(s80) * sw, max(s80) * sw] if s80 else None,
                          "scale_80": [min(s80), max(s80)] if s80 else None})
    # per-bin calibration: share of (hour, kept bin) above the per-bin chi-square level
    cal = {}
    for k, s in zip(keys, S):
        lvl = np.array([stats.chi2.isf(PER_BIN_LEVEL, v) / v for v in nu_t])
        r = s["ratios"] * m
        cal[k] = {"learning_half": float(np.mean(r[set_idx] > lvl[set_idx, None])),
                  "scored_half": float(np.mean(r[score_idx] > lvl[score_idx, None])),
                  "median_ratio_scored": float(np.median(r[score_idx])),
                  "level_scatter_db_learning": level_scatter_db(r[set_idx], nu_c),
                  "level_scatter_db_scored": level_scatter_db(r[score_idx], nu_c)}
    row["per_bin_exceedance_at_1e-3"] = cal
    # Follum's SNR metric at the fundamental, per series: A^2/(2 phi_c), phi_c per-sample ambient PSD
    f = S[0]["freqs"]; f0 = st["trace_f0_hz"]; b0 = int(np.argmin(np.abs(f - f0)))
    row["f0_bin_hz"] = float(f[b0])
    row["f0_bin_searched"] = [bool(s["keep"][b0]) for s in S]
    snr1 = [FP.snr_metric(st["trace_A_kw"], s["baseline"][b0] / m, FS) for s in S]   # at the logged size
    row["snr_db_at_logged_size"] = [10 * np.log10(v) for v in snr1]

    def snr_at(scale_pair):
        if scale_pair is None:
            return None
        v = [10 * np.log10(snr1[i] * sc ** 2) for i in range(len(S)) for sc in scale_pair]
        return [min(v), max(v)]
    for r in row["empirical"]:
        r["snr_db_at_80"] = snr_at(r["scale_80"])
    for r in row["fp"]:
        r["snr_db_at_80"] = snr_at(r["scale_80"])
    return row


def summarise_pnnl(st, fold):
    _, (set_idx, score_idx) = T2.fold_hours(fold)
    q = st["pnnl_sum"]
    sw = st["trace_swing_kw"]
    out = {"gmsc_median": float(np.median(q["G"])), "gmsc_p10_p90": [float(np.percentile(q["G"], 10)), float(np.percentile(q["G"], 90))],
           "bins_searched": int(q["kept_bins"]), "rows": []}
    for (mode, p), d in q["norm"].items():
        per = []
        for k in range(d["inj"].shape[0]):
            rates = [float(np.mean(d["inj"][k, j, score_idx] > 1.0)) for j in range(len(T2.SCALES))]
            per.append(T2.scale_for(rates, 0.8))
        s80 = [v for v in per if v is not None]
        out["rows"].append({"mode": mode, "p_fa_per_hour": p, "predicted_alarms": p * len(score_idx),
                            "alarm_hours": int(np.sum(d["null"][score_idx] > 1.0)),
                            "learning_half_alarm_hours": int(np.sum(d["null"][set_idx] > 1.0)),
                            "swing_kw_80": [min(s80) * sw, max(s80) * sw] if s80 else None})
    return out


def SEG_OF(st):
    return SEG[st["stage"].split(":")[1]]


def summarise(cache: Path, out_dir: Path):
    res = {"design": __doc__.split("Statistic:")[0].split("Data and design")[1].strip(), "p_fas": P_FAS, "rows": []}
    curves = {}
    for fpk in sorted(cache.glob("*.pkl")):
        st = pickle.loads(fpk.read_bytes())
        fold, band, workload = st["stage"].split(":")
        base = {"fold": fold, "band": band, "seg_s": SEG[band], "workload": workload, "trace_f0_hz": st["trace_f0_hz"],
                "trace_A_kw": st["trace_A_kw"], "trace_swing_kw": st["trace_swing_kw"]}
        for g in KEYS:
            r = summarise_group(st, g, fold)
            r.update(base)
            if g == "imbalance":
                r["pnnl_sum"] = summarise_pnnl(st, fold)
            res["rows"].append(r)
        if workload == "llama2_70b_lora":            # curves are signal-free: one run is enough
            curves[(fold, band)] = calibration_curves(st, fold)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "thresholds_side_by_side.json").write_text(json.dumps(res, indent=1, default=float))
    if curves:
        plot_curves(curves, out_dir / "thresholds_per_bin_calibration.png")
    return res


def calibration_curves(st, fold):
    _, (set_idx, score_idx) = T2.fold_hours(fold)
    band = st["stage"].split(":")[1]
    L = int(SEG[band] * FS)
    out = {}
    for k in ("sum", "imbalance1"):
        s = st["series"][k]
        nu = np.array([FP.effective_dof(L, L // 2, n) for n in s["segments"]])
        char_idx = list(range(RH.N_CHAR)) + T2.fold_hours(fold)[0]
        nu_c = float(np.median(nu[char_idx]))
        r = s["ratios"] * FP.median_factor(nu_c)
        grid = np.linspace(0.5, 6.0, 111)
        out[k] = {"grid": grid, "nu": nu_c,
                  "learning": [float(np.mean(r[set_idx] > t)) for t in grid],
                  "scored": [float(np.mean(r[score_idx] > t)) for t in grid],
                  "theory": stats.chi2.sf(nu_c * grid, nu_c)}
    return out


def plot_curves(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = sorted(curves)
    fig, axes = plt.subplots(len(keys), 2, figsize=(10, 3.2 * len(keys)), squeeze=False)
    for i, ck in enumerate(keys):
        for j, k in enumerate(("sum", "imbalance1")):
            c = curves[ck][k]; ax = axes[i, j]
            ax.semilogy(c["grid"], c["theory"], "k-", label="chi-square theory (stationary null)")
            ax.semilogy(c["grid"], np.maximum(c["learning"], 1e-7), "b--", label="training hours (in-sample)")
            ax.semilogy(c["grid"], np.maximum(c["scored"], 1e-7), "r-", label="test hours (held-out)")
            ax.set_ylim(1e-6, 1.2); ax.set_xlim(0.5, 6)
            ax.set_title(f"fold {ck[0]}, {ck[1]} band ({int(SEG[ck[1]])} s segs, nu = {c['nu']:.1f}), "
                         + ("three-phase total" if k == "sum" else "phase-unbalance residual, phase 1"), fontsize=9)
            ax.set_xlabel("per-bin P / Phi_c (Phi_c from training hours)")
            ax.set_ylabel("per-bin exceedance rate")
            ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Ambient PSD and site lines from the learning hours only. Red = HELD-OUT (test hours); blue = in-sample (training hours).\n"
                 "LBNL 12 kV bus, no signal added. Gap between the data and black = non-stationarity of the ambient noise.",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main(upmu_dir: Path, nlr_dir: Path, out_dir: Path, stage: str | None = None):
    cache = upmu_dir / CACHE
    if stage:
        r = cached(cache, stage.replace(":", "__"), lambda: run_stage(upmu_dir, nlr_dir, stage))
        print(stage, "done", round(r.get("seconds", 0)), "s")
        return None
    return summarise(cache, out_dir)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else None)
