"""Measure the pulse-channel noise floor on the LBNL 12 kV bus and run the
injection experiment. Writes pulse_measurement.json (small, committed).

Usage:  python -m island_detectability_map.run_pulse [DATA_DIR] [OUT_DIR]
Quantity analysed: three-phase apparent power proxy  S = V1 * (I1 + I2 + I3)  in kVA
(magnitude streams only; an added load near unity power factor adds ~1 kVA per kW).

Origin: verification@f7a9006e
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

from island_detectability_map import pulse

SEED = 20260930
SWINGS_KVA = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
BAND = (0.1, 3.0)          # search band for a training cadence, Hz
F0_RANGE = (0.2, 3.0)      # Choukse et al. (arXiv 2508.14318): energy at 0.2-3 Hz
DUTY, JITTER = 0.8, 0.01
DEC = 10                   # 120 Hz -> 12 Hz for the injection experiment
WINDOWS_S = [600, 3600]
# (label, range of the per-job median cadence in Hz, training_trace parameters)
REALISTIC = [
    ("near-rectangle, 10% spread, rho 0.95", (0.1, 3.0), dict(sigma_log=0.1, rho=0.95, straggle=0.1)),
    ("sawtooth, 10% spread, rho 0.95", (0.1, 3.0), dict(sigma_log=0.1, rho=0.95, straggle=1.0)),
    ("sawtooth, 30% spread, rho 0.95", (0.1, 3.0), dict(sigma_log=0.3, rho=0.95, straggle=1.0)),
    ("sawtooth, 30% spread, rho 0.99", (0.1, 3.0), dict(sigma_log=0.3, rho=0.99, straggle=1.0)),
    # The range as first posed, run as stated beside the literature's 0.02-3 Hz (above ~6 Hz nothing
    # survives at this sampling): one job's period wandering over 0.1-100 Hz (+-2 sigma), median 3.16 Hz
    ("sawtooth, 0.1-100 Hz wander, rho 0.99", (3.16, 3.17), dict(sigma_log=1.73, rho=0.99, straggle=1.0)),
]
CADENCE_BANDS = [("20-100 mHz (one drop per 10-50 s)", (0.02, 0.1), 500.0),
                 ("0.1-1 Hz", (0.1, 1.0), 100.0),
                 ("1-3 Hz", (1.0, 3.0), 100.0)]
# The test period contains a strong pulsed load of the site's own at exactly 1.00 Hz
# (+ harmonics), absent from the characterisation period. The detector flags it (see
# "test_windows_with_unnotched_stat_over_100"). Results are reported twice: as flagged,
# and with that load whitelisted after the alarm is resolved by inspection.
RESOLVED_LINES_HZ = [1.0, 2.0, 3.0]
RESOLVED_HALF_WIDTH_HZ = 0.03


def load(data_dir: Path):
    z = np.load(data_dir / "grizzly_head.npz")
    s = z["L1MAG"].astype(np.float64) * (z["C1MAG"].astype(np.float64) + z["C2MAG"] + z["C3MAG"]) / 1e3
    return s, float(z["fs"]), int(z["t0_ns"])


def block_mean(x, k):
    n = len(x) // k * k
    return x[:n].reshape(-1, k).mean(axis=1)


def variability(s, fs, mean):
    out = {}
    for sec in (1, 10, 60):
        b = block_mean(s, int(sec * fs))
        out[f"rms_step_{sec}s_pct_of_mean"] = float(np.sqrt(np.mean(np.diff(b) ** 2)) / mean * 100)
    return out


def band_rms(f, psd, lo, hi):
    sel = (f >= lo) & (f < hi)
    return float(np.sqrt(np.sum(psd[sel]) * (f[1] - f[0])))


def experiment(x, fs, win_s, rng, char_hours=36.0, notch_ratio=10.0, notch_pad=2, whitelist=(),
               band=BAND, f0_range=F0_RANGE, seg_s=100.0, swings=None, jitter=JITTER, make_signal=None,
               thr_override=None, return_null=False, char_idx=None, return_stats=False):
    """Split the record: the first `char_hours` characterise the site (median baseline
    spectrum + a notch list of bins where the site itself shows intermittent lines,
    max/median >= notch_ratio, padded by notch_pad bins). Everything after is the test
    period: null statistics and injections. Nothing in the test period tunes the detector.
    thr_override: optional {"max_null", "p99_null"} used instead of this series' own null
    thresholds (e.g. one threshold shared across three phases when the island's phase is
    unknown). The series' own null statistics are still reported; return_null adds the
    per-window null statistics and the threshold actually used.
    char_idx: optional subset of the characterisation windows to build the baseline and
    notch list from (e.g. one hour-of-day class); the test period is unchanged.
    return_stats adds "inj_stats", an array (swings x test windows) of the injected-run
    statistic (-inf where the hit was at the wrong frequency), so thresholds can be
    applied per class afterwards."""
    n = int(win_s * fs)
    wins = [x[i:i + n] for i in range(0, len(x) - n + 1, n)]
    n_char = int(char_hours * 3600 / win_s)
    spectra = np.array([pulse.welch_psd(w, fs, seg_s)[1] for w in wins])
    f = np.fft.rfftfreq(int(round(seg_s * fs)), 1.0 / fs)
    sel = (f >= band[0]) & (f <= band[1])
    cidx = list(range(n_char)) if char_idx is None else list(char_idx)
    baseline = np.median(spectra[cidx], axis=0)
    noisy = (spectra[cidx].max(axis=0) / baseline >= notch_ratio) & sel
    notch = noisy.copy()
    for k in range(1, notch_pad + 1):
        notch[k:] |= noisy[:-k]; notch[:-k] |= noisy[k:]
    for w0 in whitelist:
        notch |= (np.abs(f - w0) <= max(RESOLVED_HALF_WIDTH_HZ, 2.0 / seg_s)) & sel   # cover Hann leakage at coarse resolution
    keep = sel & ~notch
    test = list(range(n_char, len(wins)))
    null = (spectra[test][:, keep] / baseline[keep]).max(axis=1)
    null_unnotched = (spectra[test][:, sel] / baseline[sel]).max(axis=1)
    own = {"max_null": float(null.max()), "p99_null": float(np.quantile(null, 0.99))}
    thr = dict(thr_override) if thr_override else own
    rows = []
    inj_stats = []
    tol = max(0.03, 3.0 / seg_s)
    for swing in (swings or SWINGS_KVA):
        hits_max = hits_p99 = 0
        inj_stats.append([])
        for i in test:
            f0 = float(np.exp(rng.uniform(np.log(f0_range[0]), np.log(f0_range[1]))))
            sig = make_signal(n, fs, f0, swing, rng) if make_signal else pulse.pulse_train(n, fs, f0, swing, DUTY, jitter, rng)
            y = wins[i] + sig
            sp = pulse.welch_psd(y, fs, seg_s)[1]
            r = sp[keep] / baseline[keep]
            j = int(np.argmax(r)); stat, f_hit = float(r[j]), float(f[keep][j])
            # a wandering cadence has no single line to match, so any exceedance counts for make_signal
            ok_f = make_signal is not None or min(abs(f_hit - m * f0) for m in (1, 2, 3)) < tol
            hits_max += stat > thr["max_null"] and ok_f
            hits_p99 += stat > thr["p99_null"] and ok_f
            inj_stats[-1].append(stat if ok_f else -np.inf)
        rows.append({"swing_kva": swing, "detect_rate_zero_fa": hits_max / len(test),
                     "detect_rate_1pct_fa": hits_p99 / len(test)})
    return {"window_s": win_s, "n_char_windows": n_char, "n_test_windows": len(test),
            "notched_bins": int(notch.sum()), "band_bins": int(sel.sum()),
            "notched_freqs_hz": [round(float(v), 2) for v in f[noisy]],
            "null_stat_median": float(np.median(null)), **own,
            **({"threshold_used": thr, "null_stats": [float(v) for v in null]} if return_null else {}),
            **({"inj_stats": np.array(inj_stats)} if return_stats else {}),
            "null_unnotched_max": float(null_unnotched.max()),
            "test_windows_with_unnotched_stat_over_100": int((null_unnotched > 100).sum()),
            "rows": rows}


def swing_at(rows, key, level=0.9):
    """log-interpolated swing at which the detection rate first reaches `level`."""
    prev = None
    for r in rows:
        if r[key] >= level:
            if prev is None or r[key] == prev[key]:
                return r["swing_kva"]
            a = (level - prev[key]) / (r[key] - prev[key])
            return float(np.exp(np.log(prev["swing_kva"]) + a * np.log(r["swing_kva"] / prev["swing_kva"])))
        prev = r
    return None


def main(data_dir: Path, out_dir: Path):
    rng = np.random.default_rng(SEED)
    s, fs, t0 = load(data_dir)
    mean = float(s.mean())
    res = {"source": "LBNL Open uPMU, grizzly_bus1_2, first 200 MiB of C1/C2/C3/L1 MAG streams",
           "t0_utc_ns": t0, "hours": len(s) / fs / 3600, "fs_hz": fs,
           "mean_kva": mean, "hourly_mean_min_kva": float(block_mean(s, int(3600 * fs)).min()),
           "hourly_mean_max_kva": float(block_mean(s, int(3600 * fs)).max()),
           "seed": SEED, "duty": DUTY, "jitter": JITTER, "band_hz": BAND, "f0_range_hz": F0_RANGE}
    res["variability"] = variability(s, fs, mean)
    f, psd = pulse.welch_psd(s, fs, 100.0)
    res["band_rms_pct_of_mean"] = {f"{lo}-{hi}Hz": band_rms(f, psd, lo, hi) / mean * 100
                                   for lo, hi in [(0.03, 3.0), (0.1, 3.0), (0.2, 3.0), (3.0, 60.0)]}
    centers = [0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0]
    res["psd_kva2_per_hz"] = {str(c): float(np.median(psd[(f >= c / 1.15) & (f <= c * 1.15)])) for c in centers}
    edges = np.logspace(np.log10(0.01), np.log10(60), 61)
    res["psd_curve"] = [[float(np.sqrt(a * b)), float(np.median(psd[(f >= a) & (f < b)]))]
                        for a, b in zip(edges[:-1], edges[1:]) if np.any((f >= a) & (f < b))]
    x = block_mean(s, DEC)
    res["experiments"] = []
    for win in WINDOWS_S:
        for label, wl in (("as_flagged", ()), ("site_load_whitelisted", RESOLVED_LINES_HZ)):
            # 10-minute windows average only ~11 segments, so chance ratios are larger
            e = experiment(x, fs / DEC, win, rng, notch_ratio=10.0 if win >= 3600 else 30.0, whitelist=wl)
            e["case"] = label
            for lvl in (0.5, 0.8, 0.9):   # ~8% of cadences land in notched bins, so rates cap near 0.9
                e[f"swing{int(lvl*100)}_zero_fa_kva"] = swing_at(e["rows"], "detect_rate_zero_fa", lvl)
                e[f"swing{int(lvl*100)}_1pct_fa_kva"] = swing_at(e["rows"], "detect_rate_1pct_fa", lvl)
            res["experiments"].append(e)
    # Cadence bands from the observed range (see params.yaml: observed_pulse). The low band
    # needs longer segments to resolve a 20-100 mHz line; all use one-hour windows.
    res["by_cadence_band"] = []
    for name, (lo, hi), seg in CADENCE_BANDS:
        e = experiment(x, fs / DEC, 3600, rng, whitelist=[float(k) for k in range(1, 7)],  # all harmonics of the site's 1 Hz load
                       band=(lo / 1.25, min(3.0 * hi, 5.5)),
                       f0_range=(lo, hi), seg_s=seg, swings=SWINGS_KVA + [32.0, 64.0, 128.0])
        e.update(band=name, cadence_hz=[lo, hi], seg_s=seg)
        for lvl in (0.5, 0.8):
            e[f"swing{int(lvl*100)}_zero_fa_kva"] = swing_at(e["rows"], "detect_rate_zero_fa", lvl)
        res["by_cadence_band"].append(e)
    # Line width vs integration: a perfectly regular cadence is a delta function in frequency,
    # and a longer FFT segment puts all of its power in a narrower bin (coherent gain). Step-time
    # jitter spreads the line and caps that gain. One-hour windows, 0.1-3 Hz cadences.
    res["by_segment_and_jitter"] = []
    for seg in (100.0, 600.0, 1800.0):
        for jit in (0.0, 0.01, 0.05):
            e = experiment(x, fs / DEC, 3600, rng, whitelist=[float(k) for k in range(1, 7)], band=(0.08, 5.5),
                           f0_range=(0.1, 3.0), seg_s=seg, jitter=jit, notch_ratio=10.0 if seg <= 100 else 30.0,
                           swings=[0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
            row = {"seg_s": seg, "jitter": jit, "max_null": e["max_null"], "notched_bins": e["notched_bins"]}
            for lvl in (0.5, 0.8):
                row[f"swing{int(lvl*100)}_zero_fa_kva"] = swing_at(e["rows"], "detect_rate_zero_fa", lvl)
            res["by_segment_and_jitter"].append(row)
    # Straggler signal model: correlated lognormal step durations, straggler sawtooth (pulse.training_trace).
    res["realistic"] = []
    for name, f0r, kw in REALISTIC:
        for seg in (10.0, 30.0, 100.0):
            mk = lambda n, fs_, f0, swing, rng_, kw=kw: pulse.training_trace(n, fs_, f0, swing, rng=rng_, **kw)
            e = experiment(x, fs / DEC, 3600, rng, whitelist=[float(k) for k in range(1, 7)], band=(max(0.08, 2.0 / seg), 5.5),
                           f0_range=f0r, seg_s=seg, swings=[2.0, 4.0, 8.0, 16.0, 32.0, 64.0], make_signal=mk)
            row = {"model": name, "f_med_range_hz": list(f0r), **kw, "seg_s": seg, "max_null": e["max_null"],
                   "rates": [[r_["swing_kva"], r_["detect_rate_zero_fa"]] for r_ in e["rows"]]}
            for lvl in (0.5, 0.8):
                row[f"swing{int(lvl*100)}_zero_fa_kva"] = swing_at(e["rows"], "detect_rate_zero_fa", lvl)
            res["realistic"].append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pulse_measurement.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    dd = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/upmu")
    od = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/20260930_island_detectability_map")
    r = main(dd, od)
    print("mean kVA", round(r["mean_kva"], 1), r["variability"], r["band_rms_pct_of_mean"])
    for e in r["experiments"]:
        print(e["case"], e["window_s"], "s windows: char", e["n_char_windows"], "test", e["n_test_windows"], "| notched", e["notched_bins"], "of", e["band_bins"],
              "| null median/p99/max", round(e["null_stat_median"], 2),
              round(e["p99_null"], 2), round(e["max_null"], 2), "| un-notched null max", round(e["null_unnotched_max"], 1), "windows>100:", e["test_windows_with_unnotched_stat_over_100"],
              "| swing50/80 zero-FA", e["swing50_zero_fa_kva"], e["swing80_zero_fa_kva"])
        print("   ", [(r_["swing_kva"], round(r_["detect_rate_zero_fa"], 2), round(r_["detect_rate_1pct_fa"], 2)) for r_ in e["rows"]])
    for e in r["by_cadence_band"]:
        print(e["band"], "| seg", e["seg_s"], "s | null max", round(e["max_null"], 1), "| swing50/80 kVA:", e["swing50_zero_fa_kva"], e["swing80_zero_fa_kva"],
              "|", [(r_["swing_kva"], round(r_["detect_rate_zero_fa"], 2)) for r_ in e["rows"] if r_["swing_kva"] >= 1])
    for e in r["by_segment_and_jitter"]:
        print("seg", e["seg_s"], "jitter", e["jitter"], "| null max", round(e["max_null"], 1), "| swing50/80 kVA:",
              e["swing50_zero_fa_kva"] and round(e["swing50_zero_fa_kva"], 2), e["swing80_zero_fa_kva"] and round(e["swing80_zero_fa_kva"], 2))
    for e in r["realistic"]:
        print(e["model"], "| seg", int(e["seg_s"]), "| null max", round(e["max_null"], 1), "| swing50/80:",
              e["swing50_zero_fa_kva"] and round(e["swing50_zero_fa_kva"], 1), e["swing80_zero_fa_kva"] and round(e["swing80_zero_fa_kva"], 1))
