"""False alarms with thresholds set on held-out data, by hour-of-day class.
Writes heldout_measurement.json.

EXPLORATORY orchestration. Tested pieces: holdout.* (tests/test_holdout.py), hours.*,
upmu_full.*, run_pulse.experiment. This script wires them and has no unit tests.

Three months of one LBNL sensor were wanted. The public files hold 11.57 days
(2015-10-01 00:00 to 2015-10-12 13:46 UTC; 277 whole hours), so a rate "per month" cannot be
measured. What is done instead:

* Detectors exactly as in run_phases / run_phase_edges / run_hours: the first 36 hours characterise the site (baseline
  spectrum, notch list, edge-count medians); map-reference sawtooth signal; fine swing grid.
* Hours 36-74 are the old test hours: every threshold "used so far" is the worst no-signal
  hour among them. Hours 75-276 (202 hours) were never looked at before this run.
  (1) "as used": the old thresholds, scored on the 202 new hours.
  (2) two halves: A = hours 36-155 (Fri 05:00 to Wed 05:00 Pacific), B = hours 156-276.
      Threshold = worst no-signal hour of one half; false alarms and detection scored on the
      other; then swapped. Also with one threshold per hour-of-day class.
  (3) margin: the same with the threshold multiplied by a safety factor.
* A threshold that is the worst of N hours is expected to be exceeded about once in every
  N+1 later hours even if nothing changes. "expected_if_alike" gives that count; more than
  that means the later hours differ from the earlier ones.
* Unknown phase: one threshold for the three phases (and one for the three residuals); an
  hour is a false alarm if any of the three exceeds it.
* Edge-count detector: hours where the site's own 1 Hz load trips it (z > 100) are set aside,
  as before; they are counted and reported.
* Step and notch look-alikes on 1-second data are counted per day over the whole record. Their
  thresholds are sizes in kVA fixed in advance, so days 4-12 are held out for them as they stand.

Stages are cached under DATA_DIR/heldout_cache/ (gitignored). STAGE = spectral:<family>:<series>
or edge:<family>:<series> runs one series; "flagged:<family>" the un-whitelisted statistic.
Usage: python -m island_detectability_map.run_heldout ../data/upmu OUT_DIR [STAGE]
       python -m island_detectability_map.run_heldout ../data/upmu OUT_DIR score=mag,maglp:heldout_lowpass.json
("maglp" is the low-passed magnitude series of build_upmu_lowpass.py.)

Origin: verification@367891f4
"""
from __future__ import annotations

import json, pickle, sys
from pathlib import Path

import numpy as np

from island_detectability_map import holdout as HO, hours
from island_detectability_map import run_pulse as R, run_sync as S, run_phases as P, run_hours as H
from island_detectability_map.run_phasor import cached, fill_gaps, drop_near

FX = 12.0
FINE = H.FINE
FAMILIES = H.FAMILIES
N_CHAR, OLD_END, CUT = 36, 75, 156
MARGINS = [1.0, 1.5, 2.0, 3.0]
STEP_SIZES = [2.0, 5.0, 10.0, 20.0]
SERIES = ["sum", "phase1", "phase2", "phase3", "imbalance1", "imbalance2", "imbalance3"]


def load12(data_dir: Path, quantity: str):
    x = np.load(data_dir / "series" / f"{quantity}12.npy")
    n = x.shape[1] // int(3600 * FX) * int(3600 * FX)
    return x[:, :n]


def spectral_one(x):
    """No-signal and with-signal statistics for every hour after the characterisation hours."""
    e = R.experiment(x, FX, 3600, np.random.default_rng(H.SEED), return_null=True, return_stats=True,
                     make_signal=H.MK, **dict(H.REF_KW, swings=FINE))
    return np.array(e["null_stats"]), e["inj_stats"]


def spectral_fold(x, fold: str):
    """Everything the detector learns comes from one half: the baseline spectrum and the list
    of the site's own lines are built from the 36 characterisation hours plus the SET half
    (fold "A": hours 36-155; fold "B": hours 156-276). The other half is only scored."""
    n_hours = len(x) // int(3600 * FX)
    set_hours = list(range(N_CHAR, CUT)) if fold == "A" else list(range(CUT, n_hours))
    e = R.experiment(x, FX, 3600, np.random.default_rng(H.SEED), return_null=True, return_stats=True,
                     make_signal=H.MK, char_idx=list(range(N_CHAR)) + set_hours, **dict(H.REF_KW, swings=FINE))
    return np.array(e["null_stats"]), e["inj_stats"], e["notched_bins"], e["band_bins"], e["notched_freqs_hz"]


def score_folds(cache: Path, fam: str, labels, n_test):
    """Tables for the two-half runs where baseline, line list and threshold all come from the set half."""
    lab = labels[N_CHAR:N_CHAR + n_test]
    a = list(range(0, CUT - N_CHAR)); b = list(range(CUT - N_CHAR, n_test))
    rows = []
    for fold, set_idx, score_idx in (("A", a, b), ("B", b, a)):
        files = {k: cache / f"spectral2_{fam}_{k}_{fold}.pkl" for k in SERIES}
        st = {k: pickle.loads(p.read_bytes()) for k, p in files.items() if p.exists()}
        for family, keys in FAMILIES.items():
            if not all(k in st for k in keys):          # a family is scored only when all its series were run
                continue
            nulls = {k: st[k][0] for k in keys}
            for margin in MARGINS:
                thr = HO.family_threshold(nulls, set_idx) * margin
                fa = HO.family_false_alarms(nulls, thr, score_idx)
                lo, hi = HO.rate_interval(len(fa), len(score_idx), 0.90 if fa else 0.95)
                rows.append({"family": family, "set_half": fold, "margin": margin, "set_hours": len(set_idx), "score_hours": len(score_idx),
                             "threshold": thr, "false_alarm_hours": len(fa), "false_alarm_hour_index": [int(i) + N_CHAR for i in fa],
                             "false_alarm_classes": [lab[i] for i in fa],
                             "worst_excess_ratio": float(max(max(nulls[k][i] for k in keys) for i in fa) / thr) if fa else None,
                             "per_month_low_high": [lo * 720, hi * 720],
                             "share_of_band_notched": [st[k][2] / st[k][3] for k in keys],
                             "swing80": H.summarise([HO.swing_for(st[k][1], FINE, thr, score_idx, 0.8) for k in keys]),
                             "swing50": H.summarise([HO.swing_for(st[k][1], FINE, thr, score_idx, 0.5) for k in keys]),
                             "swing80_quiet_score_hours": H.summarise([HO.swing_for(st[k][1], FINE, thr, [i for i in score_idx if lab[i] != "work"], 0.8) for k in keys]),
                             "swing80_work_score_hours": H.summarise([HO.swing_for(st[k][1], FINE, thr, [i for i in score_idx if lab[i] == "work"], 0.8) for k in keys])})
    return rows


def flagged_family(ser):
    """Spectral statistic without whitelisting the site's own 1 Hz load (one cheap swing)."""
    out = {}
    for key, x in ser.items():
        e = R.experiment(x, FX, 3600, np.random.default_rng(H.SEED), return_null=True, make_signal=H.MK,
                         **dict(H.REF_KW, whitelist=(), swings=[1.0]))
        out[key] = np.array(e["null_stats"])
    return out


def edge_one(x):
    pos, null, inj = H.edge_stats(x, FX, list(range(N_CHAR)), FINE)
    return pos, null, inj


def to_full(pos, null, inj, n_test):
    """Edge statistics on the full hour axis: NaN where the hour was set aside."""
    nf = np.full(n_test, np.nan); nf[pos] = null
    jf = np.full((inj.shape[0], n_test), np.nan); jf[:, pos] = inj
    return nf, jf


def score(stats, labels, n_test):
    """stats: {series: (null[n_test], inj[swings, n_test])} on the test-hour axis (hour 36 = 0).
    Returns the tables for every family."""
    old = list(range(0, OLD_END - N_CHAR))
    new = list(range(OLD_END - N_CHAR, n_test))
    a = list(range(0, CUT - N_CHAR)); b = list(range(CUT - N_CHAR, n_test))
    lab = labels[N_CHAR:N_CHAR + n_test]
    out = {"hours": {"old_test": len(old), "new": len(new), "half_A": len(a), "half_B": len(b),
                     "by_class": {name: {c: len(v) for c, v in HO.by_class(lab, idx).items()}
                                  for name, idx in (("old_test", old), ("new", new), ("half_A", a), ("half_B", b))}},
           "rows": []}

    def usable(keys, idx):
        return [i for i in idx if all(np.isfinite(stats[k][0][i]) for k in keys)]

    def one(fam, name, set_idx, score_idx, margin=1.0, cls=None):
        keys = FAMILIES[fam]
        if cls:
            set_idx = [i for i in set_idx if (lab[i] == "work") == (cls == "work")]
            score_idx = [i for i in score_idx if (lab[i] == "work") == (cls == "work")]
        set_u, score_u = usable(keys, set_idx), usable(keys, score_idx)
        if not set_u or not score_u:
            return None
        nulls = {k: stats[k][0] for k in keys}
        thr = HO.family_threshold(nulls, set_u) * margin
        fa = HO.family_false_alarms(nulls, thr, score_u)
        lo, hi = HO.rate_interval(len(fa), len(score_u), 0.90 if fa else 0.95)
        sw = [HO.swing_for(stats[k][1], FINE, thr, score_u, 0.8) for k in keys]
        sw50 = [HO.swing_for(stats[k][1], FINE, thr, score_u, 0.5) for k in keys]
        return {"family": fam, "rule": name, "class": cls or "all", "margin": margin,
                "set_hours": len(set_u), "score_hours": len(score_u), "set_aside_score_hours": len(score_idx) - len(score_u),
                "threshold": thr, "false_alarm_hours": len(fa), "expected_if_alike": len(score_u) / (len(set_u) + 1) if margin == 1.0 else None,
                "false_alarm_hour_index": [int(i) + N_CHAR for i in fa],
                "false_alarm_classes": [lab[i] for i in fa],
                "worst_excess_ratio": float(max((max(stats[k][0][i] for k in keys) for i in fa), default=np.nan) / thr) if fa else None,
                "per_month_low_high": [lo * 720, hi * 720],
                "swing80": H.summarise(sw), "swing50": H.summarise(sw50)}

    for fam in FAMILIES:
        for cls in (None, "work", "quiet"):
            for name, s_idx, c_idx in (("as used: set on old test hours, scored on new", old, new),
                                       ("set on A, scored on B", a, b), ("set on B, scored on A", b, a),
                                       ("same hours (the old rule), all test hours", old + new, old + new)):
                for m in (MARGINS if cls is None and "same hours" not in name else [1.0]):
                    r = one(fam, name, s_idx, c_idx, m, cls)
                    if r:
                        out["rows"].append(r)
    return out


def daily_events(data_dir: Path, quantity: str, t0_ns: int):
    """Sharp up-steps per calendar day of the record (24 h blocks from t0), by size, for the
    sum, each phase and each residual; plus rates per 24 h of each hour-of-day class."""
    x1, bad = fill_gaps(np.load(data_dir / "series" / f"{quantity}1.npy"))
    n = x1.shape[1]
    lab = hours.hour_labels(t0_ns, n // 3600 + 1)
    sec_lab = np.array(lab)[np.minimum(np.arange(n) // 3600, len(lab) - 1)]
    cls_hours = {c: float(np.sum(sec_lab == c)) / 3600.0 for c in ("work", "weeknight", "weekend")}
    n_days = int(np.ceil(n / 86400))
    out = {"class_hours": cls_hours, "day_lengths_h": [min(86400, n - d * 86400) / 3600 for d in range(n_days)], "series": {}}
    for key, v in P.series_set(x1).items():
        t, st = S.upsteps(v)
        row = {}
        for a in STEP_SIZES:
            ev = drop_near(S.events(t, st, a), bad)
            row[str(a)] = {"per_day": np.bincount(ev // 86400, minlength=n_days).tolist(),
                           "per_24h_by_class": {c: float(np.sum(sec_lab[ev] == c) / h * 24) for c, h in cls_hours.items()},
                           "total": int(len(ev))}
        out["series"][key] = row
    # notches on the sum (stall look-alikes), as run_dips / run_hours
    notch = {}
    for d in H.NOTCH_DURATIONS:
        w = max(5, min(d, 60))
        ev = H.notch_times(x1.sum(axis=0), d, w)
        tt, dep = ev[:, 0].astype(int), ev[:, 1]
        keep = np.isin(tt, drop_near(tt, bad, margin=d + w))
        tt, dep = tt[keep], dep[keep]
        notch[str(d)] = {str(a): {"per_day": np.bincount(tt[dep >= a] // 86400, minlength=n_days).tolist(),
                                  "per_24h_by_class": {c: float(np.sum((sec_lab[tt] == c) & (dep >= a)) / h * 24) for c, h in cls_hours.items()}}
                         for a in H.NOTCH_DEPTHS}
    out["notches_sum"] = notch
    return out


def main(data_dir: Path, out_dir: Path, stage: str | None = None, fams=("mag", "p"), out_name: str = "heldout_measurement.json"):
    """fams / out_name: which series families to score and where to write (a run on the
    low-passed series scores the family "maglp" beside "mag" into its own file; the default call is unchanged)."""
    cache = data_dir / "heldout_cache"
    meta = json.loads((data_dir / "series" / "meta.json").read_text())
    t0_ns = meta["t0_ns"]
    if stage:
        kind, fam, *rest = stage.split(":")
        ser = P.series_set(load12(data_dir, fam))
        if kind == "flagged":
            cached(cache, f"flagged_{fam}", lambda: flagged_family(ser))
        elif kind == "events":
            cached(cache, f"events_{fam}", lambda: daily_events(data_dir, fam, t0_ns))
        elif kind == "spectral2":
            cached(cache, f"spectral2_{fam}_{rest[0]}_{rest[1]}", lambda: spectral_fold(ser[rest[0]], rest[1]))
        else:
            fn = spectral_one if kind == "spectral" else edge_one
            cached(cache, f"{kind}_{fam}_{rest[0]}", lambda: fn(ser[rest[0]]))
        return None
    n_hours = 277
    n_test = n_hours - N_CHAR
    labels = hours.hour_labels(t0_ns, n_hours)
    res = {"source": "LBNL Open uPMU grizzly_bus1_2, whole public record: 2015-10-01 00:00 to 2015-10-12 13:46 UTC (277 whole hours)",
           "detectors": "as run_phases / run_phase_edges: first 36 h characterise; sawtooth 10% spread; fine swing grid (steps of 2^(1/4))",
           "labels_test_hours": labels[N_CHAR:], "families": {}}
    for fam in fams:
        block = {}
        for kind in ("spectral", "edge"):
            files = {k: cache / f"{kind}_{fam}_{k}.pkl" for k in SERIES}
            if not all(f.exists() for f in files.values()):
                continue
            stats = {}
            set_aside = {}
            for k, f in files.items():
                v = pickle.loads(f.read_bytes())
                if kind == "edge":
                    stats[k] = to_full(*v, n_test)
                    set_aside[k] = [int(i) + N_CHAR for i in range(n_test) if i not in set(v[0].tolist())]
                else:
                    stats[k] = v
            block[kind] = score(stats, labels, n_test)
            block[kind]["null_by_hour"] = {k: [None if not np.isfinite(x) else float(x) for x in stats[k][0]] for k in SERIES}
            if kind == "edge":
                block[kind]["site_load_hours_set_aside"] = set_aside
        folds = score_folds(cache, fam, labels, n_test)
        if folds:
            block["spectral_learned_on_one_half"] = folds
        for extra in ("flagged", "events"):
            f = cache / f"{extra}_{fam}.pkl"
            if f.exists():
                v = pickle.loads(f.read_bytes())
                block[extra] = {k: [float(x) for x in a] for k, a in v.items()} if extra == "flagged" else v
        if block:
            res["families"][fam] = block
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / out_name).write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[3].startswith("score="):       # score=mag,maglp:heldout_lowpass.json
        fams_, name_ = sys.argv[3][6:].split(":")
        r = main(Path(sys.argv[1]), Path(sys.argv[2]), None, tuple(fams_.split(",")), name_)
    else:
        r = main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else None)
    if r is not None:
        for fam, block in r["families"].items():
            for kind in ("spectral", "edge"):
                if kind not in block:
                    continue
                print("##", fam, kind, block[kind]["hours"])
                for row in block[kind]["rows"]:
                    f = lambda v: "-" if v is None else (f"{v[0]:.1f}" if abs(v[0] - v[1]) < 0.05 else f"{v[0]:.1f}-{v[1]:.1f}")
                    print(f"  {row['family']:<10}{row['class']:<6}{row['rule']:<48} x{row['margin']:<4} set {row['set_hours']:>3} score {row['score_hours']:>3} "
                          f"FA {row['false_alarm_hours']:>3} (exp {'' if row['expected_if_alike'] is None else round(row['expected_if_alike'], 1)}) "
                          f"worst x{'' if row['worst_excess_ratio'] is None else round(row['worst_excess_ratio'], 2)} | swing80 {f(row['swing80'])} swing50 {f(row['swing50'])}")
