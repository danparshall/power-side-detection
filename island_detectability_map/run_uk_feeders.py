"""The logged 16-H100 job added to UK low-voltage feeders built from IDEAL homes, with the
power-side report's detector, folds and scoring (run_task3). Writes uk_feeders.json and
uk_feeders_tables.md (the same rows as tables).
The window, folds, home selection, feeder sizes and scales were
fixed before any injection was scored. A dry run on 12 homes then showed the report's rule setting
aside the whole band, and the "no bins set aside" variant was added; nothing was changed after
that dry run's rates were seen.

EXPLORATORY orchestration. Tested pieces: ideal.* and fetch_ideal.window_rows (tests/test_ideal.py);
jumps, usable_hours, rate_at (tests/test_uk_feeders.py); run_pulse.experiment, phases.imbalance,
nlr.*. run_task3.score_stage is reused as it stands and its figures are asserted equal to this
module's own scoring.

Background: one-second whole-home mains of IDEAL homes (Edinburgh and the Lothians), 56 days from
2018-03-05 00:00 UTC. Clamp readings in watts at an assumed voltage, summed as kVA.
* Study-wide outages (more than half the homes empty in the same second) and the dataset's own
  bad hour (2018-04-17 08:50-09:50 UTC) are left empty; an hour touching them is not used.
* A home is dropped if more than 2% of its non-outage seconds lie in gaps longer than 10 s.
  Every remaining gap is filled by holding the last reading, and the filled share is reported.
* Feeders: the kept homes in one seeded random order; a feeder of N homes is the first N, dealt
  onto three phases in turn (UK practice: each home on one phase). N = 12, 24, 48, 96 and all
  (largest multiple of 12).
Signal: NLR Llama-2 fine-tune (GPU one-second means), 1 a second, looped from a random start, times
a scale (1 = as logged, 2.45 kW swing). Three-phase island -> added to the sum. Single-phase load
-> added to one phase, looked for in that phase minus the mean of the other two. A UK home supply
cannot carry the whole island on one phase, so the single-phase rows are also read at scales 0.5
and 0.25 (8 and 4 chips).
Detector: run_task3's (spectral, one-hour windows, slow and fast band settings), in two variants.
"as in the report" sets aside every bin whose worst learning hour is 10 times its median or more
(the report's rule for a site's own intermittent lines). On small residential feeders that rule can
set aside the whole band, and then the detector cannot run; this is recorded, not worked around.
"no bins set aside" keeps every bin, so the threshold alone has to absorb the busy hours.
Folds: "first two weeks" = the first 336 hours; "eight weeks" = all 1,344. In each, fold A learns
on the usable hours of the first half and is scored on the second (time order); fold B the reverse.
Threshold = worst no-signal learning hour (one per family) x margin.
Turn-on count: jumps() on one-minute means of the three-phase total.

Usage: python -m island_detectability_map.run_uk_feeders IDEAL_DIR NLR_DIR OUT_DIR [build|stages|report]

Origin: verification@776d1246
"""
from __future__ import annotations

import json, pickle, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from island_detectability_map import ideal, run_nlr_inject as NI, run_pulse as R, run_task3 as T

SEED = 20261008
START = "2018-03-05T00:00:00"
DAYS = 56
SPANS = {"first two weeks": 336, "eight weeks": 1344}
LEVELS = [12, 24, 48, 96]
HOLD_S = 10
MAX_OWN_LONG_GAP = 0.02
OUTAGE_SHARE = 0.5
MARGINS = [1.0, 2.0]
EXTRA_SCALES = [0.5, 0.25]
VARIANTS = {"as in the report": {}, "no bins set aside": {"notch_ratio": np.inf}}
ISLAND_KW = 27.0
STEPS_KW = [25.0, 12.5]


def jumps(x, w: int, step: float) -> int:
    """Count rises of at least `step` in (mean of the next w samples) - (mean of the previous w).
    Exceedances within w samples of each other are one event; a window touching NaN is skipped."""
    x = np.asarray(x, dtype=np.float64)
    bad = np.cumsum(np.r_[0, ~np.isfinite(x)])
    c = np.cumsum(np.r_[0.0, np.nan_to_num(x)])
    tt = np.arange(w, len(x) - w + 1)
    diff = (c[tt + w] - c[tt]) / w - (c[tt] - c[tt - w]) / w
    diff[(bad[tt + w] - bad[tt - w]) > 0] = -np.inf
    ev, last = 0, -10 ** 9
    for i in np.flatnonzero(diff >= step):
        ev += i - last > w
        last = i
    return int(ev)


def usable_hours(x) -> list:
    """Indices of the whole hours in which every series has a value at every second."""
    n_h = x.shape[1] // 3600
    ok = np.isfinite(x[:, :n_h * 3600]).all(axis=0).reshape(n_h, 3600).all(axis=1)
    return [int(i) for i in np.flatnonzero(ok)]


def rate_at(rates, scales, scale: float) -> float:
    return float(np.interp(np.log(scale), np.log(scales), rates))


def grids(ideal_dir: Path):
    return {int(f.stem[4:]): f for f in sorted((ideal_dir / "grid").glob("home*.npy"))}


def build(ideal_dir: Path):
    """Outage mask, home selection, nested feeders. Writes IDEAL_DIR/feeders/level_<N>.npy and build.json."""
    files = grids(ideal_dir)
    n = DAYS * 86400
    count = np.zeros(n, dtype=np.int32)
    for f in files.values():
        count += np.load(f) == ideal.MISSING
    outage = ideal.outage_mask(count, len(files), OUTAGE_SHARE)
    t0 = np.datetime64(START, "s")
    a, b = [int((np.datetime64(v, "s") - t0).astype(np.int64)) for v in ideal.BAD_SPAN_UTC]
    empty = outage.copy(); empty[a:b] = True
    share = {h: ideal.own_long_gap_share(np.load(f), outage, HOLD_S) for h, f in files.items()}
    kept = [h for h in files if share[h] <= MAX_OWN_LONG_GAP]
    order = ideal.shuffled(kept, SEED)
    n_all = len(order) // 12 * 12
    levels = [v for v in LEVELS if v < n_all] + [n_all]
    out = ideal_dir / "feeders"; out.mkdir(exist_ok=True)
    kva = np.zeros((3, n)); phase_of = {}; filled_short = filled_long = 0; info = {}
    for i in range(0, n_all, 12):
        chunk = order[i:i + 12]
        ser = {}
        for h in chunk:
            g = np.load(files[h])
            ser[h], _ = ideal.fill(g, back_fill=True)
            miss = (g == ideal.MISSING) & ~empty
            long = miss.sum() - _short_gap_seconds(g, empty)
            filled_long += int(long); filled_short += int(miss.sum() - long)
        k, ph = ideal.feeder(ser, chunk)
        kva += k; phase_of.update(ph)
        if i + 12 in levels:
            x = kva.copy(); x[:, empty] = np.nan
            np.save(out / f"level_{i + 12}.npy", x)
            used = (~empty).sum() * (i + 12)
            info[i + 12] = {"home_ids": order[:i + 12], "filled_share_gaps_up_to_10s": filled_short / used,
                            "filled_share_longer_gaps": filled_long / used}
    res = {"start_utc": START, "days": DAYS, "homes_with_a_file": len(files), "homes_kept": len(kept),
           "homes_dropped_for_own_gaps": sorted(h for h in files if h not in kept),
           "own_long_gap_share_kept_max": max(share[h] for h in kept),
           "outage_seconds": int(outage.sum()), "outage_share": float(outage.mean()),
           "empty_seconds_with_bad_hour": int(empty.sum()), "levels": levels, "seed": SEED,
           "phase_of_home": {str(h): int(p) for h, p in phase_of.items()}, "level_info": {str(k): v for k, v in info.items()}}
    (out / "build.json").write_text(json.dumps(res, indent=1))
    return res


def _short_gap_seconds(g, empty) -> int:
    """Missing non-outage seconds that lie in gaps of at most HOLD_S seconds."""
    miss = g == ideal.MISSING
    b = np.r_[False, miss, False]
    d = np.diff(b.astype(np.int8))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    short = np.zeros(len(g), dtype=bool)
    for a_, e_ in zip(s[(e - s) <= HOLD_S], e[(e - s) <= HOLD_S]):
        short[a_:e_] = True
    return int((short & ~empty).sum())


def folds(x, hours: int):
    ok = [h for h in usable_hours(x[:, :hours * 3600])]
    half = hours // 2
    a = [i for i, h in enumerate(ok) if h < half]; b = [i for i, h in enumerate(ok) if h >= half]
    return ok, {"A": (a, b), "B": (b, a)}


def stage(args):
    ideal_dir, nlr_dir, level, span, fold, band, variant = args
    cache = Path(ideal_dir) / "feeders" / "stage_cache"; cache.mkdir(exist_ok=True)
    f = cache / f"{level}__{span.replace(' ', '_')}__{fold}__{band}__{variant.replace(' ', '_')}.pkl"
    if f.exists():
        return str(f)
    x = np.load(Path(ideal_dir) / "feeders" / f"level_{level}.npy")
    ok, fd = folds(x, SPANS[span])
    learn, score = fd[fold]
    xc = np.concatenate([x[:, h * 3600:(h + 1) * 3600] for h in ok], axis=1)      # usable hours only, in order
    tr = NI.trace_kw(Path(nlr_dir), T.WORKLOAD, dict(NI.CASES)[T.WORKLOAD], 1.0)
    st = {"n_hours": len(ok), "hours": ok, "learn": learn, "score": score, "mean_kva": float(xc.sum(axis=0).mean()),
          "phase_mean_kva": [float(v) for v in xc.mean(axis=1)],
          "trace_swing_kw": float(np.percentile(tr, 95) - np.percentile(tr, 5)), "series": {}}
    for key, s in T.series_of(xc).items():
        try:
            e = R.experiment(s, 1.0, 3600, np.random.default_rng(SEED), char_hours=0, char_idx=learn, swings=T.SCALES,
                             make_signal=NI.make_signal_from(tr), return_null=True, return_stats=True,
                             **{**T.BANDS[band], **VARIANTS[variant]})
        except ValueError as err:                            # every bin of the band set aside: nothing left to maximise over
            if "zero-size array" not in str(err):
                raise
            st["series"][key] = None
            continue
        st["series"][key] = {"null": np.array(e["null_stats"]), "inj": np.array(e["inj_stats"]),
                             "set_aside": e["notched_bins"] / e["band_bins"]}
    f.write_bytes(pickle.dumps(st))
    return str(f)


def score(st):
    """One row per series family and margin; same rule as run_task3.score_stage, plus other scales."""
    rows = []
    for g, keys in T.GROUPS.items():
        if any(st["series"][k] is None for k in keys):
            rows.append({"group": g, "cannot_run": "the whole search band is set aside by the rule for a site's own lines",
                         "score_hours": len(st["score"]), "learn_hours": len(st["learn"])})
            continue
        nul = {k: st["series"][k]["null"] for k in keys}
        for m in MARGINS:
            thr = max(np.max(nul[k][st["learn"]]) for k in keys) * m
            fa = [i for i in st["score"] if any(nul[k][i] > thr for k in keys)]
            per = [[float(np.mean(st["series"][k]["inj"][j, st["score"]] > thr)) for j in range(len(T.SCALES))] for k in keys]
            s80 = [T.scale_for(r, 0.8) for r in per]
            rows.append({"group": g, "margin": m, "alarm_hours": len(fa), "score_hours": len(st["score"]),
                         "learn_hours": len(st["learn"]),
                         "rate_at_scale_1": [rate_at(r, T.SCALES, 1.0) for r in per],
                         **{f"rate_at_scale_{s}": [rate_at(r, T.SCALES, s) for r in per] for s in EXTRA_SCALES},
                         "scale_80_min_max": [min(s80), max(s80)] if all(v is not None for v in s80) else None,
                         "share_of_band_set_aside": [float(st["series"][k]["set_aside"]) for k in keys]})
    return rows


def report(ideal_dir: Path, out_dir: Path):
    fdir = ideal_dir / "feeders"
    b = json.loads((fdir / "build.json").read_text())
    res = {"design": __doc__.split("Background:")[1].split("Usage")[0].strip(), "build": b, "scales": T.SCALES,
           "island_kw": ISLAND_KW, "levels": {}, "rows": []}
    for level in b["levels"]:
        x = np.load(fdir / f"level_{level}.npy")
        tot = x.sum(axis=0)
        m1 = tot[: len(tot) // 60 * 60].reshape(-1, 60).mean(axis=1)                 # NaN if any second of the minute is empty
        weeks = float(np.isfinite(m1).sum() / (7 * 1440))
        d = {"homes": level, "mean_kva": float(np.nanmean(tot)), "phase_mean_kva": [float(v) for v in np.nanmean(x, axis=1)],
             "usable_hours": {s: len(usable_hours(x[:, :h * 3600])) for s, h in SPANS.items()},
             "island_as_share_of_mean": ISLAND_KW / float(np.nanmean(tot)), "weeks_of_one_minute_data": weeks,
             **b["level_info"][str(level)], "turn_on": {}}
        hr = tot[: len(tot) // 3600 * 3600].reshape(-1, 3600).mean(axis=1)
        d["hourly_mean_min_max_kva"] = [float(np.nanmin(hr)), float(np.nanmax(hr))]
        for step in STEPS_KW:
            for w in (5, 15, 60):
                n = jumps(m1, w, step)
                d["turn_on"][f"step {step} kW, {w} min"] = {"jumps": n, "per_week": n / weeks}
        res["levels"][str(level)] = d
    for f in sorted((fdir / "stage_cache").glob("*.pkl")):
        level, span, fold, band, variant = f.stem.split("__")
        st = pickle.loads(f.read_bytes())
        mine = score(st)
        if all(v is not None for v in st["series"].values()):                       # same figures as the report's scoring code
            theirs = [r for r in T.score_stage(st) if r["margin"] != "scored_hours_max"]
            for a, t in zip(mine, theirs):
                assert a["alarm_hours"] == t["alarm_hours"] and np.allclose(a["rate_at_scale_1"], t["rate_at_scale_1"]), (f.name, a, t)
        for r in mine:
            res["rows"].append({"homes": int(level), "span": span.replace("_", " "), "fold": fold, "band": band,
                                "variant": variant.replace("_", " "),
                                "mean_kva": st["mean_kva"], "usable_hours": st["n_hours"], "trace_swing_kw": st["trace_swing_kw"], **r})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "uk_feeders.json").write_text(json.dumps(res, indent=1))
    (out_dir / "uk_feeders_tables.md").write_text(tables(res))
    return res


def _pct(v):
    lo, hi = min(v), max(v)
    return f"{round(100 * lo)}%" if round(100 * lo) == round(100 * hi) else f"{round(100 * lo)}–{round(100 * hi)}%"


def tables(res) -> str:
    """The rows of uk_feeders.json as markdown tables, one cell per feeder size, band and margin.
    Cells: found in % of scored hours (false-alarm hours / scored hours). A range spans the three phases."""
    by = {(r["homes"], r["span"], r["fold"], r["band"], r["variant"], r["group"], r.get("margin")): r for r in res["rows"]}
    lv = sorted(int(k) for k in res["levels"])
    out = ["<!-- Generated by run_uk_feeders.py -->"]
    for span in SPANS:
        for variant in VARIANTS:
            for fold, name in (("A", "time order"), ("B", "reverse")):
                for group, scale, title in (("sum", "1", "three-phase island on the total, as logged"),
                                            ("resid", "1", "whole island on one phase (not possible on a UK home supply), on the phase unbalance"),
                                            ("resid", "0.5", "half island (8 chips) on one phase, on the phase unbalance"),
                                            ("resid", "0.25", "quarter island (4 chips) on one phase, on the phase unbalance")):
                    out.append(f"\n**{span}, {variant}, {name}: {title}.**\n")
                    out.append("| Homes | Mean load, kVA | Slow band, margin 1 | Slow band, margin 2 | Fast band, margin 1 | Fast band, margin 2 |")
                    out.append("|---|---|---|---|---|---|")
                    for n in lv:
                        cells, mean = [], None
                        for band in ("slow", "fast"):
                            for m in MARGINS:
                                r = by.get((n, span, fold, band, variant, group, m))
                                if r is None:
                                    c = by.get((n, span, fold, band, variant, group, None))
                                    cells.append("cannot run" if c else "not run")
                                    mean = mean or (c or {}).get("mean_kva")
                                    continue
                                mean = r["mean_kva"]
                                cells.append(f"{_pct(r['rate_at_scale_' + scale])} ({r['alarm_hours']}/{r['score_hours']})")
                        mean = res["levels"][str(n)]["mean_kva"] if mean is None else mean
                        out.append(f"| {n} | {mean:.1f} | " + " | ".join(cells) + " |")
            out.append(f"\n**{span}, {variant}: share of the logged swing needed to be found in 80% of scored hours "
                       "(time order; range over phases).**\n")
            out.append("| Homes | Total, slow, margin 1 | Total, slow, margin 2 | Total, fast, margin 1 | Total, fast, margin 2 "
                       "| One phase, slow, margin 2 | One phase, fast, margin 2 |")
            out.append("|---|---|---|---|---|---|---|")
            for n in lv:
                def s80(group, band, m):
                    r = by.get((n, span, "A", band, variant, group, m))
                    if r is None:
                        return "cannot run"
                    v = r["scale_80_min_max"]
                    return "not reached" if v is None else (f"{v[0]:.2f}" if abs(v[0] - v[1]) < 0.005 else f"{v[0]:.2f}–{v[1]:.2f}")
                out.append(f"| {n} | {s80('sum', 'slow', 1.0)} | {s80('sum', 'slow', 2.0)} | {s80('sum', 'fast', 1.0)} | {s80('sum', 'fast', 2.0)} "
                           f"| {s80('resid', 'slow', 2.0)} | {s80('resid', 'fast', 2.0)} |")
    out.append("\n**Feeders.**\n")
    out.append("| Homes | Mean load, kVA | Lowest and highest hourly mean, kVA | Island (27 kW) as a share of the mean | Usable hours, two weeks / eight weeks "
               "| Natural 25 kW jumps a week (15-minute means) | Natural 12.5 kW jumps a week | Home-seconds filled: gaps to 10 s / longer |")
    out.append("|---|---|---|---|---|---|---|---|")
    for n in lv:
        d = res["levels"][str(n)]
        t = d["turn_on"]
        out.append(f"| {n} | {d['mean_kva']:.1f} | {d['hourly_mean_min_max_kva'][0]:.1f} – {d['hourly_mean_min_max_kva'][1]:.1f} | {d['island_as_share_of_mean']:.2f} "
                   f"| {d['usable_hours']['first two weeks']} / {d['usable_hours']['eight weeks']} "
                   f"| {t['step 25.0 kW, 15 min']['per_week']:.1f} ({t['step 25.0 kW, 15 min']['jumps']} in {d['weeks_of_one_minute_data']:.1f} weeks) "
                   f"| {t['step 12.5 kW, 15 min']['per_week']:.1f} | {100 * d['filled_share_gaps_up_to_10s']:.1f}% / {100 * d['filled_share_longer_gaps']:.2f}% |")
    return "\n".join(out) + "\n"


def main(ideal_dir: Path, nlr_dir: Path, out_dir: Path, what: str = "all"):
    if what in ("build", "all"):
        b = build(ideal_dir)
        print({k: b[k] for k in ("homes_with_a_file", "homes_kept", "outage_share", "levels")}, flush=True)
    if what in ("stages", "all"):
        b = json.loads((ideal_dir / "feeders" / "build.json").read_text())
        jobs = [(str(ideal_dir), str(nlr_dir), lv, sp, fo, ba, va) for sp in SPANS for lv in b["levels"] for fo in "AB"
                for ba in T.BANDS for va in VARIANTS]
        with ProcessPoolExecutor(2) as ex:
            for f in ex.map(stage, jobs):
                print(Path(f).name, flush=True)
    if what in ("report", "all"):
        return report(ideal_dir, out_dir)


if __name__ == "__main__":
    a = sys.argv
    r = main(Path(a[1]), Path(a[2]), Path(a[3]), a[4] if len(a) > 4 else "all")
    if r:
        for row in r["rows"]:
            if row["fold"] == "A" and "cannot_run" in row:
                print(f"{row['homes']:>4} homes {row['mean_kva']:6.1f} kVA | {row['span']:<15} {row['band']:<4} {row['variant']:<17} {row['group']:<5} cannot run")
            elif row["fold"] == "A":
                print(f"{row['homes']:>4} homes {row['mean_kva']:6.1f} kVA | {row['span']:<15} {row['band']:<4} {row['variant']:<17} {row['group']:<5} x{row['margin']:<3} "
                      f"| alarms {row['alarm_hours']:>3}/{row['score_hours']:<4} | rate@1 {[round(v, 2) for v in row['rate_at_scale_1']]} "
                      f"@0.5 {[round(v, 2) for v in row['rate_at_scale_0.5']]} @0.25 {[round(v, 2) for v in row['rate_at_scale_0.25']]}")
