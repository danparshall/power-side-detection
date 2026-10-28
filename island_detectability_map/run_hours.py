"""Hour-of-day split of the current-monitor detectors on the LBNL bus.
Writes hour_split.json.

EXPLORATORY orchestration. The tested pieces are hours.hour_labels / class_rate /
swing_for / split_control (tests/test_hours.py) and run_pulse.experiment's char_idx and
return_stats arguments. Everything else here wires existing detectors to per-class
thresholds and has no unit tests.

Question: every detection figure so far uses one threshold set by the worst of the 39 test
hours. Do quiet hours do better if each class of hours gets its own threshold?

Design
* Classes: 'work' = weekday 07:00-19:00 local (America/Los_Angeles); 'quiet' = the rest.
  'quiet' is reported whole and split into 'weeknight' and 'weekend', because in this
  record the test half's quiet hours are mostly one Saturday and the characterisation
  half has no weekend.
* Spectral and edge-count detectors: same signal model, seed and settings as
  run_phases / run_phase_edges, so the all-hours case must reproduce
  phase_measurement.json and phase_edges.json. Injections run once over every test
  window; class thresholds are applied afterwards, so classes see the same injections.
  Two variants: 'threshold only' (baseline from all 36 characterisation hours) and
  'own baseline' (baseline from that class's characterisation hours only).
* Two passes. Coarse: the run_phases / run_phase_edges swing grids, for the reproduction check and the
  'own baseline' comparison. Fine: a swing grid in steps of 2^(1/4), all-hours baseline,
  which is what the per-class figures and the control are read from.
* Control: random subsets of the same size as each class, own worst-window threshold.
  A class beats 'all hours' for a real reason only if it also beats this control.
* Restart edges and notches (1-s data, whole record): look-alike rates per class,
  quoted per 24 h of that class.

Each expensive stage is cached under DATA_DIR/hour_split_cache/ (gitignored) so a run that
is cut off resumes where it stopped; delete that folder to recompute. STAGE runs one
stage only (series, spectral, spectral_fine, edge, edge_fine); omit it to run all and
write the JSON.

Usage: python -m island_detectability_map.run_hours ../data/upmu OUT_DIR [STAGE]

Origin: verification@67698f42
"""
from __future__ import annotations

import json, pickle, sys
from pathlib import Path

import numpy as np

from island_detectability_map import hours, phases, pulse, run_pulse as R, run_sync as S
from island_detectability_map import run_phases as P, run_phase_edges as E, run_dips as D

SEED = P.SEED
LEVELS = (0.5, 0.8)
N_DRAWS = 2000
FINE = [float(v) for v in np.geomspace(0.5, 32.0, 25)]   # ratio 2^(1/4). The run_phases grids step by 2x, and both
#   detectors go from ~20% to ~95% detection inside one such step, so they cannot resolve a gain smaller than 2x.
STEP_SIZES = [2.0, 3.0, 5.0, 10.0, 20.0]
NOTCH_DURATIONS = [5, 30, 120, 600]
NOTCH_DEPTHS = [10.0, 20.0]
REF_KW = dict(whitelist=P.HARMONICS, band=(0.08, 5.5), f0_range=(0.1, 3.0), seg_s=100.0, swings=P.SWINGS)
MK = lambda n, fs_, f0, swing, rng_: pulse.training_trace(n, fs_, f0, swing, rng=rng_, sigma_log=0.1, rho=0.95, straggle=1.0)
FAMILIES = {"sum": ["sum"], "phase": ["phase1", "phase2", "phase3"], "imbalance": ["imbalance1", "imbalance2", "imbalance3"]}


def cached(cache: Path, name: str, fn):
    f = cache / f"{name}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    v = fn()
    cache.mkdir(parents=True, exist_ok=True)
    f.write_bytes(pickle.dumps(v))
    return v


def family_swings(stats, fam, swings, idx_by_key, level):
    """Unknown-phase rule: one threshold = worst no-signal window over the family's series
    (each over its own usable windows in the class); swing per series. Returns the list."""
    keys = FAMILIES[fam]
    thr = max(float(np.max(stats[k][0][idx_by_key[k]])) for k in keys)
    return [hours.swing_for(stats[k][0], stats[k][1], swings, idx_by_key[k], level, thr) for k in keys]


def family_control(stats, fam, swings, n_windows, size, level, rng):
    """Random subsets of `size` test windows; worst (largest) swing across the family."""
    keys = FAMILIES[fam]
    out = np.full(N_DRAWS, np.nan)
    for d in range(N_DRAWS):
        idx = rng.choice(n_windows, size=size, replace=False)
        v = family_swings(stats, fam, swings, {k: idx for k in keys}, level)
        if all(x is not None for x in v):
            out[d] = max(v)
    return out


def summarise(v):
    v = [x for x in v if x is not None]
    return [float(min(v)), float(max(v))] if v else None


def spectral(ser12, fx, lab2_char, baselines=("all", "work", "quiet"), swings=None):
    kw = dict(REF_KW, swings=swings) if swings else REF_KW
    stats = {}
    for b in baselines:
        cidx = None if b == "all" else [i for i, c in enumerate(lab2_char) if c == b]
        stats[b] = {}
        for key, x in ser12.items():
            e = R.experiment(x, fx, 3600, np.random.default_rng(SEED), return_null=True, return_stats=True,
                             char_idx=cidx, make_signal=MK, **kw)
            stats[b][key] = (np.array(e["null_stats"]), e["inj_stats"])
    return stats


def edge_stats(x, fx, cidx, swings=None):
    """run_phase_edges.edge_null / edge_rates with per-window z kept. Returns
    (clean test positions 0..38, null z over clean, injected z [swings x clean])."""
    n = int(3600 * fx)
    wins = [x[i:i + n] for i in range(0, len(x) - n + 1, n)]
    c = np.array([E.counts(w) for w in wins])
    med = np.median(c[cidx], axis=0)
    mad = 1.4826 * np.median(np.abs(c[cidx] - med), axis=0) + 1.0
    z = lambda cc: float(np.max((cc - med) / mad))
    test = list(range(36, len(wins)))
    null_all = {i: z(c[i]) for i in test}
    clean = [i for i in test if null_all[i] <= 100.0]
    name, f0r, kw = E.REF_MODEL
    rng = np.random.default_rng(SEED)
    inj = []
    for swing in (swings or E.SWINGS):
        row = []
        for i in clean:
            f0 = float(np.exp(rng.uniform(np.log(f0r[0]), np.log(f0r[1]))))
            row.append(z(E.counts(wins[i] + pulse.training_trace(n, fx, f0, swing, rng=rng, **kw))))
        inj.append(row)
    return np.array([i - 36 for i in clean]), np.array([null_all[i] for i in clean]), np.array(inj)


def events_with_times(x1):
    t, st = S.upsteps(x1)
    return {a: S.events(t, st, a) for a in STEP_SIZES}


def restart_recovery(x1, depth):
    """run_phase_edges.restart for one depth, keeping the restart times."""
    rng = np.random.default_rng(SEED)
    for d in E.DEPTHS:                     # same rng path as run_phase_edges.restart: a fresh rng per depth
        if d == depth:
            break
    y = x1.copy(); restarts = []; pos = float(rng.uniform(0, 600))
    while pos + S.PAUSE_S + 10 < len(y):
        a, b = int(pos), int(pos + S.PAUSE_S)
        y[a:b] -= depth; restarts.append(b)
        pos = b + 500.0 * (1 + 0.1 * rng.standard_normal())
    t2, s2 = S.upsteps(y)
    ev = S.events(t2, s2, 0.75 * depth)
    hit = np.array([bool(np.any(np.abs(ev - r) <= 2)) for r in restarts])
    return np.array(restarts), hit


def notch_times(x, d, w):
    """run_dips.notch_events, returning (start second, depth) per event."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    t = np.arange(w, len(x) - d - w)
    pre = (c[t] - c[t - w]) / w
    dur = (c[t + d] - c[t]) / d
    post = (c[t + d + w] - c[t + d]) / w
    depth = 0.5 * (pre + post) - dur
    ok = np.abs(pre - post) <= 0.5 * np.maximum(depth, 1e-9)
    depth = np.where(ok, depth, 0.0)
    out, i, n = [], 0, len(depth)
    while i < n:
        j = min(n, i + d)
        k = i + int(np.argmax(depth[i:j]))
        if depth[k] > 0 and depth[k] >= depth[max(0, k - d):min(n, k + d)].max():
            out.append((int(t[k]), float(depth[k])))
            i = k + d
        else:
            i = j
    return np.array(out).reshape(-1, 2)


def load_series(data_dir: Path):
    z = np.load(Path(data_dir) / "grizzly_head6.npz")
    t0_ns = int(z["t0_ns"]); del z
    x120, fs = phases.load_phases(data_dir, voltage="L1", fname="grizzly_head6.npz")
    x12 = np.vstack([R.block_mean(r, R.DEC) for r in x120])
    x1 = np.vstack([R.block_mean(r, int(fs)) for r in x120])
    return t0_ns, fs / R.DEC, x12, x1


def main(data_dir: Path, out_dir: Path, stage: str | None = None):
    cache = Path(data_dir) / "hour_split_cache"
    t0_ns, fx, x12, x1 = cached(cache, "series", lambda: load_series(data_dir))
    ser12, ser1 = P.series_set(x12), P.series_set(x1)
    n_hours = x12.shape[1] // int(3600 * fx)
    lab = hours.hour_labels(t0_ns, n_hours + 1)
    lab2 = [hours.two_class(v) for v in lab]
    lab_test, lab2_test = lab[36:n_hours], lab2[36:n_hours]
    base_idx = lambda b: list(range(36)) if b == "all" else [i for i, c in enumerate(lab2[:36]) if c == b]
    stages = {
        "spectral": lambda: spectral(ser12, fx, lab2[:36]),
        "spectral_fine": lambda: spectral(ser12, fx, lab2[:36], baselines=("all",), swings=FINE)["all"],
        "edge": lambda: {b: {k: edge_stats(v, fx, base_idx(b)) for k, v in ser12.items()} for b in ("all", "work", "quiet")},
        "edge_fine": lambda: {k: edge_stats(v, fx, base_idx("all"), FINE) for k, v in ser12.items()},
    }
    if stage:
        if stage != "series":
            cached(cache, stage, stages[stage])
        return None
    classes = {"all": list(range(n_hours - 36)),
               "work": [i for i, c in enumerate(lab2_test) if c == "work"],
               "quiet": [i for i, c in enumerate(lab2_test) if c == "quiet"],
               "weeknight": [i for i, c in enumerate(lab_test) if c == "weeknight"],
               "weekend": [i for i, c in enumerate(lab_test) if c == "weekend"]}
    hourly = R.block_mean(ser12["sum"], int(3600 * fx))
    res = {"source": "LBNL Open uPMU grizzly_bus1_2, 75.8 h from 2015-10-01 00:00 UTC", "seed": SEED,
           "class_rule": "work = weekday 07:00-19:00 America/Los_Angeles; quiet = the rest (weeknight + weekend)",
           "hours": {"characterisation": {c: lab[:36].count(c) for c in ("work", "weeknight", "weekend")},
                     "test": {c: lab_test.count(c) for c in ("work", "weeknight", "weekend")}},
           "mean_kva_by_class": {c: float(np.mean([hourly[i] for i in range(n_hours) if lab[i] == c]))
                                 for c in ("work", "weeknight", "weekend")},
           "signal_model": "sawtooth, 10% spread, rho 0.95 (map reference)"}

    # ---- 1. spectral detector
    st = cached(cache, "spectral", stages["spectral"])
    stf = cached(cache, "spectral_fine", stages["spectral_fine"])
    swings = P.SWINGS
    spec = {"null_by_class": {}, "rows": [], "rows_fine": [], "control": []}
    for c, idx in classes.items():
        nl = st["all"]["sum"][0][idx]
        spec["null_by_class"][c] = {"n_hours": len(idx), "median": float(np.median(nl)),
                                    "p90": float(np.quantile(nl, 0.9)), "max": float(nl.max())}
    for c, idx in classes.items():
        for variant, b in (("threshold only", "all"), ("own baseline", c if c in ("work", "quiet") else None)):
            if b is None or (c == "all" and variant == "own baseline"):
                continue
            row = {"class": c, "n_hours": len(idx), "variant": variant}
            for fam in FAMILIES:
                for lvl in LEVELS:
                    v = family_swings(st[b], fam, swings, {k: idx for k in FAMILIES[fam]}, lvl)
                    row[f"{fam}_swing{int(lvl*100)}"] = summarise(v)
            spec["rows"].append(row)
    for c, idx in classes.items():
        row = {"class": c, "n_hours": len(idx), "variant": "threshold only, fine grid"}
        for fam in FAMILIES:
            for lvl in LEVELS:
                row[f"{fam}_swing{int(lvl*100)}"] = summarise(family_swings(stf, fam, FINE, {k: idx for k in FAMILIES[fam]}, lvl))
            row[f"{fam}_rates"] = {k: [hours.class_rate(stf[k][0], stf[k][1][j], idx,
                                                        max(float(np.max(stf[q][0][idx])) for q in FAMILIES[fam]))
                                       for j in range(len(FINE))] for k in FAMILIES[fam]}
        spec["rows_fine"].append(row)
    spec["fine_swings_kva"] = FINE
    rng = np.random.default_rng(SEED + 7)
    for size in sorted({len(classes[c]) for c in ("work", "quiet", "weeknight", "weekend")}):
        for fam in FAMILIES:
            ctl = family_control(stf, fam, FINE, len(classes["all"]), size, 0.8, rng)
            spec["control"].append({"size": size, "family": fam, "level": 0.8, "n_draws": N_DRAWS,
                                    "q05_q50_q95": [float(np.nanquantile(ctl, q)) for q in (0.05, 0.5, 0.95)],
                                    "share_never_reaching_level": float(np.mean(np.isnan(ctl))),
                                    "draws": [float(v) for v in np.sort(ctl[~np.isnan(ctl)])[::20]]})
    res["spectral"] = spec

    # ---- 2. edge-count detector
    edge = {"rows": [], "control": []}
    es = cached(cache, "edge", stages["edge"])

    es["fine"] = cached(cache, "edge_fine", stages["edge_fine"])

    def edge_family(b, fam, cls_idx, lvl, subset=None):
        keys = FAMILIES[fam]
        sw = FINE if b == "fine" else E.SWINGS
        use = {}
        for k in keys:
            pos, null, inj = es[b][k]
            m = np.isin(pos, cls_idx if subset is None else subset)
            use[k] = np.where(m)[0]
        if any(len(u) == 0 for u in use.values()):
            return None, {k: 0 for k in keys}
        thr = max(float(np.max(es[b][k][1][use[k]])) for k in keys)
        v = [hours.swing_for(es[b][k][1], es[b][k][2], sw, use[k], lvl, thr) for k in keys]
        return v, {k: int(len(use[k])) for k in keys}

    for c, idx in classes.items():
        for variant, b in (("threshold only", "all"), ("own baseline", c if c in ("work", "quiet") else None)):
            if b is None or (c == "all" and variant == "own baseline"):
                continue
            row = {"class": c, "variant": variant}
            for fam in FAMILIES:
                for lvl in LEVELS:
                    v, nuse = edge_family(b, fam, idx, lvl)
                    row[f"{fam}_swing{int(lvl*100)}"] = summarise(v) if v else None
                row[f"{fam}_clean_hours"] = nuse
            edge["rows"].append(row)
    edge["rows_fine"] = []
    for c, idx in classes.items():
        row = {"class": c, "variant": "threshold only, fine grid"}
        for fam in FAMILIES:
            for lvl in LEVELS:
                v, nuse = edge_family("fine", fam, idx, lvl)
                row[f"{fam}_swing{int(lvl*100)}"] = summarise(v) if v else None
            row[f"{fam}_clean_hours"] = nuse
        edge["rows_fine"].append(row)
    rng = np.random.default_rng(SEED + 8)
    for cname in ("work", "quiet", "weeknight", "weekend"):
        for fam in FAMILIES:
            # same number of usable hours as the class has for this family's first series
            n_use = len(np.where(np.isin(es["fine"][FAMILIES[fam][0]][0], classes[cname]))[0])
            pool = es["fine"][FAMILIES[fam][0]][0]
            ctl = []
            for d in range(N_DRAWS):
                sub = rng.choice(pool, size=n_use, replace=False)
                v, _ = edge_family("fine", fam, None, 0.8, subset=sub)
                ctl.append(max(v) if v and all(x is not None for x in v) else np.nan)
            ctl = np.array(ctl)
            edge["control"].append({"matched_to": cname, "size": n_use, "family": fam, "level": 0.8, "n_draws": N_DRAWS,
                                    "q05_q50_q95": [float(np.nanquantile(ctl, q)) if not np.all(np.isnan(ctl)) else None for q in (0.05, 0.5, 0.95)],
                                    "share_never_reaching_level": float(np.mean(np.isnan(ctl)))})
    res["edge_count"] = edge

    # ---- 3. restart edges and notches by class (1-s data, whole record)
    sec_lab = np.array([lab[min(i // 3600, len(lab) - 1)] for i in range(x1.shape[1])])
    cls_hours = {c: float(np.sum(sec_lab == c)) / 3600.0 for c in ("work", "weeknight", "weekend")}
    cls_hours["quiet"] = cls_hours["weeknight"] + cls_hours["weekend"]
    cls_hours["all"] = x1.shape[1] / 3600.0
    def member(tt, c):
        tt = np.asarray(tt, dtype=int)          # an empty event list comes back as float
        if c == "all":
            return np.ones(len(tt), bool)
        return (sec_lab[tt] != "work") if c == "quiet" else (sec_lab[tt] == c)
    rest = {"class_hours": cls_hours, "upsteps_per_24h": {}, "recovery_5kva": {}}
    for key, v in ser1.items():
        ev = events_with_times(v)
        rest["upsteps_per_24h"][key] = {c: {str(a): float(member(ev[a], c).sum() / cls_hours[c] * 24) for a in STEP_SIZES}
                                        for c in cls_hours}
        rs, hit = restart_recovery(v, 5.0)
        ev_thr = S.events(*S.upsteps(v), 0.75 * 5.0)
        rest["recovery_5kva"][key] = {c: {"restarts": int(member(rs, c).sum()),
                                          "recovered": float(hit[member(rs, c)].mean()) if member(rs, c).any() else None,
                                          "lookalikes_per_24h_at_threshold": float(member(ev_thr, c).sum() / cls_hours[c] * 24)}
                                      for c in cls_hours}
    res["restart_edges"] = rest
    notch = {}
    for d in NOTCH_DURATIONS:
        w = max(5, min(d, 60))
        ev = notch_times(ser1["sum"], d, w)
        tt, dep = ev[:, 0].astype(int), ev[:, 1]
        notch[str(d)] = {c: {str(a): float((member(tt, c) & (dep >= a)).sum() / cls_hours[c] * 24) for a in NOTCH_DEPTHS}
                         for c in cls_hours}
    res["notches_per_24h_sum"] = notch
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hour_split.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    r = main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else None)
    if r is None:
        sys.exit(0)
    f = lambda v: "-" if v is None else (f"{v[0]:.1f}" if abs(v[0] - v[1]) < 0.05 else f"{v[0]:.1f}-{v[1]:.1f}")
    print("hours:", r["hours"], "| mean kVA:", {k: round(v) for k, v in r["mean_kva_by_class"].items()})
    print("spectral null (sum, all-hours baseline):", {k: (v["n_hours"], round(v["median"], 2), round(v["p90"], 2), round(v["max"], 2)) for k, v in r["spectral"]["null_by_class"].items()})
    for name in ("spectral", "edge_count"):
        print("##", name, "- 80% swing kVA (50% in brackets): sum | one phase | imbalance residual")
        for row in r[name]["rows"] + r[name]["rows_fine"]:
            print(f"  {row['class']:<10}{row['variant']:<15}", " | ".join(f"{f(row[f'{fam}_swing80'])} ({f(row[f'{fam}_swing50'])})" for fam in FAMILIES),
                  row.get("sum_clean_hours", ""))
        for c in r[name]["control"]:
            print("   control", {k: v for k, v in c.items() if k not in ("draws", "n_draws", "level")})
    print("## up-steps per 24 h of class:", r["restart_edges"]["class_hours"])
    for key, v in r["restart_edges"]["upsteps_per_24h"].items():
        print(f"  {key:<11}", {c: {a: round(n) for a, n in d.items()} for c, d in v.items()})
    for key, v in r["restart_edges"]["recovery_5kva"].items():
        print(f"  rec {key:<11}", {c: (d["restarts"], d["recovered"] and round(d["recovered"], 2), round(d["lookalikes_per_24h_at_threshold"])) for c, d in v.items()})
    print("## notches per 24 h of class (sum):")
    for d, v in r["notches_per_24h_sum"].items():
        print(f"  {d:>4} s", {c: {a: round(n, 1) for a, n in dd.items()} for c, dd in v.items()})
