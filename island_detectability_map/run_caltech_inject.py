"""Injection run on the Caltech 1-second data (follow-on to caltech.py
and run_caltech_levels). Writes caltech_injection.json.

EXPLORATORY orchestration, no tests of its own. Reuses run_pulse.experiment (spectral
detector, one-hour windows, zero false alarms over the test hours), the map-reference signal
(sawtooth, 10% spread; pulse.training_trace returns exact per-sample averages, i.e. it models
a meter that averages over each second), hours.* for per-class thresholds and the random-split
control, and the unknown-phase rule of run_phases (one threshold = worst no-signal hour over
the three phases or the three residuals).

What differs from the LBNL runs: 1 sample a second, so cadences stop below 0.5 Hz. Two bands:
  slow  20-100 mHz, 500 s segments, search 0.016-0.3 Hz (run_pulse's slow cadence band)
  fast  0.1-0.4 Hz, 100 s segments, search 0.08-0.5 Hz
The first half of each record characterises the site; the second half is the test period.
The LBNL bus goes through the same 1-second pipeline as a cross-check against run_hours.
Also: recovery of injected restart edges (run_sync model) on the longest gap-free stretch.

Usage: python -m island_detectability_map.run_caltech_inject CALTECH_ROOT NETWORK_JSON UPMU_DIR OUT_DIR START END [site,...]
(with a site list: compute and cache those sites only; without: assemble the JSON)

Origin: verification@1ddddb46
"""
from __future__ import annotations

import json, pickle, sys
from pathlib import Path

import numpy as np

from island_detectability_map import caltech, hours, pulse, run_pulse as R, run_sync as S
from island_detectability_map import run_caltech_levels as L, run_hours as H

SEED = 20261002
SWINGS = [float(v) for v in np.geomspace(0.25, 64.0, 25)]          # steps of 2^(1/3)
BANDS = {"slow 20-100 mHz": dict(f0_range=(0.02, 0.1), seg_s=500.0, band=(0.016, 0.3), notch_ratio=10.0),   # = run_pulse CADENCE_BANDS
         "fast 0.1-0.4 Hz": dict(f0_range=(0.1, 0.4), seg_s=100.0, band=(0.08, 0.5), notch_ratio=10.0)}
SITES = {"egauge_19": "Caltech 16.5 kV line, 5.5 MVA", "egauge_17": "Caltech 2.4 kV side, 1.2 MVA",
         "egauge_12": "Caltech building main, 0.25 MVA", "lbnl": "LBNL grizzly_bus1_2, 0.7 MVA (1-s means of the 120/s data)"}
FAM = {"sum": ["sum"], "phase": ["phase_a", "phase_b", "phase_c"], "resid": ["resid_a", "resid_b", "resid_c"]}
DEPTHS = [5.0, 10.0, 20.0]
N_DRAWS = 2000


def load_site(site, root, net, upmu, start, end):
    if site == "lbnl":
        t0_ns, fx, x12, x1 = pickle.loads((Path(upmu) / "hour_split_cache" / "series.pkl").read_bytes())
        n_h = x1.shape[1] // 3600
        return x1[:, :n_h * 3600], hours.hour_labels(t0_ns, n_h)
    d = caltech.load_element(Path(root) / site / "magnitudes", caltech.meter_map(net)[site], start, end)
    n_h = d.kva.shape[1] // 3600
    return d.kva[:, :n_h * 3600], caltech.hour_labels(start, n_h)


def site_stats(kva, lab):
    n_h = kva.shape[1] // 3600
    char_h = n_h // 2
    ser = L.series_of(kva)
    out = {"n_hours": n_h, "char_hours": char_h, "labels_test": lab[char_h:n_h], "bands": {}, "restart": {}}
    for bname, kw in BANDS.items():
        out["bands"][bname] = {}
        for key, x in ser.items():
            mk = lambda n, fs_, f0, swing, rng_: pulse.training_trace(n, fs_, f0, swing, rng=rng_, sigma_log=0.1, rho=0.95, straggle=1.0)
            e = R.experiment(x, 1.0, 3600, np.random.default_rng(SEED), char_hours=char_h, swings=SWINGS,
                             make_signal=mk, return_null=True, return_stats=True, **kw)
            out["bands"][bname][key] = (np.array(e["null_stats"]), e["inj_stats"], e["notched_bins"], e["band_bins"])
    # restart edges on the longest gap-free stretch
    mask = np.isfinite(kva).all(axis=0)
    a, b = max(L.runs(mask), key=lambda r: r[1] - r[0])
    days = (b - a) / 86400.0
    for key, x in ser.items():
        seg = x[a:b]
        out["restart"][key] = {}
        for depth in DEPTHS:
            rng = np.random.default_rng(SEED)
            y = seg.copy(); restarts = []; pos = float(rng.uniform(0, 600))
            while pos + S.PAUSE_S + 10 < len(y):
                i0, i1 = int(pos), int(pos + S.PAUSE_S)
                y[i0:i1] -= depth; restarts.append(i1)
                pos = i1 + 500.0 * (1 + 0.1 * rng.standard_normal())
            ev = S.events(*S.upsteps(y), 0.75 * depth)
            bg = S.events(*S.upsteps(seg), 0.75 * depth)
            hit = np.mean([np.any(np.abs(ev - r) <= 2) for r in restarts])
            out["restart"][key][str(depth)] = {"recovered": float(hit), "lookalikes_per_day_at_threshold": float(len(bg) / days),
                                               "n_restarts": len(restarts), "days": days}
    return out


def fam_swing(st, fam, idx, level):
    keys = FAM[fam]
    thr = max(float(np.max(st[k][0][idx])) for k in keys)
    return [hours.swing_for(st[k][0], st[k][1], SWINGS, idx, level, thr) for k in keys]


def summarise(st_site):
    lab = st_site["labels_test"]
    n = len(lab)
    classes = {"all": list(range(n)), "work": [i for i, c in enumerate(lab) if c == "work"],
               "quiet": [i for i, c in enumerate(lab) if c != "work"]}
    res = {"test_hours": {c: len(i) for c, i in classes.items()}, "char_hours": st_site["char_hours"], "bands": {}, "restart": {}}
    rng = np.random.default_rng(SEED + 1)
    for bname, st in st_site["bands"].items():
        rows = {}
        for c, idx in classes.items():
            if not idx:
                continue
            rows[c] = {}
            for fam in FAM:
                for lvl in (0.5, 0.8):
                    v = fam_swing(st, fam, idx, lvl)
                    rows[c][f"{fam}_swing{int(lvl * 100)}"] = None if any(x is None for x in v) else [float(min(v)), float(max(v))]
        # a less fragile threshold: the level 5% of no-signal hours exceed (worst of the family)
        rows["all, threshold exceeded by 5% of hours"] = {}
        for fam in FAM:
            thr95 = max(float(np.quantile(st[k][0], 0.95)) for k in FAM[fam])
            for lvl in (0.5, 0.8):
                v = [hours.swing_for(st[k][0], st[k][1], SWINGS, classes["all"], lvl, thr95) for k in FAM[fam]]
                rows["all, threshold exceeded by 5% of hours"][f"{fam}_swing{int(lvl * 100)}"] = None if any(x is None for x in v) else [float(min(v)), float(max(v))]
        worst = {fam: sorted({int(np.argmax(st[k][0])) for k in FAM[fam]}) for fam in FAM}
        ctl = {}
        for fam in FAM:
            draws = []
            for _ in range(N_DRAWS):
                idx = rng.choice(n, size=len(classes["quiet"]), replace=False)
                v = fam_swing(st, fam, idx, 0.8)
                draws.append(max(v) if all(x is not None for x in v) else np.nan)
            ctl[fam] = [float(np.nanquantile(draws, q)) for q in (0.05, 0.5, 0.95)]
        res["bands"][bname] = {"rows": rows, "control_quiet_size_q05_q50_q95": ctl,
                               "worst_null_test_hours": {fam: [[h, lab[h]] for h in w] for fam, w in worst.items()},
                               "notched_of_band_bins": {k: [int(v[2]), int(v[3])] for k, v in st.items()}}
    for depth in DEPTHS:
        res["restart"][str(depth)] = {}
        for fam, keys in FAM.items():
            rec = [st_site["restart"][k][str(depth)]["recovered"] for k in keys]
            la = [st_site["restart"][k][str(depth)]["lookalikes_per_day_at_threshold"] for k in keys]
            res["restart"][str(depth)][fam] = {"recovered": [min(rec), max(rec)], "lookalikes_per_day_at_threshold": [min(la), max(la)]}
    res["restart_days"] = st_site["restart"]["sum"][str(DEPTHS[0])]["days"]
    return res


def main(root, net, upmu, out_dir, start, end, only):
    cache = Path(root) / "_cache"
    for site in (only or []):
        f = cache / f"inject_{site}.pkl"
        if not f.exists():
            kva, lab = load_site(site, root, net, upmu, start, end)
            f.write_bytes(pickle.dumps(site_stats(kva, lab)))
        print(site, "done", flush=True)
    if only:
        return None
    res = {"signal_model": "sawtooth, 10% spread, rho 0.95 (map reference); 1-s averaging meter", "window_s": 3600,
           "swings_kva": SWINGS, "seed": SEED, "bands": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()} for k, v in BANDS.items()},
           "sites": {}}
    for site, desc in SITES.items():
        f = cache / f"inject_{site}.pkl"
        if f.exists():
            res["sites"][site] = dict(description=desc, **summarise(pickle.loads(f.read_bytes())))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "caltech_injection.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    a = sys.argv
    r = main(a[1], a[2], a[3], a[4], a[5], a[6], a[7].split(",") if len(a) > 7 else None)
    if r:
        f = lambda v: "-" if v is None else (f"{v[0]:.1f}" if abs(v[0] - v[1]) < 0.05 else f"{v[0]:.1f}-{v[1]:.1f}")
        for site, s in r["sites"].items():
            print("\n##", site, s["description"], "| test hours", s["test_hours"])
            for bname, b in s["bands"].items():
                for c, row in b["rows"].items():
                    print(f"  {bname:<16}{c[:22]:<23} 80% (50%): sum {f(row['sum_swing80'])} ({f(row['sum_swing50'])}) | phase {f(row['phase_swing80'])} ({f(row['phase_swing50'])}) | resid {f(row['resid_swing80'])} ({f(row['resid_swing50'])})")
                print(f"  {bname:<16}control (quiet size) 80%:", {k: [round(x, 1) for x in v] for k, v in b["control_quiet_size_q05_q50_q95"].items()}, "| notched", b["notched_of_band_bins"]["sum"])
            for depth, d in s["restart"].items():
                print(f"  restart {depth} kVA:", {fam: (f"{v['recovered'][0]:.0%}-{v['recovered'][1]:.0%}", f"{v['lookalikes_per_day_at_threshold'][0]:.0f}-{v['lookalikes_per_day_at_threshold'][1]:.0f}/day") for fam, v in d.items()})
