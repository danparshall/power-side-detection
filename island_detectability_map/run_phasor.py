"""Phasor residual and real/reactive split on the LBNL bus.
Writes phasor_measurement.json.

EXPLORATORY orchestration. The tested pieces are phasor.* (tests/test_phasor.py),
upmu_full.* (tests/test_upmu_full.py) and the detectors' own pieces; this script wires them and
has no unit tests. Every estimator here is first run on a planted signal (stage "plant" and the
planted rows of stage "steps").

Question: the per-phase detectors (run_phases, run_phase_edges, run_hours,
run_fast_dips) use current magnitudes. Magnitudes do not
add. By how much does that misstate (a) the island's signal and (b) the background, and does the
linear version (real power per phase, and its imbalance residual) detect a smaller island?

Series, per phase k, 12 a second and 1 a second (build_upmu_series):
  mag_k = V1 * I_k               (kVA; what those detectors use)
  p_k, q_k                       (kW, kvar; each phase's current in its own voltage's frame)
Families: sum, one phase, imbalance residual (phase minus the mean of the other two), for mag
and for p. A unity-power-factor island adds to p with gain 1 by construction; its gain in mag is
measured in stage "plant".

Record: the same 75.8 h as run_phases / run_hours (first 36 h characterise, 39 test hours), so the mag
family must reproduce run_hours' fine-grid figures. Stage "steps" also runs on the whole
11.57 days. Thresholds in stages "spectral" and "edge" are set on the hours they are scored on,
as in run_pulse to run_fast_dips (held-out thresholds are run_heldout's).

Stages are cached under DATA_DIR/phasor_cache/ (gitignored); STAGE runs one stage only.
Usage: python -m island_detectability_map.run_phasor ../data/upmu OUT_DIR [STAGE]

Origin: verification@10d761ed
"""
from __future__ import annotations

import json, pickle, sys
from pathlib import Path

import numpy as np

from island_detectability_map import hours, phasor, phases, upmu_full as U
from island_detectability_map import run_pulse as R, run_sync as S, run_phases as P, run_phase_edges as E, run_hours as H

N_HEAD12 = 32765171 // 10            # the 75.8 h record of run_phases / run_hours, at 12 a second
FX = 12.0
FINE = H.FINE
FAMILIES = H.FAMILIES
COV_BANDS = P.COV_BANDS
STEP_SIZES = [2.0, 5.0, 10.0, 15.0]
TOTAL_SIZES = [10.0, 15.0, 20.0, 30.0]
SENSOR = U.DEFAULT_SENSOR


def cached(cache: Path, name: str, fn):
    f = cache / f"{name}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    v = fn()
    cache.mkdir(parents=True, exist_ok=True)
    f.write_bytes(pickle.dumps(v))
    return v


def load12(data_dir: Path, n=N_HEAD12):
    return {q: np.array(np.load(data_dir / "series" / f"{q}12.npy", mmap_mode="r")[:, :n]) for q in ("mag", "p", "q")}


# ---------------------------------------------------------------- stage: plant
def plant(data_dir: Path, hour0: float = 40.0, seconds: int = 1200):
    """Plant known loads in the RAW 120-a-second streams and read them back through the same
    arithmetic as build_upmu_series. Gains are least-squares slopes of (series after - before)
    on the planted waveform."""
    a = int(hour0 * 3600 * 120); b = a + seconds * 120
    z = {f"{c}{k}{m}": np.array(np.load(data_dir / "full" / f"{SENSOR}_{c}{k}{m}.npy", mmap_mode="r")[a:b], dtype=np.float64)
         for k in (1, 2, 3) for c in "CL" for m in ("MAG", "ANG")}
    t = np.arange(b - a) / 120.0
    sq = (np.floor(t / 5.0) % 2 == 1).astype(float)            # 10 s period square wave, 0/1

    def series(zz):
        mag = np.vstack([zz["L1MAG"] * zz[f"C{k}MAG"] / 1e3 for k in (1, 2, 3)])
        s = np.vstack([phasor.complex_power(zz[f"L{k}MAG"], zz[f"L{k}ANG"], zz[f"C{k}MAG"], zz[f"C{k}ANG"]) for k in (1, 2, 3)])
        return mag, s

    def slope(y):
        w = sq - sq.mean()
        return float(np.dot(y - y.mean(), w) / np.dot(w, w))

    mag0, s0 = series(z)
    pf_angle = np.rad2deg(np.angle(s0.mean(axis=1)))
    out = {"window": f"hours {hour0}-{hour0 + seconds / 3600:.2f} of the record, raw 120/s",
           "background_pf_angle_deg": pf_angle.tolist(), "cos_pf_angle": np.cos(np.deg2rad(pf_angle)).tolist(),
           "v_k_over_v1": [float(np.mean(z[f"L{k}MAG"]) / np.mean(z["L1MAG"])) for k in (1, 2, 3)],
           "single_phase": [], "balanced": []}
    for k in range(3):
        for p_kw in (5.0, 25.0):
            zz = dict(z)
            zz[f"C{k+1}MAG"], zz[f"C{k+1}ANG"] = phasor.add_load(z[f"L{k+1}MAG"], z[f"L{k+1}ANG"], z[f"C{k+1}MAG"], z[f"C{k+1}ANG"], p_kw * sq)
            mag1, s1 = series(zz)
            row = {"phase": k + 1, "planted_kw": p_kw,
                   "gain_p_own": slope(s1[k].real - s0[k].real) / p_kw,
                   "gain_q_own": slope(s1[k].imag - s0[k].imag) / p_kw,
                   "gain_p_residual_own": slope((phasor.complex_residual(s1, k) - phasor.complex_residual(s0, k)).real) / p_kw,
                   "gain_p_residual_others": [slope((phasor.complex_residual(s1, j) - phasor.complex_residual(s0, j)).real) / p_kw
                                              for j in range(3) if j != k],
                   "gain_mag_own": slope(mag1[k] - mag0[k]) / p_kw,
                   "gain_mag_residual_own": slope(phases.imbalance(mag1, k) - phases.imbalance(mag0, k)) / p_kw,
                   "gain_mag_sum": slope(mag1.sum(0) - mag0.sum(0)) / p_kw}
            out["single_phase"].append(row)
    for name, p_kw, q_kvar in (("real only, 6.7 kW a phase (the three-phase island)", 6.7, 0.0),
                               ("motor-like, 30 kW + 45 kvar a phase", 30.0, 45.0)):
        zz = dict(z)
        for k in (1, 2, 3):
            zz[f"C{k}MAG"], zz[f"C{k}ANG"] = phasor.add_load(z[f"L{k}MAG"], z[f"L{k}ANG"], z[f"C{k}MAG"], z[f"C{k}ANG"], p_kw * sq, q_kvar * sq)
        mag1, s1 = series(zz)
        out["balanced"].append({"load": name,
                                "p_sum_gain": slope(s1.real.sum(0) - s0.real.sum(0)) / (3 * p_kw),
                                "mag_sum_shift_kva": slope(mag1.sum(0) - mag0.sum(0)),
                                "true_apparent_kva": 3 * float(np.hypot(p_kw, q_kvar)),
                                "p_residual_shift_kw": [slope((phasor.complex_residual(s1, k) - phasor.complex_residual(s0, k)).real) for k in range(3)],
                                "q_residual_shift_kvar": [slope((phasor.complex_residual(s1, k) - phasor.complex_residual(s0, k)).imag) for k in range(3)],
                                "mag_residual_shift_kva": [slope(phases.imbalance(mag1, k) - phases.imbalance(mag0, k)) for k in range(3)]})
    return out


# ---------------------------------------------------------------- stage: noise
def noise(x):
    nc = int(36 * 3600 * FX)
    rows = []
    for (band, seg) in COV_BANDS:
        for part, sl in (("char", slice(0, nc)), ("test", slice(nc, None))):
            row = {"band_hz": list(band), "part": part}
            for fam in ("mag", "p", "q"):
                C = phases.band_covariance(x[fam][:, sl], FX, band, seg)
                row[f"{fam}_sum"] = float(np.sqrt(np.ones(3) @ C @ np.ones(3)))
                row[f"{fam}_phase"] = [float(np.sqrt(C[k, k])) for k in range(3)]
                res = []
                for k in range(3):
                    w = -0.5 * np.ones(3); w[k] = 1.0
                    res.append(float(np.sqrt(w @ C @ w)))
                row[f"{fam}_residual"] = res
                d = np.sqrt(np.diag(C))
                row[f"{fam}_corr"] = (C / np.outer(d, d))[[0, 0, 1], [1, 2, 2]].tolist()
            rows.append(row)
    return rows


# ---------------------------------------------------------------- stages: spectral, edge
def spectral(ser):
    out = {}
    for key, v in ser.items():
        e = R.experiment(v, FX, 3600, np.random.default_rng(H.SEED), return_null=True, return_stats=True,
                         make_signal=H.MK, **dict(H.REF_KW, swings=FINE))
        out[key] = (np.array(e["null_stats"]), e["inj_stats"])
    return out


def family_table(stats, idx=None):
    """80% and 50% swings per family at one threshold per family (unknown phase), over idx."""
    n = len(next(iter(stats.values()))[0])
    idx = list(range(n)) if idx is None else idx
    out = {}
    for fam in FAMILIES:
        for lvl in (0.5, 0.8):
            out[f"{fam}_swing{int(lvl * 100)}"] = H.summarise(H.family_swings(stats, fam, FINE, {k: idx for k in FAMILIES[fam]}, lvl))
    return out


def edge(ser):
    return {k: H.edge_stats(v, FX, list(range(36)), FINE) for k, v in ser.items()}


def edge_table(es):
    out = {}
    for fam in FAMILIES:
        keys = FAMILIES[fam]
        thr = max(float(np.max(es[k][1])) for k in keys)
        for lvl in (0.5, 0.8):
            v = [hours.swing_for(es[k][1], es[k][2], FINE, list(range(len(es[k][0]))), lvl, thr) for k in keys]
            out[f"{fam}_swing{int(lvl * 100)}"] = H.summarise(v)
        out[f"{fam}_clean_hours"] = [int(len(es[k][0])) for k in keys]
    return out


# ---------------------------------------------------------------- stage: restart edges (1-s)
def restart(x1):
    days = x1.shape[1] / 86400.0
    out = {}
    for key, v in P.series_set(x1).items():
        r = E.restart(v, days)
        out[key] = {"upsteps_per_day": {a: r["upsteps_per_day"][a] for a in ("2.0", "5.0", "10.0", "20.0")},
                    "recovery": {d: r["recovery"][d] for d in ("5.0", "10.0", "20.0")}}
    return out


# ---------------------------------------------------------------- stage: steps (real / reactive)
def fill_gaps(x):
    """1-s series with the record's two bad seconds: carry the previous value forward so the
    running sums work, and return the bad positions so events near them can be dropped."""
    x = np.array(x, dtype=np.float64, copy=True)
    bad = np.where(~np.isfinite(x).all(axis=0))[0] if x.ndim == 2 else np.where(~np.isfinite(x))[0]
    for i in bad:
        x[..., i] = x[..., i - 1]
    return x, bad


def drop_near(ev, bad, margin=6):
    ev = np.asarray(ev, dtype=int)
    if len(bad) == 0 or len(ev) == 0:
        return ev
    return ev[np.min(np.abs(ev[:, None] - np.asarray(bad)[None, :]), axis=1) > margin]


def per_phase_steps(p, q, bad, days):
    """Addendum 11's table for each phase: sharp steps in real power on one phase, and how
    many come with almost no reactive change (|dQ| <= 0.1 |dP|)."""
    out = {}
    for k in range(3):
        for direction, sgn in (("up", 1.0), ("down", -1.0)):
            t, st = S.upsteps(sgn * p[k])
            for size in STEP_SIZES:
                ev = drop_near(S.events(t, st, size), bad)
                ev = ev[(ev > 4) & (ev < p.shape[1] - 5)]
                dp, dq = phasor.step_pq(p[k], q[k], ev, w=3)
                ratio = dq / dp if len(ev) else np.array([])
                out[f"phase{k + 1}, {direction}, >= {size:g} kW"] = {
                    "per_day": len(ev) / days,
                    "ratio_quartiles": [float(v) for v in np.percentile(ratio, [25, 50, 75])] if len(ev) else None,
                    "per_day_almost_no_reactive": float(np.sum(np.abs(ratio) <= 0.1) / days) if len(ev) else 0.0}
    return out


def total_step_events(p, q, bad, size, sgn):
    t, st = S.upsteps(sgn * p.sum(axis=0))
    ev = drop_near(S.events(t, st, size), bad)
    ev = ev[(ev > 4) & (ev < p.shape[1] - 5)]
    dp3, dq3 = phasor.step_pq(p, q, ev, w=3)
    return ev, dp3, dq3


def three_phase_steps(p, q, bad, t0_ns):
    """Look-alikes for a three-phase island's edge: sharp steps in total real power that are
    balanced over the phases and carry almost no reactive power. Counted per day, and per 24 h
    of each hour-of-day class."""
    n = p.shape[1]
    lab = hours.hour_labels(t0_ns, n // 3600 + 1)
    sec_lab = np.array(lab)[np.minimum(np.arange(n) // 3600, len(lab) - 1)]
    cls_hours = {c: float(np.sum(sec_lab == c)) / 3600.0 for c in ("work", "weeknight", "weekend")}
    cls_hours["all"] = n / 3600.0
    out = {"class_hours": cls_hours, "rows": []}
    for direction, sgn in (("up", 1.0), ("down", -1.0)):
        for size in TOTAL_SIZES:
            ev, dp3, dq3 = total_step_events(p, q, bad, size, sgn)
            bal = phasor.balanced_real_only(dp3, dq3, q_ratio=1e9)            # balanced, any reactive part
            for q_ratio in (0.1, 0.15):
                m = phasor.balanced_real_only(dp3, dq3, q_ratio=q_ratio)
                row = {"direction": direction, "min_total_kw": size, "q_ratio": q_ratio}
                for c, h in cls_hours.items():
                    in_c = np.ones(len(ev), bool) if c == "all" else (sec_lab[ev] == c)
                    row[c] = {"all_steps_per_24h": float(in_c.sum() / h * 24),
                              "balanced_per_24h": float((bal & in_c).sum() / h * 24),
                              "balanced_real_only_per_24h": float((m & in_c).sum() / h * 24),
                              "balanced_real_only_count": int((m & in_c).sum())}
                row["times_s"] = [int(v) for v in ev[m]][:200]
                row["dp_total_kw"] = [float(v) for v in dp3.sum(0)[m]][:200]
                row["dq_total_kvar"] = [float(v) for v in dq3.sum(0)[m]][:200]
                out["rows"].append(row)
    return out


def planted_steps(p, q, bad, rng, n_plant=200):
    """Plant island-like edges (20 kW over three phases, no reactive part) and motor-like edges
    (20 kW with 30 kvar) at random seconds; report how many are found and how they are classed."""
    n = p.shape[1]
    out = {}
    for name, dq_tot in (("island-like: 20 kW balanced, no reactive part", 0.0), ("motor-like: 20 kW balanced with 30 kvar", 30.0)):
        pp, qq = p.copy(), q.copy()
        times = np.sort(rng.choice(np.arange(600, n - 600), size=n_plant, replace=False))
        times = times[np.r_[True, np.diff(times) > 120]]
        for i, t in enumerate(times):                    # alternate drop / restore so the level does not run away
            sgn = -1.0 if i % 2 == 0 else 1.0
            pp[:, t + 1:] += sgn * 20.0 / 3; qq[:, t + 1:] += sgn * dq_tot / 3
        found = 0; flagged = {0.1: 0, 0.15: 0}
        for sgn in (1.0, -1.0):
            ev, dp3, dq3 = total_step_events(pp, qq, bad, 15.0, sgn)
            ms = {r: phasor.balanced_real_only(dp3, dq3, q_ratio=r) for r in flagged}
            for t in times:
                j = np.where(np.abs(ev - t) <= 2)[0]
                if len(j):
                    found += 1
                    for r in flagged:
                        flagged[r] += bool(ms[r][j].any())
        out[name] = {"planted": int(len(times)), "found_as_step_of_15kw_or_more": found / len(times),
                     "flagged_balanced_real_only_q0.1": flagged[0.1] / len(times),
                     "flagged_balanced_real_only_q0.15": flagged[0.15] / len(times)}
    return out


def steps(data_dir: Path):
    meta = json.loads((data_dir / "series" / "meta.json").read_text())
    p_all, bad = fill_gaps(np.load(data_dir / "series" / "p1.npy"))
    q_all, _ = fill_gaps(np.load(data_dir / "series" / "q1.npy"))
    out = {}
    for name, n in (("first 2.73 days (the earlier run's record)", 28300536 // 120), ("first 75.8 h", N_HEAD12 // 12), ("whole record, 11.57 days", p_all.shape[1])):
        p, q = p_all[:, :n], q_all[:, :n]
        b = bad[bad < n]
        days = n / 86400.0
        out[name] = {"days": days, "mean_p_kw": p.mean(axis=1).tolist(), "mean_q_kvar": q.mean(axis=1).tolist(),
                     "power_factor": (p.mean(axis=1) / np.hypot(p.mean(axis=1), q.mean(axis=1))).tolist(),
                     "per_phase": per_phase_steps(p, q, b, days),
                     "three_phase": three_phase_steps(p, q, b, meta["t0_ns"])}
    out["planted"] = planted_steps(p_all, q_all, bad, np.random.default_rng(H.SEED + 11))
    return out


# ---------------------------------------------------------------- main
def main(data_dir: Path, out_dir: Path, stage: str | None = None):
    cache = data_dir / "phasor_cache"
    x = None

    def series():
        nonlocal x
        if x is None:
            x = load12(data_dir)
        return x

    stages = {
        "plant": lambda: plant(data_dir),
        "noise": lambda: noise(series()),
        "spectral_mag": lambda: spectral(P.series_set(series()["mag"])),
        "spectral_p": lambda: spectral(P.series_set(series()["p"])),
        "edge_mag": lambda: edge(P.series_set(series()["mag"])),
        "edge_p": lambda: edge(P.series_set(series()["p"])),
        "restart": lambda: {fam: restart(np.array(np.load(data_dir / "series" / f"{fam}1.npy", mmap_mode="r")[:, :N_HEAD12 // 12]))
                            for fam in ("mag", "p")},
        "steps": lambda: steps(data_dir),
    }
    if stage:
        cached(cache, stage, stages[stage])
        return None
    c = {k: cached(cache, k, fn) for k, fn in stages.items()}
    res = {"source": "LBNL Open uPMU grizzly_bus1_2; 75.8 h from 2015-10-01 00:00 UTC unless a row says otherwise",
           "signal_model": "sawtooth, 10% spread, rho 0.95 (map reference); fine swing grid, steps of 2^(1/4)",
           "thresholds": "worst no-signal hour of the same 39 test hours (as the earlier fixed-threshold runs); not held out",
           "plant": c["plant"], "noise_kva_or_kw_rms": c["noise"],
           "spectral": {"mag": family_table(c["spectral_mag"]), "p": family_table(c["spectral_p"])},
           "edge_count": {"mag": edge_table(c["edge_mag"]), "p": edge_table(c["edge_p"])},
           "restart_edges": c["restart"], "steps": c["steps"]}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phasor_measurement.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    r = main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else None)
    if r is not None:
        print(json.dumps({k: r[k] for k in ("spectral", "edge_count")}, indent=1))
