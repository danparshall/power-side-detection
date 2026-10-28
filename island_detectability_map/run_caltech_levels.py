"""Background by level and by phase on the Caltech SoCal 28-bus network.
Writes caltech_levels.json.

EXPLORATORY orchestration: the loader (caltech.py) and the hour labels are unit-tested; the
measures below reuse run_sync.upsteps/events, run_hours.notch_times and pulse.welch_psd
unchanged and have no tests of their own.

Per meter, for the three-phase sum, each phase and each imbalance residual (phase minus the
mean of the other two), in kVA (current magnitude x line-to-neutral voltage magnitude):
* mean load; the meter's own real power over that (the power factor);
* background RMS in 0.02-0.1 Hz and 0.1-0.5 Hz (1-second data stops at 0.5 Hz): computed per
  hour, reported as the median hour, overall and by hour-of-day class;
* sharp up-steps (3-s means either side, as run_sync.upsteps) and notches of 30 s and 120 s
  (as run_dips) per 24 h at 2, 5, 10, 20 kVA, overall and by class.
Seconds missing from any channel are never filled: spectra skip segments that contain one,
and step and notch counts run inside each gap-free stretch.
Then: a log-log fit of background RMS against mean load across meters.

Stages are cached per meter under ROOT/_cache (gitignored), so the run can be done in pieces:
  python -m island_detectability_map.run_caltech_levels ROOT NETWORK_JSON OUT_DIR START END [meter,meter,...]
With a meter list it computes and caches those meters only; without one it assembles the JSON.

Origin: verification@72055af1
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

from island_detectability_map import caltech, hours, phases, pulse, run_sync as S, run_hours as H

SIZES = [2.0, 5.0, 10.0, 20.0]
BANDS = [((0.02, 0.1), 500.0), ((0.1, 0.5), 100.0)]
NOTCH_DURATIONS = [30, 120]
CLASSES = ("all", "work", "weeknight", "weekend")
LEVELS = {"egauge_19": "16.5 kV", "egauge_20": "16.5 kV", "egauge_21": "16.5 kV",
          "egauge_17": "2.4 kV", "egauge_18": "2.4 kV"}        # everything else: building main, 480 V


def series_of(kva):
    s = {"sum": kva.sum(axis=0)}
    for k, p in enumerate("abc"):
        s[f"phase_{p}"] = kva[k]
        s[f"resid_{p}"] = phases.imbalance(kva, k)
    return s


def runs(mask, min_len=600):
    """Gap-free stretches [a, b) of at least min_len seconds."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_len]


def band_rms_hour(x, band, seg):
    try:
        f, p = pulse.welch_psd(x, 1.0, seg)
    except ValueError:
        return np.nan
    sel = (f >= band[0]) & (f < band[1])
    return float(np.sqrt(np.sum(p[sel]) * (f[1] - f[0])))


def per_class(values_by_hour, lab):
    v = np.asarray(values_by_hour, float)
    out = {}
    for c in CLASSES:
        sel = np.isfinite(v) & (np.ones(len(v), bool) if c == "all" else np.array([l == c for l in lab]))
        out[c] = float(np.median(v[sel])) if sel.any() else None
    return out


def measure(d, start, name):
    n = len(d.total)
    n_hours = n // 3600
    lab = caltech.hour_labels(start, n_hours)
    sec_lab = np.repeat(np.array(lab), 3600)
    ser = series_of(d.kva)
    valid_h = {c: float(np.sum(d.mask[:n_hours * 3600] & ((sec_lab == c) if c != "all" else True))) / 3600.0 for c in CLASSES}
    hourly_total = [float(np.nanmean(d.total[h * 3600:(h + 1) * 3600])) if d.mask[h * 3600:(h + 1) * 3600].mean() > 0.5 else np.nan
                    for h in range(n_hours)]
    res = {"meter": name, "level": LEVELS.get(name, "building main, 480 V"),
           "mean_kva": float(np.nanmean(d.total)), "mean_kva_per_phase": [float(v) for v in np.nanmean(d.kva, axis=1)],
           "mean_kva_by_class": per_class(hourly_total, lab),
           "hourly_mean_min_max_kva": [float(np.nanmin(hourly_total)), float(np.nanmax(hourly_total))],
           "power_factor_median": float(np.nanmedian(d.p_kw / d.total)) if d.p_kw is not None else None,
           "masked_seconds": d.report["masked_seconds"], "n_gaps": len(d.report["gaps"]),
           "empty_files": d.report["empty_files"], "valid_hours_by_class": valid_h,
           "band_rms_kva": {}, "phase_corr": {}, "upsteps_per_24h": {}, "notches_per_24h": {}}
    for band, seg in BANDS:
        key = f"{band[0]}-{band[1]}Hz"
        res["band_rms_kva"][key] = {k: per_class([band_rms_hour(v[h * 3600:(h + 1) * 3600], band, seg) for h in range(n_hours)], lab)
                                    for k, v in ser.items()}
        try:
            c = phases.band_correlation(d.kva, 1.0, band, seg)
            res["phase_corr"][key] = [float(c[0, 1]), float(c[0, 2]), float(c[1, 2])]
        except ValueError:
            res["phase_corr"][key] = None
    rr = runs(d.mask[:n_hours * 3600])
    for k, v in ser.items():
        ev = {a: [] for a in SIZES}
        nt = {dur: [] for dur in NOTCH_DURATIONS}
        for a, b in rr:
            t, st = S.upsteps(v[a:b])
            for size in SIZES:
                ev[size].append(S.events(t, st, size) + a)
            for dur in NOTCH_DURATIONS:
                e = H.notch_times(v[a:b], dur, max(5, min(dur, 60)))
                if len(e):
                    e[:, 0] += a
                    nt[dur].append(e)
        def rate(times, c):
            times = np.asarray(times, dtype=int)
            m = np.ones(len(times), bool) if c == "all" else (sec_lab[times] == c)
            return float(m.sum() / valid_h[c] * 24) if valid_h[c] > 0 else None
        res["upsteps_per_24h"][k] = {c: {str(size): rate(np.concatenate(ev[size]) if ev[size] else [], c) for size in SIZES} for c in CLASSES}
        res["notches_per_24h"][k] = {}
        for dur in NOTCH_DURATIONS:
            e = np.vstack(nt[dur]) if nt[dur] else np.zeros((0, 2))
            res["notches_per_24h"][k][str(dur)] = {c: {str(size): rate(e[e[:, 1] >= size, 0], c) for size in SIZES} for c in CLASSES}
    return res


def fit(xs, ys):
    """Least-squares slope of log y on log x with its standard error."""
    x, y = np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float))
    n = len(x)
    b, a = np.polyfit(x, y, 1)
    r = y - (a + b * x)
    se = float(np.sqrt(np.sum(r ** 2) / (n - 2) / np.sum((x - x.mean()) ** 2))) if n > 2 else None
    return {"n": n, "exponent": float(b), "se": se, "amp_at_1mva": float(np.exp(a + b * np.log(1000.0))),
            "scatter_factor": float(np.exp(np.std(r)))}


def main(root: Path, net: Path, out_dir: Path, start: str, end: str, only: list[str] | None):
    mm = caltech.meter_map(net)
    cache = root / "_cache"; cache.mkdir(exist_ok=True)
    meters = sorted([p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("egauge_")], key=lambda s: int(s.split("_")[1]))
    for name in (only or []):
        f = cache / f"{name}.json"
        if f.exists():
            continue
        d = caltech.load_element(root / name / "magnitudes", mm[name], start, end)
        if not d.mask.any():
            f.write_text(json.dumps({"meter": name, "no_data": True, "empty_files": d.report["empty_files"]}))
            print(name, "no data", flush=True)
            continue
        r = measure(d, start, name)
        f.write_text(json.dumps(r))
        print(name, r["level"], "mean %.0f kVA" % r["mean_kva"], "masked", r["masked_seconds"], flush=True)
    if only:
        return None
    rows = [json.loads((cache / f"{m}.json").read_text()) for m in meters if (cache / f"{m}.json").exists()]
    good = [r for r in rows if not r.get("no_data")]
    res = {"source": "Caltech SoCal 28-bus (arXiv 2504.06588), 1-s magnitudes", "time_range_utc": [start, end],
           "class_rule": "work = weekday 07:00-19:00 America/Los_Angeles; weeknight; weekend", "meters": rows, "fits": {}}
    for key in ("0.02-0.1Hz", "0.1-0.5Hz"):
        res["fits"][key] = {}
        for label, sel in (("all meters", good), ("without oversized-CT meters 18, 20", [r for r in good if r["meter"] not in ("egauge_18", "egauge_20")]),
                           ("building mains only", [r for r in good if r["level"].startswith("building")])):
            res["fits"][key][label] = {
                "sum vs total load": fit([r["mean_kva"] for r in sel], [r["band_rms_kva"][key]["sum"]["all"] for r in sel]),
                "one phase vs that phase's load": fit([v for r in sel for v in r["mean_kva_per_phase"]],
                                                      [r["band_rms_kva"][key][f"phase_{p}"]["all"] for r in sel for p in "abc"]),
                "residual vs that phase's load": fit([v for r in sel for v in r["mean_kva_per_phase"]],
                                                     [r["band_rms_kva"][key][f"resid_{p}"]["all"] for r in sel for p in "abc"])}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "caltech_levels.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    a = sys.argv
    r = main(Path(a[1]), Path(a[2]), Path(a[3]), a[4], a[5], a[6].split(",") if len(a) > 6 else None)
    if r:
        for row in r["meters"]:
            if row.get("no_data"):
                print(row["meter"], "no data"); continue
            b = row["band_rms_kva"]["0.1-0.5Hz"]
            print(f"{row['meter']:<10}{row['level']:<22} {row['mean_kva']:7.0f} kVA  pf {row['power_factor_median'] or float('nan'):.2f}  "
                  f"rms 0.1-0.5: sum {b['sum']['all']:.2f} phase {min(b[f'phase_{p}']['all'] for p in 'abc'):.2f}-{max(b[f'phase_{p}']['all'] for p in 'abc'):.2f} "
                  f"resid {min(b[f'resid_{p}']['all'] for p in 'abc'):.2f}-{max(b[f'resid_{p}']['all'] for p in 'abc'):.2f}  masked {row['masked_seconds']}")
        for k, v in r["fits"].items():
            for lab, f3 in v.items():
                print(k, "|", lab, "|", {n: (round(f["exponent"], 2), round(f["se"], 2), f["n"]) for n, f in f3.items()})
