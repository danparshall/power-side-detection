"""A second logged training job through run_task3's design: one Elsayed et al. node trace
(8 x H100 or 8 x B200, see elsayed.py) added unscaled to the Caltech 1-second data, learned
in both directions, scored by the spectral detector in both bands. Writes
task3_caltech_elsayed_job.json with the same row layout as task3_caltech_real_job.json.

EXPLORATORY orchestration, no tests of its own. Tested pieces: elsayed.node_total, caltech.*,
run_pulse.experiment, phases.imbalance; run_task3.stage / score_stage.

Signal: the 8-GPU total, averaged within each second, busy stretch only (run_nlr.busy_span), the
few empty seconds bridged by interpolation, mean removed, looped from a random start, times a
scale (1 = as logged, kW taken as kVA). Everything else (sites, folds, bands, scales, margins,
threshold rule) is run_task3's.

Usage: python -m island_detectability_map.run_elsayed_inject CSV CALTECH_ROOT NETWORK_JSON CACHE_DIR OUT_DIR [STAGE]
STAGE = <site>:<fold>:<band slow|fast>  (all four series of that site; cached under CACHE_DIR)

Origin: verification@b28fe07c
"""
from __future__ import annotations

import json, pickle, sys
from pathlib import Path

import numpy as np

from island_detectability_map import elsayed, run_nlr as RN, run_task3 as T3
from island_detectability_map.run_phasor import cached


def trace_kw(csv: Path, fx: float = 1.0, remove_mean: bool = True):
    """8-GPU total in kW at fx samples a second; mean removed (the swing alone) unless remove_mean=False
    (the whole job, its mean draw included)."""
    tot, _, _ = elsayed.node_total(csv, dt_s=1.0 / fx)
    a, b = RN.busy_span(tot)
    x = tot[a:b] / 1e3
    ok = np.isfinite(x)
    t = np.arange(len(x))
    y = np.interp(t, t[ok], x[ok])
    return y - y.mean() if remove_mean else y


def main(csv, root, net, cache, out_dir, only=None):
    csv, cache = Path(csv), Path(cache)
    if only:
        site, fold, band = only.split(":")
        tr = trace_kw(csv)
        cached(cache, f"{site}__{fold}__{band}", lambda: T3.stage(site, fold, band, root, net, None, None, trace=tr))
        return None
    res = {"design": T3.__doc__.split("Folds:")[1].split("Usage")[0].strip(), "job": csv.name,
           "signal": __doc__.split("Signal:")[1].split("Usage")[0].strip(), "sites": T3.SITES, "scales": T3.SCALES, "rows": []}
    for f in sorted(cache.glob("*.pkl")):
        site, fold, band = f.stem.split("__")
        st = pickle.loads(f.read_bytes())
        for r in T3.score_stage(st):
            res["rows"].append({"site": site, "level": T3.SITES[site], "mean_kva": st["mean_kva"], "fold": fold, "band": band,
                                "trace_swing_kw": st["trace_swing_kw"], **r})
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "task3_caltech_elsayed_job.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    a = sys.argv
    r = main(a[1], a[2], a[3], a[4], a[5], a[6] if len(a) > 6 else None)
    if r:
        for row in r["rows"]:
            f = lambda v: "-" if v is None else f"{v[0]:.1f}-{v[1]:.1f}"
            print(f"{row['site']:<10}{row['band']:<5} fold {row['fold']} {row['group']:<6} x{str(row['margin'])[:6]:<6} "
                  f"| alarms {row['alarm_hours']:>3}/{row['score_hours']:<3} | rate@1 {[round(v, 2) for v in row['rate_at_scale_1']]} "
                  f"| 80% at {f(row['swing_kw_80_min_max'])} kW | set aside {[round(v, 2) for v in row['share_of_band_set_aside']]}")
