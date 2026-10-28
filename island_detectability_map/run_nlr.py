"""What one 16-H100 job looks like in the NLR power logs.
Writes nlr_island_pulse.json.

EXPLORATORY orchestration. Tested pieces: nlr.* (tests/test_nlr.py). The rest wires them.

Data: data.nlr.gov/submissions/312 (Vercellino et al., arXiv 2604.07345), the ten 4-node
training runs (four H100s a node): five Llama-2 70B LoRA fine-tunes, five Stable Diffusion
trainings. See nlr.py for what the readings are: software readings of GPU and CPU package
power, GPU readings a one-second running mean, logged every 0.2 s. Not AC-side.

Per run, on the 16-GPU total:
* the busy stretch (first to last slot above half the run's 99th percentile);
* a two-level description: slots are split at the midpoint of the 5th and 95th percentiles;
  runs above and below it give high and low dwell times, and up-crossings give the period;
* blocks: a 60 s window is "swinging" if it contains both levels, else "flat";
* how much of the per-GPU swing survives in the total (1 = all GPUs move together);
* what the CPUs do at the same moments (they partly fill the GPU dips or not);
* the one-second running mean's gain at the measured cadence.

Usage: python -m island_detectability_map.run_nlr NLR_DATA_DIR OUT_DIR

Origin: verification@fa3c1e3c
"""
from __future__ import annotations

import glob, json, sys
from pathlib import Path

import numpy as np

from island_detectability_map import nlr

DT = 0.2
RUNS = {"llama2_70b_lora": ("training_llama2_70b_lora", ["10742829", "10742831", "10742832", "10742833", "10742834"]),
        "stable_diffusion": ("training_stable_diffusion", ["10742981", "10742982", "10742983", "10742986", "10742988"])}
GPU_TDP_W, N_GPU = 700.0, 16


def busy_span(x):
    hi = np.nanpercentile(x, 99)
    on = np.where(x > 0.5 * hi)[0]
    return int(on[0]), int(on[-1]) + 1


def two_level(x):
    """Dwell times and period from crossings of the midpoint between the 5th and 95th percentiles."""
    s = nlr.swing(x)
    mid = 0.5 * (s["low"] + s["high"])
    lo_starts, lo_len = nlr.runs_below(x, mid)
    hi_starts, hi_len = nlr.runs_below(-x, -mid)
    ups = hi_starts[hi_starts > 0]
    per = np.diff(ups) * DT
    return s, mid, lo_starts, lo_len * DT, hi_starts, hi_len * DT, per


def q(v, ps=(10, 50, 90)):
    v = np.asarray(v, dtype=float)
    return [float(np.percentile(v, p)) for p in ps] if len(v) else None


def one_run(data_dir: Path, folder: str, sid: str):
    g = sorted(glob.glob(str(data_dir / "00_raw_datasets" / folder / "4node" / f"nvml*{sid}*")))
    c = sorted(glob.glob(str(data_dir / "00_raw_datasets" / folder / "4node" / f"rapl*{sid}*")))
    assert len(g) == 4 and len(c) == 4, (sid, len(g), len(c))
    G, t0, rep = nlr.island_total(g, DT)
    C, t0c, repc = nlr.island_total(c, DT)
    n = min(len(G), len(C)); G, C = G[:n], C[:n]
    assert abs(t0 - t0c) < 0.1e9
    a, b = busy_span(G)
    x, cx = G[a:b], C[a:b]
    s, mid, lo_st, lo_len, hi_st, hi_len, per = two_level(x)
    # blocks of 60 s: swinging if both levels are visited
    w = int(60 / DT)
    kinds = []
    for i in range(0, len(x) - w + 1, w):
        seg = x[i:i + w]; seg = seg[np.isfinite(seg)]
        frac_hi = float(np.mean(seg > mid))
        kinds.append("swinging" if 0.1 < frac_hi < 0.9 else ("flat_high" if frac_hi >= 0.9 else "flat_low"))
    kinds = np.array(kinds)
    sw_slots = np.repeat(kinds == "swinging", w)
    xs = x[:len(sw_slots)][sw_slots]
    # per-GPU swings against the total's (same slots)
    per_gpu = []
    for p in g:
        t, cols = nlr.read_wattameter(p)
        for v in cols.values():
            y, _ = nlr.to_grid(t, v, t0, DT, n)
            per_gpu.append(nlr.swing(y[a:b])["swing"])
    cs = nlr.swing(cx)
    ok = np.isfinite(x) & np.isfinite(cx)
    cpu_when_gpu_high = float(np.mean(cx[ok & (x > mid)])); cpu_when_gpu_low = float(np.mean(cx[ok & (x <= mid)]))
    tot = nlr.swing(x + cx)
    # periods inside swinging blocks only (a long low stretch is not a step)
    med_per = float(np.median(per)) if len(per) else None
    step_per = per[per < 2.0 * med_per] if med_per else per
    long_low = lo_len[lo_len > 3.0 * np.median(lo_len)] if len(lo_len) else lo_len
    long_low_starts = lo_st[lo_len > 3.0 * np.median(lo_len)] * DT if len(lo_len) else []
    ups_between = None
    if len(long_low_starts) > 2:
        ups_t = hi_st * DT
        ups_between = [int(np.sum((ups_t > u) & (ups_t < v))) for u, v in zip(long_low_starts[:-1], long_low_starts[1:])]
    f_step = 1.0 / float(np.median(step_per)) if len(step_per) else None
    main = per[np.abs(per - med_per) <= 0.2 * med_per] if med_per else per      # the main mode: within 20% of the median
    pairs = [(u, v) for u, v in zip(per[:-1], per[1:]) if abs(u - med_per) <= 0.2 * med_per and abs(v - med_per) <= 0.2 * med_per] if med_per else []
    return {"slurm_id": sid, "minutes_logged": n * DT / 60, "busy_minutes": (b - a) * DT / 60,
            "nan_slots_gpu": rep["nan_slots"], "nan_slots_cpu": repc["nan_slots"],
            "gpu_total_kw": {k: v / 1e3 for k, v in s.items()},
            "gpu_swing_share_of_high": s["swing"] / s["high"],
            "gpu_high_share_of_16x700W": s["high"] / (GPU_TDP_W * N_GPU),
            "gpu_low_share_of_16x700W": s["low"] / (GPU_TDP_W * N_GPU),
            "midpoint_kw": mid / 1e3,
            "high_dwell_s_p10_50_90": q(hi_len), "low_dwell_s_p10_50_90": q(lo_len[lo_len <= 3.0 * np.median(lo_len)]),
            "period_s_p10_50_90": q(step_per), "n_periods": int(len(step_per)),
            "period_cv": float(np.std(step_per) / np.mean(step_per)) if len(step_per) > 2 else None,
            "period_lag1_corr": float(np.corrcoef(step_per[:-1], step_per[1:])[0, 1]) if len(step_per) > 3 else None,
            "main_mode_periods": {"n": int(len(main)), "share_of_all": float(len(main) / max(1, len(per))),
                                  "mean_s": float(np.mean(main)) if len(main) else None,
                                  "cv": float(np.std(main) / np.mean(main)) if len(main) > 2 else None,
                                  "lag1_corr": float(np.corrcoef(np.array(pairs).T)[0, 1]) if len(pairs) > 3 else None},
            "cadence_hz": f_step,
            "running_mean_gain_at_cadence": float(nlr.running_mean_gain(f_step, 1.0)) if f_step else None,
            "duty_high": float(np.median(hi_len) / np.median(step_per)) if len(step_per) else None,
            "long_low_stretches": {"count": int(len(long_low)), "seconds_p10_50_90": q(long_low),
                                   "spacing_s_p10_50_90": q(np.diff(long_low_starts)) if len(long_low_starts) > 1 else None,
                                   "up_crossings_between_p10_50_90": q(ups_between) if ups_between else None},
            "blocks_60s": {k: int(np.sum(kinds == k)) for k in ("swinging", "flat_high", "flat_low")},
            "gpu_total_kw_in_swinging_blocks": {k: v / 1e3 for k, v in nlr.swing(xs).items()} if len(xs) else None,
            "gpu_std_kw_in_flat_high_blocks": float(np.nanmedian([np.nanstd(x[i * w:(i + 1) * w]) for i in np.where(kinds == "flat_high")[0]]) / 1e3) if np.any(kinds == "flat_high") else None,
            "sum_of_per_gpu_swings_kw": float(np.sum(per_gpu) / 1e3), "coherence_total_over_sum": float(s["swing"] / np.sum(per_gpu)),
            "cpu_total_kw": {k: v / 1e3 for k, v in cs.items()},
            "cpu_kw_when_gpu_high": cpu_when_gpu_high / 1e3, "cpu_kw_when_gpu_low": cpu_when_gpu_low / 1e3,
            "gpu_plus_cpu_kw": {k: v / 1e3 for k, v in tot.items()}}


def main(data_dir: Path, out_dir: Path):
    res = {"source": "NLR Data Catalog submission 312 (DOI 10.7799/3025227), dataset.zip of 2026-04-10; 4-node training runs",
           "what": "software readings: 16 H100 GPUs (one-second running mean, every 0.2 s) and 8 CPU packages; not AC-side",
           "runs": {}}
    for name, (folder, ids) in RUNS.items():
        res["runs"][name] = [one_run(data_dir, folder, sid) for sid in ids]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "nlr_island_pulse.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    r = main(Path(sys.argv[1]), Path(sys.argv[2]))
    f = lambda v: "-" if v is None else "/".join(f"{x:.1f}" for x in v)
    for name, rows in r["runs"].items():
        print("##", name)
        for e in rows:
            g = e["gpu_total_kw"]
            print(f"  {e['slurm_id']} busy {e['busy_minutes']:.0f} min | GPU low/mid/high {g['low']:.2f}/{g['mid']:.2f}/{g['high']:.2f} kW swing {g['swing']:.2f} "
                  f"({100 * e['gpu_swing_share_of_high']:.0f}% of high) | period {f(e['period_s_p10_50_90'])} s cv {e['period_cv'] and round(e['period_cv'], 3)} "
                  f"r1 {e['period_lag1_corr'] and round(e['period_lag1_corr'], 2)} n {e['n_periods']} | high {f(e['high_dwell_s_p10_50_90'])} low {f(e['low_dwell_s_p10_50_90'])} duty {e['duty_high'] and round(e['duty_high'], 2)}")
            ll = e["long_low_stretches"]
            print(f"      long lows {ll['count']} of {f(ll['seconds_p10_50_90'])} s every {f(ll['spacing_s_p10_50_90'])} s, up-crossings between {f(ll['up_crossings_between_p10_50_90'])} | blocks {e['blocks_60s']} | "
                  f"flat-high std {e['gpu_std_kw_in_flat_high_blocks'] and round(e['gpu_std_kw_in_flat_high_blocks'], 3)} kW | coherence {e['coherence_total_over_sum']:.2f} | "
                  f"CPU {e['cpu_total_kw']['mean']:.2f} kW (GPU high {e['cpu_kw_when_gpu_high']:.2f}, low {e['cpu_kw_when_gpu_low']:.2f}) | GPU+CPU swing {e['gpu_plus_cpu_kw']['swing']:.2f}")
