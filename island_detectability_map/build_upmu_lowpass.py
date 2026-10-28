"""Magnitude series of the LBNL bus with a proper low-pass before the 120-to-12 step (see decimate.py for why).
EXPLORATORY orchestration: the pieces are tested (decimate.lowpass_decimate, upmu_full.block_nanmean);
this script only wires them, and is checked by its `mag12` output reproducing the committed
held-out figures (run_heldout) before the low-passed `maglp12` is used.

Needs only four whole streams (C1MAG, C2MAG, C3MAG, L1MAG; see upmu_full). Per phase k:
  mag12, mag1  = V1 * I_k / 1e3, block means of 10 and 120 samples (as build_upmu_series)
  maglp12      = the same product through decimate.lowpass_decimate (unit gain to 5.5 Hz,
                 70 dB down from 6.5 Hz), on the same time grid as mag12
A block or filter span that touches a missing or doubtful sample is NaN, never filled.
Writes DATA_DIR/series/{mag12,mag1,maglp12}.npy, each (3, n), and meta.json.
Staged: each 50,000-second chunk is cached under series/chunks_lp/; run again to resume.

Usage: python -m island_detectability_map.build_upmu_lowpass DATA_DIR

Origin: verification@f68ea5ab
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np

from island_detectability_map import decimate, upmu_full as U

SENSOR = U.DEFAULT_SENSOR
CHUNK = 6_000_000            # 50,000 s; a multiple of 120
STREAMS = ["C1MAG", "C2MAG", "C3MAG", "L1MAG"]


def main(data_dir: Path, budget_s: float = 200.0):
    t_start = time.time()
    data_dir = Path(data_dir)
    full, out = data_dir / "full", data_dir / "series"
    (out / "chunks_lp").mkdir(parents=True, exist_ok=True)
    prov = json.loads((full / "PROVENANCE_full.json").read_text())
    z = {s: np.load(full / f"{SENSOR}_{s}.npy", mmap_mode="r") for s in STREAMS}
    doubt = sorted({tuple(r) for s in STREAMS for r in prov[f"{SENSOR}/{s}"]["overwritten_ranges"]})
    h = decimate.design(10, U.FS)
    pad = int(np.ceil(len(h) / 2 / 120)) * 120
    n = U.N_FULL
    starts = list(range(0, n, CHUNK))
    for a in starts:
        cf = out / "chunks_lp" / f"{a:010d}.npz"
        if cf.exists():
            continue
        if time.time() - t_start > budget_s:
            print("budget spent; run again to resume"); return None
        b = min(n, a + CHUNK)
        lo, hi = max(0, a - pad), min(n, b + pad)
        c = {}
        for s in STREAMS:
            x = np.array(z[s][lo:hi], dtype=np.float64)
            for d0, d1 in doubt:
                if d0 < hi and d1 > lo:
                    x[max(d0, lo) - lo:min(d1, hi) - lo] = np.nan
            c[s] = x
        res = {"mag12": [], "mag1": [], "maglp12": []}
        for k in (1, 2, 3):
            m = c["L1MAG"] * c[f"C{k}MAG"] / 1e3
            core = m[a - lo:a - lo + (b - a)]
            res["mag12"].append(U.block_nanmean(core, 10, 1.0))
            res["mag1"].append(U.block_nanmean(core, 120, 1.0))
            y = decimate.lowpass_decimate(m, 10, U.FS, h)
            res["maglp12"].append(y[(a - lo) // 10:(a - lo) // 10 + (b - a) // 10])
        tmp = cf.with_suffix(".tmp.npz")
        np.savez(tmp, **{k: np.vstack(v) for k, v in res.items()})
        tmp.rename(cf)
        print(f"  chunk {a // CHUNK + 1}/{len(starts)} {time.time() - t_start:.0f}s", flush=True)
    meta = {"sensor": SENSOR, "t0_ns": prov[f"{SENSOR}/C1MAG"]["t0_ns"], "doubtful_ranges_120hz": [list(r) for r in doubt],
            "missing_ranges_120hz": prov[f"{SENSOR}/C1MAG"]["missing_ranges"], "units": "kVA (V1*I_k)",
            "lowpass": {"taps": int(len(h)), "keep_hz": 5.5, "stop_hz": 6.5, "stop_db": 70.0}}
    for key in ("mag12", "mag1", "maglp12"):
        x = np.concatenate([np.load(out / "chunks_lp" / f"{a:010d}.npz")[key] for a in starts], axis=1)
        np.save(out / f"{key}.npy", x)
        meta[f"{key}_shape"] = list(x.shape); meta[f"{key}_nan"] = int(np.isnan(x[0]).sum())
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


if __name__ == "__main__":
    m = main(Path(sys.argv[1]))
    if m is not None:
        print({k: v for k, v in m.items() if k.endswith(("shape", "nan"))})
