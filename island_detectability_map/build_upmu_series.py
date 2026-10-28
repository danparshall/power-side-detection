"""Reduce the twelve whole LBNL streams (upmu_full) to the per-phase series the detectors use.
EXPLORATORY orchestration: the pieces are tested (phasor.complex_power, upmu_full.block_nanmean,
upmu_full.void_ranges, upmu_full.make_head_npz); this script only wires them and is checked by
comparing its output with the old 76-hour cache (see __main__).

Per phase k, at 12 samples a second and at 1 a second (block means; a block with any missing
or doubtful sample is NaN, never filled):
  mag  = V1 * I_k / 1e3              the magnitude series of run_pulse to run_fast_dips (kVA)
  p, q = real and reactive power     V_k * I_k * cos/sin(angle_V_k - angle_I_k) / 1e3 (kW, kvar)
Writes DATA_DIR/series/{mag,p,q}{12,1}.npy, each (3, n), float64, and meta.json.
Also rebuilds the old caches grizzly_head.npz / grizzly_head6.npz / grizzly_head_ang.npz.

Usage: python -m island_detectability_map.build_upmu_series DATA_DIR

Origin: verification@10d761ed
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np

from island_detectability_map import phasor, upmu_full as U

SENSOR = U.DEFAULT_SENSOR
CHUNK = 6_000_000            # 50,000 s; a multiple of 120


def main(data_dir: Path, budget_s: float = 200.0):
    """Staged: each 50,000-second chunk is cached under series/chunks/, so a run that is cut
    off resumes. Returns None until every chunk exists; then assembles and returns the meta."""
    t_start = time.time()
    data_dir = Path(data_dir)
    full, out = data_dir / "full", data_dir / "series"
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    prov = json.loads((full / "PROVENANCE_full.json").read_text())
    names = [f"{a}{k}{b}" for k in (1, 2, 3) for a in "CL" for b in ("MAG", "ANG")]
    z = {s: np.load(full / f"{SENSOR}_{s}.npy", mmap_mode="r") for s in names}
    doubt = sorted({tuple(r) for s in names for r in prov[f"{SENSOR}/{s}"]["overwritten_ranges"]})
    n = U.N_FULL
    starts = list(range(0, n, CHUNK))
    for a in starts:
        cf = out / "chunks" / f"{a:010d}.npz"
        if cf.exists():
            continue
        if time.time() - t_start > budget_s:
            print("budget spent; run again to resume"); return None
        b = min(n, a + CHUNK)
        c = {}
        for s in names:
            x = np.array(z[s][a:b], dtype=np.float32)
            for lo, hi in doubt:
                if lo < b and hi > a:
                    x[max(lo, a) - a:min(hi, b) - a] = np.nan
            c[s] = x
        v1 = c["L1MAG"].astype(np.float64)
        res = {}
        for k in (1, 2, 3):
            m = v1 * c[f"C{k}MAG"] / 1e3
            s_k = phasor.complex_power(c[f"L{k}MAG"], c[f"L{k}ANG"], c[f"C{k}MAG"], c[f"C{k}ANG"])
            for name, r in (("mag", m), ("p", s_k.real), ("q", s_k.imag)):
                res.setdefault(f"{name}12", []).append(U.block_nanmean(r, 10, 1.0))
                res.setdefault(f"{name}1", []).append(U.block_nanmean(r, 120, 1.0))
            del s_k, m
        tmp = cf.with_suffix(".tmp.npz")
        np.savez(tmp, **{k: np.vstack(v) for k, v in res.items()})
        tmp.rename(cf)
        print(f"  chunk {a // CHUNK + 1}/{len(starts)} {time.time() - t_start:.0f}s", flush=True)
    meta = {"sensor": SENSOR, "t0_ns": prov[f"{SENSOR}/C1MAG"]["t0_ns"], "doubtful_ranges_120hz": [list(r) for r in doubt],
            "missing_ranges_120hz": prov[f"{SENSOR}/C1MAG"]["missing_ranges"], "units": "mag kVA (V1*I_k), p kW, q kvar"}
    for key in ("mag12", "p12", "q12", "mag1", "p1", "q1"):
        x = np.concatenate([np.load(out / "chunks" / f"{a:010d}.npz")[key] for a in starts], axis=1)
        np.save(out / f"{key}.npy", x)
        meta[f"{key}_shape"] = list(x.shape); meta[f"{key}_nan"] = int(np.isnan(x[0]).sum())
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


if __name__ == "__main__":
    dd = Path(sys.argv[1])
    m = main(dd)
    if m is None:
        sys.exit(0)
    print({k: v for k, v in m.items() if k.endswith(("shape", "nan"))})
    for name, streams in (("grizzly_head.npz", ["C1MAG", "C2MAG", "C3MAG", "L1MAG"]),
                          ("grizzly_head6.npz", ["C1MAG", "C2MAG", "C3MAG", "L1MAG", "L2MAG", "L3MAG"]),
                          ("grizzly_head_ang.npz", ["L1ANG", "C1ANG"])):
        print(name, "rows", U.make_head_npz(dd, streams, name))
