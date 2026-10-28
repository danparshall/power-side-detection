"""How often does the background itself produce a 'notch' - a drop that recovers to the
same level - of a given depth and duration? A stalled or restarting
island looks like a notch of ~20 kW lasting seconds to minutes. If the feeder rarely
produces look-alikes, a single notch is evidence; if it produces many, only coincidence
across sites helps.
Usage:  python -m island_detectability_map.run_dips [DATA_DIR] [OUT_DIR]

Origin: verification@5080a582
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

from island_detectability_map import run_pulse as R

DURATIONS_S = [5, 30, 120, 600, 1000]
DEPTHS_KVA = [5.0, 10.0, 15.0, 20.0]


def notch_events(x, d, w):
    """Local maxima of notch depth for duration d (s) with pre/post windows w (s); x at 1 Hz.
    depth = mean(pre, post) - mean(during); kept only if pre and post agree within half the depth."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    t = np.arange(w, len(x) - d - w)
    pre = (c[t] - c[t - w]) / w
    dur = (c[t + d] - c[t]) / d
    post = (c[t + d + w] - c[t + d]) / w
    depth = 0.5 * (pre + post) - dur
    ok = np.abs(pre - post) <= 0.5 * np.maximum(depth, 1e-9)
    depth = np.where(ok, depth, 0.0)
    out, i, n = [], 0, len(depth)
    while i < n:                       # greedy peak picking, one event per d-second neighbourhood
        j = min(n, i + d)
        k = i + int(np.argmax(depth[i:j]))
        if depth[k] > 0 and depth[k] >= depth[max(0, k - d):min(n, k + d)].max():
            out.append(float(depth[k]))
            i = k + d
        else:
            i = j
    return np.array(out)


def main(data_dir: Path, out_dir: Path):
    s, fs, _ = R.load(data_dir)
    x = R.block_mean(s, int(fs))
    days = len(x) / 86400.0
    res = {"mean_kva": float(x.mean()), "days": days, "rows": []}
    for d in DURATIONS_S:
        w = max(5, min(d, 60))
        ev = notch_events(x, d, w)
        res["rows"].append({"duration_s": d, "window_s": w,
                            "per_day": {str(a): float((ev >= a).sum() / days) for a in DEPTHS_KVA}})
    (out_dir / "dip_lookalikes.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    dd = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/upmu")
    od = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/20260930_island_detectability_map")
    r = main(dd, od)
    print("mean kVA", round(r["mean_kva"]), "days", round(r["days"], 2))
    for row in r["rows"]:
        print(f"notch {row['duration_s']:>5} s:", {k: round(v, 1) for k, v in row["per_day"].items()}, "per day at depth >= kVA")
