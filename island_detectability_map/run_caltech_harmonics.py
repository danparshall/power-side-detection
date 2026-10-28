"""Current harmonics at three Caltech meters from the waveform captures: how big are they and
how steady from one capture to the next? (Question: would a rectifier or
battery converter show in the harmonics with present equipment?) Writes caltech_harmonics.json.
EXPLORATORY; the harmonic estimator is tested in tests/test_caltech_wave.py.

For each ~1 s capture and each current channel: RMS of harmonics 1..13 (odd) at the capture's own
fundamental. Reported per phase: median size (A and % of the fundamental) and the spread
(standard deviation) across the captures of a ten-minute set. The spread is what a new source
has to beat. For scale, the harmonic current a 25 kW three-phase load would add at that voltage,
for a supply with a corrected input (2% per harmonic) and for a plain six-pulse rectifier
(fifth harmonic 20%, seventh 14%).

Usage: python -m island_detectability_map.run_caltech_harmonics WAVE_ROOT OUT_JSON

Origin: verification@91ee680f
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

from island_detectability_map import caltech_wave as W

ORDERS = (1, 3, 5, 7, 9, 11, 13)
SETS = {"egauge_12": ("building main, 480 V, about 0.25 MVA", 277.0), "egauge_17": ("2.4 kV side, about 1.2 MVA", 1386.0),
        "egauge_19": ("16.5 kV line, about 5.5 MVA", 9526.0)}
TAG = "20250312T180000"


def main(root, out_json):
    res = {"captures_at_utc": TAG, "island_kw": 25.0, "meters": {}}
    for meter, (desc, vln) in SETS.items():
        d = Path(root) / f"{meter}_{TAG}" / "waveforms"
        per_phase = {}
        for ch in ("S1", "S2", "S3"):
            rows = []
            for p in sorted((d / f"{meter}-{ch}").rglob("*.parquet")):
                df = pd.read_parquet(p)
                t = df["t"].to_numpy().astype("datetime64[us]").astype(np.int64) / 1e6
                rows.append(W.harmonics(t - t[0], df["v"].to_numpy(float), ORDERS))
            a = np.array([[r[n] for n in ORDERS] for r in rows])
            per_phase[ch] = {"captures": len(rows), "median_a": dict(zip(map(str, ORDERS), np.median(a, axis=0).round(4).tolist())),
                             "sd_a": dict(zip(map(str, ORDERS), a.std(axis=0).round(4).tolist())),
                             "sd_of_change_between_captures_a": dict(zip(map(str, ORDERS), np.diff(a, axis=0).std(axis=0).round(4).tolist()))}
        i_island = 25000.0 / (3 * vln)
        res["meters"][meter] = {"description": desc, "per_phase": per_phase, "island_fundamental_a": i_island,
                                "island_harmonic_a": {"corrected input, 2% per harmonic": 0.02 * i_island,
                                                      "six-pulse rectifier, fifth (20%)": 0.20 * i_island, "six-pulse rectifier, seventh (14%)": 0.14 * i_island}}
    Path(out_json).write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    r = main(sys.argv[1], sys.argv[2])
    for m, v in r["meters"].items():
        print(f"\n{m} ({v['description']}): a 25 kW island is {v['island_fundamental_a']:.2f} A per phase; its harmonics: " + ", ".join(f"{k}: {x:.3f} A" for k, x in v["island_harmonic_a"].items()))
        for ch, p in v["per_phase"].items():
            f1 = p["median_a"]["1"]
            print(f"  {ch}: fundamental {f1:.1f} A | " + " | ".join(f"h{n}: {p['median_a'][n]:.2f} A ({100 * p['median_a'][n] / f1:.1f}%), spread {p['sd_a'][n]:.3f}, step {p['sd_of_change_between_captures_a'][n]:.3f}" for n in ("3", "5", "7", "11", "13")))
