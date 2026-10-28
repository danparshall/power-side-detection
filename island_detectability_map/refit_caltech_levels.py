"""Refit the Caltech background-against-load fits with the 2.4 kV meter's point fixed (ledger row
"WRONG POINT IN A FIT", 2026-10-03). EXPLORATORY; reads the committed caltech_levels.json only
(run_caltech_levels.fit, unchanged), writes caltech_levels_refit.json.

egauge_17 measures the net of about 1,661 kW of building load (the ten building mains on its
section) and about 711 kW of on-site generation (inverter board, meter 22), so its 1.16 MVA is not
the load behind its background. Three versions: as committed; the point dropped; the point at gross
load (1,661 kW, phases scaled in proportion; kW taken as kVA, power factor near 1 on these boards).
Usage: python -m island_detectability_map.refit_caltech_levels RESULTS_DIR

Origin: verification@fdff18d2
"""
from __future__ import annotations

import json, sys
from pathlib import Path

from island_detectability_map.run_caltech_levels import fit

GROSS_17_KVA = 1661.0     # power out through the ten building mains (caltech_tree / run_caltech_checks; origin: verification@fdff18d2)


def fits(sel, key):
    return {"sum vs total load": fit([r["mean_kva"] for r in sel], [r["band_rms_kva"][key]["sum"]["all"] for r in sel]),
            "one phase vs that phase's load": fit([v for r in sel for v in r["mean_kva_per_phase"]],
                                                  [r["band_rms_kva"][key][f"phase_{p}"]["all"] for r in sel for p in "abc"]),
            "residual vs that phase's load": fit([v for r in sel for v in r["mean_kva_per_phase"]],
                                                 [r["band_rms_kva"][key][f"resid_{p}"]["all"] for r in sel for p in "abc"])}


def main(res_dir: Path):
    d = json.loads((res_dir / "caltech_levels.json").read_text())
    good = [r for r in d["meters"] if not r.get("no_data")]
    gross = []
    for r in good:
        if r["meter"] == "egauge_17":
            f = GROSS_17_KVA / r["mean_kva"]
            r = dict(r, mean_kva=GROSS_17_KVA, mean_kva_per_phase=[v * f for v in r["mean_kva_per_phase"]])
        gross.append(r)
    no18 = lambda rs: [r for r in rs if r["meter"] not in ("egauge_18", "egauge_20")]
    out = {"note": __doc__.split("Usage")[0].strip(), "fits": {}}
    for key in ("0.02-0.1Hz", "0.1-0.5Hz"):
        out["fits"][key] = {}
        for label, rs in (("as committed", good), ("egauge_17 dropped", [r for r in good if r["meter"] != "egauge_17"]),
                          ("egauge_17 at gross load", gross)):
            out["fits"][key][label] = {"all meters": fits(rs, key), "without 18, 20": fits(no18(rs), key)}
    (res_dir / "caltech_levels_refit.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    o = main(Path(sys.argv[1]))
    for key, v in o["fits"].items():
        for label, w in v.items():
            for sub, f in w.items():
                print(f"{key:<11} {label:<24} {sub:<15}", " | ".join(f"{k.split(' vs')[0]}: {x['exponent']:.2f}±{x['se']:.2f} (n{x['n']}, ×{x['scatter_factor']:.1f})" for k, x in f.items()))
