"""Regenerate the frontier table and the map figure from params.yaml and, for the pulse
reference, the held-out result row named in params.yaml `pulse.reference` (falls back to
pulse_measurement.json when no reference is named).
Usage:  python -m island_detectability_map.run [OUT_DIR]

Origin: verification@f7a9006e
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from island_detectability_map import model

HERE = Path(__file__).parent


def pulse_reference(cfg, heldout):
    """The map's pulse reference from one named row of a held-out result file.

    cfg names the row (family, detector, series, rule, margin) and gives the mean load at the
    measurement point over the scored hours (m_ref_kva). The row must have had no false alarm in
    its scored hours, and the top of its 80%-detection range is used. No fallback: a missing row
    or a row with alarms raises, so a re-run on new data cannot silently anchor on a worse figure.
    """
    rows = heldout["families"][cfg["family"]][cfg["detector"]]["rows"]
    hits = [r for r in rows if r["family"] == cfg["series"] and r["rule"] == cfg["rule"]
            and r.get("class", "all") == "all" and r["margin"] == cfg["margin"]]
    if not hits:
        raise ValueError(f"no row for {cfg}")
    r = hits[0]
    if r["false_alarm_hours"]:
        raise ValueError(f"{r['false_alarm_hours']} false alarm hours in the scored hours of {cfg}")
    return dict(swing_ref_kva=float(max(r["swing80"])), m_ref_kw=float(cfg["m_ref_kva"]), t_ref_h=1.0)


def frontiers(p, meas, heldout=None):
    ms, bal, pu = p["mean_shift"], p["balance"], p["pulse"]
    if "reference" in pu:
        ref = pulse_reference(pu["reference"], heldout)
    else:
        ref = legacy_reference(pu, meas)
    return frontier_fns(p, ms, bal, pu, ref), ref


def legacy_reference(pu, meas):
    cands = [e[pu["level"]] for e in meas.get("realistic", []) if e["model"] == pu.get("signal_model") and e[pu["level"]]]
    if cands:      # correlated-lognormal straggler model, best segment length of the bank
        swing = min(cands)
    else:          # fall back to the fixed-cadence rectangle
        swing = next(e for e in meas["experiments"] if e["window_s"] == 3600 and e["case"] == "site_load_whitelisted")[pu["level"]]
    return dict(swing_ref_kva=swing, m_ref_kw=meas["mean_kva"], t_ref_h=1.0)


def frontier_fns(p, ms, bal, pu, ref):
    f = {
        "A energy mean shift, aggregate alone — residential-synthetic noise":
            lambda M: model.mean_shift_min_load(M, "residential_synthetic", ms["tpr"], ms["fpr"], ms["k"]),
        "A energy mean shift, aggregate alone — real mixed-load noise":
            lambda M: model.mean_shift_min_load(M, "real_mixed", ms["tpr"], ms["fpr"], ms["k"]),
        f"B balance vs downstream meters, e={bal['e_low']:.1%}": lambda M: model.balance_min_load(M, bal["e_low"], bal["kfac"]),
        f"B balance vs downstream meters, e={bal['e_high']:.1%}": lambda M: model.balance_min_load(M, bal["e_high"], bal["kfac"]),
    }
    for t in pu["t_hours"]:
        for d in pu["depth"]:
            for b in pu["beta"]:
                name = f"C pulse, unsmoothed, depth {d:.0%}, T={'1 h' if t == 1 else f'{t/24:.0f} d'}, noise~M^{b}"
                f[name] = (lambda M, d=d, b=b, t=t: model.pulse_min_load(M, d, b, t, **ref))
    return f


def table(p, f):
    rows = []
    for name, fn in f.items():
        row = {"method": name}
        for L in p["loads_kw"]:
            row[f"{L['kw']:g} kW"] = model.max_aggregation(fn, L["kw"])
        row["min load at 12.1 MW substation (kW)"] = fn(12100.0)
        rows.append(row)
    return rows


def fmt_kw(v):
    if not np.isfinite(v):
        return "below 1 kW"
    if v >= 1e7 * 0.99:
        return "> 10 GW"
    return f"{v/1000:.1f} MW" if v >= 1000 else f"{v:.0f} kW"


def figure(p, f, ref, out, meas=None, rc=None):
    ax_ = p["axes"]
    M = np.logspace(np.log10(ax_["m_min"]), np.log10(ax_["m_max"]), 200)
    fig, ax = plt.subplots(figsize=(9.5, 8.0))
    for (name, (lo, hi)), c in zip(p["tiers_kw"].items(), ["#f2f2f2", "#e6e6e6", "#d9d9d9"]):
        ax.axvspan(lo, hi, color=c, zorder=0)
        ax.text(np.sqrt(lo * hi), ax_["p_max"] * 0.72, name, ha="center", va="top", fontsize=8, color="#555")
    names = list(f)
    ax.plot(M, [f[names[0]](m) for m in M], color="#b2182b", lw=2, label="A  energy mean shift on the total alone (residential noise, measured ≤0.28 MW)")
    ax.plot(M, [f[names[1]](m) for m in M], color="#b2182b", lw=2, ls="--", label="A  same, real mixed-load noise (measured at 3.2 and 12.1 MW; extrapolated below)")
    ax.fill_between(M, [f[names[2]](m) for m in M], [f[names[3]](m) for m in M], color="#2166ac", alpha=0.25,
                    label="B  balance against downstream meters, 0.5–2% systematic error")
    pulse_1h = [n for n in names if n.startswith("C") and "T=1 h" in n and "depth 30%" in n]
    pulse_30d = [n for n in names if n.startswith("C") and "T=30 d" in n and "depth 30%" in n]
    ax.fill_between(M, [f[pulse_1h[0]](m) for m in M], [f[pulse_1h[1]](m) for m in M], color="#1b7837", alpha=0.30,
                    label="C  training pulse (wandering sawtooth), unsmoothed, 30% depth, 1 hour (band: noise ∝ M^0.5 … M^1)")
    ax.fill_between(M, [f[pulse_30d[0]](m) for m in M], [f[pulse_30d[1]](m) for m in M], color="#1b7837", alpha=0.12,
                    hatch="//", label="C  same, 30 days (T^-1/4 scaling assumed" + ("; not tested for an edge counter)" if rc else ")"))
    ax.plot([ref["m_ref_kw"]], [ref["swing_ref_kva"] / 0.3], "o", color="#1b7837", ms=7, zorder=5)
    ax.annotate((rc["label"] if rc else "measured, wandering\nsawtooth model"), (ref["m_ref_kw"], ref["swing_ref_kva"] / 0.3), xytext=(-78, 10),
                textcoords="offset points", fontsize=8, color="#1b7837")
    if meas and meas.get("by_cadence_band"):
        # measured 80%-detection swing by cadence band, shown as island size = swing / depth.
        # marker at 30% depth; bar from 45% (unsmoothed) to 10% (vendor-smoothed) depth.
        for e, mk, dx in zip(meas["by_cadence_band"], ["v", "s", "^"], [0.80, 1.0, 1.25]):
            sw, xm = e["swing80_zero_fa_kva"], ref["m_ref_kw"] * dx
            ax.plot([xm, xm], [sw / 0.45, sw / 0.10], color="k", lw=1.2, zorder=6)
            ax.plot([xm], [sw / 0.3], mk, color="k", mfc="w", ms=7, zorder=7,
                    label=f"measured, fixed cadence {e['band'].split(' (')[0]} (bar: pulse depth 45% … 10%)")
    for L in p["loads_kw"]:
        ax.axhline(L["kw"], color="k", lw=0.6, ls=":")
        ax.text(ax_["m_min"] * 1.1, L["kw"] * 1.08, L["label"].split(" [")[0], fontsize=8)
    ax.plot(M, M, color="#888", lw=0.8)
    ax.text(1500, 2300, "load = everything else on the conductor", rotation=36, fontsize=7, color="#666")
    ax.set(xscale="log", yscale="log", xlim=(ax_["m_min"], ax_["m_max"]), ylim=(ax_["p_min"], ax_["p_max"]),
           xlabel="mean load at the measurement point, kW (aggregation level)",
           ylabel="covert load, kW (detectable above each line)")
    ax.set_title("Where a covert compute load becomes visible, by method and aggregation level", fontsize=11)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=1, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(out_dir: Path):
    p = yaml.safe_load((HERE / "params.yaml").read_text())
    meas = json.loads((out_dir / p["pulse"]["measurement"]).read_text())
    rc = p["pulse"].get("reference")
    heldout = json.loads((out_dir / rc["file"]).read_text()) if rc else None
    f, ref = frontiers(p, meas, heldout)
    rows = table(p, f)
    (out_dir / "frontier_table.json").write_text(json.dumps(rows, indent=1))
    cols = [c for c in rows[0] if c != "method"]
    md = ["| Method | " + " | ".join(("max aggregation for " + c) if c.endswith(" kW") else c for c in cols) + " |",
          "|---|" + "---|" * len(cols)]
    for r in rows:
        md.append("| " + r["method"] + " | " + " | ".join(fmt_kw(r[c]) for c in cols) + " |")
    (out_dir / "frontier_table.md").write_text("\n".join(md) + "\n")
    figure(p, f, ref, out_dir / "island_detectability_map.png", None if rc else meas, rc)
    return rows, md


if __name__ == "__main__":
    od = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/20260930_island_detectability_map")
    print("\n".join(main(od)[1]))
