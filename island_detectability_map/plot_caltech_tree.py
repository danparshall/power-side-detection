"""Figure: the Caltech network file as sections joined by transformers, laid out left to right
by voltage (16.5 kV, 2.4 kV, 480 V), with each meter on the transformer it sits beside and its
week-mean real power. EXPLORATORY plotting script (graphviz), no tests.
Convo 20261002_caltech_login_and_hour_of_day_split, 2026-10-03 (second version: strict columns).

A "section" is a set of buses joined by closed lines, switches, breakers and fuses (no
transformer in between). Dashed grey: a tie the network file marks normally open (ties between 480 V boards are not drawn).

Usage: python -m island_detectability_map.plot_caltech_tree NETWORK_JSON WEEK_ROOT PLANT_POWER_DIR OUT_PNG [CHECKS_JSON]

Origin: verification@dfa939d5
"""
from __future__ import annotations

import json, subprocess, sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from island_detectability_map import caltech_tree as T


def mean_kw(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    return float(pd.read_csv(path)["v"].mean())


def meter_beside(t, adj, start_bus, skip_edge):
    """Walk from a transformer terminal along an unbranched run of closed elements and return
    the first metered element met (the meter that measures that transformer's flow)."""
    prev, cur = skip_edge, start_bus
    for _ in range(6):
        nxt = [(v, e) for v, e in adj[cur] if e["name"] != prev and e["cls"] != "Transformer"]
        if len(nxt) != 1:
            return None
        v, e = nxt[0]
        if e["name"] in t["meter_on"] and "/" not in t["meter_on"][e["name"]]:
            return t["meter_on"][e["name"]]
        prev, cur = e["name"], v
    return None


def main(net, week_root, plant_dir, out_png, checks_json=None):
    t = T.build(net)
    nv = lambda b: round(t["bus"][b].get("nominal_voltage") or 0)
    par = {b: b for b in t["bus"]}
    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    adj = defaultdict(list)
    for e in t["edges"]:
        adj[e["a"]].append((e["b"], e)); adj[e["b"]].append((e["a"], e))
        if e["cls"] != "Transformer":
            par[find(e["a"])] = find(e["b"])
    sec = defaultdict(list)
    for b in t["bus"]:
        sec[find(b)].append(b)
    trs = [e for e in t["edges"] if e["cls"] == "Transformer"]
    used = {find(e["a"]) for e in trs} | {find(e["b"]) for e in trs}
    week_root, plant_dir = Path(week_root), Path(plant_dir)
    kw, kva = {}, {}
    for m in {v.split("/")[0] for v in t["meter_on"].values()}:
        kw[m] = mean_kw(week_root / m / "magnitudes" / f"{m}-Mains_Power.csv") or mean_kw(plant_dir / f"{m}-Mains_Power.csv")
        c = week_root / "_cache" / f"{m}.json"
        if c.exists():
            kva[m] = json.loads(c.read_text()).get("mean_kva")
    def mlabel(m):
        s = m.replace("egauge_", "meter ")
        if kw.get(m) is not None:
            return s + f": {kw[m]:+,.0f} kW"
        if kva.get(m) is not None:
            return s + f": {kva[m]:,.0f} kVA"
        return s + ": no data"
    checks = json.loads(Path(checks_json).read_text()) if checks_json else None
    level = lambda r: max(nv(b) for b in sec[r])
    col = {9526: "#f5b7b1", 1386: "#fad7a0", 277: "#aed6f1", 240: "#aed6f1"}
    vname = {9526: "16.5 kV", 1386: "2.4 kV", 277: "480 V", 240: "415 V"}
    fed = {find(e["b"]) for e in trs}
    title = ("Caltech network file (2023-08-01 snapshot). Left to right: 16.5 kV grid connections, 2.4 kV, building boards at 480 V. "
             "Each arrow is a transformer, labelled with the meter beside it.\\l"
             "Numbers: mean real power for the week of 2025-03-10. Plus = power flowing in the arrow's direction; minus = flowing back (the board generates). "
             "kVA where a meter has no power register.\\l")
    out = ['digraph G { rankdir=LR; nodesep=0.22; ranksep=2.2; newrank=true; fontname="Helvetica"; labelloc=t; labeljust=l; fontsize=13; label="' + title + '"; '
           'node [fontname="Helvetica", fontsize=11, shape=box, style="filled,rounded"]; edge [fontname="Helvetica", fontsize=10];']
    ranks = defaultdict(list)
    for r in used:
        v = level(r); bs = sec[r]
        att = [n for b in bs for c, n, g in t["at_bus"].get(b, [])]
        loads = [a for a in att if a.startswith("load")]; gens = [a for a in att if a.startswith("gen")]; invs = [a for a in att if a.startswith("inv")]
        grid = any(b in t["roots"] for b in bs)
        lines = []
        if v > 5000:
            ms = sorted({t["meter_on"][e["name"]] for e in t["edges"] if e["name"] in t["meter_on"] and find(e["a"]) == r and e["cls"] != "Transformer"})
            lines.append("<b>16.5 kV grid connection</b>")
            if "egauge_19" in ms:
                lines += [mlabel("egauge_19") + " (whole line)",
                          f"<i>of which about {round(kw['egauge_19'] - kw['egauge_21'], -2):,.0f} kW goes to a load with no meter</i>"]
            else:
                lines += [mlabel(m) for m in ms]
        elif v > 1000:
            if r in fed and checks:
                n = checks["node_2400V"]; best = n["closures_best_first"][0]
                lines += ["<b>2.4 kV section</b>", f"in through the substation: {n['inflow_mean_kw']:,.0f} kW",
                          f"out to ten building mains: {n['building_mains_mean_kw']:,.0f} kW",
                          f"inverter board feeds in: {-n['mean_kw']['egauge_22']:,.0f} kW",
                          f"<i>left over: {best['mean_kw']:+,.0f} kW (transformer losses, anything unmetered)</i>"]
            elif r in fed:
                lines.append("<b>2.4 kV section</b>")
            else:
                lines += ["<b>2.4 kV section</b>", "<i>cut off from the substation in this file;</i>", "<i>fed from somewhere not shown</i>"]
        else:
            lines.append(f"<b>{vname.get(v, '480 V')} board</b>")
            tail = []
            if loads: tail.append(f"{len(loads)} load{'s' if len(loads) > 1 else ''}")
            if invs: tail.append(f"{len(invs)} inverters")
            if gens: tail.append("standby generator")
            lines.append("<i>" + ", ".join(tail) + "</i>" if tail else "<i>nothing attached</i>")
        ranks[9526 if v > 5000 else 1386 if v > 1000 else 277].append(r)
        out.append(f'"{r}" [fillcolor="{col.get(v, "#eeeeee")}", label=<{"<br/>".join(lines)}>];')
    for v, rs in ranks.items():
        out.append("{rank=same; " + "; ".join(f'"{r}"' for r in rs) + ";}")
    for e in trs:
        ma = meter_beside(t, adj, e["a"], e["name"]); mb = meter_beside(t, adj, e["b"], e["name"])
        lab = [f"{e['name']}, {e['kva']:g} kVA"] + [mlabel(m) for m in (ma, mb) if m]
        if not (ma or mb):
            lab.append("no meter in service")
        out.append(f'"{find(e["a"])}" -> "{find(e["b"])}" [label="{chr(92) + "n".join(lab) if False else (chr(92) + "n").join(lab)}", arrowsize=0.7, penwidth=1.3];')
    for e in t["opened"]:
        a, b = find(e["a"]), find(e["b"])
        if a != b and a in used and b in used and max(level(a), level(b)) > 1000:      # ties between 480 V boards are left out (clutter)
            mt = t["meter_on"].get(e["name"])
            lab = f"open tie ({mlabel(mt)})" if mt else "open tie"
            out.append(f'"{a}" -> "{b}" [style=dashed, color="#bbbbbb", fontcolor="#888888", fontsize=9, dir=none, constraint=false, label="{lab}"];')
    out.append("}")
    dot = Path(out_png).with_suffix(".dot")
    dot.write_text("\n".join(out))
    subprocess.run(["dot", "-Tpng", "-Gdpi=110", str(dot), "-o", str(out_png)], check=True)
    dot.unlink()
    return kw


if __name__ == "__main__":
    main(*sys.argv[1:6])
