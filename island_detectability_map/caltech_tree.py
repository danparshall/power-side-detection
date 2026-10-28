"""The Caltech network file as a tree: which meter sits on which element, what is downstream
of it, and which meters should add up to which. EXPLORATORY (no tests yet); written to answer
"how far down do we have measurements, and do the branches sum to the node?"

Buses are nodes. Lines, switches, breakers, fuses, VFIs and transformers are edges (an edge to
a bus marked normally open is dropped). The tree is rooted at the GridPower buses.

Origin: verification@dfa939d5
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

EDGE_CLASSES = ("Line", "Switch", "SwitchMultiPosition", "CB", "Fuse", "VFI", "Transformer")


def build(network_json):
    d = json.loads(Path(network_json).read_text())
    bus = {b["name"]: b for b in d["Bus"]}
    edges, opened = [], []
    for cls in EDGE_CLASSES:
        for e in d.get(cls, []):
            for tb in e["tbus"]:
                st = tb.get("status") or tb.get("nominal_status")
                (opened if st == "NO" else edges).append({"name": e["name"], "cls": cls, "a": e["fbus"], "b": tb["name"],
                                                           "kva": (e.get("kva") or [None])[0] if cls == "Transformer" else None})
    adj = defaultdict(list)
    for e in edges:
        adj[e["a"]].append((e["b"], e)); adj[e["b"]].append((e["a"], e))
    roots = [g["bus"] for g in d.get("GridPower", [])]
    parent, order, extra = {r: None for r in roots}, [], []
    q = deque(roots)
    while q:
        u = q.popleft(); order.append(u)
        for v, e in adj[u]:
            if v not in parent:
                parent[v] = (u, e); q.append(v)
            elif parent[u] is None or parent[u][1] is not e:
                if parent.get(v) is None or parent[v][1] is not e:
                    extra.append(e["name"])
    children = defaultdict(list)
    for v, pe in parent.items():
        if pe:
            children[pe[0]].append((v, pe[1]))
    at_bus = defaultdict(list)
    for cls in ("Load", "Generator", "Inverter"):
        for x in d.get(cls, []):
            at_bus[x["bus"]].append((cls, x["name"], x.get("groups", [])))
    meter_on = {}                                   # element name -> meter key (main entries and sub-circuits)
    for m in d["EgaugeMeter"]:
        for r in m["registers"]:
            if r.get("unit") == "A" and r.get("element", "").endswith(".a"):
                el = r["element"][:-2]
                main = r["name"].startswith("CT")
                meter_on[el] = m["name"] if main else f"{m['name']}/{el}"
    return dict(d=d, bus=bus, edges=edges, opened=opened, parent=parent, children=children, roots=roots,
                at_bus=at_bus, meter_on=meter_on, unreached=[b for b in bus if b not in parent], loops=sorted(set(extra)))


def edge_to(t, busname):
    return t["parent"][busname][1] if t["parent"].get(busname) else None


def downstream(t, edge_name):
    """For a metered edge: nearest metered edges below it, and loads/generators below it that
    no lower meter covers."""
    start = next(v for v, pe in t["parent"].items() if pe and pe[1]["name"] == edge_name)
    kids, bare = [], []
    stack = [start]
    while stack:
        u = stack.pop()
        bare += [x[1] for x in t["at_bus"].get(u, [])]
        for v, e in t["children"].get(u, []):
            if e["name"] in t["meter_on"]:
                kids.append(t["meter_on"][e["name"]])
            else:
                stack.append(v)
    return kids, bare


def collapsed(t, levels_json=None):
    """Graph with runs of plain series elements merged. Returns (nodes, links): nodes are buses
    that branch, end, or carry a load/generator/inverter; a link lists the elements between two
    such buses in order."""
    from collections import defaultdict
    adj = defaultdict(list)
    for e in t["edges"]:
        adj[e["a"]].append((e["b"], e, False)); adj[e["b"]].append((e["a"], e, False))
    for e in t["opened"]:
        adj[e["a"]].append((e["b"], e, True)); adj[e["b"]].append((e["a"], e, True))
    keep = {b for b in t["bus"] if len(adj[b]) != 2 or t["at_bus"].get(b) or b in t["roots"]}
    links, seen = [], set()
    for u in keep:
        for v, e, op in adj[u]:
            if (u, e["name"]) in seen:
                continue
            chain, prev, cur, is_open = [e], u, v, op
            seen.add((u, e["name"]))
            while cur not in keep:
                nxt = [(w, e2, o2) for w, e2, o2 in adj[cur] if e2 is not chain[-1]]
                if not nxt:
                    break
                w, e2, o2 = nxt[0]
                chain.append(e2); is_open = is_open or o2; prev, cur = cur, w
            seen.add((cur, chain[-1]["name"]))
            links.append({"a": u, "b": cur, "elements": chain, "open": is_open})
    return keep, links
