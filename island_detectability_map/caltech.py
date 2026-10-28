"""Loader for the Caltech SoCal 28-bus dataset's 1-second magnitude files. Tests: tests/test_caltech.py.

Layout read: <dir>/magnitudes/<register data_file>.csv with columns t,v (one row per second,
timestamps in UTC without a zone suffix), and a network JSON whose EgaugeMeter entries list
registers {name, element, unit, data_file}. Facts the loader depends on (results note
20261002_caltech_meter_map_and_first_pull.md):

* A register's phase is the suffix of its element ('cb_140.a', 'bus_1098.ag'), not its
  channel number: four meters are wired out of order.
* Two substation meters measure voltage line-to-line on two channels. V_ab is stored positive,
  V_bc negative, and V_ca is a formula on RMS values that means nothing. The loader uses
  mean(|V_ab|, |V_bc|) / sqrt(3) as every phase's line-to-neutral voltage and never reads V_ca.
* The API returns an empty file, with no error, when it has no data.
* Power registers are real power in kW (the metadata says W). Against current x voltage they
  give the power factor, so they are returned beside the series, not used as a check on it.

Nothing is filled. A second missing from any channel is NaN in all three phases and False in
`mask`; the report lists what was missing where.

Origin: verification@d36e6d97
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from island_detectability_map import hours as _hours

PHASES = ("a", "b", "c")


@dataclass
class Meter:
    key: str                      # 'egauge_11', or 'egauge_16/line_403' for a sub-circuit
    meter: str
    element: str
    current: dict                 # phase -> register file name
    voltage: dict                 # 'a','b','c' (line-to-neutral) or 'ab','bc' (line-to-line) -> file name
    v_kind: str                   # 'ln' or 'll'
    power: str | None = None      # real-power register for this element, if any (kW)
    i_rating: float | None = None


@dataclass
class Loaded:
    t0: np.datetime64
    i: np.ndarray                 # (3, n) A, phases a, b, c
    v_ln: np.ndarray              # (3, n) V line-to-neutral
    kva: np.ndarray               # (3, n) i * v_ln / 1e3
    total: np.ndarray             # (n,) sum of the three phases
    mask: np.ndarray              # (n,) True where every channel has a value
    p_kw: np.ndarray | None       # meter's own real-power register, if present
    report: dict = field(default_factory=dict)


def meter_map(network_json: Path) -> dict[str, Meter]:
    net = json.loads(Path(network_json).read_text())
    out: dict[str, Meter] = {}
    for m in net["EgaugeMeter"]:
        regs = [r for r in m["registers"] if r.get("element") and r.get("data_file")]
        volt_ln = {r["element"].rsplit(".", 1)[1][0]: r["data_file"] for r in regs
                   if r["unit"] == "V" and r["element"].rsplit(".", 1)[1] in ("ag", "bg", "cg")}
        volt_ll = {r["element"].rsplit(".", 1)[1]: r["data_file"] for r in regs
                   if r["unit"] == "V" and r["element"].rsplit(".", 1)[1] in ("ab", "bc")}
        voltage, v_kind = (volt_ln, "ln") if len(volt_ln) == 3 else (volt_ll, "ll")
        power = {r["element"]: r["data_file"] for r in regs if r["unit"] == "W"}
        groups: dict[str, dict] = {}
        rating: dict[str, float] = {}
        for r in regs:
            if r["unit"] != "A" or "." not in r["element"]:
                continue
            el, ph = r["element"].rsplit(".", 1)
            if ph in PHASES:
                groups.setdefault(el, {})[ph] = r["data_file"]
                if r.get("I_rating"):
                    rating[el] = float(r["I_rating"])
        full = {el: g for el, g in groups.items() if set(g) == set(PHASES)}
        if not full or len(voltage) < (3 if v_kind == "ln" else 2):
            continue
        mains = [el for el in full if any(c.endswith(("-CT1", "-CT2", "-CT3")) for c in full[el].values())]
        main = mains[0] if mains else next(iter(full))
        for el, g in full.items():
            key = m["name"] if el == main else f"{m['name']}/{el}"
            out[key] = Meter(key, m["name"], el, {p: g[p] for p in PHASES}, dict(voltage), v_kind,
                             power.get(el), rating.get(el))
    return out


def _read(mag_dir: Path, name: str, index: pd.DatetimeIndex, report: dict) -> np.ndarray:
    f = Path(mag_dir) / f"{name}.csv"
    if not f.exists() or f.stat().st_size == 0:
        report["empty_files"].append(name)
        report["missing_seconds"][name] = len(index)
        report["duplicate_rows"][name] = 0
        return np.full(len(index), np.nan)
    df = pd.read_csv(f)
    t = pd.to_datetime(df["t"])
    dup = t.duplicated(keep="first")
    report["duplicate_rows"][name] = int(dup.sum())
    s = pd.Series(df["v"].to_numpy(float)[~dup.to_numpy()], index=t[~dup]).reindex(index)
    report["missing_seconds"][name] = int(s.isna().sum())
    return s.to_numpy()


def load_element(mag_dir: Path, m: Meter, start: str, end: str) -> Loaded:
    """Per-phase series for one metered element over [start, end) on a 1-second grid."""
    index = pd.date_range(start, end, freq="1s", inclusive="left")
    report = {"empty_files": [], "missing_seconds": {}, "duplicate_rows": {}}
    i = np.vstack([_read(mag_dir, m.current[p], index, report) for p in PHASES])
    if m.v_kind == "ln":
        v = np.vstack([_read(mag_dir, m.voltage[p], index, report) for p in PHASES])
    else:
        vab, vbc = (np.abs(_read(mag_dir, m.voltage[k], index, report)) for k in ("ab", "bc"))
        v = np.tile(0.5 * (vab + vbc) / np.sqrt(3.0), (3, 1))
    mask = np.isfinite(i).all(axis=0) & np.isfinite(v).all(axis=0)
    i = np.where(mask, i, np.nan)
    v = np.where(mask, v, np.nan)
    kva = i * v / 1e3
    p_kw = None
    if m.power:
        side = {"empty_files": [], "missing_seconds": {}, "duplicate_rows": {}}   # the power register never masks
        p_kw = _read(mag_dir, m.power, index, side)
        if side["empty_files"]:
            p_kw = None
    bad = np.flatnonzero(~mask)
    gaps = []
    if len(bad):
        cut = np.flatnonzero(np.diff(bad) > 1)
        for a, b in zip(np.r_[0, cut + 1], np.r_[cut, len(bad) - 1]):
            gaps.append((str(index[bad[a]].isoformat()), int(bad[b] - bad[a] + 1)))
    report["masked_seconds"] = int((~mask).sum())
    report["gaps"] = gaps
    return Loaded(np.datetime64(index[0], "s"), i, v, kva, kva.sum(axis=0), mask, p_kw, report)


def hour_labels(start: str, n_hours: int, tz: str = "America/Los_Angeles") -> list[str]:
    """hours.hour_labels for a UTC start time given as an ISO string."""
    t0_ns = int(pd.Timestamp(start, tz="UTC").value)
    return _hours.hour_labels(t0_ns, n_hours, tz)
