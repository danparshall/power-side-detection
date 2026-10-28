"""How often do unrelated meters jump at the same second by chance? (Motivation:
once a site is flagged, look for the same pause-and-restart times at
other sites.) Tests run_sync's assumption that step events at different meters are independent,
so chance coincidences = rate_i x rate_j x window.

EXPLORATORY, no tests of its own. Tested pieces: caltech.load_element, phases.imbalance; the step
statistic is run_sync.upsteps (k = 3 s means either side, transition second skipped), computed here
NaN-safely (a step is invalid if any of its 7 seconds is missing).

Data: Caltech week 2025-03-10..17, 1-second magnitudes, seven 480 V building mains on the same 2.4 kV
section (siblings: they share the substation's voltage, the worst case for chance coincidence;
sites in different towns share less). Series: each meter's three-phase sum (kVA) and the three
residuals (phase minus the mean of the other two). Events: up-steps >= size, one per 5 s
(run_sync.events). Coincidence: an event at meter i with an event at meter j within +-2 s
(run_sync.WINDOW_S = 4 s span, 5 one-second slots).
Also: voltage up-steps on each meter (line-to-neutral, mean of the three phases), to see whether
coincidences are grid voltage events (which move every meter's kVA together).

Usage: python -m island_detectability_map.run_coincidence CALTECH_ROOT NETWORK_JSON OUT_DIR

Origin: verification@6875ceb7
"""
from __future__ import annotations

import itertools, json, sys
from pathlib import Path

import numpy as np

from island_detectability_map import caltech, phases, run_sync as S

START, END = "2025-03-10T00:00:00", "2025-03-17T00:00:00"
METERS = ["egauge_3", "egauge_4", "egauge_7", "egauge_9", "egauge_11", "egauge_12", "egauge_13"]
SIZES = {"sum": [5.0, 10.0, 20.0], "resid": [2.0, 5.0, 10.0]}
HALF_W = 2            # +-2 s
K = 3


def steps(x, k=K):
    """run_sync.upsteps, NaN-safe: step[t] = mean(x[t+1:t+1+k]) - mean(x[t-k:t])."""
    ok = np.isfinite(x)
    z = np.where(ok, x, 0.0)
    c = np.concatenate([[0.0], np.cumsum(z)])
    cv = np.concatenate([[0], np.cumsum(~ok)])
    t = np.arange(k, len(x) - k - 1)
    s = (c[t + 1 + k] - c[t + 1]) / k - (c[t] - c[t - k]) / k
    bad = (cv[t + 1 + k] - cv[t - k]) > 0          # any missing second in t-k .. t+k
    s[bad] = np.nan
    return t, s, int((~bad).sum())


def event_times(x, size):
    t, s, valid = steps(x)
    s = np.nan_to_num(s, nan=-np.inf)
    return S.events(t, s, size), valid


def coincide(a, b, w=HALF_W):
    """Number of events in a with at least one event in b within +-w seconds."""
    if len(a) == 0 or len(b) == 0:
        return 0
    j = np.searchsorted(b, a)
    lo = np.abs(a - b[np.clip(j - 1, 0, len(b) - 1)]) <= w
    hi = np.abs(b[np.clip(j, 0, len(b) - 1)] - a) <= w
    return int(np.sum(lo | hi))


def main(root, net, out_dir):
    mm = caltech.meter_map(net)
    ev, valid, volt = {}, {}, {}
    for m in METERS:
        d = caltech.load_element(Path(root) / m / "magnitudes", mm[m], START, END)
        ser = {"sum": d.kva.sum(axis=0), **{f"resid_{p}": phases.imbalance(d.kva, k) for k, p in enumerate("abc")}}
        vmean = d.v_ln.mean(axis=0)
        volt[m] = {"v_mean": float(np.nanmean(vmean))}
        for fam, sizes in SIZES.items():
            keys = ["sum"] if fam == "sum" else ["resid_a", "resid_b", "resid_c"]
            for size in sizes:
                ts, vs = [], []
                for k in keys:
                    e, v = event_times(ser[k], size)
                    ts.append(e); vs.append(v)
                ev[(m, fam, size)] = np.unique(np.concatenate(ts)) if ts else np.array([])
                valid[(m, fam, size)] = min(vs)
        # voltage steps of 0.5% of nominal, either direction
        t, s, _ = steps(vmean)
        thr = 0.005 * volt[m]["v_mean"]
        volt[m]["events"] = np.unique(np.concatenate([S.events(t, np.nan_to_num(s, nan=-np.inf), thr),
                                                      S.events(t, np.nan_to_num(-s, nan=-np.inf), thr)]))
    out = {"design": __doc__.split("Data:")[1].split("Usage")[0].strip(), "meters": METERS, "rows": [], "per_meter": {}}
    for (m, fam, size), e in ev.items():
        out["per_meter"][f"{m}|{fam}|{size}"] = {"events": len(e), "valid_days": valid[(m, fam, size)] / 86400.0,
                                                 "per_day": len(e) / (valid[(m, fam, size)] / 86400.0)}
    for fam, sizes in SIZES.items():
        for size in sizes:
            obs = exp = 0.0
            pairs = []
            for a, b in itertools.combinations(METERS, 2):
                ea, eb = ev[(a, fam, size)], ev[(b, fam, size)]
                T = min(valid[(a, fam, size)], valid[(b, fam, size)])
                o = coincide(ea, eb)
                x = len(ea) * len(eb) * (2 * HALF_W + 1) / T
                # coincidences that are also a voltage step at either meter (+-2 s)
                both = ea[[np.any(np.abs(eb - t) <= HALF_W) for t in ea]] if len(ea) and len(eb) else np.array([])
                vco = sum(1 for t in both if np.any(np.abs(volt[a]["events"] - t) <= HALF_W) or np.any(np.abs(volt[b]["events"] - t) <= HALF_W))
                pairs.append({"pair": [a, b], "events": [len(ea), len(eb)], "observed": o, "expected_if_independent": x,
                              "with_voltage_step": vco})
                obs += o; exp += x
            # events seen at >= 3 meters within +-2 s (fleet-wide)
            allt = np.sort(np.concatenate([ev[(m, fam, size)] for m in METERS]))
            multi = 0
            for t in np.unique(allt):
                n = sum(np.any(np.abs(ev[(m, fam, size)] - t) <= HALF_W) for m in METERS)
                multi += int(n >= 3)
            out["rows"].append({"family": fam, "size_kva": size, "pairs": len(pairs), "observed_total": obs,
                                "expected_total_if_independent": exp, "ratio": obs / exp if exp else None,
                                "pairs_detail": pairs, "event_times_seen_at_3plus_meters": multi})
    out["voltage_steps_per_meter"] = {m: len(volt[m]["events"]) for m in METERS}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "coincidence_caltech.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    r = main(sys.argv[1], sys.argv[2], sys.argv[3])
    for row in r["rows"]:
        vs = sum(p["with_voltage_step"] for p in row["pairs_detail"])
        print(f"{row['family']:<6} >= {row['size_kva']:>4} kVA | coincidences over {row['pairs']} pairs: observed {row['observed_total']:.0f}, "
              f"independent {row['expected_total_if_independent']:.2f}, ratio {row['ratio'] if row['ratio'] is None else round(row['ratio'], 1)} "
              f"| with a voltage step {vs} | times at 3+ meters {row['event_times_seen_at_3plus_meters']}")
    print("voltage steps (0.5%) per meter:", r["voltage_steps_per_meter"])
    print("events/day:", {k: round(v["per_day"], 1) for k, v in r["per_meter"].items()})
