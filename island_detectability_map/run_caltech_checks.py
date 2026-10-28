"""Accounting checks on the Caltech data: do the meters add up?
Writes caltech_checks.json. EXPLORATORY, no tests.

1. Across a substation transformer: 2.4 kV-side kVA over 16.5 kV-side kVA.
2. At the 2.4 kV section: real power in through the two substation transformers against real
   power out through every metered 480 V board on that section (1-minute means). Which boards
   belong to the section is tested, not assumed: every subset of the doubtful ones is tried.
   Then the weight at which the solar/plant board (meter 16) fits best is fitted.
3. At the top: the 16.5 kV line meter against the one metered branch below it.
4. Inside one meter, from raw waveforms: sum of mean(v*i) against the power register, and
   sum of Vrms*Irms against the product of the magnitude files.
5. Grid frequency per waveform capture, and agreement between two meters.

Usage: python -m island_detectability_map.run_caltech_checks NETWORK_JSON WEEK_ROOT PLANT_POWER_DIR WAVE_ROOT OUT_JSON

Origin: verification@dfa939d5
"""
from __future__ import annotations

import itertools, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from island_detectability_map import caltech, caltech_wave as W

S, E = "2025-03-10T00:00:00", "2025-03-17T00:00:00"
SECTION = [f"egauge_{k}" for k in (3, 4, 7, 8, 9, 10, 11, 12, 13, 14)]      # building mains on the 2.4 kV section in the network file
DOUBTFUL = ["egauge_16", "egauge_22", "egauge_24", "egauge_5"]


def m1(x):
    return np.nanmean(x[: len(x) // 60 * 60].reshape(-1, 60), axis=1)


def power_file(path, idx):
    if not path.exists() or path.stat().st_size == 0:
        return None
    df = pd.read_csv(path)
    s = pd.Series(df["v"].to_numpy(float), index=pd.to_datetime(df["t"]))
    return s[~s.index.duplicated()].reindex(idx).to_numpy()


def main(net, week_root, plant_dir, wave_root, out_json):
    week_root, plant_dir, wave_root = Path(week_root), Path(plant_dir), Path(wave_root)
    mm = caltech.meter_map(net)
    idx = pd.date_range(S, E, freq="1s", inclusive="left")
    P, K = {}, {}
    for n in SECTION + ["egauge_5", "egauge_17", "egauge_18", "egauge_19", "egauge_20", "egauge_21"]:   # P[...] are 1-minute means
        d = caltech.load_element(week_root / n / "magnitudes", mm[n], S, E)
        K[n] = m1(d.total); P[n] = None if d.p_kw is None else m1(d.p_kw)
    for n in ("egauge_16", "egauge_22", "egauge_24"):
        P[n] = m1(power_file(plant_dir / f"{n}-Mains_Power.csv", idx))
    res = {"time_range_utc": [S, E], "averaging": "1-minute means"}
    res["transformer"] = {}
    for hv, lv in (("egauge_21", "egauge_17"), ("egauge_20", "egauge_18")):
        r = K[lv] / K[hv]
        res["transformer"][f"{lv}/{hv}"] = {"kva_ratio_median": float(np.nanmedian(r)), "kva_ratio_5_95": [float(v) for v in np.nanpercentile(r, [5, 95])]}
    inflow = P["egauge_21"] + P["egauge_20"]; base = sum(P[n] for n in SECTION)
    rows = []
    for k in range(len(DOUBTFUL) + 1):
        for combo in itertools.combinations(DOUBTFUL, k):
            out = base + sum(P[c] for c in combo); r = inflow - out; ok = np.isfinite(r)
            rows.append({"extra_boards": list(combo), "mean_kw": float(np.nanmean(r)), "sd_kw": float(np.nanstd(r)),
                         "rms_kw": float(np.sqrt(np.nanmean(r ** 2))), "corr": float(np.corrcoef(inflow[ok], out[ok])[0, 1])})
    rows.sort(key=lambda x: x["rms_kw"])
    res["node_2400V"] = {"inflow_mean_kw": float(np.nanmean(inflow)), "building_mains_mean_kw": float(np.nanmean(base)),
                         "mean_kw": {n: float(np.nanmean(P[n])) for n in SECTION + DOUBTFUL}, "closures_best_first": rows}
    # what the wobble in the best closure follows: let the solar/plant board (meter 16) count at a fitted weight
    r0 = inflow - base - P["egauge_22"]; ok = np.isfinite(r0 + P["egauge_16"])
    A = np.vstack([np.ones(ok.sum()), P["egauge_16"][ok]]).T
    co = np.linalg.lstsq(A, r0[ok], rcond=None)[0]; left = r0[ok] - A @ co
    res["node_2400V"]["meter_16_weight_fit"] = {"constant_kw": float(co[0]), "weight": float(co[1]), "sd_left_kw": float(left.std()),
                                                "minute_to_minute_change_sd_kw": float(np.diff(left).std()),
                                                "meter_16_p01_p99_kw": [float(v) for v in np.nanpercentile(P["egauge_16"], [1, 99])],
                                                "metered_flow_kw": float(np.nanmean(inflow) + np.nanmean(base) + abs(np.nanmean(P["egauge_22"])))}
    # what the balance could see of an UNMETERED load: its own steps, dips and drift (1-minute means)
    def events(r):
        d = np.diff(r); d = d[np.isfinite(d)]; days = len(r) / 1440.0
        c = np.cumsum(np.r_[0, np.nan_to_num(r, nan=np.nanmean(r))]); w = 3; tt = np.arange(w, len(r) - 2 * w)
        dip = 0.5 * ((c[tt] - c[tt - w]) / w + (c[tt + 2 * w] - c[tt + w]) / w) - (c[tt + w] - c[tt]) / w
        daily = [float(np.nanmean(r[i * 1440:(i + 1) * 1440])) for i in range(len(r) // 1440)]
        hourly = np.nanmean(r[: len(r) // 60 * 60].reshape(-1, 60), axis=1)
        return {"mean_kw": float(np.nanmean(r)), "sd_kw": float(np.nanstd(r)), "daily_means_kw": daily, "daily_sd_kw": float(np.std(daily)),
                "hourly_sd_kw": float(np.nanstd(hourly)), "hourly_min_max_kw": [float(np.nanmin(hourly)), float(np.nanmax(hourly))],
                "upsteps_per_day": {str(a): float((d >= a).sum() / days) for a in (10, 15, 20, 25)},
                "three_minute_dips_per_day": {str(a): float((dip >= a).sum() / days) for a in (10, 15, 20)}}
    res["node_2400V"]["balance_events"] = {"as_network_file": events(inflow - base - P["egauge_22"] - P["egauge_16"]),
                                           "meter_16_at_fitted_weight": events(inflow - base - P["egauge_22"] - co[1] * P["egauge_16"])}
    flows = [P["egauge_21"], P["egauge_22"]] + [P[n] for n in SECTION]
    rss = float(np.sqrt(sum(np.nanmean(f) ** 2 for f in flows)))
    res["node_2400V"]["level_uncertainty_from_meter_class_kw"] = {"0.5%": 0.005 * rss, "0.2%": 0.002 * rss, "note": "root-sum-square of each meter's class times its mean flow, errors taken as independent"}
    # Test: switch a 25 kW island on. A before/after difference cancels every static term
    # (loss constant, gain errors). How often does each series jump up by that much and stay up, with no island?
    step = 25.0
    ser = {"node balance, as the network file gives it": inflow - base - P["egauge_22"] - P["egauge_16"],
           "node balance, meter 16 at fitted weight": inflow - base - P["egauge_22"] - co[1] * P["egauge_16"]}
    for n in SECTION + ["egauge_5"]:
        ser[f"building main {n}"] = P[n]
    ser["substation egauge_21"] = P["egauge_21"]; ser["16.5 kV line egauge_19"] = P["egauge_19"]
    turn = {}
    for name, x in ser.items():
        turn[name] = {"mean_kw": float(np.nanmean(x))}
        for w in (5, 15, 60):
            c = np.cumsum(np.r_[0, np.nan_to_num(x, nan=np.nanmean(x))]); tt = np.arange(w, len(x) - w)
            diff = (c[tt + w] - c[tt]) / w - (c[tt] - c[tt - w]) / w
            ev, last = 0, -10 ** 9
            for i in np.flatnonzero(diff >= step):
                ev += i - last > w; last = i
            turn[name][f"{w}min"] = {"sd_kw": float(diff.std()), "largest_kw": float(diff.max()), "jumps_ge_step_per_week": int(ev)}
    res["turn_on_test"] = {"step_kw": step, "statistic": "mean of the next W minutes minus mean of the previous W minutes, 1-minute real power",
                           "series": turn}
    res["top"] = {"egauge_19_kw": float(np.nanmean(P["egauge_19"])), "egauge_21_kw": float(np.nanmean(P["egauge_21"])),
                  "unmetered_remainder_kw": float(np.nanmean(P["egauge_19"] - P["egauge_21"]))}
    # 4. inside one meter, from waveforms
    wdir = wave_root / "egauge_19_20250312T180000" / "waveforms"; mdir = week_root / "egauge_19" / "magnitudes"
    rd = lambda n: pd.read_csv(mdir / f"egauge_19-{n}.csv").assign(t=lambda d: pd.to_datetime(d["t"])).set_index("t")["v"]
    mag = {n: rd(n) for n in ("CT1", "CT2", "CT3", "L1", "L2", "L3", "Mains_Power")}
    acc = []
    for p in sorted((wdir / "egauge_19-S1").rglob("*.parquet")):
        key = p.stem[:19]; cap = {}
        for ch in ("S1", "S2", "S3", "L1", "L2", "L3"):
            f = list((wdir / f"egauge_19-{ch}").rglob(key + "*.parquet"))
            if f:
                df = pd.read_parquet(f[0]); cap[ch] = (df["t"].to_numpy().astype("datetime64[us]").astype(np.int64) / 1e6, df["v"].to_numpy(float))
        if len(cap) < 6:
            continue
        pw = sw = 0.0
        for k in (1, 2, 3):
            tv, v = cap[f"L{k}"]; ti, i = cap[f"S{k}"]
            a = max(tv[0], ti[0]); b = a + int((min(tv[-1], ti[-1]) - a) * 60) / 60.0
            tu = np.arange(a, b, 1 / 4800.0); vu = CubicSpline(tv, v)(tu); iu = CubicSpline(ti, i)(tu)
            pw += np.mean(vu * iu) / 1e3; sw += np.sqrt(np.mean(vu ** 2)) * np.sqrt(np.mean(iu ** 2)) / 1e3
        sec = pd.Timestamp(key[:10] + " " + key[11:].replace("-", ":")); w = slice(sec, sec + pd.Timedelta(seconds=1))
        acc.append((pw, sw, mag["Mains_Power"].loc[w].mean(), sum((mag[f"CT{k}"].loc[w] * mag[f"L{k}"].loc[w]).mean() for k in (1, 2, 3)) / 1e3))
    a = np.array(acc)
    res["inside_meter_egauge_19"] = {"captures": len(a), "real_power_waveform_over_register_median": float(np.median(a[:, 0] / a[:, 2])),
                                     "real_power_ratio_5_95": [float(v) for v in np.percentile(a[:, 0] / a[:, 2], [5, 95])],
                                     "apparent_waveform_over_magnitude_product_median": float(np.median(a[:, 1] / a[:, 3])),
                                     "apparent_ratio_5_95": [float(v) for v in np.percentile(a[:, 1] / a[:, 3], [5, 95])],
                                     "power_factor_from_waveforms": float(np.median(a[:, 0] / a[:, 1]))}
    # 5. grid frequency
    res["frequency"] = {}
    for tag in ("20250312T180000", "20250312T100000"):
        f = {}
        for m in ("egauge_19", "egauge_17"):
            f[m] = {}
            for p in sorted((wave_root / f"{m}_{tag}" / "waveforms" / f"{m}-L1").rglob("*.parquet")):
                df = pd.read_parquet(p); t = df["t"].to_numpy().astype("datetime64[us]").astype(np.int64) / 1e6
                f[m][p.stem[:19]] = W._fundamental(t - t[0], df["v"].to_numpy(float))
        common = sorted(set(f["egauge_19"]) & set(f["egauge_17"]))
        fa = np.array([f["egauge_19"][k] for k in common]); fb = np.array([f["egauge_17"][k] for k in common])
        res["frequency"][tag] = {"captures": len(common), "min_hz": float(fa.min()), "median_hz": float(np.median(fa)), "max_hz": float(fa.max()),
                                 "sd_mhz": float(fa.std() * 1e3), "two_meters_rms_difference_mhz": float(np.sqrt(np.mean((fa - fb) ** 2)) * 1e3),
                                 "two_meters_corr": float(np.corrcoef(fa, fb)[0, 1])}
    Path(out_json).write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    r = main(*sys.argv[1:6])
    print(json.dumps({k: v for k, v in r.items() if k != "node_2400V"}, indent=1)[:1800])
    print("node closures:", [(c["extra_boards"], round(c["mean_kw"]), round(c["sd_kw"])) for c in r["node_2400V"]["closures_best_first"][:3]])
