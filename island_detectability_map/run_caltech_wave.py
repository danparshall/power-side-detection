"""Background at 2-30 Hz from Caltech point-on-wave captures.
Writes/extends caltech_wave.json. EXPLORATORY orchestration; the envelope maths is tested in
tests/test_caltech_wave.py.

Each capture (~1 s, one every 10 s) gives per-phase kVA envelopes (fundamental RMS current x
fundamental RMS voltage) and from them a band RMS for the sum, each phase and each imbalance
residual. Reported: the median capture and the RMS over captures, per meter and time window.
Raw channel names: S1-S3 are the current sensors, L1-L3 the voltage inputs. For the two
line-to-line meters L1 = V_ab and L2 = -V_bc.

Usage: python -m island_detectability_map.run_caltech_wave OUT_ROOT METER START END PROVENANCE_JSON RESULT_JSON
One request, no retry. Skips the pull if the folder already holds captures.

Origin: verification@b2855af0
"""
from __future__ import annotations

import hashlib, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

from island_detectability_map import caltech_login, caltech_wave as W

BANDS = [(2.0, 10.0), (10.0, 30.0), (2.0, 30.0)]
LINE_TO_LINE = {"egauge_17", "egauge_18"}


def channels(meter):
    return ["S1", "S2", "S3"] + (["L1", "L2"] if meter in LINE_TO_LINE else ["L1", "L2", "L3"])


def pull(out: Path, meter: str, t0: str, t1: str, prov_path: Path, key: str):
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    have = any(out.rglob("*.parquet"))
    if have and key in prov:
        return
    c = caltech_login.client()
    st = time.time(); status = "ok"
    if have:
        status = "ok (files already on disk from an earlier request this session)"
    else:
        try:
            c.download_data(waveforms_for=[f"{meter}-{ch}" for ch in channels(meter)], time_range=(t0, t1), output_dir=out)
        except Exception as e:
            status = f"failed: {type(e).__name__}: {str(e)[:300]}"
    files = sorted(out.rglob("*.parquet"))
    h = hashlib.sha256()
    for p in files:
        h.update(p.name.encode()); h.update(hashlib.sha256(p.read_bytes()).digest())
    prov[key] = {"api": c.base_url + "/data", "kind": "waveforms, 2.5 kHz, ~1 s every 10 s", "time_range": [t0, t1], "status": status,
                 "seconds": round(time.time() - st, 1), "pulled_utc": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()),
                 "n_files": len(files), "bytes": sum(p.stat().st_size for p in files), "sha256_of_file_hashes": h.hexdigest()}
    prov_path.write_text(json.dumps(prov, indent=1))
    print(key, status, len(files), "files", f"{time.time() - st:.0f} s", flush=True)


def analyse(out: Path, meter: str):
    chans = channels(meter)
    by_time: dict[str, dict] = {}
    for ch in chans:
        for p in sorted((out / "waveforms" / f"{meter}-{ch}").rglob("*.parquet")):
            by_time.setdefault(p.stem[:19], {})[ch] = p         # captures of one meter start within ~0.1 s; match to the second
    rows, means, f0s = [], [], []
    for stamp, files in sorted(by_time.items()):
        if len(files) != len(chans):
            continue
        caps = {}
        t_ref = None
        for ch, p in files.items():
            df = pd.read_parquet(p)
            t = df["t"].to_numpy().astype("datetime64[us]").astype(np.int64)
            t_ref = t[0] if t_ref is None else min(t_ref, t[0])
            caps[ch] = (t, df["v"].to_numpy(float))
        caps = {ch: ((t - t_ref) / 1e6, v) for ch, (t, v) in caps.items()}
        ll = meter in LINE_TO_LINE
        te, kva = W.phase_kva(caps, current=("S1", "S2", "S3"), voltage=("L1", "L2") if ll else ("L1", "L2", "L3"), line_to_line=ll)
        rows.append({f"{lo:g}-{hi:g}Hz": W.band_rms_kva(te, kva, lo, hi) for lo, hi in BANDS})
        means.append(kva.mean(axis=1)); f0s.append(W._fundamental(*caps["L1"]))
    res = {"captures": len(rows), "mean_kva": float(np.sum(np.mean(means, axis=0))), "mean_kva_per_phase": [float(v) for v in np.mean(means, axis=0)],
           "fundamental_hz_median": float(np.median(f0s)), "bands": {}}
    for lo, hi in BANDS:
        k = f"{lo:g}-{hi:g}Hz"
        s = np.array([r[k]["sum"] for r in rows]); ph = np.array([r[k]["phase"] for r in rows]); rs = np.array([r[k]["resid"] for r in rows])
        res["bands"][k] = {"sum": {"median": float(np.median(s)), "rms": float(np.sqrt(np.mean(s ** 2)))},
                           "phase": {"median": [float(v) for v in np.median(ph, axis=0)], "rms": [float(v) for v in np.sqrt(np.mean(ph ** 2, axis=0))]},
                           "resid": {"median": [float(v) for v in np.median(rs, axis=0)], "rms": [float(v) for v in np.sqrt(np.mean(rs ** 2, axis=0))]}}
    return res


if __name__ == "__main__":
    root, meter, t0, t1, prov, result = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], Path(sys.argv[5]), Path(sys.argv[6])
    key = f"wave/{meter}_{t0.replace(':', '').replace('-', '')}"
    out = root / key.split("/")[1]
    pull(out, meter, t0, t1, prov, key)
    r = analyse(out, meter)
    allr = json.loads(result.read_text()) if result.exists() else {}
    allr[key] = dict(meter=meter, time_range_utc=[t0, t1], **r)
    result.write_text(json.dumps(allr, indent=1))
    b = r["bands"]
    print(key, f"captures {r['captures']} mean {r['mean_kva']:.0f} kVA f0 {r['fundamental_hz_median']:.3f} Hz")
    for k, v in b.items():
        print(f"   {k:<9} sum {v['sum']['median']:.2f} (rms {v['sum']['rms']:.2f}) | phase {min(v['phase']['median']):.2f}-{max(v['phase']['median']):.2f} | resid {min(v['resid']['median']):.2f}-{max(v['resid']['median']):.2f} kVA")
