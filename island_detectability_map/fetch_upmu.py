"""Fetch the head of four LBNL Open uPMU streams (12 kV 'grizzly_bus1_2') and cache
a parsed copy. Source: http://powerdata-download.lbl.gov/data/ (Stewart et al. 2016,
LBNL-1006408; data begin 2015-10-01 00:00 UTC, 120 samples/s).

Only the first HEAD_BYTES of each gzip stream are pulled (HTTP range request); gzip
decompresses a truncated stream up to the cut, which yields the first ~3 days.
Raw parts and the parsed cache live under data/ (gitignored); sha256 of each part is
written to data/upmu/PROVENANCE.json so a re-fetch can be checked.

Usage:  python -m island_detectability_map.fetch_upmu [DATA_DIR] [ROWS] [SENSOR] [STREAMS] [OUT_NPZ]
  defaults: grizzly_bus1_2, C1MAG,C2MAG,C3MAG,L1MAG, grizzly_head.npz (the 2026-09-30 run).
  Per-phase voltages (2026-10-01):  ... ../data/upmu 40000000 grizzly_bus1_2 \
      C1MAG,C2MAG,C3MAG,L1MAG,L2MAG,L3MAG grizzly_head6.npz
  Other sensors in the directory (verified 2026-10-01): a6_bus1, bank_514.

Origin: verification@f7a9006e
"""
from __future__ import annotations

import hashlib, io, json, sys, urllib.request, zlib
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "http://powerdata-download.lbl.gov/data/_LBNL_{sensor}_{s}.gz"
DEFAULT_SENSOR = "grizzly_bus1_2"
STREAMS = ["C1MAG", "C2MAG", "C3MAG", "L1MAG"]
HEAD_BYTES = 200 * 1024 * 1024  # ~3 days at ~6.4 compressed bytes per row
FS = 120.0


def part_path(data_dir: Path, sensor: str, s: str) -> Path:
    prefix = "g" if sensor == DEFAULT_SENSOR else sensor   # keep the 2026-09-30 file names
    return data_dir / f"{prefix}_{s}.part.gz"


def fetch(data_dir: Path, sensor: str = DEFAULT_SENSOR, streams=STREAMS) -> dict:
    """Range-fetch the head of each stream; provenance is merged into PROVENANCE.json
    (keys: stream name for the default sensor, 'sensor/stream' otherwise)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    pf = data_dir / "PROVENANCE.json"
    prov = json.loads(pf.read_text()) if pf.exists() else {}
    for s in streams:
        part = part_path(data_dir, sensor, s)
        url = BASE.format(sensor=sensor, s=s)
        if not part.exists():  # resume: never re-download an existing part
            req = urllib.request.Request(url, headers={"Range": f"bytes=0-{HEAD_BYTES - 1}"})
            part.write_bytes(urllib.request.urlopen(req, timeout=300).read())
        key = s if sensor == DEFAULT_SENSOR else f"{sensor}/{s}"
        prov[key] = {"url": url, "bytes": part.stat().st_size,
                     "sha256": hashlib.sha256(part.read_bytes()).hexdigest()}
    pf.write_text(json.dumps(prov, indent=1))
    return prov


def parse(part: Path, max_rows: int):
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    ts, vals, buf, n = [], [], b"", 0
    with open(part, "rb") as f:
        while n < max_rows:
            c = f.read(1 << 23)
            if not c:
                break
            buf += d.decompress(c)
            cut = buf.rfind(b"\n")
            block, buf = buf[:cut], buf[cut + 1:]
            df = pd.read_csv(io.BytesIO(block), header=None, names=["t", "v"], dtype={"t": "int64", "v": "float32"})
            ts.append(df["t"].to_numpy()); vals.append(df["v"].to_numpy()); n += len(df)
    t = np.concatenate(ts)[:max_rows]
    return t, np.concatenate(vals)[:max_rows]


def main(data_dir: Path, rows: int, sensor: str = DEFAULT_SENSOR, streams=STREAMS,
         out_npz: str = "grizzly_head.npz"):
    fetch(data_dir, sensor, streams)
    out, t0 = {}, None
    for s in streams:
        t, v = parse(part_path(data_dir, sensor, s), rows)
        gaps = int((np.diff(t) > 1.5e9 / FS).sum())
        # streams compress differently, so equal byte ranges hold unequal row counts;
        # they must share a start time and be gap-free, then are cut to a common length
        assert gaps == 0, f"{s}: {gaps} gaps - index alignment would be wrong"
        assert t0 is None or t[0] == t0, f"{s}: different start time"
        t0 = int(t[0])
        print(f"{s}: {len(v):,} rows, {(t[-1]-t[0])/3.6e12:.1f} h, gaps {gaps}, mean {v.mean():.3f}")
        out[s] = v
        del t
    n = min(len(v) for v in out.values())
    np.savez(data_dir / out_npz, sensor=sensor, t0_ns=t0, fs=FS, **{k: v[:n] for k, v in out.items()})
    print(f"common length {n:,} rows = {n/FS/3600:.1f} h")


if __name__ == "__main__":
    dd = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/upmu")
    a = sys.argv
    main(dd, int(a[2]) if len(a) > 2 else 40_000_000, a[3] if len(a) > 3 else DEFAULT_SENSOR,
         a[4].split(",") if len(a) > 4 else STREAMS, a[5] if len(a) > 5 else "grizzly_head.npz")
