"""Whole-stream reader for the LBNL Open uPMU files.

fetch_upmu.py takes the first 200 MiB of each stream and aligns streams by row number, which
is right only while the stream has no gap. Each full file holds exactly 120,000,000 rows:
2015-10-01 00:00:00 to 2015-10-12 13:46:40 UTC, 11.57 days (not the three months the dataset's
citation names; checked 2026-10-04). The full streams are not gap-free, so here every sample
is placed on a 120-per-second grid by its own time-stamp. Missing samples stay NaN and are
counted; a repeated stretch overwrites (the later row wins) and is counted. Nothing is filled.

Usage (downloads and reduces one stream at a time; raw files are deleted unless KEEP=1):
    python -m island_detectability_map.upmu_full DATA_DIR STREAM[,STREAM...] [SENSOR]
Writes DATA_DIR/full/<sensor>_<stream>.npy (float32, 120 M samples) and merges a report into
DATA_DIR/full/PROVENANCE_full.json (bytes, sha256 of the whole file and of its first 200 MiB,
rows, gaps, back-steps).

Origin: verification@cf2dc041
"""
from __future__ import annotations

import hashlib, io, json, os, sys, urllib.request, zlib
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://powerdata-download.lbl.gov/data/_LBNL_{sensor}_{s}.gz"     # http:// now redirects here
DEFAULT_SENSOR = "grizzly_bus1_2"
FS = 120.0
N_FULL = 120_000_000
HEAD_BYTES = 200 * 1024 * 1024


def grid_index(t_ns: np.ndarray, t0_ns: int, fs: float = FS) -> np.ndarray:
    """Sample number of each time-stamp. The files truncate to whole nanoseconds
    (0, 8333333, 16666666, 25000000 ...), so round rather than floor."""
    return np.rint((np.asarray(t_ns, dtype=np.int64) - int(t0_ns)) * (fs / 1e9)).astype(np.int64)


def place(dest: np.ndarray, t_ns: np.ndarray, v: np.ndarray, t0_ns: int, fs: float = FS) -> dict:
    """Write v into dest at the grid positions of t_ns. Later rows overwrite earlier ones.
    The report's "overwritten_idx" lists the grid samples that were sent more than once."""
    idx = grid_index(t_ns, t0_ns, fs)
    ok = (idx >= 0) & (idx < len(dest))
    idx_ok, v_ok = idx[ok], v[ok]
    if len(idx_ok) == 0:
        return {"written": 0, "overwritten": 0, "beyond_end": int((~ok).sum()), "overwritten_idx": np.empty(0, np.int64)}
    if np.all(np.diff(idx_ok) > 0):                       # usual case: strictly increasing, no repeat inside the call
        lo, hi = int(idx_ok[0]), int(idx_ok[-1])
        if hi - lo + 1 == len(idx_ok):                    # and no gap: one slice
            already = np.isfinite(dest[lo:hi + 1])
            dest[lo:hi + 1] = v_ok
        else:
            already = np.isfinite(dest[idx_ok])
            dest[idx_ok] = v_ok
        return {"written": int(len(idx_ok)), "overwritten": int(already.sum()), "beyond_end": int((~ok).sum()),
                "overwritten_idx": idx_ok[already]}
    uniq, first_pos, counts = np.unique(idx_ok, return_index=True, return_counts=True)
    pre = np.isfinite(dest[uniq])
    dest[idx_ok] = v_ok            # numpy assigns in order, so the last duplicate wins
    return {"written": int(len(uniq)), "overwritten": int(pre.sum() + (counts - 1).sum()), "beyond_end": int((~ok).sum()),
            "overwritten_idx": uniq[pre | (counts > 1)]}


def _ranges(idx: np.ndarray) -> list:
    """Sorted sample numbers -> list of half-open [start, end) runs."""
    idx = np.unique(np.asarray(idx, dtype=np.int64))
    if len(idx) == 0:
        return []
    brk = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[brk + 1]]; ends = np.r_[idx[brk], idx[-1]] + 1
    return [[int(a), int(b)] for a, b in zip(starts, ends)]


def void_ranges(x: np.ndarray, ranges) -> np.ndarray:
    """Copy of x with the given half-open sample ranges set to NaN (stretches that were sent
    twice, whose content is therefore in doubt)."""
    y = np.array(x, dtype=x.dtype, copy=True)
    for a, b in ranges:
        y[a:b] = np.nan
    return y


def parse_to_grid(gz_path: Path, out_npy: Path, n: int = N_FULL, fs: float = FS, chunk_bytes: int = 1 << 24) -> dict:
    """Decode a (possibly truncated) gzip stream of 't_ns,value' rows onto an n-sample grid
    starting at the first row's time-stamp. Returns a report; writes float32 to out_npy."""
    dest = np.full(n, np.nan, dtype=np.float32)
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    rep = {"rows_read": 0, "overwritten": 0, "beyond_end": 0, "back_steps": 0, "forward_gaps": 0, "events": []}
    t0, prev, buf = None, None, b""
    step = 1e9 / fs
    ow = []

    def take(block: bytes):
        nonlocal t0, prev
        if not block:
            return
        df = pd.read_csv(io.BytesIO(block), header=None, names=["t", "v"], dtype={"t": "int64", "v": "float32"})
        t, v = df["t"].to_numpy(), df["v"].to_numpy()
        if t0 is None:
            t0 = int(t[0])
        dt = np.diff(np.concatenate([[prev], t])) if prev is not None else np.diff(t)
        off = 0 if prev is not None else 1
        for j in np.where(dt <= 0)[0]:
            rep["back_steps"] += 1
            rep["events"].append({"kind": "back_step", "row": rep["rows_read"] + int(j) + off, "seconds": float(dt[j] / 1e9)})
        for j in np.where(dt > 1.5 * step)[0]:
            rep["forward_gaps"] += 1
            rep["events"].append({"kind": "gap", "row": rep["rows_read"] + int(j) + off, "seconds": float(dt[j] / 1e9)})
        r = place(dest, t, v, t0, fs)
        rep["overwritten"] += r["overwritten"]; rep["beyond_end"] += r["beyond_end"]
        ow.append(r["overwritten_idx"])
        rep["rows_read"] += len(t); prev = int(t[-1])

    with open(gz_path, "rb") as f:
        while True:
            c = f.read(chunk_bytes)
            if not c:
                break
            buf += d.decompress(c)
            cut = buf.rfind(b"\n")
            if cut < 0:
                continue
            block, buf = buf[:cut], buf[cut + 1:]
            take(block)
    if d.eof and buf.strip():          # a complete stream may end without a newline; a truncated one ends mid-row
        take(buf)
    rep["t0_ns"] = t0
    rep["complete_stream"] = bool(d.eof)
    nan = np.where(np.isnan(dest[:grid_index(np.array([prev]), t0, fs)[0] + 1]))[0] if prev is not None else np.empty(0, int)
    rep["missing"] = int(len(nan))                       # inside the span that was read (not the unread tail)
    rep["missing_ranges"] = _ranges(nan)
    rep["overwritten_ranges"] = _ranges(np.concatenate(ow) if ow else np.empty(0, np.int64))
    np.save(out_npy, dest)
    return rep


def block_nanmean(x: np.ndarray, k: int, min_frac: float = 0.9) -> np.ndarray:
    """Mean of each block of k samples over the samples that are present; NaN where fewer
    than min_frac of the block is present. Never interpolates."""
    n = len(x) // k * k
    b = np.asarray(x[:n], dtype=np.float64).reshape(-1, k)
    ok = np.isfinite(b)
    cnt = ok.sum(axis=1)
    s = np.where(ok, b, 0.0).sum(axis=1)
    out = np.full(len(cnt), np.nan)
    good = cnt >= max(1, int(np.ceil(min_frac * k - 1e-9)))
    out[good] = s[good] / cnt[good]
    return out


def head_sha256(path: Path, nbytes: int = HEAD_BYTES) -> str:
    h = hashlib.sha256()
    left = nbytes
    with open(path, "rb") as f:
        while left > 0:
            c = f.read(min(1 << 24, left))
            if not c:
                break
            h.update(c); left -= len(c)
    return h.hexdigest()


def head_rows(gz_path: Path, nbytes: int = HEAD_BYTES) -> int:
    """Number of complete rows in the first nbytes of a gzip stream (what a range-fetched head yields)."""
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    rows, left = 0, nbytes
    with open(gz_path, "rb") as f:
        while left > 0:
            c = f.read(min(1 << 24, left))
            if not c:
                break
            left -= len(c)
            rows += d.decompress(c).count(b"\n")
    return rows


def make_head_npz(data_dir: Path, streams, out_name: str, sensor: str = DEFAULT_SENSOR, max_rows: int = 40_000_000) -> int:
    """Rebuild a fetch_upmu-style cache (first rows of each stream, cut to the shortest 200 MiB
    head, at most max_rows) from the whole-stream arrays. Valid only where the head has no gap
    or back-step, which is asserted."""
    data_dir = Path(data_dir)
    prov = json.loads((data_dir / "full" / "PROVENANCE_full.json").read_text())
    n = min(min(prov[f"{sensor}/{s}"]["rows_in_first_200MiB"] for s in streams), max_rows)
    t0 = {prov[f"{sensor}/{s}"]["t0_ns"] for s in streams}
    assert len(t0) == 1, "streams start at different times"
    out = {}
    for s in streams:
        ev = [e for e in prov[f"{sensor}/{s}"]["events"] if e["row"] < n]
        assert not ev, f"{s}: gap or back-step inside the head: {ev}"
        x = np.load(data_dir / "full" / f"{sensor}_{s}.npy", mmap_mode="r")[:n]
        assert np.isfinite(x).all()
        out[s] = np.array(x)
    np.savez(data_dir / out_name, sensor=sensor, t0_ns=t0.pop(), fs=FS, **out)
    return n


def fetch_one(data_dir: Path, stream: str, sensor: str = DEFAULT_SENSOR, keep_raw: bool = False) -> dict:
    full = Path(data_dir) / "full"
    full.mkdir(parents=True, exist_ok=True)
    out = full / f"{sensor}_{stream}.npy"
    pf = full / "PROVENANCE_full.json"
    prov = json.loads(pf.read_text()) if pf.exists() else {}
    key = f"{sensor}/{stream}"
    if out.exists() and key in prov:            # resume: never redo a finished stream
        return prov[key]
    gz = full / f"{sensor}_{stream}.gz"
    url = BASE.format(sensor=sensor, s=stream)
    if not gz.exists():
        tmp = gz.with_suffix(".tmp")
        with urllib.request.urlopen(url, timeout=280) as r, open(tmp, "wb") as f:
            while True:
                c = r.read(1 << 24)
                if not c:
                    break
                f.write(c)
        tmp.rename(gz)
    rep = parse_to_grid(gz, out)
    rep.update(url=url, bytes=gz.stat().st_size, sha256=head_sha256(gz, 10**12), sha256_first_200MiB=head_sha256(gz),
               rows_in_first_200MiB=head_rows(gz))
    rep["events"] = rep["events"][:50]
    prov[key] = rep
    pf.write_text(json.dumps(prov, indent=1))
    if not keep_raw:
        gz.unlink()
    return rep


if __name__ == "__main__":
    dd = Path(sys.argv[1])
    sensor = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_SENSOR
    for s in sys.argv[2].split(","):
        r = fetch_one(dd, s, sensor, keep_raw=os.environ.get("KEEP") == "1")
        print(f"{s}: rows {r['rows_read']:,} | missing {r['missing']:,} | overwritten {r['overwritten']:,} | "
              f"gaps {r['forward_gaps']} back-steps {r['back_steps']} | head sha {r['sha256_first_200MiB'][:12]}", flush=True)
