"""Pull the one-second electric-mains records of the IDEAL homes (Edinburgh DataShare, DOI
10.7488/ds/2836; Pullinger et al. 2021) for one time window, without keeping the 15.9 GB archive.
EXPLORATORY: no tests of its own; each member is checked against the zip directory's CRC
and the gridding is ideal.to_grid (tests/test_ideal.py).

household_sensors.zip answers HTTP range requests. This reads its directory, then, for every home
whose electric-mains file exists and whose study dates (metadata home.csv) span the window, streams
that home's csv.gz, keeps the rows inside the window (window_rows, tests/test_ideal.py) and writes them on a one-second grid as int32
watts (-1 = no reading in that second).

Usage: python -m island_detectability_map.fetch_ideal OUT_DIR HOME_CSV START END [WORKERS]
  START, END  UTC, e.g. 2018-03-05T00:00:00 2018-04-30T00:00:00
Writes OUT_DIR/grid/home<ID>.npy and OUT_DIR/ideal_PROVENANCE.json (url, window, per-home member
name, bytes, CRC, rows kept, seconds with a reading).

Origin: verification@776d1246
"""
from __future__ import annotations

import io, json, os, re, struct, sys, time, urllib.request, zipfile, zlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from island_detectability_map import ideal

URL = "https://datashare.ed.ac.uk/server/api/core/bitstreams/32316855-78ab-4279-8543-2992b12d322f/content"
SIZE = 15863138394                                   # household_sensors.zip, as served 2026-10-08
MEMBER = re.compile(r"^sensordata/home(\d+)_.*_electric-mains_electric-combined\.csv\.gz$")


def get_range(a: int, b: int, tries: int = 12) -> bytes:
    """Bytes a..b (inclusive) of the archive. The server drops connections now and then: retried."""
    for k in range(tries):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(req, timeout=180) as r:
                if r.status != 206:
                    raise OSError(f"server did not honour the range request (status {r.status})")
                d = r.read()
            if len(d) != b - a + 1:
                raise OSError("short read")
            return d
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(1 + k)


class HttpRange(io.RawIOBase):
    """A seekable read-only file over HTTP range requests; used only to read the zip directory."""

    def __init__(self, size: int, block: int = 1 << 20):
        self.size, self.pos, self.block = size, 0, block
        self._cache: dict = {}

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            i, o = divmod(self.pos, self.block)
            if i not in self._cache:
                self._cache[i] = get_range(i * self.block, min(self.size, (i + 1) * self.block) - 1)
            chunk = self._cache[i][o:o + n]
            out += chunk; self.pos += len(chunk); n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d
        return len(d)


def directory():
    return zipfile.ZipFile(io.BufferedReader(HttpRange(SIZE), 1 << 20)).infolist()


def read_member(offset: int, compressed: int, method: int, crc: int, threads: int = 6, part: int = 1 << 23) -> bytes:
    """One member's bytes, fetched as parallel ranges and checked against the directory's CRC.
    The server serves about 3 MB/s per connection, so the ranges go out side by side."""
    head = get_range(offset, offset + 29)
    if head[:4] != b"PK\x03\x04":
        raise OSError("not a local file header")
    n_name, n_extra = struct.unpack("<HH", head[26:30])
    a = offset + 30 + n_name + n_extra
    spans = [(s, min(a + compressed, s + part) - 1) for s in range(a, a + compressed, part)]
    with ThreadPoolExecutor(threads) as ex:
        raw = b"".join(ex.map(lambda ab: get_range(*ab), spans))
    data = raw if method == zipfile.ZIP_STORED else zlib.decompress(raw, -15)
    if zlib.crc32(data) != crc:
        raise OSError("CRC mismatch")
    return data


def window_rows(gz: bytes, a: str, b: str, chunk: int = 1 << 21) -> bytes:
    """The whole lines of a gzipped 'time-stamp,value' file from the blocks that can hold rows with
    a <= time-stamp < b. Rows are in time order, so blocks wholly before a are dropped unparsed and
    reading stops at the first block that starts at or after b. The caller trims the edges.
    chunk is the number of compressed bytes fed to the decompressor at a time."""
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    out, tail = [], b""
    lo, hi = a.encode(), b.encode()
    for pos in range(0, len(gz), chunk):
        blk = tail + d.decompress(gz[pos:pos + chunk])
        cut = blk.rfind(b"\n") + 1
        blk, tail = blk[:cut], blk[cut:]
        if not blk:
            continue
        if blk[:19] >= hi:
            return b"".join(out)
        if blk[blk.rfind(b"\n", 0, len(blk) - 1) + 1:][:19] >= lo:
            out.append(blk)
    tail += d.flush()
    if tail and tail[:19] < hi:
        out.append(tail if tail.endswith(b"\n") else tail + b"\n")
    return b"".join(out)


def spanning_homes(home_csv: Path, start: str, end: str) -> set:
    h = pd.read_csv(home_csv)
    s = pd.to_datetime(h["starttime"], dayfirst=True); e = pd.to_datetime(h["endtime"], dayfirst=True)
    return set(h.loc[(s <= pd.Timestamp(start)) & (e >= pd.Timestamp(end)), "homeid"].astype(int))


def one(args):
    name, crc, offset, compressed, method, out_dir, start, end = args
    hid = int(MEMBER.match(name).group(1))
    dst = Path(out_dir) / "grid" / f"home{hid}.npy"
    meta = dst.with_suffix(".json")
    if dst.exists() and meta.exists():                   # resume: never refetch a finished home
        return json.loads(meta.read_text())
    gz = read_member(offset, compressed, method, crc)
    n_bytes = len(gz)
    a, b = start.replace("T", " "), end.replace("T", " ")
    txt = window_rows(gz, a, b)
    del gz
    if txt:
        df = pd.read_csv(io.BytesIO(txt), header=None, names=["t", "w"])
        df = df[(df["t"] >= a) & (df["t"] < b)]
        t = pd.to_datetime(df["t"]).to_numpy().astype("datetime64[s]").astype(np.int64)
        w = df["w"].to_numpy()
    else:
        t, w = np.zeros(0, dtype=np.int64), np.zeros(0)
    t0 = int(pd.Timestamp(start).value // 10**9); n = int(pd.Timestamp(end).value // 10**9) - t0
    g, rep = ideal.to_grid(t, w, t0, n)
    tmp = dst.with_suffix(".tmp.npy")
    np.save(tmp, g); os.replace(tmp, dst)
    m = {"home": hid, "member": name, "bytes": n_bytes, "crc32": crc, "rows_kept": len(t), **rep}
    meta.write_text(json.dumps(m))
    return m


def main(out_dir: Path, home_csv: Path, start: str, end: str, workers: int = 2):
    (out_dir / "grid").mkdir(parents=True, exist_ok=True)
    keep = spanning_homes(home_csv, start, end)
    jobs = []
    for i in directory():
        m = MEMBER.match(i.filename)
        if m and int(m.group(1)) in keep:
            jobs.append((i.filename, i.CRC, i.header_offset, i.compress_size, i.compress_type, str(out_dir), start, end))
    done = []
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(one, jobs):
            done.append(r)
            print(f"home {r['home']:>4}  {r['bytes'] / 1e6:6.1f} MB  seconds with a reading {r['seconds_read']}", flush=True)
    prov = {"url": URL, "zip_bytes": SIZE, "start_utc": start, "end_utc": end, "homes_spanning_window": len(keep),
            "homes_with_mains_file": len(jobs), "members": sorted(done, key=lambda r: r["home"])}
    (out_dir / "ideal_PROVENANCE.json").write_text(json.dumps(prov, indent=1))
    return len(done)


if __name__ == "__main__":
    a = sys.argv
    print(main(Path(a[1]), Path(a[2]), a[3], a[4], int(a[5]) if len(a) > 5 else 2), "homes")
