"""Pull the 4-node (16 H100) training runs out of the NLR dataset zip without downloading it whole.
EXPLORATORY: no tests; checked by the zip's own CRCs (zipfile verifies each member on read).

The zip (1.07 GB) is at data.nlr.gov/submissions/312 and answers HTTP range requests. This reads its
directory, then only the members wanted: the raw GPU and CPU logs of the 4-node training runs, the
aggregated training results, and the README. About 240 MB on disk, 120 MB fetched.

Usage: python -m island_detectability_map.fetch_nlr OUT_DIR
Writes OUT_DIR/<member path> and OUT_DIR/nlr_PROVENANCE.json (url, zip size, sha256 per file).

Origin: verification@43c17c88
"""
from __future__ import annotations

import hashlib, io, json, os, sys, urllib.request, zipfile
from pathlib import Path

URL = "https://data.nlr.gov/system/files/312/1775845072-dataset.zip"      # version 2, 2026-04-10


class HttpRange(io.RawIOBase):
    """A seekable read-only file over HTTP range requests, with a small block cache."""

    def __init__(self, url: str, block: int = 1 << 22):
        self.url, self.pos, self.block = url, 0, block
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self._cache: dict = {}
        self.fetched = 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def _blk(self, i):
        if i not in self._cache:
            a = i * self.block; b = min(self.size, a + self.block) - 1
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status != 206:
                    raise IOError(f"server did not honour the range request (status {r.status})")
                self._cache[i] = r.read()
            self.fetched += b - a + 1
            if len(self._cache) > 64:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[i]

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            i, o = divmod(self.pos, self.block)
            chunk = self._blk(i)[o:o + n]
            out += chunk; self.pos += len(chunk); n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d
        return len(d)


def wanted(name: str) -> bool:
    return (("/4node/" in name and "training" in name) or name.startswith("01_aggregated_datasets/training/")
            or name in ("01_aggregated_datasets/utilities.py", "requirements.txt", "README.md"))


def main(out_dir: Path, url: str = URL):
    h = HttpRange(url)
    z = zipfile.ZipFile(io.BufferedReader(h, 1 << 20))
    prov = {}
    for i in z.infolist():
        if i.is_dir() or not wanted(i.filename):
            continue
        p = out_dir / i.filename
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():                                  # resume: never refetch a finished file
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_bytes(z.read(i)); os.replace(tmp, p)
        b = p.read_bytes()
        prov[i.filename] = {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest(), "crc32": i.CRC}
    (out_dir / "nlr_PROVENANCE.json").write_text(json.dumps({"url": url, "zip_bytes": h.size, "members": prov}, indent=1))
    return len(prov), sum(v["bytes"] for v in prov.values()), h.fetched


if __name__ == "__main__":
    n, size, fetched = main(Path(sys.argv[1]))
    print(f"{n} files, {size / 1e6:.0f} MB on disk, {fetched / 1e6:.0f} MB fetched")
