"""Pull 1-second magnitudes from the Caltech SoCal 28-bus dataset and record provenance.
EXPLORATORY tooling, no tests; the loader that reads these files is caltech.py.

One request, no retry loop (the server has a record of failing on large pulls; a request is
capped at an estimated 1 GiB, ~34 bytes per point). Data goes under data/ (gitignored);
PROVENANCE (elements, time range, bytes, rows, sha256) is merged into a JSON kept with the results.
Needs a login first: see caltech_login.py.

Meter mode pulls every channel the loader (caltech.py) needs for each named meter, one request
per meter into ROOT/<meter>/, and skips a meter whose provenance entry already says ok and whose
files are on disk (safe to re-run after an interruption):
  python -m island_detectability_map.fetch_caltech --meters ROOT egauge_17,egauge_11 START END PROVENANCE_JSON NETWORK_JSON

Usage: python -m island_detectability_map.fetch_caltech OUT_DIR ELEMENTS START END PROVENANCE_JSON
  e.g. data/caltech/first_pull_20250312 egauge_17-CT1,egauge_17-CT2 2025-03-12T00:00:00 2025-03-13T00:00:00 \\
       results/20260930_island_detectability_map/caltech_PROVENANCE.json

Origin: verification@67698f42
"""
from __future__ import annotations

import hashlib, json, subprocess, sys, time
from pathlib import Path

from island_detectability_map import caltech_login


def main(out: Path, elements: list[str], t0: str, t1: str, prov_path: Path, key: str | None = None):
    c = caltech_login.client()
    st = time.time()
    status = "ok"
    try:
        c.download_data(magnitudes_for=elements, time_range=(t0, t1), output_dir=out)
    except Exception as e:                                     # record and stop; do not retry
        status = f"failed: {type(e).__name__}: {str(e)[:300]}"
    files = {}
    for p in sorted(out.rglob("*.csv")):
        b = p.read_bytes()
        files[p.stem] = {"bytes": len(b), "rows": b.count(b"\n") - 1, "sha256": hashlib.sha256(b).hexdigest()}
    commit = subprocess.run(["git", "-C", caltech_login.CLIENT_DIR, "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    prov[key or out.name] = {"api": c.base_url + "/data", "kind": "magnitudes, 1 s", "time_range": [t0, t1],
                      "status": status, "seconds": round(time.time() - st, 1),
                      "pulled_utc": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()),
                      "client_repo_commit": commit, "files": files}
    prov_path.write_text(json.dumps(prov, indent=1))
    print(key or out.name, status, f"{len(files)} files, {sum(f['bytes'] for f in files.values()):,} bytes, {time.time() - st:.0f} s", flush=True)
    return status


def channels(m) -> list[str]:
    """Every register file caltech.load_element reads for a meter entry."""
    return list(m.current.values()) + list(m.voltage.values()) + ([m.power] if m.power else [])


def pull_meters(root: Path, meters: list[str], t0: str, t1: str, prov_path: Path, network_json: Path):
    from island_detectability_map import caltech
    mm = caltech.meter_map(network_json)
    for name in meters:
        key = f"{root.name}/{name}"
        prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
        done = prov.get(key, {})
        if done.get("status") == "ok" and all((root / name / "magnitudes" / f"{f}.csv").exists() for f in done.get("files", {})):
            print(key, "already pulled, skipping", flush=True)
            continue
        if main(root / name, channels(mm[name]), t0, t1, prov_path, key) != "ok":
            print("stopping: a pull failed (no retry)", flush=True)
            return


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "--meters":
        pull_meters(Path(a[2]), a[3].split(","), a[4], a[5], Path(a[6]), Path(a[7]))
    else:
        main(Path(a[1]), a[2].split(","), a[3], a[4], Path(a[5]))
