"""Whole-stream reader for the LBNL uPMU files.
Written before the code.

fetch_upmu.parse() assumes a gap-free stream and aligns streams by row number. The full
streams are not gap-free (found 2026-10-04: one second repeated and one second missing in
C1MAG near 2015-10-06 15:45 UTC), so the whole-stream reader places every sample by its
time-stamp, leaves missing samples as NaN, and reports what it found.

Origin: verification@23d3a497
"""
import gzip

import numpy as np
import pytest

from island_detectability_map import upmu_full

FS = 120.0
T0 = 1443657600 * 10**9


def _stamps(idx):
    """The files' time-stamps: whole nanoseconds, truncated (0, 8333333, 16666666, 25000000 ...)."""
    idx = np.asarray(idx, dtype=np.int64)
    return T0 + (idx * 10**9) // 120


def _write_gz(path, idx, vals):
    with gzip.open(path, "wb") as f:
        for t, v in zip(_stamps(idx), vals):
            f.write(f"{t},{v:.6f}\n".encode())


def test_grid_index_recovers_sample_numbers_from_truncated_stamps():
    idx = np.array([0, 1, 2, 3, 119, 120, 121, 7_199_999, 119_999_999])
    np.testing.assert_array_equal(upmu_full.grid_index(_stamps(idx), T0, FS), idx)


def test_place_leaves_missing_samples_as_nan_and_counts_them(tmp_path):
    idx = np.r_[np.arange(0, 240), np.arange(360, 600)]          # one second missing
    dest = np.full(600, np.nan, dtype=np.float32)
    rep = upmu_full.place(dest, _stamps(idx), np.arange(len(idx), dtype=np.float32), T0, FS)
    assert np.isnan(dest[240:360]).all() and np.isfinite(dest[:240]).all() and np.isfinite(dest[360:]).all()
    assert dest[360] == 240.0
    assert rep["written"] == 480 and rep["overwritten"] == 0 and rep["beyond_end"] == 0


def test_place_lets_a_repeated_block_overwrite_and_counts_it():
    idx = np.r_[np.arange(0, 240), np.arange(120, 480)]          # the stream steps back one second
    vals = np.r_[np.zeros(240), np.ones(360)].astype(np.float32)
    dest = np.full(480, np.nan, dtype=np.float32)
    rep = upmu_full.place(dest, _stamps(idx), vals, T0, FS)
    assert rep["overwritten"] == 120
    assert (dest[:120] == 0).all() and (dest[120:] == 1).all()   # later rows win


def test_parse_whole_gz_stream_onto_the_grid(tmp_path):
    idx = np.r_[np.arange(0, 1000), np.arange(880, 1300), np.arange(1420, 2400)]   # a back-step, then a gap
    vals = np.arange(len(idx), dtype=np.float64) * 0.5
    p = tmp_path / "s.gz"
    _write_gz(p, idx, vals)
    out = tmp_path / "s.npy"
    rep = upmu_full.parse_to_grid(p, out, n=2400, fs=FS, chunk_bytes=4096)       # small chunks: rows split across reads
    x = np.load(out)
    assert x.dtype == np.float32 and x.shape == (2400,)
    assert rep["t0_ns"] == T0 and rep["rows_read"] == len(idx)
    assert rep["missing"] == 120 and np.isnan(x[1300:1420]).all()
    assert rep["overwritten"] == 120
    assert rep["back_steps"] == 1 and rep["forward_gaps"] == 1
    assert x[0] == 0.0 and x[2399] == pytest.approx(vals[-1])
    assert x[900] == pytest.approx(vals[1000 + 20])              # sample 900 was sent twice; the later one is kept
    assert rep["overwritten_ranges"] == [[880, 1000]]            # half-open range of grid samples sent twice
    assert rep["missing_ranges"] == [[1300, 1420]]


def test_parse_truncated_gz_gives_the_prefix_without_error(tmp_path):
    """A range-fetched head is a gzip stream cut mid-way; everything before the cut must parse."""
    idx = np.arange(0, 50_000)
    rng = np.random.default_rng(0)
    vals = rng.normal(30.0, 1.0, len(idx))
    p = tmp_path / "s.gz"
    _write_gz(p, idx, vals)
    raw = p.read_bytes()
    cut = tmp_path / "cut.gz"
    cut.write_bytes(raw[: len(raw) // 2])
    rep = upmu_full.parse_to_grid(cut, tmp_path / "cut.npy", n=50_000, fs=FS)
    x = np.load(tmp_path / "cut.npy")
    k = rep["rows_read"]
    assert 10_000 < k < 40_000
    np.testing.assert_allclose(x[:k], vals[:k].astype(np.float32), rtol=1e-6)
    assert np.isnan(x[k:]).all()


def test_block_nanmean_never_fills_a_block_that_is_mostly_missing():
    x = np.arange(40, dtype=np.float64)
    x[10:20] = np.nan                    # block 1 wholly missing
    x[20] = np.nan                       # block 2 lacks one sample of ten
    m = upmu_full.block_nanmean(x, 10, min_frac=0.9)
    assert m.shape == (4,)
    assert m[0] == pytest.approx(4.5) and np.isnan(m[1])
    assert m[2] == pytest.approx(np.mean(np.arange(21, 30)))     # mean of what is there
    m2 = upmu_full.block_nanmean(x, 10, min_frac=1.0)
    assert np.isnan(m2[2])                                        # strict: any missing sample voids the block


def test_head_of_a_full_file_has_the_recorded_hash(tmp_path):
    """The 76-hour results were made from the first 200 MiB of each stream (sha256 in
    upmu_PROVENANCE.json). head_sha256 must hash exactly that many leading bytes."""
    import hashlib
    b = bytes(range(256)) * 100
    p = tmp_path / "f.bin"; p.write_bytes(b)
    assert upmu_full.head_sha256(p, 1000) == hashlib.sha256(b[:1000]).hexdigest()
    assert upmu_full.head_sha256(p, 10**9) == hashlib.sha256(b).hexdigest()


def test_head_rows_counts_the_complete_rows_in_the_leading_bytes(tmp_path):
    """Needed to rebuild the 76-hour caches (fetch_upmu's output) from the whole-stream arrays:
    the old cache is the first rows of each stream, cut to the shortest 200 MiB head."""
    idx = np.arange(0, 50_000)
    vals = np.random.default_rng(1).normal(30.0, 1.0, len(idx))
    p = tmp_path / "s.gz"
    _write_gz(p, idx, vals)
    nbytes = p.stat().st_size // 2
    cut = tmp_path / "cut.gz"; cut.write_bytes(p.read_bytes()[:nbytes])
    rep = upmu_full.parse_to_grid(cut, tmp_path / "cut.npy", n=50_000, fs=FS)
    assert upmu_full.head_rows(p, nbytes) == rep["rows_read"]
    assert upmu_full.head_rows(p, 10**9) == 50_000


def test_make_head_npz_matches_the_old_cache_layout(tmp_path):
    full = tmp_path / "full"; full.mkdir()
    rng = np.random.default_rng(2)
    prov = {}
    for s, rows in (("C1MAG", 900), ("L1MAG", 700)):
        np.save(full / f"grizzly_bus1_2_{s}.npy", rng.normal(size=1200).astype(np.float32))
        prov[f"grizzly_bus1_2/{s}"] = {"rows_in_first_200MiB": rows, "t0_ns": T0, "events": []}
    import json
    (full / "PROVENANCE_full.json").write_text(json.dumps(prov))
    upmu_full.make_head_npz(tmp_path, ["C1MAG", "L1MAG"], "head.npz", max_rows=800)
    z = np.load(tmp_path / "head.npz")
    assert len(z["C1MAG"]) == 700 and len(z["L1MAG"]) == 700          # cut to the shortest head
    assert int(z["t0_ns"]) == T0 and float(z["fs"]) == 120.0
    np.testing.assert_array_equal(z["C1MAG"], np.load(full / "grizzly_bus1_2_C1MAG.npy")[:700])


def test_void_ranges_blanks_the_stretches_whose_content_is_in_doubt():
    x = np.arange(20, dtype=np.float32)
    y = upmu_full.void_ranges(x, [[3, 5], [10, 12]])
    assert np.isnan(y[[3, 4, 10, 11]]).all() and np.isfinite(np.delete(y, [3, 4, 10, 11])).all()
    assert np.isfinite(x).all()                                  # the input is not changed
