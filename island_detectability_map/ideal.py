"""IDEAL household energy dataset (Pullinger et al. 2021, Scientific Data; Edinburgh DataShare,
DOI 10.7488/ds/2836): one-second electric-mains readings of 255 homes in Edinburgh and the
Lothians, 2016-2018. Tests in tests/test_ideal.py.

What the readings are (read from the dataset's IDEALdata.pdf, 2026-10-08):
* Whole-home mains from current clamps (a 30 A and a 100 A clamp merged into one stream),
  reported in watts, whole numbers, time-stamps in UTC. A clamp measures current, so this is
  apparent power at an assumed voltage, not real power, and carries no voltage variation.
* Nominally one reading a second. Seconds are missing where the radio link lost a reading.
* All sensor data between 08:50 and 09:50 UTC on 2018-04-17 are unreliable (dataset's own note).

Origin: verification@776d1246
"""
from __future__ import annotations

import numpy as np

MISSING = -1                                              # grid marker: no reading in that second
BAD_SPAN_UTC = ("2018-04-17T08:50:00", "2018-04-17T09:50:00")


def to_grid(t_s, w, t0_s: int, n: int):
    """Place readings (whole UTC seconds, watts) on a grid of n seconds starting at t0_s.
    Returns (int32 grid with MISSING where no reading fell, report). Two readings in one
    second: the later is kept."""
    idx = np.asarray(t_s, dtype=np.int64) - int(t0_s)
    ok = (idx >= 0) & (idx < n)
    g = np.full(n, MISSING, dtype=np.int32)
    g[idx[ok]] = np.asarray(w)[ok]                        # numpy assigns in order: the later reading wins
    placed = len(np.unique(idx[ok]))
    return g, {"seconds_read": placed, "duplicates": int(ok.sum() - placed), "outside": int((~ok).sum())}


def gap_lengths(g) -> np.ndarray:
    """Length in seconds of each run of missing seconds that follows a reading."""
    miss = np.asarray(g) == MISSING
    have = np.flatnonzero(~miss)
    if len(have) == 0:
        return np.zeros(0, dtype=np.int64)
    b = np.r_[False, miss[have[0]:], False]
    d = np.diff(b.astype(np.int8))
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def fill(g, max_hold_s: int | None = None, back_fill: bool = False):
    """Hold the last reading across gaps. Returns (float series in watts, report).
    Seconds before the first reading stay NaN unless back_fill (then they take the first
    reading); any gap longer than max_hold_s stays NaN."""
    g = np.asarray(g)
    miss = g == MISSING
    last = np.where(~miss, np.arange(len(g)), -1)
    np.maximum.accumulate(last, out=last)
    x = np.where(last >= 0, g[np.maximum(last, 0)], np.nan).astype(np.float64)
    gaps = gap_lengths(g)
    left = 0
    if max_hold_s is not None:
        have = np.flatnonzero(~miss)
        first = have[0] if len(have) else len(g)
        b = np.r_[False, miss[first:], False]
        d = np.diff(b.astype(np.int8))
        for a, e in zip(np.flatnonzero(d == 1) + first, np.flatnonzero(d == -1) + first):
            if e - a > max_hold_s:
                x[a:e] = np.nan
                left += int(e - a)
    lead = int(np.argmax(~miss)) if (~miss).any() else len(g)
    if back_fill and 0 < lead < len(g):
        x[:lead] = g[lead]
    return x, {"filled_s": int(gaps.sum()) - left, "left_empty_s": left, "leading_empty_s": lead,
               "gaps": len(gaps), "longest_gap_s": int(gaps.max()) if len(gaps) else 0}


def outage_mask(missing_count, n_homes: int, share: float = 0.5) -> np.ndarray:
    """Seconds in which more than `share` of the homes have no reading: the study's own outages."""
    return np.asarray(missing_count) > share * n_homes


def own_long_gap_share(g, outage, longer_than_s: int = 10) -> float:
    """Share of the non-outage seconds that lie in this home's gaps longer than longer_than_s
    (seconds before its first reading count as one gap)."""
    miss = np.asarray(g) == MISSING
    b = np.r_[False, miss, False]
    d = np.diff(b.astype(np.int8))
    long = np.zeros(len(miss), dtype=bool)
    for a, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if e - a > longer_than_s:
            long[a:e] = True
    keep = ~np.asarray(outage, dtype=bool)
    return float((long & keep).sum() / keep.sum())


def shuffled(homes, seed: int) -> list:
    """The homes in a seeded random order that does not depend on the order they were given in."""
    h = sorted(int(v) for v in homes)
    return [h[i] for i in np.random.default_rng(seed).permutation(len(h))]


def feeder(series: dict, homes: list):
    """Deal homes onto three phases in the order given and sum each phase.
    Returns ((3, n) array in kVA, {home: phase}). NaN wherever any home on that phase is NaN."""
    phase_of = {h: k % 3 for k, h in enumerate(homes)}
    n = len(series[homes[0]])
    kva = np.zeros((3, n))
    for h in homes:
        kva[phase_of[h]] += series[h]
    return kva / 1e3, phase_of
