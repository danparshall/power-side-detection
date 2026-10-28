"""Gridding and feeder-building for the IDEAL one-second household mains records.
Written before the code.

A home's file is rows of 'UTC time-stamp, watts', nominally one a second, with seconds missing
where the radio link dropped a reading. Homes are placed on one grid of whole seconds; a second
with no reading is marked, not invented. Gaps are then filled by holding the last reading, and
what was filled is counted, because a held value is flatter than the home really was.
A UK low-voltage feeder is three-phase with each home on one phase, so a feeder here is a set of
homes dealt onto three phases and summed per phase.

Origin: verification@776d1246
"""
import numpy as np
import pytest

from island_detectability_map import ideal


def test_readings_land_in_their_own_second_and_missing_seconds_are_marked():
    g, rep = ideal.to_grid(np.array([100, 101, 103]), np.array([5, 6, 8]), t0_s=100, n=5)
    assert g.dtype == np.int32
    assert g.tolist() == [5, 6, ideal.MISSING, 8, ideal.MISSING]
    assert rep == {"seconds_read": 3, "duplicates": 0, "outside": 0}


def test_readings_outside_the_window_are_dropped_and_counted():
    g, rep = ideal.to_grid(np.array([99, 100, 105]), np.array([1, 2, 3]), t0_s=100, n=5)
    assert g.tolist() == [2] + [ideal.MISSING] * 4
    assert rep["outside"] == 2 and rep["seconds_read"] == 1


def test_two_readings_in_one_second_keep_the_later_and_are_counted():
    g, rep = ideal.to_grid(np.array([100, 100, 101]), np.array([7, 9, 4]), t0_s=100, n=2)
    assert g.tolist() == [9, 4]
    assert rep["duplicates"] == 1 and rep["seconds_read"] == 2


def test_gaps_are_filled_by_holding_the_last_reading():
    M = ideal.MISSING
    x, rep = ideal.fill(np.array([5, M, M, 8, M, 9], dtype=np.int32))
    assert x.tolist() == [5.0, 5.0, 5.0, 8.0, 8.0, 9.0]
    assert rep["filled_s"] == 3 and rep["longest_gap_s"] == 2 and rep["gaps"] == 2


def test_seconds_before_the_first_reading_stay_empty():
    M = ideal.MISSING
    x, rep = ideal.fill(np.array([M, M, 5, M], dtype=np.int32))
    assert np.isnan(x[:2]).all() and x[2:].tolist() == [5.0, 5.0]
    assert rep["leading_empty_s"] == 2 and rep["filled_s"] == 1


def test_a_gap_longer_than_the_hold_limit_stays_empty():
    M = ideal.MISSING
    x, rep = ideal.fill(np.array([5, M, M, M, 8, M, 9], dtype=np.int32), max_hold_s=2)
    assert x[0] == 5.0 and np.isnan(x[1:4]).all() and x[4:].tolist() == [8.0, 8.0, 9.0]
    assert rep["filled_s"] == 1 and rep["left_empty_s"] == 3 and rep["longest_gap_s"] == 3


def test_gap_report_lists_gaps_longer_than_a_given_length():
    M = ideal.MISSING
    g = np.array([5, M, 6, M, M, M, 7, M, M, 8], dtype=np.int32)
    assert ideal.gap_lengths(g).tolist() == [1, 3, 2]


def test_a_feeder_deals_homes_onto_three_phases_and_sums_each_in_kva():
    homes = {h: np.full(4, 1000.0 * h) for h in (1, 2, 3, 4, 5, 6)}
    kva, phase_of = ideal.feeder(homes, [1, 2, 3, 4, 5, 6])
    assert phase_of == {1: 0, 2: 1, 3: 2, 4: 0, 5: 1, 6: 2}        # dealt in the order given
    assert kva.shape == (3, 4)
    assert kva[:, 0].tolist() == [5.0, 7.0, 9.0]                    # watts summed, in kVA


def test_a_feeder_is_empty_wherever_one_of_its_homes_is_empty():
    homes = {1: np.array([1000.0, np.nan]), 2: np.array([1000.0, 1000.0]), 3: np.array([1000.0, 1000.0])}
    kva, _ = ideal.feeder(homes, [1, 2, 3])
    assert kva[:, 0].tolist() == [1.0, 1.0, 1.0]
    assert np.isnan(kva[0, 1]) and kva[1, 1] == 1.0


def test_the_dealing_order_is_seeded_and_ignores_input_order():
    order = ideal.shuffled([5, 3, 9, 1, 7, 2], seed=1)
    assert sorted(order) == [1, 2, 3, 5, 7, 9]
    assert ideal.shuffled([5, 3, 9, 1, 7, 2], seed=1) == order       # seeded
    assert ideal.shuffled([2, 7, 1, 9, 3, 5], seed=1) == order       # and independent of input order


# The study's servers went down from time to time, and then every home is empty at once. Those
# stretches are not a home's own gaps: the feeder is left empty there and the hours are not used.

def test_an_outage_is_where_more_than_the_given_share_of_homes_are_empty_at_once():
    count = np.array([0, 1, 5, 6, 10])
    assert ideal.outage_mask(count, n_homes=10, share=0.5).tolist() == [False, False, False, True, True]


def test_a_homes_own_long_gaps_are_counted_outside_outages_only():
    M = ideal.MISSING
    #            0  1  2  3  4  5  6  7  8  9
    g = np.array([5, M, M, M, 6, M, 7, M, M, 8], dtype=np.int32)
    outage = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0], dtype=bool)
    # gaps: seconds 1-3 (3 s), second 5 (1 s), seconds 7-8 (inside the outage). Longer than 2 s and
    # outside the outage: seconds 1-3 only, out of 8 non-outage seconds.
    assert ideal.own_long_gap_share(g, outage, longer_than_s=2) == pytest.approx(3 / 8)


def test_seconds_before_a_homes_first_reading_count_as_its_own_gap():
    M = ideal.MISSING
    g = np.array([M, M, M, 5, 6], dtype=np.int32)
    assert ideal.own_long_gap_share(g, np.zeros(5, dtype=bool), longer_than_s=2) == pytest.approx(3 / 5)


def test_holding_can_start_from_the_first_reading_backwards():
    M = ideal.MISSING
    x, rep = ideal.fill(np.array([M, M, 5, M], dtype=np.int32), back_fill=True)
    assert x.tolist() == [5.0, 5.0, 5.0, 5.0] and rep["leading_empty_s"] == 2


# The download keeps only the rows of one time window from a file that covers a year or more.

def _gz(rows):
    import gzip
    return gzip.compress("".join(f"{t},{w}\n" for t, w in rows).encode())


def _inside(txt, a, b):
    return [ln for ln in txt.decode().splitlines() if a <= ln[:19] < b]


def test_window_rows_returns_every_row_inside_the_window_as_whole_lines():
    from island_detectability_map import fetch_ideal
    rows = [(f"2018-03-{d:02d} 00:00:{s:02d}", d * 100 + s) for d in range(1, 29) for s in range(60)]
    a, b = "2018-03-05 00:00:00", "2018-03-07 00:00:00"
    txt = fetch_ideal.window_rows(_gz(rows), a, b, chunk=64)
    want = [f"{t},{w}" for t, w in rows if a <= t < b]
    assert _inside(txt, a, b) == want and len(want) == 120
    assert all(ln.count(",") == 1 and len(ln) > 20 for ln in txt.decode().splitlines())
    assert len(txt.decode().splitlines()) < len(rows)                 # blocks outside the window are dropped


def test_window_rows_keeps_a_last_row_that_has_no_final_newline():
    from island_detectability_map import fetch_ideal
    import gzip
    gz = gzip.compress(b"2018-03-01 00:00:00,1\n2018-03-02 00:00:00,2\n2018-03-03 00:00:00,3")
    a, b = "2018-03-02 00:00:00", "2018-03-09 00:00:00"
    assert _inside(fetch_ideal.window_rows(gz, a, b, chunk=16), a, b) == ["2018-03-02 00:00:00,2", "2018-03-03 00:00:00,3"]
