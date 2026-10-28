"""Tests for island_detectability_map.phase_split: a training island's swing split over three phases by
NVIDIA's DGX SuperPOD H100 PSU phase tables, mapped through a delta-wye (Dyn) service transformer, and
scored by the report's spectral detector (and a harmonic-sum variant) through welch_engine.

Behaviour under test:
  * the transcribed PSU tables give the phase counts read off NVIDIA's Tables 11-14;
  * per-PSU current-sharing error moves the shares only when it differs between phases;
  * the Dyn mapping keeps a balanced load balanced, puts a single line-to-neutral load on two primary
    lines at 1/sqrt(3) each, keeps the in-phase total, and halves the in-phase unbalance;
  * unbalance coefficients agree with phases.imbalance applied to the injected currents;
  * the prepare/inject split of welch_engine.experiment reproduces experiment exactly, at any coefficient;
  * the family score counts an hour found when any channel in it clears the shared threshold;
  * the harmonic-sum statistic is the mean, over harmonics, of the best ratio near each harmonic.

Origin: verification@29bb8624
"""
import itertools

import numpy as np
import pytest

from island_detectability_map import phases, welch_engine as WE
from island_detectability_map import phase_split as PS


# --- NVIDIA tables -------------------------------------------------------------------------------

@pytest.mark.parametrize("config, counts", [
    ("n1_1dgx", (2, 3, 1)),
    ("n1_2dgx", (5, 4, 3)),
    ("n1_4dgx", (8, 8, 8)),
    ("enh_1dgx", (3, 3, 0)),
    ("enh_2dgx", (6, 3, 3)),
    ("enh_4dgx", (9, 9, 6)),
])
def test_nvidia_tables_give_the_phase_counts_read_off_the_page(config, counts):
    assert PS.phase_counts(config) == counts


@pytest.mark.parametrize("config, gpus", [("n1_1dgx", 8), ("n1_2dgx", 16), ("n1_4dgx", 32), ("enh_2dgx", 16)])
def test_gpu_count_is_eight_per_dgx(config, gpus):
    assert PS.n_gpus(config) == gpus


def test_equal_sharing_gives_counts_over_six_per_system():
    np.testing.assert_allclose(PS.shares("n1_2dgx"), np.array([5, 4, 3]) / 12)


def test_a_common_sharing_error_on_every_psu_changes_nothing():
    n = len(PS.psu_phases("n1_2dgx"))
    np.testing.assert_allclose(PS.shares("n1_2dgx", np.full(n, 0.05)), np.array([5, 4, 3]) / 12)


def test_sharing_error_moves_current_only_within_each_system():
    """Each system's load is fixed by its GPUs; a PSU drawing 5% more takes it from its own system's
    other PSUs. The four N+1 systems have 2, 3, 2 and 1 PSUs on phase A (Tables 11-12)."""
    ph = PS.psu_phases("n1_4dgx")
    eps = np.where(np.array(ph) == "A", 0.05, 0.0)
    f = PS.shares("n1_4dgx", eps)
    n_a = np.array([2, 3, 2, 1])
    assert f[0] == pytest.approx(np.mean(1.05 * n_a / (1.05 * n_a + 6 - n_a)))
    assert f[0] > 1 / 3
    assert f.sum() == pytest.approx(1.0)


# --- Dyn transformer -----------------------------------------------------------------------------

def test_dyn_keeps_a_balanced_load_balanced():
    J = PS.dyn_line_currents(np.full(3, 1 / 3))
    np.testing.assert_allclose(np.abs(J), 1 / 3)
    np.testing.assert_allclose(PS.dyn_inphase(np.full(3, 1 / 3)), 1 / 3)


@pytest.mark.parametrize("group", ["Dyn1", "Dyn11"])
def test_dyn_puts_one_line_to_neutral_load_on_two_primary_lines(group):
    one = np.array([1.0, 0.0, 0.0])
    assert sorted(np.abs(PS.dyn_line_currents(one, group))) == pytest.approx([0.0, 1 / np.sqrt(3), 1 / np.sqrt(3)])
    assert sorted(PS.dyn_inphase(one, group)) == pytest.approx([0.0, 0.5, 0.5])


@pytest.mark.parametrize("group", ["Dyn1", "Dyn11"])
def test_a_lagging_background_splits_a_single_load_unevenly_between_its_two_lines(group):
    """A magnitude meter sees the job along the background current. At a 26 degree lag the two
    lines a single load lands on are 30 -/+ 26 degrees off that direction."""
    one = np.array([1.0, 0.0, 0.0])
    got = sorted(PS.dyn_inphase(one, group, bg_lag_deg=26.0))
    want = sorted([0.0, np.cos(np.radians(4.0)) / np.sqrt(3), np.cos(np.radians(56.0)) / np.sqrt(3)])
    assert got == pytest.approx(want, abs=1e-12)


def test_a_lagging_background_scales_an_unmapped_load_by_the_cosine():
    f = PS.shares("n1_2dgx")
    np.testing.assert_allclose(PS.project_unmapped(f, bg_lag_deg=26.0), f * np.cos(np.radians(26.0)))
    np.testing.assert_allclose(PS.project_unmapped(f), f)


@pytest.mark.parametrize("seed", range(5))
def test_dyn_keeps_the_inphase_total_and_halves_the_unbalance(seed):
    f = np.random.default_rng(seed).dirichlet(np.ones(3))
    g = PS.dyn_inphase(f)
    assert g.sum() == pytest.approx(1.0)
    assert sorted(PS.unbalance_coefs(g)) == pytest.approx(sorted(-0.5 * PS.unbalance_coefs(f)))


def test_unbalance_coefs_agree_with_phases_imbalance():
    rng = np.random.default_rng(3)
    x = rng.normal(100.0, 5.0, (3, 500))
    job = rng.normal(0.0, 1.0, 500)
    f = PS.shares("enh_2dgx")
    y = x + np.outer(f, job)
    c = PS.unbalance_coefs(f)
    for k in range(3):
        np.testing.assert_allclose(phases.imbalance(y, k) - phases.imbalance(x, k), c[k] * job, atol=1e-9)


def test_the_two_vector_groups_put_the_unbalance_on_different_lines():
    f = PS.shares("n1_2dgx")
    g1, g11 = PS.dyn_inphase(f, "Dyn1"), PS.dyn_inphase(f, "Dyn11")
    assert not np.allclose(g1, g11)
    assert sorted(g1) == pytest.approx(sorted(g11))


def test_assignments_are_every_distinct_ordering_onto_the_bus_phases():
    v = PS.dyn_inphase(PS.shares("n1_2dgx"))
    got = {tuple(np.round(a, 9)) for a in PS.assignments(v)}
    assert got == set(itertools.permutations(np.round(v, 9)))
    assert len(PS.assignments(np.full(3, 1 / 3))) == 1


# --- detector engine -----------------------------------------------------------------------------

FS, WIN, SEG = 1.0, 3600.0, 100.0
KW = dict(band=(0.08, 0.5), f0_range=(0.1, 0.4), seg_s=SEG, char_hours=0)
SCALES = [0.5, 1.0, 2.0, 4.0]


def _hours(n_h, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(n_h * WIN * FS)) / FS
    return rng.standard_normal(len(t)) + 3.0 * np.sin(2 * np.pi * 0.31 * t)


def _trace(seed=1):
    t = np.arange(1500) / FS
    return np.sign(np.sin(2 * np.pi * t / 10.2)) + 0.1 * np.random.default_rng(seed).standard_normal(len(t))


@pytest.mark.parametrize("coef", [1.0, 0.5, -0.25])
def test_prepared_injection_reproduces_experiment_at_any_coefficient(coef):
    x, tr = _hours(8), _trace()
    char = [0, 1, 2, 3]
    ref = WE.experiment(x, FS, WIN, tr, 7, [coef * s for s in SCALES], char, **KW)
    st = WE.prepare(x, FS, WIN, tr, 7, len(SCALES), char, **KW)
    null, inj = WE.inject(st, [coef * s for s in SCALES])
    np.testing.assert_array_equal(null, ref["null_stats"])
    np.testing.assert_array_equal(inj, ref["inj_stats"])


def test_family_score_counts_an_hour_found_if_any_channel_clears_the_shared_threshold():
    """Threshold = worst learning-hour null over the family x margin; an hour is a false alarm, or the
    job is found in it, when a statistic is strictly above the threshold. Found and false alarm are
    counted separately, as the report counts them."""
    nulls = [np.array([1.0, 2.0, 1.0, 1.0, 1.0]), np.array([1.5, 1.0, 1.0, 3.0, 1.0])]
    injs = [np.array([[1.0, 1.0, 5.0, 1.0, 2.0]]), np.array([[1.0, 1.0, 1.0, 9.0, 1.0]])]
    r = PS.score_family(nulls, injs, learn=[0, 1], score=[2, 3, 4], margin=1.0)
    assert r["threshold"] == pytest.approx(2.0)
    assert r["false_alarms"] == 1                      # hour 3: channel 2's null is 3.0
    assert r["found"] == pytest.approx([2 / 3])        # hours 2 and 3; hour 4 ties the threshold
    assert r["scored_hours"] == 3


DF = 1 / 60.0
FREQS = np.arange(1, 300) * DF


def _bin(f):
    return int(np.argmin(np.abs(FREQS - f)))


def test_harmonic_sum_is_the_mean_ratio_at_the_harmonics():
    K = 5
    r = np.ones(len(FREQS))
    for k in range(1, K + 1):
        r[_bin(k * 0.1)] = 4.0
    comb = PS.HarmonicSum(FREQS, f0_grid=np.array([0.1]), n_harm=K)
    assert comb(r[None])[0] == pytest.approx(4.0)
    r1 = np.ones(len(FREQS)); r1[_bin(0.1)] = 4.0
    assert comb(r1[None])[0] == pytest.approx((4.0 + K - 1) / K)


def test_harmonic_sum_takes_the_best_bin_within_one_bin_of_each_harmonic():
    r = np.ones(len(FREQS))
    r[_bin(0.1) + 1] = 4.0                             # one bin above harmonic 1: counted
    r[_bin(0.2) + 2] = 9.0                             # two bins above harmonic 2: not counted
    comb = PS.HarmonicSum(FREQS, f0_grid=np.array([0.1]), n_harm=2)
    assert comb(r[None])[0] == pytest.approx((4.0 + 1.0) / 2)


def test_harmonic_sum_skips_harmonics_whose_bins_are_all_notched():
    keep = np.abs(FREQS - 0.2) > 0.03                  # harmonic 2 of 0.1 Hz notched out with its neighbours
    r = np.ones(keep.sum())
    fk = FREQS[keep]
    r[np.argmin(np.abs(fk - 0.1))] = 3.0
    r[np.argmin(np.abs(fk - 0.3))] = 5.0
    comb = PS.HarmonicSum(fk, f0_grid=np.array([0.1]), n_harm=3)
    assert comb(r[None])[0] == pytest.approx(4.0)      # (3 + 5) / 2; counting harmonic 2 as 1 would give 3


def test_harmonic_sum_reports_the_best_fundamental_on_the_grid():
    r = np.ones(len(FREQS))
    for k in (1, 2, 3):
        r[_bin(k * 0.15)] = 6.0
    comb = PS.HarmonicSum(FREQS, f0_grid=np.array([0.1, 0.15, 0.2]), n_harm=3)
    assert comb(r[None])[0] == pytest.approx(6.0)
