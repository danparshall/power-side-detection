"""Low-pass before the 120-to-12 samples-a-second step. Written before the code.

The pipeline so far took a plain mean of every ten samples. A ten-sample mean passes a
fraction of a 10 Hz line, which then appears at 2 Hz in the 12-a-second series (and 20 Hz
at 4 Hz), inside the band the spectral detector searches. It also attenuates the top of that
band (about 10% at 3 Hz). The replacement must:
  * keep the band the detector searches (up to 5.5 Hz) at unit gain,
  * remove what would fold into that band (6.5 Hz and above),
  * put each output sample at the centre of the same ten-sample block as before, so hour
    boundaries and time-stamps do not move,
  * never fill: an output sample whose filter span touches a missing input sample is NaN.

Origin: verification@f68ea5ab
"""
import numpy as np
import pytest

from island_detectability_map import decimate as D
from island_detectability_map.upmu_full import block_nanmean

FS, K = 120.0, 10


def tone(f, n=120 * 200, amp=1.0, phase=0.3):
    return amp * np.sin(2 * np.pi * f * np.arange(n) / FS + phase)


def rms_inside(y, margin=100):
    v = y[margin:-margin]
    assert np.isfinite(v).all()
    return float(np.sqrt(np.mean(v ** 2)))


def test_block_mean_folds_a_10_hz_line_to_2_hz_which_is_the_flaw_being_fixed():
    y = block_nanmean(tone(10.0), K)
    assert rms_inside(y) / np.sqrt(0.5) == pytest.approx(0.193, abs=0.005)   # sin(150 deg) / (10 sin(15 deg))
    spec = np.abs(np.fft.rfft(y[:2400] * np.hanning(2400)))
    f = np.fft.rfftfreq(2400, 1 / 12.0)
    assert f[np.argmax(spec)] == pytest.approx(2.0, abs=0.01)


@pytest.mark.parametrize("f_alias", [6.5, 7.0, 10.0, 14.0, 20.0, 30.0, 50.0])
def test_lowpass_removes_what_would_fold_into_the_search_band(f_alias):
    y = D.lowpass_decimate(tone(f_alias), K, FS)
    assert rms_inside(y) / np.sqrt(0.5) < 10 ** (-60 / 20)                  # at least 60 dB down


@pytest.mark.parametrize("f_keep", [0.1, 1.0, 3.0, 5.0, 5.5])
def test_lowpass_keeps_the_search_band_at_unit_gain(f_keep):
    y = D.lowpass_decimate(tone(f_keep), K, FS)
    assert rms_inside(y) / np.sqrt(0.5) == pytest.approx(1.0, abs=0.005)


def test_block_mean_by_contrast_loses_a_tenth_at_3_hz():
    y = block_nanmean(tone(3.0), K)
    assert rms_inside(y) / np.sqrt(0.5) == pytest.approx(0.901, abs=0.005)


def test_constant_stays_constant():
    y = D.lowpass_decimate(np.full(12000, 709.0), K, FS)
    inside = y[np.isfinite(y)]
    assert len(inside) > 1000
    assert np.allclose(inside, 709.0, rtol=0, atol=1e-9)


def test_output_sample_sits_at_the_centre_of_the_same_block_as_the_block_mean():
    x = np.arange(24000, dtype=float)                  # a ramp: a symmetric filter returns the value at its centre
    y = D.lowpass_decimate(x, K, FS)
    j = np.where(np.isfinite(y))[0]
    assert len(y) == len(x) // K
    assert np.allclose(y[j], K * j + (K - 1) / 2, atol=1e-6)
    assert np.allclose(block_nanmean(x, K)[j], y[j], atol=1e-6)


def test_edges_without_full_filter_span_are_nan_not_padded():
    y = D.lowpass_decimate(np.ones(24000), K, FS)
    half = D.design(K, FS).size // 2
    n_edge = int(np.ceil((half - K / 2) / K))
    assert np.isnan(y[:n_edge]).all() and np.isnan(y[-n_edge:]).all()
    assert np.isfinite(y[n_edge:-n_edge]).all()


def test_missing_input_blanks_exactly_the_outputs_whose_span_touches_it_and_nothing_is_filled():
    x = tone(1.0, n=48000)
    clean = D.lowpass_decimate(x, K, FS)
    xm = x.copy(); xm[24000:24120] = np.nan            # one missing second, as in the LBNL record
    y = D.lowpass_decimate(xm, K, FS)
    h = D.design(K, FS).size
    centre = K * np.arange(len(y)) + (K - 1) / 2
    touches = (centre + h / 2 > 24000) & (centre - h / 2 < 24120)
    edge = np.isnan(clean)
    assert np.isnan(y[touches]).all()
    assert np.array_equal(np.isnan(y), touches | edge)
    assert np.allclose(y[~touches & ~edge], clean[~touches & ~edge], atol=1e-12)


def test_chunked_run_equals_whole_run():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(60000); x[31000:31050] = np.nan
    whole = D.lowpass_decimate(x, K, FS)
    parts = D.lowpass_decimate_chunked(x, K, FS, chunk=12000)
    assert parts.shape == whole.shape
    assert np.array_equal(np.isnan(parts), np.isnan(whole))
    ok = np.isfinite(whole)
    assert np.allclose(parts[ok], whole[ok], atol=1e-10)


def test_chunk_must_be_a_multiple_of_the_step():
    with pytest.raises(ValueError):
        D.lowpass_decimate_chunked(np.zeros(1000), K, FS, chunk=1234)
