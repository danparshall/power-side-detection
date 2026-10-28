# Origin: verification@81ea80da
import numpy as np

from synthetic_lookalikes import PARAMS, loads, spectrogram


def test_every_load_is_one_hour_at_10ms():
    for name, p in loads(PARAMS).items():
        assert p.shape == (360_000,), name
        assert np.all(p >= 0), name


def test_training_pulse_period_and_swing():
    p = loads(PARAMS)["training_16gpu"]
    seg = p[: 60 * 100]  # first minute, before any checkpoint
    high = seg > seg.mean()
    swing = np.median(seg[high]) - np.median(seg[~high])
    assert np.isclose(swing, PARAMS["training_16gpu"]["swing_kw"], atol=0.05)
    f, _, S = spectrogram(p, PARAMS)
    col = S[:, 0]
    band = (f > 0.05) & (f < 0.2)
    assert np.isclose(f[band][np.argmax(col[band])], 1 / 10.2, rtol=0.03)


def test_flat_load_has_no_line():
    p = loads(PARAMS)["crypto_miner"]
    f, _, S = spectrogram(p, PARAMS)
    band = (f > 0.05) & (f < 0.2)
    col = S[band, 0]
    assert col.max() / np.median(col) < 20


def test_spectrogram_grid():
    p = loads(PARAMS)["ev_charger"]
    f, t, S = spectrogram(p, PARAMS)
    assert len(t) == 11  # 10-min window, 5-min stride, one hour
    assert S.shape == (len(f), 11)
    assert f[-1] <= 50


def test_monitor_current_uses_locked_rotor_kva():
    from synthetic_lookalikes import currents

    q = PARAMS["well_pump"]
    i = currents(PARAMS)["well_pump"]
    peak_a = q["locked_rotor_kva"] * 1e3 / PARAMS["volts"]
    assert np.isclose(i.max(), peak_a, rtol=0.05)
    run_a = q["run_kw"] * 1e3 / q["pf"] / PARAMS["volts"]
    assert np.isclose(np.median(i[i > 1]), run_a, rtol=0.05)


def test_resistive_load_current_is_p_over_v():
    from synthetic_lookalikes import currents, loads

    p = loads(PARAMS)["laser_fuser"]
    i = currents(PARAMS)["laser_fuser"]
    k = PARAMS["laser_fuser"]
    heater_on = p > k["motor_kw"] + k["heater_kw"] / 2
    assert np.isclose(np.median(i[heater_on]), 1e3 * np.median(p[heater_on]) / PARAMS["volts"], rtol=0.03)
