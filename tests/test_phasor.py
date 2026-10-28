"""Phasor residual and real/reactive split for the current-monitor thread.
Written before the code.

Conventions under test: per-phase complex power S_k = V_k * I_k * exp(j(angle_V_k - angle_I_k)),
in kW + j kvar, so each phase's current is expressed in its own voltage's frame. Positive
reactive power means the current lags the voltage (motor-like).

Origin: verification@23d3a497
"""
import numpy as np
import pytest

from island_detectability_map import phasor

V = 7200.0                                   # line-to-neutral volts, as on the LBNL 12 kV bus
VANG = np.array([10.0, -110.0, 130.0])       # three phases 120 degrees apart, arbitrary reference


def _background(n=50, pf_angles_deg=(20.0, 30.0, 15.0), amps=(30.0, 34.0, 33.0)):
    """Steady three-phase background whose power-factor angle differs by phase."""
    vmag = np.full((3, n), V)
    vang = np.repeat(VANG[:, None], n, axis=1)
    imag = np.repeat(np.array(amps)[:, None], n, axis=1)
    iang = vang - np.array(pf_angles_deg)[:, None]      # current lags voltage
    return vmag, vang, imag, iang


def test_complex_power_has_the_known_real_and_reactive_parts():
    s = phasor.complex_power(np.array([V]), np.array([40.0]), np.array([50.0]), np.array([40.0 - 25.0]))
    apparent = V * 50.0 / 1e3
    assert s.real[0] == pytest.approx(apparent * np.cos(np.deg2rad(25.0)))
    assert s.imag[0] == pytest.approx(apparent * np.sin(np.deg2rad(25.0)))     # lagging current -> positive
    assert s.imag[0] > 0


def test_complex_power_handles_angle_wrap_at_180_degrees():
    # voltage at +179, current lagging by 20 degrees wraps to -161 + 360 -> 159; reported as 159 or -201
    a = phasor.complex_power(np.array([V]), np.array([179.0]), np.array([50.0]), np.array([159.0]))
    b = phasor.complex_power(np.array([V]), np.array([-179.0]), np.array([50.0]), np.array([161.0]))  # -179 - 20 = -199 = +161
    assert a.real[0] == pytest.approx(b.real[0])
    assert a.imag[0] == pytest.approx(b.imag[0])


def test_complex_power_ignores_a_common_rotation_of_both_angles():
    """The sensor's angles turn together when the grid is off 60 Hz; power must not."""
    vmag, vang, imag, iang = _background()
    s0 = phasor.complex_power(vmag, vang, imag, iang)
    drift = np.linspace(0.0, 700.0, vmag.shape[1])       # degrees, several full turns
    s1 = phasor.complex_power(vmag, vang + drift, imag, iang + drift)
    np.testing.assert_allclose(s1, s0, rtol=1e-9)


def test_equal_loads_on_three_phases_give_equal_complex_power_in_own_frame():
    vmag, vang, imag, iang = _background(pf_angles_deg=(25.0, 25.0, 25.0), amps=(30.0, 30.0, 30.0))
    s = phasor.complex_power(vmag, vang, imag, iang)
    np.testing.assert_allclose(s[1], s[0], rtol=1e-9)
    np.testing.assert_allclose(s[2], s[0], rtol=1e-9)


def test_add_load_raises_complex_power_by_exactly_the_load():
    vmag, vang, imag, iang = _background()
    s0 = phasor.complex_power(vmag, vang, imag, iang)
    imag2, iang2 = phasor.add_load(vmag[1], vang[1], imag[1], iang[1], p_kw=20.0, q_kvar=5.0)
    s1 = phasor.complex_power(vmag[1], vang[1], imag2, iang2)
    np.testing.assert_allclose(s1 - s0[1], 20.0 + 5.0j, rtol=1e-9)


def test_balanced_complex_step_cancels_exactly_in_the_phasor_residual():
    vmag, vang, imag, iang = _background()
    s0 = phasor.complex_power(vmag, vang, imag, iang)
    new = [phasor.add_load(vmag[k], vang[k], imag[k], iang[k], p_kw=30.0, q_kvar=45.0) for k in range(3)]  # motor-like
    s1 = phasor.complex_power(vmag, vang, np.array([m for m, _ in new]), np.array([a for _, a in new]))
    for k in range(3):
        np.testing.assert_allclose(phasor.complex_residual(s1, k), phasor.complex_residual(s0, k), atol=1e-9)


def test_single_phase_step_passes_the_phasor_residual_with_unit_gain():
    vmag, vang, imag, iang = _background()
    s0 = phasor.complex_power(vmag, vang, imag, iang)
    imag2, iang2 = imag.copy(), iang.copy()
    imag2[2], iang2[2] = phasor.add_load(vmag[2], vang[2], imag[2], iang[2], p_kw=25.0)
    s1 = phasor.complex_power(vmag, vang, imag2, iang2)
    d = [phasor.complex_residual(s1, k) - phasor.complex_residual(s0, k) for k in range(3)]
    np.testing.assert_allclose(d[2], 25.0 + 0.0j, atol=1e-9)            # its own phase: full size
    np.testing.assert_allclose(d[0], -12.5 + 0.0j, atol=1e-9)           # the other two: minus half each
    np.testing.assert_allclose(d[1], -12.5 + 0.0j, atol=1e-9)


def test_magnitude_and_phasor_residuals_differ_when_background_angles_differ():
    """A balanced real-power step added to phases with different background angles moves
    each current magnitude by a different amount, so the magnitude residual shifts; the
    phasor residual does not."""
    vmag, vang, imag, iang = _background(pf_angles_deg=(10.0, 40.0, 25.0))
    new = [phasor.add_load(vmag[k], vang[k], imag[k], iang[k], p_kw=30.0) for k in range(3)]
    imag2 = np.array([m for m, _ in new]); iang2 = np.array([a for _, a in new])
    mag0 = vmag * imag / 1e3; mag1 = vmag * imag2 / 1e3
    shift_mag = [float(np.mean((mag1[k] - np.delete(mag1, k, 0).mean(0)) - (mag0[k] - np.delete(mag0, k, 0).mean(0))))
                 for k in range(3)]
    s0 = phasor.complex_power(vmag, vang, imag, iang); s1 = phasor.complex_power(vmag, vang, imag2, iang2)
    shift_pha = [float(np.mean(np.abs(phasor.complex_residual(s1, k) - phasor.complex_residual(s0, k)))) for k in range(3)]
    assert max(abs(v) for v in shift_mag) > 1.0          # kVA: a visible artefact
    assert max(shift_pha) < 1e-9


def test_a_real_power_step_shows_in_the_current_magnitude_at_about_the_cosine_of_the_background_angle():
    vmag, vang, imag, iang = _background(pf_angles_deg=(23.0, 23.0, 23.0))
    imag2, _ = phasor.add_load(vmag[0], vang[0], imag[0], iang[0], p_kw=2.0)      # small step: first-order regime
    gain = float(np.mean(vmag[0] * (imag2 - imag[0]) / 1e3)) / 2.0
    assert gain == pytest.approx(np.cos(np.deg2rad(23.0)), abs=0.01)


def test_step_pq_recovers_a_planted_step_and_its_reactive_share():
    rng = np.random.default_rng(3)
    n = 400
    p = 200.0 + 0.05 * rng.standard_normal(n); q = 80.0 + 0.05 * rng.standard_normal(n)
    p[211:] += 12.0; q[211:] += 18.0                    # step lands between samples 210 and 211
    dp, dq = phasor.step_pq(p, q, np.array([210]), w=3)
    assert dp[0] == pytest.approx(12.0, abs=0.2)
    assert dq[0] == pytest.approx(18.0, abs=0.2)


def test_step_pq_skips_the_transition_sample():
    """An event index t means the step happens during sample t+... the detector's convention
    (run_sync.upsteps) is: after = x[t+1 : t+1+w], before = x[t-w : t]; sample t is skipped."""
    p = np.zeros(40); q = np.zeros(40)
    p[20] = 5.0; p[21:] = 10.0                          # sample 20 is a half-way value
    dp, _ = phasor.step_pq(p, q, np.array([20]), w=3)
    assert dp[0] == pytest.approx(10.0)


def test_balanced_real_only_flags_the_three_phase_island_edge_and_not_a_motor_or_a_single_phase_step():
    island = (np.array([[6.7], [6.6], [6.8]]), np.array([[0.2], [-0.1], [0.3]]))        # kW, kvar per phase
    motor = (np.array([[6.7], [6.6], [6.8]]), np.array([[9.0], [9.5], [8.8]]))
    single = (np.array([[20.0], [0.1], [-0.2]]), np.array([[0.5], [0.0], [0.0]]))
    assert phasor.balanced_real_only(*island)[0]
    assert not phasor.balanced_real_only(*motor)[0]
    assert not phasor.balanced_real_only(*single)[0]
