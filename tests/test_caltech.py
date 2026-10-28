"""Loader for the Caltech SoCal 28-bus 1-second magnitude files.
Written before the code.

Fixtures are small synthetic files in the dataset's own layout (magnitudes/<register>.csv with
columns t,v; a network JSON with EgaugeMeter -> registers), modelled on real meters:
egauge_11 (plain), egauge_13 (channels in reverse phase order), egauge_17 (line-to-line
voltage on two channels, V_bc stored negative, V_ca meaningless), egauge_16 (sub-circuits).

Origin: verification@d36e6d97
"""
import json

import numpy as np
import pandas as pd
import pytest

from island_detectability_map import caltech

T0 = "2025-03-12T00:00:00"
N = 600


def reg(name, element, unit, meter):
    return {"name": name, "element": element, "unit": unit, "value": name, "data_file": f"{meter}-{name}"}


NETWORK = {"EgaugeMeter": [
    {"name": "egauge_11", "model": "EG4015", "registers": [
        {"name": "Mains_Power", "element": "cb_134", "unit": "W", "value": "S1*L1+S2*L2+S3*L3", "data_file": "egauge_11-Mains_Power"},
        reg("L1", "bus_1070.ag", "V", "egauge_11"), reg("L2", "bus_1070.bg", "V", "egauge_11"), reg("L3", "bus_1070.cg", "V", "egauge_11"),
        reg("CT1", "cb_134.a", "A", "egauge_11"), reg("CT2", "cb_134.b", "A", "egauge_11"), reg("CT3", "cb_134.c", "A", "egauge_11")]},
    {"name": "egauge_13", "model": "EG4015", "registers": [
        reg("L3", "bus_1098.ag", "V", "egauge_13"), reg("L2", "bus_1098.bg", "V", "egauge_13"), reg("L1", "bus_1098.cg", "V", "egauge_13"),
        reg("CT3", "cb_140.a", "A", "egauge_13"), reg("CT2", "cb_140.b", "A", "egauge_13"), reg("CT1", "cb_140.c", "A", "egauge_13")]},
    {"name": "egauge_17", "model": "EG4015", "registers": [
        {"name": "L1", "unit": "V", "value": "L1"}, {"name": "L2", "unit": "V", "value": "L2"},
        reg("V_ab", "bus_1041.ab", "V", "egauge_17"), reg("V_bc", "bus_1041.bc", "V", "egauge_17"), reg("V_ca", "bus_1041.ca", "V", "egauge_17"),
        reg("CT1", "line_382.a", "A", "egauge_17"), reg("CT2", "line_382.b", "A", "egauge_17"), reg("CT3", "line_382.c", "A", "egauge_17")]},
    {"name": "egauge_16", "model": "EG4015", "registers": [
        {"name": "Mains_Power", "element": "line_402", "unit": "W", "value": "S1*L1+S2*L2+S3*L3", "data_file": "egauge_16-Mains_Power"},
        reg("L1", "bus_1161.ag", "V", "egauge_16"), reg("L2", "bus_1161.bg", "V", "egauge_16"), reg("L3", "bus_1161.cg", "V", "egauge_16"),
        reg("CT1", "line_402.a", "A", "egauge_16"), reg("CT2", "line_402.b", "A", "egauge_16"), reg("CT3", "line_402.c", "A", "egauge_16"),
        reg("SouthPV_A", "line_403.a", "A", "egauge_16"), reg("SouthPV_B", "line_403.b", "A", "egauge_16"), reg("SouthPV_C", "line_403.c", "A", "egauge_16")]},
]}


def write(mag, name, values, t0=T0, drop=()):
    t = pd.date_range(t0, periods=len(values), freq="1s")
    df = pd.DataFrame({"t": t.strftime("%Y-%m-%d %H:%M:%S"), "v": values})
    df = df.drop(index=list(drop))
    df.to_csv(mag / f"{name}.csv", index=False)


@pytest.fixture
def site(tmp_path):
    (tmp_path / "net.json").write_text(json.dumps(NETWORK))
    mag = tmp_path / "magnitudes"; mag.mkdir()
    rng = np.random.default_rng(0)
    cur = {k: 100.0 * k + rng.normal(0, 1, N) for k in (1, 2, 3)}       # CT1 ~100 A, CT2 ~200 A, CT3 ~300 A
    for m in ("egauge_11", "egauge_13", "egauge_17"):
        for k in (1, 2, 3):
            write(mag, f"{m}-CT{k}", cur[k])
    for m in ("egauge_11", "egauge_13"):
        for k in (1, 2, 3):
            write(mag, f"{m}-L{k}", np.full(N, 270.0 + k))                # L1 271 V, L2 272 V, L3 273 V
    write(mag, "egauge_11-Mains_Power", 0.85 * sum(cur[k] * (270.0 + k) for k in (1, 2, 3)) / 1e3)   # kW
    write(mag, "egauge_17-V_ab", np.full(N, 2400.0))
    write(mag, "egauge_17-V_bc", np.full(N, -2380.0))
    write(mag, "egauge_17-V_ca", np.full(N, 7.0))
    return tmp_path, mag, cur


def test_three_per_phase_series_on_a_common_one_second_base(site):
    root, mag, cur = site
    m = caltech.meter_map(root / "net.json")["egauge_11"]
    d = caltech.load_element(mag, m, T0, "2025-03-12T00:10:00")
    assert d.i.shape == d.v_ln.shape == d.kva.shape == (3, N)
    assert str(d.t0) == "2025-03-12T00:00:00" and d.mask.all()
    for p, k in enumerate((1, 2, 3)):
        assert np.allclose(d.i[p], cur[k], atol=1e-3)
        assert np.allclose(d.kva[p], cur[k] * (270.0 + k) / 1e3, atol=1e-3)


def test_gap_in_one_phase_is_reported_and_masked_in_all_three_never_filled(site):
    root, mag, cur = site
    write(mag, "egauge_11-CT2", cur[2], drop=range(100, 120))
    m = caltech.meter_map(root / "net.json")["egauge_11"]
    d = caltech.load_element(mag, m, T0, "2025-03-12T00:10:00")
    assert not d.mask[100:120].any() and d.mask[:100].all() and d.mask[120:].all()
    assert np.isnan(d.i[:, 100:120]).all() and np.isnan(d.kva[:, 100:120]).all() and np.isnan(d.total[100:120]).all()
    assert d.report["missing_seconds"]["egauge_11-CT2"] == 20 and d.report["missing_seconds"]["egauge_11-CT1"] == 0
    assert d.report["masked_seconds"] == 20
    assert d.report["gaps"] == [("2025-03-12T00:01:40", 20)]
    assert np.isclose(d.i[1, 120], cur[2][120], atol=1e-3)       # rows after the gap keep their own second
    assert np.isclose(d.i[0, 99], cur[1][99], atol=1e-3)


def test_total_is_the_sum_of_phases_and_the_meters_own_power_gives_the_power_factor(site):
    """The target was 'phases sum to the element total within 0.5%'. The meter's power
    register is real power in kW, so against current x voltage it gives the power factor,
    not 1. The loader's total is the phase sum; the register is returned beside it."""
    root, mag, cur = site
    m = caltech.meter_map(root / "net.json")["egauge_11"]
    d = caltech.load_element(mag, m, T0, "2025-03-12T00:10:00")
    assert np.allclose(d.total, d.kva.sum(axis=0))
    assert d.p_kw is not None and abs(np.nanmedian(d.p_kw / d.total) - 0.85) < 0.005


def test_channels_map_to_phases_by_the_metadata_not_by_channel_number(site):
    root, mag, cur = site
    m = caltech.meter_map(root / "net.json")["egauge_13"]
    assert m.current == {"a": "egauge_13-CT3", "b": "egauge_13-CT2", "c": "egauge_13-CT1"}
    d = caltech.load_element(mag, m, T0, "2025-03-12T00:10:00")
    assert np.allclose(d.i[0], cur[3], atol=1e-3) and np.allclose(d.i[2], cur[1], atol=1e-3)   # phase a is channel 3
    assert np.allclose(d.v_ln[0], 273.0) and np.allclose(d.v_ln[2], 271.0)


def test_line_to_line_meter_uses_the_two_real_channels_and_ignores_the_derived_one(site):
    root, mag, cur = site
    m = caltech.meter_map(root / "net.json")["egauge_17"]
    assert m.v_kind == "ll"
    d = caltech.load_element(mag, m, T0, "2025-03-12T00:10:00")
    expect = (2400.0 + 2380.0) / 2 / np.sqrt(3)
    assert np.allclose(d.v_ln, expect) and (d.v_ln > 0).all()
    assert d.p_kw is None                                         # this meter has no power register
    (mag / "egauge_17-V_ca.csv").write_text("")                   # the derived channel is never needed
    d2 = caltech.load_element(mag, m, T0, "2025-03-12T00:10:00")
    assert d2.mask.all()


def test_empty_file_is_reported_and_masks_everything_without_raising(site):
    root, mag, cur = site
    (mag / "egauge_11-CT3.csv").write_text("")                   # what the API returns when it has no data
    m = caltech.meter_map(root / "net.json")["egauge_11"]
    d = caltech.load_element(mag, m, T0, "2025-03-12T00:10:00")
    assert not d.mask.any() and d.report["empty_files"] == ["egauge_11-CT3"]
    assert d.report["missing_seconds"]["egauge_11-CT3"] == N


def test_duplicate_timestamps_keep_the_first_and_are_counted(site):
    root, mag, cur = site
    f = mag / "egauge_11-CT1.csv"
    lines = f.read_text().splitlines()
    f.write_text("\n".join(lines[:6] + ["2025-03-12 00:00:04,999.0"] + lines[6:]) + "\n")
    m = caltech.meter_map(root / "net.json")["egauge_11"]
    d = caltech.load_element(mag, m, T0, "2025-03-12T00:10:00")
    assert d.report["duplicate_rows"]["egauge_11-CT1"] == 1 and np.isclose(d.i[0, 4], cur[1][4], atol=1e-3)


def test_sub_circuits_on_one_meter_get_their_own_entries(site):
    root, mag, cur = site
    mm = caltech.meter_map(root / "net.json")
    assert mm["egauge_16"].element == "line_402" and mm["egauge_16"].current["a"] == "egauge_16-CT1"
    assert mm["egauge_16/line_403"].current == {"a": "egauge_16-SouthPV_A", "b": "egauge_16-SouthPV_B", "c": "egauge_16-SouthPV_C"}
    assert mm["egauge_16/line_403"].voltage == mm["egauge_16"].voltage


def test_local_hour_labels_use_utc_timestamps(site):
    """Timestamps are UTC (solar output on 2025-03-12 starts ~14:00 and ends ~02:00)."""
    lab = caltech.hour_labels("2025-03-12T00:00:00", 24)
    assert lab[0] == "work"        # Tue 17:00 PDT
    assert lab[2] == "weeknight"   # Tue 19:00 PDT
    assert lab[14] == "work"       # Wed 07:00 PDT
