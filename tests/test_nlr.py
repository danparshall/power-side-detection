"""Reader for the NLR generative-AI power profiles.
Written before the code.

The raw files are WattAMeter logs, one per node: a header line starting with '#', then one row
per reading with a time-stamp, the time the reading took, and one column per device. GPU power
is in mW, CPU power in W; there are also temperature and energy columns that are not power.
Nodes are logged separately, each with its own clock ticks, so an island's total needs the
nodes placed on one time grid. Nothing is filled: a slot with no reading is NaN, and the
island total is NaN wherever any node is.

Origin: verification@f68ea5ab
"""
import numpy as np
import pytest

from island_detectability_map import nlr

NVML = """# timestamp reading-time[ns] gpu-0[mW] gpu-1[mW] gpu-0[C] gpu-1[C]
  2025-09-20_06:51:56.800000 367568 70000 72000 40 41
  2025-09-20_06:51:57.000500 54052 170000 172000 40 41
  2025-09-20_06:51:57.200900 66942 270000 272000 40 41
  2025-09-20_06:51:57.600100 46450 470000 472000 40 41
"""
RAPL = """# timestamp                  reading-time[ns] cpu-0[uJ] cpu-0-core[uJ] cpu-1[uJ] cpu-0[W] cpu-0-core[W] cpu-1[W]
  2025-09-20_06:51:56.800000 89486 4308436614 21685053460 47609283400 69.5 0.04 67.25
  2025-09-20_06:51:57.000000 113823 4322422799 21685061791 47622737339 70.5 0.02 68.25
"""


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_gpu_power_is_read_in_watts_and_temperature_columns_are_dropped(tmp_path):
    t, cols = nlr.read_wattameter(write(tmp_path, "nvml_a.log", NVML))
    assert sorted(cols) == ["gpu-0", "gpu-1"]
    assert cols["gpu-0"].tolist() == [70.0, 170.0, 270.0, 470.0]
    assert cols["gpu-1"][0] == 72.0
    assert t.dtype == np.int64 and (t[1] - t[0]) == 200_500_000          # nanoseconds


def test_cpu_package_power_is_read_and_energy_and_core_columns_are_dropped(tmp_path):
    t, cols = nlr.read_wattameter(write(tmp_path, "rapl_a.log", RAPL))
    assert sorted(cols) == ["cpu-0", "cpu-1"]                             # package power only
    assert cols["cpu-0"].tolist() == [69.5, 70.5]


def test_readings_go_to_the_nearest_grid_slot_and_a_slot_with_no_reading_is_nan(tmp_path):
    t, cols = nlr.read_wattameter(write(tmp_path, "nvml_a.log", NVML))
    y, rep = nlr.to_grid(t, cols["gpu-0"], t0_ns=int(t[0]), dt_s=0.2, n=6)
    assert np.allclose(y[[0, 1, 2, 4]], [70.0, 170.0, 270.0, 470.0])
    assert np.isnan(y[3]) and np.isnan(y[5])                              # the skipped reading, and past the end
    assert rep == {"placed": 4, "empty_slots": 2, "collisions": 0, "outside": 0}


def test_two_readings_in_one_slot_are_counted_and_the_later_one_is_kept():
    t = np.array([0, 190_000_000, 210_000_000, 400_000_000], dtype=np.int64)
    y, rep = nlr.to_grid(t, np.array([1.0, 2.0, 3.0, 4.0]), t0_ns=0, dt_s=0.2, n=3)
    assert y.tolist() == [1.0, 3.0, 4.0]
    assert rep["collisions"] == 1


def test_island_total_sums_every_device_of_every_node_on_one_grid(tmp_path):
    a = write(tmp_path, "nvml_a.log", NVML)
    later = NVML.replace("06:51:56.800000", "06:51:57.000000").replace("06:51:57.000500", "06:51:57.200000") \
                .replace("06:51:57.200900", "06:51:57.400000").replace("06:51:57.600100", "06:51:57.600000")
    b = write(tmp_path, "nvml_b.log", later)
    tot, t0, rep = nlr.island_total([a, b], dt_s=0.2)
    # common span: from the later start (57.0) to the earlier end (57.6): slots at 57.0, 57.2, 57.4, 57.6
    assert t0 == nlr.read_wattameter(b)[0][0]
    assert len(tot) == 4
    assert tot[0] == pytest.approx(170 + 172 + 70 + 72)
    assert tot[1] == pytest.approx(270 + 272 + 170 + 172)
    assert np.isnan(tot[2])                                               # node a has no reading at 57.4
    assert tot[3] == pytest.approx(470 + 472 + 470 + 472)
    assert rep["devices"] == 4 and rep["nan_slots"] == 1


def test_gain_of_a_running_mean():
    # a reading that is the mean of the last second passes slow swings and nulls whole multiples of 1 Hz
    assert nlr.running_mean_gain(0.0, 1.0) == pytest.approx(1.0)
    assert nlr.running_mean_gain(0.1, 1.0) == pytest.approx(0.9836, abs=1e-3)
    assert nlr.running_mean_gain(0.5, 1.0) == pytest.approx(2 / np.pi, abs=1e-6)
    assert nlr.running_mean_gain(1.0, 1.0) == pytest.approx(0.0, abs=1e-12)
    assert nlr.running_mean_gain(2.0, 1.0) == pytest.approx(0.0, abs=1e-12)


def test_swing_is_the_gap_between_two_percentiles_and_ignores_missing_slots():
    x = np.r_[np.full(50, 10.0), np.full(50, 4.0), np.nan]
    s = nlr.swing(x, 5, 95)
    assert s == {"low": 4.0, "mid": 7.0, "high": 10.0, "swing": 6.0, "mean": 7.0}


def test_runs_below_a_level_give_dip_start_times_and_lengths():
    x = np.array([9, 9, 3, 3, 3, 9, 9, 2, 9, np.nan, 1, 1, 9], dtype=float)
    starts, lengths = nlr.runs_below(x, 5.0)
    assert starts.tolist() == [2, 7, 10] and lengths.tolist() == [3, 1, 2]
