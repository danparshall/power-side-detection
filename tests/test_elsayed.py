"""Reader for the Elsayed et al. (Sci Data 2026) node traces: one CSV per 15-minute session,
one row per ~20 ms, a timestamp column, host columns, and per-GPU columns gpuN_<field>.
Only gpuN_power_W is power; gpuN_Power_TDP is a percentage and cpu_power_W is empty.
Readings are placed in fixed slots by floor of the time since the first row and averaged
within a slot; a slot with no reading is NaN. Written before the code.

Origin: verification@b28fe07c
"""
import numpy as np
import pytest

from island_detectability_map import elsayed

CSV = """timestamp,cpu_utilization_percent,cpu_power_W,gpu0_utilization_percent,gpu0_Power_TDP,gpu0_power_W,gpu0_temp_C,gpu1_Power_TDP,gpu1_power_W
2025-09-20 12:52:20.300000,1.6,,100,96.2,100.0,49,95.7,200.0
2025-09-20 12:52:20.700000,4.1,,100,96.2,110.0,49,95.7,210.0
2025-09-20 12:52:21.200000,4.1,,100,96.2,120.0,49,95.7,220.0
2025-09-20 12:52:22.900000,4.1,,100,96.2,130.0,49,95.7,230.0
"""


def write(tmp_path, text):
    p = tmp_path / "LLM_test.csv"
    p.write_text(text)
    return p


def test_read_node_returns_timestamps_and_only_the_power_columns(tmp_path):
    t, cols = elsayed.read_node(write(tmp_path, CSV))
    assert t.dtype == np.int64 and len(t) == 4
    assert (t[1] - t[0]) == 400_000_000                      # ns
    assert set(cols) == {"gpu0", "gpu1"}
    assert cols["gpu0"].tolist() == [100.0, 110.0, 120.0, 130.0]


def test_node_total_block_averages_the_gpu_sum_on_a_fixed_grid(tmp_path):
    tot, _, rep = elsayed.node_total(write(tmp_path, CSV), dt_s=1.0)
    # slots from the first row (12:52:20.3): [0,1) holds rows 0-2, [1,2) is empty, [2,3) holds row 3
    assert len(tot) == 3
    assert tot[0] == pytest.approx(((100 + 200) + (110 + 210) + (120 + 220)) / 3)
    assert np.isnan(tot[1])
    assert tot[2] == pytest.approx(130 + 230)
    assert rep["devices"] == 2 and rep["empty_slots"] == 1 and rep["rows"] == 4


def test_node_total_at_the_native_rate_keeps_every_reading(tmp_path):
    tot, _, _ = elsayed.node_total(write(tmp_path, CSV), dt_s=0.1)
    assert np.nansum(tot) == pytest.approx(300 + 320 + 340 + 360)
    assert np.isfinite(tot).sum() == 4
