"""run_task2 takes its signal from either an NLR run or an Elsayed et al. node session.
A workload named "elsayed-<job>" is the Elsayed session listed under that job; its trace is the
one run_elsayed_inject builds (8-GPU total, busy stretch, mean removed), at the requested rate.
Written before the code.

Origin: verification@5175e2fe
"""
from pathlib import Path

import numpy as np
import pytest

from island_detectability_map import run_elsayed_inject as EI, run_task2 as T2

DATA = Path(__file__).resolve().parents[2] / "data"
ROOT = DATA / "external" / "elsayed_2026" / "High-resolution-AI-Data-Center-Training-Workloads-Dataset_FigShare"
needs_data = pytest.mark.skipif(not ROOT.exists(), reason="Elsayed archive not unpacked on this machine")


@needs_data
@pytest.mark.parametrize("job", ["llama8b", "diffusion"])
def test_elsayed_workload_is_the_inject_trace_at_the_requested_rate(job):
    tr = T2.load_trace(f"elsayed-{job}", None, ROOT, 12.0)
    ref = EI.trace_kw(ROOT / T2.ELSAYED_JOBS[job], 12.0)
    np.testing.assert_array_equal(tr, ref)
    assert abs(tr.mean()) < 1e-9
    assert 0.9 * 900 * 12 < len(tr) <= 900 * 12          # a 15-minute session, busy stretch only


def test_unknown_elsayed_job_is_refused():
    with pytest.raises(KeyError):
        T2.load_trace("elsayed-nonesuch", None, ROOT, 12.0)
