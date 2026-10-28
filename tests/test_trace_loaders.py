"""The logged-job loaders can return the whole job (its mean draw included) as well as the swing alone.
Uses the real NLR and Elsayed logs; skipped where data/ is absent.

Origin: verification@80f0c35c
"""
from pathlib import Path

import numpy as np
import pytest

from island_detectability_map import run_elsayed_inject as EI, run_nlr_inject as NI, run_task2 as T2

DATA = Path(__file__).resolve().parents[2] / "data"
NLR = DATA / "nlr"
ELS = DATA / T2.ELSAYED_ROOT / T2.ELSAYED_JOBS["llama8b"]


@pytest.mark.skipif(not NLR.exists(), reason="NLR logs not present")
def test_nlr_whole_job_is_the_swing_plus_its_positive_mean_draw():
    swing = NI.trace_kw(NLR, "llama2_70b_lora", "10742829", 1.0)
    whole = NI.trace_kw(NLR, "llama2_70b_lora", "10742829", 1.0, remove_mean=False)
    assert abs(swing.mean()) < 1e-9
    offset = whole - swing
    assert np.allclose(offset, offset[0])
    assert 5.0 < offset[0] < 16 * 0.7               # 16 H100s, each under its 700 W TDP on average


@pytest.mark.skipif(not ELS.exists(), reason="Elsayed logs not present")
def test_elsayed_whole_job_is_the_swing_plus_its_positive_mean_draw():
    swing = EI.trace_kw(ELS, 1.0)
    whole = EI.trace_kw(ELS, 1.0, remove_mean=False)
    offset = whole - swing
    assert np.allclose(offset, offset[0])
    assert 2.0 < offset[0] < 8 * 0.7                # 8 H100s
