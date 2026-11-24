from pathlib import Path

import pytest

from virtualgpu import _core

ROOT = Path(__file__).resolve().parents[2]
IR = (ROOT / "schemas/examples/gemm_sm89.json").read_text()
PROFILE = (ROOT / "profiles/rtx4070.json").read_text()


def test_predict_example_gemm_on_rtx4070():
    r = _core.predict(IR, PROFILE)
    assert r["time_ns"] > r["makespan_ns"] > 0
    assert r["ctas_per_sm"] >= 1
    assert 0 < r["tail_efficiency"] <= 1


def test_predict_rejects_bad_l2_state():
    with pytest.raises(ValueError):
        _core.predict(IR, PROFILE, l2_state="lukewarm")
