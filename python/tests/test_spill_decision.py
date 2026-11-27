"""The spill decision script: its lookup copy matches the core and its rule reads as registered."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("spill_decision", ROOT / "scripts/spill_decision.py")
sd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sd)


def load(name):
    return json.loads((ROOT / f"profiles/{name}.json").read_text())


@pytest.mark.parametrize(("name", "mape"), [("rtx4070", 0.126), ("gb10", 0.131)])
def test_boundary_lookup_reproduces_the_core_basis(name, mape):
    # design.md 6.4 and the core test spill_reload_latency_at_the_l1_boundary_matches_the_microbenchmark
    rows = sd.boundary(load(name))
    assert len(rows) == 4
    assert sd.boundary_test(rows)["mape"] == pytest.approx(mape, abs=0.0005)


def test_leave_one_out_matches_design_md_on_gb10():
    rows = sd.leave_one_out(load("gb10"))["l1_hit"]
    assert max(r["error"] for r in rows) == pytest.approx(0.186, abs=0.0005)


def test_whole_core_rows_are_the_boundary_points():
    rows = sd.whole_core(load("rtx4070"), (ROOT / "profiles/rtx4070.json").read_text())
    assert [r["point"] for r in rows] == [r["point"] for r in sd.boundary(load("rtx4070"))]


def test_rule_thresholds():
    rows = [{"error": e} for e in (0.05, -0.08, 0.09, 0.2)]
    assert sd.count_test(rows)["verdict"] == "pass"
    assert sd.count_test(rows + [{"error": 0.3}])["verdict"] == "fail"
    assert sd.count_test([{"error": 0.11}, {"error": -0.12}])["verdict"] == "fail"
    assert sd.boundary_test([{"error": e} for e in (0.1, 0.2, -0.2, 0.3)])["verdict"] == "fail"
    assert sd.boundary_test([{"error": e} for e in (0.1, 0.2, -0.2, 0.15)])["verdict"] == "pass"
    assert sd.boundary_test([{"error": 0.1}])["verdict"] == "not decided"


def test_short_launch_profiles_are_not_scored(tmp_path):
    assert sd.eligible(load("gb10"), None) == "not a long-launch profile"
    profile = {"provenance": {"capture_protocol": {"rule_three": {"spill-latency": {}}}}}
    assert "capture rules" in sd.eligible(profile, None)
    profile["provenance"]["capture_protocol"]["rule_three"]["spill-latency"]["status"] = "passed"
    raw = tmp_path / "merged-spill-latency.json"
    raw.write_text(json.dumps({"reload_latency_by_count": []}))
    assert "launch cost" in sd.eligible(profile, raw)
    raw.write_text(json.dumps({"launch_cost_by_frame": [{"frame_bytes_per_thread": 128}]}))
    assert sd.eligible(profile, raw) is None


def test_verdict_needs_both_reference_gpus():
    passed = {
        "count_test": {"l1_hit": {"verdict": "pass"}, "l2_spill": {"verdict": "pass"}},
        "boundary_test": {"verdict": "pass"},
    }
    assert sd.verdict([{"name": "gb10", **passed}])[0] == "not decided"
    assert sd.verdict([{"name": "gb10", **passed}, {"name": "rtx4070", **passed}])[0] == "pass"
    failed = {**passed, "boundary_test": {"verdict": "fail"}}
    assert sd.verdict([{"name": "gb10", **passed}, {"name": "rtx4070", **failed}])[0] == "fail"
    skipped = {"name": "rtx4070", "not_run": "spill-latency did not pass the capture rules"}
    assert sd.verdict([{"name": "gb10", **passed}, skipped])[0] == "not decided"
