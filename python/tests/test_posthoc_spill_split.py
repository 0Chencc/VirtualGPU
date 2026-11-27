"""Tests for scripts/posthoc_spill_split.py."""

import json

from scripts.posthoc_spill_split import compare


def _loop(visits, loads, threads):
    return {
        "cta_iterations": visits,
        "thread_local_load": loads,
        "per_thread_iteration": {"load": loads / (visits * threads)},
    }


def test_compare_splits_static_reloads_and_pools_measured_loops():
    row = {
        "config_id": "m64n32-w4-s2",
        "loops": {"main": _loop(30, 30 * 128 * 10, 128), "diagonal": _loop(10, 10 * 128 * 6, 128)},
    }
    resources = {
        "spill_loads_bytes": 80,
        "spill_scope": "inner_loop_assumed",
        "spill_loop_count": 2,
    }
    r = compare(row, resources, num_warps=4)
    assert r["static_unsplit"] == 20
    assert r["static_split"] == 10
    assert r["measured_by_loop"] == {"main": 10, "diagonal": 6}
    assert r["measured_all_loops"] == 9


def test_compare_defaults_to_one_loop():
    row = {"config_id": "x", "loops": {"main": _loop(4, 4 * 64 * 2, 64)}}
    r = compare(row, {"spill_loads_bytes": 8, "spill_scope": "inner_loop_confirmed"}, num_warps=2)
    assert r["spill_loop_count"] == 1
    assert r["static_split"] == r["static_unsplit"] == 2


def test_main_compiles_for_the_captured_target(tmp_path, monkeypatch):
    from scripts import posthoc_spill_split as split

    row = {"config_id": "m64n32-w4-s2", "registers_per_thread": 1, "loops": {}}
    capture = {
        "provenance": {"commit": "abc", "target": "sm_86"},
        "causal_attention": {"case": {}, "configs": [row]},
    }
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(capture))
    targets = []

    def resources(case, config, target):
        targets.append(target)
        return {"registers_per_thread": 1, "spill_loads_bytes": 0, "spill_scope": "none"}

    monkeypatch.setattr(split, "frontend_resources", resources)
    monkeypatch.setattr(split.sys, "argv", ["posthoc_spill_split", str(path)])
    assert split.main() == 0
    assert targets == ["sm_86"]
