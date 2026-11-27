"""Tests for scripts/posthoc_spill_split.py."""

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
