"""CPU checks for the v0.3 candidate selection policy."""

from copy import deepcopy

import pytest

from validation.v03_candidates import TARGETS, create_manifests, read


@pytest.fixture
def inputs():
    return (
        {
            "excluded_gemm_axes": dict.fromkeys(("M", "N", "K"), []),
            "excluded_attention_seq_len": [],
        },
        {device: read(f"profiles/{device}.json")[0] for device in TARGETS},
    )


def test_candidate_census_and_tile_alignment(inputs):
    manifests = create_manifests(*inputs)
    for device in TARGETS:
        gemm = manifests["gemm"]["devices"][device]
        assert sum(len(case["config_ids"]) for case in gemm["cases"]) == 324
        capacity = gemm["wave_reference"]["planning_full_wave_ctas"]
        assert [case["reference_ctas"] for case in gemm["cases"] if "reference_ctas" in case] == [
            capacity - 1,
            capacity,
            capacity + 1,
        ]
        for case in gemm["cases"]:
            if "prefill" in case["case_id"]:
                m, n, k = case["shape"]
                assert m % 128 == n % 128 == k % 64 == 0
        attention = manifests["attention"]["devices"][device]
        assert sum(len(case["config_ids"]) for case in attention["cases"]) == 228
        assert {
            (case["causal"], case["query_heads"] == case["kv_heads"], case["head_dim"])
            for case in attention["cases"]
        } == {
            (causal, mha, dim)
            for causal in (False, True)
            for mha in (False, True)
            for dim in (64, 128)
        }
    lengths = [
        {case["seq_len"] for case in cohort["cases"]}
        for cohort in manifests["attention"]["devices"].values()
    ]
    assert lengths[0].isdisjoint(lengths[1])


@pytest.mark.parametrize("axis,index", [("M", 0), ("N", 1), ("K", 2)])
def test_rejects_reusing_one_gemm_axis(inputs, axis, index):
    inventory, profiles = deepcopy(inputs)
    candidate = create_manifests(inventory, profiles)["gemm"]["devices"]["rtx4070"]["cases"][0]
    inventory["excluded_gemm_axes"][axis] = [candidate["shape"][index]]
    with pytest.raises(ValueError, match="GEMM shape reuses"):
        create_manifests(inventory, profiles)


def test_rejects_reusing_attention_length(inputs):
    inventory, profiles = deepcopy(inputs)
    candidate = create_manifests(inventory, profiles)["attention"]["devices"]["gb10"]["cases"][0]
    inventory["excluded_attention_seq_len"] = [candidate["seq_len"]]
    with pytest.raises(ValueError, match="attention sequence length is reused"):
        create_manifests(inventory, profiles)
