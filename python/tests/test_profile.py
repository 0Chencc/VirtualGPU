"""Tests for GPU profile schema and validator."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from virtualgpu.profile import (
    DEFAULT_SCHEMA_PATH,
    ProfileValidationError,
    load_profile,
    validate_profile,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PROFILE = REPO_ROOT / "schemas" / "examples" / "profile_example.json"
RTX4070_PROFILE = REPO_ROOT / "profiles" / "rtx4070.json"


def test_example_profile_is_valid() -> None:
    profile = load_profile(EXAMPLE_PROFILE)
    assert profile["schema_version"] == "0.1"
    assert profile["identity"]["arch"] == "sm_89"
    assert DEFAULT_SCHEMA_PATH.is_file()


def test_rtx4070_profile_is_valid() -> None:
    if not RTX4070_PROFILE.is_file():
        pytest.skip("profiles/rtx4070.json not present")
    profile = load_profile(RTX4070_PROFILE)
    assert profile["identity"]["arch"] == "sm_89"
    assert profile["specs"]["sm_count"] == 46
    mem = profile["measured"]["memory"]
    assert mem["dram_bw_read_gbps"]["source"] == "measured"
    assert mem["l2_bw_read_gbps"]["source"] == "measured"
    assert mem["l2_effective_capacity_bytes"]["source"] == "measured"
    assert mem["l2_effective_capacity_bytes"]["value"] <= profile["specs"]["l2_size_bytes"]
    assert mem["dram_latency_loaded_ns"]["source"] == "measured"
    assert len(mem["dram_latency_by_bw"]) >= 5
    assert len(mem["l2_latency_by_bw"]) >= 5
    assert len(mem["dram_bw_by_bytes_in_flight"]) >= 40
    assert len(mem["l2_bw_by_bytes_in_flight"]) >= 40
    assert max(pt["bytes_in_flight"] for pt in mem["dram_bw_by_bytes_in_flight"]) >= 6 * 1024 * 1024
    assert max(pt["bytes_in_flight"] for pt in mem["l2_bw_by_bytes_in_flight"]) >= 6 * 1024 * 1024
    assert len(mem["dram_closed_loop"]) >= 100
    assert len(mem["l2_closed_loop"]) >= 100
    assert len(mem["l2_merge_by_sharing"]) >= 20
    assert len(mem["l2_rescan_hit_rate_by_working_set"]) >= 12
    assert len(profile["measured"]["pipeline"]["stage_sync_overhead_ns"]) >= 8
    assert profile["measured"]["async_pipeline"]["cp_async_bw_per_sm_gbps"]["source"] == "measured"
    assert profile["measured"]["overhead"]["launch_overhead_us"]["source"] == "measured"
    mma_f16 = profile["measured"]["compute"]["mma_tflops_per_sm_by_warps"]["mma.sync:f16:f16:f32:16x8x16"]
    assert len(mma_f16) >= 4
    assert all(pt["tflops_per_sm"]["source"] == "measured" for pt in mma_f16)


def test_rejects_missing_required_field() -> None:
    profile = load_profile(EXAMPLE_PROFILE)
    del profile["specs"]["sm_count"]
    with pytest.raises(ProfileValidationError, match="sm_count"):
        validate_profile(profile)


def test_rejects_unknown_field() -> None:
    profile = load_profile(EXAMPLE_PROFILE)
    profile["identity"]["unexpected_field"] = True
    with pytest.raises(ProfileValidationError, match="unexpected_field"):
        validate_profile(profile)


def test_rejects_invalid_ci_bounds() -> None:
    profile = copy.deepcopy(load_profile(EXAMPLE_PROFILE))
    mv = profile["measured"]["memory"]["dram_bw_read_gbps"]
    mv["ci95_low"] = mv["value"] + 10.0
    mv["ci95_high"] = mv["value"] + 20.0
    with pytest.raises(ProfileValidationError, match="outside"):
        validate_profile(profile)


def test_rejects_non_monotonic_active_sms_curve() -> None:
    profile = copy.deepcopy(load_profile(EXAMPLE_PROFILE))
    curve = profile["measured"]["memory"]["dram_bw_by_active_sms"]
    curve[1]["active_sms"] = curve[0]["active_sms"]
    with pytest.raises(ProfileValidationError, match="strictly increasing"):
        validate_profile(profile)


def test_rejects_active_sms_exceeding_sm_count() -> None:
    profile = copy.deepcopy(load_profile(EXAMPLE_PROFILE))
    curve = profile["measured"]["memory"]["dram_bw_by_active_sms"]
    curve[-1]["active_sms"] = profile["specs"]["sm_count"] + 1
    with pytest.raises(ProfileValidationError, match="exceeds hardware limit"):
        validate_profile(profile)


def test_rejects_dispatch_policy_not_in_candidates() -> None:
    profile = copy.deepcopy(load_profile(EXAMPLE_PROFILE))
    profile["measured"]["cta_dispatch"]["observed_policy"] = "grouped"
    profile["measured"]["cta_dispatch"]["candidate_policies"] = ["round_robin"]
    with pytest.raises(ProfileValidationError, match="candidate_policies"):
        validate_profile(profile)


def test_rejects_latency_curve_bw_exceeding_peak() -> None:
    profile = copy.deepcopy(load_profile(EXAMPLE_PROFILE))
    peak_mv = profile["measured"]["memory"]["dram_bw_read_gbps"]
    max_allowed = peak_mv["value"] * (1.0 + peak_mv["rel_uncertainty"])
    profile["measured"]["memory"]["dram_latency_by_bw"] = [
        {"achieved_bw_gbps": 0.0, "latency_ns": profile["measured"]["memory"]["dram_latency_ns"]},
        {"achieved_bw_gbps": max_allowed + 1.0, "latency_ns": profile["measured"]["memory"]["dram_latency_ns"]},
    ]
    with pytest.raises(ProfileValidationError, match="dram_bw_read_gbps"):
        validate_profile(profile)
