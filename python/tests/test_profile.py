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
    assert profile["measured"]["memory"]["dram_bw_read_gbps"]["source"] == "measured"


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
