"""The registered non-MMA same-device check: its rule and its model side."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("non_mma_check", ROOT / "scripts/non_mma_check.py")
nm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nm)

SM = 48


def launch(steps, mhz=2500.0):
    return {"length": steps, "sm_mhz_before": [mhz] * 5, "sm_mhz_after": [mhz] * 5}


def campaign(field, ffma, mix_peak, method="cuda_ptx_ffma_lop3_mix", qualified=True):
    """A profile copy whose field is `field`, and merged raws whose FFMA rate is `ffma` and
    whose mix curve peaks at `mix_peak` (16 warps)."""
    profile = json.loads((ROOT / "profiles/gb10.json").read_text())
    value = profile["measured"]["compute"]["non_mma_issue_rate_per_sm_gips"]
    value.update(source="measured", value=field, method=method)
    value["per_cycle"] = {"value": field / 2500.0, "unit": "Ginst/s per MHz"}
    if qualified:
        profile["provenance"]["capture_protocol"] = {"minimum_launch_ns": 1e6}
    tflops = ffma * SM * 64.0 / 1e3
    mma = {
        "fp32_fma": {"warps_per_sm": 16, "launch": launch(52000)},
        "fp32_fma_tflops_samples": [tflops] * 5,
    }
    curve = [
        {"warps_per_sm": w, "rate_gwarp_inst_per_sm_samples": [r] * 5, "launch": launch(s)}
        for w, r, s in (
            (8, mix_peak * 0.98, 9886),
            (16, mix_peak, 8192),
            (32, mix_peak * 0.99, 4096),
        )
    ]
    sfu = {"alu_issue_rate_per_sm_gips": curve}
    return profile, json.dumps(profile), mma, sfu


def test_the_rule_is_two_sided_and_strict_at_ten_percent():
    assert not nm.missed(11.0, 10.0)  # exactly 10% above: not rejected
    assert nm.missed(11.0 + 1e-9, 10.0)
    assert not nm.missed(9.0, 10.0)  # exactly 10% below: not rejected
    assert nm.missed(9.0 - 1e-9, 10.0)
    assert nm.missed(10.0, 11.2)  # the field below a source's rate counts too


def test_reading_b_returns_the_field_and_reading_a_pays_the_step_sync():
    out = nm.check_gpu(*campaign(field=8.7, ffma=8.7, mix_peak=8.0))
    ffma, mix = out["judged"]
    for row in (ffma, mix):
        assert abs(row["predicted_b"] / 8.7 - 1) < 1e-9
    assert ffma["predicted_a"] < 0.5 * 8.7
    assert (mix["kernel"], mix["warps_per_sm"], mix["steps"]) == ("mix", 16, 8192)
    assert out["verdict"] == "pass"
    assert abs(out["ratio_mix_over_ffma"] - 8.0 / 8.7) < 1e-12


def test_a_slower_source_beyond_ten_percent_rejects():
    out = nm.check_gpu(*campaign(field=8.7, ffma=8.7, mix_peak=7.85))
    assert out["judged"][1]["rejects"] and out["verdict"] == "reject"
    assert not out["judged"][0]["rejects"]


def test_per_mhz_uses_each_rounds_probes_and_the_fields_clock():
    profile, text, mma, sfu = campaign(field=8.7, ffma=8.7, mix_peak=8.0)
    for q in sfu["alu_issue_rate_per_sm_gips"]:
        q["launch"]["sm_mhz_before"] = [2400.0] * 5
        q["launch"]["sm_mhz_after"] = [2600.0] * 5
    mix = nm.check_gpu(profile, text, mma, sfu)["judged"][1]
    assert abs(mix["measured_per_mhz"] - 8.0 / 2500.0) < 1e-12
    assert abs(mix["predicted_b_per_mhz"] - 8.7 / 2500.0) < 1e-9


def test_an_unqualified_profile_is_not_decided():
    assert nm.check_gpu(*campaign(8.7, 8.7, 8.0, qualified=False))["verdict"] == "not decided"
