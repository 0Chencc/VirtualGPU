"""Tests for the non-MMA issue rate in microbench/collect_profile.py."""

from __future__ import annotations

from microbench.collect_profile import non_mma_issue_rate, peak_rate_samples


def _curve(medians: dict[int, float]) -> list[dict]:
    return [
        {"warps_per_sm": w, "rate_gwarp_inst_per_sm_samples": [m * 0.99, m, m * 1.01]}
        for w, m in medians.items()
    ]


def _tflops(rate: float, sm_count: int = 46) -> list[float]:
    return [rate * sm_count * 64.0 / 1e3] * 3


def test_peak_rate_samples_takes_highest_median() -> None:
    samples = peak_rate_samples(_curve({4: 3.0, 8: 7.4, 16: 7.6, 32: 7.5}))
    assert samples == [7.6 * 0.99, 7.6, 7.6 * 1.01]


def test_non_mma_issue_rate_takes_mix_when_higher() -> None:
    raw = {"alu_issue_rate_per_sm_gips": _curve({8: 7.4, 16: 7.6})}
    mv = non_mma_issue_rate(raw, _tflops(7.0), 46, 1.9, [1890.0, 1905.0], 0)
    assert mv["source"] == "measured"
    assert mv["method"] == "cuda_ptx_ffma_lop3_mix"
    assert mv["unit"] == "Ginst/s"
    assert abs(mv["value"] - 7.6) < 1e-9


def test_non_mma_issue_rate_takes_pure_ffma_when_higher() -> None:
    raw = {"alu_issue_rate_per_sm_gips": _curve({8: 6.1, 16: 6.2})}
    mv = non_mma_issue_rate(raw, _tflops(7.17), 46, 1.9, [1890.0, 1905.0], 0)
    assert mv["method"] == "cuda_fp32_fma_unrolled"
    assert abs(mv["value"] - 7.17) < 1e-4


def test_non_mma_issue_rate_estimated_without_alu_curve() -> None:
    mv = non_mma_issue_rate({}, _tflops(7.17), 46, 1.9, None, 0)
    assert mv["source"] == "estimated"
    assert mv["value"] == 7.6


def _timed(point: dict, mhz: float) -> dict:
    return point | {"launch": {"sm_mhz_before": [mhz] * 3, "sm_mhz_after": [mhz] * 3}}


def test_non_mma_issue_rate_compares_per_mhz_with_clock_probes() -> None:
    # Pure FFMA is higher as measured, but its process ran 10% faster: per MHz the mix wins.
    curve = [_timed(p, 1770.0) for p in _curve({8: 6.9, 16: 7.0})]
    raw = {"alu_issue_rate_per_sm_gips": curve}
    ffma_point = _timed({"warps_per_sm": 16}, 1950.0)
    mv = non_mma_issue_rate(raw, _tflops(7.3), 46, 1.9, None, 0, ffma_point)
    assert mv["method"] == "cuda_ptx_ffma_lop3_mix"
    assert abs(mv["value"] - 7.0) < 1e-9
    no_probe = non_mma_issue_rate(raw, _tflops(7.3), 46, 1.9, None, 0, {"warps_per_sm": 16})
    assert no_probe["method"] == "cuda_fp32_fma_unrolled"
