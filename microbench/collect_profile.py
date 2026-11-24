"""Run M1 CUDA microbenchmarks and assemble a validated VirtualGPU profile JSON."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from virtualgpu.profile import validate_profile  # noqa: E402

DEFAULT_BIN = Path(__file__).resolve().parent / "build" / "vgpu_bench"
SUITE_VERSION = "0.1.0"
PROFILE_REVISION = "0.1.0-m1"


def find_nvidia_smi() -> str | None:
    for candidate in (shutil.which("nvidia-smi"), "/usr/lib/wsl/lib/nvidia-smi"):
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def sample_nvidia_smi(device: int) -> dict[str, Any]:
    smi = find_nvidia_smi()
    if smi is None:
        return {}
    fields = [
        "driver_version",
        "clocks.current.sm",
        "clocks.current.memory",
        "power.limit",
        "clocks_event_reasons.hw_thermal_slowdown",
        "clocks_event_reasons.sw_thermal_slowdown",
    ]
    cmd = [
        smi,
        f"--id={device}",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not proc.stdout.strip():
        return {}
    parts = [p.strip() for p in proc.stdout.strip().splitlines()[0].split(",")]
    if len(parts) < 4:
        return {}

    def parse_float(s: str) -> float | None:
        try:
            return float(s)
        except ValueError:
            return None

    throttled = any(p.lower() == "active" for p in parts[4:])
    return {
        "driver_version": parts[0],
        "sm_clock_mhz": parse_float(parts[1]),
        "mem_clock_mhz": parse_float(parts[2]),
        "power_limit_w": parse_float(parts[3]),
        "throttled": throttled,
    }


def run_bench_with_clock_polling(
    binary: Path,
    mode: str,
    device: int,
    extra_args: list[str] | None = None,
    poll_interval_s: float = 0.15,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run vgpu_bench while polling nvidia-smi under active load (avoids idle P8 clocks)."""
    import time

    cmd = [str(binary), "--mode", mode, "--device", str(device)]
    if extra_args:
        cmd.extend(extra_args)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    samples: list[dict[str, Any]] = []
    # Wait briefly for kernel warmup before polling clocks.
    time.sleep(poll_interval_s)
    while proc.poll() is None:
        snap = sample_nvidia_smi(device)
        if snap:
            samples.append(snap)
        time.sleep(poll_interval_s)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=stdout, stderr=stderr)
    if not samples:
        snap = sample_nvidia_smi(device)
        if snap:
            samples.append(snap)
    return json.loads(stdout), samples


def bootstrap_median_ci(samples: list[float], n_boot: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    med = float(statistics.median(samples))
    if len(samples) < 2:
        return med, med, med
    rng = random.Random(seed)
    n = len(samples)
    boots = [statistics.median(rng.choices(samples, k=n)) for _ in range(n_boot)]
    boots.sort()
    lo = float(boots[int(0.025 * n_boot)])
    hi = float(boots[int(0.975 * n_boot)])
    return med, min(lo, med), max(hi, med)


def unlocked_rel_uncertainty(
    med: float,
    ci_lo: float,
    ci_hi: float,
    clock_range: list[float] | None,
    locked: bool,
    throttle_events: int,
) -> float:
    half_ci_rel = (ci_hi - ci_lo) / (2.0 * med) if med > 0 else 0.10
    unc = max(0.10, half_ci_rel)
    if not locked:
        if clock_range is not None and clock_range[0] > 0 and clock_range[1] >= clock_range[0]:
            mid = 0.5 * (clock_range[0] + clock_range[1])
            drift_rel = (clock_range[1] - clock_range[0]) / (2.0 * mid)
            unc = max(unc, drift_rel + 0.05)
        else:
            unc = max(unc, 0.15)
    if throttle_events > 0:
        unc += 0.05
    return round(min(unc, 1.0), 4)


def make_measured(
    samples: list[float],
    unit: str,
    method: str,
    perturb_group: str | None,
    clock_range: list[float] | None,
    locked: bool,
    throttle_events: int,
) -> dict[str, Any]:
    med, lo, hi = bootstrap_median_ci(samples)
    rel_unc = unlocked_rel_uncertainty(med, lo, hi, clock_range, locked, throttle_events)
    return {
        "value": round(med, 4),
        "ci95_low": round(lo, 4),
        "ci95_high": round(hi, 4),
        "rel_uncertainty": rel_unc,
        "unit": unit,
        "method": method,
        "source": "measured",
        "perturb_group": perturb_group,
    }


def make_estimated(
    value: float,
    unit: str,
    method: str,
    perturb_group: str | None,
    rel_uncertainty: float = 0.30,
) -> dict[str, Any]:
    return {
        "value": round(value, 4),
        "ci95_low": None,
        "ci95_high": None,
        "rel_uncertainty": rel_uncertainty,
        "unit": unit,
        "method": method,
        "source": "estimated",
        "perturb_group": perturb_group,
    }


def classify_dispatch(dispatch_raw: dict[str, Any]) -> dict[str, Any]:
    sm_count = dispatch_raw["sm_count"]
    probes = dispatch_raw["probes"]
    if not probes:
        return {
            "observed_policy": "unknown",
            "candidate_policies": ["round_robin", "grouped", "unknown"],
            "policy_params": {},
            "confidence": "low",
            "probes": [],
        }

    consistencies = [p["order_consistency"] for p in probes]
    avg_consistency = sum(consistencies) / len(consistencies)

    # Check whether CTA i maps to (i % sm_count) or contiguous groups on 1-CTA/SM first wave.
    rr_matches = 0
    rr_total = 0
    for p in probes:
        trace = p["first_wave_trace"]
        if p["ctas_per_sm"] == 1 and len(trace) >= sm_count:
            for rec in trace[:sm_count]:
                rr_total += 1
                if rec["sm_id"] == (rec["cta_id"] % sm_count):
                    rr_matches += 1
    rr_ratio = (rr_matches / rr_total) if rr_total > 0 else 0.0

    if rr_ratio >= 0.90 and avg_consistency >= 0.85:
        policy = "round_robin"
        candidates = ["round_robin"]
        confidence = "high"
    elif avg_consistency >= 0.85:
        policy = "measured_table"
        candidates = ["measured_table", "round_robin"]
        confidence = "medium"
    else:
        policy = "unknown"
        candidates = ["round_robin", "grouped", "unknown"]
        confidence = "low"

    return {
        "observed_policy": policy,
        "candidate_policies": candidates,
        "policy_params": {"rr_exact_match_ratio": round(rr_ratio, 4)},
        "confidence": confidence,
        "probes": probes,
    }


def is_wsl() -> bool:
    rel = platform.release().lower()
    return "microsoft" in rel or "wsl" in rel


def arch_carveout_options(major: int, minor: int, smem_per_sm: int) -> list[int]:
    if major == 8 and minor == 9:
        return [0, 8192, 16384, 32768, 65536, 102400]
    if major == 12:
        return [0, 8192, 16384, 32768, 65536, 102400]
    return sorted({0, smem_per_sm // 4, smem_per_sm // 2, smem_per_sm})


def build_profile(
    query: dict[str, Any],
    dram_raw: dict[str, Any],
    dispatch_raw: dict[str, Any],
    smi_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    major = int(query["major"])
    minor = int(query["minor"])
    arch = f"sm_{major}{minor}"
    sm_count = int(query["sm_count"])
    max_warps_per_sm = int(query["max_warps_per_sm"])
    l2_size = int(query["l2_size_bytes"])
    smem_per_sm = int(query["shared_mem_per_sm_bytes"])
    max_smem_block = int(query["max_shared_mem_per_block_optin_bytes"] or query["shared_mem_per_block_bytes"])
    reserved_smem = int(query["reserved_shared_mem_per_cta_bytes"])

    # Aggregate active-load clocks from nvidia-smi snapshots (exclude idle P8 transitions < 50% peak).
    raw_sm_clocks = [s["sm_clock_mhz"] for s in smi_samples if s.get("sm_clock_mhz")]
    raw_mem_clocks = [s["mem_clock_mhz"] for s in smi_samples if s.get("mem_clock_mhz")]
    sm_clocks = [c for c in raw_sm_clocks if c >= 0.5 * max(raw_sm_clocks)] if raw_sm_clocks else []
    mem_clocks = [c for c in raw_mem_clocks if c >= 0.5 * max(raw_mem_clocks)] if raw_mem_clocks else []
    sm_range = [min(sm_clocks), max(sm_clocks)] if sm_clocks else None
    mem_range = [min(mem_clocks), max(mem_clocks)] if mem_clocks else None
    throttle_events = sum(1 for s in smi_samples if s.get("throttled"))

    driver_str = next((s["driver_version"] for s in smi_samples if s.get("driver_version")), str(query["driver_version"]))
    power_limit = next((s["power_limit_w"] for s in smi_samples if s.get("power_limit_w")), None)
    runtime_ver = int(query["runtime_version"])
    toolkit_str = f"{runtime_ver // 1000}.{(runtime_ver % 1000) // 10}"

    # Build measured DRAM read curve (keyed by target_sms; use observed_sms if strictly increasing).
    read_curve_points: list[dict[str, Any]] = []
    seen_sms: set[int] = set()
    for pt in dram_raw["read_curve"]:
        active = int(pt["target_sms"])
        if active in seen_sms:
            continue
        seen_sms.add(active)
        mv = make_measured(
            pt["gbps_samples"],
            unit="GB/s",
            method="cuda_stream_read_cold_l2",
            perturb_group="dram_bw",
            clock_range=mem_range,
            locked=False,
            throttle_events=throttle_events,
        )
        read_curve_points.append({"active_sms": active, "bw_gbps": mv})
    read_curve_points.sort(key=lambda x: x["active_sms"])

    full_read_mv = read_curve_points[-1]["bw_gbps"]
    full_write_samples = dram_raw["write_full"][-1]["gbps_samples"]
    full_write_mv = make_measured(
        full_write_samples,
        unit="GB/s",
        method="cuda_stream_write_cold_l2",
        perturb_group="dram_bw",
        clock_range=mem_range,
        locked=False,
        throttle_events=throttle_events,
    )

    # Estimated placeholders for M1 unmeasured fields (marked source="estimated", rel_uncertainty=0.30).
    sm_clock_ghz = (
        (statistics.median(sm_clocks) / 1000.0)
        if sm_clocks
        else (query["clock_khz"] / 1e6 if query["clock_khz"] > 0 else 2.4)
    )
    est_l2_peak = max(full_read_mv["value"] * 4.5, 1000.0)
    est_mma_1w = 0.25 * sm_clock_ghz  # conservative placeholder per SM for 1 warp
    est_mma_4w = 0.95 * sm_clock_ghz  # conservative placeholder per SM for 4 warps
    est_mma_8w = 1.00 * sm_clock_ghz

    l1_unified = 131072 if (major == 8 and minor == 9) or major == 12 else smem_per_sm
    tma_supported = major >= 9

    now_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_id = f"cal-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{arch}"

    profile: dict[str, Any] = {
        "schema_version": "0.1",
        "identity": {
            "vendor": "nvidia",
            "arch": arch,
            "sku": query["name"],
            "memory_type": "unified" if query["integrated"] else "discrete",
        },
        "provenance": {
            "profile_revision": PROFILE_REVISION,
            "calibration_run_id": run_id,
            "calibration_manifest_digest": None,
            "source": "partial",
            "driver_version": driver_str,
            "toolkit_version": toolkit_str,
            "suite_version": SUITE_VERSION,
            "host_os": platform.system().lower(),
            "is_wsl": is_wsl(),
            "power_limit_w": power_limit,
            "timestamp_utc": now_utc,
            "clock_policy": {
                "locked": False,
                "sm_clock_locked": False,
                "mem_clock_locked": False,
                "sm_clock_mhz": round(statistics.median(sm_clocks), 1) if sm_clocks else (round(query["clock_khz"] / 1000.0, 1) if query["clock_khz"] > 0 else None),
                "mem_clock_mhz": round(statistics.median(mem_clocks), 1) if mem_clocks else (round(query["mem_clock_khz"] / 1000.0, 1) if query["mem_clock_khz"] > 0 else None),
                "observed_sm_clock_range_mhz": sm_range,
                "observed_mem_clock_range_mhz": mem_range,
                "throttle_events": throttle_events,
            },
        },
        "specs": {
            "sm_count": sm_count,
            "warp_size": int(query["warp_size"]),
            "max_threads_per_sm": int(query["max_threads_per_sm"]),
            "max_warps_per_sm": max_warps_per_sm,
            "max_blocks_per_sm": int(query["max_blocks_per_sm"]),
            "max_threads_per_block": int(query["max_threads_per_block"]),
            "registers_per_sm": int(query["registers_per_sm"]),
            "max_registers_per_thread": 255,
            "register_alloc_granularity": "warp",
            "register_alloc_unit_regs": 256,
            "l1_smem_unified_bytes": l1_unified,
            "l1_size_bytes": l1_unified,
            "shared_mem_per_sm_bytes": smem_per_sm,
            "max_shared_mem_per_block_bytes": max_smem_block,
            "reserved_shared_mem_per_cta_bytes": reserved_smem,
            "shared_mem_alloc_granularity_bytes": 128,
            "shared_mem_carveout_options_bytes": arch_carveout_options(major, minor, smem_per_sm),
            "l2_size_bytes": l2_size,
            "l2_line_bytes": 128,
            "l2_sector_bytes": 32,
            "l2_partitions": 1,
            "l1_line_bytes": 128,
            "capabilities": {
                "cp_async": major >= 8,
                "tma": tma_supported,
                "wgmma": major == 9,
                "tcgen05_tmem": major == 10,
                "cluster": major in (9, 10),
            },
            "mma_shapes": [
                {"op_class": "mma.sync", "a_dtype": "f16", "b_dtype": "f16", "acc_dtype": "f32", "m": 16, "n": 8, "k": 16},
                {"op_class": "mma.sync", "a_dtype": "bf16", "b_dtype": "bf16", "acc_dtype": "f32", "m": 16, "n": 8, "k": 16},
            ],
        },
        "measured": {
            "memory": {
                "dram_bw_read_gbps": full_read_mv,
                "dram_bw_write_gbps": full_write_mv,
                "dram_bw_by_active_sms": read_curve_points,
                "l2_bw_read_gbps": make_estimated(est_l2_peak, "GB/s", "m1_estimated", "l2_bw"),
                "l2_bw_write_gbps": make_estimated(est_l2_peak * 0.85, "GB/s", "m1_estimated", "l2_bw"),
                "l2_bw_by_active_sms": [
                    {"active_sms": 1, "bw_gbps": make_estimated(est_l2_peak / max(sm_count / 3.0, 1.0), "GB/s", "m1_estimated", "l2_bw")},
                    {"active_sms": sm_count, "bw_gbps": make_estimated(est_l2_peak, "GB/s", "m1_estimated", "l2_bw")},
                ],
                "l2_effective_capacity_bytes": make_estimated(l2_size * 0.85, "bytes", "m1_estimated", None),
                "smem_bw_per_sm_gbps": make_estimated(128.0 * sm_clock_ghz, "GB/s", "m1_estimated", None),
                "dram_latency_ns": make_estimated(350.0, "ns", "m1_estimated", None),
                "dram_latency_loaded_ns": None,
                "l2_latency_ns": make_estimated(120.0, "ns", "m1_estimated", None),
                "l2_latency_loaded_ns": None,
                "smem_latency_ns": make_estimated(15.0, "ns", "m1_estimated", None),
            },
            "compute": {
                "mma_tflops_per_sm_by_warps": {
                    "mma.sync:f16:f16:f32:16x8x16": [
                        {"mma_warps_per_sm": 1, "tflops_per_sm": make_estimated(est_mma_1w, "TFLOP/s", "m1_estimated", "mma_dense")},
                        {"mma_warps_per_sm": 4, "tflops_per_sm": make_estimated(est_mma_4w, "TFLOP/s", "m1_estimated", "mma_dense")},
                        {"mma_warps_per_sm": 8, "tflops_per_sm": make_estimated(est_mma_8w, "TFLOP/s", "m1_estimated", "mma_dense")},
                    ]
                },
                "non_mma_issue_rate_per_sm_gips": make_estimated(4.0 * sm_clock_ghz, "Ginst/s", "m1_estimated", None),
                "fp32_fma_tflops": make_estimated(sm_count * 128.0 * 2.0 * sm_clock_ghz / 1000.0, "TFLOP/s", "m1_estimated", None),
                "sfu_tflops": make_estimated(sm_count * 16.0 * sm_clock_ghz / 1000.0, "TFLOP/s", "m1_estimated", None),
            },
            "occupancy": {
                "smem_carveout_observed": [],
            },
            "async_pipeline": {
                "cp_async_bw_per_sm_gbps": make_estimated(100.0, "GB/s", "m1_estimated", "async_cp"),
                "tma_bw_per_sm_gbps": make_estimated(150.0, "GB/s", "m1_estimated", "async_tma") if tma_supported else None,
                "tma_setup_ns": make_estimated(80.0, "ns", "m1_estimated", None) if tma_supported else None,
            },
            "spill": {
                "local_mem_l1_hit_bw_per_sm_gbps": make_estimated(150.0, "GB/s", "m1_estimated", "spill_l1"),
                "local_mem_l1_miss_bw_per_sm_gbps": make_estimated(40.0, "GB/s", "m1_estimated", "spill_l2"),
                "local_mem_l1_hit_latency_ns": make_estimated(25.0, "ns", "m1_estimated", "spill_l1"),
                "local_mem_l1_miss_latency_ns": make_estimated(160.0, "ns", "m1_estimated", "spill_l2"),
                "l1_to_l2_spill_threshold_bytes_per_sm": make_estimated(65536.0, "bytes", "m1_estimated", None),
            },
            "overhead": {
                "launch_overhead_us": make_estimated(5.0, "us", "m1_estimated", None),
                "per_cta_launch_latency_ns": make_estimated(15.0, "ns", "m1_estimated", None),
            },
            "cta_dispatch": classify_dispatch(dispatch_raw),
        },
    }

    validate_profile(profile)
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect M1 GPU profile via CUDA microbenchmarks.")
    parser.add_argument("--binary", type=Path, default=DEFAULT_BIN, help="Path to compiled vgpu_bench binary.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--samples", type=int, default=15, help="Timed iterations per DRAM BW point.")
    parser.add_argument("--repeats", type=int, default=5, help="Repeats per CTA dispatch probe.")
    parser.add_argument("--output", type=Path, required=True, help="Output profile JSON path.")
    args = parser.parse_args()

    query, _ = run_bench_with_clock_polling(args.binary, "query", args.device, poll_interval_s=0.05)
    dram_raw, smi_dram = run_bench_with_clock_polling(
        args.binary, "dram-bw", args.device, ["--samples", str(args.samples)]
    )
    dispatch_raw, smi_disp = run_bench_with_clock_polling(
        args.binary, "cta-dispatch", args.device, ["--repeats", str(args.repeats)]
    )
    smi_samples = smi_dram + smi_disp

    profile = build_profile(query, dram_raw, dispatch_raw, smi_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2)
        fh.write("\n")

    ident = profile["identity"]
    full_bw = profile["measured"]["memory"]["dram_bw_read_gbps"]
    disp = profile["measured"]["cta_dispatch"]
    print(
        f"Wrote {args.output}: {ident['sku']} ({ident['arch']}, {profile['specs']['sm_count']} SMs) | "
        f"DRAM read={full_bw['value']:.1f} GB/s (±{full_bw['rel_uncertainty']*100:.1f}%) | "
        f"dispatch={disp['observed_policy']} ({disp['confidence']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
