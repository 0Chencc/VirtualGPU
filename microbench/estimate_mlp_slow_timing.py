"""Estimate archived MLP slow-segment timings from a separate flush control."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from statistics import median

from microbench import repeat_mlp as mlp

MS = 1e6


def load(path):
    return json.loads(path.read_bytes())


def controls(path):
    cold, active = [], []
    coverage = []
    for trial in load(path)["runs"]:
        if trial["condition"] != "original":
            continue
        origin = trial["result"]["origin_realtime_ns"] - trial["started_realtime_ns"]
        points = trial["result"]["points"]
        for point in points:
            target = active if point["point"].startswith("dram-bw") else cold
            target.extend(point["flush_ns"])
        coverage.append(
            {
                "order": trial["order"],
                "first_sample_ms": (origin + points[0]["start_ns"][0] + points[0]["flush_ns"][0])
                / MS,
                "last_sample_end_ms": (
                    origin
                    + points[-1]["start_ns"][-1]
                    + points[-1]["flush_ns"][-1]
                    + points[-1]["launch_ns"][-1]
                )
                / MS,
                "process_wall_ms": (trial["finished_realtime_ns"] - trial["started_realtime_ns"])
                / MS,
                "point_medians_gbps": {
                    point["point"]: median(
                        point["transfer_bytes"] / ns for ns in point["launch_ns"]
                    )
                    for point in points
                },
                "median_host_gap_ms": median(
                    (b - a - launch - flush) / MS
                    for point in points
                    for a, b, launch, flush in zip(
                        point["start_ns"][:-1],
                        point["start_ns"][1:],
                        point["launch_ns"][:-1],
                        point["flush_ns"][:-1],
                        strict=True,
                    )
                ),
            }
        )
    return {
        "contiguous": {
            "min_ms": min(cold) / MS,
            "median_ms": median(cold) / MS,
            "max_ms": max(cold) / MS,
        },
        "active_sm": {
            "min_ms": min(active) / MS,
            "median_ms": median(active) / MS,
            "max_ms": max(active) / MS,
        },
        "coverage": coverage,
    }


def summarize(points, receipt, proxy, host_gap):
    wall = (
        datetime.fromisoformat(receipt["finished_utc"])
        - datetime.fromisoformat(receipt["started_utc"])
    ).total_seconds() * 1000
    timed = sum(sum(p["launch_ns_samples"]) for p in points) / MS
    segments = []
    estimate = 0.0
    timed_prefix = 0.0
    for point in points:
        durations = [ns / MS for ns in point["launch_ns_samples"]]
        flush = proxy if point["geometry"]["field"].startswith("dram_") else 0.0
        estimate += mlp.WARMUP * (median(durations) + flush + host_gap)
        for segment in point["segments"]:
            lo, hi = segment["first_sample"] - 1, segment["last_sample"]
            prefix = sum(durations[:lo])
            suffix_after = timed - (timed_prefix + prefix)
            e = estimate + prefix + (lo + 1) * flush + lo * host_gap
            duration = sum(durations[lo:hi]) + (hi - lo - 1) * (flush + host_gap)
            segments.append(
                {
                    **segment,
                    "point": point["point"],
                    "field": point["geometry"]["field"],
                    "estimated_start_ms_excluding_startup": e,
                    "estimated_end_ms_excluding_startup": e + duration,
                    "popen_start_lower_bound_ms": timed_prefix + prefix,
                    "popen_start_upper_bound_ms": wall - suffix_after,
                }
            )
        estimate += sum(durations) + len(durations) * (flush + host_gap)
        timed_prefix += sum(durations)
    cold = [s for s in segments if s["field"].startswith("dram_")]
    gaps = [
        b["estimated_start_ms_excluding_startup"] - a["estimated_start_ms_excluding_startup"]
        for a, b in zip(cold, cold[1:])
    ]
    return {
        "report": receipt["report"],
        "process_wall_ms": wall,
        "timed_events_ms": timed,
        "estimated_span_ms_excluding_startup": estimate,
        "unallocated_wall_ms": wall - estimate,
        "cold_start_gap_ms": gaps,
        "segments": segments,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("flush_control", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    base = load(args.analysis)
    control = controls(args.flush_control)
    host_gap = median(row["median_host_gap_ms"] for row in control["coverage"])
    modes = {}
    for mode, evidence in base["modes"].items():
        rate = control["active_sm" if mode == "dram-bw" else "contiguous"]["median_ms"]
        modes[mode] = [
            summarize(points, receipt, rate, host_gap)
            for points, receipt in zip(evidence["runs"], evidence["receipts"], strict=True)
        ]
    mlp.write_json(
        args.output,
        {
            "position_analysis_sha256": mlp.digest(args.analysis),
            "flush_control_sha256": mlp.digest(args.flush_control),
            "control": control,
            "proxy_host_gap_ms": host_gap,
            "proxy_strided_flush": "Use the contiguous control's read-only flush, which has the same 256 MiB size.",
            "bounds": "Exclude all unobserved startup and overhead from the lower bound. The upper bound subtracts the remaining timed events from the outer process envelope.",
            "modes": modes,
        },
    )


if __name__ == "__main__":
    main()
