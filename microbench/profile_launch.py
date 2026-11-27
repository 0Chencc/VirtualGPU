"""Attach the long-launch records of merged raws (microbench/long_capture.py) to a profile.

For each timed point of a merged raw, the profile entry with the same design fields
gets, on every value built from one of the point's sample lists: `launch`, the range
of the round medians, and for compute rates the per-cycle value (median over rounds
of each round's rate over its mean SM clock). Entries a device exception covers get
the exception's flag and an uncertainty that covers the round medians. A point
without its profile entry, or a value that is not the median of its list, is an error.
"""

from __future__ import annotations

import statistics

try:
    import long_launch
except ImportError:  # imported as microbench.profile_launch
    from microbench import long_launch

# (mode, raw field) -> (profile section, profile field)
PROFILE_PATH = {
    ("mlp", "dram_bw_by_bytes_in_flight"): ("memory", "dram_bw_by_bytes_in_flight"),
    ("mlp", "l2_bw_by_bytes_in_flight"): ("memory", "l2_bw_by_bytes_in_flight"),
    ("strided-mlp", "dram_bw_by_bytes_in_flight_strided"): (
        "memory",
        "dram_bw_by_bytes_in_flight_strided",
    ),
    ("dram-bw", "read_curve"): ("memory", "dram_bw_by_active_sms"),
    ("l2", "read_curve"): ("memory", "l2_bw_by_active_sms"),
    ("closed-loop", "dram_closed_loop"): ("memory", "dram_closed_loop"),
    ("closed-loop", "l2_closed_loop"): ("memory", "l2_closed_loop"),
    ("l2-merge", "l2_merge_by_sharing"): ("memory", "l2_merge_by_sharing"),
    ("mma", "mma"): ("compute", "mma_tflops_per_sm_by_warps"),
    ("mma", "fp32_fma"): ("compute", "fp32_fma_tflops"),
    ("gmem-issue", "gmem_load_issue_per_sm"): ("memory", "gmem_load_issue_per_sm"),
    ("gmem-issue-controlled", "gmem_load_issue_controlled"): (
        "memory",
        "gmem_load_issue_controlled",
    ),
    ("sync-loop", "sync_stage_loop"): ("pipeline", "sync_stage_loop"),
    ("closed-loop-mma", "dram_closed_loop_mma"): ("memory", "dram_closed_loop_mma"),
    ("spill-latency", "reload_latency_by_count"): ("memory", "spill_reload_latency_by_count"),
    ("mma-chain", "mma_chain_latency"): ("compute", "mma_chain_latency"),
}
# Raw sample lists whose profile value has another name than the list without "_samples".
STAGE_LOOP = {
    "bw_gbps_samples": "delivered_bw_gbps",
    "turnaround_ns_samples": "mean_stage_turnaround_ns",
    "mma_tflops_per_sm_samples": "achieved_mma_tflops_per_sm",
    "tensor_util_samples": "tensor_utilization",
}
METRIC = {
    "closed-loop": STAGE_LOOP,
    "sync-loop": STAGE_LOOP,
    "closed-loop-mma": STAGE_LOOP,
}
# A point whose rate list sits outside it: (mode, raw field) -> top-level list.
OUTSIDE = {("mma", "fp32_fma"): "fp32_fma_tflops_samples"}
# Profile values that are compute rates and get a per-cycle value.
COMPUTE = {"tflops_per_sm", "fp32_fma_tflops", "issue_rate_gwarp_inst_per_sm"}
# Design fields the profile names differently.
ALIAS = {"target_sms": "active_sms"}
FLAG = {"low_in_flight_unstable": "low_in_flight_unstable", "bistable_band": "bistable_band"}


def metric(mode, key):
    if key == "gbps_samples":
        return "bw_gbps"
    return METRIC.get(mode, {}).get(key, key.removesuffix("_samples"))


def entry_for(profile, mode, path, point):
    """The profile entry (or, for a scalar, its container and key) of a merged point."""
    section, field = PROFILE_PATH[(mode, path[0])]
    target = profile["measured"][section][field]
    if (mode, path[0]) in OUTSIDE:
        return profile["measured"][section], field
    for key in path[1:]:
        target = target[key]
    names = long_launch.DESIGN_FIELDS[mode][path[0]]
    matches = [e for e in target if all(e[ALIAS.get(k, k)] == point[k] for k in names)]
    if len(matches) != 1:
        raise ValueError(
            f"{mode} {path}: {len(matches)} profile entries for {[point[k] for k in names]}"
        )
    return matches[0], None


def per_cycle(rates, launch, unit):
    clocks = [
        (b + a) / 2 for b, a in zip(launch["sm_mhz_before"], launch["sm_mhz_after"], strict=True)
    ]
    if None in clocks or len(clocks) != len(rates):
        raise ValueError("per-cycle values need every round's SM clock")
    return {
        "value": statistics.median(r / c for r, c in zip(rates, clocks)),
        "unit": f"{unit} per MHz",
    }


def annotate(value, rounds, launch, compute):
    """Attach one value's records; `rounds` is the list of its round medians."""
    if abs(value["value"] - statistics.median(rounds)) > 1e-3 * max(abs(value["value"]), 1e-9):
        raise ValueError(f"value {value['value']} is not the median of its round medians")
    value["launch"] = {
        k: launch[k] for k in ("length", "min_ns", "median_ns", "sm_mhz_before", "sm_mhz_after")
    }
    value["round_median_range"] = [min(rounds), max(rounds)]
    if compute:
        value["per_cycle"] = per_cycle(rounds, launch, value["unit"])


def cover(value):
    """Widen the uncertainty so value +/- uncertainty spans the round medians."""
    low, high = value["round_median_range"]
    spread = max(value["value"] - low, high - value["value"]) / value["value"]
    value["rel_uncertainty"] = round(min(1.0, max(value["rel_uncertainty"], spread)), 4)


def attach(profile, mode, merged):
    """Annotate `profile` in place from one mode's merged raw."""
    for path, point in long_launch.points(merged):
        entry, scalar = entry_for(profile, mode, path, point)
        if scalar is not None:
            lists = {OUTSIDE[(mode, path[0])]: merged[OUTSIDE[(mode, path[0])]]}
            values = {scalar: lists[OUTSIDE[(mode, path[0])]]}
            targets = {scalar: entry[scalar]}
        else:
            values = {
                metric(mode, k): v
                for k, v in point.items()
                if k.endswith("_samples") and k != "launch_ns_samples"
            }
            targets = {name: entry[name] for name in values}
        for name, rounds in values.items():
            annotate(targets[name], rounds, point["launch"], name in COMPUTE)
        for name in point.get("exceptions", ()):
            if scalar is None:
                entry[FLAG[name]] = True
            for value in targets.values():
                cover(value)
    return profile
