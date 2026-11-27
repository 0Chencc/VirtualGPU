"""Compare normalized slow-segment onsets for a forward/forward/reverse replay."""

import argparse
import hashlib
import json
import math
from itertools import groupby
from pathlib import Path

TOLERANCE_NS = 20_000_000
SLOW_FRACTION = 0.85


def segments(key, bandwidth, starts_ns, origin_ns):
    cutoff = SLOW_FRACTION * max(bandwidth)
    flags = [value < cutoff for value in bandwidth]
    out = []
    for slow, group in groupby(enumerate(flags), key=lambda item: item[1]):
        samples = [i for i, _ in group]
        if slow and len(samples) >= 2:
            out.append(
                {
                    "key": key,
                    "first_sample": samples[0] + 1,
                    "last_sample": samples[-1] + 1,
                    "elapsed_ns": starts_ns[samples[0]] - origin_ns,
                }
            )
    return out


def validate_and_extract(order, points, origin_ns):
    """Check normalized telemetry against the frozen geometry and extract slow onsets."""
    if type(origin_ns) is not int or origin_ns < 0 or len(order) != len(points):
        raise ValueError("Invalid process origin or point count")
    previous_end = origin_ns
    onsets = []
    for frozen, point in zip(order, points, strict=True):
        for key in ("field", "key", "steps", "transfer_bytes", "source_buffer_bytes"):
            if frozen[key] != point[key]:
                raise ValueError(f"Point differs from its frozen geometry: {key}")
        if any(
            not math.isfinite(point[key]) or point[key] <= 0
            for key in ("sm_before_mhz", "sm_after_mhz")
        ):
            raise ValueError("Invalid per-point clock observations")
        durations = point["launch_ns_samples"]
        starts = point["starts_ns"]
        ends = point["ends_ns"]
        flushes = point["flush_ends_ns"]
        if any(len(values) != 15 for values in (durations, starts, ends, flushes)):
            raise ValueError("Timed sample count mismatch")
        cold = frozen["field"].startswith("dram_")
        for begin, finish, flush_end, duration in zip(
            starts, ends, flushes, durations, strict=True
        ):
            if not math.isfinite(duration) or duration < 1_000_000:
                raise ValueError("Timed launch is below 1 ms or invalid")
            if (
                type(begin) is not int
                or type(finish) is not int
                or not previous_end <= begin < finish
            ):
                raise ValueError("Process timestamps are out of order")
            if cold and (type(flush_end) is not int or not begin <= flush_end <= finish):
                raise ValueError("Missing or unordered cold flush timestamp")
            if not cold and flush_end is not None:
                raise ValueError("Warm sample unexpectedly reports a flush")
            previous_end = finish
        bandwidth = [point["transfer_bytes"] / value for value in durations]
        onsets.extend(segments([frozen["field"], *frozen["key"]], bandwidth, starts, origin_ns))
    return onsets


def normalize_raw(raw, order):
    if raw["mode"] != "mlp-order" or raw["samples"] != 15 or raw["warmup"] != 3:
        raise ValueError("Unexpected sweep protocol")
    points = []
    for index, (point, frozen) in enumerate(zip(raw["points"], order, strict=True)):
        if point["exec_index"] != index:
            raise ValueError("Execution index differs from recorded point order")
        is_cold = point["field"].startswith("dram_")
        points.append(
            {
                "field": point["field"],
                "key": [
                    point[key]
                    for key in ("issue_mode", "bytes_in_flight", "active_ctas", "bytes_per_cta")
                ],
                "steps": point["steps"],
                "transfer_bytes": point["transfer_bytes"],
                "source_buffer_bytes": point.get(
                    "source_buffer_bytes", frozen["source_buffer_bytes"]
                ),
                "sm_before_mhz": point["sm_mhz_before"],
                "sm_after_mhz": point["sm_mhz_after"],
                "launch_ns_samples": point["launch_ns"],
                "starts_ns": point["host_start_ns"],
                "ends_ns": point["launch_end_ns"],
                "flush_ends_ns": point["flush_end_ns"] if is_cold else [None] * 15,
            }
        )
    return validate_and_extract(order, points, raw["entry_monotonic_ns"])


def compare(forward_a, forward_b, reverse):
    matches = {"time_effect": [], "point_or_predecessor_effect": []}
    for a in forward_a:
        for b in forward_b:
            for c in reverse:
                times = [row["elapsed_ns"] for row in (a, b, c)]
                if c["key"] not in (a["key"], b["key"]) and max(times) - min(times) <= TOLERANCE_NS:
                    matches["time_effect"].append([a, b, c])
                if (
                    a["key"] == b["key"] == c["key"]
                    and min(abs(times[-1] - t) for t in times[:2]) > TOLERANCE_NS
                ):
                    matches["point_or_predecessor_effect"].append([a, b, c])
    supported = [name for name, rows in matches.items() if rows]
    return {
        "tolerance_ns": TOLERANCE_NS,
        "decision": supported[0] if len(supported) == 1 else "unresolved",
        "matches": matches,
        "onsets": [forward_a, forward_b, reverse],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "receipt", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    receipt = json.loads(args.receipt.read_bytes())
    if receipt["manifest_sha256"] != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError("Capture manifest digest mismatch")
    if receipt["base_freeze_sha256"] != manifest["base_freeze_sha256"]:
        raise ValueError("Capture base freeze digest mismatch")
    if [run["order"] for run in receipt["runs"]] != ["forward", "forward", "reverse"]:
        raise ValueError("Expected two forward sweeps followed by one reverse sweep")
    if any(run["host_check"]["status"] != "passed" for run in receipt["runs"]):
        raise ValueError("A capture failed the host gate")
    runs = []
    for run in receipt["runs"]:
        order = manifest["orders"][run["order"]]["points"]
        if len(order) != 96:
            raise ValueError("Expected 96 frozen geometry keys")
        runs.append(normalize_raw(run["result"], order))
    with args.output.open("x") as out:
        json.dump(compare(*runs), out, indent=2)
        out.write("\n")


if __name__ == "__main__":
    main()
