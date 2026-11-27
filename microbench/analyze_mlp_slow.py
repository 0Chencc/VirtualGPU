"""CPU-only audit of slow sample segments in frozen long-read sweeps."""

import argparse
import json
from itertools import groupby
from pathlib import Path
from statistics import median, quantiles

from microbench import long_mlp
from microbench import repeat_mlp as mlp
from microbench.capture_posthoc import iqr_ratio

MODES = ("mlp", "strided-mlp", "dram-bw")
REPEAT_FLOOR = 3
POINT_RADIUS = 2
SLOW_FRACTION = 0.85


def geometry(field, row):
    return {
        "field": field,
        "key": list(mlp.point_key(field, row)),
        **{
            key: row[key]
            for key in (
                "issue_mode",
                "bytes_per_cta",
                "active_ctas",
                "bytes_in_flight",
                "segment_bytes",
                "stride_bytes",
                "target_sms",
                "grid",
                "source_buffer_bytes",
                "transfer_bytes",
                "steps",
            )
            if key in row
        },
    }


def split_ratio(samples):
    """Descriptive fast/slow ratio under the preceding diagnostic's split rule."""
    values = sorted(samples)
    split = max(range(3, len(values) - 2), key=lambda i: values[i] - values[i - 1])
    lower, upper = values[:split], values[split:]
    gap = upper[0] - lower[-1]
    width = max(iqr_ratio(part) * median(part) for part in (lower, upper))
    if gap >= 0.05 * median(values) and gap > 2.5 * width:
        return median(upper) / median(lower)
    return None


def describe(record, use_p90):
    points = []
    clock_ns = 0.0
    index = 0
    for ordinal, (field, row) in enumerate(long_mlp.mode_rows(record["mode"], record["raw"])):
        times = row["launch_ns_samples"]
        values = row["gbps_samples"]
        reference = quantiles(values, n=10)[-1] if use_p90 else max(values)
        flags = [value < SLOW_FRACTION * reference for value in values]
        prefix_ns = []
        for duration in times:
            prefix_ns.append(clock_ns)
            clock_ns += duration
        segments = []
        for slow, group in groupby(enumerate(flags), key=lambda item: item[1]):
            positions = [i for i, _ in group]
            if slow and len(positions) >= 2:
                start, end = positions[0], positions[-1]
                segments.append(
                    {
                        "first_sample": start + 1,
                        "last_sample": end + 1,
                        "first_global_sample": index + start + 1,
                        "last_global_sample": index + end + 1,
                        "timed_only_start_ms": prefix_ns[start] / 1e6,
                        "timed_only_end_ms": (prefix_ns[end] + times[end]) / 1e6,
                    }
                )
        levels = [
            [v for v, slow in zip(values, flags, strict=True) if slow == bucket]
            for bucket in (True, False)
        ]
        points.append(
            {
                "point": ordinal + 1,
                "geometry": geometry(field, row),
                "previous_point": points[-1]["geometry"] if points else None,
                "pattern": "".join("L" if flag else "." for flag in flags),
                "first_global_sample": index + 1,
                "segments": segments,
                "median_gbps": median(values),
                "iqr_over_median": iqr_ratio(values),
                "two_cluster_fast_over_slow": split_ratio(values),
                "slow_fast_transitions": sum(a != b for a, b in zip(flags, flags[1:])),
                "slow_median_gbps": median(levels[0]) if levels[0] else None,
                "other_median_gbps": median(levels[1]) if levels[1] else None,
                "samples_gbps": values,
                "launch_ns_samples": times,
            }
        )
        index += len(values)
    return points


def align(runs):
    anchors = []
    width = len(runs[0])
    for anchor in range(1, width + 1):
        matches = [
            [
                p["point"]
                for p in points
                if abs(p["point"] - anchor) <= POINT_RADIUS and p["segments"]
            ]
            for points in runs
        ]
        exact = sum(anchor in match for match in matches)
        nearby = sum(bool(match) for match in matches)
        if nearby >= REPEAT_FLOOR:
            anchors.append(
                {
                    "anchor_point": anchor,
                    "matching_repeats": nearby,
                    "exact_repeats": exact,
                    "matching_points_by_repeat": matches,
                }
            )
    return anchors


def run(root):
    freeze_path = root / "freeze.json"
    freeze = json.loads(freeze_path.read_bytes())
    digest = mlp.digest(freeze_path)
    audit = json.loads((root / "audit.json").read_bytes())
    if audit["freeze_sha256"] != digest:
        raise ValueError("Capture audit binds a different schedule freeze")
    modes = {}
    for mode in MODES:
        records, receipts = [], []
        for repeat in range(1, mlp.REPEATS + 1):
            path = root / "capture" / f"{mode}-{repeat}.json"
            record = json.loads(path.read_bytes())
            receipt = audit["modes"][mode]["repeats"][repeat - 1]
            if (
                record["repeat"] != repeat
                or record["mode"] != mode
                or record["status"] != "complete"
                or not receipt["host_quiet_qualified"]
                or record["freeze_sha256"] != digest
                or receipt["path"] != path.name
                or receipt["sha256"] != mlp.digest(path)
            ):
                raise ValueError(f"{path}: invalid frozen receipt")
            records.append(record)
            receipts.append(
                {
                    "report": path.name,
                    "sha256": receipt["sha256"],
                    "started_utc": record["started_utc"],
                    "finished_utc": record["finished_utc"],
                }
            )
        primary = [describe(r, use_p90=False) for r in records]
        sensitivity = [describe(r, use_p90=True) for r in records]
        expected = [
            (row["field"], row["key"], row["steps"]) for row in freeze["modes"][mode]["points"]
        ]
        for points in primary:
            observed = [
                (p["geometry"]["field"], p["geometry"]["key"], p["geometry"]["steps"])
                for p in points
            ]
            if observed != expected:
                raise ValueError(f"{mode}: emission order or geometry changed")
        modes[mode] = {
            "receipts": receipts,
            "runs": primary,
            "matching_anchors": align(primary),
            "p90_matching_anchors": align(sensitivity),
        }
    return {
        "freeze_sha256": digest,
        "clock": "Cumulative CUDA event duration from the first timed launch. Excludes flushes, warmups, and host gaps.",
        "slow_definition": "Strictly below 0.85 times the within-point maximum bandwidth. A segment is at least two consecutive L samples in the same point.",
        "decision": "a" if any(mode["matching_anchors"] for mode in modes.values()) else "b",
        "modes": modes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    mlp.write_json(args.output, run(args.capture_root))


if __name__ == "__main__":
    main()
