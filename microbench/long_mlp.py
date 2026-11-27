"""Size, freeze, capture, and audit long-launch MLP and active-SM read sweeps."""

import argparse
import json
import math
from argparse import Namespace
from pathlib import Path

from microbench import host_quiet
from microbench import repeat_mlp as mlp

PILOT_MIN_NS = 1_250_000
CAPTURE_MIN_NS = 1_000_000


def read(path):
    return json.loads(path.read_bytes())


def mode_rows(mode, raw):
    for field in mlp.MODES[mode]:
        name = "read_curve" if mode in mlp.ACTIVE_MODES else field
        yield from ((field, row) for row in raw[name])


def reference(profile_path, revision):
    profile = read(profile_path)
    memory = profile["measured"]["memory"]
    return {
        "created_utc": mlp.utc(),
        "source_revision": revision,
        "base_profile_sha256": mlp.digest(profile_path),
        "profile_revision": profile["provenance"]["profile_revision"],
        "prediction": "Current lookup-table expectation at each registered geometry",
        "curves": {field: memory[field] for fields in mlp.MODES.values() for field in fields},
    }


def validate_readout(profile, mode, raw, pilot=False):
    sample_count = 3 if pilot else mlp.SAMPLES
    mlp.point_maps(profile, mode, raw, sample_count)
    points = []
    for field, row in mode_rows(mode, raw):
        key = mlp.point_key(field, row)
        times = row["launch_ns_samples"]
        transferred = row["transfer_bytes"]
        steps = row["steps"]
        buffer_bytes = row["source_buffer_bytes"]
        if any(
            type(value) is not int or value <= 0 for value in (transferred, steps, buffer_bytes)
        ):
            raise ValueError(f"{field} {key}: invalid transfer geometry")
        if len(times) != sample_count or not all(math.isfinite(ns) and ns > 0 for ns in times):
            raise ValueError(f"{field} {key}: invalid CUDA event samples")
        if any(
            not math.isclose(bandwidth, transferred / ns, rel_tol=1e-6, abs_tol=1e-6)
            for bandwidth, ns in zip(row["gbps_samples"], times, strict=True)
        ):
            raise ValueError(f"{field} {key}: transfer bytes and bandwidth disagree")
        if mode == "strided-mlp" and transferred > buffer_bytes // 2:
            raise ValueError(f"{field} {key}: strided half-buffer cap exceeded")
        if field.startswith("dram_") and transferred > buffer_bytes:
            raise ValueError(f"{field} {key}: cold address range wraps")
        points.append(
            {
                "field": field,
                "key": list(key),
                "steps": steps,
                "transfer_bytes": transferred,
                "source_buffer_bytes": buffer_bytes,
                "minimum_launch_ns": min(times),
            }
        )
    return points


def identity(args):
    return {
        "source_revision": args.revision,
        "base_profile_sha256": mlp.digest(args.profile),
        "source_bundle_sha256": args.source_sha256,
        "binary_sha256": mlp.digest(args.binary),
    }


def host_qualified(record):
    if not record.get("device_query", {}).get("integrated"):
        return True
    before = record.get("host_quiet_gate") or {}
    check = record.get("host_quiet_check") or {}
    samples = record.get("host_samples")
    roots = record.get("host_quiet_roots")
    return (
        before.get("check", {}).get("status") == "passed"
        and check.get("status") == "passed"
        and bool(samples and roots)
        and host_quiet.check(samples, roots) == check
    )


def collect(args, frozen):
    args.binary = args.binary.resolve(strict=True)
    profile = read(args.profile)
    expected = identity(args)
    if frozen:
        freeze = read(args.freeze)
        if freeze["capture_identity"] != expected:
            raise ValueError("Capture identity does not match the freeze")
        if (
            mlp.digest(Path(mlp.__file__).with_name("capture_posthoc.py"))
            != freeze["shared_gate_sha256"]
        ):
            raise ValueError("Shared capture gate differs from the freeze")
        schedule = args.freeze.parent / freeze["modes"][args.mode]["schedule"]
        if mlp.digest(schedule) != freeze["modes"][args.mode]["schedule_sha256"]:
            raise ValueError("Frozen schedule digest mismatch")
        session_id = freeze["session_id"]
    else:
        prior = read(args.reference)
        if any(prior[key] != expected[key] for key in ("source_revision", "base_profile_sha256")):
            raise ValueError("Pilot identity does not match the reference freeze")
        schedule = "pilot"
        session_id = None
    if args.output.exists():
        raise FileExistsError(args.output)
    record = mlp.capture(
        Namespace(
            **vars(args), session_id=session_id, warmup_seconds=60, long_read_schedule=schedule
        )
    )
    record["kind"] = "capture" if frozen else "pilot"
    record["freeze_sha256"] = mlp.digest(args.freeze if frozen else args.reference)
    try:
        if frozen and record["nvcc"] != freeze["nvcc"]:
            raise ValueError("Capture toolchain differs from the freeze")
        points = validate_readout(profile, args.mode, record["raw"], pilot=not frozen)
        if frozen:
            mapped = mlp.point_maps(profile, args.mode, record["raw"])
            record["dispersion_check"] = mlp.dispersion_for_points(mapped)
            geometry = [
                {k: v for k, v in point.items() if k != "minimum_launch_ns"} for point in points
            ]
            if geometry != freeze["modes"][args.mode]["points"]:
                raise ValueError("Capture differs from frozen geometry or step counts")
        minimum_ns = CAPTURE_MIN_NS if frozen else PILOT_MIN_NS
        short_points = [point for point in points if point["minimum_launch_ns"] < minimum_ns]
        record["minimum_launch_ns"] = min(point["minimum_launch_ns"] for point in points)
        if short_points:
            record["below_minimum"] = short_points
            raise ValueError(f"{len(short_points)} points below {minimum_ns} ns")
        if not host_qualified(record):
            raise ValueError("Integrated GPU capture failed the host CPU quiet check")
        record["status"] = "complete"
    except ValueError as exc:
        record.update(status="protocol_failed", reason=str(exc))
        mlp.write_json(args.output, record)
        raise
    mlp.write_json(args.output, record)
    print(f"{args.mode}: {len(points)} points, min {record['minimum_launch_ns'] / 1e6:.3f} ms")


def freeze_schedule(args):
    profile = read(args.profile)
    expected = identity(args)
    ref_digest = mlp.digest(args.reference)
    prior = read(args.reference)
    if any(prior[key] != expected[key] for key in ("source_revision", "base_profile_sha256")):
        raise ValueError("Reference and schedule freeze use different inputs")
    modes = {}
    schedules = {}
    toolchains = set()
    for mode in mlp.MODES:
        path = args.pilot_dir / f"{mode}.json"
        record = read(path)
        if (
            record["kind"] != "pilot"
            or record["status"] != "complete"
            or record["mode"] != mode
            or record["freeze_sha256"] != ref_digest
            or not host_qualified(record)
            or any(record[key] != value for key, value in expected.items())
        ):
            raise ValueError(f"{path}: inconsistent pilot receipt")
        points = validate_readout(profile, mode, record["raw"], pilot=True)
        if min(point.pop("minimum_launch_ns") for point in points) < PILOT_MIN_NS:
            raise ValueError(f"{path}: sizing pilot failed")
        schedule = args.output.parent / f"{mode}.steps"
        schedules[mode] = schedule
        toolchains.add(record["nvcc"])
        modes[mode] = {
            "schedule": schedule.name,
            "pilot_sha256": mlp.digest(path),
            "points": points,
        }
    if len(toolchains) != 1:
        raise ValueError("The sizing pilot used different CUDA toolchains")
    if args.output.exists() or any(path.exists() for path in schedules.values()):
        raise FileExistsError("Schedule output already exists")
    for mode, schedule in schedules.items():
        with schedule.open("x") as output:
            output.writelines(f"{point['steps']}\n" for point in modes[mode]["points"])
        modes[mode]["schedule_sha256"] = mlp.digest(schedule)
    mlp.write_json(
        args.output,
        {
            "created_utc": mlp.utc(),
            "session_id": args.session_id,
            "capture_identity": expected,
            "nvcc": toolchains.pop(),
            "shared_gate_sha256": mlp.digest(Path(mlp.__file__).with_name("capture_posthoc.py")),
            "reference_sha256": ref_digest,
            "processes_per_mode": mlp.REPEATS,
            "samples_per_point": mlp.SAMPLES,
            "warmups_per_point": mlp.WARMUP,
            "capture_minimum_ns": CAPTURE_MIN_NS,
            "modes": modes,
        },
    )


def audit(args):
    profile = read(args.profile)
    freeze = read(args.freeze)
    freeze_digest = mlp.digest(args.freeze)
    result = {
        "created_utc": mlp.utc(),
        "freeze_sha256": freeze_digest,
        "checker_sha256": mlp.digest(Path(mlp.__file__).with_name("capture_posthoc.py")),
        "status": "passed",
        "modes": {},
    }
    expected = freeze["capture_identity"]
    if expected["base_profile_sha256"] != mlp.digest(args.profile):
        raise ValueError("Audit profile digest mismatch")
    for mode in mlp.MODES:
        repeats = []
        point_runs = []
        for repeat in range(1, mlp.REPEATS + 1):
            path = args.capture_dir / f"{mode}-{repeat}.json"
            record = read(path)
            if (
                record["mode"] != mode
                or record["repeat"] != repeat
                or record["kind"] != "capture"
                or record["freeze_sha256"] != freeze_digest
                or record["session_id"] != freeze["session_id"]
                or record["nvcc"] != freeze["nvcc"]
                or any(record[key] != value for key, value in expected.items())
            ):
                raise ValueError(f"{path}: inconsistent capture receipt")
            mapped = mlp.point_maps(profile, mode, record["raw"])
            point_runs.append(mapped)
            points = validate_readout(profile, mode, record["raw"])
            minimum_ns = min(point.pop("minimum_launch_ns") for point in points)
            if minimum_ns < CAPTURE_MIN_NS or points != freeze["modes"][mode]["points"]:
                result["status"] = "failed"
            checks = mlp.dispersion_for_points(mapped)
            quiet = host_qualified(record)
            if record["status"] != "complete" or checks["status"] != "passed" or not quiet:
                result["status"] = "failed"
            repeats.append(
                {
                    "path": path.name,
                    "sha256": mlp.digest(path),
                    "status": record["status"],
                    "minimum_launch_ns": minimum_ns,
                    "host_quiet_qualified": quiet,
                    "checks": checks,
                }
            )
        cross = mlp.reproducibility_for_runs(point_runs)
        if cross["status"] != "passed":
            result["status"] = "failed"
        result["modes"][mode] = {"repeats": repeats, "cross_process_range_gt_0.10": cross}
    mlp.write_json(args.output, result)
    print(result["status"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("reference", "pilot", "freeze", "capture", "audit"):
        cmd = commands.add_parser(action)
        cmd.add_argument("--profile", required=True, type=Path)
        cmd.add_argument("--output", required=True, type=Path)
        if action != "audit":
            cmd.add_argument("--revision", required=True)
        if action in ("pilot", "freeze", "capture"):
            cmd.add_argument("--binary", required=True, type=Path)
            cmd.add_argument("--source-sha256", required=True)
        if action in ("pilot", "capture"):
            cmd.add_argument("--nvcc", required=True, type=Path)
            cmd.add_argument("--mode", choices=mlp.MODES, required=True)
        if action == "pilot":
            cmd.set_defaults(repeat=1)
        if action == "capture":
            cmd.add_argument("--repeat", required=True, type=int, choices=range(1, mlp.REPEATS + 1))
        if action in ("pilot", "freeze"):
            cmd.add_argument("--reference", required=True, type=Path)
        if action in ("audit", "capture"):
            cmd.add_argument("--freeze", required=True, type=Path)
        if action == "freeze":
            cmd.add_argument("--pilot-dir", required=True, type=Path)
            cmd.add_argument("--session-id", required=True)
        if action == "audit":
            cmd.add_argument("--capture-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "reference":
        mlp.write_json(args.output, reference(args.profile, args.revision))
    elif args.action == "pilot":
        collect(args, frozen=False)
    elif args.action == "freeze":
        freeze_schedule(args)
    elif args.action == "capture":
        collect(args, frozen=True)
    else:
        audit(args)


if __name__ == "__main__":
    main()
