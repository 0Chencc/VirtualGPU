"""Capture a v0.3 case-session under the published plan, then seal it."""

import argparse
import json
import math
import os
import stat
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

from microbench import host_quiet
from microbench.dram_reference import require_lock
from validation.holdout import seal_report, sha256
from validation.provenance import REPO_ROOT
from validation.v02_score import check_cache_control
from validation.v03_plan import (
    DEVICES,
    FAMILIES,
    aot_case,
    checked_json,
    load_plan,
    published_main,
    require_capture,
    require_reviewed_code,
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, record):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("x") as output:
        json.dump(record, output, indent=2, allow_nan=False)
        output.write("\n")


def receipt_name(case_id, session, attempt):
    stem = f"{case_id}-session-{session}"
    return f"{stem}.json" if attempt == 1 else f"{stem}-attempt-{attempt}.json"


def check_sealed(path):
    details = path.lstat()
    if not stat.S_ISREG(details.st_mode) or stat.S_IMODE(details.st_mode) != 0:
        raise ValueError(f"{path.name}: expected a regular mode-zero seal")


@contextmanager
def host_trace(samples):
    done = Event()

    def monitor():
        while not done.wait(1):
            samples.append({"timestamp_utc": utc(), **host_quiet.sample()})

    samples.append({"timestamp_utc": utc(), **host_quiet.sample()})
    thread = Thread(target=monitor)
    thread.start()
    try:
        yield
    finally:
        done.set()
        thread.join()
        samples.append({"timestamp_utc": utc(), **host_quiet.sample()})


def run_session(device, family, case, configs, manifest, manifest_sha, profile, aot, session, trace):
    import pynvml
    import triton
    import virtualgpu.frontend

    from validation import attention, runner
    from validation.attention_runner import verify_artifacts
    from validation.baseline import residency
    from validation.phase2_capture import SparkSession
    from validation.v02_capture import Rtx4070Session
    from validation.v02_predict import require_imports_from_checkout

    require_imports_from_checkout(virtualgpu.frontend)
    if triton.__version__ != "3.9.0" or triton.knobs.nvidia.disable_ptxas_opt:
        raise ValueError("Expected Triton 3.9.0 with assembler optimization enabled")
    specs = profile["specs"]
    provenance = profile["provenance"]
    driver = provenance["driver_version"]
    if provenance["clock_policy"]["locked"] or profile["identity"]["arch"] != DEVICES[device]:
        raise ValueError("Expected the approved record-only profile for the target")
    pynvml.nvmlInit()
    try:
        if pynvml.nvmlSystemGetDriverVersion() != driver:
            raise ValueError(f"Driver differs from capture policy (expected {driver})")
    finally:
        pynvml.nvmlShutdown()

    policy = {
        "l2_bytes": specs["l2_size_bytes"], "driver_version": driver,
        "clock_policy": "record_only", "power_mode": {"unavailable": "nvpmodel is not installed"},
    }
    gb10 = device == "gb10"
    environment = SparkSession(policy, trace) if gb10 else Rtx4070Session(policy)
    arch = int(DEVICES[device].removeprefix("sm_"))

    def verify(artifacts):
        for key, row in aot["configs"].items():
            if row["status"] == "resource_exceeded":
                artifacts[key] = {"status": "resource_exceeded", "reason": row["exceeded"]}
        verify_artifacts(aot, artifacts)
        if aot["wave"]:
            expected = aot["wave"]
            row = artifacts[expected["config_id"]]
            if (row["status"] != "ok" or math.prod(row["grid"]) != expected["ctas"]
                    or residency(row, specs) * specs["sm_count"] != expected["full_wave_capacity"]):
                raise ValueError("Untimed GEMM reference does not match the published wave audit")

    # The published AOT resource-exceeded rows are recorded without launching them.
    enabled = [cfg for cfg in configs if aot["configs"][cfg["config_id"]]["status"] != "resource_exceeded"]
    prepare = None
    if family == "attention":
        case = dict(case, shape=[case[key] for key in attention.SHAPE_FIELDS])

        def prepare(case, configs, seed):
            return attention.prepare_candidates(case, configs, seed, arch=arch)

    args = SimpleNamespace(cache="both", blocks=10, seed=100 + session, warmup_seconds=60 if gb10 else 2)
    kernel = runner.dense_gemm if family == "gemm" else attention.fused_attention
    report = runner.measure(
        case, enabled, manifest["input_seed"], args, manifest_sha,
        verify_artifacts=verify, device_arch=arch, kernel=kernel,
        spark=environment, prepare=prepare, retry_rejected_blocks=False,
    )
    return report, kernel.cache_key


def capture(directory, aot_directory, device, family, case_id, session, attempt):
    main = published_main()
    release, release_sha, inputs = load_plan(main)
    head = require_reviewed_code(release["code_commit"], capture=True)
    manifest = inputs[family]
    case = next((row for row in manifest["devices"][device]["cases"] if row["case_id"] == case_id), None)
    if case is None or attempt < 1:
        raise ValueError("Expected a published case and a positive attempt")
    base = directory / f"{device}-{family}"
    name = receipt_name(case_id, session, attempt)
    receipt = base / "receipts" / name
    seal = base / "sealed" / name
    trace = base / "thermal-timeouts" / name
    host_log = base / "host" / name
    if any(path.exists() or path.is_symlink() for path in (receipt, seal, trace, host_log)):
        raise FileExistsError(name)
    failed_sha = None
    if attempt > 1:
        raw = (base / "receipts" / receipt_name(case_id, session, attempt - 1)).read_bytes()
        previous = json.loads(raw)
        if previous["status"] != "protocol_failed" or previous["release_sha256"] != release_sha:
            raise ValueError("Only a recorded protocol failure from this release can be retried")
        failed_sha = sha256(raw)
    approval_sha, retry_sha = require_capture(main, device, release_sha, case_id, session, attempt, failed_sha)
    profile_sha = release["capture_profiles"][device]
    profile = checked_json(REPO_ROOT / "profiles" / f"{device}.json", profile_sha)
    cohort = inputs["aot"]["cohorts"][f"{device}-{family}"]
    if cohort["profile_sha256"] != profile_sha:
        raise ValueError("Capture policy differs from the profile used for the published resource audit")
    aot = aot_case(aot_directory, inputs, device, family, case)
    by_id = {row["config_id"]: row for row in manifest["configs"]}
    configs = [by_id[key] for key in case["config_ids"]]

    require_lock()
    samples, gate, report = [], None, None
    roots = [os.getpid()]
    details = {
        "timestamp_utc": utc(), "release_sha256": release_sha, "capture_approval_sha256": approval_sha,
        "retry_approval_sha256": retry_sha, "repeat_of_receipt_sha256": failed_sha, "device": device,
        "family": family, "case_id": case_id, "session": session, "attempt": attempt, "source_commit": head,
        "manifest_sha256": release["inputs"][family], "aot_sha256": aot["sha256"],
        "profile_sha256": profile_sha,
    }
    failures = []
    try:
        if device == "rtx4070":
            from microbench.capture_posthoc import wait_for_quiet_desktop

            gate = wait_for_quiet_desktop()
            if gate["status"] != "passed":
                raise RuntimeError("The approved Windows quiet gate was unavailable")
            report, kernel = run_session(
                device, family, case, configs, manifest, release["inputs"][family], profile, aot, session, trace,
            )
        else:
            gate = host_quiet.wait_for_quiet_host(roots)
            with host_trace(samples):
                report, kernel = run_session(
                    device, family, case, configs, manifest, release["inputs"][family], profile, aot, session, trace,
                )
        after = report["spark"]["after"]["compute_processes"]
        if report["complete_blocks"] != 10:
            failures.append("incomplete randomized blocks")
        if after["status"] or set(after["stdout"].split()) - {str(os.getpid())}:
            failures.append("foreign compute process or unavailable process check")
        check_cache_control(report["cache_control_ns"], case_id)
        if set(report["configs"]) != set(case["config_ids"]):
            raise ValueError("The report must account for every declared candidate")
    except (RuntimeError, ValueError) as error:
        failures.append(str(error))

    host = {"gate": gate}
    if samples:
        host |= {"roots": roots, "samples": host_quiet.slim(samples, roots), "check": host_quiet.check(samples, roots)}
        if host["check"]["status"] != "passed":
            failures.append("host quiet check failed")
    write_json(host_log, host)
    details["host_sha256"] = sha256(host_log.read_bytes())
    if trace.exists():
        details["thermal_timeout_sha256"] = sha256(trace.read_bytes())
    details.update(status="protocol_failed" if failures else "complete", failure_reasons=failures)
    if report is not None:
        report.update(profile_sha256=profile_sha, aot_sha256=aot["sha256"], kernel_key=kernel,
                      collection_status=details["status"], release_sha256=release_sha)
        seal.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        details.update(seal_report(seal, report))
        details["complete_blocks"] = report["complete_blocks"]
        details["kernel_key"] = kernel
        details["config_status_counts"] = dict(Counter(row["status"] for row in report["configs"].values()))
    write_json(receipt, details)
    print(f"{name}: {details['status']}", flush=True)
    if failures:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("aot_directory", type=Path)
    parser.add_argument("device", choices=DEVICES)
    parser.add_argument("family", choices=FAMILIES)
    parser.add_argument("case")
    parser.add_argument("session", type=int, choices=(1, 2, 3))
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    capture(args.directory, args.aot_directory, args.device, args.family, args.case, args.session, args.attempt)


if __name__ == "__main__":
    main()
