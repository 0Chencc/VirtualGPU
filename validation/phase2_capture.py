"""Capture a sealed GB10 case-session after the published prediction freeze."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import time
from types import SimpleNamespace

from validation.holdout import seal_report
from validation.phase2_aot import REVISION, TMA_CASE, load_tma_kernel, write_json
from validation.provenance import REPO_ROOT, read_core_revision, source_info

THERMAL_AUDIT = "validation/phase2-thermal-audit.json"
THERMAL_TIMEOUT_SECONDS = 20 * 60


def thermal_harness_revision(policy):
    revision = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--diff-filter=A", "origin/main", "--", THERMAL_AUDIT],
        cwd=REPO_ROOT, text=True,
    ).strip()
    if not revision:
        return None
    audit = json.loads(subprocess.check_output(
        ["git", "show", f"{revision}:{THERMAL_AUDIT}"], cwd=REPO_ROOT,
    ))
    if (audit["previous_harness_commit"] != policy["harness_commit"]
            or audit["new_timeout_seconds"] != THERMAL_TIMEOUT_SECONDS):
        raise ValueError("Published GB10 thermal revision does not match the frozen policy")
    return revision


def command(*args):
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    return {"status": result.returncode, "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip()}


def power_mode():
    if shutil.which("nvpmodel"):
        return command("nvpmodel", "-q")
    return {"unavailable": "nvpmodel is not installed"}


def host_snapshot():
    return {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "uptime": command("uptime"),
        "memory": Path("/proc/meminfo").read_text(),
        "gpu": command("nvidia-smi"),
        "compute_processes": command("nvidia-smi", "--query-compute-apps=pid",
                                     "--format=csv,noheader,nounits"),
        "power_mode": power_mode(),
    }


class SparkSession:
    def __init__(self, policy, thermal_trace):
        self.thermal_trace = thermal_trace
        self.clock_policy = policy["clock_policy"]
        self.before = host_snapshot()
        self.initial_power = policy["power_mode"]
        self.driver_version = policy["driver_version"]
        self.l2_bytes = policy["l2_bytes"]
        if platform.node() != "spark-cff6":
            raise ValueError("GB10 collection requires Spark 1")
        processes = self.before["compute_processes"]
        if (self.before["gpu"]["status"] or processes["status"] or processes["stdout"]
                or self.before["uptime"]["status"]):
            raise ValueError("Spark 1 GPU availability could not be verified")
        if self.before["power_mode"] != self.initial_power:
            raise ValueError("Spark 1 power mode does not match the calibration record")
        self.soaks = []

    def stabilize(self, warmup, telemetry, has_operations):
        if not has_operations:
            return
        start = time.monotonic()
        while time.monotonic() - start < THERMAL_TIMEOUT_SECONDS:
            warmup()
            now = time.monotonic()
            if now - start < 60:
                continue
            recent_ns = int((now - 30) * 1e9)
            values = [row["temperature_c"] for row in telemetry
                      if row["monotonic_ns"] >= recent_ns]
            measured = [value for value in values if isinstance(value, int)]
            if values and (not measured or len(measured) == len(values) and max(measured) - min(measured) <= 2):
                self.soaks.append({"elapsed_seconds": now - start,
                                   "temperature_range_c": [min(measured), max(measured)] if measured else None})
                return
        trace = {}
        for row in telemetry:
            elapsed = row["monotonic_ns"] / 1e9 - start
            if elapsed >= 0:
                trace[int(elapsed)] = row["temperature_c"]
        self.thermal_trace.parent.mkdir(mode=0o700, exist_ok=True)
        with self.thermal_trace.open("x") as stream:
            json.dump({"utc": datetime.now(timezone.utc).isoformat(),
                       "temperature_c_by_elapsed_second": trace}, stream, allow_nan=False)
            stream.write("\n")
        raise RuntimeError("Spark 1 temperature did not stabilize within twenty minutes")

    def begin_block(self):
        counters = {}
        for line in Path("/proc/vmstat").read_text().splitlines():
            name, value = line.split()
            if name in ("pswpin", "pswpout"):
                counters[name] = int(value)
        return counters

    def check_block(self, handle, telemetry, start_ns, begin):
        import pynvml
        from validation.runner import telemetry_value

        thermal_mask = (pynvml.nvmlClocksEventReasonSwThermalSlowdown |
                        pynvml.nvmlClocksEventReasonHwThermalSlowdown)
        events = [row["clock_event_reasons"] for row in telemetry if row["monotonic_ns"] >= start_ns]
        events.append(telemetry_value(pynvml.nvmlDeviceGetCurrentClocksEventReasons, handle))
        if any(isinstance(event, int) and event & thermal_mask for event in events):
            return "thermal_throttling"
        if power_mode() != self.initial_power:
            return "power_mode_changed"
        if self.begin_block() != begin:
            return "host_swap_activity"
        return None

    def provenance(self):
        return {"before": self.before, "after": host_snapshot(), "thermal_soaks": self.soaks}


def verify_release(directory, amendment):
    amendment = read_core_revision(amendment)
    subprocess.run(["git", "merge-base", "--is-ancestor", amendment, "origin/main"],
                   cwd=REPO_ROOT, check=True)
    prereg = subprocess.check_output(
        ["git", "show", f"{amendment}:validation/phase2-preregistration.md"], cwd=REPO_ROOT, text=True,
    )
    index_bytes = (directory / "aot-index.json").read_bytes()
    with (directory / "predictions.json").open("rb") as archive:
        archive_digest = hashlib.file_digest(archive, "sha256").hexdigest()
    index_digest = hashlib.sha256(index_bytes).hexdigest()
    policy_bytes = (directory / "measurement-policy.json").read_bytes()
    policy_digest = hashlib.sha256(policy_bytes).hexdigest()
    for label, digest in (
        ("GB10 prediction archive", archive_digest),
        ("GB10 AOT index", index_digest),
        ("GB10 measurement policy", policy_digest),
    ):
        if f"| {label} | `{digest}` |" not in prereg:
            raise ValueError(f"{label} does not match the published freeze")
    return json.loads(index_bytes), json.loads(policy_bytes), archive_digest


def capture(directory, frozen, cohort, case_id, session, amendment):
    import triton
    from validation.runner import dense_gemm, measure
    from validation.score import check_artifact

    index, policy, archive_sha256 = verify_release(directory, amendment)
    source = source_info()
    if (source["commit"] != thermal_harness_revision(policy) or source["python_dirty"]
            or source["validation_dirty"] or triton.__version__ != "3.9.0"):
        raise ValueError("Collect using the published GB10 harness and Triton 3.9.0")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=frozen, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=frozen, text=True)
    if revision != REVISION or dirty:
        raise ValueError("Collect using the clean phase1-final kernel sources")
    if hashlib.sha256((frozen / "validation/runner.py").read_bytes()).hexdigest() != index["dense_runner_sha256"]:
        raise ValueError("Pinned dense kernel source changed")
    if hashlib.sha256(dense_gemm.src.encode()).hexdigest() != index["dense_kernel_sha256"]:
        raise ValueError("Harness dense kernel does not match phase1-final")

    def checked_json(path):
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != index["files"][path.relative_to(directory).as_posix()]:
            raise ValueError(f"{path.name}: digest mismatch")
        return json.loads(data)

    base = directory / cohort
    manifest = base / "manifest.json"
    suite = checked_json(manifest)
    case = next(item for item in suite["cases"] if item["case_id"] == case_id)
    configs = [cfg for cfg in suite["configs"] if cfg["config_id"] in case["config_ids"]]
    diagnostic = cohort == "diagnostic" and case_id == TMA_CASE
    kernel = load_tma_kernel(frozen) if diagnostic else dense_gemm
    if diagnostic:
        import torch
        triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.uint8))

    def verify(artifacts):
        for key, jit in artifacts.items():
            if jit["status"] != "ok":
                continue
            path = base / "aot" / case_id / f"{key}.json"
            if path.relative_to(directory).as_posix() not in index["files"]:
                jit.update(status="aot_missing", reason="No frozen AOT artifact")
                continue
            ir = checked_json(path)
            check = check_artifact(jit, ir)
            jit["aot_check"] = check
            if check["status"] != "matched" or not check["ttgir_match"]:
                jit.update(status="artifact_mismatch", reason="Pinned AOT/JIT artifact check failed")

    raw_dir = base / "sealed"
    receipts = base / "receipts"
    raw_dir.mkdir(mode=0o700, exist_ok=True)
    receipts.mkdir(mode=0o700, exist_ok=True)
    name = f"{case_id}-session-{session}.json"
    report_path, receipt_path = raw_dir / name, receipts / name
    if report_path.exists() or receipt_path.exists():
        raise FileExistsError(name)
    trace = base / "thermal-timeouts" / name
    spark = SparkSession(policy, trace)
    args = SimpleNamespace(cache="both", blocks=10, seed=100 + session, warmup_seconds=60)
    report = measure(case, configs, suite["input_seed"], args,
                     index["files"][manifest.relative_to(directory).as_posix()],
                     verify_artifacts=verify, device_arch=121, kernel=kernel,
                     tensor_copy=diagnostic, spark=spark)
    report.update(cohort=cohort, frozen_revision=revision,
                  prediction_archive_sha256=archive_sha256, preregistration_commit=amendment)
    receipt = seal_report(report_path, report)
    write_json(receipt_path, receipt)
    checks = [row["status"] for row in report["configs"].values()]
    if any(state in ("aot_missing", "artifact_mismatch") for state in checks):
        return "FAIL"
    return "PASS" if "ok" in checks else "NO ELIGIBLE CONFIGS"


def audit(directory, output):
    reports = {}
    index = json.loads((directory / "aot-index.json").read_bytes())
    for cohort in ("primary", "secondary", "diagnostic"):
        base = directory / cohort
        raw = (base / "manifest.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != index["files"][f"{cohort}/manifest.json"]:
            raise ValueError(f"{cohort}: manifest digest mismatch")
        names = {f"{case['case_id']}-session-{session}.json"
                 for case in json.loads(raw)["cases"] for session in (1, 2, 3)}
        if any({p.name for p in (base / kind).glob("*.json")} != names
               for kind in ("sealed", "receipts")):
            raise ValueError(f"{cohort}: incomplete or extra sealed reports")
        for name in sorted(names):
            path = base / "sealed" / name
            if path.is_symlink() or stat.S_IMODE(path.stat().st_mode) != 0:
                raise ValueError(f"{cohort}/{name}: report is not sealed")
            reports[f"{cohort}/sealed/{name}"] = json.loads((base / "receipts" / name).read_bytes())
    with output.open("x") as stream:
        json.dump({"device": "GB10", "utc": datetime.now(timezone.utc).isoformat(),
                   "reports": reports}, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("capture")
    run.add_argument("directory", type=Path)
    run.add_argument("frozen", type=Path)
    run.add_argument("cohort", choices=("primary", "secondary", "diagnostic"))
    run.add_argument("case")
    run.add_argument("session", type=int, choices=(1, 2, 3))
    run.add_argument("amendment")
    receipt = commands.add_parser("audit")
    receipt.add_argument("directory", type=Path)
    receipt.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "audit":
        audit(args.directory, args.output)
    else:
        print("AOT/JIT " + capture(args.directory, args.frozen, args.cohort, args.case,
                                   args.session, args.amendment), flush=True)


if __name__ == "__main__":
    main()
