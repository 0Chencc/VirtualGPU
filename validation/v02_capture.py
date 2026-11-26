"""Capture preregistered v0.2 holdouts without releasing their labels."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import partial
import json
import math
import os
import shutil
from pathlib import Path
import stat
import subprocess
import time
from types import SimpleNamespace

import triton
from virtualgpu.frontend.compiler import MockTensor, specialize_triton_call

from validation import attention, attention_runner, runner
from validation.baseline import residency
from validation.holdout import seal_report, sha256
from validation.phase2_capture import SparkSession, command, power_mode
from validation.provenance import REPO_ROOT, source_info

APPROVAL = "29a6ac2"
DEVICE_ARCH = {"gb10": 121, "rtx4070": 89}
RTX_WARMUP_SECONDS = 2
GEMM_POINTERS = tuple(MockTensor("f16") for _ in range(3))


def utc():
    return datetime.now(timezone.utc).isoformat()


def inputs(family, device):
    subprocess.run(["git", "merge-base", "--is-ancestor", APPROVAL, "origin/main"],
                   cwd=REPO_ROOT, check=True)
    relative = f"validation/v02-{family}-holdout.json"
    raw = (REPO_ROOT / relative).read_bytes()
    published = subprocess.check_output(["git", "show", f"{APPROVAL}:{relative}"], cwd=REPO_ROOT)
    if published != raw:
        raise ValueError("Manifest differs from the approved GB10 matrix")
    manifest = json.loads(raw)
    source = source_info()
    profiles_dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", "profiles/"], cwd=REPO_ROOT)
    if (source["python_dirty"] or source["validation_dirty"] or profiles_dirty or triton.__version__ != "3.9.0"
            or manifest["devices"][device]["target"] != f"sm_{DEVICE_ARCH[device]}"):
        raise ValueError("Use a clean capture checkout and Triton 3.9.0")
    return manifest, sha256(raw), source


def case_configs(manifest, case_id, device):
    case = next(row for row in manifest["devices"][device]["cases"] if row["case_id"] == case_id)
    keys = set(case["config_ids"])
    configs = [row for row in manifest["configs"] if row["config_id"] in keys]
    if len(configs) != len(keys):
        raise ValueError("The case candidate matrix is incomplete")
    return case, configs


def identity(manifest_digest, case, kernel):
    return {"manifest_sha256": manifest_digest, "case_id": case["case_id"],
            "kernel_key": kernel.cache_key}


def gemm_artifact(case, config, target):
    m, n, k = case["shape"]
    bm, bn, bk = config["tile"]
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn), 1, 1)
    options = {"num_warps": config["num_warps"], "num_stages": config["num_stages"],
               "enable_fp_fusion": True}
    args = (*GEMM_POINTERS, m, n, k, bm, bn, bk, config["group_m"])
    parsed, signature, constexprs, attrs = specialize_triton_call(runner.dense_gemm, target, args, options)
    return attention_runner.compile_identity(runner.dense_gemm, target, grid, options,
                                             parsed, signature, constexprs, attrs)


def compile_case(directory, family, case_id, device):
    from triton.runtime.errors import PTXASError

    manifest, digest, source = inputs(family, device)
    case, configs = case_configs(manifest, case_id, device)
    kernel = runner.dense_gemm if family == "gemm" else attention.fused_attention
    compile_artifact = gemm_artifact if family == "gemm" else attention_runner.compile_artifact
    rows = {}
    for config in configs:
        try:
            row = compile_artifact(case, config, f"sm_{DEVICE_ARCH[device]}")
        except (triton.CompilationError, PTXASError, RuntimeError) as error:
            row = {"status": "compile_failed", "reason": str(error)}
        rows[config["config_id"]] = row
    output = directory / family / "aot" / f"{case_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    attention_runner.write_json(output, {**identity(digest, case, kernel), "source": source, "configs": rows})
    print(f"{case_id}: {dict(Counter(row['status'] for row in rows.values()))}", flush=True)


def read_aot(base, case, digest, kernel, source):
    raw = (base / "aot" / f"{case['case_id']}.json").read_bytes()
    archive = attention_runner.load_aot(raw, case, identity(digest, case, kernel))
    if archive["source"] != source:
        raise ValueError("AOT archive and capture checkout differ")
    return archive, sha256(raw)


def load_policy(device):
    raw = (REPO_ROOT / "profiles" / f"{device}.json").read_bytes()
    profile = json.loads(raw)
    fields = profile["provenance"]
    if profile["identity"]["arch"] != f"sm_{DEVICE_ARCH[device]}" or fields["clock_policy"]["locked"]:
        raise ValueError("Capture requires the target record-only profile")
    policy = {"l2_bytes": profile["specs"]["l2_size_bytes"], "clock_policy": "record_only",
              "driver_version": fields["driver_version"],
              "power_mode": {"unavailable": "nvpmodel is not installed"}}
    return policy, profile["specs"], sha256(raw)


class Rtx4070Session(SparkSession):
    """Use the established record-only RTX 4070 collection protocol."""

    def __init__(self, policy):
        self.clock_policy = policy["clock_policy"]
        self.driver_version = policy["driver_version"]
        self.l2_bytes = policy["l2_bytes"]
        self.initial_power = power_mode()
        self.smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
        self.before = self.snapshot()
        self.soaks = []
        processes = self.before["compute_processes"]
        if (self.before["gpu"]["status"] or processes["status"] or processes["stdout"]
                or self.before["uptime"]["status"]):
            raise ValueError("RTX 4070 GPU availability could not be verified")

    def snapshot(self):
        return {
            "time_utc": utc(), "uptime": command("uptime"), "gpu": command(self.smi),
            "compute_processes": command(self.smi, "--query-compute-apps=pid",
                                         "--format=csv,noheader,nounits"),
        }

    def stabilize(self, warmup, telemetry, has_operations):
        if not has_operations:
            return
        start = time.monotonic()
        while time.monotonic() - start < RTX_WARMUP_SECONDS:
            warmup()
        self.soaks.append({"elapsed_seconds": time.monotonic() - start})

    def provenance(self):
        return {"before": self.before, "after": self.snapshot(), "thermal_soaks": self.soaks}


def capture_session(device, policy, trace):
    if device == "gb10":
        return SparkSession(policy, trace)
    return Rtx4070Session(policy)


def boundary(directory, device):
    manifest, digest, source = inputs("gemm", device)
    policy, specs, profile_sha = load_policy(device)
    base = directory / "gemm"
    output = base / "wave-boundary.json"
    if output.exists():
        raise FileExistsError(output)
    spark = capture_session(device, policy, base / "wave-boundary-untimed.json")
    ref = manifest["devices"][device]["wave_reference"]
    key = ref["config_id"]
    records = {}
    for case in manifest["devices"][device]["cases"]:
        if "reference_ctas" not in case:
            continue
        config = next(cfg for cfg in manifest["configs"] if cfg["config_id"] == key)
        archive, aot_sha = read_aot(base, case, digest, runner.dense_gemm, source)
        _, rows = runner.prepare_candidates(case, [config], manifest["input_seed"], DEVICE_ARCH[device])
        attention_runner.verify_artifacts(archive, rows)
        jit = rows[key]
        if jit["status"] != "ok":
            raise ValueError(f"Untimed wave reference failed for {case['case_id']}: {jit['status']}")
        ctas = math.prod(jit["grid"])
        resident = residency(jit, specs)
        capacity = specs["sm_count"] * resident
        records[case["case_id"]] = {
            "aot_sha256": aot_sha, "ctas": ctas, "resident_ctas_per_sm": resident,
            "full_wave_capacity": capacity, "ctas_relative_to_capacity": ctas - capacity,
            "registers_per_thread": jit["registers_per_thread"], "shared_bytes": jit["shared_bytes"],
        }
    attention_runner.write_json(output, {"timestamp_utc": utc(), "source": source,
                                        "manifest_sha256": digest, "profile_sha256": profile_sha,
                                        "environment": spark.provenance(), "cases": records})
    print({name: (row["ctas"], row["full_wave_capacity"]) for name, row in records.items()})


def config_status_counts(report):
    # Counts only: a receipt is public before the holdout opens.
    return dict(sorted(Counter(row["status"] for row in report["configs"].values()).items()))


def capture(directory, family, case_id, session, device):
    manifest, digest, source = inputs(family, device)
    case, configs = case_configs(manifest, case_id, device)
    policy, _, profile_sha = load_policy(device)
    base = directory / family
    if family == "gemm":
        wave = json.loads((base / "wave-boundary.json").read_bytes())
        if (wave["source"] != source or wave["manifest_sha256"] != digest
                or wave["profile_sha256"] != profile_sha or len(wave["cases"]) != 3):
            raise ValueError("GEMM requires the untimed capture-resource wave check")
    is_attention = family == "attention"
    kernel = attention.fused_attention if is_attention else runner.dense_gemm
    prepare = partial(attention.prepare_candidates, arch=DEVICE_ARCH[device]) if is_attention else None
    archive, aot_sha = read_aot(base, case, digest, kernel, source)
    if is_attention:
        case = dict(case, shape=[case[key] for key in attention.SHAPE_FIELDS])
    name = f"{case_id}-session-{session}.json"
    sealed, receipt, failure, trace = (
        base / folder / name for folder in ("sealed", "receipts", "failures", "thermal-timeouts")
    )
    for path in (sealed, receipt, failure, trace):
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(mode=0o700, exist_ok=True)
    spark = capture_session(device, policy, trace)
    args = SimpleNamespace(cache="both", blocks=10, seed=100 + session,
                           warmup_seconds=60 if device == "gb10" else RTX_WARMUP_SECONDS)
    try:
        report = runner.measure(
            case, configs, manifest["input_seed"], args, digest,
            verify_artifacts=partial(attention_runner.verify_artifacts, archive), device_arch=DEVICE_ARCH[device],
            kernel=kernel, spark=spark, prepare=prepare, retry_rejected_blocks=False,
        )
    except (RuntimeError, ValueError) as error:
        details = {"timestamp_utc": utc(), "status": "protocol_failed", "reason": str(error)}
        if trace.exists():
            details["thermal_trace_sha256"] = sha256(trace.read_bytes())
        attention_runner.write_json(failure, details)
        raise
    if device == "rtx4070":
        report["rtx4070"] = report.pop("spark")
    environment = report["spark" if device == "gb10" else "rtx4070"]
    after = environment["after"]["compute_processes"]
    other_compute = set(after["stdout"].split()) - {str(os.getpid())}
    valid = report["complete_blocks"] == 10 and after["status"] == 0 and not other_compute
    report.update(aot_sha256=aot_sha, kernel_key=kernel.cache_key,
                  profile_sha256=profile_sha, collection_status="complete" if valid else "protocol_failed")
    receipt_data = seal_report(sealed, report)
    receipt_data.update(status=report["collection_status"], complete_blocks=report["complete_blocks"],
                        config_status_counts=config_status_counts(report))
    attention_runner.write_json(receipt, receipt_data)
    if not valid:
        raise RuntimeError(f"Case-session failed the capture protocol. See {receipt.name}")
    print(f"{case_id} session {session}: sealed", flush=True)


def audit(directory, family, output, device):
    manifest, digest, source = inputs(family, device)
    base = directory / family
    _, _, profile_sha = load_policy(device)
    reports, failures, aot = {}, {}, {}
    expected = set()
    for case in manifest["devices"][device]["cases"]:
        case_id = case["case_id"]
        kernel = runner.dense_gemm if family == "gemm" else attention.fused_attention
        _, aot[case_id] = read_aot(base, case, digest, kernel, source)
        for session in (1, 2, 3):
            name = f"{case_id}-session-{session}.json"
            expected.add(name)
            path = base / "sealed" / name
            receipt = base / "receipts" / name
            failure = base / "failures" / name
            if receipt.exists():
                if path.is_symlink() or stat.S_IMODE(path.stat().st_mode) != 0:
                    raise ValueError(f"{name}: report is not sealed")
                reports[name] = json.loads(receipt.read_bytes())
            elif failure.exists():
                failures[name] = json.loads(failure.read_bytes())
    record = {"timestamp_utc": utc(), "source": source,
              "device": "GB10" if device == "gb10" else "RTX 4070", "family": family,
              "preregistration_commit": APPROVAL, "manifest_sha256": digest,
              "profile_sha256": profile_sha, "aot_sha256_by_case": aot, "reports": reports,
              "protocol_failures_without_report": failures,
              "pending": sorted(expected - reports.keys() - failures.keys())}
    if family == "gemm":
        record["wave_boundary_sha256"] = sha256((base / "wave-boundary.json").read_bytes())
    attention_runner.write_json(output, record)
    print(f"{family}: {len(reports)} sealed reports, {len(failures)} failures, {len(record['pending'])} pending")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=DEVICE_ARCH, default="gb10")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("aot", "boundary", "capture", "audit"):
        cmd = commands.add_parser(name)
        cmd.add_argument("directory", type=Path)
        if name != "boundary":
            cmd.add_argument("family", choices=("gemm", "attention"))
        if name in ("aot", "capture"):
            cmd.add_argument("case")
        if name == "capture":
            cmd.add_argument("session", type=int, choices=(1, 2, 3))
        if name == "audit":
            cmd.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.device == "rtx4070" and getattr(args, "family", "gemm") == "attention":
        parser.error("RTX 4070 attention already uses the original sealed reports")
    if args.command == "aot":
        compile_case(args.directory, args.family, args.case, args.device)
    elif args.command == "boundary":
        boundary(args.directory, args.device)
    elif args.command == "capture":
        capture(args.directory, args.family, args.case, args.session, args.device)
    else:
        audit(args.directory, args.family, args.output, args.device)


if __name__ == "__main__":
    main()
