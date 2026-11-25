"""Capture preregistered GB10 v0.2 holdouts without releasing their labels."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import partial
import json
import math
import os
from pathlib import Path
import stat
import subprocess
from types import SimpleNamespace

import triton
from virtualgpu.frontend.compiler import MockTensor, specialize_triton_call

from validation import attention, attention_runner, runner
from validation.baseline import residency
from validation.holdout import seal_report, sha256
from validation.phase2_capture import SparkSession
from validation.provenance import REPO_ROOT, source_info

APPROVAL = "29a6ac2"
TARGET = "sm_121"
ARCH = 121
GEMM_POINTERS = tuple(MockTensor("f16") for _ in range(3))


def utc():
    return datetime.now(timezone.utc).isoformat()


def inputs(family):
    subprocess.run(["git", "merge-base", "--is-ancestor", APPROVAL, "origin/main"],
                   cwd=REPO_ROOT, check=True)
    relative = f"validation/v02-{family}-holdout.json"
    raw = (REPO_ROOT / relative).read_bytes()
    published = subprocess.check_output(["git", "show", f"{APPROVAL}:{relative}"], cwd=REPO_ROOT)
    if published != raw:
        raise ValueError("Manifest differs from the approved GB10 matrix")
    manifest = json.loads(raw)
    source = source_info()
    if (source["python_dirty"] or source["validation_dirty"] or triton.__version__ != "3.9.0"
            or manifest["devices"]["gb10"]["target"] != TARGET):
        raise ValueError("Use a clean capture checkout and Triton 3.9.0 for GB10")
    return manifest, sha256(raw), source


def case_configs(manifest, case_id):
    case = next(row for row in manifest["devices"]["gb10"]["cases"] if row["case_id"] == case_id)
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


def compile_case(directory, family, case_id):
    from triton.runtime.errors import PTXASError

    manifest, digest, source = inputs(family)
    case, configs = case_configs(manifest, case_id)
    kernel = runner.dense_gemm if family == "gemm" else attention.fused_attention
    compile_artifact = gemm_artifact if family == "gemm" else attention_runner.compile_artifact
    rows = {}
    for config in configs:
        try:
            row = compile_artifact(case, config, TARGET)
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


def load_policy():
    raw = (REPO_ROOT / "profiles" / "gb10.json").read_bytes()
    profile = json.loads(raw)
    fields = profile["provenance"]
    if profile["identity"]["arch"] != TARGET or fields["clock_policy"]["locked"]:
        raise ValueError("GB10 capture requires the record-only profile")
    policy = {"l2_bytes": profile["specs"]["l2_size_bytes"], "clock_policy": "record_only",
              "driver_version": fields["driver_version"],
              "power_mode": {"unavailable": "nvpmodel is not installed"}}
    return policy, profile["specs"], sha256(raw)


def boundary(directory):
    manifest, digest, source = inputs("gemm")
    policy, specs, profile_sha = load_policy()
    base = directory / "gemm"
    output = base / "wave-boundary.json"
    if output.exists():
        raise FileExistsError(output)
    spark = SparkSession(policy, base / "wave-boundary-untimed.json")
    ref = manifest["devices"]["gb10"]["wave_reference"]
    key = ref["config_id"]
    records = {}
    for case in manifest["devices"]["gb10"]["cases"]:
        if "reference_ctas" not in case:
            continue
        config = next(cfg for cfg in manifest["configs"] if cfg["config_id"] == key)
        archive, aot_sha = read_aot(base, case, digest, runner.dense_gemm, source)
        _, rows = runner.prepare_candidates(case, [config], manifest["input_seed"], ARCH)
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


def capture(directory, family, case_id, session):
    manifest, digest, source = inputs(family)
    case, configs = case_configs(manifest, case_id)
    policy, _, profile_sha = load_policy()
    base = directory / family
    if family == "gemm":
        wave = json.loads((base / "wave-boundary.json").read_bytes())
        if (wave["source"] != source or wave["manifest_sha256"] != digest
                or wave["profile_sha256"] != profile_sha or len(wave["cases"]) != 3):
            raise ValueError("GB10 GEMM requires the untimed capture-resource wave check")
    is_attention = family == "attention"
    kernel = attention.fused_attention if is_attention else runner.dense_gemm
    prepare = partial(attention.prepare_candidates, arch=ARCH) if is_attention else None
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
    spark = SparkSession(policy, trace)
    args = SimpleNamespace(cache="both", blocks=10, seed=100 + session, warmup_seconds=60)
    try:
        report = runner.measure(
            case, configs, manifest["input_seed"], args, digest,
            verify_artifacts=partial(attention_runner.verify_artifacts, archive), device_arch=ARCH,
            kernel=kernel, spark=spark, prepare=prepare, retry_rejected_blocks=False,
        )
    except (RuntimeError, ValueError) as error:
        details = {"timestamp_utc": utc(), "status": "protocol_failed", "reason": str(error)}
        if trace.exists():
            details["thermal_trace_sha256"] = sha256(trace.read_bytes())
        attention_runner.write_json(failure, details)
        raise
    after = report["spark"]["after"]["compute_processes"]
    other_compute = set(after["stdout"].split()) - {str(os.getpid())}
    valid = report["complete_blocks"] == 10 and after["status"] == 0 and not other_compute
    report.update(aot_sha256=aot_sha, kernel_key=kernel.cache_key,
                  profile_sha256=profile_sha, collection_status="complete" if valid else "protocol_failed")
    receipt_data = seal_report(sealed, report)
    receipt_data.update(status=report["collection_status"], complete_blocks=report["complete_blocks"])
    attention_runner.write_json(receipt, receipt_data)
    if not valid:
        raise RuntimeError(f"Case-session failed the capture protocol. See {receipt.name}")
    print(f"{case_id} session {session}: sealed", flush=True)


def audit(directory, family, output):
    manifest, digest, source = inputs(family)
    base = directory / family
    _, _, profile_sha = load_policy()
    reports, failures, aot = {}, {}, {}
    expected = set()
    for case in manifest["devices"]["gb10"]["cases"]:
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
    record = {"timestamp_utc": utc(), "source": source, "device": "GB10", "family": family,
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
    if args.command == "aot":
        compile_case(args.directory, args.family, args.case)
    elif args.command == "boundary":
        boundary(args.directory)
    elif args.command == "capture":
        capture(args.directory, args.family, args.case, args.session)
    else:
        audit(args.directory, args.family, args.output)


if __name__ == "__main__":
    main()
