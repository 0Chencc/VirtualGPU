"""Compile the attention suite offline or collect an RTX 4070 case-session."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import triton
from triton.runtime.errors import PTXASError
from virtualgpu.frontend.compiler import (
    ARTIFACT_HASH_VERSION, MockTensor, compile_triton_aot, specialize_triton_call,
)

from validation.attention import SHAPE_FIELDS, fused_attention, launch_spec, prepare_candidates
from validation.holdout import seal_report
from validation.provenance import source_info
from validation.score import check_artifact

# Triton's native binder caches dtype objects by identity for the process.
_AOT_POINTERS = tuple(MockTensor(dtype) for dtype in ("f16", "f16", "f16", "f16", "f32"))
_EXPECTED_POINTER_TYPES = {"Q": "*fp16", "K": "*fp16", "V": "*fp16", "O": "*fp16", "LSE": "*fp32"}


def compile_artifact(case, config, target):
    grid, constants, options = launch_spec(case, config)
    parsed, signature, constexprs, attrs = specialize_triton_call(
        fused_attention, target, (*_AOT_POINTERS, *constants), options,
    )
    if any(signature.get(key) != dtype for key, dtype in _EXPECTED_POINTER_TYPES.items()):
        raise ValueError("Attention AOT pointer types do not match the JIT input types")
    artifact = compile_triton_aot(
        fused_attention, signature, constexprs, target,
        options["num_warps"], options["num_stages"], attrs=attrs, options=parsed,
    )
    # Store the resource and toolchain fields used by the shared artifact check.
    return {
        "status": "ok",
        "metadata": {
            "config": {**options, "ptx_options": parsed["ptx_options"]},
            "toolchain": {
                "compiler": "triton",
                "compiler_version": artifact.triton_version,
                "assembler": artifact.ptxas.assembler,
                "assembler_version": artifact.ptxas.assembler_version,
                "target_sm_compiled": artifact.ptxas.target_sm,
                "artifact_hash_version": ARTIFACT_HASH_VERSION,
                "ptx_sha256": artifact.ptx_sha256,
                "ttgir_sha256": artifact.ttgir_sha256,
            },
        },
        "resources": {"registers_per_thread": artifact.ptxas.registers_per_thread,
                      "static_smem_bytes": artifact.static_smem_bytes},
        "launch": {"grid": list(grid), "dynamic_smem_bytes": 0},
    }


def write_json(path, record):
    with path.open("x") as output:
        json.dump(record, output, indent=2, allow_nan=False)
        output.write("\n")


def compile_case(case, configs, manifest, identity, output):
    rows = {}
    for config in configs:
        try:
            row = compile_artifact(case, config, manifest["target"])
        except (triton.CompilationError, PTXASError, RuntimeError) as error:
            row = {"status": "compile_failed", "reason": str(error)}
        rows[config["config_id"]] = row
    write_json(output, {**identity, "source": source_info(), "configs": rows})
    return Counter(row["status"] for row in rows.values())


def measure_case(args, case, configs, manifest, identity):
    from validation.runner import measure

    archive_bytes = args.aot.read_bytes()
    archive = json.loads(archive_bytes)
    if any(archive.get(key) != value for key, value in identity.items()):
        raise ValueError("AOT archive does not match the manifest, kernel, or case")
    if set(archive["configs"]) != set(case["config_ids"]):
        raise ValueError("AOT archive must cover every declared config")

    def verify(artifacts):
        for key, jit in artifacts.items():
            if jit["status"] != "ok":
                continue
            aot = archive["configs"][key]
            if aot["status"] != "ok":
                jit.update(status="aot_compile_failed", reason=aot["reason"])
                continue
            match = check_artifact(jit, aot)
            jit["aot_check"] = match
            if match["status"] != "matched" or not match["ttgir_match"]:
                jit.update(status="artifact_mismatch", reason="AOT/JIT artifact check failed")

    run = SimpleNamespace(cache="both", blocks=10, seed=100 + args.session, warmup_seconds=2)
    report = measure(
        dict(case, shape=[case[key] for key in SHAPE_FIELDS]), configs, manifest["input_seed"], run,
        identity["manifest_sha256"], verify_artifacts=verify, kernel=fused_attention,
        prepare=prepare_candidates,
    )
    report.update(workload={key: case[key] for key in SHAPE_FIELDS}, kernel_key=identity["kernel_key"],
                  aot_sha256=sha256(archive_bytes).hexdigest())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    aot = commands.add_parser("aot")
    aot.add_argument("--config", action="append", help="Limit the offline compilation check")
    capture = commands.add_parser("capture")
    capture.add_argument("--aot", type=Path, required=True)
    capture.add_argument("--session", type=int, choices=(1, 2, 3), required=True)
    capture.add_argument("--sealed", action="store_true")
    for command in (aot, capture):
        command.add_argument("manifest", type=Path)
        command.add_argument("case")
        command.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw)
    schema = {"schema_version": 1, "kernel": "fused_attention_forward",
              "target": "sm_89", "triton_version": "3.9.0", "input_dtype": "float16",
              "output_dtype": "float16", "accumulator_dtype": "float32", "layout": "BHSD",
              "enable_fp_fusion": True, "warp_specialize": False, "softmax_scale": "1/sqrt(head_dim)"}
    if (any(manifest.get(key) != value for key, value in schema.items())
            or triton.__version__ != schema["triton_version"]):
        parser.error("Expected the pinned Phase 4 attention suite and Triton version")
    case = next((row for row in manifest["cases"] if row["case_id"] == args.case), None)
    if case is None:
        parser.error("Unknown attention case")
    selected = set(args.config or case["config_ids"]) if args.command == "aot" else set(case["config_ids"])
    if not selected <= set(case["config_ids"]):
        parser.error("Requested config is outside the case")
    configs = [config for config in manifest["configs"] if config["config_id"] in selected]
    if (len(configs) != len(selected) or {cfg["config_id"] for cfg in configs} != selected
            or case["split"] not in ("development", "holdout") or case["kv_heads"] <= 0
            or case["query_heads"] % case["kv_heads"]):
        parser.error("Invalid attention case or candidate list")
    identity = {"manifest_sha256": sha256(raw).hexdigest(), "case_id": args.case,
                "kernel_key": fused_attention.cache_key}
    if args.output.exists():
        parser.error("Output already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "aot":
        print(dict(compile_case(case, configs, manifest, identity, args.output)))
        return
    if args.sealed != (case["split"] == "holdout"):
        parser.error("Holdout labels require --sealed, and development labels use open capture")
    receipt_path = args.output.with_suffix(".receipt.json")
    if args.sealed and receipt_path.exists():
        parser.error("Seal receipt already exists")
    report = measure_case(args, case, configs, manifest, identity)
    if args.sealed:
        write_json(receipt_path, seal_report(args.output, report))
    else:
        write_json(args.output, report)
    print(dict(Counter(row["status"] for row in report["configs"].values())))


if __name__ == "__main__":
    main()
