"""Compile development GEMM artifacts offline for AOT/JIT matching."""

import argparse
import json
from pathlib import Path

import triton
from triton.runtime.errors import PTXASError
from virtualgpu.frontend.compiler import MockTensor
from virtualgpu.frontend.triton_gemm import extract_gemm_kernel_ir

from validation.provenance import source_info
from validation.runner import dense_gemm


def compile_case(case, configs, output, source):
    output.mkdir(parents=True, exist_ok=True)
    m, n, k = case["shape"]
    failures = {}
    for key in case["config_ids"]:
        config = configs[key]
        bm, bn, bk = config["tile"]
        try:
            ir = extract_gemm_kernel_ir(
                dense_gemm,
                M=m, N=n, K=k,
                config=dict(BM=bm, BN=bn, BK=bk, GROUP_M=config["group_m"],
                            num_warps=config["num_warps"], num_stages=config["num_stages"]),
                target_arch="sm_89",
                call_args=(MockTensor("f16"), MockTensor("f16"), MockTensor("f16"), m, n, k),
            )
        except (triton.CompilationError, PTXASError, RuntimeError, ValueError) as error:
            failures[key] = {"error_type": type(error).__name__, "reason": str(error)}
            continue
        ir["metadata"]["toolchain"]["source"] = source
        (output / f"{key}.json").write_text(json.dumps(ir) + "\n")
    (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--case", action="append", help="Development case ID, repeatable")
    args = parser.parse_args()
    suite = json.loads(args.manifest.read_text())
    development = {case["case_id"]: case for case in suite["cases"]
                   if case["split"] == "development"}
    keys = args.case or list(development)
    if not set(keys) <= development.keys():
        parser.error("Only development cases are accepted")
    configs = {config["config_id"]: config for config in suite["configs"]}
    source = source_info()
    for key in keys:
        case = development[key]
        failures = compile_case(case, configs, args.output / key, source)
        print(f"{key}: {len(case['config_ids']) - len(failures)} artifacts, {len(failures)} failures",
              flush=True)


if __name__ == "__main__":
    main()
