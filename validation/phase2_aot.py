"""Compile and index the preregistered GB10 artifacts from phase1-final."""

import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path

from validation.holdout import sha256
from validation.provenance import REPO_ROOT, source_info

REVISION = "129a6f55d3dcbf5c261a8397296400f91f291fe1"
MANIFESTS = {
    "primary": ("development", 255, "cbfe410e95f327943f6f0cd59250f217b874cbb4654a3a161f7e52e3756698d4"),
    "secondary": ("holdout", 324, "b6c8fd089543ec8444e820993acc962f25eb3e662f9d47d3fe3bb79dbf03eaf1"),
}
TMA_CASE = "gb10-tma"
TMA_SOURCE = "python/virtualgpu/frontend/tests/test_triton_frontend.py"


def load_tma_kernel(root):
    import triton
    import triton.language as tl

    # Load the exact pinned function without importing the frontend test suite.
    path = root / TMA_SOURCE
    module = ast.parse(path.read_text(), filename=str(path))
    function = next(node for node in module.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_matmul_tma_kernel")
    scope = {"triton": triton, "tl": tl, "__name__": __name__}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
    return scope[function.name]


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def compile_cohort(output, name, manifest, kernel, source):
    import triton
    from triton.runtime.errors import PTXASError
    from virtualgpu.frontend import MockTensor, extract_gemm_kernel_ir

    configs = {row["config_id"]: row for row in manifest["configs"]}
    for case in manifest["cases"]:
        directory = output / name / "aot" / case["case_id"]
        directory.mkdir(parents=True)
        m, n, k = case["shape"]
        failures = {}
        for key in case["config_ids"]:
            cfg = configs[key]
            bm, bn, bk = cfg["tile"]
            knobs = (dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk)
                     if name == "diagnostic" else dict(BM=bm, BN=bn, BK=bk, GROUP_M=cfg["group_m"]))
            knobs.update(num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
            try:
                ir = extract_gemm_kernel_ir(
                    kernel, M=m, N=n, K=k, config=knobs, target_arch="sm_121",
                    call_args=(*(MockTensor("f16") for _ in range(3)), m, n, k),
                    enable_fp_fusion=True,
                )
            except (triton.CompilationError, PTXASError, RuntimeError, ValueError) as error:
                failures[key] = {"error_type": type(error).__name__, "reason": str(error)}
                continue
            ir["metadata"]["toolchain"]["source"] = source
            write_json(directory / f"{key}.json", ir)
        write_json(directory / "failures.json", failures)
        print(f"{name}/{case['case_id']}: {len(case['config_ids']) - len(failures)} artifacts, "
              f"{len(failures)} failures", flush=True)


def main():
    import triton
    import virtualgpu.frontend.compiler as compiler
    from validation import runner

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary", type=Path)
    parser.add_argument("secondary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = source_info()
    if (source["commit"] != REVISION or source["python_dirty"] or source["validation_dirty"]
            or not Path(compiler.__file__).resolve().is_relative_to(REPO_ROOT / "python")
            or not Path(runner.__file__).resolve().is_relative_to(REPO_ROOT / "validation")
            or triton.__version__ != "3.9.0"):
        raise ValueError("GB10 AOT requires the clean phase1-final frontend and Triton 3.9.0")

    suites = {}
    for name, (split, count, digest) in MANIFESTS.items():
        raw = getattr(args, name).read_bytes()
        if sha256(raw) != digest:
            raise ValueError(f"{name}: source manifest digest mismatch")
        suite = json.loads(raw)
        suite["cases"] = [row for row in suite["cases"] if row["split"] == split]
        if len(suite["cases"]) != 9 or sum(len(row["config_ids"]) for row in suite["cases"]) != count:
            raise ValueError(f"{name}: unexpected candidate matrix")
        suites[name] = suite

    configs = [{"config_id": f"m128n128k32-w4-s{stage}-g1", "tile": [128, 128, 32],
                "group_m": 1, "num_warps": 4, "num_stages": stage} for stage in (2, 3, 4)]
    suites["diagnostic"] = {
        "input_seed": 17,
        "configs": configs,
        "cases": [{"case_id": TMA_CASE, "split": "diagnostic", "shape": [1024, 1024, 512],
                   "config_ids": [cfg["config_id"] for cfg in configs]}],
    }
    args.output.mkdir(mode=0o700)
    for name, suite in suites.items():
        kernel = load_tma_kernel(REPO_ROOT) if name == "diagnostic" else runner.dense_gemm
        compile_cohort(args.output, name, suite, kernel, source)
        write_json(args.output / name / "manifest.json", suite)
    index = {path.relative_to(args.output).as_posix(): sha256(path.read_bytes())
             for path in sorted(args.output.rglob("*.json"))}
    write_json(args.output / "aot-index.json", {
        "source": source, "recorded_utc": datetime.now(timezone.utc).isoformat(), "files": index,
        "dense_runner_sha256": sha256(Path(runner.__file__).read_bytes()),
        "dense_kernel_sha256": sha256(runner.dense_gemm.src.encode()),
    })


if __name__ == "__main__":
    main()
