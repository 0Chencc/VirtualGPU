"""Archive the v0.2 holdout predictions on CPU without opening timing labels."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import triton
from triton.runtime.errors import PTXASError

from virtualgpu.frontend import extract_attention_kernel_ir, extract_gemm_kernel_ir

from validation import attention, attention_baseline, baseline, neusight
from validation.attention_runner import _AOT_POINTERS
from validation.holdout import sha256
from validation.provenance import REPO_ROOT, require_formal_build, source_info
from validation.runner import dense_gemm
from validation.score import PRACTICAL_MARGIN, check_artifact
from validation.v02_capture import GEMM_POINTERS
from validation.v02_inputs import COHORTS, DRAWS, SEED, TAG, TARGETS, case_inputs, git


def require_imports_from_checkout(*modules):
    source = (REPO_ROOT / "python").resolve()
    for module in modules:
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise ValueError(f"{module.__name__} is imported from {module.__file__}, not this checkout")


def require_frozen_checkout(core):
    import virtualgpu
    import virtualgpu.frontend

    require_imports_from_checkout(virtualgpu, virtualgpu.frontend, core)
    revision = git("rev-parse", "--verify", f"refs/tags/{TAG}^{{commit}}").decode().strip()
    tag = git("rev-parse", "--verify", f"refs/tags/{TAG}").decode().strip()
    subprocess.run(["git", "merge-base", "--is-ancestor", revision, "origin/main"],
                   cwd=REPO_ROOT, check=True)
    if source_info()["commit"] != revision or git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError(f"Predictions require the clean {TAG} checkout")
    if triton.__version__ != "3.9.0" or triton.knobs.nvidia.disable_ptxas_opt:
        raise ValueError("The archive requires Triton 3.9.0 with the captured assembler settings")
    build = core.build_info()
    require_formal_build(build, revision, "Loaded binary")
    return revision, tag, build


def load_fits(raw, expected_digest, revision, profile_digests):
    if sha256(raw) != expected_digest:
        raise ValueError("Frozen fit digest mismatch")
    data = json.loads(raw)
    if data["core_revision"] != revision or data["profile_sha256"] != profile_digests:
        raise ValueError("Frozen fits differ from the core or profile revisions")
    fits = {}
    for family, width in (("gemm", 13), ("attention", 14)):
        row = data["fits"][family]
        training = row["training_inputs_sha256"]
        if len(training) != 64 or any(ch not in "0123456789abcdef" for ch in training):
            raise ValueError(f"{family}: expected a training input digest")
        center, scale, weights = (np.asarray(row[key], dtype=float)
                                  for key in ("center", "scale", "weights"))
        intercept = float(row["intercept"])
        if (any(arr.shape != (width,) or not np.isfinite(arr).all()
                for arr in (center, scale, weights)) or not (scale > 0).all()
                or not math.isfinite(intercept)):
            raise ValueError(f"{family}: invalid frozen fit")
        fits[family] = (neusight.EfficiencyFit(center, scale, weights, intercept), training)
    return fits


def read_capture_artifact(directory, case, seal, manifest_digest, kernel):
    case_id = case["case_id"]
    raw = (directory / f"{case_id}.json").read_bytes()
    if sha256(raw) != seal["aot_sha256_by_case"][case_id]:
        raise ValueError(f"{case_id}: capture AOT archive differs from the seal")
    capture = json.loads(raw)
    if (capture["case_id"] != case_id or capture["manifest_sha256"] != manifest_digest
            or capture["kernel_key"] != kernel.cache_key
            or set(capture["configs"]) != set(case["config_ids"])):
        raise ValueError(f"{case_id}: capture AOT metadata does not match the frozen inputs")
    return capture["configs"]


def frozen_ir(family, case, config, target):
    # No KernelIR or AOT cache: their keys do not identify the frontend
    # source, so an entry written by another checkout would be returned.
    if family == "gemm":
        m, n, k = case["shape"]
        bm, bn, bk = config["tile"]
        knobs = dict(BM=bm, BN=bn, BK=bk, GROUP_M=config["group_m"],
                     num_warps=config["num_warps"], num_stages=config["num_stages"])
        return extract_gemm_kernel_ir(dense_gemm, M=m, N=n, K=k, config=knobs, target_arch=target,
                                      call_args=(*GEMM_POINTERS, m, n, k), use_cache=False)
    _, constants, options = attention.launch_spec(case, config)
    hq, hk, seq, dim, causal, scale, bm, bn = constants
    # The frontend supplies CAUSAL by keyword. Keep the identical JIT
    # specialization while passing the other constexprs by name.
    knobs = dict(num_warps=options["num_warps"], num_stages=options["num_stages"],
                 BM=bm, BN=bn, SCALE_LOG2=scale)
    return extract_attention_kernel_ir(
        attention.fused_attention, case["batch"], hq, seq, dim, knobs, target,
        H_kv=hk, causal=causal, call_args=(*_AOT_POINTERS, hq, hk, seq, dim), use_cache=False,
    )


def artifact_match(captured, ir):
    toolchain = ir["metadata"]["toolchain"]
    resources = ir["resources"]
    launch = ir["launch"]
    context = {"grid": launch["grid"], "registers_per_thread": resources["registers_per_thread"],
               "shared_bytes": resources["static_smem_bytes"] + launch["dynamic_smem_bytes"]}
    actual = {**context, **toolchain, "disable_ptxas_opt": False,
              "ptx_options": ir["metadata"]["config"].get("ptx_options") or None}
    match = check_artifact(actual, captured)
    mma_shapes = sorted({"x".join(map(str, op["inst_mnk"]))
                         for op in ir["body"]["pipeline_loop"]["ops"] if op["kind"] == "mma"})
    return match, {**context, "mma_shapes": mma_shapes}


def core_prediction(core, ir_text, profile_text, cache, revision):
    result = core.predict(ir_text, profile_text, l2_state=cache, draws=DRAWS, seed=SEED)
    require_formal_build(result.get("build"), revision, "Core prediction")
    if result["status"] == "ok":
        nominal = result["time_ns"]
        draws = result["draws"]
        if (set(draws) != {str(i) for i in range(DRAWS + 1)} or draws["0"] != nominal
                or any(not math.isfinite(value) or value <= 0 for value in draws.values())):
            raise ValueError("Unexpected core perturbation draws")
        result["nominal_latency_ns"] = nominal
    return result


def baseline_predictions(family, case, config, profile, fit, cache):
    if family == "gemm":
        key = config["config_id"]
        roof = baseline.predict({"case_id": case["case_id"], "shape": case["shape"],
                                 "gpu": {"arch": profile["identity"]["arch"]},
                                 "configs": {key: dict(config, status="ok")}}, profile)[0]
        features, workload = neusight.tile_features, case["shape"]
    else:
        try:
            roof = {"status": "ok", "nominal_latency_ns": attention_baseline.roofline(case, config, profile)}
        except ValueError as error:
            roof = {"status": "unsupported", "reason": str(error)}
        features, workload = attention_baseline.tile_features, case
    try:
        tile = features(workload, config, profile, cache)
        learned = {"status": "ok", "nominal_latency_ns": fit.predict_ns(tile)}
    except ValueError as error:
        learned = {"status": "unsupported", "reason": str(error)}
    return roof, learned


def predict_case(name, case, configs, captured, target, profile_raw, profile, fit, build, revision, core, output):
    family = name.split("-")[1]
    methods = {cache: {name: {} for name in ("core", "roofline_wave", "neusight_style")}
               for cache in ("warm", "cold")}
    index = {}
    for key in case["config_ids"]:
        artifact = captured[key]
        ir_text = None
        if artifact["status"] != "ok":
            row = {"status": "failed", "reason": artifact["reason"], "stage": "capture_aot"}
        else:
            try:
                ir = frozen_ir(family, case, configs[key], target)
            except (triton.CompilationError, PTXASError, ValueError, RuntimeError) as error:
                row = {"status": "failed", "reason": str(error), "stage": "frozen_frontend"}
            else:
                match, resources = artifact_match(artifact, ir)
                row = {"status": "ok", "artifact_check": match}
                if match["status"] != "matched" or not match["ttgir_match"]:
                    row.update(status="failed", reason="Frozen artifact differs from capture AOT",
                               stage="artifact_match")
                formatted = json.dumps(ir, sort_keys=True, separators=(",", ":"), allow_nan=False)
                path = Path("ir") / name / case["case_id"] / f"{key}.json"
                destination = output / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(formatted, encoding="utf-8")
                row.update(ir=path.as_posix(), ir_sha256=sha256(formatted.encode()))
                if row["status"] == "ok":
                    ir_text = formatted
        index[key] = row
        for cache, records in methods.items():
            if ir_text is None:
                failure = {name: row[name] for name in ("status", "reason", "stage")}
                records["core"][key] = {**failure, "build": build}
                records["roofline_wave"][key] = dict(failure)
                records["neusight_style"][key] = dict(failure)
                continue
            records["core"][key] = core_prediction(core, ir_text, profile_raw, cache, revision)
            roof, learned = baseline_predictions(family, case, {**configs[key], **resources}, profile, fit, cache)
            records["roofline_wave"][key], records["neusight_style"][key] = roof, learned
    return methods, index


def write_json(path, data):
    raw = (json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    with path.open("xb") as output:
        output.write(raw)
    return sha256(raw)


def main():
    from virtualgpu import _core

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--fit-sha256", required=True)
    for name in COHORTS:
        parser.add_argument(f"--{name}-aot", type=Path, required=True)
    args = parser.parse_args()
    revision, tag, build = require_frozen_checkout(_core)
    if args.output.exists():
        raise FileExistsError(args.output)
    profile_paths = {device: f"profiles/{device}.json" for device in TARGETS}
    profiles = {device: git("show", f"{revision}:{path}") for device, path in profile_paths.items()}
    profile_digests = {device: sha256(raw) for device, raw in profiles.items()}
    fits = load_fits(args.fit.read_bytes(), args.fit_sha256, revision, profile_digests)
    cohorts = {name: case_inputs(name, revision) for name in COHORTS}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{args.output.name}-", dir=args.output.parent) as staging:
        temporary = Path(staging)
        status = Counter()
        artifact_index, inputs, result = {}, {}, {}
        for name, (cases, configs, seal, evidence) in cohorts.items():
            device, family = name.split("-")
            directory = getattr(args, name.replace("-", "_") + "_aot")
            profile_raw = profiles[device].decode()
            profile = json.loads(profile_raw)
            target = TARGETS[device]
            if profile["identity"]["arch"] != target:
                raise ValueError(f"{name}: profile architecture differs from the manifest")
            fit, training = fits[family]
            inputs[name] = {**evidence, "profile_sha256": profile_digests[device],
                            "profile_revision": profile["provenance"]["profile_revision"],
                            "training_inputs_sha256": training}
            result[name], artifact_index[name] = {}, {}
            kernel = dense_gemm if family == "gemm" else attention.fused_attention
            for case in cases:
                case_id = case["case_id"]
                captured = read_capture_artifact(directory, case, seal, evidence["manifest_sha256"], kernel)
                methods, matches = predict_case(name, case, configs, captured, target, profile_raw, profile,
                                                fit, build, revision, _core, temporary)
                result[name][case_id] = methods
                artifact_index[name][case_id] = {"capture_aot_sha256": seal["aot_sha256_by_case"][case_id],
                                                 "configs": matches}
                status.update(f"{name}:{method}:{row['status']}"
                              for cache in methods.values() for method, rows in cache.items()
                              for row in rows.values())
                print(f"{name}/{case_id}: archived", flush=True)
        index_digest = write_json(temporary / "aot-index.json", artifact_index)
        evidence = {"result_class": "formal", "tag": TAG, "tag_object": tag, "build": build,
                    "binary_sha256": sha256(Path(_core.__file__).read_bytes()),
                    "recorded_utc": datetime.now(timezone.utc).isoformat(),
                    "draws": DRAWS, "seed": SEED, "pair_margin": PRACTICAL_MARGIN,
                    "fit_sha256": args.fit_sha256, "aot_index_sha256": index_digest,
                    "cohort_inputs": inputs, "status_counts": dict(status)}
        evidence["prediction_archive_sha256"] = write_json(temporary / "predictions.json",
                                                           {**evidence, "cohorts": result})
        write_json(temporary / "prediction-evidence.json", evidence)
        temporary.rename(args.output)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
