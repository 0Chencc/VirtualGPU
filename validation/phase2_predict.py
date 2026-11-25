"""Archive formal GB10 predictions without accessing any timing labels."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
from virtualgpu import _core

from validation.baseline import predict as roofline
from validation.holdout import sha256
from validation.neusight import EfficiencyFit, tile_features
from validation.phase2_aot import REVISION, write_json
from validation.provenance import REPO_ROOT, require_formal_build, source_info

FIT_SHA256 = "20f7b3869b8e7e79fba98a05d91d38c6c305a3a3bdfe475bd7838111a264ed72"
COHORTS = ("primary", "secondary", "diagnostic")


def predict_cohort(cohort, directory, index, profile_text, profile, fit, build, counts):
    def frozen_artifact(path):
        raw = path.read_bytes()
        if sha256(raw) != index["files"][path.relative_to(directory).as_posix()]:
            raise ValueError(f"{path}: AOT index digest mismatch")
        return raw.decode()

    root = directory / cohort
    suite = json.loads(frozen_artifact(root / "manifest.json"))
    configs = {cfg["config_id"]: cfg for cfg in suite["configs"]}
    cases = {}
    for case in suite["cases"]:
        case_id = case["case_id"]
        aot = root / "aot" / case_id
        failures = json.loads(frozen_artifact(aot / "failures.json"))
        candidates, resources = {}, {}
        for key in case["config_ids"]:
            if key in failures:
                resources[key] = {"status": "failed", "reason": failures[key]["reason"]}
                continue
            raw = frozen_artifact(aot / f"{key}.json")
            ir = json.loads(raw)
            toolchain = ir["metadata"]["toolchain"]
            if (toolchain["source"] != index["source"] or toolchain["compiler_version"] != "3.9.0"
                    or toolchain["artifact_hash_version"] != "2"
                    or toolchain["target_sm_compiled"] not in ("sm_121", "sm_121a")
                    or toolchain["enable_fp_fusion"] is not True):
                raise ValueError(f"{cohort}/{case_id}/{key}: inconsistent AOT provenance")
            mma_shapes = sorted({"x".join(map(str, op["inst_mnk"]))
                                 for op in ir["body"]["pipeline_loop"]["ops"] if op["kind"] == "mma"})
            candidates[key] = raw
            resources[key] = dict(configs[key], status="ok", mma_shapes=mma_shapes,
                                  registers_per_thread=ir["resources"]["registers_per_thread"],
                                  shared_bytes=ir["resources"]["static_smem_bytes"]
                                  + ir["launch"]["dynamic_smem_bytes"])

        contexts = {"case_id": case_id, "shape": case["shape"], "gpu": {"arch": "sm_121"},
                    "configs": resources}
        fixed = {row["config_id"]: row for row in roofline(contexts, profile)} if cohort != "diagnostic" else {}
        cases[case_id] = {}
        for cache in ("warm", "cold"):
            core, trained = {}, {}
            for key, context in resources.items():
                if key in failures:
                    row = {**context, "build": build}
                else:
                    row = _core.predict(candidates[key], profile_text, l2_state=cache, draws=32, seed=17)
                require_formal_build(row.get("build"), REVISION, f"{cohort}/{case_id}/{cache}/{key}")
                if row["status"] == "ok":
                    row["nominal_latency_ns"] = row["time_ns"]
                    if set(row["draws"]) != {str(i) for i in range(33)} or row["draws"]["0"] != row["time_ns"]:
                        raise ValueError(f"{case_id}/{cache}/{key}: incorrect perturbation draws")
                core[key] = row
                if cohort == "diagnostic":
                    continue
                if key in failures or fixed[key]["status"] != "ok":
                    trained[key] = {"status": fixed[key]["status"], "reason": fixed[key]["reason"]}
                    continue
                latency = fit.predict_ns(tile_features(case["shape"], context, profile, cache))
                trained[key] = {"status": "ok", "nominal_latency_ns": latency}
            methods = {"core": core}
            if cohort != "diagnostic":
                methods.update(roofline_wave=fixed, neusight_style=trained)
            cases[case_id][cache] = methods
            counts.update(f"{cohort}:{method}:{row['status']}"
                          for method, records in methods.items() for row in records.values())
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("fit", type=Path)
    parser.add_argument("profile_commit")
    parser.add_argument("aot_seal_commit")
    args = parser.parse_args()
    if source_info() != {"commit": REVISION, "python_dirty": False, "validation_dirty": False}:
        raise ValueError("GB10 predictions require a clean phase1-final checkout")
    build = _core.build_info()
    require_formal_build(build, REVISION, "Loaded binary")
    profile_commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{args.profile_commit}^{{commit}}"], cwd=REPO_ROOT, text=True,
    ).strip()
    subprocess.run(["git", "merge-base", "--is-ancestor", profile_commit, "origin/main"],
                   cwd=REPO_ROOT, check=True)
    profile_bytes = subprocess.check_output(
        ["git", "show", f"{profile_commit}:profiles/gb10.json"], cwd=REPO_ROOT,
    )
    profile = json.loads(profile_bytes)
    if profile["identity"]["arch"] != "sm_121":
        raise ValueError("Expected the published sm_121 microbenchmark profile")
    profile_text = profile_bytes.decode()
    weights = args.fit.read_bytes()
    if sha256(weights) != FIT_SHA256:
        raise ValueError("The frozen RTX 4070 fit digest did not match")
    array = json.loads(weights)
    fit = EfficiencyFit(*(np.asarray(array[key]) for key in ("center", "scale", "weights")),
                        array["intercept"])
    index_bytes = (args.directory / "aot-index.json").read_bytes()
    seal_commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{args.aot_seal_commit}^{{commit}}"], cwd=REPO_ROOT, text=True,
    ).strip()
    subprocess.run(["git", "merge-base", "--is-ancestor", seal_commit, "origin/main"],
                   cwd=REPO_ROOT, check=True)
    seal = json.loads(subprocess.check_output(
        ["git", "show", f"{seal_commit}:validation/phase2-aot-seal.json"], cwd=REPO_ROOT,
    ))
    if sha256(index_bytes) != seal["aot_index_sha256"]:
        raise ValueError("GB10 AOT index does not match the published seal")
    index = json.loads(index_bytes)
    counts = Counter()
    cohorts = {name: predict_cohort(name, args.directory, index, profile_text, profile, fit, build, counts)
               for name in COHORTS}
    evidence = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "binary_sha256": sha256(Path(_core.__file__).read_bytes()),
        "aot_index_sha256": sha256(index_bytes),
        "profile_commit": profile_commit,
        "profile_revision": profile["provenance"]["profile_revision"],
        "profile_sha256": sha256(profile_bytes),
        "fit_sha256": FIT_SHA256,
        "status_counts": dict(counts),
    }
    archive = args.directory / "predictions.json"
    with archive.open("x") as stream:
        json.dump({"build": build, "draws": 32, "seed": 17, "pair_margin": 0.05,
                   **evidence, "cohorts": cohorts}, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
    evidence["prediction_archive_sha256"] = sha256(archive.read_bytes())
    write_json(args.directory / "prediction-evidence.json", evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
