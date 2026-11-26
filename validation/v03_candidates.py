"""Generate v0.3 candidate manifests and audit their resources with CPU AOT."""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEMM = {
    "rtx4070": (
        (8, 5376, 3712),
        (28, 10752, 6400),
        (417, 4867, 3203),
        (833, 9471, 4739),
        (60, 11584, 2816),
        (60, 11712, 2816),
        (60, 11840, 2816),
        (384, 5888, 3968),
        (1408, 7424, 5248),
    ),
    "gb10": (
        (6, 6912, 4480),
        (44, 13824, 7936),
        (673, 5381, 3717),
        (1217, 9983, 6911),
        (52, 12096, 3072),
        (52, 12224, 3072),
        (52, 12352, 3072),
        (640, 6400, 3456),
        (1920, 7936, 6016),
    ),
}
FAMILIES = (
    "decode-small",
    "decode-wide",
    "mask-mnk-small",
    "mask-mnk-wide",
    "wave-below",
    "wave-exact",
    "wave-above",
    "prefill-small",
    "prefill-wide",
)
ATTENTION = {
    "rtx4070": (
        (1, 16, 16, 640, 64, False),
        (2, 16, 16, 2688, 128, False),
        (2, 20, 20, 1217, 64, True),
        (1, 24, 24, 3840, 128, True),
        (2, 24, 6, 1347, 64, False),
        (1, 32, 4, 3328, 128, False),
        (1, 32, 8, 1795, 64, True),
        (2, 24, 6, 2305, 128, True),
    ),
    "gb10": (
        (1, 20, 20, 896, 64, False),
        (2, 12, 12, 2176, 128, False),
        (2, 24, 24, 1407, 64, True),
        (1, 28, 28, 4352, 128, True),
        (2, 28, 7, 1539, 64, False),
        (1, 36, 6, 2944, 128, False),
        (1, 48, 8, 1985, 64, True),
        (2, 32, 8, 2563, 128, True),
    ),
}
ATTENTION_FIELDS = ("batch", "query_heads", "kv_heads", "seq_len", "head_dim", "causal")
REFERENCE = "m64n128k32-w4-s4-g1"
TARGETS = {"rtx4070": "sm_89", "gb10": "sm_121"}


def read(path, expected=None):
    raw = (ROOT / path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"{path}: historical manifest digest changed")
    return json.loads(raw), digest


def history():
    phase1, _ = read("validation/holdout-seal.json")
    phase2, _ = read("validation/phase2-aot-seal.json")
    index, _ = read("results/phase2-gb10/aot-index.json", phase2["aot_index_sha256"])
    v02_seals = {device: read(f"validation/v02-{device}-gemm-seal.json")[0] for device in TARGETS}
    gb10_attention_seal, _ = read("validation/v02-gb10-attention-seal.json")
    v02_attention_sha = gb10_attention_seal["manifest_sha256"]
    v02_attention, _ = read("validation/v02-attention-holdout.json", v02_attention_sha)
    attention_seal, _ = read("validation/attention-holdout-seal.json")
    sources = {
        "gemm": [
            (
                "m1-and-initial-suite",
                "validation/suite.json",
                phase1["manifest_materialization"]["source_manifest_sha256"],
                None,
            ),
            (
                "rtx4070-m2-development",
                "results/m2/suite.json",
                phase2["source_manifests"]["primary"],
                None,
            ),
            (
                "rtx4070-phase1-workload-holdout",
                "results/holdout-sealed/manifest.json",
                phase1["manifest_sha256"],
                None,
            ),
            *(
                (
                    f"gb10-phase2-{group}",
                    f"results/phase2-gb10/{group}/manifest.json",
                    index["files"][f"{group}/manifest.json"],
                    None,
                )
                for group in ("primary", "secondary", "diagnostic")
            ),
            *(
                (
                    f"v02-{device}-gemm",
                    "validation/v02-gemm-holdout.json",
                    v02_seals[device]["manifest_sha256"],
                    device,
                )
                for device in TARGETS
            ),
        ],
        "attention": [
            (
                "rtx4070-phase4-development",
                "validation/attention-suite.json",
                attention_seal["manifest_sha256"],
                "development",
            ),
            (
                "rtx4070-phase4-v02-holdout",
                "validation/attention-suite.json",
                attention_seal["manifest_sha256"],
                "holdout",
            ),
            (
                "v02-gb10-attention",
                "validation/v02-attention-holdout.json",
                v02_attention_sha,
                "gb10",
            ),
        ],
    }
    result = {}
    for family, cohorts in sources.items():
        result[family] = {}
        for name, path, expected, selector in cohorts:
            data, digest = read(path, expected)
            if "devices" in data:
                cases = data["devices"][selector]["cases"]
            else:
                cases = [case for case in data["cases"] if selector in (None, case["split"])]
            dimensions = {
                case["case_id"]: (case["shape"] if family == "gemm" else case["seq_len"])
                for case in cases
            }
            result[family][name] = {
                "manifest": path,
                "manifest_sha256": digest,
                "cases": dimensions,
            }
    if set(v02_attention["devices"]["rtx4070"]["case_ids"]) != set(
        result["attention"]["rtx4070-phase4-v02-holdout"]["cases"]
    ):
        raise ValueError("v0.2 RTX 4070 attention case references do not match Phase 4")
    axes = [
        sorted(
            {shape[i] for source in result["gemm"].values() for shape in source["cases"].values()}
        )
        for i in range(3)
    ]
    lengths = sorted(
        {seq for source in result["attention"].values() for seq in source["cases"].values()}
    )
    return {
        **result,
        "excluded_gemm_axes": dict(zip(("M", "N", "K"), axes)),
        "excluded_attention_seq_len": lengths,
    }


def create_manifests(inventory, profiles):
    bases = {
        family: read(f"validation/v02-{family}-holdout.json")[0] for family in ("gemm", "attention")
    }
    manifests = {
        family: {**base, "status": "candidate", "devices": {}} for family, base in bases.items()
    }
    used_shapes, used_lengths = set(), set()
    for device, target in TARGETS.items():
        sm_count = profiles[device]["specs"]["sm_count"]
        cases = []
        for label, shape in zip(FAMILIES, GEMM[device], strict=True):
            if shape in used_shapes or any(
                value in inventory["excluded_gemm_axes"][axis]
                for axis, value in zip(("M", "N", "K"), shape)
            ):
                raise ValueError(
                    f"{device}: GEMM shape reuses a historical axis or candidate tuple: {shape}"
                )
            used_shapes.add(shape)
            row = {
                "case_id": f"v03-{device}-{label}",
                "split": "holdout",
                "shape": shape,
                "config_ids": [config["config_id"] for config in bases["gemm"]["configs"]],
            }
            if label.startswith("wave"):
                row["reference_ctas"] = (
                    2 * sm_count + {"wave-below": -1, "wave-exact": 0, "wave-above": 1}[label]
                )
            cases.append(row)
        manifests["gemm"]["devices"][device] = {
            "target": target,
            "wave_reference": {
                "config_id": REFERENCE,
                "sm_count": sm_count,
                "planning_resident_ctas_per_sm": 2,
                "planning_full_wave_ctas": 2 * sm_count,
            },
            "cases": cases,
        }
        cases, strata = [], set()
        for batch, hq, hk, seq, dim, causal in ATTENTION[device]:
            if seq in used_lengths or seq in inventory["excluded_attention_seq_len"]:
                raise ValueError(f"{device}: attention sequence length is reused: {seq}")
            used_lengths.add(seq)
            strata.add((hq == hk, dim, causal))
            case = dict(zip(ATTENTION_FIELDS, (batch, hq, hk, seq, dim, causal)))
            case_id = f"b{batch}-q{hq}-kv{hk}-s{seq}-d{dim}-{'causal' if causal else 'full'}"
            ids = [
                config["config_id"]
                for config in bases["attention"]["configs"]
                if config["block_m"] <= seq
                and config["block_n"] <= dim
                and (not causal or config["block_n"] <= config["block_m"])
            ]
            cases.append({"case_id": case_id, "split": "holdout", **case, "config_ids": ids})
        if len(strata) != 8:
            raise ValueError(f"{device}: attention strata are incomplete")
        manifests["attention"]["devices"][device] = {"target": target, "cases": cases}
    return manifests


def inspect_case(task):
    import triton
    from triton.runtime.errors import PTXASError

    from validation.attention_runner import compile_artifact
    from validation.baseline import residency
    from validation.v02_capture import gemm_artifact

    family, target, case, configs, specs = task
    compile_case = gemm_artifact if family == "gemm" else compile_artifact
    records = {}
    for key in case["config_ids"]:
        config = configs[key]
        try:
            item = compile_case(case, config, target)
        except (triton.CompilationError, PTXASError, RuntimeError) as error:
            records[key] = {"status": "compile_failed", "reason": str(error)}
            continue
        registers = item["resources"]["registers_per_thread"]
        shared = item["resources"]["static_smem_bytes"] + item["launch"]["dynamic_smem_bytes"]
        warps = config["num_warps"]
        resident = residency(
            {"num_warps": warps, "registers_per_thread": registers, "shared_bytes": shared}, specs
        )
        limits = {
            "shared_per_cta": shared > specs["max_shared_mem_per_block_bytes"],
            "registers_per_thread": registers > specs["max_registers_per_thread"],
            "threads_per_cta": warps * specs["warp_size"] > specs["max_threads_per_block"],
            "residency": resident <= 0,
        }
        exceeded = [name for name, exceeded in limits.items() if exceeded]
        records[key] = {
            **item,
            "status": "resource_exceeded" if exceeded else "fits_profile",
            "exceeded": exceeded,
            "resident_ctas_per_sm": resident if not exceeded else 0,
        }
    return records


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Set CUDA_VISIBLE_DEVICES to an empty string for CPU-only AOT")
    import triton

    if triton.__version__ != "3.9.0" or triton.knobs.nvidia.disable_ptxas_opt:
        raise ValueError(
            "Use the capture toolchain: Triton 3.9.0 and default assembler optimization"
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    aot = args.output / "aot"
    aot.mkdir()
    inventory = history()
    inputs = {device: read(f"profiles/{device}.json") for device in TARGETS}
    profiles = {device: pair[0] for device, pair in inputs.items()}
    manifests = create_manifests(inventory, profiles)
    index = {
        "audit_class": "candidate_cpu_aot",
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "host_arch": platform.machine(),
        "triton_version": triton.__version__,
        "history_sha256": write(args.output / "v03-holdout-history.json", inventory),
        "manifests": {},
        "cohorts": {},
    }
    tasks, labels = [], []
    for family, manifest in manifests.items():
        index["manifests"][family] = write(args.output / f"v03-{family}-holdout.json", manifest)
        for device, cohort in manifest["devices"].items():
            configs = {config["config_id"]: config for config in manifest["configs"]}
            for case in cohort["cases"]:
                tasks.append((family, cohort["target"], case, configs, profiles[device]["specs"]))
                labels.append((family, device, case))
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for (family, device, case), records in zip(
            labels, pool.map(inspect_case, tasks), strict=True
        ):
            target = manifests[family]["devices"][device]["target"]
            cohort_name = f"{device}-{family}"
            cohort = index["cohorts"].setdefault(
                cohort_name,
                {
                    "target": target,
                    "profile_sha256": inputs[device][1],
                    "profile_revision": profiles[device]["provenance"]["profile_revision"],
                    "cases": {},
                },
            )
            filename = f"{cohort_name}-{case['case_id']}.json"
            row = {
                "aot_file": filename,
                "aot_sha256": write(aot / filename, records),
                "status_counts": dict(Counter(record["status"] for record in records.values())),
                "invalid": {
                    key: record.get("exceeded", record.get("reason"))
                    for key, record in records.items()
                    if record["status"] != "fits_profile"
                },
            }
            if "reference_ctas" in case:
                ref = records[REFERENCE]
                ctas = math.prod(ref["launch"]["grid"])
                capacity = profiles[device]["specs"]["sm_count"] * ref["resident_ctas_per_sm"]
                expected_capacity = manifests[family]["devices"][device]["wave_reference"][
                    "planning_full_wave_ctas"
                ]
                if (
                    ref["status"] != "fits_profile"
                    or ctas != case["reference_ctas"]
                    or capacity != expected_capacity
                ):
                    raise ValueError(
                        f"{case['case_id']}: AOT changes planned wave geometry: {ctas}/{capacity}"
                    )
                row["wave_reference"] = {
                    "config_id": REFERENCE,
                    "ctas": ctas,
                    "full_wave_capacity": capacity,
                    "resident_ctas_per_sm": ref["resident_ctas_per_sm"],
                    "resources": ref["resources"],
                }
            cohort["cases"][case["case_id"]] = row
            print(cohort_name, case["case_id"], row["status_counts"], flush=True)
    write(args.output / "v03-holdout-aot-audit.json", index)


if __name__ == "__main__":
    main()
