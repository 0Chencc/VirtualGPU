"""Generate the versioned GEMM experiment manifest."""

import argparse
import itertools
import json
from pathlib import Path

TILES = ((16, 64, 32), (32, 64, 32), (64, 64, 32),
         (64, 128, 32), (128, 64, 64), (128, 128, 64))


def config(tile, warps, stages, group_m=1):
    bm, bn, bk = tile
    return {
        "config_id": f"m{bm}n{bn}k{bk}-w{warps}-s{stages}-g{group_m}",
        "tile": list(tile),
        "num_warps": warps,
        "num_stages": stages,
        "group_m": group_m,
    }


def case(case_id, split, family, shape, config_ids):
    m, n, k = shape
    return {
        "case_id": case_id,
        "split": split,
        "family": family,
        "shape": list(shape),
        "strides": {"a": [k, 1], "b": [n, 1], "c": [n, 1]},
        "config_ids": config_ids,
    }


def manifest():
    configs = [config(tile, warps, stages)
               for tile, warps, stages in itertools.product(TILES, (4, 8), (1, 2, 4))]
    core = [entry["config_id"] for entry in configs]
    reference = config((64, 64, 32), 4, 2)
    grouped = config((64, 64, 32), 4, 2, group_m=8)
    swizzle = [reference["config_id"], grouped["config_id"]]
    m1 = config((128, 128, 32), 4, 3, group_m=8)
    return {
        "schema_version": 1,
        "kernel": "dense_gemm",
        "input_dtype": "float16",
        "output_dtype": "float16",
        "accumulator_dtype": "float32",
        "input_seed": 17,
        "configs": [*configs, grouped, m1],
        "cases": [
            case("decode-1", "development", "decode", (1, 4096, 4096), core),
            case("decode-16", "development", "decode", (16, 8192, 4096), core),
            case("decode-32", "development", "decode", (32, 4096, 11008), core),
            case("prefill-256", "holdout", "prefill", (256, 4096, 4096), core),
            case("prefill-1024", "holdout", "prefill", (1024, 4096, 11008), core),
            case("prefill-2048", "holdout", "prefill", (2048, 8192, 4096), core),
            case("mask-m", "development", "mask-m", (129, 4096, 4096), core),
            case("mask-mnk", "holdout", "mask-mnk", (513, 4103, 4093), core),
            case("swizzle-320", "development", "swizzle", (320, 4096, 4096), swizzle),
            case("m1-cube", "development", "m1-reference", (4096, 4096, 4096),
                 [m1["config_id"]]),
        ],
        "boundary_rules": [
            {"family": "sm-round", "split": "development", "capacity": "sm_count",
             "offsets": [-1, 0, 1], "reference_config_id": reference["config_id"]},
            {"family": "resident-wave", "split": "holdout", "capacity": "sm_count_times_residency",
             "offsets": [-1, 0, 1], "reference_config_id": reference["config_id"]},
            {"family": "l2-working-set", "split": "holdout", "capacity": "l2_capacity_bytes",
             "ratios": [0.75, 1.25], "reference_config_id": reference["config_id"]},
        ],
    }


def development_manifest(profile):
    """Resolve development-only boundary cases against the measured device."""
    suite = manifest()
    suite["cases"] = [entry for entry in suite["cases"] if entry["split"] == "development"]
    configs = {entry["config_id"]: entry for entry in suite["configs"]}
    core_ids = suite["cases"][0]["config_ids"]
    rule = next(rule for rule in suite.pop("boundary_rules")
                if rule["family"] == "sm-round" and rule["split"] == "development")
    bm, bn, _ = configs[rule["reference_config_id"]]["tile"]
    sm_count = profile["specs"]["sm_count"]
    for name, offset in zip(("below", "exact", "above"), rule["offsets"], strict=True):
        suite["cases"].append(case(f"sm-round-{name}", "development", "sm-round",
                                   (bm, bn * (sm_count + offset), 4096), core_ids))
    suite["boundary_profile_revision"] = profile["provenance"]["profile_revision"]
    return suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--development-profile", type=Path,
                        help="Emit the development subset and resolve its device-dependent cases")
    args = parser.parse_args()
    suite = (development_manifest(json.loads(args.development_profile.read_text()))
             if args.development_profile else manifest())
    args.output.write_text(json.dumps(suite, indent=2) + "\n")


if __name__ == "__main__":
    main()
