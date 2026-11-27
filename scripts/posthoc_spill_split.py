"""Post-hoc check of the causal attention spill split against measured local loads.

The core charges `spill_loads_bytes / 4 / spill_loop_count` reloads per thread per iteration.
This compares that static figure, and the old one without the split, with the dynamic local
loads per thread per iteration that `microbench.spill_counters` measured in each loop. It
recompiles each config offline (CPU only, needs Triton) and fits nothing.

  python scripts/posthoc_spill_split.py results/post-hoc/spill-counters-<stamp>.json [OUT.json]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "validation/attention-suite.json"
RELOAD_BYTES = 4


def compare(row, resources, num_warps):
    loops = resources.get("spill_loop_count", 1)
    unsplit = resources["spill_loads_bytes"] / RELOAD_BYTES
    measured = {part: row["loops"][part]["per_thread_iteration"]["load"] for part in row["loops"]}
    visits = sum(row["loops"][part]["cta_iterations"] for part in row["loops"])
    loads = sum(row["loops"][part]["thread_local_load"] for part in row["loops"])
    return {
        "config_id": row["config_id"],
        "spill_scope": resources["spill_scope"],
        "spill_loop_count": loops,
        "static_unsplit": unsplit,
        "static_split": unsplit / loops,
        "measured_by_loop": measured,
        "measured_all_loops": loads / (visits * num_warps * 32) if visits else 0.0,
    }


def frontend_resources(case, config, target):
    from virtualgpu.frontend import extract_attention_kernel_ir

    from validation import attention
    from validation.attention_runner import _AOT_POINTERS

    _, constants, options = attention.launch_spec(case, config)
    hq, hk, seq, dim, causal, scale, bm, bn = constants
    knobs = dict(
        num_warps=options["num_warps"],
        num_stages=options["num_stages"],
        BM=bm,
        BN=bn,
        SCALE_LOG2=scale,
    )
    ir = extract_attention_kernel_ir(
        attention.fused_attention,
        case["batch"],
        hq,
        seq,
        dim,
        knobs,
        target,
        H_kv=hk,
        causal=causal,
        call_args=(*_AOT_POINTERS, hq, hk, seq, dim),
        use_cache=False,
    )
    return ir["resources"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("counters", type=Path)
    parser.add_argument("out", type=Path, nargs="?")
    args = parser.parse_args()
    data = json.loads(args.counters.read_text())
    attention = data["causal_attention"]
    target = data["provenance"]["target"]
    configs = {row["config_id"]: row for row in json.loads(MANIFEST.read_text())["configs"]}
    rows = []
    for row in attention["configs"]:
        config = configs[row["config_id"]]
        resources = frontend_resources(attention["case"], config, target)
        if resources["registers_per_thread"] != row["artifact"]["registers_per_thread"]:
            print(f"{row['config_id']}: registers differ from the capture, toolchain mismatch?")
        rows.append(compare(row, resources, config["num_warps"]))
    for r in rows:
        by_loop = ", ".join(f"{k} {v:.2f}" for k, v in r["measured_by_loop"].items())
        print(
            f"{r['config_id']}: static {r['static_unsplit']:.2f} unsplit, "
            f"{r['static_split']:.2f} split by {r['spill_loop_count']}; "
            f"measured {r['measured_all_loops']:.2f} ({by_loop})"
        )
    if args.out:
        result = {"counters": str(args.counters), "commit": data["provenance"]["commit"]}
        args.out.write_text(json.dumps({**result, "configs": rows}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
