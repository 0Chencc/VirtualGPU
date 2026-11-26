"""Synthetic KernelIR for the post-hoc GB10 microbenchmarks, and their predictions.

Each microbenchmark point of `wave-rescan`, `b-stream` and `staged-mma`
(microbench/src/posthoc_gb10_kernels.cuh) is written as the KernelIR of the
kernel that runs, and predicted with the installed core. The point grid
mirrors the CUDA sweep, so predictions can be frozen before any capture.

Usage:
    python microbench/posthoc_gb10_ir.py --profile profiles/gb10.json --out predictions.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from virtualgpu import _core

THREADS_B_STREAM = 128
THREADS_STAGED = 256
RATIOS = (0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0)


def padded_smem(profile, ctas_per_sm):
    """Dynamic shared memory the CUDA driver pads each CTA to, as in b_stream_run."""
    s = profile["specs"]
    per_sm = s["shared_mem_per_sm_bytes"]
    reserved = s["reserved_shared_mem_per_cta_bytes"]
    return min((per_sm // ctas_per_sm - reserved) // 128 * 128, s["max_shared_mem_per_block_bytes"])


def kernel_ir(*, name, grid, warps, smem, tokens, trips, ops, access, tensors, config):
    return {
        "schema_version": "0.1",
        "metadata": {
            "kernel_name": name,
            "frontend": "triton",
            "target_vendor": "nvidia",
            "target_arch": "sm_121",
            "config": config,
        },
        "launch": {
            "grid": [grid, 1, 1],
            "num_warps": warps,
            "warp_size": 32,
            "warp_groups": [{"id": 0, "num_warps": warps, "role": "unified"}],
            "cluster_shape": [1, 1, 1],
            "dynamic_smem_bytes": 0,
        },
        "resources": {
            "registers_per_thread": 64,
            "static_smem_bytes": smem,
            "spill_loads_bytes": 0,
            "spill_stores_bytes": 0,
            "stack_frame_bytes": 0,
            "spill_scope": "none",
            "barriers": 1,
            "source": "assembler_report",
        },
        "tensors": tensors,
        "body": {
            "prologue": {"ops": []},
            "pipeline_loop": {
                "async_copy": "async_element",
                "num_stages": tokens,
                "prefetch_distance": tokens - 1,
                "trip_count_mode": "uniform",
                "trip_counts": trips,
                "ops": ops,
            },
            "outer_loop": {"trip_count_mode": "uniform", "trip_counts": 1},
            "epilogue": {"ops": [], "overlaps_next_tile": False},
        },
        "access_map": {
            "acc_B": {
                "encoding": "strided_2d_per_outer_tile",
                "sidecar_ref": None,
                "entries": access,
            }
        },
        "diagnostics": [],
        "unsupported_features": [],
    }


def stream_ops(tokens, stage, smem_bytes, mma_count, warps):
    ops = [
        {
            "kind": "gmem_load",
            "op_id": "load_B",
            "tensor_id": "B",
            "access_ref": "acc_B",
            "nominal_bytes_per_cta": stage,
            "transfer_width_bytes": 16,
            "stage_offset": tokens - 1,
            "warp_group_id": 0,
        },
        {
            "kind": "smem_load",
            "op_id": "smem_load",
            "bytes_per_cta": smem_bytes,
            "stage_offset": 0,
            "warp_group_id": 0,
        },
    ]
    ops.append(
        {
            "kind": "mma",
            "op_id": "mma_0",
            "op_class": "mma.sync",
            "a_dtype": "f16",
            "b_dtype": "f16",
            "acc_dtype": "f32",
            "inst_mnk": [16, 8, 16],
            "tile_mnk": [16 * warps, 32, 16 * mma_count // (4 * warps)],
            "count": mma_count,
            "stage_offset": 0,
            "warp_group_id": 0,
        }
    )
    ops.append(
        {
            "kind": "barrier",
            "op_id": "stage_sync",
            "count": 2,
            "stage_offset": 0,
            "warp_group_id": 0,
        }
    )
    return ops


def b_stream_ir(
    profile, *, tokens, stage, ctas_per_sm, grid, panels, row_bytes, k_steps, mma_iters
):
    """b_stream_kernel: CTA b reads panel b % panels of a row-major [k_rows, panels * row_bytes]."""
    rows = stage // row_bytes
    stride = panels * row_bytes
    k_rows = rows * k_steps
    access = [
        [
            b // panels,
            b % panels,
            (b % panels) * row_bytes,
            rows,
            row_bytes,
            stride,
            rows * stride,
            False,
        ]
        for b in range(grid)
    ]
    tensors = [
        {
            "id": "B",
            "dtype": "f16",
            "element_bits": 16,
            "shape": [k_rows, stride // 2],
            "strides": [stride // 2, 1],
            "base_addr": 0,
            "cache_policy": "default",
        }
    ]
    return kernel_ir(
        name="b_stream",
        grid=grid,
        warps=4,
        smem=padded_smem(profile, ctas_per_sm),
        tokens=tokens,
        trips=k_steps,
        ops=stream_ops(tokens, stage, THREADS_B_STREAM * 16, 16 * mma_iters, 4),
        access=access,
        tensors=tensors,
        config={"tokens": tokens, "stage_bytes": stage, "ctas_per_sm": ctas_per_sm},
    )


def staged_mma_ir(profile, *, tokens, stage, ctas_per_sm, mma_iters, steps):
    """staged_mma_kernel: every stage of CTA b reads the same stage bytes at b * stage."""
    grid = profile["specs"]["sm_count"] * ctas_per_sm
    access = [[0, b, b * stage, 1, stage, stage, 0, False] for b in range(grid)]
    tensors = [
        {
            "id": "B",
            "dtype": "f16",
            "element_bits": 16,
            "shape": [grid, stage // 2],
            "strides": [stage // 2, 1],
            "base_addr": 0,
            "cache_policy": "default",
        }
    ]
    return kernel_ir(
        name="staged_mma",
        grid=grid,
        warps=8,
        smem=padded_smem(profile, ctas_per_sm),
        tokens=tokens,
        trips=steps,
        ops=stream_ops(tokens, stage, 32768, 32 * mma_iters, 8),
        access=access,
        tensors=tensors,
        config={"tokens": tokens, "stage_bytes": stage, "ctas_per_sm": ctas_per_sm},
    )


def wave_rescan_points(profile):
    sms, l2 = profile["specs"]["sm_count"], profile["specs"]["l2_size_bytes"]
    row_bytes, stage = 256, 8192
    for cps in (1, 2, 4):
        for rows_per_wave in (1, 2):
            for ratio in RATIOS:
                slots = sms * cps
                panels = slots // rows_per_wave
                stride = panels * row_bytes
                k_steps = max(int(ratio * l2 / (stride * (stage // row_bytes)) + 0.5), 4)
                yield {
                    "ctas_per_sm": cps,
                    "rows_per_wave": rows_per_wave,
                    "footprint_ratio_nominal": ratio,
                    "slots": slots,
                    "panels": panels,
                    "row_bytes": row_bytes,
                    "stage_bytes": stage,
                    "k_steps": k_steps,
                }


def b_stream_points(profile):
    sms, s = profile["specs"]["sm_count"], profile["specs"]
    target = 256 << 20
    for tokens in (1, 2, 3):
        for cps in (1, 2, 3, 4, 6, 9):
            for vpt in (2, 4, 8, 16):
                stage = THREADS_B_STREAM * 16 * vpt
                if padded_smem(profile, cps) < tokens * stage or cps > s["max_blocks_per_sm"]:
                    continue
                for row_bytes in (0, 256, 128):
                    for mma_iters in (1, 8):
                        grid = sms * cps
                        yield {
                            "tokens": tokens,
                            "ctas_per_sm": cps,
                            "stage_bytes": stage,
                            "row_bytes": row_bytes or stage,
                            "pattern": "rows" if row_bytes else "linear",
                            "mma_iters_per_stage": mma_iters,
                            "grid": grid,
                            "k_steps": target // (grid * stage),
                        }


def staged_mma_points(profile):
    for cps in (1, 2):
        for tokens in (1, 2, 3):
            for vpt in (4, 8):
                stage = 4096 * vpt
                if padded_smem(profile, cps) < tokens * stage:
                    continue
                for mma_iters in (1, 8, 16, 32):
                    yield {
                        "ctas_per_sm": cps,
                        "tokens": tokens,
                        "stage_bytes": stage,
                        "mma_iters_per_stage": mma_iters,
                        "steps": 1024,
                    }


def interpolate(points, x):
    points = sorted(points)
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


def candidate_hits(profile, footprint):
    """Re-reading wave hit fraction under the zero-parameter candidate forms b and c.

    b: every chunk that ends in one event gets the clock after the event, so a
       re-read sees the whole wave footprint: the profile's rescan curve at it.
    c: LRU at k-slice grain with capacity l2_effective_capacity_bytes. Forward
       re-reads start at the oldest slices: all hit if the wave fits, else none.
       Reversed re-reads start at the newest: min(1, capacity / footprint).
    Form a (the current core) is the core prediction itself; KernelIR cannot
    express a reversed K walk, so for reversed points it equals the forward one.
    """
    mem = profile["measured"]["memory"]
    curve = [
        (q["working_set_bytes"], q["estimated_hit_rate"]["value"])
        for q in mem["l2_rescan_hit_rate_by_working_set"]
    ]
    for i in range(len(curve) - 2, -1, -1):  # monotone non-increasing, as the core builds it
        curve[i] = (curve[i][0], max(curve[i][1], curve[i + 1][1]))
    cap = mem["l2_effective_capacity_bytes"]["value"]
    return {
        "form_b": interpolate(curve, footprint),
        "form_c_forward": 1.0 if footprint <= cap else 0.0,
        "form_c_reversed": min(1.0, cap / footprint),
    }


def run(ir, profile_text, cache):
    r = _core.predict(json.dumps(ir), profile_text, l2_state=cache, draws=0, seed=17)
    return {
        "time_ns": r["time_ns"],
        "ctas_per_sm": r["ctas_per_sm"],
        "dram_bytes": r["dram_bytes"],
        "l2_hit_bytes": r["l2_hit_bytes"],
        "merged_bytes": r["merged_bytes"],
        "bound_fraction": {k: v for k, v in r["bound_fraction"].items() if v > 0},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    text = args.profile.read_text()
    profile = json.loads(text)
    out = {
        "post_hoc": True,
        "build": _core.build_info(),
        "profile_revision": profile["provenance"]["profile_revision"],
        "wave_rescan": [],
        "b_stream": [],
        "staged_mma": [],
    }
    for p in wave_rescan_points(profile):
        common = dict(
            tokens=3,
            stage=p["stage_bytes"],
            ctas_per_sm=p["ctas_per_sm"],
            panels=p["panels"],
            row_bytes=p["row_bytes"],
            k_steps=p["k_steps"],
            mma_iters=1,
        )
        one = run(b_stream_ir(profile, grid=p["slots"], **common), text, "cold")
        four = run(b_stream_ir(profile, grid=4 * p["slots"], **common), text, "cold")
        footprint = (
            p["panels"] * p["row_bytes"] * (p["stage_bytes"] // p["row_bytes"]) * p["k_steps"]
        )
        form_a = 1 - (four["dram_bytes"] - one["dram_bytes"]) / (3 * footprint)
        out["wave_rescan"].append(
            {
                **p,
                "footprint_bytes": footprint,
                "one_wave": one,
                "four_wave": four,
                "rewave_hit_fraction": {"form_a": form_a, **candidate_hits(profile, footprint)},
            }
        )
    for p in b_stream_points(profile):
        ir = b_stream_ir(
            profile,
            tokens=p["tokens"],
            stage=p["stage_bytes"],
            ctas_per_sm=p["ctas_per_sm"],
            grid=p["grid"],
            panels=p["grid"],
            row_bytes=p["row_bytes"],
            k_steps=p["k_steps"],
            mma_iters=p["mma_iters_per_stage"],
        )
        out["b_stream"].append({**p, "prediction": run(ir, text, "cold")})
    for p in staged_mma_points(profile):
        ir = staged_mma_ir(
            profile,
            tokens=p["tokens"],
            stage=p["stage_bytes"],
            ctas_per_sm=p["ctas_per_sm"],
            mma_iters=p["mma_iters_per_stage"],
            steps=p["steps"],
        )
        r = run(ir, text, "warm")
        out["staged_mma"].append(
            {**p, "prediction": r, "stage_turnaround_ns": r["time_ns"] / p["steps"]}
        )
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
