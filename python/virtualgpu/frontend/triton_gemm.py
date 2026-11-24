"""First-version Triton GEMM extractor emitting KernelIR v0.1."""

from __future__ import annotations

import math
from typing import Any

from virtualgpu.frontend.compiler import compile_triton_aot
from virtualgpu.frontend.ttgir import dtype_bits, parse_ttgir_summary


def _align_up(value: int, alignment: int = 256) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _default_gemm_signature_and_attrs(
    fn: Any,
    dtype: str,
    M: int,
    N: int,
    K: int,
) -> tuple[dict[str, str], dict[tuple[int, ...], list[list[Any]]], dict[str, Any]]:
    """Build ASTSource signature, 16B divisibility attrs, and stride constexprs."""
    ptr_ty = f"*{dtype}" if dtype != "f16" else "*fp16"
    if dtype == "bf16":
        ptr_ty = "*bf16"
    elif dtype == "f32":
        ptr_ty = "*fp32"

    arg_names = list(fn.arg_names)
    signature: dict[str, str] = {}
    extra_constexprs: dict[str, Any] = {}
    attrs: dict[tuple[int, ...], list[list[Any]]] = {}

    for idx, name in enumerate(arg_names):
        if name.endswith("_ptr"):
            signature[name] = ptr_ty
            attrs[(idx,)] = [["tt.divisibility", 16]]
        elif name in ("stride_ak", "stride_bn", "stride_cn"):
            signature[name] = "constexpr"
            extra_constexprs[name] = 1
        elif name in ("M", "N", "K", "stride_am", "stride_bk", "stride_cm"):
            signature[name] = "i32"
            dim_val = {"M": M, "N": N, "K": K, "stride_am": K, "stride_bk": N, "stride_cm": N}[name]
            if dim_val % 16 == 0:
                attrs[(idx,)] = [["tt.divisibility", 16]]
        elif name.startswith(("BLOCK_", "GROUP_")):
            continue

    return signature, attrs, extra_constexprs


def _build_linear_access_maps(
    M: int,
    N: int,
    K: int,
    block_m: int,
    block_n: int,
    block_k: int,
    elem_bytes: int,
    base_a: int,
    base_b: int,
    base_c: int,
) -> dict[str, dict[str, Any]]:
    """Build linear CTA traversal access_map entries (M1; swizzle evaluator in M2)."""
    num_pid_m = math.ceil(M / block_m)
    num_pid_n = math.ceil(N / block_n)

    entries_a: list[list[Any]] = []
    entries_b: list[list[Any]] = []
    entries_c: list[list[Any]] = []

    stride_a_bytes = K * elem_bytes
    stride_b_bytes = N * elem_bytes
    stride_c_bytes = N * elem_bytes

    step_a_bytes = block_k * elem_bytes
    step_b_bytes = block_k * stride_b_bytes
    k_masked = (K % block_k) != 0

    for pid_m in range(num_pid_m):
        row_start = pid_m * block_m
        eff_m = min(block_m, M - row_start)
        m_masked = eff_m < block_m

        for pid_n in range(num_pid_n):
            col_start = pid_n * block_n
            eff_n = min(block_n, N - col_start)
            n_masked = eff_n < block_n

            # A tile: [eff_m, block_k] stepping along K
            entries_a.append([
                pid_m,
                pid_n,
                base_a + row_start * stride_a_bytes,
                eff_m,
                min(block_k, K) * elem_bytes,
                stride_a_bytes,
                step_a_bytes,
                m_masked or k_masked,
            ])

            # B tile: [block_k, eff_n] stepping along K
            entries_b.append([
                pid_m,
                pid_n,
                base_b + col_start * elem_bytes,
                min(block_k, K),
                eff_n * elem_bytes,
                stride_b_bytes,
                step_b_bytes,
                n_masked or k_masked,
            ])

            # C tile: [eff_m, eff_n] epilogue store (no K step)
            entries_c.append([
                pid_m,
                pid_n,
                base_c + row_start * stride_c_bytes + col_start * elem_bytes,
                eff_m,
                eff_n * elem_bytes,
                stride_c_bytes,
                0,
                m_masked or n_masked,
            ])

    return {
        "acc_A": {"encoding": "strided_2d_per_outer_tile", "sidecar_ref": None, "entries": entries_a},
        "acc_B": {"encoding": "strided_2d_per_outer_tile", "sidecar_ref": None, "entries": entries_b},
        "acc_C": {"encoding": "strided_2d_per_outer_tile", "sidecar_ref": None, "entries": entries_c},
    }


def extract_gemm_kernel_ir(
    fn: Any,
    *,
    M: int,
    N: int,
    K: int,
    config: dict[str, int],
    target_arch: str,
    dtype: str = "f16",
    acc_dtype: str = "f32",
    signature: dict[str, str] | None = None,
    attrs: dict[tuple[int, ...], list[list[Any]]] | None = None,
    extra_constexprs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile a Triton GEMM kernel offline and emit a KernelIR v0.1 dictionary."""
    block_m = int(config.get("BLOCK_SIZE_M", config.get("BLOCK_M", 128)))
    block_n = int(config.get("BLOCK_SIZE_N", config.get("BLOCK_N", 128)))
    block_k = int(config.get("BLOCK_SIZE_K", config.get("BLOCK_K", 32)))
    num_warps = int(config.get("num_warps", 4))
    num_stages = int(config.get("num_stages", 3))

    if signature is None:
        signature, auto_attrs, auto_consts = _default_gemm_signature_and_attrs(fn, dtype, M, N, K)
        if attrs is None:
            attrs = auto_attrs
        if extra_constexprs is None:
            extra_constexprs = auto_consts

    constexprs: dict[str, Any] = {}
    for k, v in config.items():
        if k in fn.arg_names:
            constexprs[k] = v
    if extra_constexprs:
        for k, v in extra_constexprs.items():
            if k in fn.arg_names:
                constexprs[k] = v

    artifacts = compile_triton_aot(
        fn=fn,
        signature=signature,
        constexprs=constexprs,
        target_arch=target_arch,
        num_warps=num_warps,
        num_stages=num_stages,
        attrs=attrs,
    )

    elem_bits = dtype_bits(dtype)
    elem_bytes = elem_bits // 8
    summary = parse_ttgir_summary(
        ttgir=artifacts.ttgir,
        ptx=artifacts.ptx,
        requested_num_stages=num_stages,
        element_bytes=elem_bytes,
    )

    diagnostics = list(summary.diagnostics)
    if artifacts.ptxas.ptxas_only_spill:
        diagnostics.append("ptxas_only_spill")

    num_pid_m = math.ceil(M / block_m)
    num_pid_n = math.ceil(N / block_n)
    grid_x = num_pid_m * num_pid_n
    k_trips = math.ceil(K / block_k)

    size_a = M * K * elem_bytes
    size_b = K * N * elem_bytes
    base_a = 0
    base_b = _align_up(size_a)
    base_c = _align_up(base_b + size_b)

    tile_bytes_a = block_m * block_k * elem_bytes
    tile_bytes_b = block_k * block_n * elem_bytes
    tile_bytes_c = block_m * block_n * elem_bytes

    # Prologue prefills the pipeline buffers when software pipelining is active.
    prologue_ops: list[dict[str, Any]] = []
    if summary.prefetch_distance > 0:
        prologue_ops.extend([
            {
                "op_id": "prefill_A",
                "kind": "gmem_load",
                "warp_group_id": 0,
                "stage_offset": 0,
                "tensor_id": "A",
                "nominal_bytes_per_cta": tile_bytes_a,
                "transfer_width_bytes": summary.transfer_width_bytes,
                "access_ref": "acc_A",
            },
            {
                "op_id": "prefill_B",
                "kind": "gmem_load",
                "warp_group_id": 0,
                "stage_offset": 0,
                "tensor_id": "B",
                "nominal_bytes_per_cta": tile_bytes_b,
                "transfer_width_bytes": summary.transfer_width_bytes,
                "access_ref": "acc_B",
            },
        ])

    # Steady-state pipeline loop ops.
    smem_load_bytes = summary.smem_load_bytes_per_iter or (tile_bytes_a + tile_bytes_b)
    pipeline_ops: list[dict[str, Any]] = [
        {
            "op_id": "load_A",
            "kind": "gmem_load",
            "warp_group_id": 0,
            "stage_offset": summary.prefetch_distance,
            "tensor_id": "A",
            "nominal_bytes_per_cta": tile_bytes_a,
            "transfer_width_bytes": summary.transfer_width_bytes,
            "access_ref": "acc_A",
        },
        {
            "op_id": "load_B",
            "kind": "gmem_load",
            "warp_group_id": 0,
            "stage_offset": summary.prefetch_distance,
            "tensor_id": "B",
            "nominal_bytes_per_cta": tile_bytes_b,
            "transfer_width_bytes": summary.transfer_width_bytes,
            "access_ref": "acc_B",
        },
        {
            "op_id": "smem_load_AB",
            "kind": "smem_load",
            "warp_group_id": 0,
            "stage_offset": 0,
            "bytes_per_cta": smem_load_bytes,
        },
    ]

    if summary.mma is not None:
        im, in_, ik = summary.mma.inst_mnk
        mma_count = max(
            1,
            math.ceil(block_m / im) * math.ceil(block_n / in_) * math.ceil(block_k / ik),
        )
        pipeline_ops.append({
            "op_id": "mma_0",
            "kind": "mma",
            "warp_group_id": 0,
            "stage_offset": 0,
            "op_class": summary.mma.op_class,
            "tile_mnk": [block_m, block_n, block_k],
            "inst_mnk": [im, in_, ik],
            "a_dtype": summary.mma.a_dtype,
            "b_dtype": summary.mma.b_dtype,
            "acc_dtype": summary.mma.acc_dtype,
            "count": mma_count,
        })

    if summary.loop_barriers > 0:
        pipeline_ops.append({
            "op_id": "stage_sync",
            "kind": "barrier",
            "warp_group_id": 0,
            "stage_offset": 0,
            "count": summary.loop_barriers,
        })

    epilogue_ops: list[dict[str, Any]] = [
        {
            "op_id": "epilogue_cast",
            "kind": "alu",
            "warp_group_id": 0,
            "stage_offset": 0,
            "dtype": dtype,
            "count": max(1, (block_m * block_n) // (summary.num_warps * summary.warp_size)),
        },
        {
            "op_id": "store_C",
            "kind": "gmem_store",
            "warp_group_id": 0,
            "stage_offset": 0,
            "tensor_id": "C",
            "nominal_bytes_per_cta": tile_bytes_c,
            "transfer_width_bytes": summary.transfer_width_bytes,
            "access_ref": "acc_C",
        },
    ]

    access_map = _build_linear_access_maps(
        M=M,
        N=N,
        K=K,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        elem_bytes=elem_bytes,
        base_a=base_a,
        base_b=base_b,
        base_c=base_c,
    )

    return {
        "schema_version": "0.1",
        "metadata": {
            "kernel_name": artifacts.kernel_name,
            "target_vendor": "nvidia",
            "target_arch": artifacts.target_arch,
            "frontend": "triton",
            "toolchain": {
                "compiler": "triton",
                "compiler_version": artifacts.triton_version,
                "assembler": artifacts.ptxas.assembler,
                "assembler_version": artifacts.ptxas.assembler_version,
                "target_sm_compiled": artifacts.ptxas.target_sm,
            },
            "config": dict(config),
        },
        "launch": {
            "grid": [grid_x, 1, 1],
            "cluster_shape": list(summary.cluster_shape),
            "num_warps": summary.num_warps,
            "warp_size": summary.warp_size,
            "warp_groups": [
                {"id": 0, "role": "unified", "num_warps": summary.num_warps}
            ],
            "dynamic_smem_bytes": 0,
        },
        "resources": {
            "source": "assembler_report",
            "registers_per_thread": artifacts.ptxas.registers_per_thread,
            "static_smem_bytes": artifacts.static_smem_bytes,
            "stack_frame_bytes": artifacts.ptxas.stack_frame_bytes,
            "spill_stores_bytes": artifacts.ptxas.spill_stores_bytes,
            "spill_loads_bytes": artifacts.ptxas.spill_loads_bytes,
            "spill_scope": artifacts.ptxas.spill_scope,
            "barriers": artifacts.ptxas.barriers,
        },
        "tensors": [
            {
                "id": "A",
                "dtype": dtype,
                "element_bits": elem_bits,
                "shape": [M, K],
                "strides": [K, 1],
                "base_addr": base_a,
                "cache_policy": "default",
            },
            {
                "id": "B",
                "dtype": dtype,
                "element_bits": elem_bits,
                "shape": [K, N],
                "strides": [N, 1],
                "base_addr": base_b,
                "cache_policy": "default",
            },
            {
                "id": "C",
                "dtype": dtype,
                "element_bits": elem_bits,
                "shape": [M, N],
                "strides": [N, 1],
                "base_addr": base_c,
                "cache_policy": "streaming",
            },
        ],
        "body": {
            "outer_loop": {
                "trip_count_mode": "uniform",
                "trip_counts": 1,
            },
            "prologue": {"ops": prologue_ops},
            "pipeline_loop": {
                "trip_count_mode": "uniform",
                "trip_counts": k_trips,
                "num_stages": summary.num_stages,
                "prefetch_distance": summary.prefetch_distance,
                "async_copy": summary.async_copy,
                "ops": pipeline_ops,
            },
            "epilogue": {
                "overlaps_next_tile": False,
                "ops": epilogue_ops,
            },
        },
        "access_map": access_map,
        "diagnostics": diagnostics,
        "unsupported_features": summary.unsupported_features,
    }
