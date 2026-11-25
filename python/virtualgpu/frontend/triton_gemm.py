"""First-version Triton GEMM extractor emitting KernelIR v0.1."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from virtualgpu.frontend.compiler import (
    ARTIFACT_HASH_VERSION,
    MockTensor,
    compile_triton_aot,
    specialize_triton_call,
)
from virtualgpu.frontend.scalar_eval import evaluate_ttgir_cta_offsets
from virtualgpu.frontend.ttgir import dtype_bits, parse_ttgir_summary


def _align_up(value: int, alignment: int = 256) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _build_gemm_call_args(
    fn: Any,
    *,
    dtype: str,
    M: int,
    N: int,
    K: int,
    base_a: int,
    base_b: int,
    base_c: int,
    strides_a: tuple[int, int] | None = None,
    strides_b: tuple[int, int] | None = None,
    strides_c: tuple[int, int] | None = None,
) -> tuple[Any, ...]:
    """Construct positional runtime arguments matching a real GEMM launch."""
    sa = strides_a if strides_a is not None else (K, 1)
    sb = strides_b if strides_b is not None else (N, 1)
    sc = strides_c if strides_c is not None else (N, 1)

    value_by_name: dict[str, Any] = {
        "a_ptr": MockTensor(dtype, ptr=base_a),
        "b_ptr": MockTensor(dtype, ptr=base_b),
        "c_ptr": MockTensor(dtype, ptr=base_c),
        "M": M,
        "N": N,
        "K": K,
        "stride_am": sa[0],
        "stride_ak": sa[1],
        "stride_bk": sb[0],
        "stride_bn": sb[1],
        "stride_cm": sc[0],
        "stride_cn": sc[1],
    }

    positional: list[Any] = []
    for name in fn.arg_names:
        if name in value_by_name:
            positional.append(value_by_name[name])
    return tuple(positional)


def _build_access_maps_from_coords(
    cta_tile_coords: list[tuple[int, int]],
    *,
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
    """Build strided_2d_per_outer_tile access_map entries for a given CTA tile sequence."""
    entries_a: list[list[Any]] = []
    entries_b: list[list[Any]] = []
    entries_c: list[list[Any]] = []

    stride_a_bytes = K * elem_bytes
    stride_b_bytes = N * elem_bytes
    stride_c_bytes = N * elem_bytes

    step_a_bytes = block_k * elem_bytes
    step_b_bytes = block_k * stride_b_bytes
    k_masked = (K % block_k) != 0

    for pid_m, pid_n in cta_tile_coords:
        row_start = pid_m * block_m
        col_start = pid_n * block_n
        eff_m = max(0, min(block_m, M - row_start))
        eff_n = max(0, min(block_n, N - col_start))
        m_masked = eff_m < block_m
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


def _recover_cta_tile_coords(
    ttgir: str,
    grid: tuple[int, int, int],
    *,
    fn: Any,
    call_args: tuple[Any, ...],
    signature: dict[str, str],
    M: int,
    N: int,
    K: int,
    block_m: int,
    block_n: int,
    stride_am: int,
    stride_bk: int,
) -> tuple[list[tuple[int, int]], bool, bool, bool]:
    """Use the TTGIR scalar evaluator to recover per-CTA (pid_m, pid_n) order.

    Returns (coords, used_fallback, data_dependent, has_nested_loop).
    """
    # Map non-constexpr runtime parameters in tt.func order to their integer values.
    param_int_values: list[int | None] = []
    runtime_int_args: dict[str, int] = {}
    for arg_name, val in zip(fn.arg_names, call_args):
        if signature.get(arg_name) == "constexpr":
            continue
        if isinstance(val, int) and not isinstance(val, bool):
            param_int_values.append(val)
            runtime_int_args[arg_name] = val
        else:
            param_int_values.append(None)

    ev = evaluate_ttgir_cta_offsets(
        ttgir=ttgir,
        grid=grid,
        runtime_int_args=runtime_int_args,
        param_int_values=param_int_values,
    )
    total_ctas = grid[0] * grid[1] * grid[2]

    # Case 1: Pointer offsets recovered for A (arg 0) and B (arg 1).
    offs_a = ev.ptr_elem_offsets.get(0)
    offs_b = ev.ptr_elem_offsets.get(1)
    if offs_a is not None and offs_b is not None and len(offs_a) == total_ctas and len(offs_b) == total_ctas:
        coords: list[tuple[int, int]] = []
        for oa, ob in zip(offs_a, offs_b):
            row_start = oa // stride_am if stride_am > 0 else 0
            col_start = ob % stride_bk if stride_bk > 0 else ob
            coords.append((row_start // block_m, col_start // block_n))
        return coords, False, ev.data_dependent, ev.has_nested_loop

    # Case 2: TMA 2D coordinates recovered for A (arg 0) and B (arg 1).
    tma_a = ev.tma_coords.get(0)
    tma_b = ev.tma_coords.get(1)
    if tma_a is not None and tma_b is not None and len(tma_a) == total_ctas and len(tma_b) == total_ctas:
        coords = [(ca[0] // block_m, cb[1] // block_n) for ca, cb in zip(tma_a, tma_b)]
        return coords, False, ev.data_dependent, ev.has_nested_loop

    # Fallback: linear row-major tile order.
    num_pid_m = math.ceil(M / block_m)
    num_pid_n = math.ceil(N / block_n)
    linear = [(pm, pn) for pm in range(num_pid_m) for pn in range(num_pid_n)]
    return linear, True, ev.data_dependent, ev.has_nested_loop


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
    call_args: tuple[Any, ...] | None = None,
    strides_a: tuple[int, int] | None = None,
    strides_b: tuple[int, int] | None = None,
    strides_c: tuple[int, int] | None = None,
    ptr_offset_a: int = 0x10000,
    ptr_offset_b: int | None = None,
    ptr_offset_c: int | None = None,
    enable_fp_fusion: bool = True,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Compile a Triton GEMM kernel offline using Triton's runtime JIT binder
    and emit a KernelIR v0.1 dictionary.
    """
    block_m = int(config.get("BLOCK_SIZE_M", config.get("BLOCK_M", config.get("BM", 128))))
    block_n = int(config.get("BLOCK_SIZE_N", config.get("BLOCK_N", config.get("BN", 128))))
    block_k = int(config.get("BLOCK_SIZE_K", config.get("BLOCK_K", config.get("BK", 32))))
    num_warps = int(config.get("num_warps", 4))
    num_stages = int(config.get("num_stages", 3))

    elem_bits = dtype_bits(dtype)
    elem_bytes = elem_bits // 8
    size_a = M * K * elem_bytes
    size_b = K * N * elem_bytes

    # Virtual aligned addresses for L2 modeling (base_addr in KernelIR tensors)
    base_a = 0
    base_b = _align_up(size_a)
    base_c = _align_up(base_b + size_b)

    # Pointer addresses passed to Triton's runtime specialization binder
    sim_ptr_a = ptr_offset_a
    sim_ptr_b = ptr_offset_b if ptr_offset_b is not None else _align_up(sim_ptr_a + size_a)
    sim_ptr_c = ptr_offset_c if ptr_offset_c is not None else _align_up(sim_ptr_b + size_b)

    if call_args is None:
        call_args = _build_gemm_call_args(
            fn,
            dtype=dtype,
            M=M,
            N=N,
            K=K,
            base_a=sim_ptr_a,
            base_b=sim_ptr_b,
            base_c=sim_ptr_c,
            strides_a=strides_a,
            strides_b=strides_b,
            strides_c=strides_c,
        )

    call_kwargs = dict(config)
    call_kwargs.setdefault("enable_fp_fusion", enable_fp_fusion)
    options_dict, signature, constexprs, attrs = specialize_triton_call(
        fn=fn,
        target_arch=target_arch,
        args=call_args,
        kwargs=call_kwargs,
        enable_fp_fusion=enable_fp_fusion,
    )

    artifacts = compile_triton_aot(
        fn=fn,
        signature=signature,
        constexprs=constexprs,
        target_arch=target_arch,
        num_warps=num_warps,
        num_stages=num_stages,
        attrs=attrs,
        options=options_dict,
        enable_fp_fusion=enable_fp_fusion,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )

    summary = parse_ttgir_summary(
        ttgir=artifacts.ttgir,
        ptx=artifacts.ptx,
        requested_num_stages=num_stages,
        element_bytes=elem_bytes,
    )

    diagnostics = list(summary.diagnostics)
    unsupported_features = list(summary.unsupported_features)
    if artifacts.ptxas.ptxas_only_spill:
        diagnostics.append("ptxas_only_spill")

    num_pid_m = math.ceil(M / block_m)
    num_pid_n = math.ceil(N / block_n)
    uses_2d_grid = "tt.get_program_id y" in artifacts.ttgir
    grid = (num_pid_m, num_pid_n, 1) if uses_2d_grid else (num_pid_m * num_pid_n, 1, 1)
    k_trips = math.ceil(K / block_k)

    stride_am = strides_a[0] if strides_a is not None else K
    stride_bk = strides_b[0] if strides_b is not None else N

    cta_tile_coords, used_linear_fallback, data_dependent, has_nested_loop = _recover_cta_tile_coords(
        artifacts.ttgir,
        grid,
        fn=fn,
        call_args=call_args,
        signature=signature,
        M=M,
        N=N,
        K=K,
        block_m=block_m,
        block_n=block_n,
        stride_am=stride_am,
        stride_bk=stride_bk,
    )
    if has_nested_loop:
        diagnostics.append("persistent_kernel_unmodeled")
    if used_linear_fallback:
        diagnostics.append("linear_access_fallback")
    if data_dependent and "data_dependent_address" not in unsupported_features:
        unsupported_features.append("data_dependent_address")

    tile_bytes_a = block_m * block_k * elem_bytes
    tile_bytes_b = block_k * block_n * elem_bytes
    tile_bytes_c = block_m * block_n * elem_bytes

    # Keep prologue.ops empty for standard GEMM: software-pipeline prefill and
    # drain are accounted for by pipeline_loop.prefetch_distance in the model core
    # to avoid double-counting global loads.
    prologue_ops: list[dict[str, Any]] = []

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

    access_map = _build_access_maps_from_coords(
        cta_tile_coords,
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
                "enable_fp_fusion": artifacts.enable_fp_fusion,
                "artifact_hash_version": ARTIFACT_HASH_VERSION,
                "ptx_sha256": artifacts.ptx_sha256,
                "ttgir_sha256": artifacts.ttgir_sha256,
            },
            "config": dict(config),
        },
        "launch": {
            "grid": list(grid),
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
        "unsupported_features": unsupported_features,
    }
