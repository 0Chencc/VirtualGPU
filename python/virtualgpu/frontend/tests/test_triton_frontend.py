"""Offline CPU-only tests for the Triton frontend and KernelIR v0.1 extraction."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
import triton
import triton.language as tl

from virtualgpu.frontend import (
    MockTensor,
    extract_gemm_kernel_ir,
    hash_ptx,
    hash_ttgir,
    parse_ptxas_verbose,
    specialize_triton_call,
)


SCHEMA_PATH = Path(__file__).resolve().parents[4] / "schemas" / "kernel_ir.schema.json"


@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def _matmul_tma_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N]
    )
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = a_desc.load([pid_m * BLOCK_SIZE_M, k * BLOCK_SIZE_K])
        b = b_desc.load([k * BLOCK_SIZE_K, pid_n * BLOCK_SIZE_N])
        acc = tl.dot(a, b, acc)
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N]
    )
    c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], acc.to(tl.float16))


def _load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_parse_ptxas_verbose_zero_and_ptxas_only_spill() -> None:
    clean = (
        "ptxas info    : 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
        "ptxas info    : Used 128 registers, used 1 barriers, 416 bytes cmem[0]\n"
    )
    r_clean = parse_ptxas_verbose(clean, ptx="", assembler="ptxas", assembler_version="12.9", target_sm="sm_89")
    assert r_clean.registers_per_thread == 128
    assert r_clean.barriers == 1
    assert r_clean.spill_scope == "none"
    assert not r_clean.ptxas_only_spill

    spilled = (
        "ptxas info    : 320 bytes stack frame, 328 bytes spill stores, 316 bytes spill loads\n"
        "ptxas info    : Used 255 registers, used 1 barriers, 320 bytes cumulative stack size\n"
    )
    r_spill = parse_ptxas_verbose(spilled, ptx="", assembler="ptxas", assembler_version="12.9", target_sm="sm_89")
    assert r_spill.stack_frame_bytes == 320
    assert r_spill.spill_stores_bytes == 328
    assert r_spill.spill_loads_bytes == 316
    assert r_spill.spill_scope == "inner_loop_assumed"
    assert r_spill.ptxas_only_spill


@pytest.mark.parametrize("target_arch", ["sm_89", "sm_121"])
def test_extract_gemm_validates_against_schema(target_arch: str) -> None:
    schema = _load_schema()
    config = {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 3,
    }
    ir = extract_gemm_kernel_ir(
        _matmul_kernel,
        M=1024,
        N=1024,
        K=512,
        config=config,
        target_arch=target_arch,
    )
    jsonschema.validate(instance=ir, schema=schema)

    assert ir["metadata"]["target_arch"] == target_arch
    assert ir["launch"]["grid"] == [64, 1, 1]
    assert ir["resources"]["registers_per_thread"] > 0
    assert ir["resources"]["static_smem_bytes"] == 32768
    assert ir["body"]["pipeline_loop"]["async_copy"] == "async_element"
    assert ir["body"]["pipeline_loop"]["num_stages"] == 3
    assert ir["body"]["pipeline_loop"]["prefetch_distance"] == 2
    assert ir["unsupported_features"] == []
    assert len(ir["metadata"]["toolchain"]["ptx_sha256"]) == 64
    assert len(ir["metadata"]["toolchain"]["ttgir_sha256"]) == 64

    mma_ops = [op for op in ir["body"]["pipeline_loop"]["ops"] if op["kind"] == "mma"]
    assert len(mma_ops) == 1
    assert mma_ops[0]["op_class"] == "mma.sync"
    assert mma_ops[0]["inst_mnk"] == [16, 8, 16]
    assert mma_ops[0]["count"] == 256


def test_extract_tma_gemm_sm121_uses_async_tensor() -> None:
    schema = _load_schema()
    ir = extract_gemm_kernel_ir(
        _matmul_tma_kernel,
        M=1024,
        N=1024,
        K=512,
        config={"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "num_warps": 4, "num_stages": 3},
        target_arch="sm_121",
    )
    jsonschema.validate(instance=ir, schema=schema)
    assert ir["body"]["pipeline_loop"]["async_copy"] == "async_tensor"
    assert ir["body"]["pipeline_loop"]["num_stages"] == 3


def test_unaligned_shape_and_unpipelined_fallback_diagnostic() -> None:
    schema = _load_schema()
    ir = extract_gemm_kernel_ir(
        _matmul_kernel,
        M=300,
        N=300,
        K=100,
        config={
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": 3,
        },
        target_arch="sm_89",
        # Misaligned pointer (not 16B aligned) causes Triton's runtime binder to
        # omit tt.divisibility=16, matching real runtime JIT fallback behavior.
        ptr_offset_a=0x10002,
        ptr_offset_b=0x20002,
    )
    jsonschema.validate(instance=ir, schema=schema)
    assert ir["body"]["pipeline_loop"]["async_copy"] == "none"
    assert ir["body"]["pipeline_loop"]["num_stages"] == 1
    assert "unpipelined_fallback" in ir["diagnostics"]
    # Tail tiles should be clamped and marked masked.
    last_a = ir["access_map"]["acc_A"]["entries"][-1]
    assert last_a[3] == 44  # 300 - 2*128
    assert last_a[7] is True


def test_runtime_specialization_parity_and_artifact_hashing() -> None:
    # Aligned pointers & 16-divisible dims vs misaligned pointer & non-unit stride
    args_aligned = (
        MockTensor("f16", ptr=0x10000),
        MockTensor("f16", ptr=0x20000),
        MockTensor("f16", ptr=0x30000),
        1024, 1024, 512, 512, 1, 1024, 1, 1024, 1,
    )
    kwargs = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3}
    _, sig_a, consts_a, attrs_a = specialize_triton_call(_matmul_kernel, "sm_89", args_aligned, kwargs)
    assert sig_a["stride_ak"] == "constexpr"
    assert consts_a[(7,)] == 1
    assert ["tt.divisibility", 16] in attrs_a[(0,)]
    assert ["tt.divisibility", 16] in attrs_a[(3,)]

    args_unaligned = (
        MockTensor("f16", ptr=0x10002),
        MockTensor("f16", ptr=0x20000),
        MockTensor("f16", ptr=0x30000),
        300, 300, 100, 100, 2, 300, 1, 300, 1,
    )
    _, sig_u, _, attrs_u = specialize_triton_call(_matmul_kernel, "sm_89", args_unaligned, kwargs)
    assert sig_u["stride_ak"] == "i32"
    assert attrs_u[(0,)] == []
    assert attrs_u[(3,)] == []

    # Normalized hashes ignore comments and debug locations
    ptx1 = "// comment a\n.loc 1 10 0\nmma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 d, a, b, c;\n"
    ptx2 = "// different path\n.loc 2 99 0\n  mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32   d, a, b, c; // trailing\n"
    assert hash_ptx(ptx1) == hash_ptx(ptx2)

    ttgir1 = '#loc1 = loc("/a/b.py":1:1)\n%0 = tt.dot %a, %b, %c : tensor<128x32xf16> loc(#loc1)\n'
    ttgir2 = '#loc1 = loc("/x/y.py":99:5)\n%0 = tt.dot %a, %b, %c : tensor<128x32xf16> loc(#loc1)\n'
    assert hash_ttgir(ttgir1) == hash_ttgir(ttgir2)
