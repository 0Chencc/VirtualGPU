"""Extract post-pipelining structure from Triton TTGIR and PTX."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MmaInfo:
    """Hardware MMA instruction details extracted from TTGIR and PTX."""

    op_class: str
    inst_mnk: tuple[int, int, int]
    a_dtype: str
    b_dtype: str
    acc_dtype: str


@dataclass(frozen=True)
class TtgirSummary:
    """Structured summary extracted from post-pipelining TTGIR + PTX."""

    num_warps: int
    warp_size: int
    cluster_shape: tuple[int, int, int]
    async_copy: str  # "none" | "async_element" | "async_tensor"
    num_stages: int
    prefetch_distance: int
    transfer_width_bytes: int
    mma: MmaInfo | None
    smem_load_bytes_per_iter: int
    loop_barriers: int
    diagnostics: list[str] = field(default_factory=list)
    unsupported_features: list[str] = field(default_factory=list)


_DTYPE_BITS: dict[str, int] = {
    "f4": 4,
    "f8E4M3FN": 8,
    "f8E5M2": 8,
    "i8": 8,
    "f16": 16,
    "bf16": 16,
    "i16": 16,
    "f32": 32,
    "tf32": 32,
    "i32": 32,
    "f64": 64,
    "i64": 64,
}

_RE_NUM_WARPS = re.compile(r'"ttg\.num-warps"\s*=\s*(\d+)')
_RE_WARP_SIZE = re.compile(r'"ttg\.threads-per-warp"\s*=\s*(\d+)')
_RE_NUM_CTAS = re.compile(r'"ttg\.num-ctas"\s*=\s*(\d+)')
_RE_MULTIBUF_MEMDESC = re.compile(
    r"ttg\.local_alloc\s*:\s*\(\)\s*->\s*!ttg\.memdesc<(\d+)x(\d+)x(\d+)x([a-zA-Z0-9_]+)"
)
_RE_CONTIGUITY = re.compile(r"contiguity\s*=\s*(\d+)")
_RE_PTX_MMA_SYNC = re.compile(
    r"mma\.sync\.aligned\.m(\d+)n(\d+)k(\d+)\.[a-z.]+\.(f32|f16|bf16|s32)\.(f16|bf16|tf32|f8e4m3|f8e5m2|s8|u8)\.(f16|bf16|tf32|f8e4m3|f8e5m2|s8|u8)\.(f32|f16|bf16|s32)"
)
_RE_PTX_WGMMA = re.compile(r"wgmma\.mma_async\.sync\.aligned\.m(\d+)n(\d+)k(\d+)")
_RE_PTX_TCGEN05 = re.compile(r"tcgen05\.mma\.")
_RE_TTGIR_MMA_LAYOUT = re.compile(r"#ttg\.nvidia_mma<[^>]*instrShape\s*=\s*\[(\d+),\s*(\d+)\]")
_RE_TT_DOT = re.compile(
    r"tt\.dot\s+[^:]+:\s*tensor<(\d+)x(\d+)x([a-zA-Z0-9_]+),\s*[^>]+>\s*\*\s*tensor<(\d+)x(\d+)x([a-zA-Z0-9_]+),\s*[^>]+>\s*->\s*tensor<(\d+)x(\d+)x([a-zA-Z0-9_]+)"
)
_RE_LOCAL_LOAD = re.compile(
    r"ttg\.local_load\b[^:]*:\s*!ttg\.memdesc<(\d+)x(\d+)x([a-zA-Z0-9_]+)"
)


def dtype_bits(dtype: str) -> int:
    """Return bit width for a canonical dtype name."""
    if dtype in _DTYPE_BITS:
        return _DTYPE_BITS[dtype]
    raise ValueError(f"Unknown dtype: {dtype!r}")


def _extract_loop_body(ttgir: str) -> str:
    """Return the text inside the primary scf.for loop, or empty string if none."""
    idx = ttgir.find("scf.for ")
    if idx == -1:
        return ""
    brace_start = ttgir.find("{", idx)
    if brace_start == -1:
        return ""
    depth = 0
    for pos in range(brace_start, len(ttgir)):
        ch = ttgir[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return ttgir[brace_start + 1 : pos]
    return ttgir[brace_start + 1 :]


def _extract_mma(ttgir: str, ptx: str) -> MmaInfo | None:
    """Extract hardware MMA class, instruction shape, and dtypes."""
    m_dot = _RE_TT_DOT.search(ttgir)
    a_dtype = m_dot.group(3) if m_dot else "f16"
    b_dtype = m_dot.group(6) if m_dot else "f16"
    acc_dtype = m_dot.group(9) if m_dot else "f32"

    m_sync = _RE_PTX_MMA_SYNC.search(ptx)
    if m_sync:
        return MmaInfo(
            op_class="mma.sync",
            inst_mnk=(int(m_sync.group(1)), int(m_sync.group(2)), int(m_sync.group(3))),
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            acc_dtype=acc_dtype,
        )

    m_wgmma = _RE_PTX_WGMMA.search(ptx)
    if m_wgmma:
        return MmaInfo(
            op_class="wgmma",
            inst_mnk=(int(m_wgmma.group(1)), int(m_wgmma.group(2)), int(m_wgmma.group(3))),
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            acc_dtype=acc_dtype,
        )

    if _RE_PTX_TCGEN05.search(ptx):
        return MmaInfo(
            op_class="tcgen05",
            inst_mnk=(64, 64, 16),
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            acc_dtype=acc_dtype,
        )

    m_layout = _RE_TTGIR_MMA_LAYOUT.search(ttgir)
    if m_layout:
        inst_k = 16 if dtype_bits(a_dtype) <= 16 else 8
        return MmaInfo(
            op_class="mma.sync",
            inst_mnk=(int(m_layout.group(1)), int(m_layout.group(2)), inst_k),
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            acc_dtype=acc_dtype,
        )

    return None


def parse_ttgir_summary(
    ttgir: str,
    ptx: str,
    requested_num_stages: int,
    element_bytes: int = 2,
) -> TtgirSummary:
    """Parse post-pipelining TTGIR and PTX into a structured summary."""
    diagnostics: list[str] = []
    unsupported: list[str] = []

    m_warps = _RE_NUM_WARPS.search(ttgir)
    num_warps = int(m_warps.group(1)) if m_warps else 4

    m_ws = _RE_WARP_SIZE.search(ttgir)
    warp_size = int(m_ws.group(1)) if m_ws else 32

    m_ctas = _RE_NUM_CTAS.search(ttgir)
    num_ctas = int(m_ctas.group(1)) if m_ctas else 1
    cluster_shape = (num_ctas, 1, 1)

    if "bCluster = true" in ttgir or num_ctas > 1:
        unsupported.append("cluster_smem")

    if "tt.atomic_cas" in ttgir or "tt.atomic_xchg" in ttgir:
        unsupported.append("atomic_work_stealing")

    # Detect actual async copy mechanism emitted after software pipelining.
    if "ttng.async_tma_copy_global_to_local" in ttgir:
        async_copy = "async_tensor"
    elif "ttg.async_copy_global_to_local" in ttgir:
        async_copy = "async_element"
    else:
        async_copy = "none"

    # Extract materialized multi-buffer depth from 3D SMEM memdesc allocations
    # (excluding 1-element i64 mbarrier arrays).
    buf_depths = [
        int(m.group(1))
        for m in _RE_MULTIBUF_MEMDESC.finditer(ttgir)
        if not (int(m.group(2)) == 1 and m.group(4) == "i64")
    ]
    if async_copy != "none" and buf_depths:
        prefetch_distance = max(buf_depths)
        num_stages = prefetch_distance + 1
    else:
        prefetch_distance = 0
        num_stages = 1
        if requested_num_stages > 1:
            diagnostics.append("unpipelined_fallback")

    # Vectorized transfer width per thread (bytes).
    contigs = [int(m.group(1)) for m in _RE_CONTIGUITY.finditer(ttgir)]
    if async_copy == "async_tensor":
        transfer_width_bytes = 128
    elif contigs:
        transfer_width_bytes = max(1, min(16, max(contigs) * element_bytes))
    else:
        transfer_width_bytes = 16

    loop_body = _extract_loop_body(ttgir)
    smem_load_bytes = 0
    for m in _RE_LOCAL_LOAD.finditer(loop_body):
        rows, cols, dt = int(m.group(1)), int(m.group(2)), m.group(3)
        smem_load_bytes += (rows * cols * dtype_bits(dt)) // 8

    # Count synchronization points inside the pipelined loop.
    loop_barriers = (
        loop_body.count("ttg.async_wait")
        + loop_body.count("ttng.wait_barrier")
        + loop_body.count("gpu.barrier")
    )
    if loop_barriers == 0 and async_copy == "none":
        loop_barriers = 1

    mma = _extract_mma(ttgir, ptx)

    return TtgirSummary(
        num_warps=num_warps,
        warp_size=warp_size,
        cluster_shape=cluster_shape,
        async_copy=async_copy,
        num_stages=num_stages,
        prefetch_distance=prefetch_distance,
        transfer_width_bytes=transfer_width_bytes,
        mma=mma,
        smem_load_bytes_per_iter=smem_load_bytes,
        loop_barriers=loop_barriers,
        diagnostics=diagnostics,
        unsupported_features=unsupported,
    )
