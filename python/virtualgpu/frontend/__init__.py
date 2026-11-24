"""Kernel frontends that lower source kernels to VirtualGPU KernelIR."""

from virtualgpu.frontend.compiler import (
    ARTIFACT_HASH_VERSION,
    CompiledArtifacts,
    MockTensor,
    PtxasReport,
    artifact_hashes,
    compile_triton_aot,
    hash_ptx,
    hash_ttgir,
    parse_ptxas_verbose,
    specialize_triton_call,
)
from virtualgpu.frontend.scalar_eval import ScalarEvalResult, evaluate_ttgir_cta_offsets
from virtualgpu.frontend.triton_gemm import extract_gemm_kernel_ir

__all__ = [
    "ARTIFACT_HASH_VERSION",
    "CompiledArtifacts",
    "MockTensor",
    "PtxasReport",
    "ScalarEvalResult",
    "artifact_hashes",
    "compile_triton_aot",
    "evaluate_ttgir_cta_offsets",
    "hash_ptx",
    "hash_ttgir",
    "parse_ptxas_verbose",
    "specialize_triton_call",
    "extract_gemm_kernel_ir",
]
