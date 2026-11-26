"""Kernel frontends that lower source kernels to VirtualGPU KernelIR."""

from virtualgpu.frontend.ascend_compiler import (
    AscendCompiledArtifacts,
    bishengir_target_arch,
    canonical_ascend_arch,
    compile_triton_ascend_aot,
    hash_hivm,
    is_ascend_arch,
)
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
from virtualgpu.frontend.hivm import HivmMmaOp, HivmSummary, parse_hivm_summary
from virtualgpu.frontend.scalar_eval import ScalarEvalResult, evaluate_ttgir_cta_offsets
from virtualgpu.frontend.triton_ascend_gemm import (
    assemble_ascend_gemm_kernel_ir,
    extract_ascend_gemm_kernel_ir,
)
from virtualgpu.frontend.triton_attention import extract_attention_kernel_ir
from virtualgpu.frontend.triton_gemm import extract_gemm_kernel_ir

__all__ = [
    "ARTIFACT_HASH_VERSION",
    "AscendCompiledArtifacts",
    "CompiledArtifacts",
    "HivmMmaOp",
    "HivmSummary",
    "MockTensor",
    "PtxasReport",
    "ScalarEvalResult",
    "artifact_hashes",
    "assemble_ascend_gemm_kernel_ir",
    "bishengir_target_arch",
    "canonical_ascend_arch",
    "compile_triton_aot",
    "compile_triton_ascend_aot",
    "evaluate_ttgir_cta_offsets",
    "extract_ascend_gemm_kernel_ir",
    "extract_attention_kernel_ir",
    "extract_gemm_kernel_ir",
    "hash_hivm",
    "hash_ptx",
    "hash_ttgir",
    "is_ascend_arch",
    "parse_hivm_summary",
    "parse_ptxas_verbose",
    "specialize_triton_call",
]
