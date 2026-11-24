"""VirtualGPU: predict the relative performance of GPU kernel variants."""

try:
    from virtualgpu._core import __version__
except ImportError:
    __version__ = "0.1.0"

from virtualgpu.api import (
    extract_gemm_configs,
    predict_gemm_configs,
    predict_ir_dir,
    predict_kernel_ir,
)

__all__ = [
    "__version__",
    "extract_gemm_configs",
    "predict_gemm_configs",
    "predict_ir_dir",
    "predict_kernel_ir",
]
