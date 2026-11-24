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
from virtualgpu.autotune import (
    AutotuneReport,
    autotune,
    autotune_offline,
    prune_configs_by,
)

__all__ = [
    "__version__",
    "AutotuneReport",
    "autotune",
    "autotune_offline",
    "prune_configs_by",
    "extract_gemm_configs",
    "predict_gemm_configs",
    "predict_ir_dir",
    "predict_kernel_ir",
]
