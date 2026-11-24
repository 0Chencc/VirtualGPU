"""VirtualGPU: predict the relative performance of GPU kernel variants."""

try:
    from virtualgpu._core import __version__
except ImportError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
