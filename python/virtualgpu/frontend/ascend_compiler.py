"""Ahead-of-time Triton-Ascend compilation (CPU-only, no NPU hardware required)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from virtualgpu.frontend.compiler import (
    ARTIFACT_HASH_VERSION,
    _canonical_json,
    default_aot_cache_dir,
    hash_ttgir,
)

ASCEND_AOT_CACHE_VERSION = "2"

_CANONICAL_TO_BISHENGIR_ARCH: dict[str, str] = {
    "ascend910b1": "Ascend910B1",
    "ascend910b2": "Ascend910B2",
    "ascend910b3": "Ascend910B3",
    "ascend910b4": "Ascend910B4",
    "ascend910_9391": "Ascend910_9391",
    "ascend910_9392": "Ascend910_9392",
}

_DEFAULT_AI_CORES_BY_ARCH: dict[str, int] = {
    "ascend910b1": 24,
    "ascend910b2": 24,
    "ascend910b3": 20,
    "ascend910b4": 20,
}


def is_ascend_arch(target_arch: str) -> bool:
    """Return True if target_arch names a Huawei Ascend NPU target."""
    return target_arch.strip().lower().startswith("ascend")


def canonical_ascend_arch(target_arch: str) -> str:
    """Normalize an Ascend target arch string to lowercase (e.g. 'ascend910b3')."""
    norm = target_arch.strip().lower()
    if not norm.startswith("ascend"):
        raise ValueError(f"Expected Ascend arch starting with 'ascend', got {target_arch!r}")
    return norm


def bishengir_target_arch(target_arch: str) -> str:
    """Convert canonical arch ('ascend910b3') to BiShengIR --target format ('Ascend910B3')."""
    canon = canonical_ascend_arch(target_arch)
    if canon in _CANONICAL_TO_BISHENGIR_ARCH:
        return _CANONICAL_TO_BISHENGIR_ARCH[canon]
    # Generic fallback: Ascend910B3-style capitalization
    return re.sub(r"^ascend", "Ascend", canon).upper().replace("ASCEND", "Ascend", 1)


@dataclass(frozen=True)
class AscendCompiledArtifacts:
    """Artifacts collected from offline Triton-Ascend AOT compilation."""

    kernel_name: str
    target_arch: str
    triton_version: str
    ttir: str
    ttadapter: str
    hivm_ir: str
    ttir_sha256: str
    hivm_sha256: str
    mix_mode: str
    workspace_size: int
    bs_task_type: int
    requested_num_stages: int
    requested_multibuffer: bool


def hash_hivm(hivm_ir: str) -> str:
    """Compute a normalized SHA-256 over HIVM MLIR, stripping source locations and warnings."""
    norm_lines: list[str] = []
    for raw in hivm_ir.splitlines():
        s = raw.strip()
        if not s or s.startswith(("//", "bisheng:", "ld.lld:", "warning:", "1 warning")):
            continue
        norm_lines.append(" ".join(s.split()))
    return hashlib.sha256("\n".join(norm_lines).encode("utf-8")).hexdigest()


def _ascend_cache_key(
    *,
    fn: Any,
    signature: Mapping[str, str],
    constants: Mapping[str, Any],
    target_arch: str,
    num_stages: int,
    multibuffer: bool,
    extra_options: Mapping[str, Any],
    triton_version: str,
) -> str:
    src_text = getattr(fn, "src", None) or getattr(fn, "__name__", repr(fn))
    payload = {
        "cache_version": ASCEND_AOT_CACHE_VERSION,
        "artifact_hash_version": ARTIFACT_HASH_VERSION,
        "kernel_src_sha256": hashlib.sha256(src_text.encode("utf-8")).hexdigest(),
        "signature": dict(signature),
        "constants": _canonical_json(dict(constants)),
        "target_arch": canonical_ascend_arch(target_arch),
        "num_stages": num_stages,
        "multibuffer": multibuffer,
        "extra_options": _canonical_json(dict(extra_options)),
        "triton_version": triton_version,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_ascend_cache(key: str, cache_dir: Path) -> AscendCompiledArtifacts | None:
    p = cache_dir / "ascend_aot" / f"{key}.json"
    if not p.is_file():
        return None
    try:
        return AscendCompiledArtifacts(**json.loads(p.read_text(encoding="utf-8")))
    except (OSError, TypeError, ValueError):
        return None


def _store_ascend_cache(key: str, artifacts: AscendCompiledArtifacts, cache_dir: Path) -> None:
    out_dir = cache_dir / "ascend_aot"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_name = tempfile.mkstemp(prefix=".ascend_", suffix=".tmp", dir=str(out_dir))
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(asdict(artifacts), fh)
            os.replace(tmp_name, out_dir / f"{key}.json")
        except OSError:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
    except OSError:
        pass


def _mock_signature_for_fn(
    fn: Any,
    call_args: Sequence[Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Build Triton-Ascend ASTSource (signature, constants) without invoking NPU driver."""
    constexpr_names = {
        p.name for p in getattr(fn, "params", []) if getattr(p, "is_constexpr", False)
    }
    do_not_specialize = {
        p.name for p in getattr(fn, "params", []) if getattr(p, "do_not_specialize", False)
    }
    dtype_to_triton = {
        "torch.float16": "*fp16",
        "torch.bfloat16": "*bf16",
        "torch.float32": "*fp32",
        "torch.int8": "*i8",
        "torch.int32": "*i32",
        "float16": "*fp16",
        "bfloat16": "*bf16",
        "float32": "*fp32",
        "f16": "*fp16",
        "bf16": "*bf16",
        "f32": "*fp32",
    }
    signature: dict[str, str] = {}
    constants: dict[str, Any] = {}
    pos_iter = iter(call_args)

    for name in fn.arg_names:
        if name in constexpr_names:
            if name in config:
                constants[name] = config[name]
            continue
        val = next(pos_iter, None)
        if hasattr(val, "data_ptr") and hasattr(val, "dtype"):
            signature[name] = dtype_to_triton.get(str(val.dtype), "*fp16")
        elif isinstance(val, bool):
            signature[name] = "i1"
        elif isinstance(val, int):
            # Match Triton's runtime equal_to_1 specialization (e.g., contiguous stride == 1
            # allows BiShengIR to lower unmasked GEMM as pure AIC_ONLY without AIV workspace).
            if val == 1 and name not in do_not_specialize:
                constants[name] = 1
            else:
                signature[name] = "i32"
        elif isinstance(val, float):
            signature[name] = "fp32"
        else:
            signature[name] = "i32"
    return signature, constants


def compile_triton_ascend_aot(
    fn: Any,
    *,
    signature: Mapping[str, str],
    constants: Mapping[str, Any],
    target_arch: str = "ascend910b3",
    num_stages: int = 2,
    multibuffer: bool | None = None,
    extra_options: Mapping[str, Any] | None = None,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> AscendCompiledArtifacts:
    """Compile a @triton.jit kernel via Triton-Ascend's backend stages purely on CPU."""
    import triton
    from triton._C.libtriton import ir
    from triton.backends.ascend.driver import NPUUtils
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource, make_backend
    from triton.runtime.cache import get_dump_manager

    canon_arch = canonical_ascend_arch(target_arch)
    bs_arch = bishengir_target_arch(canon_arch)
    eff_multibuffer = (num_stages != 1) if multibuffer is None else bool(multibuffer)
    extra = dict(extra_options) if extra_options else {}

    resolved_cache = Path(cache_dir) if cache_dir is not None else default_aot_cache_dir()
    key = _ascend_cache_key(
        fn=fn,
        signature=signature,
        constants=constants,
        target_arch=canon_arch,
        num_stages=num_stages,
        multibuffer=eff_multibuffer,
        extra_options=extra,
        triton_version=triton.__version__,
    )
    if use_cache:
        cached = _load_ascend_cache(key, resolved_cache)
        if cached is not None:
            return cached

    # Stub NPUUtils arch/core queries so CPU-only compilation never touches the NPU driver.
    orig_get_arch = NPUUtils.get_arch
    orig_get_aicore = NPUUtils.get_aicore_num
    ai_cores = _DEFAULT_AI_CORES_BY_ARCH.get(canon_arch, 20)
    prev_dump_dir = os.environ.get("TRITON_DUMP_DIR")
    prev_kernel_dump = os.environ.get("TRITON_KERNEL_DUMP")
    prev_compile_only = os.environ.get("TRITON_COMPILE_ONLY")

    with tempfile.TemporaryDirectory(prefix="vgpu_ascend_dump_") as dump_root:
        try:
            NPUUtils.get_arch = lambda _self, _a=bs_arch: _a  # type: ignore[method-assign]
            NPUUtils.get_aicore_num = lambda _self, _c=ai_cores: _c  # type: ignore[method-assign]
            os.environ["TRITON_DUMP_DIR"] = dump_root
            os.environ["TRITON_KERNEL_DUMP"] = "1"
            os.environ["TRITON_COMPILE_ONLY"] = "1"

            target = GPUTarget("npu", bs_arch, 32)
            backend = make_backend(target)
            opts_dict: dict[str, Any] = {
                "debug": True,
                "num_stages": int(num_stages),
                "multibuffer": eff_multibuffer,
            }
            opts_dict.update(extra)
            options = backend.parse_options(opts_dict)

            context = ir.context()
            ir.load_dialects(context)
            backend.load_dialects(context)

            src = ASTSource(fn=fn, signature=dict(signature), constants=dict(constants))
            mod = src.make_ir(
                options, backend.get_codegen_implementation(), backend.get_module_map(), context
            )
            stages: dict[str, Any] = {}
            backend.add_stages(stages, options)
            metadata: dict[str, Any] = dict(options.__dict__)
            metadata["target"] = target
            metadata["hash"] = key

            mod = stages["ttir"](mod, metadata)
            ttir_text = str(mod)
            ttadapter_text = stages["ttadapter"](mod, metadata)
            _npubin = stages["npubin"](ttadapter_text, metadata)

            dump_dir = Path(get_dump_manager(key).cache_dir)
            hivm_path = dump_dir / "kernel.npuir.mlir"
            hivm_text = hivm_path.read_text(encoding="utf-8") if hivm_path.is_file() else ""
        finally:
            NPUUtils.get_arch = orig_get_arch  # type: ignore[method-assign]
            NPUUtils.get_aicore_num = orig_get_aicore  # type: ignore[method-assign]
            if prev_dump_dir is None:
                os.environ.pop("TRITON_DUMP_DIR", None)
            else:
                os.environ["TRITON_DUMP_DIR"] = prev_dump_dir
            if prev_kernel_dump is None:
                os.environ.pop("TRITON_KERNEL_DUMP", None)
            else:
                os.environ["TRITON_KERNEL_DUMP"] = prev_kernel_dump
            if prev_compile_only is None:
                os.environ.pop("TRITON_COMPILE_ONLY", None)
            else:
                os.environ["TRITON_COMPILE_ONLY"] = prev_compile_only

    artifacts = AscendCompiledArtifacts(
        kernel_name=str(metadata.get("kernel_name", getattr(fn, "__name__", "kernel"))),
        target_arch=canon_arch,
        triton_version=triton.__version__,
        ttir=ttir_text,
        ttadapter=ttadapter_text,
        hivm_ir=hivm_text,
        ttir_sha256=hash_ttgir(ttir_text),
        hivm_sha256=hash_hivm(hivm_text),
        mix_mode=str(metadata.get("mix_mode", "aic")),
        workspace_size=int(metadata.get("workspace_size") or 0),
        bs_task_type=int(metadata.get("bs_task_type") or 0),
        requested_num_stages=int(num_stages),
        requested_multibuffer=eff_multibuffer,
    )
    if use_cache:
        _store_ascend_cache(key, artifacts, resolved_cache)
    return artifacts
