"""Ahead-of-time Triton compilation and ptxas resource/spill parsing (CPU-only)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from typing import Any


_CANONICAL_TO_TORCH_DTYPE: dict[str, str] = {
    "f16": "float16",
    "fp16": "float16",
    "bf16": "bfloat16",
    "f32": "float32",
    "fp32": "float32",
    "f64": "float64",
    "i8": "int8",
    "u8": "uint8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
    "f8E4M3FN": "float8_e4m3fn",
    "f8E5M2": "float8_e5m2",
}


class MockTensor:
    """Minimal tensor stand-in for CPU-only Triton JIT argument specialization.

    Exposes the same ``data_ptr()`` and ``dtype`` interface that Triton's
    runtime binder inspects, without requiring PyTorch or a physical GPU.
    """

    def __init__(self, dtype: str, ptr: int = 0x10000) -> None:
        torch_dt = _CANONICAL_TO_TORCH_DTYPE.get(dtype, dtype)
        self._ptr = int(ptr)
        self._torch_dtype = torch_dt
        self.dtype = type("_MockDType", (), {"__str__": lambda _s: f"torch.{torch_dt}"})()

    def data_ptr(self) -> int:
        return self._ptr


@dataclass(frozen=True)
class PtxasReport:
    """Resource usage reported by ptxas --verbose."""

    assembler: str
    assembler_version: str
    target_sm: str
    registers_per_thread: int
    barriers: int
    stack_frame_bytes: int
    spill_stores_bytes: int
    spill_loads_bytes: int
    spill_scope: str
    ptxas_only_spill: bool
    raw_stderr: str


@dataclass(frozen=True)
class CompiledArtifacts:
    """Artifacts collected from offline Triton AOT compilation."""

    kernel_name: str
    target_arch: str
    triton_version: str
    ttgir: str
    ptx: str
    ttgir_sha256: str
    ptx_sha256: str
    static_smem_bytes: int
    requested_num_warps: int
    requested_num_stages: int
    enable_fp_fusion: bool
    ptxas: PtxasReport


_RE_PTX_TARGET = re.compile(r"^\s*\.target\s+(sm_[0-9a-zA-Z_]+)", re.MULTILINE)
_RE_SPILL_LINE = re.compile(
    r"(\d+)\s+bytes\s+stack\s+frame,\s*(\d+)\s+bytes\s+spill\s+stores,\s*(\d+)\s+bytes\s+spill\s+loads"
)
_RE_USED_LINE = re.compile(
    r"Used\s+(\d+)\s+registers(?:,\s*used\s+(\d+)\s+barriers)?"
)
_RE_VERSION = re.compile(r"release\s+([0-9.]+),\s*V([0-9.]+)")
_RE_PTX_LOCAL_LDST = re.compile(r"\b(?:ld|st)\.local\b")
_RE_TTGIR_LOC_DEF = re.compile(r"^#loc\d*\s*=.*$")

ARTIFACT_HASH_VERSION = "2"


def _strip_ttgir_locs(line: str) -> str:
    """Strip all loc(...) annotations from a TTGIR line, handling nested parens."""
    out: list[str] = []
    i = 0
    n = len(line)
    while i < n:
        if line.startswith("loc(", i) and (i == 0 or not (line[i - 1].isalnum() or line[i - 1] == "_")):
            depth = 0
            j = i + 3
            while j < n:
                if line[j] == "(":
                    depth += 1
                elif line[j] == ")":
                    depth -= 1
                    if depth == 0:
                        j += 1
                        break
                j += 1
            # Remove a single preceding space if we stripped a loc(...)
            if out and out[-1] == " ":
                out.pop()
            i = j
        else:
            out.append(line[i])
            i += 1
    return "".join(out).strip()


def hash_ptx(ptx: str) -> str:
    """Compute a normalized SHA-256 over PTX, stripping comments, .file/.loc, and debug sections.

    Shared between offline AOT compilation and on-GPU runtime validation to
    verify that both paths compiled the exact same kernel regardless of source
    file path or line numbers.
    """
    norm_lines: list[str] = []
    in_debug_section = False
    debug_brace_depth = 0
    for raw in ptx.splitlines():
        line = " ".join(raw.split("//", 1)[0].split())
        if not line:
            continue
        if in_debug_section:
            debug_brace_depth += line.count("{") - line.count("}")
            if debug_brace_depth <= 0:
                in_debug_section = False
                debug_brace_depth = 0
            continue
        if line.startswith(".section .debug"):
            in_debug_section = True
            debug_brace_depth = line.count("{") - line.count("}")
            continue
        if line.startswith((".loc ", ".file ")) or line in (".loc", ".file"):
            continue
        norm_lines.append(line)
    payload = "\n".join(norm_lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def hash_ttgir(ttgir: str) -> str:
    """Compute a normalized SHA-256 over TTGIR, stripping all source location metadata."""
    norm_lines: list[str] = []
    for raw in ttgir.splitlines():
        stripped = raw.strip()
        if not stripped or _RE_TTGIR_LOC_DEF.match(stripped):
            continue
        cleaned = _strip_ttgir_locs(stripped)
        if cleaned:
            norm_lines.append(cleaned)
    payload = "\n".join(norm_lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def artifact_hashes(ptx: str, ttgir: str) -> dict[str, str]:
    """Return the versioned AOT/JIT artifact consistency hash bundle."""
    return {
        "artifact_hash_version": ARTIFACT_HASH_VERSION,
        "ptx_sha256": hash_ptx(ptx),
        "ttgir_sha256": hash_ttgir(ttgir),
    }


def specialize_triton_call(
    fn: Any,
    target_arch: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    warp_size: int = 32,
    enable_fp_fusion: bool = True,
) -> tuple[dict[str, Any], dict[str, str], dict[Any, Any], dict[Any, Any]]:
    """Derive (options, signature, constexprs, attrs) using Triton's own runtime binder.

    Guarantees AOT specialization (16B divisibility, equal-to-1 constexprs,
    do_not_specialize, integer bit widths) matches runtime JIT identically.
    Always passes `enable_fp_fusion` explicitly rather than relying on Triton's
    version-dependent default.
    """
    from triton.backends.compiler import GPUTarget
    from triton.compiler import make_backend
    from triton.runtime.jit import create_function_from_signature

    arch_num = parse_arch_number(target_arch)
    target = GPUTarget("cuda", arch_num, warp_size)
    backend = make_backend(target)
    binder = create_function_from_signature(fn.signature, fn.params, backend)
    call_kwargs = dict(kwargs)
    call_kwargs.setdefault("enable_fp_fusion", enable_fp_fusion)
    bound_args, specialization, options = binder(*args, **call_kwargs)
    options_obj, signature, constexprs, attrs = fn._pack_args(
        backend, call_kwargs, bound_args, specialization, options
    )
    return dict(options_obj.__dict__), signature, constexprs, attrs


def find_bundled_ptxas(arch_num: int) -> tuple[str, str]:
    """Locate the Triton-bundled ptxas binary appropriate for the target SM."""
    import triton

    bin_dir = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin")
    candidates = ("ptxas-blackwell", "ptxas") if arch_num >= 100 else ("ptxas",)
    for name in candidates:
        path = os.path.join(bin_dir, name)
        if os.path.isfile(path):
            return name, path
    raise FileNotFoundError(f"Could not find ptxas binary under {bin_dir}")


def get_ptxas_version(ptxas_path: str) -> str:
    """Return the version string of a ptxas binary."""
    out = subprocess.check_output([ptxas_path, "--version"], text=True, stderr=subprocess.STDOUT)
    m = _RE_VERSION.search(out)
    if m:
        return f"{m.group(1)} (V{m.group(2)})"
    return out.strip().splitlines()[-1] if out.strip() else "unknown"


def detect_ptx_target(ptx: str, fallback_arch: int) -> str:
    """Extract the exact .target directive from PTX (e.g. sm_89 or sm_121a)."""
    m = _RE_PTX_TARGET.search(ptx)
    if m:
        return m.group(1)
    return f"sm_{fallback_arch}a" if fallback_arch >= 90 else f"sm_{fallback_arch}"


def classify_spill_scope(ptx: str, spill_stores: int, spill_loads: int) -> tuple[str, bool]:
    """Classify where spills occur and whether they are ptxas-only."""
    if spill_stores == 0 and spill_loads == 0:
        return "none", False

    has_ptx_local = bool(_RE_PTX_LOCAL_LDST.search(ptx))
    if not has_ptx_local:
        # Spills were introduced by ptxas register allocation; conservatively
        # attribute to the hot inner loop and flag lower confidence.
        return "inner_loop_assumed", True

    # Check if any ld.local/st.local appears inside a PTX backward-branch loop body.
    lines = ptx.splitlines()
    labels: dict[str, int] = {}
    for idx, line in enumerate(lines):
        lm = re.match(r"^\s*(\$[A-Za-z0-9_]+):", line)
        if lm:
            labels[lm.group(1)] = idx

    loop_ranges: list[tuple[int, int]] = []
    for idx, line in enumerate(lines):
        bm = re.search(r"\bbra(?:\.uni)?\s+(\$[A-Za-z0-9_]+)", line)
        if bm and bm.group(1) in labels and labels[bm.group(1)] < idx:
            loop_ranges.append((labels[bm.group(1)], idx))

    for idx, line in enumerate(lines):
        if _RE_PTX_LOCAL_LDST.search(line):
            if any(start <= idx <= end for start, end in loop_ranges):
                return "inner_loop_confirmed", False

    return "prologue_epilogue", False


def parse_ptxas_verbose(
    stderr: str,
    ptx: str,
    assembler: str,
    assembler_version: str,
    target_sm: str,
) -> PtxasReport:
    """Parse register, barrier, stack frame, and spill metrics from ptxas -v stderr."""
    stack_frame = 0
    spill_stores = 0
    spill_loads = 0
    registers = 0
    barriers = 0

    m_spill = _RE_SPILL_LINE.search(stderr)
    if m_spill:
        stack_frame = int(m_spill.group(1))
        spill_stores = int(m_spill.group(2))
        spill_loads = int(m_spill.group(3))

    m_used = _RE_USED_LINE.search(stderr)
    if m_used:
        registers = int(m_used.group(1))
        if m_used.group(2) is not None:
            barriers = int(m_used.group(2))

    spill_scope, ptxas_only = classify_spill_scope(ptx, spill_stores, spill_loads)
    return PtxasReport(
        assembler=assembler,
        assembler_version=assembler_version,
        target_sm=target_sm,
        registers_per_thread=registers,
        barriers=barriers,
        stack_frame_bytes=stack_frame,
        spill_stores_bytes=spill_stores,
        spill_loads_bytes=spill_loads,
        spill_scope=spill_scope,
        ptxas_only_spill=ptxas_only,
        raw_stderr=stderr,
    )


def run_ptxas_verbose(
    ptx: str,
    arch_num: int,
    options: dict[str, Any] | None = None,
) -> PtxasReport:
    """Assemble PTX with ptxas -v on CPU using the same flags as Triton's CUDABackend."""
    assembler_name, ptxas_path = find_bundled_ptxas(arch_num)
    target_sm = detect_ptx_target(ptx, arch_num)
    version = get_ptxas_version(ptxas_path)

    opts = options or {}
    flags: list[str] = ["-lineinfo"]
    if opts.get("enable_fp_fusion") is False:
        flags.append("--fmad=false")
    flags.extend(["-v", "--regAllocOptLevel=2"])
    ptx_extra = opts.get("ptx_options")
    if ptx_extra:
        flags.extend(str(ptx_extra).split())
    flags.append(f"--gpu-name={target_sm}")

    with tempfile.TemporaryDirectory(prefix="vgpu_ptxas_") as tmpdir:
        ptx_file = os.path.join(tmpdir, "kernel.ptx")
        cubin_file = os.path.join(tmpdir, "kernel.cubin")
        with open(ptx_file, "w", encoding="utf-8") as f:
            f.write(ptx)
        proc = subprocess.run(
            [ptxas_path, *flags, ptx_file, "-o", cubin_file],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"{assembler_name} failed for {target_sm} (rc={proc.returncode}): {proc.stderr.strip()}"
            )

    return parse_ptxas_verbose(
        stderr=proc.stderr,
        ptx=ptx,
        assembler=assembler_name,
        assembler_version=version,
        target_sm=target_sm,
    )


def parse_arch_number(target_arch: str) -> int:
    """Parse an NVIDIA SM string like 'sm_89' or 'sm_121' into its integer code."""
    m = re.fullmatch(r"sm_(\d+)a?", target_arch)
    if not m:
        raise ValueError(f"Unsupported target_arch format: {target_arch!r} (expected e.g. 'sm_89', 'sm_121')")
    return int(m.group(1))


_MEMORY_CACHE: dict[str, CompiledArtifacts] = {}


def default_aot_cache_dir() -> Path:
    """Return the default on-disk cache directory for AOT compilation artifacts."""
    env = os.environ.get("VIRTUALGPU_CACHE_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "virtualgpu" / "aot"


def _canonical_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _canonical_json(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonical_json(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def _compute_aot_cache_key(
    *,
    fn: Any,
    signature: dict[str, str],
    constexprs: dict[Any, Any],
    attrs: dict[Any, Any] | None,
    options: dict[str, Any],
    target_arch: str,
    warp_size: int,
    triton_version: str,
    assembler_version: str,
) -> str:
    """Compute a deterministic cache key from source, signature, config, target, and versions."""
    src_text = getattr(fn, "src", None) or getattr(fn, "__name__", repr(fn))
    source_sha = hashlib.sha256(src_text.encode("utf-8")).hexdigest()
    # Keep only codegen-relevant options so transient paths/objects do not bust cache.
    opt_keys = (
        "num_warps",
        "num_stages",
        "num_ctas",
        "warp_size",
        "maxnreg",
        "enable_fp_fusion",
        "ptx_options",
        "default_dot_input_precision",
        "sanitize_overflow",
        "debug",
    )
    filtered_opts = {k: options.get(k) for k in opt_keys if k in options}
    payload = {
        "cache_format": "1",
        "artifact_hash_version": ARTIFACT_HASH_VERSION,
        "source_sha256": source_sha,
        "signature": _canonical_json(signature),
        "constexprs": _canonical_json(constexprs),
        "attrs": _canonical_json(attrs or {}),
        "options": _canonical_json(filtered_opts),
        "target_arch": target_arch,
        "warp_size": warp_size,
        "triton_version": triton_version,
        "assembler_version": assembler_version,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifacts_to_dict(art: CompiledArtifacts) -> dict[str, Any]:
    return {
        "kernel_name": art.kernel_name,
        "target_arch": art.target_arch,
        "triton_version": art.triton_version,
        "ttgir": art.ttgir,
        "ptx": art.ptx,
        "ttgir_sha256": art.ttgir_sha256,
        "ptx_sha256": art.ptx_sha256,
        "static_smem_bytes": art.static_smem_bytes,
        "requested_num_warps": art.requested_num_warps,
        "requested_num_stages": art.requested_num_stages,
        "enable_fp_fusion": art.enable_fp_fusion,
        "ptxas": asdict(art.ptxas),
    }


def _artifacts_from_dict(data: dict[str, Any]) -> CompiledArtifacts:
    return CompiledArtifacts(
        kernel_name=data["kernel_name"],
        target_arch=data["target_arch"],
        triton_version=data["triton_version"],
        ttgir=data["ttgir"],
        ptx=data["ptx"],
        ttgir_sha256=data["ttgir_sha256"],
        ptx_sha256=data["ptx_sha256"],
        static_smem_bytes=int(data["static_smem_bytes"]),
        requested_num_warps=int(data["requested_num_warps"]),
        requested_num_stages=int(data["requested_num_stages"]),
        enable_fp_fusion=bool(data.get("enable_fp_fusion", True)),
        ptxas=PtxasReport(**data["ptxas"]),
    )


def compile_triton_aot(
    fn: Any,
    signature: dict[str, str],
    constexprs: dict[Any, Any],
    target_arch: str,
    num_warps: int,
    num_stages: int,
    attrs: dict[Any, Any] | None = None,
    options: dict[str, Any] | None = None,
    warp_size: int = 32,
    enable_fp_fusion: bool = True,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> CompiledArtifacts:
    """Compile a @triton.jit kernel ahead-of-time on CPU without a physical GPU."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    arch_num = parse_arch_number(target_arch)
    compile_options = dict(options) if options is not None else {}
    compile_options.setdefault("num_warps", num_warps)
    compile_options.setdefault("num_stages", num_stages)
    compile_options.setdefault("enable_fp_fusion", enable_fp_fusion)
    resolved_fp_fusion = bool(compile_options["enable_fp_fusion"])

    cache_key: str | None = None
    disk_path: Path | None = None
    if use_cache:
        _, ptxas_path = find_bundled_ptxas(arch_num)
        assembler_ver = get_ptxas_version(ptxas_path)
        cache_key = _compute_aot_cache_key(
            fn=fn,
            signature=signature,
            constexprs=constexprs,
            attrs=attrs,
            options=compile_options,
            target_arch=f"sm_{arch_num}",
            warp_size=warp_size,
            triton_version=triton.__version__,
            assembler_version=assembler_ver,
        )
        if cache_key in _MEMORY_CACHE:
            return _MEMORY_CACHE[cache_key]
        resolved_dir = Path(cache_dir) if cache_dir is not None else default_aot_cache_dir()
        disk_path = resolved_dir / f"{cache_key}.json"
        if disk_path.is_file():
            try:
                art = _artifacts_from_dict(json.loads(disk_path.read_text(encoding="utf-8")))
                _MEMORY_CACHE[cache_key] = art
                return art
            except (OSError, KeyError, ValueError, TypeError):
                pass

    target = GPUTarget("cuda", arch_num, warp_size)
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs, attrs=attrs)
    compiled = triton.compile(src, target=target, options=compile_options)
    ttgir = compiled.asm["ttgir"]
    ptx = compiled.asm["ptx"]
    ptxas_report = run_ptxas_verbose(ptx, arch_num, options=compile_options)
    hashes = artifact_hashes(ptx, ttgir)

    artifacts = CompiledArtifacts(
        kernel_name=getattr(compiled.metadata, "name", fn.__name__),
        target_arch=f"sm_{arch_num}",
        triton_version=triton.__version__,
        ttgir=ttgir,
        ptx=ptx,
        ttgir_sha256=hashes["ttgir_sha256"],
        ptx_sha256=hashes["ptx_sha256"],
        static_smem_bytes=int(getattr(compiled.metadata, "shared", 0) or 0),
        requested_num_warps=num_warps,
        requested_num_stages=num_stages,
        enable_fp_fusion=resolved_fp_fusion,
        ptxas=ptxas_report,
    )

    if use_cache and cache_key is not None:
        _MEMORY_CACHE[cache_key] = artifacts
        if disk_path is not None:
            try:
                disk_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_fd, tmp_name = tempfile.mkstemp(prefix=".aot_", suffix=".tmp", dir=str(disk_path.parent))
                try:
                    with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                        json.dump(_artifacts_to_dict(artifacts), f)
                    os.replace(tmp_name, disk_path)
                except OSError:
                    if os.path.exists(tmp_name):
                        os.unlink(tmp_name)
            except OSError:
                pass

    return artifacts
