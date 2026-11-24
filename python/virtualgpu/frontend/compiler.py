"""Ahead-of-time Triton compilation and ptxas resource/spill parsing (CPU-only)."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
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

ARTIFACT_HASH_VERSION = "1"


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
    """Compute a normalized SHA-256 over PTX, stripping comments and debug locs.

    Shared between offline AOT compilation and on-GPU runtime validation to
    verify that both paths compiled the exact same kernel.
    """
    norm_lines: list[str] = []
    for raw in ptx.splitlines():
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        if line.startswith((".loc ", ".file ", ".section .debug")):
            continue
        norm_lines.append(" ".join(line.split()))
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
) -> tuple[dict[str, Any], dict[str, str], dict[Any, Any], dict[Any, Any]]:
    """Derive (options, signature, constexprs, attrs) using Triton's own runtime binder.

    Guarantees AOT specialization (16B divisibility, equal-to-1 constexprs,
    do_not_specialize, integer bit widths) matches runtime JIT identically.
    """
    from triton.backends.compiler import GPUTarget
    from triton.compiler import make_backend
    from triton.runtime.jit import create_function_from_signature

    arch_num = parse_arch_number(target_arch)
    target = GPUTarget("cuda", arch_num, warp_size)
    backend = make_backend(target)
    binder = create_function_from_signature(fn.signature, fn.params, backend)
    bound_args, specialization, options = binder(*args, **kwargs)
    options_obj, signature, constexprs, attrs = fn._pack_args(
        backend, kwargs, bound_args, specialization, options
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
) -> CompiledArtifacts:
    """Compile a @triton.jit kernel ahead-of-time on CPU without a physical GPU."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    arch_num = parse_arch_number(target_arch)
    target = GPUTarget("cuda", arch_num, warp_size)
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs, attrs=attrs)
    compile_options = dict(options) if options is not None else {}
    compile_options.setdefault("num_warps", num_warps)
    compile_options.setdefault("num_stages", num_stages)

    compiled = triton.compile(src, target=target, options=compile_options)
    ttgir = compiled.asm["ttgir"]
    ptx = compiled.asm["ptx"]
    ptxas_report = run_ptxas_verbose(ptx, arch_num, options=compile_options)
    hashes = artifact_hashes(ptx, ttgir)

    return CompiledArtifacts(
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
        ptxas=ptxas_report,
    )
