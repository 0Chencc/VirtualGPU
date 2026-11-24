"""Minimal TTGIR scalar evaluator for recovering swizzled per-CTA tile footprints."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PtrVal:
    """Symbolic pointer rooted at a kernel pointer argument with an element offset."""

    arg_index: int
    elem_offset: int = 0


@dataclass(frozen=True)
class ScalarEvalResult:
    """Per-CTA evaluated base element offsets for each pointer argument."""

    # Maps pointer arg index -> list of element offsets (length = grid_size)
    ptr_elem_offsets: dict[int, list[int]]
    # Optional 2D TMA coordinates: ptr arg index -> list of (coord0, coord1)
    tma_coords: dict[int, list[tuple[int, int]]]
    data_dependent: bool = False
    has_nested_loop: bool = False


_RE_FUNC_HEADER = re.compile(r"tt\.func\s+[^{]+\((.*)\)\s*(?:attributes\s*\{[^}]*\}\s*)?\{")
_RE_PARAM = re.compile(r"(%[A-Za-z0-9_]+)\s*:\s*([^,)]+)")
_RE_ASSIGN = re.compile(r"^\s*(%[A-Za-z0-9_]+)\s*=\s*(.+)$")
_RE_CONST_SCALAR = re.compile(r"^arith\.constant\s+(-?\d+|true|false)\s*:")
_RE_CONST_DENSE = re.compile(r"^arith\.constant\s+dense<(-?\d+|true|false)>\s*:")
_RE_GET_PID = re.compile(r"^tt\.get_program_id\s+(x|y|z)\b")
_RE_GET_NPROG = re.compile(r"^tt\.get_num_programs\s+(x|y|z)\b")
_RE_MAKE_RANGE = re.compile(r"^tt\.make_range\s*\{[^}]*start\s*=\s*(-?\d+)")
_RE_BINOP = re.compile(
    r"^arith\.(addi|subi|muli|divsi|divui|remsi|remui|minsi|maxsi|minui|maxui|andi|ori|xori|shli|shrsi|shrui)\s+(%[A-Za-z0-9_]+),\s*(%[A-Za-z0-9_]+)\b"
)
_RE_CMPI = re.compile(
    r"^arith\.cmpi\s+([a-z]+),\s*(%[A-Za-z0-9_]+),\s*(%[A-Za-z0-9_]+)\b"
)
_RE_SELECT = re.compile(
    r"^arith\.select\s+(%[A-Za-z0-9_]+),\s*(%[A-Za-z0-9_]+),\s*(%[A-Za-z0-9_]+)\b"
)
_RE_UNARY_PASS = re.compile(
    r"^(?:arith\.(?:extsi|extui|trunci|index_cast)|tt\.(?:splat|expand_dims|broadcast|bitcast)|ttg\.convert_layout)\s+(%[A-Za-z0-9_]+)\b"
)
_RE_ADDPTR = re.compile(
    r"^tt\.addptr\s+(%[A-Za-z0-9_]+),\s*(%[A-Za-z0-9_]+)\b"
)
_RE_TENSORMAP_CREATE = re.compile(
    r"ttng\.tensormap_create\s+(%[A-Za-z0-9_]+),\s*(%[A-Za-z0-9_]+),"
)
_RE_REINTERPRET_DESC = re.compile(
    r"^ttng\.reinterpret_tensor_descriptor\s+(%[A-Za-z0-9_]+)\b"
)
_RE_TMA_LOAD_STORE = re.compile(
    r"ttng\.async_tma_copy_(?:global_to_local|local_to_global)\s+(%[A-Za-z0-9_]+)\[(%[A-Za-z0-9_]+),\s*(%[A-Za-z0-9_]+)\]"
)
_RE_LOAD_USE = re.compile(
    r"(?:ttg\.async_copy_global_to_local|tt\.load)\s+(%[A-Za-z0-9_]+)\b"
)
_RE_STORE_USE = re.compile(r"tt\.store\s+(%[A-Za-z0-9_]+)\b")


def _c_div(a: int, b: int) -> int:
    if b == 0:
        return 0
    return int(a / b)


def _c_rem(a: int, b: int) -> int:
    if b == 0:
        return 0
    return a - int(a / b) * b


def _eval_cmpi(pred: str, a: int, b: int) -> int:
    if pred == "eq":
        return int(a == b)
    if pred == "ne":
        return int(a != b)
    if pred in ("slt", "ult"):
        return int(a < b)
    if pred in ("sle", "ule"):
        return int(a <= b)
    if pred in ("sgt", "ugt"):
        return int(a > b)
    if pred in ("sge", "uge"):
        return int(a >= b)
    return 0


def _parse_func_params(ttgir: str) -> list[tuple[str, bool]]:
    """Return list of (ssa_name, is_ptr) for the tt.func parameters."""
    m = _RE_FUNC_HEADER.search(ttgir)
    if not m:
        return []
    param_str = m.group(1)
    # Strip loc(...) annotations inside parameter list before splitting
    cleaned = re.sub(r'loc\("[^"]*"\([^)]*\)\)', "", param_str)
    cleaned = re.sub(r"loc\([^)]*\)", "", cleaned)
    cleaned = re.sub(r"\{[^}]*\}", "", cleaned)
    params: list[tuple[str, bool]] = []
    for part in cleaned.split(","):
        pm = _RE_PARAM.search(part.strip())
        if pm:
            ssa, ty = pm.group(1), pm.group(2).strip()
            params.append((ssa, "!tt.ptr<" in ty))
    return params


def _extract_top_level_lines(ttgir: str) -> tuple[list[str], list[str]]:
    """Split tt.func body into top-level lines (outside scf.for) and inner-loop lines."""
    func_pos = ttgir.find("tt.func ")
    if func_pos == -1:
        return [], []
    body_start = ttgir.find("{", func_pos)
    if body_start == -1:
        return [], []

    top_lines: list[str] = []
    loop_lines: list[str] = []
    depth = 0
    in_loop = False
    for raw in ttgir[body_start + 1 :].splitlines():
        line = raw.strip()
        if not line:
            continue
        opens = line.count("{")
        closes = line.count("}")
        if depth == 0 and line.startswith("}"):
            break
        if depth == 0 and "scf.for " in line:
            in_loop = True
            depth += opens - closes
            continue
        if in_loop:
            depth += opens - closes
            if depth <= 0:
                in_loop = False
                depth = 0
            else:
                loop_lines.append(line)
        else:
            top_lines.append(line)
            depth += opens - closes
    return top_lines, loop_lines


def _compile_instructions(lines: list[str]) -> tuple[
    list[tuple[str, ...]],
    dict[str, str],
    list[tuple[str, str, str]],
    list[str],
]:
    """Parse SSA lines into compact evaluator instructions plus pointer/TMA sinks."""
    instrs: list[tuple[str, ...]] = []
    # desc_scratch_ssa -> ptr_ssa, desc_ssa -> scratch_ssa
    scratch_to_ptr: dict[str, str] = {}
    tma_uses: list[tuple[str, str, str]] = []  # (desc_ssa, c0_ssa, c1_ssa)
    ptr_sinks: list[str] = []

    for line in lines:
        m_tm = _RE_TENSORMAP_CREATE.search(line)
        if m_tm:
            scratch_to_ptr[m_tm.group(1)] = m_tm.group(2)
            continue

        m_tma = _RE_TMA_LOAD_STORE.search(line)
        if m_tma:
            tma_uses.append((m_tma.group(1), m_tma.group(2), m_tma.group(3)))
            continue

        m_ld = _RE_LOAD_USE.search(line)
        if m_ld:
            ptr_sinks.append(m_ld.group(1))
        m_st = _RE_STORE_USE.search(line)
        if m_st:
            ptr_sinks.append(m_st.group(1))

        ma = _RE_ASSIGN.match(line)
        if not ma:
            continue
        dst, rhs = ma.group(1), ma.group(2)

        if m := _RE_CONST_SCALAR.match(rhs):
            v = m.group(1)
            val = 1 if v == "true" else (0 if v == "false" else int(v))
            instrs.append(("const", dst, str(val)))
        elif m := _RE_CONST_DENSE.match(rhs):
            v = m.group(1)
            val = 1 if v == "true" else (0 if v == "false" else int(v))
            instrs.append(("const", dst, str(val)))
        elif m := _RE_GET_PID.match(rhs):
            instrs.append(("pid", dst, m.group(1)))
        elif m := _RE_GET_NPROG.match(rhs):
            instrs.append(("nprog", dst, m.group(1)))
        elif m := _RE_MAKE_RANGE.match(rhs):
            instrs.append(("const", dst, m.group(1)))
        elif m := _RE_BINOP.match(rhs):
            instrs.append(("binop", dst, m.group(1), m.group(2), m.group(3)))
        elif m := _RE_CMPI.match(rhs):
            instrs.append(("cmpi", dst, m.group(1), m.group(2), m.group(3)))
        elif m := _RE_SELECT.match(rhs):
            instrs.append(("select", dst, m.group(1), m.group(2), m.group(3)))
        elif m := _RE_ADDPTR.match(rhs):
            instrs.append(("addptr", dst, m.group(1), m.group(2)))
        elif m := _RE_REINTERPRET_DESC.match(rhs):
            if m.group(1) in scratch_to_ptr:
                scratch_to_ptr[dst] = scratch_to_ptr[m.group(1)]
        elif m := _RE_UNARY_PASS.match(rhs):
            instrs.append(("pass", dst, m.group(1)))
        elif rhs.startswith("tt.load "):
            instrs.append(("load_unknown", dst))

    return instrs, scratch_to_ptr, tma_uses, ptr_sinks


def evaluate_ttgir_cta_offsets(
    ttgir: str,
    grid: tuple[int, int, int],
    runtime_int_args: dict[str, int],
    param_int_values: list[int | None] | None = None,
) -> ScalarEvalResult:
    """Evaluate TTGIR address arithmetic for every CTA in grid.

    Recovers exact per-CTA base element offsets (and TMA 2D coordinates)
    for arbitrary swizzle expressions (e.g., GROUP_SIZE_M, diagonal, 2D grids)
    without pattern-matching kernel idioms.
    """
    params = _parse_func_params(ttgir)
    top_lines, loop_lines = _extract_top_level_lines(ttgir)
    has_nested_loop = any("scf.for " in ln for ln in loop_lines)
    if has_nested_loop:
        # Persistent kernels compute per-tile coordinates inside an outer scf.for;
        # do not silently return incomplete single-iteration offsets.
        return ScalarEvalResult(
            ptr_elem_offsets={},
            tma_coords={},
            data_dependent=False,
            has_nested_loop=True,
        )

    instrs, scratch_to_ptr, tma_uses, ptr_sinks = _compile_instructions(top_lines)

    # Also collect ptr/TMA sinks inside the loop if prologue didn't issue loads
    # (e.g., unpipelined kernels where tt.load is only inside scf.for).
    _, _, loop_tma, loop_sinks = _compile_instructions(loop_lines)
    all_ptr_sinks = ptr_sinks + [s for s in loop_sinks if s not in ptr_sinks]
    all_tma_uses = tma_uses + [u for u in loop_tma if u not in tma_uses]

    # Build base environment for kernel arguments.
    base_env: dict[str, Any] = {}
    ptr_arg_indices: dict[str, int] = {}
    ptr_counter = 0
    for idx, (ssa, is_ptr) in enumerate(params):
        if is_ptr:
            base_env[ssa] = PtrVal(arg_index=ptr_counter, elem_offset=0)
            ptr_arg_indices[ssa] = ptr_counter
            ptr_counter += 1
        else:
            name = ssa.lstrip("%")
            if param_int_values is not None and idx < len(param_int_values) and param_int_values[idx] is not None:
                base_env[ssa] = int(param_int_values[idx])  # type: ignore[arg-type]
            elif name in runtime_int_args:
                base_env[ssa] = int(runtime_int_args[name])

    gx, gy, gz = grid
    ptr_elem_offsets: dict[int, list[int]] = {}
    tma_coords: dict[int, list[tuple[int, int]]] = {}
    data_dependent = False

    # Map TMA descriptor SSA -> ptr arg index
    desc_to_ptr_idx: dict[str, int] = {}
    for desc_ssa, ptr_ssa in scratch_to_ptr.items():
        if ptr_ssa in ptr_arg_indices:
            desc_to_ptr_idx[desc_ssa] = ptr_arg_indices[ptr_ssa]

    for pz in range(gz):
        for py in range(gy):
            for px in range(gx):
                env = dict(base_env)
                for ins in instrs:
                    op = ins[0]
                    dst = ins[1]
                    if op == "const":
                        env[dst] = int(ins[2])
                    elif op == "pid":
                        axis = ins[2]
                        env[dst] = px if axis == "x" else (py if axis == "y" else pz)
                    elif op == "nprog":
                        axis = ins[2]
                        env[dst] = gx if axis == "x" else (gy if axis == "y" else gz)
                    elif op == "pass":
                        if ins[2] in env:
                            env[dst] = env[ins[2]]
                    elif op == "binop":
                        bop, a_s, b_s = ins[2], ins[3], ins[4]
                        av, bv = env.get(a_s), env.get(b_s)
                        if isinstance(av, int) and isinstance(bv, int):
                            if bop == "addi":
                                env[dst] = av + bv
                            elif bop == "subi":
                                env[dst] = av - bv
                            elif bop == "muli":
                                env[dst] = av * bv
                            elif bop in ("divsi", "divui"):
                                env[dst] = _c_div(av, bv)
                            elif bop in ("remsi", "remui"):
                                env[dst] = _c_rem(av, bv)
                            elif bop in ("minsi", "minui"):
                                env[dst] = min(av, bv)
                            elif bop in ("maxsi", "maxui"):
                                env[dst] = max(av, bv)
                            elif bop == "andi":
                                env[dst] = av & bv
                            elif bop == "ori":
                                env[dst] = av | bv
                            elif bop == "xori":
                                env[dst] = av ^ bv
                            elif bop == "shli":
                                env[dst] = av << bv
                            elif bop in ("shrsi", "shrui"):
                                env[dst] = av >> bv
                        elif av == "_DATA_DEP" or bv == "_DATA_DEP":
                            env[dst] = "_DATA_DEP"
                    elif op == "cmpi":
                        pred, a_s, b_s = ins[2], ins[3], ins[4]
                        av, bv = env.get(a_s), env.get(b_s)
                        if isinstance(av, int) and isinstance(bv, int):
                            env[dst] = _eval_cmpi(pred, av, bv)
                    elif op == "select":
                        c_s, t_s, f_s = ins[2], ins[3], ins[4]
                        cv = env.get(c_s)
                        if isinstance(cv, int):
                            chosen = env.get(t_s) if cv else env.get(f_s)
                            if chosen is not None:
                                env[dst] = chosen
                    elif op == "addptr":
                        p_s, o_s = ins[2], ins[3]
                        pv, ov = env.get(p_s), env.get(o_s)
                        if isinstance(pv, PtrVal) and isinstance(ov, int):
                            env[dst] = PtrVal(pv.arg_index, pv.elem_offset + ov)
                        elif ov == "_DATA_DEP":
                            data_dependent = True
                    elif op == "load_unknown":
                        env[dst] = "_DATA_DEP"

                # Record the first evaluated offset for each pointer arg on this CTA.
                seen_ptrs: set[int] = set()
                for sink in all_ptr_sinks:
                    val = env.get(sink)
                    if isinstance(val, PtrVal) and val.arg_index not in seen_ptrs:
                        seen_ptrs.add(val.arg_index)
                        ptr_elem_offsets.setdefault(val.arg_index, []).append(val.elem_offset)

                seen_tma: set[int] = set()
                for desc_ssa, c0_ssa, c1_ssa in all_tma_uses:
                    p_idx = desc_to_ptr_idx.get(desc_ssa)
                    if p_idx is None or p_idx in seen_tma:
                        continue
                    c0, c1 = env.get(c0_ssa), env.get(c1_ssa)
                    if isinstance(c0, int) and isinstance(c1, int):
                        seen_tma.add(p_idx)
                        tma_coords.setdefault(p_idx, []).append((c0, c1))

    return ScalarEvalResult(
        ptr_elem_offsets=ptr_elem_offsets,
        tma_coords=tma_coords,
        data_dependent=data_dependent,
        has_nested_loop=False,
    )
