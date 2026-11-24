# Proposal: Triton Frontend (`frontend.md`)

Owner: Agent 2  
Status: Draft for architecture alignment (no code written)  
Interface authority: Output conforms to Agent 3's Kernel IR (`docs/proposals/kernel-ir-and-l2.md`).

## 1. Scope

The Triton frontend transforms a Triton kernel, concrete launch parameters, and a target GPU specification into a structured **Kernel IR** and backend resource report, running entirely on CPU without a physical GPU.

Responsibilities:
1. **AOT compilation** for arbitrary GPU targets (`GPUTarget` + backend assembler).
2. **Post-pipelining TTGIR extraction** of loops, materialized pipeline stages, async memory ops, MMA shapes, and sync points.
3. **Minimal scalar evaluation** of `program_id` arithmetic to recover tile coordinates and byte footprints (L2 swizzle recovery).
4. **Structural detection** of persistent kernels and warp specialization.
5. **Backend resource and spill parsing** from `ptxas --verbose` (registers, shared memory, stack frame, spill stores/loads).

Out of scope: L2 reuse modeling (Agent 3), event-driven performance simulation (Agent 4), functional tensor execution, non-Triton kernels (cuBLAS/CUTLASS/CUDA), and reverse engineering of vendor binaries or SASS.

## 2. Inputs and Outputs

### Inputs
- **Kernel source**: A `@triton.jit` function (or a cached `(ttgir, ptx, ptxas_log)` bundle for reproducible offline runs).
- **Concrete invocation**: Tensor shapes, strides, dtypes, `constexpr`/autotune config (`BLOCK_M/N/K`, `GROUP_SIZE_M`, `num_warps`, `num_stages`, `maxnreg`), and grid dimensions (or grid lambda).
- **Target spec** (subset of GPU profile): `backend` (`cuda` | `hip`), `arch` (e.g., `sm_89`, `sm_121`, `sm_90`), `warp_size`, and required toolchain version.

### Outputs (aligned to Agent 3's `KernelIR` v0.1)
Output conforms directly to `KernelIR` v0.1 in `docs/proposals/kernel-ir-and-l2.md`:

| `KernelIR` Section | Frontend Mapping |
|---|---|
| `metadata` | `kernel_name`, `target_arch`, `frontend = "triton"`, `config`, `triton_version`, `ptxas_version` |
| `launch` | `grid`, `cluster_shape`, `num_warps`, `warp_groups` (1 `unified` group for standard kernels; split `producer`/`consumer` groups when warp-specialized), `dynamic_smem_bytes` |
| `resources` | `source = "ptxas"`, `registers_per_thread`, `static_smem_bytes`, `stack_frame_bytes`, `spill_stores_bytes`, `spill_loads_bytes`, `spill_scope` (`none` \| `prologue_epilogue` \| `inner_loop_confirmed` \| `inner_loop_assumed`), `barriers` |
| `tensors` | `id`, `dtype`, `element_bits`, `shape`, `strides`, `base_addr`, `cache_policy` (`ca`/`cg`/`cs`/`cv` from TTGIR load/store cache modifiers) |
| `body` | `outer_loop` (`trip_count_mode`: `uniform` \| `per_cta`, unifying standard `=1` and persistent `>1`); `prologue` / `pipeline_loop` (post-pipelining `num_stages`, `prefetch_distance`, `async_copy`: `none` \| `cp_async` \| `tma`, `ops` with `op_id`, `kind`, `warp_group_id`, `stage_offset`, `bytes_per_cta`, `transfer_width_bytes`, `tile_mnk`, `inst_mnk`, `op_class`, dtypes) / `epilogue` (with `overlaps_next_tile`) |
| `access_map` | Default `strided_2d_per_outer_tile` entries `[tile_coord_0, tile_coord_1, base_byte_offset, rows, row_bytes, row_stride_bytes, inner_step_byte_stride, is_masked]` evaluated per `(cta_id, outer_iter)` with boundary masks clamped into effective `rows`/`row_bytes`; degrades to `explicit_blocks` for irregular access |
| `diagnostics` | Soft confidence-lowering flags where simulation still runs (e.g., `ptxas_only_spill`, `unpipelined_fallback`) |
| `unsupported_features` | Hard-stop flags causing the simulator to return `unsupported` rather than guess (e.g., `atomic_work_stealing`, `cluster_smem`, `data_dependent_address`) |

## 3. Key Decisions and Trade-offs

### D1. AOT Compilation Pipeline: `triton.compile` + Direct `ptxas --verbose`
- **Decision**: Invoke Triton's public CPU-only compilation path (`triton.compile(ASTSource(...), target=GPUTarget(...), options=...)`) to produce post-pass `ttgir` and `ptx`, then run `ptxas --gpu-name <arch> --verbose` directly on the emitted PTX. Cache artifacts by `(kernel_ast_hash, signature, config, target, triton_ver, ptxas_ver)`.
- **Alternatives**:
  1. *Use `CompiledKernel.metadata` only*: Rejected. Triton's metadata exposes `n_regs`, `shared`, and a coarse `n_spills`, but omits separate spill store/load byte counts and stack frame size needed for spill modeling.
  2. *Parse Python AST directly*: Rejected. Misses Triton layout inference, MMA lowering, SMEM allocation, and whether the compiler actually pipelined the loop.
  3. *Disassemble cubin (`nvdisasm`)*: Rejected by the no-reverse-engineering principle and unnecessary for tile-level modeling.
- **Trade-off**: Requires a host `ptxas` binary that supports the target architecture (e.g., CUDA 12.9+/13.0+ for `sm_121`). Compilation caching keeps repeated autotune sweeps fast.

### D2. Extraction Stage: Post-Pipelining TTGIR (Not TTIR, Not LLIR/PTX)
- **Decision**: Parse the final `ttgir` emitted *after* `TritonGPUPipeline` and layout/MMA passes.
- **Alternatives**:
  1. *Pre-pass TTIR*: Rejected. `num_stages` is only a hint before pipelining; the compiler may silently clamp or disable pipelining when shared memory or loop structure prevents it.
  2. *LLIR or PTX*: Rejected. Tile shapes, tensor layouts, and multi-buffered stage descriptors are lowered into scalarized pointer arithmetic, making tile-level recovery fragile.
- **Trade-off**: TTGIR op names (`ttg.async_copy_global_to_local`, `ttng.async_tma_copy_global_to_local`, `ttg.local_alloc`, `ttg.warp_specialize`) drift across Triton releases. Isolate dialect matching behind a thin versioned adapter and parse MLIR text with a small dedicated parser rather than binding to LLVM/MLIR C++ internals.

### D3. Swizzle & Tile-Order Recovery: Minimal Scalar Evaluator on TTGIR
- **Decision**: Build a small deterministic scalar interpreter (~200-300 LOC) over the backward slice of global load/store addresses and masks (`tt.get_program_id`, `tt.get_num_programs`, `arith.*`, `scf.for` induction variables). Given concrete `(shapes, strides, grid, config)`, evaluate per-`pid` tile coordinates and byte ranges directly.
- **Alternatives**:
  1. *Regex/pattern matching for `GROUP_SIZE_M`*: Rejected. Breaks on diagonal swizzles, split-K, grouped GEMM, or non-standard index math.
  2. *Symbolic/polyhedral analysis*: Rejected as over-engineered; concrete launch dimensions are always known at simulation time.
  3. *CPU execution (`triton-cpu`)*: Rejected. Too slow and executes tensor math we do not need.
- **Trade-off**: Handles arbitrary integer swizzle logic cleanly. If an address slice depends on a runtime memory load (e.g., MoE routing or paged KV indices), evaluation stops on that operand and emits `data_dependent_address` in `unsupported_features` instead of guessing.

### D4. Persistent Kernel and Warp Specialization Recognition
- **Decision**: Detect persistent kernels structurally in TTGIR: an outer `scf.for` loop whose bounds/step depend on `tt.get_program_id` / `tt.get_num_programs` (or a `NUM_SMS` constant) and whose body wraps both tile-coordinate computation, an inner reduction `scf.for`, and an output tile store. Detect warp specialization from explicit TTGIR warp-partition ops (e.g., `ttg.warp_specialize` / producer-consumer barrier pairs) and record the producer/consumer warp split.
- **Alternatives**:
  1. *Assume 1-CTA-1-tile everywhere*: Rejected (fails on persistent GEMM and multi-tile kernels).
  2. *Require user annotations*: Rejected as primary mechanism because it breaks drop-in autotuning (optional hints may be supported as a fallback).
- **Trade-off**: Covers standard static-strided persistent loops cleanly; dynamic atomic work-stealing loops cannot be statically resolved and emit `atomic_work_stealing` in `unsupported_features`.

### D5. Register Spill Extraction and Attribution
- **Decision**: Parse `ptxas --verbose` for `registers`, `smem`, `stack frame`, `spill stores` (bytes), and `spill loads` (bytes). When spills are non-zero, scan the emitted PTX (public ISA) for `ld.local`/`st.local` placement relative to the inner loop:
  - If visible in PTX, tag `spill_scope` as `inner_loop_confirmed` or `prologue_epilogue`.
  - If `ptxas` reports spills not present in PTX (introduced during `ptxas` register allocation), conservatively tag `inner_loop_assumed` so Agent 4 models the worst-case per-iteration penalty with widened uncertainty.
- **Alternatives**:
  1. *Report total spill bytes only*: Rejected. A small prologue spill has negligible cost, whereas the same bytes spilled inside a K-loop scale with `K / BLOCK_K` and can flip config rankings.
- **Trade-off**: Without SASS inspection we cannot know the exact instruction placement of `ptxas`-only spills; explicit `inner_loop_assumed` tagging avoids false precision.

### D6. Initial Target Hardware Coverage (`sm_89` and `sm_121`)
- **Decision**: Prioritize op-table validation for the two available physical GPUs:
  - **RTX 4070 (`sm_89`, Ada)**: `cp.async` (`ttg.async_copy_global_to_local`) + warp-level `mma.sync`.
  - **DGX Spark GB10 (`sm_121`, client/edge Blackwell)**: TMA (`cp.async.bulk.tensor` / `ttng.async_tma_copy_global_to_local`) + warp-level `mma.sync` (including Blackwell block-scaled formats where exposed by the toolchain). Per NVIDIA public architecture/PTX documentation, `sm_121` does **not** expose datacenter `sm_100` `tcgen05`/TMEM/DSMEM pipelines or Hopper `wgmma`; exact supported MMA shapes and TMA ops must follow official CUDA 12.9+/13.0+ docs and be verified on GB10.
- Keep `sm_90` (`wgmma`) and AMD `gfx942` op mappings as extensible adapter tables, marked unvalidated until hardware is available.

## 4. Response to Aggregated Risks

| Risk | Frontend Mitigation |
|---|---|
| **R2 (Swizzle is plain `program_id` arithmetic)** | D3 scalar evaluator executes the exact `arith` slice per `pid` rather than pattern-matching `GROUP_SIZE_M`, recovering exact tile traversal order and byte offsets for Agent 3's L2 model. |
| **R3 (Register spill penalty)** | D1 + D5 parse exact `spill stores`, `spill loads`, and `stack frame` bytes from `ptxas --verbose`, plus `spill_scope` attribution so Agent 4 can penalize hot-loop spills accurately. |
| **R4 (Persistent kernels & warp specialization)** | D4 structurally detects outer persistent tile loops (emitting `execution_model = persistent` + outer loop trip count/stride + per-iteration tile sequence) and warp-specialization splits instead of assuming 1-CTA-1-tile. |
| **R5 (Post-pipelining stage structure)** | D2 extracts strictly from post-`TritonGPUPipeline` TTGIR, capturing the *actual* materialized stage count, multi-buffered SMEM allocations, async copy/TMA ops, and `wait_group`/`mbarrier` sync points rather than the requested `num_stages` parameter. |
| **R1 / R6 (Uncertainty propagation)** | Separates soft uncertainties (`diagnostics`: `ptxas_only_spill`, `unpipelined_fallback`) that widen indistinguishable bands from hard blockers (`unsupported_features`: `data_dependent_address`, `atomic_work_stealing`, `cluster_smem`) that return `unsupported`. |

## 5. What Must Be Measured / Verified Empirically

1. **AOT toolchain matrix for `sm_121` and `sm_89`**: Verify exact Triton version and host `ptxas` version (CUDA 12.9+/13.0+) needed to compile `GPUTarget("cuda", 121, 32)` and `GPUTarget("cuda", 89, 32)` on CPU without an attached GPU.
2. **Actual `sm_121` op emission**: Compile a representative GEMM autotune sweep for `sm_121` and inspect emitted TTGIR/PTX to confirm which MMA shapes, dtypes, and TMA vs `cp.async` paths Triton actually selects on GB10 (do not assume from `sm_90` or `sm_100`).
3. **Effective vs. requested `num_stages`**: Verify across configs (including high `num_stages` near SMEM limits) that post-pass TTGIR cleanly exposes when `TritonGPUPipeline` clamps or drops stages.
4. **AOT compile latency & caching**: Measure wall-clock time to compile 50-100 GEMM configs on CPU and confirm cache hit overhead is negligible compared to simulation time.

## 6. Open Questions / TBD

- **Implementation language boundary**: Driving `triton.compile` requires Python. If the core simulator (Agents 3 & 4) is written in Rust or C++, the frontend should serialize `KernelIR` to JSON/MessagePack (with optional binary sidecar for large `access_map` tables) at the process boundary; if Python, it can pass in-memory objects.
- **AMD backend priority (depends on project goals)**: Extracting `registers_per_thread`/`spill_*` for AMD (`gfx942`) requires parsing AMDGPU ELF metadata / `amdgpu` assembler output instead of `ptxas`. Defer until Phase 1 scope and target priority are finalized.
- **Triton version pinning**: Decide which Triton release line (e.g., 3.2/3.3+) to target first so TTGIR dialect adapters stay minimal.
- **Data-dependent kernels (depends on project goals)**: For future MoE/paged-attention support, define how an external workload trace feeds into the scalar evaluator when `data_dependent_address` is encountered.
