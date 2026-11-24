# Proposal: Extending the Triton Frontend to Fused Attention

Owner: Agent 2 (`s2/attention`)  
Status: Analysis for review (no schema or extractor code changes yet)  
Target: Forward FlashAttention-style fused attention (Triton 3.9 tutorial kernel, `sm_89` first)

## 1. Scope

This document analyzes how Triton's forward fused-attention kernel lowers on `sm_89` (RTX 4070) after software pipelining, how its tile structure maps to `KernelIR` v0.1 (`schemas/kernel_ir.schema.json`), and which minimal schema extensions are needed from Agent 3 before implementing the extractor.

Out of scope for this step: backward pass, paged KV cache, FlashDecoding split-KV, and `sm_121` TMA validation (deferred until Phase 4 hardware runs).

## 2. Empirical Post-Pipelining Observations (`sm_89`, Triton 3.9.0)

We compiled the Triton 3.9 fused-attention forward kernel offline (`CUDA_VISIBLE_DEVICES=""`, `ptxas 12.9 V12.9.86`, `Z=4, H=32, N_CTX=2048`, `fp16` I/O, `fp32` accumulator) using `tl.make_tensor_descriptor` (`tl.make_block_ptr` was removed in Triton 3.9; on `sm_89`, tensor descriptors lower to `tt.addptr` + `cp.async`).

| Variant (`D`, `BMxBN`, `w`, `req_s`) | `scf.for` | SMEM (B) | Regs | Spill S/L (B) | SMEM bufs (`K`, `V`) | Loop `wait_group` / `commit` | PTX `mma` / `ex2` / `shfl` (per warp/iter) |
|---|---:|---:|---:|---:|---|---:|---:|
| Non-causal `128`, `128x64`, `w=4`, `s=1` | 1 | 49152 | 255 | 1132 / 828 | none (`ld.global`) | none / 0 | 256 / 68 / 16 |
| Non-causal `128`, `128x64`, `w=4`, `s=2` | 1 | 65536 | 255 | 248 / 228 | `1x64x128`, `1x64x128` | `0` / 2 | 256 / 68 / 16 |
| Non-causal `128`, `128x64`, `w=4`, `s=3` | 1 | 98304 | 255 | 416 / 356 | `2x64x128`, `2x64x128` | `2` / 2 | 256 / 68 / 16 |
| Non-causal `128`, `128x64`, `w=4`, `s=4` | 1 | 131072 | 255 | 416 / 356 | `3x64x128`, `3x64x128` | `4` / 2 | 256 / 68 / 16 |
| Causal `128`, `128x64`, `w=4`, `s=3` | 2 | 98304 | 255 | 720 / 684 | 2 loops x (`2x64x128` x 2)* | `2` / 2 | 256 / 68 / 16 |
| Non-causal `64`, `64x64`, `w=4`, `s=3` | 1 | 40960 | 144 | 0 / 0 | `2x64x64`, `2x64x64` | `2` / 2 | 64 / 34 / 8 |
| Causal `128`, `64x64`, `w=4`, `s=3` | 2 | 90112 | 255 | 0 / 0 | 2 loops x (`2x64x128` x 2)* | `2` / 2 | 128 / 40 / 64 |

*\*In causal mode, Triton emits two sequential `scf.for` loops with separate `ttg.local_alloc` ops, but reuses the same SMEM allocation across them (`98304 B`, not doubled).*

### Key structural findings vs. GEMM

1. **Q is staged in SMEM and reloaded every KV iteration.**  
   In the prologue, Q is loaded synchronously (`tt.load`) and immediately written to SMEM via `ttg.local_alloc %q` (`128x128xf16 = 32768 B`, accounting for the extra 32 KB in `98304 = 32768(Q) + 32768(2xK) + 32768(2xV)`). Inside the KV loop, `ttg.local_load %q_smem` reloads Q from SMEM on every iteration before `dot(Q, K^T)`. Thus per-iteration SMEM read volume is `Q + K + V` (`65536 B`), not just `K + V`.
2. **K and V are co-pipelined into one stage on `sm_89`.**  
   Even though `desc_v.load()` appears after online softmax in Python, `TritonGPUPipeline` hoists both K and V loads into the same pipeline stage:
   - Prologue issues `async_copy(K) + commit`, `async_copy(V) + commit` per prefilled buffer.
   - Loop body opens with a **single** `ttg.async_wait %k_tok, %v_tok {num = N}` (`cp.async.wait_group N`) waiting on both K and V together, then executes `dot(Q, K^T)` -> online softmax -> `dot(P, V)`, and finishes by issuing the next K and V async copies (2 commit groups per iteration).
   - Therefore, our existing PTX ISA rule `prefetch_distance = wait_group_N // commits_per_stage` yields `0 / 0 / 1 / 2` for `req_stages = 1 / 2 / 3 / 4` with exact PTX/TTGIR agreement — identical to GEMM.
   - **Caveat for future variants:** kernels with split K/V waits (one `wait_group` before `dot(Q, K^T)` and a second before `dot(P, V)`) will have two distinct `wait_group` counts inside one loop body; the extractor must detect multiple distinct waits rather than blindly taking `min()`.
3. **Two distinct MMA ops + heavy SFU/shuffle softmax per iteration.**  
   Each KV iteration executes:
   - `mma_qk`: `tile_mnk = [BM, BN, D]`, `inst_mnk = [16, 8, 16]`, `count = (BM/16)*(BN/8)*(D/16)`.
   - Online softmax: row-max `tt.reduce`, scale/sub, `math.exp2` (`64` `ex2.approx`/warp for `P`), row-sum `tt.reduce`, `math.exp2` (`4` `ex2.approx`/warp for `alpha`), `acc * alpha` rescaling, and `f32 -> f16` cast. At `BM=128, BN=64, w=4`, SFU (`68 ex2.approx`/warp) and warp shuffles (`16 shfl.sync`/warp; `64` at `64x64`) are first-order latency contributors alongside MMA.
   - `mma_pv`: `tile_mnk = [BM, D, BN]`, `inst_mnk = [16, 8, 16]`, `count = (BM/16)*(D/8)*(BN/16)`.
4. **Spill pressure is much sharper than in GEMM.**  
   `BM=128, BN=64, D=128, w=4` hits 255 regs with `248–720 B` of `ptxas_only_spill` across all `num_stages`, whereas `BM=64, BN=64` has `0 B` spill. Accurate spill modeling directly governs config ranking.
5. **Causal attention compiles to two sequential `scf.for` loops with per-CTA trip counts.**  
   - Loop 1 (unmasked off-diagonal): `start_n in [0, start_m * BM)`, trip count `start_m * (BM // BN)` (varies linearly from `0` to `N_CTX/BN - BM/BN` across CTAs).
   - Inter-loop drain (`ttg.async_wait {num = 0}`) and refill.
   - Loop 2 (masked diagonal): `start_n in [start_m * BM, (start_m + 1) * BM)`, constant trip count `BM // BN` (e.g., `2`), adding causal comparison + `arith.select`.

## 3. Mapping Attention to `KernelIR` v0.1

Most of fused attention maps cleanly onto existing `KernelIR` v0.1 primitives:

| Attention Component | `KernelIR` v0.1 Mapping | Notes |
|---|---|---|
| Grid `(cdiv(N_CTX, BM), Z*H, 1)` | `launch.grid = [grid_m, Z*H, 1]` | 2D grid already supported |
| Tensors `Q, K, V, Out` | `tensors`: 4D `[Z, H, N_CTX, D]` | Arbitrary rank already supported in schema |
| `Q` load + SMEM staging | `body.prologue.ops`: `gmem_load(Q)` + `smem_store(Q)` | Uses the non-pipelined prologue slot reserved by Agents 3 & 4 |
| KV loop `K, V` loads | `body.pipeline_loop.ops`: two `gmem_load` (`acc_K`, `acc_V`) | `stage_offset = prefetch_distance` |
| `Q, K, V` SMEM reads | `body.pipeline_loop.ops`: `smem_load` (`Q + K + V` bytes/iter) | Must include Q reload bytes from `ttg.local_load` |
| `Q*K^T` and `P*V` | Two `mma` ops (`mma_qk`, `mma_pv`) in `pipeline_loop.ops` | Schema `ops` array already allows multiple `mma` ops |
| Online softmax `exp2` | `scalar_op` (`kind: "sfu"`, `dtype: "f32"`, `count: ex2_per_warp`) | Schema already defines `sfu` with `insts_per_warp_per_iter` |
| Softmax ALU & reductions | `scalar_op` (`kind: "alu"`, `dtype: "f32"`, `count: ...`) | See Section 4 for warp-shuffle serialization gap |
| Final `acc / l_i` + `Out` store | `body.epilogue.ops`: `alu`/`sfu` + `gmem_store(Out)` | Fits existing epilogue block |
| Cross-Q-block K/V L2 reuse | `access_map`: `strided_2d_per_outer_tile` for `acc_Q/K/V/O` | CTAs with the same `off_hz` (or same KV head in GQA) emit identical `base_byte_offset` for `K`/`V`, so Agent 3's L2 model captures inter-CTA KV reuse without schema changes (`tile_coord_0 = start_m`, `tile_coord_1 = off_hz`) |

## 4. Schema Gaps Requiring Agent 3 Extensions

### Gap 1 (Required for causal attention): `pipeline_loop.trip_count_mode` lacks `"per_cta"`
- **Current v0.1 schema**: `pipeline_loop.trip_count_mode` only allows `"uniform"` or `"per_outer_iter"`. For non-persistent attention (`outer_loop.trip_counts = 1`), `"per_outer_iter"` cannot express distinct KV loop lengths across `grid_m * Z * H` CTAs.
- **Proposed extension**: Add `"per_cta"` to `pipeline_loop.trip_count_mode` (matching `outer_loop.trip_count_mode`), where `trip_counts` is an integer array of length `grid_size` (with `minimum: 0` already allowed by schema). For future persistent causal attention, `"per_outer_tile"` (flat array indexed identically to `access_map.entries`) generalizes both.
- **How to represent the two sequential causal loops (`unmasked` + `masked`)**:
  - *Option A (Recommended for Phase 4 start)*: Collapse both sequential loops into one `pipeline_loop` with `trip_count_mode = "per_cta"` and `trip_counts[cta] = (start_m + 1) * (BM // BN)`, attributing the small constant mask ALU overhead and one inter-loop drain barrier to `epilogue.ops`. Keeps `body` single-loop and avoids changing the simulator's loop driver; for `N_CTX = 2048, BN = 64` (average 17 iters), the 2-iter diagonal overhead is small and constant across tiles.
  - *Option B (Exact multi-loop)*: Allow `body.pipeline_loops: list[PipelineLoop]` (or a second optional `tail_loop`). Needed only if short sequences (`N_CTX <= 512`) or split-K/decode show that inter-loop drain/refill latency shifts config rankings.

### Gap 2 (Model/schema contract check): SFU and warp-shuffle accounting
- `scalar_op` already supports `kind: "sfu"`, so the frontend can immediately emit exact per-warp `ex2.approx` counts from PTX/TTGIR. We must confirm with Agent 4 that `vgpu-core` prices `kind: "sfu"` against `measured.compute.sfu_tflops` (not generic ALU).
- `tt.reduce` lowers to serialized `shfl.sync` chains (`8–64` instructions/warp/iter) plus FP32 max/add. Decide with Agents 3 & 4 whether folding `shfl.sync` into `kind: "alu"` suffices for Phase 4 ranking or if a dedicated `warp_shuffle` / dependency-latency term is needed.

### Gap 3 (Deferred until split-wait kernels appear): Per-op async wait depth
- On Triton 3.9 `sm_89`, K and V share a single `wait_group`, so one `pipeline_loop.prefetch_distance` is exact. If future kernels or Hopper/Blackwell warp-specialized attention emit distinct waits for K and V, `prefetch_distance` should move (or be overridable) per `gmem_load` op (`stage_offset` already exists on `gmem_op`, but its semantic tie to `wait_group` would need to be formalized).

## 5. Required Frontend Changes (When Implementation Starts)

No code changes are made in this step; when approved, the extractor will need:
1. **Multi-`tt.dot` extraction** in `ttgir.py`: return a list of `MmaInfo` (one per `tt.dot` in the loop body) rather than a single MMA.
2. **Per-loop-scope parsing** in `ttgir.py`: parse each top-level `scf.for` independently so causal's two sequential loops do not double-count `local_alloc` buffers or mix `wait_group` counts.
3. **Scalar evaluator additions** in `scalar_eval.py`: add `tt.reshape` pass-through (emitted by Triton 3.9's lowered tensor descriptors) and evaluate loop upper bounds (`hi`) per CTA to populate `trip_counts` for causal masks.
4. **Prologue Q extraction & SFU/ALU counting**: emit `gmem_load(Q)` + `smem_store(Q)` in `prologue.ops`, include `ttg.local_load` of Q in `smem_load`, and count `ex2.approx` / `tt.reduce` / FP32 ops in the loop body.
5. **Split-wait guard**: if a single loop body contains multiple distinct `cp.async.wait_group` values, flag `split_async_wait_unsupported` in `unsupported_features` rather than guessing.
