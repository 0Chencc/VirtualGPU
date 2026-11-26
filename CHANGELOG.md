# Changelog

All notable changes to VirtualGPU are documented in this file.

- **v0.1** corresponds to the `phase1-final` freeze (`129a6f5`, Phase 1/2 frozen baseline).
- **v0.2** covers the post-Phase-2 model, frontend, measurement, and validation cycle up to `58a0f3f`.
- L1 carveout spill transition curves and spill reload latency modeling are **not** included in v0.2 (measured and stored in the profiles, not yet implemented in the core; planned for v0.3, see Known Limits).

---

## v0.2 (2026-11-25)

### Model Mechanisms

- Run each iteration's MMAs and the work depending on them in series within a CTA (`9c12720`).
- Floor each memory round trip at idle latency plus streaming its bytes through the SM port (`e51f8ef`), and share that SM port out of phase among co-resident CTAs (`7b1be46`).
- Bound per-SM global memory warp instruction issue rate using controlled-in-flight microbenchmark data (`3f2288a`).
- Add `sfu` and `shfl` execution pipes to the simulator core (`e7cb092`) driven by measured per-SM warp throughput curves (`f2915df`, `ac0e501`).
- Extend KernelIR v0.1 (`7ca1e50`) and the Rust IR/L2 model (`0d63893`, `e7cb092`) with `per_cta` per-tile trip counts, `shfl` ops, and prefix-overlap KV sharing for causal GQA attention.
- Charge register-spill bytes served by L2 or DRAM on the SM's local port as well as L2/DRAM (`ca411f4`).
- Update design documentation to reflect active core mechanisms (`e46b16e`) and the synchronous-load GEMM and spill charge notes (`3f65039`, `34529b2`), then align design sections 6 and 9 with the v0.2 core (`edb9ee6`).

### Frontend

- Implement the Triton forward fused-attention KernelIR extractor for non-causal and causal kernels (`136f683`), deriving `(BM, BN)` from TTGIR `tt.dot` and aligning toolchain keys with GEMM (`7f71e56`).
- Extract MMA, barrier, and shared-memory loop counts directly from PTX for both attention and GEMM (`bd2bc38`), and record both PTX and tile MMA counts in the override diagnostic (`a698ce2`).
- Derive `transfer_width_bytes` from PTX load/store widths and intern `MockTensor` dtypes (`1f8b5d5`), stabilizing mixed-dtype attention AOT specialization across garbage collection (`1cd36b7`, `e3d667f`).
- Carry forward GPU-free Triton autotuning (offline API, `virtualgpu autotune` CLI, online pruning hooks, parallel AOT compilation, and KernelIR caching; `d9e886a`, `2f779f1`) with attention support.

### Profiles & Measurements

#### Measurement corrections affecting prior profile numbers

- **Write-flush pollution in cold GEMM labels** (corrected prior to `phase1-final` in `c515438`): switched cold GEMM cache eviction from write-based flushes—which left dirty writebacks inside timed windows—to read-only L2 eviction.
- **Microbenchmark read-only flushes and streaming bandwidth accounting** (corrected prior to `phase1-final` in `35886e6`, profile `0.4.0`): switched microbenchmark L2 flushes to read-only, aligned streaming bandwidth accounting, and introduced bytes-in-flight curves.
- **Underfed L2 read bandwidth ceiling** (`b5d5aae`, profile `0.16.0`): re-measured `l2_bw_by_active_sms` with an ILP-8, 16-warp kernel on RTX 4070 and GB10 to reach the true hardware L2 bandwidth ceiling rather than saturating below it.
- **Uncoalesced spill reload benchmark** (`4502343`): fixed `spill_reload_latency` from scattered per-lane offsets to warp-uniform coalesced accesses across 128 B–4 KiB frames on RTX 4070 and GB10.
- **Double-stride bug in `serial_chain` spill reload benchmark** (`43152fd`, profile `0.20.0-posthoc`): fixed `idx = (v + stride_u32) & mask` to `idx = v & mask` in `microbench/src/m2_kernels.cuh` so `serial_chain` traverses 100% of stack-frame slots instead of half (at 128 KiB/SM, RTX 4070 rose from 25.7 to 94.4 ns/reload and GB10 from 29.4 to 59.0 ns/reload; GB10 512 KiB/SM now crosses the 24 MiB L2 boundary into DRAM), archiving the superseded half-working-set points under `spill_reload_latency_serial_halfws_diagnostic`.
- **Short `sync_stage_loop` window and runtime loop overhead** (`ffb01df`, profile `0.21.0-posthoc`): reconciled the 65–98 ns gap against `serial_stage_fill` using compile-time `MMA_ITERS` and ~1 ms adaptive kernel windows, re-measuring all 210 L2 points (superseded points archived under `sync_stage_loop_pre_reconcile_diagnostic`) and adding 144 large-stage (12–32 KiB) points on both GPUs.

#### New profiles and microbenchmarks

- Add measured DGX Spark GB10 (`sm_121`) profile and `--nvcc` build option (`4be40fe`), plus strided MLP and GB10 5x memory repeatability study (`51336b5`).
- Add SFU/SHFL issue throughput and SHFL latency benchmarks (`8a83947`) and the dependent MMA-softmax-MMA serial/overlap chain benchmark (`5024caf`).
- Add controlled-in-flight global memory load issue benchmarks (`056f2fb`, `9ff93be`, `2f139b6`) and S=1 synchronous vs. `cp.async` and multi-CTA `sync_stage_loop` benchmarks (`483ea3a`, `11fa0e7`).
- Measure L1 usable capacity vs. shared-memory carveout (`b1a6310`) and full per-tier L1 miss-rate transition curves (`b92286c`) on RTX 4070 and GB10 (stored in profiles for v0.3; not yet wired into the v0.2 core).

### Packaging

- Build cp310-abi3 Linux wheels (x86_64, aarch64) and an sdist with the GPU profiles bundled under `virtualgpu/profiles/`, add `list_profiles()` / `profile_path()`, let `--profile` take a bundled name, and add a README quickstart (`933d08e`).
- Release workflow attaches wheels and SHA256 sums to a GitHub Release, with no PyPI upload, and only runs on semantic version tags like `v0.2.0` (`933d08e`, `e4e770a`).

### Validation & Workflow

- Freeze `phase1-final` predictions and publish the Phase 1 RTX 4070 GEMM holdout report and pairwise verdict analysis (`92439e0`, `45db984`, `49cf6a7`, `5feb1b5`, `6ad60bf`).
- Execute and publish the Phase 2 sealed DGX Spark GB10 transfer validation with thermal/clock stability gating and acquisition retry handling (`d42128e`, `de3dba1`, `eae13bd`, `42689c8`, `5aaf5c3`, `1cb44d6`, `fd5c947`, `076b1da`, `c15b1a3`, `ad7fc77`, `d3ab4eb`, `fc89ad3`), plus post-hoc residual analyses (`4cd03e6`, `9491a19`).
- Define the Phase 4 RTX 4070 fused-attention workload suite, validation runner, development-set diagnosis, and sealed holdout (`aa54f13`, `d103ff2`, `fde371d`, `c1ac132`, `a5bab55`).
- Add an audited published-holdout scoring channel (`754613a`), track published holdout access summaries (`faa9071`), and refresh build identity on reflog/packed-ref changes (`0d3b4c4`).
- Preregister the v0.2 holdout evaluation (`29a6ac2`) and record the v0.2 development-set baseline before the freeze (`d518203`).
- Seal fresh v0.2 holdouts: GB10 GEMM and attention (`11b23a9`, `8a34ec4`, `1770165`) and RTX 4070 GEMM, including a recorded protocol retry (`58d07f6`, `120789a`, `4a5ebe8`, `72a8fb4`).
- Prepare the frozen v0.2 CPU prediction archive (`b38adfd`) and harden capture and prediction against stale caches, installed wheels, and dirty profiles (`139789c`); skip those tests where the CUDA modules are absent (`58a0f3f`).

### Known Limits

- **Deferred to v0.3:** carveout-aware L1-to-L2 spill transition curves and spill reload latency modeling are not implemented in the v0.2 core. The measured curves are stored in the profiles but the core does not read them yet.
- **Spill dynamic counts and causal dual-loop attribution:** static `ptxas` spill counts sum both causal `_attention_tiles` loops (main + 1-step diagonal) and do not report dynamic per-iteration reload counts; resolving this requires hardware performance counter (`ncu`) profiling.
- **Synchronous-load GEMM residual:** `num_stages = 1` register-staged GEMM configs remain predicted slow (+9.1% warm / +2.2% cold median error on the development set; see `docs/design.md` section 9.2).
- **Holdout validity for v0.2:** the RTX 4070 GEMM workload holdout and GB10 transfer datasets have been unsealed and **cannot** be reused to validate v0.2; v0.2 is instead evaluated on fresh sealed holdouts (RTX 4070 and GB10, GEMM and attention; see `validation/v02-preregistration.md`), whose results are pending.
- **Unmodeled scope:** data-dependent kernels (MoE routing, sparse/paged attention), non-Triton PTX kernels, Hopper `wgmma` / data-center Blackwell `tcgen05`, and fine-grained bank-conflict or thermal-throttling effects remain out of scope (see `docs/design.md` section 9).

---

## v0.1 — `phase1-final` (2026-11-25)

Frozen Phase 1/2 baseline (`129a6f5`, tag `phase1-final`) used to score the sealed RTX 4070 workload holdout and the DGX Spark GB10 zero-shot transfer evaluation.

### Model Mechanisms

- Tile-level, event-driven Rust performance core with closed queueing network steady-state solver (`cda8507`), empirical L2 rescan and MSHR miss-merging reuse model, occupancy/carveout accounting, and batched Latin-hypercube uncertainty draws (`42e2b31`).

### Frontend

- Triton GEMM AOT KernelIR extractor (Triton 3.9.0) with `scf.for` iteration-0 scalar evaluation (`18af028`), plus GPU-free offline autotuning API, CLI, parallel AOT compilation, and IR caching (`d9e886a`, `2f779f1`).

### Profiles & Measurements

- CUDA C++ microbenchmark suite (`vgpu_bench`) and RTX 4070 (`sm_89`) profile `0.11.0-m3`, including read-only L2 flushes and bytes-in-flight MLP curves (`35886e6`), closed-loop DRAM/L2/MMA-contention benchmarks (`e99a5a2`, `54597f0`, `67d3d0b`, `a4fc6e3`, `562fa39`), and profile library docs/CI validation (`32e38ac`).

### Validation & Workflow

- Sealed workload holdout capture, digest locking, and scoring pipeline (`2d74963`, `bc8f0a2`), with Phase 1 known limits finalized prior to unsealing (`c9d7bb2`, `71e7567`, `129a6f5`).

### Known Limits

- Serial large-tile GEMM residuals and `num_stages = 1` synchronous-load GEMM residuals documented in `docs/design.md` section 9.1; no attention frontend or `sfu`/`shfl` pipeline support.
