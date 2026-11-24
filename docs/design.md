# VirtualGPU Design

Status: design phase complete. No code exists yet.

This document is the overview. Each component has its own detailed design in [`docs/proposals/`](proposals/):

| Component | Design |
|---|---|
| Triton frontend | [frontend.md](proposals/frontend.md) |
| Kernel IR and L2 reuse model | [kernel-ir-and-l2.md](proposals/kernel-ir-and-l2.md) |
| Performance model core | [model-core.md](proposals/model-core.md) |
| Microbenchmark suite and GPU profile format | [microbench-and-profile.md](proposals/microbench-and-profile.md) |
| Validation and evaluation | [validation.md](proposals/validation.md) |

A plain-language overview is in [roadmap.html](roadmap.html).

## Decisions

- **Scope:** an open-source tool for kernel developers. The first application is GPU-free autotuning for Triton.
- **Relative, not absolute:** the product is a ranking with confidence, not a predicted time.
- **One neutral Kernel IR:** the IR is independent of the source language and of the GPU vendor. Triton is the first frontend; a PTX frontend follows and covers CUDA, CUTLASS, TileLang and other languages that compile to PTX. AMD follows through its public ISA.
- **Implementation:** the model core (scheduler, L2 model, perturbation) is written in Rust and exposed to Python through PyO3. Frontends are written in Python. The microbenchmark suite is written in CUDA C++ to control the exact instructions it measures.
- **Test GPUs:** RTX 4070 (Ada, `sm_89`) for development; DGX Spark (GB10, `sm_121`) as a held-out GPU for the cross-GPU experiment.
- **Verdict rule:** A is faster than B only if A is at least 5% faster in every perturbation draw (|log(t_A/t_B)| >= log(1.05) with the same sign in all draws). Anything else is "indistinguishable". The 5% band is the same practical band used for hardware pairs; it is fixed, not tuned. Profile perturbation cannot cover structural model error, so a draw-only rule orders pairs that differ only through modeled structure.

## 1. Problem

Inference optimization is bottlenecked on GPUs. Kernel developers need hardware to tune kernels, the hardware they target is often not the hardware they own, and every hour spent benchmarking competes with training and serving for the same scarce compute.

VirtualGPU models a GPU architecture on the CPU so that kernel variants can be compared without the target GPU.

## 2. Core idea: relative, not absolute

VirtualGPU does not try to reproduce the absolute speed of a GPU. It answers one question:

> Given kernel variants A and B and GPU X, is A faster than B, by roughly how much, and why?

This is much easier than predicting absolute time:

- Systematic errors cancel. If the model underestimates memory bandwidth by 20%, every variant gets slower by the same factor, and the ranking is unchanged.
- No cycle-exact microarchitecture is needed, so no reverse engineering is needed. The model uses public specifications, generic architectural structure, and measured parameters.

The project works if this holds: **when VirtualGPU reports a difference between A and B, the same difference exists on the real GPU.**

## 3. Goals and non-goals

Goals:

- Rank kernel variants on a target GPU in seconds on a CPU.
- Report when two variants are indistinguishable instead of forcing an order.
- Explain each prediction: which resource bounds the kernel and why.
- Support GPUs the user does not own, through community-contributed GPU profiles.
- One frontend for NVIDIA and AMD.

Non-goals:

- Cycle-accurate timing.
- Functional simulation (computing kernel outputs). Inference kernel runtimes rarely depend on data values; see section 9 for exceptions.
- Replacing real hardware for final validation.
- Reverse engineering vendor drivers, firmware, or toolchains.

## 4. Use cases

Ordered by difficulty.

1. **Config ranking.** Rank the autotuning configs of one kernel (tile sizes, `num_warps`, `num_stages`). Systematic errors cancel almost entirely. This is the first target.
2. **Algorithm comparison.** Compare different implementations of the same operator, such as split-K versus stream-K GEMM. Variants may be bound by different resources, so per-resource scales must be roughly right.
3. **Bottleneck finding in a full model.** Find which operators in an end-to-end model are far from their achievable speed. This compares unrelated operators (GEMM versus normalization versus attention), so each operator class needs coarse absolute calibration.

The most direct application of use case 1 is **GPU-free autotuning**: a drop-in replacement for the measurement step of `@triton.autotune` that ranks configs against a chosen GPU profile.

## 5. Input: one neutral IR, Triton first

Kernels are written in many languages: CUDA C++, CUTLASS and CuTe, Triton, TileLang, HIP, and more. VirtualGPU does not support languages one by one. Every frontend lowers into one Kernel IR, which is independent of language and vendor, and the performance model only reads that IR.

```
Triton   CUDA C++   CUTLASS   TileLang   ...        HIP
   |          \         |        /                   |
   |               PTX (public)                AMD GPU ISA (public)
   |                     \                        /
   +----------------->  Kernel IR (neutral)  <---+
                               |
                       Performance model
```

| Input | Pros | Cons |
|---|---|---|
| Triton IR (TTGIR) | Tiles, layouts, `num_warps`, `num_stages` are explicit; same frontend for NVIDIA and AMD; native autotuning workflow | Only covers Triton kernels |
| PTX | Public specification, covers any CUDA source | Tile structure must be recovered; misses `ptxas` decisions |
| Vendor machine code | Closest to what runs on hardware | NVIDIA machine code is undocumented |

VirtualGPU starts with a Triton frontend that reads TTGIR after the software pipelining pass, because the tile structure the performance model needs is explicit there, and Triton is where autotuning already happens. A PTX frontend comes later and recovers the same structure from public PTX, which covers every language that compiles to it.

Closed-source libraries such as cuBLAS and cuDNN often ship machine code without PTX. VirtualGPU does not model their internals; where they matter, they are treated as measured black boxes.

Register usage and spills are decided by the backend assembler, not by TTGIR. Triton can compile ahead of time for a given target without a GPU, and the vendor assembler reports per-kernel resource usage. VirtualGPU uses that report as an input.

## 6. Performance model

The model is tile-level and event-driven. The input is the kernel launch grid and per-CTA (thread block) behavior. The output is predicted time with a breakdown by resource.

### 6.1 Mechanisms

Autotuning configs usually have similar arithmetic intensity. What separates them is the list below, and each item must be modeled:

| Mechanism | Driven by | Effect |
|---|---|---|
| Arithmetic intensity / roofline | tile shape, dtype | Compute-bound or memory-bound baseline |
| Wave quantization | grid size, SM count, occupancy | Tail waves leave SMs idle |
| Occupancy | registers, shared memory, warps per CTA | CTAs resident per SM, latency hiding |
| Pipeline overlap | `num_stages`, async copy / TMA | Whether loads hide behind compute |
| L2 reuse | tile traversal order (swizzle / grouping), L2 size | DRAM traffic |
| Tensor core shape fit | tile shape versus MMA instruction shape | Wasted MMA throughput |
| Launch and prologue/epilogue cost | kernel size | Matters for small shapes |
| Register spills | assembler spill report, spill location | Extra local memory traffic per iteration or once |
| Persistent kernels and warp specialization | outer tile loop, producer/consumer warps | One CTA handles many tiles; loads and compute run in separate warps |

Items such as shared memory bank conflicts and instruction scheduling are out of scope for the first model. Variants that differ only in those will be reported as indistinguishable, which is the correct answer for a model that does not contain them.

### 6.2 Structure

- Per CTA: phases (prologue, steady-state loop, drain, epilogue). In steady state, time per iteration is the maximum of compute time, transfer time, and load latency divided by the iterations in flight. Stage counts come from the IR after pipelining, never from the requested `num_stages`.
- Per SM: resident CTAs share issue, tensor core, and shared memory bandwidth.
- Per GPU: an event-driven scheduler dispatches CTAs to SMs using the dispatch policy measured on the real GPU, and produces waves and tails naturally. L2 and DRAM bandwidth are shared among active phases by max-min fair sharing, using bandwidth curves measured against the number of active SMs.
- L2: an online model, called by the scheduler at tile boundaries and once per phase segment, splits traffic into L2 hits and DRAM bytes from the actual set of resident tiles.

Events are per CTA phase, not per instruction or per loop iteration. The target is about 1 ms per simulation of a 4096-tile GEMM, so that a full autotuning space with perturbation sampling finishes in seconds.

### 6.2.1 Resolution through perturbation

When variants are bound by different resources, profile errors do not cancel. Every measured profile field therefore carries an uncertainty. The model runs each variant against draw 0 (the nominal profile) and 32 perturbed draws (Latin hypercube within each field's uncertainty, correlated fields drawn together, dispatch policy candidates included). All variants share the same draws, so comparisons are paired. A pair is ordered only if every draw agrees on the direction and on a difference of at least 5%; otherwise it is indistinguishable.

### 6.3 Output

- Predicted time per variant.
- Pairwise verdicts (faster, slower, indistinguishable). Verdicts are kept per pair; there is no hard grouping, because "indistinguishable" is not transitive.
- `unsupported` with a reason, when a kernel uses a feature the model does not handle.
- Per variant: bounding resource, utilization of each resource, wave count and tail efficiency, estimated DRAM traffic.

## 7. GPU profiles

A GPU is described by a JSON profile: one architecture model, one profile per SKU.

A profile has two kinds of fields:

- **Specification fields** from public vendor documentation: SM/CU count, clocks, register file and shared memory per SM, L2 size, memory bandwidth, peak throughput per dtype.
- **Measured fields** from the VirtualGPU microbenchmark suite: achievable DRAM and L2 bandwidth, shared memory bandwidth, achievable tensor core throughput per dtype and instruction shape, load latencies, launch overhead.

Every profile records its provenance: GPU model, driver version, clocks, and the version of the microbenchmark suite.

### 7.1 Community calibration

"No GPU needed" holds for users, not for the project. Each architecture must be measured on real hardware at least once. This is where the community scales the project: anyone with a GPU runs the microbenchmark suite and submits the resulting profile. Everyone else can then optimize for that GPU.

The microbenchmark suite uses only public APIs and our own kernels.

GPUs that are announced but not shipping can only get specification-derived profiles. Their predictions should be read as trends and rankings, not numbers.

## 8. Validation

### 8.1 First experiment

This experiment decides whether the approach works before anything else is built:

1. Take a Triton GEMM with a realistic autotuning space (tens of configs).
2. Run all configs over a set of LLM-relevant shapes on a real GPU and record times.
3. Run the same configs through VirtualGPU with a profile for that GPU.
4. Compare.

Metrics:

- **Kendall τ** between predicted and measured order, per shape.
- **Top-k hit rate**: whether the truly fastest config is in VirtualGPU's top k.
- **Top-k regret**: how much slower the best config in VirtualGPU's top k is than the true best.
- **Pair accuracy**: on hardware, a pair is equivalent if its 95% interval lies within ±5%, ordered if it lies wholly outside, unresolved otherwise. Model verdicts are scored against resolved pairs.
- Absolute error (MAPE) is recorded but is not a pass criterion.

VirtualGPU must beat two independent baselines built by the validation work, not by the model authors: roofline with wave quantization, and a NeuSight-style learned tile predictor. If a simple formula ranks as well, the event-driven model has no reason to exist.

Pass criteria, revised and fixed on 2026-11-25, after the development baseline scores and before any model score or holdout timing was seen:

- **Ranking:** median Kendall τ-b at least 0.05 higher than each baseline.
- **Shortlist:** mean top-5 regret at most 1% absolute, and no more than 0.5 percentage points worse than any baseline (non-inferiority).

The original shortlist criterion (1 percentage point lower regret than each baseline) was infeasible: the NeuSight-style baseline already reaches 0.46% (warm) and 1.09% (cold) mean top-5 regret on the development set, measured on the same GPU it was trained on.

The same criteria apply to the cross-GPU experiment (8.2), where the learned baseline is trained only on RTX 4070 labels and VirtualGPU only receives the target GPU's profile. That experiment is the main test: a learned baseline needs labels from some GPU, while VirtualGPU needs none from any GPU.

Hardware measurements use locked clocks where possible, explicit cold or warm L2 preparation, randomized repeated blocks across sessions, and 95% intervals. Calibration data (microbenchmarks), development data, and holdout data are kept strictly separate.

### 8.2 Cross-GPU experiment

The real value is predicting GPUs the user does not own. The second experiment repeats 8.1 on a GPU whose measurements were not used in any model decisions, using only its profile. The model and baselines are frozen on the RTX 4070 before any GB10 GEMM timing is opened.

### 8.3 Dataset

The measured data is a deliverable in its own right: kernel source, config, shape, GPU, driver, clocks, and time. Few public datasets of this kind exist.

## 9. Known limits

- **Data-dependent kernels.** MoE routing, sparse attention, and paged KV cache depend on runtime data. They need a workload description (expert distribution, sparsity pattern, page table) as an extra input.
- **Non-Triton kernels.** Not covered until the PTX frontend exists. Closed-source libraries without PTX are treated as measured black boxes.
- **Hardware coverage.** Neither test GPU has Hopper `wgmma` or data-center Blackwell `tcgen05`. Kernels using those paths carry an "unvalidated feature" flag until such hardware is available.
- **Fine-grained effects.** Bank conflicts, instruction scheduling, power and clock throttling are not modeled.
- **Model resolution.** Differences smaller than the model's error cannot be resolved; real hardware decides those.

## 10. Related work

- [GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim_distribution) / [Accel-Sim](https://accel-sim.github.io/): cycle-level NVIDIA simulation; detailed but hours per large kernel and behind on recent architectures.
- [Timeloop](https://github.com/NVlabs/timeloop): loop-nest mapping analysis for accelerator architects; its data reuse analysis is relevant to tile-level GEMM modeling.
- [Triton-CPU](https://github.com/triton-lang/triton-cpu): runs Triton kernels on CPUs for correctness; no GPU performance model.
- [GPU-Tile-Sim](https://arxiv.org/abs/2607.11262) (MICRO 2026): tile-graph simulation from TileLang IR, 1.22%–8.71% MAPE on A100 and H100. Closest in spirit; no public code found.
- [NeuSight](https://arxiv.org/abs/2407.13853) (ASPLOS 2025): tile-based performance forecasting for unseen GPUs, bounded by hardware peaks.
- [KernelSight-LM](https://arxiv.org/abs/2606.28565): kernel-level prediction; 12.1% per-kernel error on an unseen GPU generation versus 22.0% for roofline.

These works establish that tile-level modeling of LLM kernels is accurate. What does not exist is an open, usable tool aimed at kernel developers: Triton in, ranking and explanation out, in seconds, on a CPU, against any GPU profile.

## 11. Open questions

1. How to obtain register usage and spills for AMD targets.
2. Which Triton release line to pin first, to keep TTGIR adapters small.
3. How to feed workload descriptions (expert routing, page tables) into data-dependent kernels.
4. Profile format versioning, and how to review community-submitted profiles.
5. Whether a softer verdict rule than unanimity is worth calibrating.
6. Which clock controls are available on each test system (WSL2 for the RTX 4070, GB10's unified memory).

Each component design lists its own open questions in more detail.
