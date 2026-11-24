# VirtualGPU Design

Status: draft for discussion. No code exists yet.

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

## 5. Input: Triton first

| Input | Pros | Cons |
|---|---|---|
| Triton IR (TTGIR) | Tiles, layouts, `num_warps`, `num_stages` are explicit; same frontend for NVIDIA and AMD; native autotuning workflow | Only covers Triton kernels |
| PTX | Public specification, covers any CUDA source | Tile structure must be recovered; misses `ptxas` decisions |
| Vendor machine code | Closest to what runs on hardware | NVIDIA machine code is undocumented |

VirtualGPU starts with Triton. The tile structure that the performance model needs is already present in the IR, and Triton is where autotuning already happens.

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

Items such as shared memory bank conflicts and instruction scheduling are out of scope for the first model. Variants that differ only in those will be reported as indistinguishable, which is the correct answer for a model that does not contain them.

### 6.2 Structure

- Per CTA: a steady-state loop of load and compute stages. Overlap is bounded by `num_stages`. Prologue and epilogue are modeled separately.
- Per SM: resident CTAs share issue, tensor core, and shared memory bandwidth.
- Per GPU: an event-driven scheduler dispatches CTAs to SMs in hardware order and produces waves and tails naturally. L2 and DRAM bandwidth are shared across SMs.

Simulating events per CTA rather than per instruction keeps a large GEMM in the millisecond-to-second range on a CPU.

### 6.3 Output

- Predicted time per variant.
- Ranking, with variants grouped as indistinguishable when their difference is below the model's resolution.
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
- **Indistinguishable accuracy**: pairs within 5% of each other on hardware should be reported as indistinguishable, not ordered.
- Absolute error (MAPE) is recorded but is not a pass criterion.

Proposed pass criteria, to be fixed before the experiment is run: median τ ≥ 0.8 and the true best config in the top 5 for at least 90% of shapes.

### 8.2 Cross-GPU experiment

The real value is predicting GPUs the user does not own. The second experiment repeats 8.1 on a GPU whose measurements were not used in any model decisions, using only its profile.

### 8.3 Dataset

The measured data is a deliverable in its own right: kernel source, config, shape, GPU, driver, clocks, and time. Few public datasets of this kind exist.

## 9. Known limits

- **Data-dependent kernels.** MoE routing, sparse attention, and paged KV cache depend on runtime data. They need a workload description (expert distribution, sparsity pattern, page table) as an extra input.
- **Non-Triton kernels.** cuBLAS, CUTLASS, and hand-written CUDA kernels are not covered by the Triton frontend. Use case 3 needs a plan for them.
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

1. Implementation language for the model and CLI.
2. How to obtain register usage for AMD targets, and whether it matters for the first experiment.
3. How much of the L2 reuse model can be analytical before it needs simulation.
4. How to define the model's resolution per prediction, so "indistinguishable" is principled rather than a fixed threshold.
5. Which GPUs to use for the first and cross-GPU experiments.
6. Profile format versioning, and how to review community-submitted profiles.
