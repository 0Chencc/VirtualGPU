# VirtualGPU

A GPU simulator for testing and predicting the performance of LLM inference kernels — without needing the target GPU.

## Why

Tuning GEMM, attention and MoE kernels means trying hundreds of variants (tile shapes, pipeline depths, layouts) on real hardware. Real hardware is scarce, expensive, and often not the generation you are targeting. VirtualGPU aims to answer one question cheaply:

> **Which kernel variant is faster on GPU X?**

The goal is **relative speed**, not cycle-exact timing. Ranking variants correctly is what tuning needs; systematic errors that shift every variant equally don't matter. Real hardware stays the final judge — VirtualGPU narrows hundreds of candidates down to a handful.

## Principles

- **Built from public information and measurement.** The model uses public specifications, generic architectural structure (pipelines, caches, bandwidth), and parameters measured with our own microbenchmarks. No reverse engineering of vendor drivers, firmware or toolchains.
- **Model what differentiates variants.** If an optimization changes a mechanism (occupancy, wave quantization, pipeline overlap, L2 reuse), the model must contain that mechanism; everything else can stay coarse.
- **One model per architecture, one profile per SKU.** H100, H800, H200 and H20 share one sm_90 model with different JSON profiles.

## How it works

```
Triton kernel + autotuning configs
      │
 compile ahead of time for the target (no GPU needed)
      │  tile structure from Triton IR + resource usage
 tile-level, event-driven performance model
      │  + GPU profile (JSON: public specs + measured parameters)
      │
 output: ranking, indistinguishable groups, bounding resource per variant
```

Triton comes first: tile shapes, layouts, `num_warps` and `num_stages` are explicit in its IR, the same frontend serves NVIDIA and AMD, and autotuning already lives there. The first application is GPU-free autotuning.

See [docs/design.md](docs/design.md) for the full design. The Kernel IR is independent of language and vendor: Triton is the first frontend, and a PTX frontend follows. The model core is written in Rust with Python bindings.

## Installation

Prebuilt Linux `abi3` wheels (`x86_64` and `aarch64`, Python 3.10–3.13) and source distributions are attached to [GitHub Releases](https://github.com/VirtualGPU/VirtualGPU/releases).

```bash
# Install a GitHub Release wheel with Triton 3.9 (not published to PyPI):
pip install "virtualgpu-0.2.0-cp310-abi3-manylinux_2_34_x86_64.whl[triton]"

# Or build from source (requires Rust toolchain and maturin):
pip install ".[triton]"
```

The base `virtualgpu` package includes the Rust performance model core (`virtualgpu._core`), embedded JSON schemas, and bundled GPU profiles (`rtx4070`, `gb10`). Add the `[triton]` extra when extracting or autotuning Triton kernels.

## Quickstart

### 1. Choose a GPU profile

Bundled profiles can be passed by short name (`rtx4070`, `gb10`) or by path to any valid profile JSON (see [profiles/README.md](profiles/README.md) to measure a new GPU):

```python
import virtualgpu

print(virtualgpu.list_profiles())          # ['gb10', 'rtx4070']
print(virtualgpu.profile_path("rtx4070"))  # path to bundled rtx4070.json
```

### 2. Run GPU-free autotuning (CLI)

Given a `@triton.autotune`-decorated GEMM kernel `my_module:matmul`, rank all configs on a CPU-only machine:

```bash
virtualgpu autotune \
    --kernel my_module:matmul \
    --shape 2048 8192 4096 \
    --profile rtx4070 \
    --cache cold \
    --top-k 5
```

Or use `@virtualgpu.autotune` in Python to prune a large search space down to `top_k=5` candidates before benchmarking on real hardware (set `VIRTUALGPU_PROFILE=rtx4070` or pass `profile="rtx4070"`):

```python
import virtualgpu

@virtualgpu.autotune(configs=CONFIGS, key=["M", "N", "K"], profile="rtx4070", top_k=5)
def matmul(...):
    ...
```

### 3. Reading the output

`virtualgpu autotune` prints the JSON report, or with `--out report.json` writes it there and prints a short summary:

- **`ranking`**: configs ordered by predicted runtime on the nominal profile (`nominal_latency_ns`), each with its predictions under the perturbed profile draws (`draws`). Lower is faster. Configs the model cannot handle come last, with `status: unsupported` and a reason.
- **`shortlist`**: the top-k candidate configs recommended for final hardware verification (`top_k=5` by default; `top_k=1` skips hardware verification and explicitly marks the result as unverified).
- **`indistinguishable_pairs`**: pairs the model cannot order: a pair is ordered only if every draw agrees on which is faster by at least 5%. Treat these as tied.
- **`primary_bottleneck`**: per config, the cause that bounds the largest share of its time and that share (one of `mma`, `issue`, `smem`, `local`, `sfu`, `shfl`, `lsu`, `sync`, `l2_latency`, `dram_latency`, `l2_bandwidth`, `dram_bandwidth`, `launch`), explaining *why* one tile beats another.

### Limitations

VirtualGPU predicts relative performance using public specifications and microbenchmark profiles rather than cycle-accurate simulation. Known modeling limits — including register-spill L1 transition and reload latency (planned for v0.3), dynamic scheduling effects, and unmodeled kernel idioms — are documented in [Known limits (`docs/design.md`, section 9)](docs/design.md#9-known-limits). Any kernel construct the frontend or core cannot safely model is reported explicitly as `unsupported` rather than guessed.

## Target architectures

| Vendor | Architecture | Examples | Priority |
|---|---|---|---|
| NVIDIA | Hopper sm_90 | H100, H800, H200, H20 | 1 |
| NVIDIA | Ampere sm_80 | A100, A800 | 1 |
| NVIDIA | Blackwell sm_100 / sm_103 | B200, GB200 / B300, GB300 | 2 |
| AMD | CDNA3 gfx942 | MI300X, MI325X | 2 |
| NVIDIA | Ada sm_89 | L40S, L20, RTX 4090 | 3 |
| NVIDIA | Blackwell sm_120 / sm_121 | RTX 5090, RTX PRO 6000 / GB10 | 3 |
| AMD | CDNA4 gfx950 | MI350X, MI355X | 3 |
| — | Rubin, CDNA5 | VR200, MI455X | when public details allow |

## Roadmap

See [docs/roadmap.html](docs/roadmap.html) for a plain-language overview.

0. **Design** (done) — [docs/design.md](docs/design.md) and the component designs in [docs/proposals/](docs/proposals/).
1. **Prove it on one GPU** — rank the autotuning configs of a Triton GEMM on an RTX 4070 and beat two simple baselines.
2. **Predict a GPU we did not tune on** — the same test on a DGX Spark, using only its profile.
3. **Open the doors** — GPU-free autotuning for Triton, a shared library of GPU profiles, and a simple way to contribute one.
4. **More kinds of kernels** — attention, mixture of experts, and whole-model bottleneck finding.
5. **More languages, more GPUs** — a PTX frontend (CUDA, CUTLASS, TileLang, ...) and AMD support.

## Validation

Accuracy is measured by what tuning needs:

- **Rank correlation** (Kendall τ) between simulated and measured times across variants.
- **Top-k hit rate** — does the simulator's top-k contain the truly fastest variant?
- **Indistinguishable accuracy** — variants within a few percent on hardware should be reported as indistinguishable, not ordered.
- Absolute error (MAPE) is tracked but secondary.

Every calibration and validation dataset records the GPU, driver, clocks and kernel source it came from.

## Related work

- [Accel-Sim](https://accel-sim.github.io/) — cycle-level, trace-driven NVIDIA simulator
- [GPU-Tile-Sim](https://arxiv.org/abs/2607.11262) — tile-centric simulation for LLM kernels (MICRO 2026)
- [NeuSight](https://arxiv.org/abs/2407.13853) — tile-based performance forecasting for unseen GPUs (ASPLOS 2025)
- [KernelSight-LM](https://arxiv.org/abs/2606.28565) — kernel-level LLM inference simulator
- [Sim-FA](https://arxiv.org/abs/2605.00555) — cycle-accurate FlashAttention-3 pipeline simulation
- [Vidur](https://github.com/microsoft/vidur), [LLMServingSim](https://github.com/casys-kaist/LLMServingSim) — serving-level simulators

## Status

**Phase 1 (RTX 4070) is complete, and it formally failed its preregistered criteria.** The model was frozen as `phase1-final`, predictions were archived, and only then was the sealed workload holdout opened. Full results: [validation/phase1-holdout-report.md](validation/phase1-holdout-report.md).

| Holdout (9 unseen shapes) | VirtualGPU | Learned baseline (NeuSight-style) | Roofline + wave |
|---|---:|---:|---:|
| Kendall tau-b, warm / cold | **0.787 / 0.750** | 0.387 / 0.476 | 0.512 / 0.514 |
| Mean top-5 regret, warm / cold | 3.11% / 3.02% | **1.19% / 1.24%** | 9.30% / 9.25% |

- When VirtualGPU says one config is faster than another, the GPU agrees about 95% of the time and disagrees about 2% of the time; the baselines are reversed about 20% of the time. See [docs/phase1-summary.md](docs/phase1-summary.md).
- Ranking passed by a wide margin: the learned baseline, trained on RTX 4070 GEMM timings, did not generalize to new shapes; VirtualGPU uses no GEMM timings at all.
- The shortlist criteria failed. Six of nine shapes have 0% top-5 regret; the mean is dominated by one shape with M, N and K all unaligned (24% warm, 22% cold). The cause is being investigated as a post-hoc analysis, which cannot change the Phase 1 decision. Known model limits are listed in [docs/design.md](docs/design.md) section 9.

GPU-free autotuning works today:

```
virtualgpu autotune --kernel my_module:matmul --shape 2048 8192 4096 \
    --profile profiles/rtx4070.json --cache cold
```

**Phase 2 (transfer to a DGX Spark GB10) is complete, and it also formally failed, by a narrow margin.** The same frozen `phase1-final` model received only a GB10 profile, measured in 24 seconds by the microbenchmark suite; it never saw a GB10 GEMM timing. Full results: [validation/phase2-gb10-report.md](validation/phase2-gb10-report.md).

| GB10 | VirtualGPU | Learned baseline (trained on RTX 4070) | Roofline + wave |
|---|---:|---:|---:|
| Primary cohort tau-b, warm / cold | **0.446 / 0.597** | 0.311 / 0.501 | 0.213 / 0.247 |
| Primary cohort top-5 regret, warm / cold | 1.10% / **0.16%** | 1.33% / 1.59% | 1.95% / 1.02% |
| Unseen shapes on unseen GPU, tau-b warm / cold | **0.762 / 0.749** | 0.432 / -0.194 | 0.523 / 0.519 |
| Unseen shapes on unseen GPU, top-5 regret warm / cold | **0.14% / 0.13%** | 36.8% / 45.2% | 4.1% / 3.9% |
| Ordered verdicts that agree with GB10 | **95% to 98%** | 41% to 53% | 43% to 70% |

- Cold passed every criterion. Warm missed the 1% absolute regret ceiling by 0.095 percentage points and passed everything else; the preregistration requires both.
- On a GPU the model was not built on, when VirtualGPU says one config is faster, GB10 agrees 95% to 98% of the time and disagrees less than 1% of the time. The learned baseline, trained on RTX 4070 timings, is reversed 15% to 41% of the time.

**v0.2 (frozen as `v0.2-final`) was evaluated on fresh sealed holdouts, GEMM and fused attention on both GPUs, and both device decisions formally failed.** The learned baselines were refit on all earlier published timings; VirtualGPU still used only the GPU profiles. Preregistration: [validation/v02-preregistration.md](validation/v02-preregistration.md). Full results: [validation/v02-holdout-report.md](validation/v02-holdout-report.md).

| Cohort | Decision | VirtualGPU tau-b, warm / cold | VirtualGPU top-5 regret, warm / cold | Ordered verdicts that agree with hardware: VirtualGPU / baselines |
|---|---|---:|---:|---:|
| RTX 4070 GEMM | **Pass** | 0.748 / 0.693 | 0.82% / 0.73% | 93% to 94% / 54% to 73% |
| RTX 4070 attention | Fail | 0.710 / 0.711 | 1.30% / 1.43% | 89% to 90% / 60% to 72% |
| GB10 GEMM | Fail | 0.683 / 0.689 | 2.12% / 2.00% | 94% / 59% to 64% |
| GB10 attention | Fail (warm) | 0.670 / 0.703 | 0.82% / 0.33% | 87% to 91% / 63% to 76% |

- VirtualGPU ranked better than both baselines by the required margin in 7 of 8 cohort and cache combinations; the exception is GB10 attention warm, 0.048 above the learned baseline against a required 0.05.
- The shortlist ceiling of 1% mean top-5 regret failed for RTX 4070 attention and GB10 GEMM. On GB10 GEMM cold the learned baseline's regret was also lower by more than the 0.5-point margin.
- A device passes only if both kernel families pass in both cache regimes, so both devices fail.

## License

[Apache-2.0](LICENSE)
