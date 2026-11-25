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

Phase 1 is in progress on an RTX 4070. The full pipeline runs end to end: Triton kernel, ahead-of-time compilation, Kernel IR, GPU profile, performance model, and scoring against measured hardware. On the development set (cold L2), VirtualGPU ranks configs better than a learned baseline trained on the same GPU, without using any GEMM timing. The workload holdout is still sealed.

GPU-free autotuning works today:

```
virtualgpu autotune --kernel my_module:matmul --shape 2048 8192 4096 \
    --profile profiles/rtx4070.json --cache cold
```

The Phase 2 cross-GPU evaluation on DGX Spark is preregistered in [validation/phase2-preregistration.md](validation/phase2-preregistration.md).

## License

[Apache-2.0](LICENSE)
