# VirtualGPU

A GPU simulator for testing and predicting the performance of LLM inference kernels — without needing the target GPU.

## Why

Tuning GEMM, attention and MoE kernels means trying hundreds of variants (tile shapes, pipeline depths, layouts) on real hardware. Real hardware is scarce, expensive, and often not the generation you are targeting. VirtualGPU aims to answer one question cheaply:

> **Which kernel variant is faster on GPU X?**

The goal is **relative speed**, not cycle-exact timing. Ranking variants correctly is what tuning needs; systematic errors that shift every variant equally don't matter. Real hardware stays the final judge — VirtualGPU narrows hundreds of candidates down to a handful.

## Principles

- **Built from public information and measurement.** Functional semantics follow public ISA specifications (NVIDIA PTX, AMD CDNA ISA manuals). Timing models use generic architectural structure (pipelines, queues, caches, bandwidth) with parameters fitted from our own microbenchmarks and official profiler counters. No disassembly of vendor drivers, firmware or toolchains.
- **Model what differentiates variants.** If an optimization changes a mechanism (occupancy, pipeline overlap, coalescing, bank conflicts), the model must contain that mechanism; everything else can stay coarse.
- **One model per architecture, one config per SKU.** H100, H800, H200 and H20 share one sm_90 model with different parameter files.

## Architecture

```
CUDA / HIP program
      │
 runtime shim (driver API replacement)
      │
 functional model   — PTX / AMDGPU ISA interpreter, bit-exact results
      │  instruction stream + synchronization events
 timing model       — pluggable fidelity: analytical / tile-level / cycle-level
   ├─ compute units: warp scheduling, issue, tensor cores
   ├─ async engines: TMA, async copy, barriers
   └─ memory: L1/shared, L2, HBM
      │
 output: cycles, profiler-style counters, per-warp timeline
```

Synchronization (barriers, async-copy completion) is timing-driven: modern kernels are asynchronous pipelines, so the timing model decides when the functional model may proceed.

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

1. **GEMM, tile-level, one architecture** — predict the ranking of GEMM variants, calibrated on a real card. Proves the approach.
2. **Attention and MoE kernels** on the same model.
3. **Functional simulation** — runtime shim + PTX interpreter; unmodified programs run and match real hardware bit for bit.
4. **Cycle-level SM and memory models** with async engines.
5. **More architectures**, AMD support, sampling and parallel simulation for speed.

## Validation

Accuracy is measured by what tuning needs:

- **Rank correlation** (Kendall τ) between simulated and measured times across variants.
- **Top-k hit rate** — does the simulator's top-k contain the truly fastest variant?
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

Early design. Nothing runs yet.

## License

[Apache-2.0](LICENSE)
