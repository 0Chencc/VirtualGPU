# Ascend 910B3 Formal Sealed Holdout Evaluation

> **Overall decision: Fail (Attention warm/cold Pass; GEMM warm/cold Pass on regret and NeuSight ranking, Fail on Roofline rank margin).** Because a device passes only when both kernel families pass all criteria in both cache regimes, `ascend910b3` remains gated with `unvalidated_feature` and `unvalidated_target_ascend`.

## Setup and Provenance

- **Protocol:** Fixed before capture in [validation/ascend-910b3-preregistration.md](../../validation/ascend-910b3-preregistration.md).
- **Frozen core and profile:** `main` plus HIVM frontend parser fixes (`hivm.py`: Ascend fractal NZ `(Cols1, Rows1, Rows0, Cols0)` tile-shape decoding and skipping `AIC`-only `hivm.hir.vbrc` `cbuf` helpers), with `profiles/ascend910b3.json` (`0.3.1-ascend`, zero GEMM or Attention timing labels). Predictions were generated CPU-only before any holdout kernel ran on NPU hardware (`draws = 32`, `seed = 17`, `pair_margin = 0.05`).
- **Frozen baselines (zero Ascend labels):**
  - `roofline_wave`: analytical roofline + CTA wave quantization (`sm_count = 20`, Cube FP16 peak, HBM read bandwidth, launch overhead).
  - `neusight_style`: frozen ridge regression fits in `validation/v02-frozen-fit.json` (trained exclusively on NVIDIA RTX 4070 and GB10 labels), evaluated with Ascend 910B3 hardware profile features.
- **Hardware and stack:** Huawei Ascend 910B3 (`npu:0`, 20 AI Cores = 20 AIC + 40 AIV, 64 GiB HBM, 192 MiB L2), Driver `26.1.1`, CANN `9.2.1`, Triton `3.5.0`, Triton-Ascend `3.2.2`, `torch_npu 2.10.0.post4`, Python `3.12.13`.
- **Capture and telemetry:** 16 fresh disjoint shapes (8 GEMM x 16 configs = 128 variants, 8 Attention x 10 configs = 80 variants; 208/208 compiled and passed PyTorch FP32 reference accuracy checks). 3 sessions x 10 randomized complete blocks per `(case, config, cache)` = 48 sealed session reports. Warm launches use a zero-L2 register spacer (`stream_spacer`) so `torch.npu.Event` measures on-device execution without host dispatch bubbles; cold launches use a clean read-only 384 MiB `read_flush`. Across all 48 sessions, the 64 MiB warm-vs-cold L2 probe ratio stayed at `1.514-1.649` (median `1.60`), `Aicore curFreq` stayed locked at `1800 MHz`, die temperature `41-43 C`, and real-time power `105.6-120.0 W`.
- **Receipts and digests:**
  - Freeze receipt: [validation/ascend-910b3-freeze.json](../../validation/ascend-910b3-freeze.json) (`recorded_utc = 2026-11-27T00:19:45.943760+00:00`, `prediction_archive_sha256 = 1cedc8578baebf656d20dae4cc2be8b02caf06d0a94e7a8ed9a36c521234d5e6`, `profile_sha256 = 8846f4baf8e10d708def63e238899bf8ddd3cdb7c7df430fac5a7577ee25f91c`, `fit_sha256 = 9bbd965c362aed8f55dbb01224e76aef5073074ef726753a4ad9b045e860bcd4`). Note: the Linux `_core` binary on the remote host was built from a clean tarball without `.git` metadata, so `core_build` in the freeze JSON reports `commit = "unknown"`.
  - Seal receipt: [validation/ascend-910b3-seal.json](../../validation/ascend-910b3-seal.json) (`48` verified session reports, all `cold_probe_ratio > 1.5`).
  - Scored output: [validation/ascend-910b3-holdout-scores.json](../../validation/ascend-910b3-holdout-scores.json) (`scored_utc = 2026-11-27T00:37:46.655449+00:00`, `freeze_sha256` and `seal_sha256` verified).

## Cohort Summary

Pass criteria per [docs/design.md section 8](../design.md#8-validation): median Kendall tau-b at least `+0.05` above both baselines, and mean top-5 regret at most `1.0%` and at most `0.5` percentage points worse than either baseline.

| Cohort | Decision | Method | Median tau-b | Mean top-5 regret | Ordered agreement (agree / calls) | Median abs rel error |
|---|---|---|---:|---:|---:|---:|
| `ascend910b3-attention-warm` | **Pass** | `core` | **0.756** | **0.00%** | **82.8%** (280/338) | 23.6% |
| `ascend910b3-attention-warm` | | `roofline_wave` | 0.545 | 0.00% | 79.5% (210/264) | - |
| `ascend910b3-attention-warm` | | `neusight_style` | 0.578 | 0.00% | 79.1% (269/340) | - |
| `ascend910b3-attention-cold` | **Pass** | `core` | **0.756** | **0.00%** | **81.9%** (280/342) | 23.9% |
| `ascend910b3-attention-cold` | | `roofline_wave` | 0.571 | 0.00% | 79.5% (210/264) | - |
| `ascend910b3-attention-cold` | | `neusight_style` | 0.622 | 0.00% | 79.1% (269/340) | - |
| `ascend910b3-gemm-warm` | **Fail** | `core` | **0.773** | **0.00%** | **89.8%** (765/852) | 34.6% |
| `ascend910b3-gemm-warm` | | `roofline_wave` | 0.803 | 0.00% | 93.8% (726/774) | - |
| `ascend910b3-gemm-warm` | | `neusight_style` | 0.642 | 4.18% | 75.1% (674/897) | - |
| `ascend910b3-gemm-cold` | **Fail** | `core` | **0.720** | **0.09%** | **86.7%** (766/884) | 44.0% |
| `ascend910b3-gemm-cold` | | `roofline_wave` | 0.840 | 0.00% | 93.4% (723/774) | - |
| `ascend910b3-gemm-cold` | | `neusight_style` | 0.267 | 33.05% | 66.3% (554/836) | - |

- **Attention (`warm` and `cold`): Pass.** Across all 8 unseen shapes spanning `{full, causal} x {MHA, GQA} x {D=64, D=128}` (including odd sequence lengths `1665` and `1921`), VirtualGPU beats both baselines by `+0.134` to `+0.211` in median Kendall tau-b (well above the `+0.05` requirement), achieves `0.00%` mean top-5 regret, and reaches `23.6%-23.9%` median absolute latency error (`0.867-0.911` tau-b on all four non-causal shapes).
- **GEMM (`warm` and `cold`): Pass on regret and NeuSight ranking; Fail on Roofline rank margin.** VirtualGPU achieves `0.00%` warm and `0.09%` cold mean top-5 regret (`0.00%` on 15 of 16 case-cache combinations, `0.76%` on `decode-32` cold), agrees with hardware on `86.7%-89.8%` of ordered pairs, and beats `neusight_style` by `+0.131` (warm) and `+0.453` (cold) in tau-b. However, it trails `roofline_wave` in median Kendall tau-b (`0.773` vs `0.803` warm; `0.720` vs `0.840` cold), missing the `+0.05` rank gain requirement against `roofline_wave`.

## Per-Case Breakdown

### Attention Holdout (`validation/ascend-910b3-attention-holdout.json`)

| Case | Cache | Core tau-b | Roofline tau-b | NeuSight tau-b | Core top-5 regret | Core med abs err |
|---|---|---:|---:|---:|---:|---:|
| `b2-q16-kv4-s896-d64-full` | warm / cold | 0.911 / 0.867 | 0.389 / 0.441 | 0.467 / 0.511 | 0.00% / 0.00% | 25.8% / 27.5% |
| `b2-q16-kv4-s896-d64-causal` | warm / cold | 0.600 / 0.600 | 0.701 / 0.701 | 0.689 / 0.689 | 0.00% / 0.00% | 18.8% / 18.4% |
| `b2-q16-kv16-s1408-d64-full` | warm / cold | 0.911 / 0.911 | 0.441 / 0.441 | 0.511 / 0.511 | 0.00% / 0.00% | 26.2% / 26.1% |
| `b1-q32-kv8-s1665-d64-causal` | warm / cold | 0.644 / 0.644 | 0.649 / 0.649 | 0.644 / 0.644 | 0.00% / 0.00% | 14.5% / 14.2% |
| `b2-q16-kv4-s1152-d128-full` | warm / cold | 0.911 / 0.867 | 0.389 / 0.493 | 0.467 / 0.600 | 0.00% / 0.00% | 32.6% / 33.9% |
| `b2-q16-kv4-s1152-d128-causal` | warm / cold | 0.600 / 0.600 | 0.701 / 0.701 | 0.689 / 0.689 | 0.00% / 0.00% | 21.1% / 21.0% |
| `b1-q16-kv16-s1921-d128-full` | warm / cold | 0.867 / 0.867 | 0.441 / 0.441 | 0.511 / 0.511 | 0.00% / 0.00% | 32.9% / 33.8% |
| `b1-q32-kv8-s2304-d128-causal` | warm / cold | 0.600 / 0.600 | 0.701 / 0.701 | 0.689 / 0.689 | 0.00% / 0.00% | 21.4% / 21.6% |

### GEMM Holdout (`validation/ascend-910b3-gemm-holdout.json`)

| Case | Shape (`M x N x K`) | Cache | Core tau-b | Roofline tau-b | NeuSight tau-b | Core top-5 regret | Core med abs err |
|---|---|---|---:|---:|---:|---:|---:|
| `ascend-v2-decode-32` | `32 x 4608 x 2560` | warm / cold | 0.588 / 0.611 | 0.723 / 0.781 | -0.276 / -0.217 | 0.00% / 0.76% | 56.7% / 58.5% |
| `ascend-v2-decode-96` | `96 x 6144 x 2816` | warm / cold | 0.611 / 0.561 | 0.639 / 0.513 | 0.517 / 0.033 | 0.00% / 0.00% | 55.4% / 55.8% |
| `ascend-v2-wave2-below` | `128 x 4992 x 2816` | warm / cold | 0.773 / 0.711 | 0.803 / 0.840 | 0.650 / 0.217 | 0.00% / 0.00% | 34.5% / 43.5% |
| `ascend-v2-wave2-exact` | `128 x 5120 x 2816` | warm / cold | 0.773 / 0.728 | 0.803 / 0.840 | 0.633 / 0.267 | 0.00% / 0.00% | 34.7% / 44.5% |
| `ascend-v2-wave2-above` | `128 x 5248 x 2816` | warm / cold | 0.756 / 0.678 | 0.767 / 0.803 | 0.617 / 0.267 | 0.00% / 0.00% | 27.0% / 41.1% |
| `ascend-v2-prefill-med` | `768 x 4608 x 2816` | warm / cold | 0.824 / 0.783 | 0.876 / 0.876 | 0.650 / 0.617 | 0.00% / 0.00% | 26.4% / 32.7% |
| `ascend-v2-prefill-large` | `2304 x 4608 x 2816` | warm / cold | 0.824 / 0.833 | 0.876 / 0.876 | 0.650 / 0.650 | 0.00% / 0.00% | 24.5% / 25.2% |
| `ascend-v2-mask-odd` | `1153 x 3329 x 2689` | warm / cold | 0.800 / 0.800 | 0.908 / 0.908 | 0.683 / 0.633 | 0.00% / 0.00% | 66.1% / 69.5% |

## Post-Hoc Residual Analysis (No Refitting Applied)

1. **Why `core` trailed `roofline_wave` in GEMM tau-b despite `0.00%-0.09%` regret.** On Ascend 910B3, every GEMM config runs at `ctas_per_sm = 1` with no warp/register occupancy cliff, so tile geometry alone gives `roofline_wave` an unusually high `tau-b = 0.803-0.840` (vs `0.334-0.390` on RTX 4070 and `0.508` on GB10). In the core, `multibuffer = False` (`prefetch_distance = 0`) pays full un-overlapped DRAM round trips (`+3%` to `+10%` error on `prefill-large`) while `multibuffer = True` (`prefetch_distance = 1`) halves exposed round trips (`-17%` to `-41%` fast on `prefill-large`, `-50%` to `-68%` fast on short `decode-32` where per-iteration Cube/MTE2 overhead dominates). Because real Ascend 910B3 hardware shows only a `1%-8%` gap between `mb0` and `mb1` on `AIC_ONLY` GEMM (hardware MTE2/MTE1 decoupling hides part of single-buffer latency), the core's ~`1.5x` `mb0/mb1` predicted gap swaps several `mb0` configs below smaller-tile `mb1` configs (for example, on `prefill-large` cold, `m128n128k64-g8-mb0` measures `636.0 us` vs `m128n128k32-g8-mb1` at `976.1 us`, while `core` predicts `656.1 us` vs `575.9 us`).
2. **Initial diagnostic v1 findings fixed prior to this run.** An earlier diagnostic attempt on a disjoint shape set caught two parser bugs in `hivm.py` (swapped fractal NZ `(Cols1, Rows1)` dimensions when `M, N, K` are `tl.constexpr`, and counting `AIC`-only L1 `cbuf` `hivm.hir.vbrc` padding as AIV vector ALU work) plus a `~160 us` `torch.npu.Event` warm-path host-dispatch floor (`cold_probe_ratio = 0.515`). Fixing those three bugs and evaluating on this fresh disjoint shape set lifted GEMM from `0.474/0.575` tau-b and `5.89%/10.63%` regret to `0.773/0.720` tau-b and `0.00%/0.09%` regret, and reduced Attention median latency error from `48.3%-49.3%` to `23.6%-23.9%`.
