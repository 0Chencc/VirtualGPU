# Ascend 910B3 FP16 GEMM Sanity Check (Unvalidated)

> **Status: Informal sanity check, not a sealed holdout.** Ascend 910B3 has not gone through VirtualGPU's preregistered holdout protocol (`validation/`). Every Ascend prediction is automatically tagged with `unvalidated_feature` and `unvalidated_target_ascend`.

## Setup

- **Hardware:** Huawei Ascend 910B3 (`npu:0`, 20 AI Cores = 20 AIC + 40 AIV, 64 GiB HBM, 192 MiB L2, 1800 MHz Cube clock).
- **Software:** Driver `26.1.1`, CANN `9.2.1`, Triton `3.5.0`, Triton-Ascend `3.2.2`, `torch_npu 2.10.0.post4`.
- **Profile:** `profiles/ascend910b3.json` (`0.3.0-ascend`, M1 + M2 microbenchmarks, zero GEMM fitting).
- **Measurement:** Cold L2 (384 MiB zero-fill flush before each timed kernel), 3 warmup runs + 10 timed runs via `torch.npu.Event`, median latency reported.
- **Simulation:** `l2_state = "cold"`, 64 Latin hypercube draws (`seed = 17`), 10 Triton-Ascend FP16 GEMM configs (7 with `multibuffer = True`, 3 with `multibuffer = False`).

## Summary Metrics (M1 vs M2 Profile)

| Shape (M x N x K) | Profile | Spearman rho | Median rel error | Pairwise verdict precision | Top-3 regret |
|---|---|---:|---:|---:|---:|
| `2048 x 2048 x 2048` | `0.2.0-ascend` (M1 only) | 0.624 | -24.9% | 33/42 (78.6%) | 0.00% |
| `2048 x 2048 x 2048` | `0.3.0-ascend` (M1 + M2) | 0.624 | -24.0% | 33/41 (80.5%) | 0.00% |
| `4096 x 4096 x 4096` | `0.2.0-ascend` (M1 only) | 0.624 | -21.9% | 33/42 (78.6%) | 0.00% |
| `4096 x 4096 x 4096` | `0.3.0-ascend` (M1 + M2) | 0.709 | -21.0% | 32/40 (80.0%) | 0.00% |

In both shapes the predicted #1 config (`m128n256k64_g8_mb1`) matched the fastest measured config on hardware.

## Per-Config Results (`0.3.0-ascend`)

### `2048 x 2048 x 2048`

| Config | Hardware (us) | Predicted (us) | Rel error | Dominant bound |
|---|---:|---:|---:|---|
| `m128n256k64_g8_mb1` | 141.5 | 115.2 | -18.6% | `dram_latency` (0.47) |
| `m128n256k64_g8_mb0` | 148.7 | 305.8 | +105.6% | `dram_bandwidth` (0.96) |
| `m128n128k64_g8_mb1` | 190.5 | 136.2 | -28.5% | `dram_latency` (0.76) |
| `m128n128k64_g8_mb0` | 201.3 | 397.9 | +97.7% | `dram_bandwidth` (0.99) |
| `m128n128k32_g8_mb1` | 308.6 | 170.9 | -44.6% | `dram_latency` (0.78) |
| `m128n128k128_g8_mb1` | 157.5 | 121.1 | -23.1% | `dram_latency` (0.75) |
| `m64n128k64_g8_mb1` | 258.3 | 194.1 | -24.8% | `l2_latency` (0.73) |
| `m64n64k64_g8_mb1` | 427.5 | 301.8 | -29.4% | `dram_latency` (0.89) |
| `m64n64k32_g8_mb0` | 786.3 | 1291.9 | +64.3% | `dram_latency` (1.00) |
| `m128n256k32_g1_mb1` | 177.6 | 132.3 | -25.5% | `l2_latency` (0.80) |

### `4096 x 4096 x 4096`

| Config | Hardware (us) | Predicted (us) | Rel error | Dominant bound |
|---|---:|---:|---:|---|
| `m128n256k64_g8_mb1` | 982.9 | 801.0 | -18.5% | `mma` (0.62) |
| `m128n256k64_g8_mb0` | 1013.0 | 2302.6 | +127.3% | `dram_bandwidth` (1.00) |
| `m128n128k64_g8_mb1` | 1345.0 | 1028.7 | -23.5% | `dram_latency` (0.88) |
| `m128n128k64_g8_mb0` | 1425.2 | 3029.8 | +112.6% | `dram_bandwidth` (1.00) |
| `m128n128k32_g8_mb1` | 2190.9 | 1307.7 | -40.3% | `dram_latency` (0.88) |
| `m128n128k128_g8_mb1` | 1099.1 | 901.6 | -18.0% | `dram_latency` (0.87) |
| `m64n128k64_g8_mb1` | 1947.2 | 1451.3 | -25.5% | `l2_latency` (0.86) |
| `m64n64k64_g8_mb1` | 3220.8 | 2330.3 | -27.6% | `dram_latency` (0.94) |
| `m64n64k32_g8_mb0` | 6017.9 | 10344.3 | +71.9% | `dram_latency` (1.00) |
| `m128n256k32_g1_mb1` | 1228.2 | 925.8 | -24.6% | `l2_latency` (0.94) |

## Observed Residuals

1. **`multibuffer = True` optimistic bias (-18% to -45%):** Cube `MTE1` (`L1 -> L0A/L0B`) and `Fixpipe` (`L0C -> GM` NZ-to-ND) formatting/transfer pipelines are not yet modeled separately from UB copy bandwidth.
2. **`multibuffer = False` pessimistic bias (+64% to +127%):** On hardware, disabling `multibuffer` barely slows large tiles (1013 us vs 983 us on `4096^3`), whereas the model currently flags these configs as heavily `dram_bandwidth`-bound. This needs follow-up investigation in the Ascend frontend / pipeline model.
