# Community GPU Profile Audit (Pre-v0.2.0 Wheel)

Independent read-only audit of the five community-contributed GPU profiles
bundled in `profiles/` prior to packaging the `v0.2.0` wheel:

- `profiles/rtx3070.json` (`5d03053`, Luke Gustafson)
- `profiles/l40s.json` (`1f499a7`, Luke Gustafson)
- `profiles/l4.json` (`df6bae8`, bbznop)
- `profiles/a100-80gb.json` (`a3d603c`, bbznop)
- `profiles/rtxpro6000.json` (`36f321a`, bbznop)

No profile files were modified during this audit.

## 1. Schema and Provenance Summary

- **Schema validation (`python3 scripts/check_profiles.py`)**: All five profiles
  pass (`OK`), with zero NaN/Inf/negative leaf values.
- **Provenance fields**: All five record `profile_revision: "0.21.0-posthoc"`,
  `suite_version: "0.21.0"`, `driver_version`, `toolkit_version`, `host_os`,
  `is_wsl`, `power_limit_w`, `timestamp_utc`, and `clock_policy` with
  `throttle_events: 0`.
- **Shared provenance notes (matching bundled `rtx4070.json` and `gb10.json`)**:
  - `calibration_manifest_digest` is `null`, and the schema does not record a
    microbenchmark git commit SHA (only `suite_version: "0.21.0"`).
  - `clock_policy.locked` is `false` on all five profiles (`l40s` and
    `a100-80gb` still held flat SM clocks with 0 MHz spread; `rtx3070`, `l4`,
    and `rtxpro6000` had unlocked SM clock swings).

## 2. Spec and Peak Comparison Table

| Profile | Arch | SMs | L2 (MiB) | DRAM Peak (GB/s) | Meas DRAM R / W (GB/s) | SM Clock Range (MHz) | Meas FP16 MMA (TFLOPS) | Peak FP16 MMA at Max Obs Clock | Verdict |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |
| `rtx3070.json` | `sm_86` | 46 | 4 | 448.0 (GDDR6 256b) | 427.5 / 397.2 | 1770-2025 | 47.31 | 47.34 (@2010 MHz) | Issues (High) |
| `l40s.json` | `sm_89` | 142 | 96 | 864.0 (GDDR6 384b) | 748.1 / 714.3 | 2100-2100 | 301.05 | 305.36 (@2100 MHz) | Clean |
| `l4.json` | `sm_89` | 58 | 48 | 300.0 (GDDR6 192b) | 266.4 / 270.8 | 1845-2040 | 119.14 | 121.15 (@2040 MHz) | Clean |
| `a100-80gb.json` | `sm_80` | 108 | 40 | 2039.0 (HBM2e 5120b) | 1800.0 / 1747.6 | 1410-1410 | 294.92 | 311.87 (@1410 MHz) | Issues (High) |
| `rtxpro6000.json` | `sm_120` | 188 | 128 | 1792.0 (GDDR7 512b) | 1510.4 / 1962.7 | 1710-2332 | 445.05 | 448.94 (@2332 MHz) | Issues (High) |

Notes on FP16 MMA (FP32 accumulate) rates per SM per cycle:

- `sm_86` (RTX 3070): 512 FLOPs/SM/cycle (measured 1.0284 TFLOPS/SM at 2,010 MHz = 99.9%).
- `sm_89` datacenter (L40S, L4): 1,024 FLOPs/SM/cycle (full-rate vs 512 on consumer GeForce RTX 4070).
- `sm_80` (A100): 1,024 FLOPs/SM/cycle (312 TFLOPS dense at 1,410 MHz; measured 294.92 TFLOPS = 94.6%).
- `sm_120` (RTX PRO 6000 Blackwell): 1,024 FLOPs/SM/cycle (measured 2.3673 TFLOPS/SM at 2,332 MHz = 99.1%).

## 3. Concrete Findings by Profile

### `profiles/rtxpro6000.json`

1. **[High] `measured.memory.dram_bw_write_gbps` exceeds theoretical DRAM peak by +9.5%.**
   - Recorded `dram_bw_write_gbps` is `1962.707 GB/s` (95% CI `[1944.057, 1973.790]`, method `cuda_stream_write_cold_l2`), which is 30.0% above `dram_bw_read_gbps` (`1510.372 GB/s`) and 9.5% above the theoretical 512-bit 28 Gbps GDDR7 ceiling (`1792.0 GB/s`).
   - On a 128 MiB L2 GPU, a streaming write buffer that is not sufficiently larger than L2 allows a large fraction of writes to complete in L2 rather than draining to DRAM.
2. **[Medium] Wide unlocked SM clock swing inflates uncertainty to 20.39% profile-wide.**
   - `clock_policy.observed_sm_clock_range_mhz` spans `[1710.0, 2332.0]` (-26.7%), producing a `0.2039` `rel_uncertainty` floor across most measured entries (compared to `0.10` on `l40s`, `a100-80gb`, and `gb10`). This will widen perturbation draws and increase indistinguishable pair rates.

### `profiles/a100-80gb.json`

1. **[High] `measured.memory.l2_bw_by_active_sms` caps 40.5% below `l2_bw_read_gbps`.**
   - `l2_bw_read_gbps` (`cuda_mlp_l2_warm_cp_async`) reaches `5007.939 GB/s`, consistent with A100's ~5.1 TB/s L2 read bandwidth.
   - `l2_bw_by_active_sms` (`cuda_l2_cg_read_warm_ilp8_16w`) saturates at `2980.163 GB/s` at 108 SMs (`(54, 2794.271), (108, 2980.163)`), a 40.5% shortfall against `l2_bw_read_gbps`. Any model path reading `l2_bw_by_active_sms` will understate full-chip A100 L2 read bandwidth by ~40%.
2. **[Medium] `specs.l1_smem_unified_bytes` and `specs.l1_size_bytes` are 164 KiB instead of 192 KiB.**
   - Both fields are `167936` (164 KiB, the maximum shared-memory carveout) rather than A100 `sm_80`'s physical 192 KiB (`196608` B) unified L1/SMEM per SM. This understates remaining L1 capacity by 28 KiB (implying 0 B L1 at the 164 KiB carveout).

### `profiles/rtx3070.json`

1. **[High] Non-monotonic full-chip DRAM bandwidth collapse and noisy WSL2 measurements.**
   - `dram_bw_by_active_sms` drops 34.8% from `426.945 GB/s` at 23 SMs to `278.284 GB/s` at 46 SMs (only 62.1% of the 448 GB/s GDDR6 peak).
   - `dram_bw_read_gbps` (`427.512 GB/s`, CI `[283.759, 431.899]`) and `dram_bw_write_gbps` (`397.188 GB/s`, CI `[289.292, 426.250]`) have ~35% CI widths.
   - `mma.sync:bf16:bf16:f32:16x8x16` has non-monotonic dips at 4 warps (`0.7442` vs FP16 `1.0237`) and 32 warps (`0.9106` vs `1.0281`), and `l2_rescan_hit_rate_by_working_set` drops to `0.54` at `0.25x L2` and rebounds to `0.191` at `4.0x L2`.
   - Consistent with unlocked WSL2 host/desktop contention (`is_wsl: true`, SM clock `[1770.0, 2025.0] MHz`, mem clock `[7001.0, 7201.0] MHz`).
2. **[Medium] `specs.l1_smem_unified_bytes` and `specs.l1_size_bytes` are 100 KiB instead of 128 KiB.**
   - Both fields are `102400` (100 KiB, the maximum shared-memory carveout) instead of Ampere `sm_86`'s physical 128 KiB (`131072` B) unified L1/SMEM per SM, understating remaining L1 capacity by 28 KiB.

### `profiles/l40s.json`

- **No blocking issues.** All specs match NVIDIA L40S (`sm_89`, 142 SMs, 96 MiB L2, 128 KiB unified L1/SMEM, 864 GB/s GDDR6). Clocks were flat (`[2100.0, 2100.0] MHz`, `[9001.0, 9001.0] MHz`), CIs are tight (`rel_uncertainty = 0.10`), and all curves are smooth and internally consistent with 2,100 MHz (note: 2,100 MHz is below the 2,520 MHz datasheet max boost, so chip MMA is 301.05 TFLOPS rather than 362 TFLOPS, which is normal for a cloud/datacenter L40S under default clocks).

### `profiles/l4.json`

- **No blocking issues.** All specs match NVIDIA L4 (`sm_89`, 58 SMs, 48 MiB L2, 128 KiB unified L1/SMEM, 300 GB/s GDDR6, 72 W TDP). Bandwidths, MMA throughput (119.14 TFLOPS vs 121.15 TFLOPS peak at 2,040 MHz), and scaling curves are smooth and well within physical limits. Observed SM clock range `[1845.0, 2040.0] MHz` (-9.6%) is mild for a 72 W card.
