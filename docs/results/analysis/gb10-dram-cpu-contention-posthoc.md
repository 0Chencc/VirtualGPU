# CPU memory traffic and GPU DRAM timing on GB10

**Post-hoc diagnostic.** GB10's CPU and GPU share one LPDDR5x memory. The H5 captures on GB10 ([h5-followups-gb10-posthoc.md](h5-followups-gb10-posthoc.md)) had tight hot-L2 lists above 1 ms but long narrow DRAM points whose samples jumped between levels at one point (2.17 to 3.77 ms). This study asks whether CPU memory traffic causes that, and what a GB10 DRAM capture needs before it starts. It changes no model or profile. Summary: [gb10-dram-cpu-contention-posthoc.json](gb10-dram-cpu-contention-posthoc.json), written by `microbench/dram_dispersion.py summarize`.

## Method

- Points: `vgpu_bench --mode dram-dispersion` (`microbench/src/narrow_validate.cuh`), the narrow-load kernel from DRAM with `.cg` loads and aligned rows: 2 and 16 B, at 1 CTA per SM with 24 loads in flight and 3 CTAs per SM with 12, 512 MiB and 2 GiB per sample; 8 points, 30 samples each, an L2 flush before each sample. Launches take 2.3 to 13 ms.
- Conditions, one process each, cycled three times in one session: quiet, a 4-thread CPU load, quiet, a 16-thread CPU load, quiet. The load is `microbench/src/cpu_stream.c`: each thread copies between two 256 MiB buffers it has touched first; the GPU process starts 5 s after the load reports ready and the load stops 5 s after the GPU process ends. No other session's process was touched.
- Recorded once per second: `/proc/stat` (busy share of all 20 CPUs), PSI, page faults, load average, the top CPU processes, GPU telemetry. Run by `microbench/dram_dispersion.py` under the GPU lock.

Captured on a DGX Spark GB10, 2026-11-26 23:17 to 23:19 UTC, after the thermal gate (50.8 / 50.8 C, 2515 / 2515 MHz). The GPU ran at 2411 to 2554 MHz with no throttle reason in samples above 50% utilization; the only low clocks (208 MHz) and power-cap flags were in samples with 0% GPU utilization, between processes and while the load started or stopped. Raw capture `dram-dispersion-1.json` (SHA-256 `3983b7e6...`) in `gb10-dram-dispersion-captures.tar.gz` (`f761c1a4b9adda11391f92b2b007eb3ab628be350547b27cd67a81589aba5762`), on the Bifrost workstation in `~/scratch/vgpu8/` and on the Spark in `~/virtualgpu/s8-narrow/runs/`.

## Results

| Condition | Runs | Load bandwidth (CPU) | Busy share of CPUs | Point medians against quiet | Largest IQR over median |
|---|---:|---:|---:|---:|---:|
| Quiet | 9 | none | 5.3% to 5.8% | -1.5% to +1.9% | 4.8% |
| 4-thread load | 3 | 117 to 120 GB/s | 20.5% to 20.6% | +21.4% to +26.2% | 2.2% |
| 16-thread load | 3 | 126 GB/s | 80.9% to 81.5% | +65.9% to +82.5% | 4.2% |

No list in any run had an interquartile range above 10% of its median or a largest sample above 1.5 times its smallest, and no point's median moved more than 2.6% (quiet), 1.7% (4 threads) or 3.2% (16 threads) across the three cycles.

1. **Quiet, long DRAM launches are tight.** Both widths and both volumes per sample, 512 MiB and 2 GiB, pass all three capture rules in every quiet run. Launch length beyond about 2 ms is not what decides stability.
2. **CPU memory traffic slows GPU DRAM time by a quarter to four fifths.** Four copying threads take about 118 GB/s, close to half of the GPU's DRAM read rate, and every point slows by 21% to 26%; sixteen threads slow it by 66% to 83%.
3. **A steady load leaves the samples tight.** The slowdown is uniform, so the dispersion rules do not see it; a load that starts and stops during a capture, like a build, would split one point's samples into two levels as in the H5 captures. That intermittent case was not run here.
4. **PSI memory and the load average do not show it.** PSI memory stayed at 0 under 126 GB/s of copying (it counts reclaim and fault stalls, not bandwidth); the 1-minute load average lags by tens of seconds (6.15 in a quiet run after the 16-thread load). The busy share from `/proc/stat` per second and the list of busy processes show it at once.

## Precondition for GB10 DRAM captures

- Before and throughout a capture, sample `/proc/stat` and the process list once per second; the capture is void if the busy share of all CPUs rises above the quiet level (5% to 6% here, of which the capture's own host thread is most; a limit of 10%, two CPUs of twenty, leaves margin) or any process outside the capture uses more than 5% of a CPU for more than one sample.
- Other sessions' CPU-heavy work on the Spark (builds, large copies, data processing) conflicts with GPU DRAM captures as much as GPU work does, so it should wait for the GPU lock or the capture should stop.
- Longer launches alone are not enough on GB10, and neither PSI memory nor the load average can stand in for the busy-CPU check.
