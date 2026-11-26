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

## A build next to the capture

Other sessions build `vgpu_bench` on the Spark while a capture holds the GPU lock, so a second session of the same eight points cycled quiet, build, quiet, build, quiet three times (`--conditions build`): during the build runs a loop of `microbench/build.py --arch sm_121a` ran in its own process group from 5 s before the GPU process to 5 s after it. Captured 2026-11-26 23:26 to 23:28 UTC (thermal gate 50.7 / 50.7 C, 2515 / 2522 MHz). Summary: [gb10-dram-build-posthoc.json](gb10-dram-build-posthoc.json); raw capture `dram-dispersion-build-1.json` (SHA-256 `af3f0a14...`) in `gb10-dram-build-captures.tar.gz` (`ba7335b31339dea76761e37d8cbe1388233af0192d450bd8a243b6ef2f3f962b`), in the same two places.

- During every build run the compiler front end (`cicc`) used one full CPU and about 600 MB; measured separately under the GPU lock, a build loop raises the busy share of the 20 CPUs from 0.3% to 5.2%, all of it `cicc`.
- Point medians during the build stayed within -0.7% to +2.3% of the quiet runs, no list had an interquartile range above 5.2% of its median, and every run passes the rules. A build does not disturb GPU DRAM timing the way a copying load does.
- The stop of the build loop was not clean in this capture: `cicc` outlived the loop's SIGTERM and ran on into the quiet run after each build run; only the first quiet run of each cycle was free of it. Those quiet runs were as tight as the clean ones. `stop_group` in `microbench/dram_dispersion.py` now waits for the whole process group and kills what remains after 30 s.

## Precondition for GB10 DRAM captures

The CPU-side load that matters is memory traffic, which neither the CPU share nor any counter readable without privileges separates from harmless work such as a compiler front end. The check is therefore conservative: it voids a capture during any sustained foreign CPU work, and so also during a build that would not have mattered.

- `microbench/host_quiet.py` implements it for the capture scripts: `sample()` once per second during the capture (from `/proc/stat` and each process's `/proc/<pid>/stat`), then `check(samples, roots)` with the capture's own process IDs. A capture is void if the busy share of all CPUs exceeds 10% in any second (the capture's own host thread spinning on events takes about 5%) or if a process outside the capture and its descendants uses more than 5% of one CPU in two consecutive seconds. Per-process CPU comes from deltas of `/proc/<pid>/stat`, since `ps` %CPU is a lifetime average.
- Other sessions' CPU-heavy work on the Spark should wait for the GPU lock; otherwise the check voids the capture.
- Longer launches alone are not enough on GB10, and neither PSI memory nor the load average can stand in for this check.

## Lock incident (provenance)

At about 2026-11-26 23:34 UTC, while the GB10 long-launch MLP capture held the shared GPU lock, this study's author ran `~/virtualgpu/s8-narrow/build/vgpu_bench --mode query` on the Spark over SSH without taking the lock, to read the device's `integrated` field for the host gate. The query creates a CUDA context and reads device properties, runs no kernel and exits within seconds. The capture's compute-process check saw it (PID 1557838) during the warmup of its second L2 process, which exited before any timing and produced no report ([v0.3-gb10-mlp-long-capture.md](v0.3-gb10-mlp-long-capture.md)); no recorded data was affected. Every other GPU run of this study, the two sessions above and a CPU test of a build loop at about 23:30 to 23:31 UTC, ran under `flock /tmp/virtualgpu-gpu.lock`, the same lock file the captures use, with the build loops and CPU loads as children of the locked script. Since then, any call on the Spark that creates a CUDA context, a device query included, takes the lock first, and device properties are read once and kept rather than queried during someone else's capture.

