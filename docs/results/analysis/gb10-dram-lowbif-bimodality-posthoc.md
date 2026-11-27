# GB10 low in-flight DRAM bimodality (post-hoc)

Status: post-hoc, record only. Three rounds, no cause found. Written so the core owner can decide how to treat the affected profile points.

## 1. The observation

In the GB10 long-launch MLP capture (session `gb10-3169632`, [v0.3-gb10-mlp-long-capture.md](v0.3-gb10-mlp-long-capture.md)), DRAM points with little in flight (up to about 256 KiB) failed rule 1 with the host CPU inside the quiet thresholds. Many of them split into two clusters 1.3 to 1.5 times apart: for example 7 and 10 GB/s at 4 KiB (`cp_async`, 1 CTA), 89 and 138 GB/s at 64 KiB (`ld_global_v4`, 16 CTAs). Points with more in flight did not split.

## 2. Facts from the existing data and the code

Both are post-hoc readings, made after the captures were known to fail.

**Slow samples repeat across processes.** Marking each sample of the 5 `mlp` reports as slow (`L`) when it is below 85% of the point's largest sample:

- `cp_async` 4 KiB reads `LL...L.LL......` in processes 2 and 4 and `L.LLLL....L.LL.` in processes 3 and 5, character for character.
- The shape `L..LLLLL` (one slow sample, two fast, five slow) recurs at other points (`ld_global_v4` 8, 64, 192 and 224 KiB) with different offsets.

The slow state is therefore not independent from sample to sample. It is an episode of fixed shape lasting about 8 samples (some 15 ms of flush and launch), and its timing repeats when the same process runs the same sequence. `nvidia-smi` reports no memory clock on GB10 (`[N/A]`); the SM clock moved between 2418 and 2496 MHz.

**The low in-flight points are latency bound.** Disassembly (local nvcc 13.5, `sm_121a`):

- `mlp_ld_v4_kernel<1>` keeps its data dependency: each load's address waits for the previous load, so one 16 B load per thread is in flight. Bandwidth is then bytes in flight over latency, and the two clusters at 4 KiB correspond to latencies of about 410 and 590 ns.
- `dram_read_kernel` (the active-SM curve) issues 4 loads per thread back to back: 32 KiB in flight for one CTA of 512 threads, 64 GB/s on one SM.
- The streaming kernel of round 2 (`sm_stream_kernel`, nominally 8 loads per thread) is scheduled with its XOR chain between the loads, so fewer are in flight; this explains its 45 GB/s on one SM without any cached data.

## 3. Rounds

All on Spark 1 under the GPU lock, after the thermal gate and the 10 s host gate.

| Round | Test | Result |
|---|---|---|
| 1 | Flush timing: `dram-lowbif`, 16 B loads on all SMs at 24, 96, 384 and 1536 KiB in flight, 30 samples each | No split; interquartile range 0.2% to 2.3% of the median |
| 2 | Per-SM placement: `dram-sm-map`, one CTA streaming 96 MiB on each SM in turn, then 200 single-CTA launches | Per-SM bandwidth uniform, 44.5 to 46.8 GB/s over 48 SMs; single CTAs all landed on SM 0, median 44.6 GB/s, 3 of 200 samples 24% to 40% slow |
| 3 | `dram-bimodal` (below) | The original protocol did not reproduce the split |

**Round 3 design** (reviewed before capture): the MLP kernels and the active-SM kernel on their own protocol at five points with the frozen lengths of `gb10-3169632` (`cp_async` 4 KiB and 48 KiB, `ld_global_v4` 64 KiB and 224 KiB, `dram-bw` 1 SM), 200 samples per point, one process per condition: original; no flush with a cold region per sample (`rotate`); the flush kept with a cold region (`rotate-flush`); a streaming kernel with the same bytes in flight per CTA (`stream`); four times the frozen length, capped by the 1 GiB buffer (`long`); original again. Probed copies of the kernels record each sample's CTA count, SM set and CTA 0 clock; their loops keep the original memory instruction sequence (integer scheduling and registers differ slightly). Frozen reading: a point is bimodal when at least 5% of its samples fall below 0.85 times its 90th percentile; a condition removes the split when both originals have at least 3 bimodal points and the condition at most 1; if the originals have fewer than 3, the capture is record only.

**Capture 1 (void).** Three of six processes failed the host check on `systemd` and `mysqld` at 8% to 9% of one CPU. The runner sampled processes every 100 ms, so the rule "above 5% of one CPU in two consecutive seconds" was applied to two consecutive 100 ms intervals, stricter than the rule. The capture is void and kept unread; the runner was fixed to sample processes once per second (`b7edf39`), and the capture repeated with the same binary.

**Capture 2.** 2026-11-27 00:49:02 to 00:49:24 UTC, every process passed the host check (busiest second 5.5% to 5.7% of all CPUs; each process lasted about 3.5 s, so its check covers one or two whole seconds). Every launch ran on the intended CTA count and SMs; SM clocks 2412 to 2566 MHz.

| Condition | Bimodal points | Interquartile range / median | Medians (GB/s): 4 KiB, 48 KiB, 64 KiB, 224 KiB, 1 SM |
|---|---|---|---|
| original | 0/5 | 0.2% to 0.7% | 9.39, 108.1, 137.2, 233.6, 63.8 |
| rotate | 0/5 | 0.2% to 0.9% | 9.44, 109.4, 139.6, 231.8, 64.6 |
| stream | 0/5 | 0.1% to 0.5% | 9.28, 107.6, 139.7, 239.2, 35.8 |
| long | 0/5 | 0.1% to 0.4% | 9.35, 107.8, 137.1, 232.8, 63.4 |
| rotate-flush | 0/5 | 0.1% to 1.0% | 9.34, 108.6, 138.6, 232.2, 64.0 |
| original | 0/5 | 0.2% to 0.5% | 9.48, 108.9, 138.3, 233.8, 63.7 |

The original protocol did not reproduce the split, so by the frozen reading the capture judges no condition. Every median sits at the fast cluster of `gb10-3169632` (10, 108, 137 to 139, 227 to 235 GB/s). The streaming kernel's 1 SM point (35.8 GB/s) issues 7 of its 8 loads back to back and the 8th after the first returns, so it is not a like-for-like replacement for `dram_read_kernel` there.

## 4. What is known

- Run alone, the affected points are stable: five of them, 200 samples each, in six processes and five protocols, never split (interquartile range at most 1%). Because the original protocol did not split either, round 3 cannot say whether the flush, the addresses, the kernel or the launch length matter; it only shows that none of them produces the split by itself.
- Per-SM DRAM bandwidth is uniform within 5%, so SM placement does not explain a 1.3 to 1.5 times split; the host CPU stayed inside the quiet thresholds in the split captures.
- Within `gb10-3169632` the slow episodes have a fixed shape and repeat with the process's sequence. That capture ran all 96 `mlp` points in one process, 15 samples each; round 3 ran 5 points, 200 samples each. What differs is the sequence around a point, so the state that slows latency-bound points is likely set by the preceding work in the sweep or by the session (the clocks of a unified-memory fabric that `nvidia-smi` does not report), not by the point itself. This is a hypothesis; it was not tested.

## 5. Options for the core owner

1. Mark the GB10 DRAM points up to 256 KiB in flight as low in-flight unstable, with an uncertainty that spans both clusters, and move on.
2. One more test, of the sequence hypothesis only: the full `mlp` sweep under its frozen schedule with the probed kernels and per-sample host times, twice, to see whether slow episodes start at the same place in the sequence.

## 6. Provenance

- Code: `s8/v03-dram-lowbif` (`dram-lowbif`, `dram-sm-map`, `dram-bimodal`, `microbench/dram_bimodal.py`); round 3 binary built on Spark 1 with CUDA 13.0, SHA-256 `e38e8d8a455d5cd22d74ba70640a1848d9ff48d1f2fa141499d1c01796062b85`.
- Captures: `dram-bimodal-1.json` (void, `a2991e04...`), `dram-bimodal-2.json` (`f40ac9a4...`), its summary (`19515b6b...`), archived on Spark 1 as `~/virtualgpu/s8-narrow/runs/gb10-dram-bimodal-captures.tar.gz` (`baa5c09b...`); rounds 1 and 2 in `~/virtualgpu/s8-ll/runs/` (`lb-quick.json`, `sm-map-1.json`).
- Source data for section 2: `results/v03-mlp-long/gb10-3169632/capture/mlp-*.json` in the archive cited by the MLP capture report.
