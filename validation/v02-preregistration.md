# VirtualGPU v0.2 preregistration

## Evaluation and scope

The evaluation has two independent decisions, one for RTX 4070 and one
for GB10. Each device has a GEMM cohort and a fused attention cohort. The
GB10 evaluation tests cross-device performance on new workloads. GB10
GEMM results from Phase 2 are already open and may inform development.
The earlier Phase 1 and Phase 2 results remain frozen.

The model and frontend come from the reviewed v0.2 release scope. The
L1 spill capacity transition and spill reload latency work are outside
v0.2 and belong in the release's known limits. GPU performance counters
may support independent microbenchmarks and hardware validation. GEMM
and attention timings do not tune model or profile parameters.

## Cohorts and splits

| Device | Kernel | Workload holdout | Candidate combinations |
| --- | --- | --- | ---: |
| RTX 4070 | GEMM | 9 new shapes | 324 |
| GB10 | GEMM | 9 different new shapes | 324 |
| RTX 4070 | Attention | 8 existing sealed Phase 4 holdout shapes | 228 |
| GB10 | Attention | 8 new shapes | 228 |

[GEMM manifest](v02-gemm-holdout.json). Each device has two decode-like
shapes, two shapes with unaligned M, N and K, a triplet around a resident
wave boundary, and two aligned prefill-like shapes. Every M, N and K in
each new shape is absent from its respective axis in the opened GEMM
workloads, including the GB10 TMA diagnostic. The reference wave uses
`m64n128k32-w4-s4-g1` at two resident CTAs per SM. CPU-only compilation
confirmed 128 registers per thread and 36,864 shared bytes for each
reference. The triplets launch 91/92/93 and 95/96/97 CTAs on the 46-SM
and 48-SM profiles, respectively. Recheck occupancy with the capture
resources before timing. If it differs, label the observed boundary
and keep the enumerated shapes fixed. Each case attempts the existing 36 configurations.

[Attention manifest](v02-attention-holdout.json). The 4070 uses the
original [manifest](attention-suite.json) and
[seal](attention-holdout-seal.json). The GB10 matrix uses fresh sequence
lengths and has one case for each combination of causal flag, head
grouping (MHA or GQA), and head dimension (64 or 128). It includes odd
tails and longer aligned lengths. Both cohorts use FP16 prefill forward
with equal query and KV sequence lengths and the reviewed tutorial
candidate space. Decode attention is evaluated separately in future
work.

The historical data are development inputs for this release. Across
cache regimes, the learned GEMM baseline uses all eligible labels from
the RTX 4070 development suite, the published RTX 4070 workload holdout,
and the GB10 primary and secondary GEMM suites. Use the original warm
and replacement clean-cold labels where applicable. The historical TMA
diagnostic stays separate. Fit the attention baseline on all eligible
RTX 4070 attention development labels. The two families have distinct
feature builders and fits. Keep original split and device provenance
when deriving v0.2 training roles, and preserve cases measured on both
devices as separate observations with their own profiles.

## Freeze and label release

1. Review and commit these matrices and this plan before timing any new
   holdout workload. No holdout result may guide development or target
   selection.
2. Capture the three new holdout cohorts using the approved kernels,
   toolchain and harness revisions. The existing RTX 4070 attention
   reports stay sealed. Record the candidate outcomes, archive setup
   and toolchain metadata, and publish a receipt with SHA-256 and UTC
   time for every mode-zero raw report. Collecting sealed reports may
   precede the freeze.
3. Tag the final clean snapshot `v0.2-final`. Record the full tag and
   commit IDs, core build, frontend and exact compiler/assembler
   settings, both hardware profile digests, scoring version, manifest
   and seal digests, and the baseline feature and training-input
   digests. Freeze the baseline coefficients from the development
   inputs above, including their normalization and intercepts. The
   profiled rates come from hardware specifications and microbenchmarks.
4. Before opening any v0.2 labels, archive the three predictors'
   outputs for all four cohorts, including unsupported or failed
   candidates. Reproduce the capture artifacts offline for the frozen
   frontend and check the versioned AOT/JIT identity. Publish a freeze
   amendment with fit and prediction archive SHA-256 digests and UTC
   times. Formal core predictions require a clean build whose commit
   matches `v0.2-final`.
5. Once the freeze amendment and all seal receipts are published and
   release is approved, verify the archive and every report digest
   before parsing labels. Use the frozen predictions, fits, scoring
   implementation, and gates. Publish the four cohort results and the
   two device decisions, including any missing or invalid candidates.

Capture and score tooling for the new cohorts must enforce the
`v0.2-final` release gate and the corresponding digest checks. The
historical post-hoc scoring route is for previously published releases.

## Measurement protocol

Use CUDA Graph events for one kernel launch. Each case has three
sessions of ten randomized complete blocks, with order seeds 101,
102, and 103. Warm trials prime the kernel and buffers. Cold trials
scan a disjoint read-only eviction buffer with L2-caching,
L1-bypassing loads, at least 128 MiB and three times the queried L2
capacity. Verify cold eviction with the read probe. Correctness,
compilation and cache preparation run outside the interval.

Use the per-device GPU lock. Record device identity, driver, clock and
thermal telemetry, compute process list, and power policy. Use
record-only clocks when the operator cannot lock and verify them. On
GB10, keep CPU and GPU memory contention out of the run. Use the
published GB10 thermal gate: at least 60 seconds of untimed warm-up,
adjacent 30-second temperature means within 0.5 C, SM clock medians
within 1%, and both windows clear of thermal and power slowdown.
Keep the 20-minute stabilization limit and preserve timeout telemetry.
[GB10 thermal audit](phase2-thermal-trend-audit.json).

Record compile, launch, reference and artifact failures separately.
Verify hash protocol v2 PTX and TTGIR, target, assembler and compiler
settings, launch grid, registers, and shared memory before accepting a
label. Preserve every eligible measured configuration in the shortlist
oracle even when a predictor omits it. Repeat measurements only for
documented protocol failures.

## Independent baselines

Retain the GEMM roofline + wave and NeuSight-style feature definitions.
For attention, use an analytical query-tile workload. For each query
tile, take the valid key prefix length (the whole sequence for full
attention, or the sequence clipped at the end of the query tile for
causal attention). Round it up to key tiles for the padded work of QK
and PV. Count Q, O and FP32 LSE once per query row and the valid K and V
rows once per visited query tile and query head. Use actual compiled
resources for residency and the measured FP16 tensor rate, bandwidth
and launch overhead from the target profile. Roofline + wave uses the
average CTA's work and requested bytes with the existing wave-quantized
formula.

The attention NeuSight-style baseline uses the same tile counts,
requested traffic and target hardware rates. Its floor is the maximum
of padded tensor work at the measured rate, requested bytes at L2
bandwidth, and compulsory bytes at DRAM bandwidth. Compulsory bytes
count each Q, K, V, O and LSE tensor once, subtracting the L2 capacity
for warm trials with a floor of zero. The fit uses the GEMM baseline's
ridge penalty of 1 and bounded log slowdown over this analytical floor.
Its features use base-2 logs for padded FLOPs and requested bytes per CTA,
log working set relative to L2, wave fill fraction, log resident CTAs
per SM, log compute-to-L2 time ratio, log query and key tile sizes,
log head dimension, log query-to-KV head ratio, log warp and stage
counts, causal flag, and warm flag. Use one median per eligible
configuration and cache regime. Pool the allowed training inputs once
per family and use the frozen fit on both devices.

## Scoring and decisions

Use [design section 8](../docs/design.md#8-validation) without changing
its thresholds. Evaluate each kernel family and cache regime separately
within each device, with equal weight per case. Report rank coverage
and undefined tau-b cases, top-5 hit and regret, pair confusion,
directional precision and coverage, and descriptive absolute error.
Use the nominal ranks for the shortlist. For pair verdicts, use draw 0
plus 32 shared perturbations, seed 17, and the fixed 5% log-ratio band.
Require a common direction and at least the margin in every core draw.
Baselines apply the same band to their nominal predictions. For
hardware, use a paired 95% bootstrap interval with 1,000 draws and seed
17, resampling complete blocks within each session. Keep equivalent
and unresolved pairs visible.

A cohort passes in a cache regime when median Kendall tau-b exceeds
both baselines by at least 0.05, mean top-5 regret is at most 1%, and
mean regret is at most 0.5 percentage points worse than each baseline.
A missing shortlist on an eligible case cannot satisfy the shortlist
criterion. Each device passes only when both kernel families meet all
three criteria in both cache regimes. Report the RTX 4070 and GB10
decisions independently. A secondary diagnostic never changes a
decision.

## v0.2 final freeze record

The annotated tag object `ea2ced7e13023f528b859e7ad8860f9373d773e8` resolves to
`e70824b9c3ea5d6818a5bef306ee21c9e5d2d74a`. That commit identifies the
core, Python frontend, and scoring implementation. The build is clean
and formal. The SHA-256 of `validation/v02_score.py` is
`05d4cb57861703636f78d926cb9ef988768263b8203e59ce585d78c35b657be9`.

| Artifact | Recorded UTC | SHA-256 |
| --- | --- | --- |
| [Frozen NeuSight-style fit bundle](v02-frozen-fit.json) | 2026-11-26T02:15:06.877007+00:00 | `9bbd965c362aed8f55dbb01224e76aef5073074ef726753a4ad9b045e860bcd4` |
| Prediction archive | 2026-11-26T03:08:34.217587+00:00 | `74caeeeb72380b0961b26db2f9ec63ebf1974b1153266b0c84bd2ff404f06d93` |
| Frozen AOT index | 2026-11-26T03:08:34.217587+00:00 | `8d752498cb7a015428a960dd1a07a10e003cc84a3891bf135776287c5512e0fd` |

The [machine-readable amendment](v02-freeze-amendment.json) binds the
binary, profiles, manifests, seals, and per-method status counts to this
archive. All 1,104 regenerated AOT artifacts matched the capture
artifacts. Each attention cohort contains 40 unsupported candidates
per method and cache state. The archive retains all 6,624 prediction
rows and their statuses. The prediction and regenerated IR files are archived in
`results/v02-formal/`. The raw historical input hash indexes, feature
vectors, and deterministic row order are archived in
`results/v02-freeze/*-training-inputs.json`. The local fit driver is
`results/v02-freeze/fit_historical.py` (SHA-256 `5e59841dc99116f0109e233cf2c67f3d53e8204ffc14ffb961b33a6b9153f65e`).
Training rows retain their original split and device. Both fits use
ridge penalty 1 and one median per eligible case, configuration, and
cache regime. The published fit bundle records the normalization,
coefficients, intercepts, and feature and training-input hashes.

| Family | Source | Training rows |
| --- | --- | ---: |
| gemm | gb10-phase2-primary | 510 |
| gemm | gb10-phase2-secondary | 648 |
| gemm | rtx4070-m2-development | 510 |
| gemm | rtx4070-phase1-workload | 648 |
| attention | rtx4070-attention-development | 376 |

Triton 3.9.0 uses artifact hash version 2, FP fusion on,
assembler optimizations on, and no extra PTX options. The frontend
runs the assembler with `-lineinfo -v --regAllocOptLevel=2` and the
target-specific GPU name.

| Device | Profile revision | Requested target | PTX target | Assembler |
| --- | --- | --- | --- | --- |
| rtx4070 | `0.21.0-posthoc` | `sm_89` | `sm_89` | `ptxas` 12.9 (V12.9.86) |
| gb10 | `0.21.0-posthoc` | `sm_121` | `sm_121a` | `ptxas-blackwell` 13.4 (V13.4.59) |

The four v0.2 label cohorts remained sealed during fitting and
prediction. Label release requires publication of this amendment
and approval under the preregistered procedure.
