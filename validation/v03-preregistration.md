# VirtualGPU v0.3 preregistration draft

This draft defines the evaluation and freeze procedure. Release mechanism
scope is pending the H1 decision. Publish a reviewed amendment naming
the included changes and the frozen implementation before opening any
v0.3 holdout labels. Earlier release decisions remain frozen.

## Evaluation design

| Device | GEMM | Attention |
| --- | --- | --- |
| RTX 4070 | Fresh workload holdout | Fresh workload holdout |
| DGX Spark GB10 | Separate fresh workload holdout | Separate fresh workload holdout |

Report four cohorts and two independent device decisions. Both families
and both cache regimes must pass for a device to pass. The evaluation
measures generalization to new workloads on two previously studied
devices. VirtualGPU uses each target's calibration profile. The
learned comparator can use historical labels from both devices.

Publish a versioned inventory of previously opened development,
diagnostic and holdout workloads through the v0.3 manifest approval.
Use disjoint workload tuples on the two devices and publish the
novelty audit. Determine eligibility and case mix from the workload
specification, compiler resource reports and profile metadata before
any timing or ranking preview.

- **GEMM.** For each new tuple, require that M, N, and K are each
  absent from the corresponding historical axis across both GPUs.
  The inventory must include RTX 4070 M2 development, Phase 1 workload
  holdout, GB10 Phase 2 primary, secondary and TMA diagnostic, and both
  v0.2 GEMM holdouts. Include other opened GEMM datasets from the
  inventory. Cover decode-like and aligned prefill-like workloads,
  shapes with non-aligned M, N and K, and an adjacent triplet below,
  at and above a resident-wave boundary per device. Declare the
  reference configuration and verify its occupancy and CTA counts
  offline from compiled resources. Use the reviewed 36-config space
  as the starting point.
- **Attention.** Require every new `seq_len` on either GPU to be
  absent from the entire opened attention inventory, including Phase 4
  development, the Phase 4/v0.2 RTX 4070 holdout and the v0.2 GB10
  holdout. Use disjoint `seq_len` sets across the two device cohorts.
  Cover forward prefill with equal query and KV lengths, causal and
  non-causal, MHA and GQA, head dimensions 64 and 128, aligned lengths,
  and distinct odd sequence tails. Include both short and longer
  sequence regimes on both GPUs. Use the existing versioned Triton
  configuration space as the starting point.
- Freeze exact case counts, configuration filters and manifests in a
  reviewed amendment before any v0.3 workload labels are collected.
  Record every attempted configuration and why any attempt is invalid.

## Candidate manifests for review

The [GEMM candidates](v03-gemm-holdout.json) cover nine shapes and
324 planned configurations per device. Each wave triplet uses one
output tile in M and moves N across three adjacent CTA counts while
holding M and K fixed. All three shapes in each triplet have the same
64-column tail for the reference configuration. Aligned prefill cases
are divisible by the largest candidate tiles on each axis.

The [attention candidates](v03-attention-holdout.json) cover eight
shapes and 228 planned configurations per device, one for each
causality, grouping, and head dimension stratum. Each device has
four sequence lengths aligned to both query tile sizes and four odd
sequence lengths. All 16 sequence lengths are distinct.

The [historical inventory](v03-holdout-history.json) records each
source manifest's SHA-256 and excluded axes or sequence lengths.
It includes the original M1 suite, the materialized Phase 1 and
Phase 2 manifests, and all v0.2 workload groups. The historical
checks use manifest metadata and published seal/index digests.

The [CPU AOT audit](v03-holdout-aot-audit.json) records resource status
by case, including every configuration exceeding a profile limit,
and identifies the full resource reports by SHA-256. Resource
eligibility is a static check. Capture still checks the approved
toolchain, compiled resources and JIT artifacts. Review the candidate
manifests and release them in a separate amendment before collection.

| Cohort | Cases | Compiled | Fits profile | Exceeds a resource limit | Wave boundary CTAs |
| --- | ---: | ---: | ---: | ---: | --- |
| RTX 4070 GEMM | 9 | 324 | 324 | 0 | 91 / 92 / 93 |
| GB10 GEMM | 9 | 324 | 324 | 0 | 95 / 96 / 97 |
| RTX 4070 attention | 8 | 228 | 188 | 40 | n/a |
| GB10 attention | 8 | 228 | 188 | 40 | n/a |

The GEMM wave reference is `m64n128k32-w4-s4-g1`. AOT reported
128 registers per thread and 36,864 bytes of shared memory for all six
boundary cases. Each admits two resident CTAs per SM. Keep all
planned configurations in the manifests, with resource-limit flags
available for review and later capture.

The audit used Triton 3.9.0, `ptxas` 12.9 (V12.9.86) for `sm_89`
and `ptxas-blackwell` 13.4 (V13.4.59) for the GB10 PTX target
`sm_121a`. The profiles are RTX 4070 `0.21.0-posthoc` and GB10
`0.22.0-posthoc`. The audit did not collect timing or run either GPU.
The complete CPU review bundle is
`results/v03-holdout-review.tar.gz` (SHA-256
`0063e98648cadd13f7773fc442527ec273929765c65e841226ab2834e88e28dd`).

Run the [candidate generator](v03_candidates.py) on a checkout with
the archived historical manifests and AOT index:

```sh
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m validation.v03_candidates results/v03-candidate-audit
```

## Hardware and baselines

Reuse the reviewed v0.2 capture protocol. Measure warm and clean-cold
with single-launch CUDA Graph events, three sessions of ten randomized
complete blocks, seed schedule 101 to 103, clean read-only L2 eviction
and a valid cold read probe. Check AOT/JIT artifact identity.
Use the GPU lock, telemetry, process checks and the existing GB10
thermal/clock gate. Record any documented protocol failure, including
its immutable receipt and any authorized retry.

Compare with roofline plus wave quantization and the independently
implemented NeuSight-style tile baseline, each for both kernel
families. Train one frozen learned fit per family on all eligible
published historical labels from both devices. Use the same fitted
coefficients on both devices with the target's hardware features.
The currently published sources are:

| Family | RTX 4070 | GB10 |
| --- | --- | --- |
| GEMM | M2 development (original warm and clean-cold replacement), Phase 1 workload holdout, v0.2 GEMM holdout | Phase 2 primary and secondary cohorts, v0.2 GEMM holdout |
| Attention | Phase 4 development, Phase 4/v0.2 holdout | v0.2 attention holdout |

Preserve the original split, device and source receipt for every
training observation. Keep observations of the same workload on
different devices distinct. Derive one row per eligible case,
configuration and cache regime for each source. Use effective valid
sessions and deduplicate reused reports, including the Phase 4/v0.2
RTX 4070 attention holdout. Exclude the Phase 2 TMA diagnostic,
invalid or failed captures, and all v0.3 holdout labels. Freeze the
exact training inventory, data digests, profile features, feature
builders and fitted coefficients before opening the new labels.
Use hardware specifications and isolated microbenchmarks for
model/profile calibration.

## Freeze and release

1. Publish the manifests, split audit, candidate space, procedure and
   scoring implementation. Capture labels only after the plan is
   approved. Store mode-zero reports and publish SHA-256 and UTC
   receipts without reading the timings.
2. At the clean release tag `v0.3-final`, freeze the model, frontend,
   toolchain, profiles, scoring implementation and both baseline fits.
   Record the build ID and the hashes of the profiles, fits, source
   inputs, manifests, seals and capture artifacts.
3. Generate and archive core, roofline and NeuSight-style predictions
   for all four cohorts, including unsupported and failed rows.
   Publish a freeze amendment with digests and UTC timestamps.
4. After the amendment is on the remote branch and release is
   approved, verify the archive and every sealed report digest before
   interpreting timing data. Score once with the frozen implementation.

## Decisions

Apply [design section 8](../docs/design.md#8-validation) independently
to each kernel family and cache regime:

- Median Kendall tau-b reaches at least 0.05 above each baseline.
- Mean top-5 regret is at most 1% and at most 0.5 percentage points
  worse than either baseline.
- Any eligible case missing a core shortlist fails both shortlist
  criteria. The oracle includes all hardware-eligible candidates.
- Publish per-case and aggregate scores, top-5 hit rate, paired
  judgments using the 5% resolution band, directional agreement,
  latency error, read probe results, coverage, and config status counts.
  The first two bullets define the numerical pass criteria.

The next review fixes the mechanism scope, manifests, exact toolchain
and training inventory. Record changes in an amendment before label
release.
