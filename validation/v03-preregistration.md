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
and both cache regimes must pass for a device to pass. GB10 tests
cross-device prediction on fresh workloads using its calibration
profile. Both devices were used in earlier releases, so this test
measures transfer on unseen workloads for these devices.

Select shapes from a versioned inventory of every previously opened
development, diagnostic and holdout workload, including the released
v0.2 cohorts. Reject previously opened exact workload tuples on either
device. Use different tuples on the two devices and record the
novelty audit. Determine eligibility and case mix from the workload
specification, compiler resource reports and profile metadata before
any timing or ranking preview.

- **GEMM.** Cover decode-like and aligned prefill-like workloads,
  shapes with non-aligned M, N and K, and an adjacent triplet below,
  at and above a resident-wave boundary per device. Favor novel
  dimensions as well as new tuples. Declare the reference configuration
  and verify its occupancy and CTA counts offline from compiled
  resources. Use the reviewed 36-config space as the starting point.
- **Attention.** Cover forward prefill with equal query and KV lengths,
  causal and non-causal, MHA and GQA, head dimensions 64 and 128,
  aligned lengths, and distinct odd sequence tails. Include both short
  and longer sequence regimes on both GPUs. Use the existing
  versioned Triton configuration space as the starting point.
- Freeze exact case counts, configuration filters and manifests in a
  reviewed amendment before any v0.3 workload labels are collected.
  Record every attempted configuration and why any attempt is invalid.

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
families. The draft uses historic eligible RTX 4070 labels only for the
frozen learned fits, including for transfer to GB10, as specified in
[design section 8](../docs/design.md#8-validation). Freeze the exact
training inventory, profile features, fitted coefficients and feature
builders before opening the new labels. Previously opened GB10 labels
may support post-hoc diagnosis. Use hardware specifications and
isolated microbenchmarks for model/profile calibration.

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
