# Phase 2 preregistration: RTX 4070 to DGX Spark

This experiment tests profile-only transfer of Triton GEMM rankings from the
RTX 4070 to DGX Spark GB10 (`sm_121`). GB10 is the held-out device. Its Grace
CPU and GPU share LPDDR5x system memory. [NVIDIA compute capability table](https://developer.nvidia.com/cuda/gpus), [DGX Spark hardware overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html).

## Freeze and data access

- Commit this reviewed plan before timing any GB10 GEMM, including pilots and
  diagnostics. Finish Phase 1 development and tag the agreed model and
  frontend snapshot `phase1-final`. At that freeze, refit and seal the
  NeuSight-style baseline using the full RTX 4070 development split.
  Freeze these inputs before opening the RTX 4070 workload holdout.
- Use `phase1-final` for the formal RTX 4070 workload-holdout score. Use that
  same immutable tag and the same frozen fit for both GB10 evaluations.
  Once the tag is created, add a separate preregistration revision with its
  full SHA, the frontend SHA and exact toolchain versions, the holdout manifest
  digest, and the source fit and training-input digests. Commit the revision
  before any timed GB10 GEMM. The formal binary must report the tagged
  snapshot SHA, `dirty=false`, and `result_class=formal`.
- AOT extraction uses the tagged clean checkout, hash protocol v2, and FP
  fusion enabled. Retain Triton `3.9.0` after the GB10 compatibility
  check below. Resolve any future toolchain change before freezing the
  toolchain and opening the Phase 1 holdout.
  Record the exact Triton release, bundled assembler version, PTX target
  suffix, and codegen options for both AOT and JIT.
- The validation owner may adapt the hardware harness and data-loading
  orchestration for GB10. Record their revisions and the toolchain before any
  timed GB10 GEMM. Build and run the formal predictions from the pinned clean
  checkout. The pinned model, frontend, kernel sources, case matrix, and
  scoring rules govern the run.
- Run the isolated calibration suite on GB10, seal the resulting profile
  and target AOT artifacts, and archive the three prediction sets before
  opening timing labels to the model and baseline authors. Record
  SHA-256 digests and UTC timestamps. A change to the plan requires a reviewed
  amendment before timing. Later changes belong to a separately labeled run.

## Phase 1 freeze record

The annotated tag `phase1-final` has tag object `ca04403df570d46fd83bd0a3be900417ea7d66ee`
and resolves to `129a6f55d3dcbf5c261a8397296400f91f291fe1`. The Phase 1 binary reports that full commit,
`dirty=false`, and `result_class=formal`. The tagged frontend includes
`18af028a26dea8b5014225a1ebdc230bd86a8207`, the single-stage access-map fix.

The RTX 4070 evaluation uses profile `0.11.0-m3`, Python `3.12.9`,
Triton `3.9.0`, bundled `ptxas` `12.9 (V12.9.86)` for `sm_89`,
FP fusion enabled, and artifact hash protocol v2. The core was built with
Rust `1.99.1`. The GB10 assembler and smoke environment are recorded
separately below.

| Artifact | SHA-256 | UTC recorded |
| --- | --- | --- |
| Formal core binary | `f94d6732bf02b9ae69c8eb95ddb82c329f7403490b017aa69040fe399d975e0f` | 2026-11-25T09:37:47.692479+00:00 |
| RTX 4070 profile | `8ddab1c37b975aaf5390d160654ae34fd8ae04301ad64df1022c26896f3a0aa8` | 2026-11-25T09:37:47.692479+00:00 |
| RTX 4070 development training input index | `80df250aebfd782cbcfca9bbb539cdb9edbeaf5b0b0682274913ced53598b754` | 2026-11-25T09:38:13.038675+00:00 |
| Frozen NeuSight-style fit | `20f7b3869b8e7e79fba98a05d91d38c6c305a3a3bdfe475bd7838111a264ed72` | 2026-11-25T09:38:13.038675+00:00 |
| RTX 4070 workload-holdout manifest | `b6c8fd089543ec8444e820993acc962f25eb3e662f9d47d3fe3bb79dbf03eaf1` | 2026-11-25T08:48:48.998112+00:00 |
| Committed RTX 4070 holdout seal index | `14df3e947b38ea7f7cd9cda9be0ad7bcad33213b75506379ef9e47575f293ce9` | 2026-11-25T08:58:27.486686+00:00 |
| Regenerated AOT match index | `3585245d009ec9afc6c2064c4d339bf744fb167b783bf8f1b50e3c070c8a177a` | 2026-11-25T09:41:19.024605+00:00 |
| Three-method RTX 4070 prediction archive | `e6869636ce5f838398bd8f859bd06a671ae468f147c99f9bfabb410a027c4852` | 2026-11-25T09:41:19.024605+00:00 |

The NeuSight-style fit uses all 510 config and cache-regime medians from the
nine RTX 4070 development cases. The 324 regenerated workload AOT artifacts
match the PTX, TTGIR, toolchain, launch grid, and resources checked against
JIT at collection. Each method has 648 archived predictions. The core uses
draw 0 plus 32 shared draws with seed 17 and the fixed 5% pair band.

This record was prepared at 2026-11-25T09:41:37.830624+00:00. The RTX 4070 workload timings
remained sealed at preparation. Open them only after this revision is
published on main, the formal prediction archive digest is verified, and
each raw report passes its committed seal digest check.

### GB10 toolchain compatibility

The coordinator reported a successful correctness smoke on 2026-11-25 on
Spark 1 (GB10) with driver `580.173.02` reporting CUDA `13.0`. Triton
`3.9.0` used its bundled `ptxas-blackwell` `13.4`. Vector add and a GEMM
with `M=N=K=1024`, tile `(128,128,32)`, and three requested stages
produced correct results. The GEMM PTX targeted `sm_121` and contained
`mma.sync` and `cp.async`. The environment used Python `3.12`, PyTorch
`2.14.0+cu132`, and NumPy `2.3.5`. This record covers toolchain
correctness. The timed GB10 GEMM gate still requires the `phase1-final`
tag and the committed preregistration revision specifying the freeze.

## Candidate matrix

Use the nine already materialized M2 development cases as the primary transfer suite.
They are held out by **device** on GB10. The source manifest
`results/m2/suite.json` has SHA-256
`cbfe410e95f327943f6f0cd59250f217b874cbb4654a3a161f7e52e3756698d4`.
Carry the exact logical shapes and config IDs across devices. In particular,
the SM-round shapes use the original 4070 dimensions.

| Case | M | N | K | Configs |
| --- | ---: | ---: | ---: | --- |
| decode-1 | 1 | 4096 | 4096 | 36 common |
| decode-16 | 16 | 8192 | 4096 | 36 common |
| decode-32 | 32 | 4096 | 11008 | 36 common |
| mask-m | 129 | 4096 | 4096 | 36 common |
| swizzle-320 | 320 | 4096 | 4096 | `m64n64k32-w4-s2-g1`, `m64n64k32-w4-s2-g8` |
| m1-cube | 4096 | 4096 | 4096 | `m128n128k32-w4-s3-g8` |
| sm-round-below | 64 | 2880 | 4096 | 36 common |
| sm-round-exact | 64 | 2944 | 4096 | 36 common |
| sm-round-above | 64 | 3008 | 4096 | 36 common |

The 36 common configurations cross tiles `(16,64,32)`, `(32,64,32)`,
`(64,64,32)`, `(64,128,32)`, `(128,64,64)`, and `(128,128,64)` with
`num_warps` in `{4,8}` and `num_stages` in `{1,2,4}`. Use `GROUP_M=1`
and the pinned runner's `dense_gemm`. All primary cases use row-major FP16
inputs and output, FP32 accumulation, and input seed 17. This yields
255 attempted case/config combinations per cache regime.

### Secondary transfer on unseen workloads

After the formal Phase 1 RTX 4070 workload-holdout score is complete, run its
entire frozen workload-holdout manifest on GB10. It includes
`prefill-256` `(256,4096,4096)`, `prefill-1024` `(1024,4096,11008)`,
`prefill-2048` `(2048,8192,4096)`, and `mask-mnk` `(513,4103,4093)`.
Carry over the exact shapes and config IDs of any other cases materialized
in that manifest, with device-derived boundaries resolved on the 4070.
Do not regenerate them for GB10. Because the model and learned fit were
frozen before the Phase 1 workload holdout opened, this test changes both
the device and the workloads relative to development. Use the same
measurement and scoring rules and report this cohort separately. Only the
nine primary transfer cases enter the Phase 2 pass decision.

### GB10-only TMA diagnostic

Use the pinned `_matmul_tma_kernel` in
`python/virtualgpu/frontend/tests/test_triton_frontend.py` with shape
`(1024,1024,512)`, tile `(128,128,32)`, four warps, and stages 2, 3, or 4.
Use its 2-D grid `(8,8)` and the same dtype and input seed as above. Record
materialized stages, prefetch distance, exact MMA opcode, CTA-local tensor
copies, and their completion barriers. Validate the emitted tensor-copy
path and resource limits before timing. Report compile, launch, extraction,
or model support failures explicitly. Keep this diagnostic outside the
primary ranking, baseline fit, and pass decision. The source uses tensor
descriptors for both operands and the output. [NVIDIA PTX ISA tensor copy target notes](https://docs.nvidia.com/cuda/archive/13.2.2/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor).

## Profile and predictors

- Record GB10 static limits from public documentation and device queries.
  Every measured profile field must come from isolated VirtualGPU
  microbenchmarks on the measured GB10. Record GB10 device ID, schema and
  suite revision, driver, OS, memory configuration, clock and power policy,
  raw run IDs, and profile digest. Use the measurement policy below for both
  calibration and labels. Keep missing probes and unsupported paths visible.
- VirtualGPU receives the pinned GB10 KernelIR and GB10 profile. Use draw 0
  plus 32 shared perturbations per config, seed 17, and the fixed 5% log-ratio
  band. Take the nominal prediction for the shortlist. For pair direction,
  require every draw to agree by at least `log(1.05)`.
- Run the independent roofline + wave baseline with the same GB10 profile
  and compiled resources. Use the `phase1-final` NeuSight-style feature
  builder and fit. Refit once at the Phase 1 freeze on the entire RTX 4070
  development split, using the original warm labels and clean-cold
  replacement labels. Record the source profile and training label digests.
  Archive the fit as compact UTF-8 JSON with sorted keys and record its
  SHA-256 in the preregistration revision. Reuse its center, scale, weights,
  and intercept verbatim for both GB10 cohorts. Evaluate target features
  and analytical floors using GB10 profile values. Keep the fit frozen
  through both workload-holdout evaluations.

## Measurement runbook

1. Reserve exclusive GPU time and use the local GPU lock for every GPU
   command. Use the same fixed system power mode for the calibration and
   label sessions. Suspend CPU memory-intensive work, other GPU programs,
   large writes or copies, and background data processing. Keep display
   activity fixed. Allocate device tensors, initialize and touch all pages,
   compile, and verify outputs against an FP32 reference before timing.
   Avoid concurrent host buffer access, page migration, and swap pressure
   during a timing block. Record the host environment and any interference.
2. Use supported graphics and memory clock locks when an operator can set
   and verify a sustainable known frequency. Pick the clock policy during
   independent microbenchmark calibration. Record requested and achieved
   clocks per domain and restore changed settings after the run. If a lock
   is unsupported or unavailable, use record-only for that domain in both
   calibration and label collection. Record domain availability, SM and
   memory clocks, temperature, power mode, and clock-event reasons. Stabilize
   with untimed work for at least 60 seconds. Where device temperature is
   exposed, require a range of at most 2 C over the last 30 seconds before
   collecting a session. Flag unavailable temperature telemetry and keep
   the same warm-up and power policy. Reacquire an entire affected block
   after a documented thermal-throttle event, clock-policy change, or host
   interference.
   Record any unavailable telemetry and retain rejected blocks with reasons.
   [NVIDIA clock and thermal telemetry](https://docs.nvidia.com/deploy/nvidia-smi/).
3. Collect three independent sessions with ten randomized complete blocks
   each and order seeds 101, 102, and 103. Each block covers every viable
   config in warm and cold conditions. Time one GEMM with CUDA events in a
   CUDA Graph. Warm preparation primes with one untimed launch of the same
   kernel and buffers. Cold preparation scans a disjoint, once-initialized
   buffer with L2-caching, L1-bypassing loads. Use at least 128 MiB and at
   least three times the queried L2 size. Cache preparation is outside the
   timing interval. Verify eviction with the separate read probe and retain
   the probe results. The timing excludes allocation, compilation, host
   transfers, and validation. Keep warm and cold scores separate.
4. Archive every attempted candidate, including compile failures, launch
   resource errors, incorrect results, missing predictions, and hash
   mismatches. Check v2 PTX hash, toolchain, launch grid, and compiled
   resources before accepting an AOT/JIT comparison. Record both TTGIR hash
   results, actual pipeline metadata, and raw timing/telemetry. Preserve
   measured eligible candidates in the shortlist oracle even when a
   predictor cannot score them. Repeat measurement only for prespecified
   protocol failures.

## Scoring and decision

Use [design.md section 8](../docs/design.md#8-validation). Score each case
and cache regime against its own measured candidate median. Use 1,000 paired
bootstrap resamples with seed 17 for 95% intervals on the log latency ratios.
Resample complete blocks within sessions. Count hardware-equivalent,
unresolved, and ordered pairs separately. Report the three methods' per-case Kendall tau-b,
top-5 hit and regret, coverage, and pair confusion. Report absolute error
as a diagnostic. The two baselines use the fixed 5% band on their nominal
pair estimates.

Aggregate cases with equal weight within each cache regime. Report cases
with undefined tau-b and all coverage denominators. Calculate top-5 regret
against the fastest eligible measured config. A missing shortlist cannot
satisfy the shortlist criterion.

For each cache regime, the core must meet all three thresholds:

- Median Kendall tau-b is at least 0.05 higher than each baseline.
- Mean top-5 regret is at most 1%.
- Mean top-5 regret is at most 0.5 percentage points above each baseline.

Report the decision for warm and cold individually. An overall pass requires
both regimes to pass on the primary transfer cohort. Report the secondary
workload-holdout cohort and the GB10 TMA diagnostic separately.
