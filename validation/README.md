# M1 GEMM validation

Run commands from the repository root. The scorer uses the installed
VirtualGPU Python package to identify the current hash protocol. The runner
uses Python 3.12, PyTorch 2.8 with CUDA 12.8, Triton 3.9,
`cuda-bindings`, `nvidia-ml-py`, and the installed VirtualGPU Python package.
Use a worktree-local `.venv` and reuse cached CUDA wheels where available.

## Run a development case

```sh
flock /tmp/virtualgpu-gpu.lock .venv/bin/python -m validation.runner \
  validation/suite.json decode-16 results/decode-16.json --blocks 15 \
  --config m32n64k32-w4-s1-g1 --config m64n64k32-w4-s2-g1 \
  --config m64n128k32-w4-s2-g1
```

Repeat in separate processes with unique output paths and order seeds. M1
accepts explicit development cases on the RTX 4070. The `m1-cube` case uses the
shape and compile knobs in `schemas/examples/gemm_sm89.json`. The hand-written
IR has illustrative resource metadata, and the runner records the actual
compiled resources. Device-dependent boundary rules in the suite reserve
development and holdout cases for a later sweep.

The runner checks every candidate against an FP32 reference before timing.
Each randomized block has one warm and one cold trial for every valid
candidate. Each CUDA Graph primes the same buffers or writes a disjoint buffer
twice the reported L2 capacity, then brackets one GEMM with CUDA events. Cache
preparation and host submission are outside that interval. A separate read
probe records the observed effect of cache preparation. Clock telemetry is
record-only and is sampled asynchronously. An unsupported telemetry query is
recorded as unavailable. Reports include every sample and its block, the
cache control, compiled artifact hashes, timing dispersion, and telemetry.
Candidate statuses distinguish compilation failures, launch failures, and
incorrect outputs. Only successful candidates contribute timing samples.
Each cache regime reports a bootstrap 95% median interval, its half-width
divided by the median, and each config's median gap to the fastest measured
candidate. The scorer also reports these statistics across input sessions,
resampling blocks within each session.

## Compare with a prediction

```sh
python3 -m validation.baseline schemas/examples/profile_example.json \
  results/decode-16.json > results/baseline.json
.venv/bin/python -m validation.score results/baseline.json results/decode-16.json \
  --profile-revision 0.1.0-example --cache cold --top-k 2 --unchecked
.venv/bin/python -m unittest validation.test_validation
```

The schema example is a substitute for a measured profile when checking the
interfaces. Use the measured profile and its revision for experiments. The
baseline uses a full-wave duration for a partial wave and the profile's DRAM
read bandwidth as one off-chip rate.

For the core comparison, pass its JSON array of prediction records in place of
`results/baseline.json`. Each record contains `case_id`, `config_id`,
`profile_revision`, `status`, and, for successful predictions,
`nominal_latency_ns`. The score report places the measured median beside the
nominal prediction. Provide the `draws` object keyed by draw ID, including
`"0"`, to enable pairwise scoring. Pass multiple raw files to score matched
blocks across sessions. Compare one cache regime at a time with a matching
initial-cache assumption.

For validated comparisons, supply `--aot CONFIG_ID=KERNEL_IR_JSON` once per
config in place of `--unchecked`. Generate the IR offline from the exact same
`dense_gemm` function, arguments, specializations, and compiler options used
by the runner. Pass explicit `call_args` containing three aligned FP16
`MockTensor` arguments followed by `M`, `N`, and `K` to the frontend
extractor. It takes the tile, traversal, warp, and stage knobs in `config`.
The scorer verifies the versioned PTX digest, toolchain, launch grid, and
resources. It reports TTGIR digest agreement as a diagnostic. Unverified or
mismatched samples remain in runner reports and are excluded from validated
scoring. The scorer requires the current hash version for both artifacts.
The exploratory flag skips the AOT comparison and still requires a
current-version JIT report.
