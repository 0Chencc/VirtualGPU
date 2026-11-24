# Contributing

## Language

Everything in this repository is written in English: code, comments, documentation, commit messages, issues, and pull requests.

## Layout

| Path | Contents |
|---|---|
| `crates/vgpu-core/` | Rust model core: scheduler, phase model, L2 model, perturbation, PyO3 bindings |
| `python/virtualgpu/` | Python package: frontends (Triton first), user-facing API |
| `schemas/` | JSON Schemas for the Kernel IR and the GPU profile, with examples |
| `microbench/` | CUDA C++ microbenchmark suite that produces GPU profiles |
| `profiles/` | Measured GPU profiles |
| `validation/` | Hardware runner, baselines, scoring, experiment manifests |
| `docs/` | Design documents |

The schemas in `schemas/` are the contracts between components. Change them only together with every consumer.

## Principles

- No reverse engineering. Use public specifications, documented tool output, and our own measurements.
- Calibration data (microbenchmarks), development data, and holdout data stay separate. Never fit model parameters to GEMM timings.
- Keep it simple. Three similar lines are better than a premature abstraction.

## Commits

- One logical change per commit, with a short imperative message.
- Commit with explicit paths: `git commit -m "..." -- <paths>`.
- Run the tests for what you changed before committing, and do not commit if they fail.

## Using the GPU

The development GPU is shared. Wrap every command that runs on it in the lock, and keep runs short:

```
flock /tmp/virtualgpu-gpu.lock <command>
```
