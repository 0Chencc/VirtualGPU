# Post-hoc: dynamic local loads on the RTX 3070

**Post-hoc.** A first look at how the core's static spill figure compares with the local loads a kernel really executes. It uses the RTX 3070, which is not a reference GPU, and the causal case from the development attention manifest. It fits nothing and changes no model. The RTX 4070 capture is still needed before a core change (see `ROADMAP.md`).

## Data and method

- Capture: [spill-counters-rtx3070-posthoc.json](spill-counters-rtx3070-posthoc.json), from `python -m microbench.spill_counters` at commit `1b9b0e9`. RTX 3070 (`sm_86`), driver 610.88, Nsight Compute 2026.4.0, nvcc 13.2, Triton 3.9.0, WSL2, clocks and caches not controlled.
- Method: [profiles.md](../../components/profiles.md#post-hoc-spill-counter-capture). Counts are per thread per iteration, from predicated-on LDL instructions.
- Static figures: `scripts/posthoc_spill_split.py` on the capture, which recompiles each config for `sm_86`. Static is `spill_loads_bytes / 4`, and split is that over `spill_loop_count`. `sm_86` and `sm_89` give the same registers and spill bytes for all three configs.

## Spill microbenchmark

For all 60 points of `spill_reload_latency_by_count` (frames 128 B to 4 KiB, both dependency modes), the measured local loads per thread per iteration equal `reload_count` exactly, and the loop has no local stores. The microbenchmark does what its curves assume: every reload is one 4 B `LDL` and it runs on every iteration.

## Causal attention (`b2-q32-kv4-s768-d128-causal`)

| Config | Static | Static split | Measured, both loops | Main loop | Diagonal loop | Local loads outside both loops |
|---|---|---|---|---|---|---|
| `m64n32-w4-s2` | 0 | 0 | 0 | 0 | 0 | 0% |
| `m128n64-w4-s2` | 81.0 | 40.5 | 29.9 | 32.4 | 23.5 | 21% |
| `m128n128-w8-s2` | 7.0 | 3.5 | 0.86 | 0 | 3.0 | 57% |

Reloads per thread per iteration. The last column is the share of all local loads in the launch that fall outside the two loop helpers (prologue and epilogue).

## What it shows

1. The split moves the static figure toward the measured one, but it still overcounts: 1.4 times the pooled measurement for `m128n64-w4-s2` and 4.1 times for `m128n128-w8-s2`, against 2.7 and 8.1 times without the split.
2. The two loops do not hold the same spill code. In `m128n128-w8-s2` the main loop has no reloads and the diagonal loop has all of them; in `m128n64-w4-s2` the main loop reloads 38% more per iteration than the diagonal one. An even split is only right on average.
3. A large share of the static figure is not in either loop. Static spill bytes cover the prologue and epilogue too, which `spill_scope` charges as per iteration code.

None of this is from a reference GPU, and the SASS for `sm_86` can differ from `sm_89` even with the same spill bytes. It says what to check on the RTX 4070, not what the core should do.
