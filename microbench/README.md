# VirtualGPU Microbenchmark Suite

CUDA C++ microbenchmarks that query static device limits and measure effective hardware parameters to generate a VirtualGPU profile (`schemas/gpu_profile.schema.json`).

## Build

Install `nvidia-cuda-nvcc` and `nvidia-cuda-runtime` inside the project virtual environment, then build:

```bash
.venv/bin/python microbench/build.py --arch sm_89
```

## Run (M1)

Always wrap GPU execution with the shared GPU lock:

```bash
flock /tmp/virtualgpu-gpu.lock .venv/bin/python microbench/collect_profile.py --output profiles/rtx4070.json
```

In WSL2, clock locking is unavailable from inside Linux; `collect_profile.py` records observed clocks via `nvidia-smi`, marks `clock_policy.locked = false`, and widens `rel_uncertainty` according to the unlocked-clock degradation rule.
