"""Measure development GEMM configurations on the RTX 4070."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import re
import statistics
from threading import Event, Thread
import time
from uuid import uuid4

from cuda.bindings import driver
import pynvml
import torch
import triton
import triton.language as tl
from triton.backends.nvidia.compiler import get_ptxas
from triton.runtime.errors import PTXASError
from virtualgpu.frontend.compiler import artifact_hashes, detect_ptx_target, get_ptxas_version

from validation.provenance import source_info
from validation.score import median_ci95


@triton.jit
def dense_gemm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BM)
    grid_n = tl.cdiv(N, BN)
    group = pid // (GROUP_M * grid_n)
    first_m = group * GROUP_M
    group_size = tl.minimum(grid_m - first_m, GROUP_M)
    local_id = pid % (GROUP_M * grid_n)
    rows = (first_m + local_id % group_size) * BM + tl.arange(0, BM)
    cols = (local_id // group_size) * BN + tl.arange(0, BN)
    offsets = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for step in range(tl.cdiv(K, BK)):
        inner = step * BK + offsets
        a = tl.load(A + rows[:, None] * K + inner[None, :],
                    (rows[:, None] < M) & (inner[None, :] < K), other=0)
        b = tl.load(B + inner[:, None] * N + cols[None, :],
                    (inner[:, None] < K) & (cols[None, :] < N), other=0)
        acc = tl.dot(a, b, acc)
    tl.store(C + rows[:, None] * N + cols[None, :], acc.to(tl.float16),
             (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def read_probe(Data, Out, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    indices = pid * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(Data + indices, cache_modifier=".cg")
    tl.store(Out + pid, tl.sum(values))


@triton.jit
def read_flush(Data, Sink, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    indices = pid * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(Data + indices, cache_modifier=".cg")
    total = tl.sum(values)
    # Keep the loads live without a store for the all-ones eviction buffer.
    if total == -1:
        tl.store(Sink + pid, total)


def make_read_flush(l2_bytes):
    block = 4096
    block_bytes = block * 4
    size = max(3 * l2_bytes, 128 * 1024**2)
    blocks = triton.cdiv(size, block_bytes)
    buffer = torch.ones(blocks * block, device="cuda", dtype=torch.int32)
    sink = torch.empty(blocks, device="cuda", dtype=torch.int32)

    def evict():
        read_flush[(blocks,)](buffer, sink, block)

    evict()
    return evict, blocks * block_bytes


def l2_capacity():
    status, = driver.cuInit(0)
    if status != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuInit failed: {status}")
    status, size = driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, 0,
    )
    if status != driver.CUresult.CUDA_SUCCESS or size <= 0:
        raise RuntimeError(f"Could not query L2 capacity: {status}")
    return size


def telemetry_value(function, *args):
    try:
        return function(*args)
    except pynvml.NVMLError as error:
        return {"unavailable": str(error)}


def record_telemetry(handle, output, stop):
    while True:
        output.append({
            "monotonic_ns": time.monotonic_ns(),
            "sm_mhz": telemetry_value(pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_SM),
            "memory_mhz": telemetry_value(pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_MEM),
            "temperature_c": telemetry_value(
                pynvml.nvmlDeviceGetTemperature, handle, pynvml.NVML_TEMPERATURE_GPU,
            ),
            "clock_event_reasons": telemetry_value(
                pynvml.nvmlDeviceGetCurrentClocksEventReasons, handle,
            ),
        })
        if stop.wait(0.02):
            return


def timed_graph(prepare, operation):
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepare()
        start.record()
        operation()
        end.record()

    def elapsed_ns():
        graph.replay()
        end.synchronize()
        return start.elapsed_time(end) * 1e6

    elapsed_ns()
    return elapsed_ns


def cache_control(evict, l2_bytes):
    blocks = max(1, l2_bytes // (8 * 1024))
    probe = torch.ones(blocks * 1024, dtype=torch.float32, device="cuda")
    output = torch.empty(blocks, dtype=torch.float32, device="cuda")

    def read():
        read_probe[(blocks,)](probe, output, 1024)

    read()
    warm = timed_graph(read, read)
    cold = timed_graph(evict, read)
    return {"warm": [warm() for _ in range(8)], "cold": [cold() for _ in range(8)]}


def prepare_candidates(case, configs, seed):
    m, n, k = case["shape"]
    torch.manual_seed(seed)
    a = torch.randn((m, k), device="cuda", dtype=torch.float16).mul_(0.25)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16).mul_(0.25)
    out = torch.empty((m, n), device="cuda", dtype=torch.float16)
    torch.backends.cuda.matmul.allow_tf32 = False
    reference = (a.float() @ b.float()).half()
    ptxas = Path(get_ptxas(89).path)
    assembler_version = get_ptxas_version(str(ptxas))
    operations = {}
    artifacts = {}

    for cfg in configs:
        key = cfg["config_id"]
        bm, bn, bk = cfg["tile"]
        grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
        kernel_args = (a, b, out, m, n, k, bm, bn, bk, cfg["group_m"])
        options = dict(num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])

        def operation(grid=grid, kernel_args=kernel_args, options=options):
            return dense_gemm[grid](*kernel_args, **options)

        try:
            compiled = dense_gemm.run(*kernel_args, grid=grid, warmup=True, **options)
        except (triton.CompilationError, PTXASError) as error:
            artifacts[key] = {"status": "compile_failed", "reason": str(error)}
            continue

        ptx = compiled.asm["ptx"]
        mma_shapes = re.findall(r"mma\.sync\.aligned\.m(\d+)n(\d+)k(\d+)\.row\.col\.f32\.f16\.f16\.f32", ptx)
        row = {
            "tile": cfg["tile"],
            "num_warps": cfg["num_warps"],
            "mma_shapes": sorted({f"{m}x{n}x{k}" for m, n, k in mma_shapes}),
            "grid": list(grid),
            "shared_bytes": compiled.metadata.shared,
            "ptx_raw_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
            **artifact_hashes(ptx, compiled.asm["ttgir"]),
            "toolchain": {
                "compiler": "triton",
                "compiler_version": triton.__version__,
                "assembler": ptxas.name,
                "assembler_version": assembler_version,
                "target_sm_compiled": detect_ptx_target(ptx, 89),
                "enable_fp_fusion": compiled.metadata.enable_fp_fusion,
                "ptx_options": compiled.metadata.ptx_options,
                "disable_ptxas_opt": triton.knobs.nvidia.disable_ptxas_opt,
            },
        }
        artifacts[key] = row
        try:
            operation()
        except (triton.OutOfResources, RuntimeError) as error:
            row.update(status="launch_failed", reason=str(error))
            continue
        # An asynchronous execution error can invalidate the CUDA context.
        torch.cuda.synchronize()
        row.update(registers_per_thread=compiled.n_regs, spill_count=compiled.n_spills)
        try:
            torch.testing.assert_close(out, reference, rtol=0.01, atol=0.015)
        except AssertionError as error:
            row.update(status="incorrect", reason=str(error))
            continue
        row["status"] = "ok"
        operations[key] = operation
    return operations, artifacts


def measure(case, configs, input_seed, args, manifest_sha):
    source = source_info()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("The M1 runner requires exactly one visible CUDA device")
    gpu = torch.cuda.get_device_properties(0)
    if gpu.name != "NVIDIA GeForce RTX 4070" or (gpu.major, gpu.minor) != (8, 9):
        raise RuntimeError(f"Only the RTX 4070 is authorized in M1: {gpu.name}")
    l2_bytes = l2_capacity()
    operations, artifacts = prepare_candidates(case, configs, input_seed)
    caches = ("warm", "cold") if args.cache == "both" else (args.cache,)
    eviction_bytes = 0
    evict = None
    if "cold" in caches and operations:
        evict, eviction_bytes = make_read_flush(l2_bytes)
    control = cache_control(evict, l2_bytes) if evict else {}
    trials = {(cache, key): timed_graph(evict if cache == "cold" else operation, operation)
              for key, operation in operations.items() for cache in caches}

    rng = random.Random(args.seed)
    samples, telemetry = [], []
    order = list(trials)
    pynvml.nvmlInit()
    try:
        uuid = str(gpu.uuid)
        handle = pynvml.nvmlDeviceGetHandleByUUID(uuid if uuid.startswith("GPU-") else "GPU-" + uuid)
        driver_version = pynvml.nvmlSystemGetDriverVersion()
        stop = Event()
        monitor = Thread(target=record_telemetry, args=(handle, telemetry, stop))
        monitor.start()
        try:
            deadline = time.monotonic() + args.warmup_seconds
            while operations and time.monotonic() < deadline:
                for operation in operations.values():
                    operation()
                torch.cuda.synchronize()
            for block in range(args.blocks):
                rng.shuffle(order)
                for cache, key in order:
                    latency = trials[cache, key]()
                    if not math.isfinite(latency) or latency <= 0:
                        raise RuntimeError("Invalid CUDA event duration")
                    samples.append({
                        "block": block, "cache": cache, "config_id": key,
                        "latency_ns": latency, "monotonic_ns": time.monotonic_ns(),
                    })
        finally:
            stop.set()
            monitor.join()
    finally:
        pynvml.nvmlShutdown()

    summary = {}
    for cache, key in sorted(order):
        times = [row["latency_ns"] for row in samples
                 if row["cache"] == cache and row["config_id"] == key]
        median = statistics.median(times)
        low, high = median_ci95([times], seed=args.seed)
        stdev = statistics.stdev(times)
        summary.setdefault(cache, {})[key] = {
            "median_ns": median,
            "median_ci95_ns": [low, high],
            "relative_ci95_half_width": (high - low) / (2 * median),
            "sample_stdev_ns": stdev,
            "cv": stdev / statistics.mean(times),
        }
    for entries in summary.values():
        best = min(entry["median_ns"] for entry in entries.values())
        for entry in entries.values():
            entry["gap_to_measured_best"] = entry["median_ns"] / best - 1
    return {
        "session_id": str(uuid4()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": case["case_id"],
        "split": case["split"],
        "shape": case["shape"],
        "gpu": {"name": gpu.name, "uuid": uuid, "arch": "sm_89", "driver": driver_version},
        "software": {"os": platform.release(), "torch": torch.__version__,
                     "triton": triton.__version__, "cuda": torch.version.cuda},
        "kernel_sha256": hashlib.sha256(dense_gemm.src.encode()).hexdigest(),
        "manifest_sha256": manifest_sha,
        "source": source,
        "protocol": {"clock_policy": "record_only", "timer": "cuda_graph_single_launch",
                     "l2_bytes": l2_bytes, "cache_states": list(caches),
                     "cold_flush": "read_cg" if "cold" in caches else None,
                     "eviction_bytes": eviction_bytes, "blocks": args.blocks,
                     "order_seed": args.seed, "thermal_warmup_seconds": args.warmup_seconds},
        "cache_control_ns": control,
        "configs": artifacts,
        "samples": samples,
        "telemetry": telemetry,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("case_id")
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", action="append")
    parser.add_argument("--cache", choices=("both", "warm", "cold"), default="both")
    parser.add_argument("--blocks", type=int, default=15)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--warmup-seconds", type=float, default=2)
    args = parser.parse_args()
    if args.blocks < 2 or args.warmup_seconds < 0:
        parser.error("Require at least two blocks and nonnegative warmup")
    raw = args.manifest.read_bytes()
    data = json.loads(raw)
    case = next((entry for entry in data["cases"] if entry["case_id"] == args.case_id), None)
    if case is None or case["split"] != "development":
        parser.error("Only explicit development cases can run in M1")
    keys = set(args.config or case["config_ids"])
    if not keys <= set(case["config_ids"]):
        parser.error("Requested config is outside the case")
    configs = [cfg for cfg in data["configs"] if cfg["config_id"] in keys]
    report = measure(case, configs, data["input_seed"], args, hashlib.sha256(raw).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
