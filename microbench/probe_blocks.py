"""Run DRAM probe blocks: probe, workload, probe, each block under one short GPU lock.

Implements the capture of docs/results/analysis/v0.3-gb10-dram-probe-plan.md.
The thermal gate of repeat_mlp.py runs once under the lock before the first
block; between blocks the lock is released and the gap is spent outside it.

Usage:
    python microbench/probe_blocks.py --binary vgpu_bench_sm121 --revision <commit> \
        --blocks 20 --gap-seconds 30 --output session.json
"""

from __future__ import annotations

import argparse
import fcntl
import json
import platform
import time
from contextlib import contextmanager
from pathlib import Path
from shutil import which

from repeat_mlp import (
    command,
    digest,
    host_snapshot,
    require_quiet,
    telemetry,
    utc,
    warm_device,
    write_json,
)

LOCK = Path("/tmp/virtualgpu-gpu.lock")
VMSTAT_KEYS = (
    "pswpin",
    "pswpout",
    "pgscan_kswapd",
    "pgscan_direct",
    "pgsteal_kswapd",
    "pgsteal_direct",
)


@contextmanager
def gpu_lock():
    with LOCK.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def system_state(smi):
    cpu = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
    vmstat = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
    meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    pressure = (
        Path("/proc/pressure/memory").read_text()
        if Path("/proc/pressure/memory").exists()
        else None
    )
    return {
        "utc": utc(),
        "cpu_jiffies": [int(x) for x in cpu],
        "vmstat": {k: int(vmstat[k]) for k in VMSTAT_KEYS if k in vmstat},
        "mem_available_kb": int(meminfo["MemAvailable"].split()[0]),
        "pressure_memory": pressure,
        "gpu": telemetry(smi),
        "compute_apps": command(
            smi, "--query-compute-apps=pid,process_name", "--format=csv,noheader"
        ),
    }


def run(binary, mode, samples):
    return json.loads(
        command(
            str(binary), "--mode", mode, "--device", "0", "--samples", str(samples), "--warmup", "1"
        )
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--blocks", type=int, default=20)
    ap.add_argument("--gap-seconds", type=float, default=30)
    ap.add_argument("--samples", type=int, default=15)
    ap.add_argument("--workload", default="mlp-plateau")
    ap.add_argument("--warmup-seconds", type=float, default=60)
    args = ap.parse_args()
    smi = which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    before = host_snapshot(smi)
    require_quiet(before)
    with gpu_lock():
        warm, stable = warm_device(
            smi,
            str(args.binary),
            args.warmup_seconds,
            timeout_report=args.output.with_suffix(".thermal-timeout.json"),
        )
    blocks = []
    for index in range(args.blocks):
        if index:
            time.sleep(args.gap_seconds)
        with gpu_lock():
            state_before = system_state(smi)
            require_quiet({"compute_apps": state_before["compute_apps"]})
            probe_before = run(args.binary, "dram-probe", args.samples)
            workload = run(args.binary, args.workload, args.samples)
            probe_after = run(args.binary, "dram-probe", args.samples)
            state_after = system_state(smi)
        blocks.append(
            {
                "index": index,
                "state_before": state_before,
                "probe_before": probe_before,
                "workload": workload,
                "probe_after": probe_after,
                "state_after": state_after,
            }
        )
    after = host_snapshot(smi)
    write_json(
        args.output,
        {
            "post_hoc": True,
            "source_revision": args.revision,
            "binary_sha256": digest(args.binary),
            "host": platform.node(),
            "platform": platform.platform(),
            "gap_seconds": args.gap_seconds,
            "samples": args.samples,
            "workload_mode": args.workload,
            "before": before,
            "after": after,
            "thermal_warmup": warm,
            "thermal_stability": stable,
            "blocks": blocks,
        },
    )


if __name__ == "__main__":
    main()
