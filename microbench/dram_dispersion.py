"""GB10 DRAM dispersion study: long DRAM launches with and without a controlled CPU memory load.

Runs `vgpu_bench --mode dram-dispersion` as one process per condition, cycling
quiet, 4-thread load, quiet, 16-thread load, quiet, three times. The load is
`cpu_stream` (microbench/src/cpu_stream.c): it touches its buffers first, and
the GPU process starts 5 s after it reports ready; it stops 5 s after the GPU
process ends. Records /proc/stat (busy CPUs), PSI, vmstat faults, load average,
the top CPU processes and GPU telemetry once per second. PSI memory counts
reclaim and fault stalls, not bandwidth, so busy CPUs and the load's own
bandwidth are the CPU-side measures. Run under the GPU lock.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from shutil import which

sys.path.insert(0, str(Path(__file__).resolve().parent))

from repeat_mlp import (
    command,
    host_snapshot,
    require_quiet,
    telemetry,
    utc,
    warm_device,
    write_json,
)  # noqa: E402

CONDITIONS = (("quiet", 0), ("load", 4), ("quiet", 0), ("load", 16), ("quiet", 0))
CYCLES = 3
SAMPLES = 30
WARMUP = 3
LOAD_LEAD_SECONDS = 5


def host_state():
    def read(path):
        try:
            return Path(path).read_text()
        except OSError:
            return ""

    vmstat = dict(line.split() for line in read("/proc/vmstat").splitlines() if line)
    cpu = [int(x) for x in read("/proc/stat").splitlines()[0].split()[1:]]
    return {
        "cpu_jiffies": {"busy": sum(cpu) - cpu[3] - cpu[4], "total": sum(cpu)},
        "utc": utc(),
        "psi": {k: read(f"/proc/pressure/{k}").strip() for k in ("cpu", "memory", "io")},
        "loadavg": read("/proc/loadavg").strip(),
        "pgfault": int(vmstat.get("pgfault", 0)),
        "pgmajfault": int(vmstat.get("pgmajfault", 0)),
        "top": command("ps", "-eo", "user,pcpu,rss,comm", "--sort=-pcpu").splitlines()[1:6],
    }


def run_condition(smi, binary, load_binary, threads):
    trace, done = [], threading.Event()

    def sample():
        while not done.is_set():
            trace.append({**host_state(), "gpu": telemetry(smi)})
            done.wait(1.0)

    sampler = threading.Thread(target=sample)
    sampler.start()
    load = None
    if threads:
        load = subprocess.Popen([str(load_binary), str(threads)], stdout=subprocess.PIPE, text=True)
        if load.stdout.readline().strip() != "ready":
            raise RuntimeError("cpu_stream did not report ready")
        time.sleep(LOAD_LEAD_SECONDS)
    started = utc()
    out = command(
        str(binary),
        "--mode",
        "dram-dispersion",
        "--device",
        "0",
        "--samples",
        str(SAMPLES),
        "--warmup",
        str(WARMUP),
    )
    finished = utc()
    load_report = None
    if load:
        time.sleep(LOAD_LEAD_SECONDS)
        load.send_signal(signal.SIGTERM)
        load_report = json.loads(load.communicate(timeout=60)[0].strip().splitlines()[-1])
    done.set()
    sampler.join()
    return {
        "load_threads": threads,
        "started_utc": started,
        "finished_utc": finished,
        "load_report": load_report,
        "host_trace": trace,
        "result": json.loads(out)["gmem_load_narrow"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("load_binary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--warmup-seconds", type=int, default=120)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    smi = which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    before = host_snapshot(smi)
    require_quiet(before)
    warm, stable = warm_device(
        smi,
        str(args.binary),
        args.warmup_seconds,
        timeout_report=args.output.with_suffix(".thermal-timeout.json"),
    )
    runs = []
    for cycle in range(CYCLES):
        for name, threads in CONDITIONS:
            runs.append(
                {
                    "cycle": cycle,
                    "condition": name,
                    **run_condition(smi, args.binary, args.load_binary, threads),
                }
            )
    write_json(
        args.output,
        {
            "samples": SAMPLES,
            "warmup": WARMUP,
            "host_before": before,
            "thermal_warmup": warm,
            "thermal_stability": stable,
            "runs": runs,
            "host_after": host_snapshot(smi),
        },
    )


if __name__ == "__main__":
    main()
