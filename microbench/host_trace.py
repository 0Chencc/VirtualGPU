"""Sample host CPU use once a second beside a GB10 capture, and judge it afterwards.

GB10 shares DRAM between CPU and GPU, so CPU work slows DRAM-bound measurements
without spreading their samples. Until the shared gate in capture_posthoc.py is
merged, this sidecar records, every second, the busy share of all CPUs and each
process's CPU share (from /proc deltas), and `judge` applies the rule: void if
the total busy share exceeds 10% in any sample, or if any process outside the
capture exceeds 5% of one CPU in two or more consecutive samples.

Usage:
    python microbench/host_trace.py record trace.jsonl      # until killed
    python microbench/host_trace.py judge trace.jsonl START_UTC END_UTC
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

TOTAL_BUSY_LIMIT = 0.10
PROCESS_LIMIT = 0.05
CAPTURE_NAMES = ("vgpu_bench", "nvidia-smi", "flock", "sleep", "ps")
CAPTURE_SCRIPTS = ("capture_posthoc.py", "probe_blocks.py", "host_trace.py", "repeat_mlp.py")
HZ = os.sysconf("SC_CLK_TCK")


def cpu_totals():
    fields = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    idle = fields[3] + fields[4]
    return sum(fields) - idle, sum(fields)


def process_ticks():
    ticks = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text()
        except OSError:
            continue
        name = raw[raw.index("(") + 1 : raw.rindex(")")]
        if name.startswith("python"):
            try:
                args = (entry / "cmdline").read_bytes().split(b"\0")
                script = next((a.decode() for a in args[1:] if a.endswith(b".py")), "")
                name = f"{name}:{Path(script).name}" if script else name
            except OSError:
                pass
        rest = raw[raw.rindex(")") + 2 :].split()
        ticks[int(entry.name)] = (name, int(rest[11]) + int(rest[12]))
    return ticks


def record(path):
    busy0, total0 = cpu_totals()
    procs0, t0 = process_ticks(), time.monotonic()
    with open(path, "a") as out:
        while True:
            time.sleep(1.0)
            busy1, total1 = cpu_totals()
            procs1, t1 = process_ticks(), time.monotonic()
            elapsed = t1 - t0
            shares = {}
            for pid, (name, ticks) in procs1.items():
                before = procs0.get(pid)
                if before and before[0] == name:
                    share = (ticks - before[1]) / HZ / elapsed
                    if share > 0.01:
                        shares[f"{pid}:{name}"] = round(share, 3)
            row = {
                "utc": datetime.now(timezone.utc).isoformat(),
                "busy_share": (busy1 - busy0) / max(total1 - total0, 1),
                "processes": shares,
            }
            out.write(json.dumps(row) + "\n")
            out.flush()
            busy0, total0, procs0, t0 = busy1, total1, procs1, t1


def is_capture(name):
    return name.startswith(CAPTURE_NAMES) or name.endswith(CAPTURE_SCRIPTS)


def judge(path, start, end):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    rows = [r for r in rows if start <= r["utc"] <= end]
    if not rows:
        return {"status": "failed", "reason": "no samples in the window"}
    over_total = [r["utc"] for r in rows if r["busy_share"] > TOTAL_BUSY_LIMIT]
    runs, streak, offenders = {}, {}, {}
    for r in rows:
        current = {
            k
            for k, v in r["processes"].items()
            if v > PROCESS_LIMIT and not is_capture(k.split(":", 1)[1])
        }
        for k in current:
            streak[k] = streak.get(k, 0) + 1
            runs[k] = max(runs.get(k, 0), streak[k])
        for k in list(streak):
            if k not in current:
                streak[k] = 0
        for k in current:
            offenders.setdefault(k, r["processes"][k])
    sustained = {k: n for k, n in runs.items() if n >= 2}
    return {
        "status": "failed" if over_total or sustained else "passed",
        "samples": len(rows),
        "max_busy_share": max(r["busy_share"] for r in rows),
        "samples_over_total_limit": len(over_total),
        "sustained_other_processes": sustained,
        "limits": {"total_busy": TOTAL_BUSY_LIMIT, "process": PROCESS_LIMIT, "consecutive": 2},
    }


def main():
    if sys.argv[1] == "record":
        record(sys.argv[2])
    else:
        print(json.dumps(judge(sys.argv[2], sys.argv[3], sys.argv[4]), indent=1))


if __name__ == "__main__":
    main()
