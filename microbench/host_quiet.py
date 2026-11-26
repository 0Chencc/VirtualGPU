"""Host CPU quiet check for captures on GPUs that share memory with the CPU (GB10).

On GB10, CPU memory traffic slows GPU DRAM time (docs/results/analysis/
gb10-dram-cpu-contention-posthoc.md), and the GPU lock does not stop other
sessions' CPU work. Sample the host while a capture runs, then judge:

- the busy share of all CPUs, from /proc/stat, stays at or below BUSY_LIMIT;
- no process outside the capture uses more than PROCESS_LIMIT of one CPU in
  two consecutive samples (per-process CPU from /proc/<pid>/stat deltas, not
  `ps` %CPU, which is a lifetime average).

PSI memory and the load average cannot replace this: PSI memory stays at 0
under full copy bandwidth and the load average lags by tens of seconds.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

BUSY_LIMIT = 0.10
PROCESS_LIMIT = 0.05
QUIET_SECONDS = 10
QUIET_TIMEOUT_SECONDS = 20 * 60


def sample():
    """One snapshot: total and busy jiffies, and (comm, jiffies, parent) per process."""
    fields = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    procs = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text()
        except OSError:
            continue
        comm = raw[raw.index("(") + 1 : raw.rindex(")")]
        rest = raw[raw.rindex(")") + 2 :].split()
        procs[int(entry.name)] = (comm, int(rest[11]) + int(rest[12]), int(rest[1]))
    return {"total": sum(fields), "busy": sum(fields) - fields[3] - fields[4], "procs": procs}


def own_pids(snapshot, roots):
    """The roots and every descendant in `snapshot`."""
    children = {}
    for pid, (_, _, parent) in snapshot["procs"].items():
        children.setdefault(parent, []).append(pid)
    own, stack = set(), list(roots)
    while stack:
        pid = stack.pop()
        if pid not in own:
            own.add(pid)
            stack.extend(children.get(pid, []))
    return own


def check(samples, roots):
    """Judge a list of `sample()` snapshots taken about once per second during a capture.

    `roots` are the capture's own process IDs (the runner and the benchmark); their
    descendants count as the capture too.
    """
    ncpu = os.cpu_count()
    busy, offenders, previous = [], {}, set()
    for a, b in zip(samples, samples[1:]):
        dt = b["total"] - a["total"]
        if dt <= 0:
            continue
        busy.append((b["busy"] - a["busy"]) / dt)
        own = own_pids(b, roots)
        heavy = set()
        for pid, (comm, jiffies, _) in b["procs"].items():
            if pid in own or pid not in a["procs"]:
                continue
            share = (jiffies - a["procs"][pid][1]) / dt * ncpu
            if share > PROCESS_LIMIT:
                heavy.add(pid)
                if pid in previous:
                    offenders[pid] = (comm, max(offenders.get(pid, (comm, 0.0))[1], share))
        previous = heavy
    max_busy = max(busy) if busy else None
    failed = not busy or max_busy > BUSY_LIMIT or bool(offenders)
    return {
        "status": "failed" if failed else "passed",
        "samples": len(busy),
        "max_busy_share": max_busy,
        "busy_limit": BUSY_LIMIT,
        "process_limit": PROCESS_LIMIT,
        "offenders": {
            str(pid): {"comm": comm, "max_cpu_share": share}
            for pid, (comm, share) in offenders.items()
        },
    }


def wait_for_quiet_host(roots, seconds=QUIET_SECONDS, timeout=QUIET_TIMEOUT_SECONDS, clock=time):
    """Block until the last `seconds` one-second samples pass `check`; return that window."""
    start = clock.monotonic()
    window = [sample()]
    while True:
        clock.sleep(1.0)
        window = (window + [sample()])[-(seconds + 1) :]
        if len(window) > seconds and check(window, roots)["status"] == "passed":
            return {
                "seconds": seconds,
                "waited_seconds": clock.monotonic() - start,
                "check": check(window, roots),
            }
        if clock.monotonic() - start > timeout:
            raise RuntimeError(f"The host did not stay quiet for {seconds} s within {timeout} s")
