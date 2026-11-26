"""CPU-only tests for the host CPU quiet check of GB10 captures."""

import os
import unittest
from unittest.mock import patch

from microbench import host_quiet


def series(foreign_per_second, busy_share=0.03, seconds=4, parent=1):
    """One snapshot per second; pid 50 uses `foreign_per_second` jiffies per second."""
    cpus = os.cpu_count()
    out = []
    for i in range(seconds):
        out.append(
            {
                "total": 100 * cpus * i,
                "busy": round(busy_share * 100 * cpus * i),
                "procs": {
                    10: ("capture", 0, 1),
                    11: ("bench", 0, 10),
                    50: ("cc1", foreign_per_second * i, parent),
                },
            }
        )
    return out


class HostQuietTests(unittest.TestCase):
    def test_quiet_host_passes(self):
        self.assertEqual(host_quiet.check(series(0), [10])["status"], "passed")

    def test_sustained_foreign_process_fails(self):
        result = host_quiet.check(series(60), [10])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["offenders"]["50"]["comm"], "cc1")

    def test_busy_share_above_the_limit_fails(self):
        self.assertEqual(host_quiet.check(series(0, busy_share=0.2), [10])["status"], "failed")

    def test_descendants_of_the_capture_are_its_own(self):
        self.assertEqual(host_quiet.check(series(60, parent=11), [10])["status"], "passed")


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class WaitForQuietHostTests(unittest.TestCase):
    def test_returns_once_the_window_is_quiet(self):
        samples = iter(series(0, seconds=20))
        with patch.object(host_quiet, "sample", lambda: next(samples)):
            gate = host_quiet.wait_for_quiet_host([10], seconds=5, clock=FakeClock())
        self.assertEqual(gate["check"]["status"], "passed")
        self.assertEqual(gate["waited_seconds"], 5.0)

    def test_times_out_while_a_foreign_process_runs(self):
        samples = iter(series(60, seconds=100))
        with patch.object(host_quiet, "sample", lambda: next(samples)):
            with self.assertRaises(RuntimeError):
                host_quiet.wait_for_quiet_host([10], seconds=5, timeout=30, clock=FakeClock())
