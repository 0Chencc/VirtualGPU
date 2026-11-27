"""CPU-only checks of the generic launch-schedule freeze and verification."""

import unittest

from microbench import long_launch


def point(ctas, steps, ns, trials=None):
    row = {
        "issue_mode": "cp_async",
        "tokens": 2,
        "active_ctas": ctas,
        "stage_bytes": 4096,
        "compute_ns": 0,
        "steps": steps,
        "launch_ns_samples": ns,
        "bw_gbps_samples": [1.0] * len(ns),
        "turnaround_ns_samples": [1.0] * len(ns),
    }
    if trials is not None:
        row["sizing_trials"] = trials
    return row


def output(*rows_per_field, samples=15):
    return {
        "samples": samples,
        "dram_closed_loop": list(rows_per_field[0]),
        "l2_closed_loop": list(rows_per_field[1]),
    }


class LongLaunchTests(unittest.TestCase):
    def pilot(self):
        trials = [{"steps": 256, "launch_ns_samples": [300e3] * 3}]
        return output(
            [point(16, 1100, [1.3e6] * 3, trials)],
            [point(32, 4000, [1.26e6] * 3, trials)],
            samples=3,
        )

    def test_freeze_writes_one_length_per_point_in_order(self):
        schedule, record = long_launch.freeze("closed-loop", self.pilot())
        self.assertEqual(schedule, "1100\n4000\n")
        self.assertEqual(
            [p["point"]["field"] for p in record["points"]], ["dram_closed_loop", "l2_closed_loop"]
        )
        self.assertNotIn("sizing_trials", record["points"][0]["point"])

    def test_freeze_rejects_a_short_pilot(self):
        pilot = output([point(16, 1024, [1.0e6] * 3)], [point(32, 4000, [1.3e6] * 3)])
        with self.assertRaisesRegex(ValueError, "pilot reached only"):
            long_launch.freeze("closed-loop", pilot)

    def test_verify_checks_points_lengths_and_launches(self):
        _, record = long_launch.freeze("closed-loop", self.pilot())
        good = output([point(16, 1100, [1.2e6] * 15)], [point(32, 4000, [1.1e6] * 15)])
        self.assertEqual(long_launch.verify(record, good)["shortest_launch_ns"], 1.1e6)
        for bad, message in (
            (output([point(16, 1000, [1.2e6] * 15)], [point(32, 4000, [1.1e6] * 15)]), "froze"),
            (
                output([point(16, 1100, [0.9e6] * 15)], [point(32, 4000, [1.1e6] * 15)]),
                "a launch took",
            ),
            (output([point(64, 1100, [1.2e6] * 15)], [point(32, 4000, [1.1e6] * 15)]), "differs"),
            (output([point(16, 1100, [1.2e6] * 15)], []), "points captured"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                long_launch.verify(record, bad)

    def test_verify_rejects_incomplete_or_invalid_samples(self):
        _, record = long_launch.freeze("closed-loop", self.pilot())
        for samples in ([1.2e6] * 14, [1.2e6] * 14 + [float("nan")], [1.2e6] * 14 + [0.0]):
            bad = output([point(16, 1100, samples)], [point(32, 4000, [1.1e6] * 15)])
            with self.assertRaisesRegex(ValueError, "CUDA event samples"):
                long_launch.verify(record, bad)

    def test_identity_ignores_diagnostics_but_checks_them(self):
        _, record = long_launch.freeze("closed-loop", self.pilot())
        row = point(16, 1100, [1.2e6] * 15)
        row["observed_sms"], row["target_sms"] = 16, 16
        good = output([row], [point(32, 4000, [1.1e6] * 15)])
        long_launch.verify(record, good)
        row["observed_sms"] = 15
        with self.assertRaisesRegex(ValueError, "SMs ran"):
            long_launch.verify(record, good)

    def test_schedule_file_must_match_the_freeze(self):
        schedule, record = long_launch.freeze("closed-loop", self.pilot())
        long_launch.check_schedule(record, schedule.encode())
        with self.assertRaisesRegex(ValueError, "differs from the freeze"):
            long_launch.check_schedule(record, b"1100\n4001\n")

    def test_unregistered_mode_is_refused(self):
        with self.assertRaisesRegex(ValueError, "no design fields"):
            long_launch.freeze("l2-merge", self.pilot())
