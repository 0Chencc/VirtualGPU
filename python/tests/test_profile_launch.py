"""CPU-only checks of attaching long-launch records to a profile."""

import copy
import json
import unittest
from pathlib import Path

from microbench import profile_launch as pl

ROOT = Path(__file__).resolve().parents[2]


def rounds_around(value):
    return [value * f for f in (0.98, 1.0, 1.01, 0.99, 1.02)]


def launch(steps=1000):
    return {
        "length": steps,
        "min_ns": 1.2e6,
        "median_ns": 1.3e6,
        "sm_mhz_before": [2500.0] * 5,
        "sm_mhz_after": [2520.0] * 5,
    }


class ProfileLaunchTests(unittest.TestCase):
    def setUp(self):
        self.profile = json.loads((ROOT / "profiles/gb10.json").read_text())

    def test_closed_loop_values_get_launch_and_round_range(self):
        entry = self.profile["measured"]["memory"]["dram_closed_loop"][3]
        point = {
            k: entry[k]
            for k in ("issue_mode", "tokens", "active_ctas", "stage_bytes", "compute_ns")
        }
        bw, turn = entry["delivered_bw_gbps"]["value"], entry["mean_stage_turnaround_ns"]["value"]
        point |= {
            "steps": 1000,
            "launch_ns_samples": [1.3e6] * 5,
            "bw_gbps_samples": rounds_around(bw),
            "turnaround_ns_samples": rounds_around(turn),
            "launch": launch(),
        }
        pl.attach(self.profile, "closed-loop", {"samples": 5, "dram_closed_loop": [point]})
        value = entry["delivered_bw_gbps"]
        self.assertEqual(value["launch"]["length"], 1000)
        self.assertEqual(value["round_median_range"], [bw * 0.98, bw * 1.02])
        self.assertNotIn("per_cycle", value)
        self.assertIn("launch", entry["mean_stage_turnaround_ns"])

    def test_compute_rates_get_per_cycle_values(self):
        table = self.profile["measured"]["compute"]["mma_tflops_per_sm_by_warps"]
        op = next(iter(table))
        entry = table[op][0]
        rate = entry["tflops_per_sm"]["value"]
        point = {
            "mma_warps_per_sm": entry["mma_warps_per_sm"],
            "steps": 9000,
            "launch_ns_samples": [1.3e6] * 5,
            "tflops_per_sm_samples": [rate] * 5,
            "launch": launch(9000),
        }
        fp32 = self.profile["measured"]["compute"]["fp32_fma_tflops"]["value"]
        merged = {
            "mma": {op: [point]},
            "fp32_fma": {
                "warps_per_sm": 16,
                "steps": 9e4,
                "launch_ns_samples": [1.3e6] * 5,
                "launch": launch(),
            },
            "fp32_fma_tflops_samples": [fp32] * 5,
        }
        pl.attach(self.profile, "mma", merged)
        self.assertAlmostEqual(entry["tflops_per_sm"]["per_cycle"]["value"], rate / 2510.0)
        self.assertEqual(entry["tflops_per_sm"]["per_cycle"]["unit"], "TFLOP/s per MHz")
        self.assertAlmostEqual(
            self.profile["measured"]["compute"]["fp32_fma_tflops"]["per_cycle"]["value"],
            fp32 / 2510.0,
        )

    def test_write_points_are_scalars_and_capacity_sweep_has_no_value(self):
        write = self.profile["measured"]["memory"]["dram_bw_write_gbps"]["value"]
        point = {
            "target_sms": 48,
            "steps": 256,
            "launch_ns_samples": [1.3e6] * 5,
            "gbps_samples": [write] * 5,
            "launch": launch(256),
        }
        pl.attach(self.profile, "dram-bw", {"write_full": [point]})
        self.assertEqual(
            self.profile["measured"]["memory"]["dram_bw_write_gbps"]["launch"]["length"], 256
        )
        sweep = {
            "working_set_bytes": 1 << 24,
            "steps": 9,
            "launch_ns_samples": [1.3e6] * 5,
            "launch": launch(9),
        }
        pl.attach(self.profile, "l2", {"capacity_sweep": [sweep]})

    def test_exception_flags_the_entry_and_covers_the_rounds(self):
        entry = self.profile["measured"]["memory"]["dram_bw_by_active_sms"][0]
        bw = entry["bw_gbps"]["value"]
        point = {
            "target_sms": entry["active_sms"],
            "steps": 81,
            "launch_ns_samples": [1.3e6] * 5,
            "gbps_samples": [bw * 0.7, bw, bw, bw * 1.01, bw],
            "launch": launch(81),
            "exceptions": ["low_in_flight_unstable"],
        }
        pl.attach(self.profile, "dram-bw", {"read_curve": [point]})
        self.assertTrue(entry["low_in_flight_unstable"])
        self.assertAlmostEqual(entry["bw_gbps"]["rel_uncertainty"], 0.3, places=3)

    def test_mismatches_are_errors(self):
        entry = self.profile["measured"]["memory"]["dram_bw_by_active_sms"][0]
        point = {
            "target_sms": 999,
            "steps": 81,
            "launch_ns_samples": [1.3e6] * 5,
            "gbps_samples": [1.0] * 5,
        }
        with self.assertRaisesRegex(ValueError, "0 profile entries"):
            pl.attach(copy.deepcopy(self.profile), "dram-bw", {"read_curve": [point]})
        point["target_sms"] = entry["active_sms"]
        point["launch"] = launch(81)
        with self.assertRaisesRegex(ValueError, "not the median"):
            pl.attach(self.profile, "dram-bw", {"read_curve": [point]})
