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
    def test_l2_capacity_freezes_passes_and_working_set(self):
        row = {"working_set_bytes": 24 * 1024**2, "steps": 32, "launch_ns_samples": [1.3e6] * 3}
        raw = {"samples": 3, "capacity_sweep": [row]}
        schedule, freeze = long_launch.freeze("l2", raw)
        self.assertEqual(schedule, "32\n")
        long_launch.verify(freeze, raw)
        row["steps"] = 31
        with self.assertRaisesRegex(ValueError, "froze"):
            long_launch.verify(freeze, raw)

    def test_l2_write_legacy_samples_do_not_duplicate_the_scheduled_point(self):
        bandwidth = [100.0] * 3
        raw = {
            "samples": 3,
            "write_full": {
                "working_set_bytes": 6 * 1024**2,
                "steps": 128,
                "launch_ns_samples": [1.3e6] * 3,
                "gbps_samples": bandwidth,
            },
            "write_full_gbps_samples": bandwidth,
        }
        schedule, freeze = long_launch.freeze("l2", raw)
        self.assertEqual(schedule, "128\n")
        long_launch.verify(freeze, raw)
        raw["write_full"]["working_set_bytes"] *= 2
        with self.assertRaisesRegex(ValueError, "differs from the freeze"):
            long_launch.verify(freeze, raw)

    def test_l2_read_instructions_are_separate_frozen_curves(self):
        row = {"target_sms": 48, "steps": 2048, "launch_ns_samples": [1.3e6] * 3}
        raw = {"samples": 3, "read_curve": [row], "read_curve_cp_async": [dict(row)]}
        _, freeze = long_launch.freeze("l2", raw)
        self.assertEqual(
            [entry["point"] for entry in freeze["points"]],
            [
                {"field": "read_curve", "target_sms": 48},
                {"field": "read_curve_cp_async", "target_sms": 48},
            ],
        )
        long_launch.verify(freeze, raw)
        raw["read_curve_cp_async"][0]["target_sms"] = 24
        with self.assertRaisesRegex(ValueError, "differs from the freeze"):
            long_launch.verify(freeze, raw)

    def test_dram_write_has_its_own_frozen_point(self):
        row = {"target_sms": 48, "steps": 384, "launch_ns_samples": [1.3e6] * 3}
        raw = {"samples": 3, "read_curve": [row], "write_full": [dict(row, steps=512)]}
        schedule, freeze = long_launch.freeze("dram-bw", raw)
        self.assertEqual(schedule, "384\n512\n")
        self.assertEqual(freeze["points"][1]["point"], {"field": "write_full", "target_sms": 48})
        long_launch.verify(freeze, raw)
        with self.assertRaisesRegex(ValueError, "points captured"):
            long_launch.verify(freeze, {"samples": 3, "read_curve": [row]})

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

    def test_verify_checks_in_kernel_windows(self):
        _, record = long_launch.freeze("closed-loop", self.pilot())
        row = point(16, 1100, [1.2e6] * 15)
        row["window_ns_samples"] = [1.19e6] * 15
        good = output([row], [point(32, 4000, [1.1e6] * 15)])
        long_launch.verify(record, good)
        row["window_ns_samples"][3] = 0.99e6
        with self.assertRaisesRegex(ValueError, "in-kernel window"):
            long_launch.verify(record, good)

    def test_verify_rejects_incomplete_or_invalid_samples(self):
        _, record = long_launch.freeze("closed-loop", self.pilot())
        for samples in ([1.2e6] * 14, [1.2e6] * 14 + [float("nan")], [1.2e6] * 14 + [0.0]):
            bad = output([point(16, 1100, samples)], [point(32, 4000, [1.1e6] * 15)])
            with self.assertRaisesRegex(ValueError, "CUDA event samples"):
                long_launch.verify(record, bad)

    def test_freeze_rejects_an_empty_sample_list(self):
        pilot = output([point(16, 1100, [])], [point(32, 4000, [1.3e6] * 3)], samples=3)
        with self.assertRaisesRegex(ValueError, "no CUDA event samples"):
            long_launch.freeze("closed-loop", pilot)

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
            long_launch.freeze("sync-loop", self.pilot())

    def test_mma_tables_and_fp32_fma_are_keyed_by_their_own_fields(self):
        def timed(steps):
            return {"steps": steps, "launch_ns_samples": [1.3e6] * 3}

        pilot = {
            "samples": 3,
            "mma": {
                "f16": [{"mma_warps_per_sm": 1, **timed(9000)}],
                "bf16": [{"mma_warps_per_sm": 1, **timed(9100)}],
            },
            "fp32_fma": {"warps_per_sm": 16, **timed(52000)},
            "fp32_fma_tflops_samples": [30.0] * 3,
        }
        schedule, record = long_launch.freeze("mma", pilot)
        self.assertEqual(schedule, "9000\n9100\n52000\n")
        self.assertEqual(
            [p["point"] for p in record["points"]],
            [
                {"field": "mma/f16", "mma_warps_per_sm": 1},
                {"field": "mma/bf16", "mma_warps_per_sm": 1},
                {"field": "fp32_fma", "warps_per_sm": 16},
            ],
        )

    def test_gmem_issue_points_are_keyed_by_grid_width_and_warps(self):
        row = {
            "active_sms": 1,
            "load_width_bytes": 4,
            "num_warps": 8,
            "steps": 9000,
            "launch_ns_samples": [1.3e6] * 3,
        }
        _, record = long_launch.freeze(
            "gmem-issue", {"samples": 3, "gmem_load_issue_per_sm": [row]}
        )
        self.assertEqual(
            record["points"][0]["point"],
            {
                "field": "gmem_load_issue_per_sm",
                "active_sms": 1,
                "load_width_bytes": 4,
                "num_warps": 8,
            },
        )

    def test_read_sweeps_bind_their_work(self):
        def row(steps, transfer):
            return {
                "issue_mode": "cp_async",
                "bytes_in_flight": 4096,
                "active_ctas": 1,
                "bytes_per_cta": 4096,
                "steps": steps,
                "transfer_bytes": transfer,
                "source_buffer_bytes": 1 << 30,
                "launch_ns_samples": [1.3e6] * 3,
            }

        _, record = long_launch.freeze(
            "mlp", {"samples": 3, "dram_bw_by_bytes_in_flight": [row(4096, 1 << 24)]}
        )
        self.assertEqual(record["points"][0]["transfer_bytes"], 1 << 24)
        capture = {"samples": 3, "dram_bw_by_bytes_in_flight": [row(4096, 1 << 25)]}
        with self.assertRaisesRegex(ValueError, "work differs"):
            long_launch.verify(record, capture)

    def test_unregistered_field_is_refused(self):
        pilot = {"samples": 3, "other": [{"steps": 1, "launch_ns_samples": [1.3e6] * 3}]}
        with self.assertRaisesRegex(ValueError, "other: no design fields"):
            long_launch.freeze("closed-loop", pilot)

    def test_l2_merge_points_are_keyed_by_sharing_and_stagger(self):
        def merge(stagger, steps):
            return {
                "active_ctas": 48,
                "sharing": 4,
                "bytes_per_cta": 8192,
                "stagger_stages": stagger,
                "steps": steps,
                "launch_ns_samples": [1.3e6] * 3,
                "turnaround_ns_samples": [900.0] * 3,
            }

        pilot = {"samples": 3, "l2_merge_by_sharing": [merge(0, 2000), merge(16, 2100)]}
        schedule, record = long_launch.freeze("l2-merge", pilot)
        self.assertEqual(schedule, "2000\n2100\n")
        self.assertEqual([p["point"]["stagger_stages"] for p in record["points"]], [0, 16])
        swapped = {"samples": 3, "l2_merge_by_sharing": [merge(16, 2000), merge(0, 2100)]}
        with self.assertRaisesRegex(ValueError, "differs from the freeze"):
            long_launch.verify(record, swapped)

    def test_sync_loop_multicta_is_keyed_by_level_mode_ctas_bytes_and_chain(self):
        def stage(level, cps, steps):
            return {
                "mem_level": level,
                "transfer_mode": "cp_async_s1",
                "ctas_per_sm": cps,
                "active_ctas": 48 * cps,
                "num_warps": 4,
                "stage_bytes": 4096,
                "mma_chain_len": 8,
                "solo_compute_ns": 120.0,
                "steps": steps,
                "launch_ns_samples": [1.3e6] * 3,
            }

        occupancy = {
            "transfer_mode": "cp_async_s1",
            "stage_bytes": 4096,
            "max_active_ctas_per_sm": 8,
        }
        pilot = {
            "samples": 3,
            "occupancy_check": [occupancy],
            "sync_stage_loop": [stage("dram", 1, 900), stage("l2", 6, 5000)],
        }
        schedule, record = long_launch.freeze("sync-loop-multicta", pilot)
        self.assertEqual(schedule, "900\n5000\n")
        self.assertEqual(
            record["points"][1]["point"],
            {
                "field": "sync_stage_loop",
                "mem_level": "l2",
                "transfer_mode": "cp_async_s1",
                "ctas_per_sm": 6,
                "num_warps": 4,
                "stage_bytes": 4096,
                "mma_chain_len": 8,
            },
        )

    def test_sync_loop_ablation_is_keyed_by_variant(self):
        def stage(variant, steps):
            return {
                "variant": variant,
                "ctas_per_sm": 1,
                "active_ctas": 48,
                "num_warps": 8,
                "stage_bytes": 16384,
                "mma_chain_len": 64,
                "steps": steps,
                "launch_ns_samples": [1.3e6] * 3,
            }

        pilot = {
            "samples": 3,
            "sync_stage_loop_ablation": [stage("cp_async_s1", 700), stage("ld_st_sync", 760)],
        }
        schedule, record = long_launch.freeze("sync-loop-ablation", pilot)
        self.assertEqual(schedule, "700\n760\n")
        self.assertEqual(
            [p["point"]["variant"] for p in record["points"]], ["cp_async_s1", "ld_st_sync"]
        )
        swapped = {
            "samples": 3,
            "sync_stage_loop_ablation": [stage("ld_st_sync", 700), stage("cp_async_s1", 760)],
        }
        with self.assertRaisesRegex(ValueError, "differs from the freeze"):
            long_launch.verify(record, swapped)
