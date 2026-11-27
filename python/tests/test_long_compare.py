"""CPU-only tests for the long-launch mechanism comparison."""

import unittest

from microbench import long_compare


def point(warps, steps, ns, trials=None):
    row = {"mma_warps_per_sm": warps, "steps": steps, "launch_ns_samples": [ns] * 3}
    if trials:
        row["sizing_trials"] = [{"steps": s, "launch_ns_samples": [1.0] * 3} for s in trials]
    return row


def capture(rows):
    return {"samples": 3, "mma": {"f16": rows}}


class LongCompareTests(unittest.TestCase):
    def test_x1_schedule_takes_each_point_s_first_trial(self):
        pilot = capture([point(1, 9000, 1.3e6, [4000, 9000]), point(2, 8000, 1.3e6, [4000])])
        self.assertEqual(long_compare.x1_schedule("mma", pilot), "4000\n4000\n")

    def test_x1_schedule_writes_frozen_parameters_ahead_of_lengths(self):
        def row(iters, trial):
            return {
                "ctas_per_sm": 1,
                "tokens": 1,
                "num_warps": 4,
                "stage_bytes": 8192,
                "ldmatrix_bytes_per_stage": 0,
                "mma_iters_per_stage": iters,
                "steps": 900,
                "launch_ns_samples": [1.3e6] * 3,
                "sizing_trials": [{"steps": trial, "launch_ns_samples": [1.0] * 3}],
            }

        pilot = {"samples": 3, "dram_closed_loop_mma": [row(0, 256), row(37, 256)]}
        self.assertEqual(long_compare.x1_schedule("closed-loop-mma", pilot), "0\n256\n37\n256\n")

    def test_shift_and_fixed_cost_follow_the_one_over_steps_model(self):
        # 10 ns per step plus a 3000 ns fixed cost per launch.
        x1 = capture([point(1, 1000, 13000.0)])
        long = capture([point(1, 4000, 43000.0)])
        result = long_compare.shift("mma", [x1, x1], [long, long])
        row = result["rows"][0]
        self.assertAlmostEqual(row["fixed_cost_ns"], 3000.0)
        self.assertAlmostEqual(result["shift_median"], 10.75 / 13.0 - 1)

    def test_points_must_match(self):
        with self.assertRaisesRegex(ValueError, "differ in points"):
            long_compare.shift(
                "mma", [capture([point(1, 10, 1.0)])], [capture([point(2, 10, 1.0)])]
            )


if __name__ == "__main__":
    unittest.main()
