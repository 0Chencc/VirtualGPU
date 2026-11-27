"""CPU-only tests for the capture rules of the post-hoc microbenchmarks."""

import json
import unittest
from pathlib import Path

from microbench import capture_posthoc as cap

ROOT = Path(__file__).resolve().parents[2]
CAPTURES = ROOT / "docs/results/analysis/v0.3-gb10-h3-h4"


def raw(path):
    return json.loads((CAPTURES / path).read_text())["raw"]


class CaptureRuleTests(unittest.TestCase):
    def test_rtx4070_attempt_shared_with_the_desktop_is_void(self):
        for name in ("wave-rescan.json", "staged-mma.json"):
            check = cap.dispersion_check(raw(f"captures-rtx4070-attempt1/{name}"))
            self.assertEqual(check["status"], "failed", name)

    def test_gb10_b_stream_passes(self):
        self.assertEqual(cap.dispersion_check(raw("captures/b-stream.json"))["status"], "passed")

    def test_gb10_wave_rescan_runs_fail_reproducibility(self):
        runs = [
            raw("captures/wave-rescan.json"),
            raw("captures-gb10-wave-rescan-replication/wave-rescan.json"),
        ]
        self.assertEqual(cap.reproducibility_check(runs)["status"], "failed")

    def test_wide_core_fails_rule_one(self):
        steady = [100.0] * 15
        wide = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0] * 2
        lists = [{"t_ns_samples": steady} for _ in range(18)] + [{"t_ns_samples": wide}] * 2
        check = cap.dispersion_check({"points": lists})
        self.assertEqual(check["status"], "failed")
        self.assertGreater(check["rules"]["iqr"]["share"], cap.IQR_MAX_SHARE)

    def test_isolated_spikes_pass_until_rule_two(self):
        spiked = [100.0] * 14 + [200.0]
        few = [{"gbps_samples": spiked}] * 1 + [{"gbps_samples": [100.0] * 15}] * 9
        self.assertEqual(cap.dispersion_check({"p": few})["status"], "passed")
        many = [{"gbps_samples": spiked}] * 2 + [{"gbps_samples": [100.0] * 15}] * 8
        self.assertEqual(cap.dispersion_check({"p": many})["status"], "failed")

    def test_empty_or_invalid_lists_fail(self):
        self.assertEqual(cap.dispersion_check({})["status"], "failed")
        self.assertEqual(cap.dispersion_check({"a_ns_samples": [1.0]})["status"], "failed")
        self.assertEqual(cap.dispersion_check({"a_ns_samples": [True, 2.0]})["status"], "failed")

    def test_other_sample_fields_are_ignored(self):
        data = {"a_ns_samples": [1.0, 1.01], "finish_spread_ns_offsets_samples": [1.0, 50.0]}
        self.assertEqual(cap.dispersion_check(data)["status"], "passed")

    def test_reproducibility_needs_matching_geometry(self):
        a = {"p": [{"t_ns_samples": [1.0, 1.0]}]}
        b = {"p": [{"t_ns_samples": [1.0, 1.0]}, {"t_ns_samples": [1.0, 1.0]}]}
        self.assertEqual(cap.reproducibility_check([a])["status"], "not_applicable")
        self.assertEqual(cap.reproducibility_check([a, b])["status"], "failed")
        self.assertEqual(cap.reproducibility_check([a, a])["status"], "passed")
        moved = {"p": [{"t_ns_samples": [1.2, 1.2]}, {"t_ns_samples": [1.2, 1.2]}]}
        self.assertEqual(cap.reproducibility_check([b, moved])["status"], "failed")

    def test_small_modes_tolerate_one_failing_list_or_point(self):
        steady = [100.0] * 15
        spiked = [100.0] * 14 + [200.0]
        one = {"p": [{"t_ns_samples": spiked}] + [{"t_ns_samples": steady}] * 5}
        self.assertEqual(cap.dispersion_check(one)["status"], "passed")
        two = {"p": [{"t_ns_samples": spiked}] * 2 + [{"t_ns_samples": steady}] * 4}
        self.assertEqual(cap.dispersion_check(two)["status"], "failed")
        base = {"p": [{"t_ns_samples": [1.0, 1.0]} for _ in range(6)]}
        moved_one = {"p": [{"t_ns_samples": [1.2, 1.2]}] + base["p"][1:]}
        self.assertEqual(cap.reproducibility_check([base, moved_one])["status"], "passed")
        moved_two = {"p": [{"t_ns_samples": [1.2, 1.2]}] * 2 + base["p"][2:]}
        self.assertEqual(cap.reproducibility_check([base, moved_two])["status"], "failed")

    def test_large_modes_keep_the_share_limits(self):
        self.assertEqual(cap.allowed_failures(cap.IQR_MAX_SHARE, 100), 5)
        self.assertEqual(cap.allowed_failures(cap.SPIKE_MAX_SHARE, 100), 15)
        self.assertEqual(cap.allowed_failures(cap.REPRO_MAX_SHARE, 100), 5)
        steady = [100.0] * 15
        spiked = [100.0] * 14 + [200.0]
        for bad, status in ((15, "passed"), (16, "failed")):
            data = {
                "p": [{"t_ns_samples": spiked}] * bad + [{"t_ns_samples": steady}] * (100 - bad)
            }
            self.assertEqual(cap.dispersion_check(data)["status"], status)
        base = {"p": [{"t_ns_samples": [1.0, 1.0]} for _ in range(100)]}
        for bad, status in ((5, "passed"), (6, "failed")):
            moved = {"p": [{"t_ns_samples": [1.2, 1.2]}] * bad + base["p"][bad:]}
            self.assertEqual(cap.reproducibility_check([base, moved])["status"], status)


if __name__ == "__main__":
    unittest.main()
