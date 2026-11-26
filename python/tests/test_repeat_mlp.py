"""CPU-only tests for the repeated MLP reduction."""

import json
import tempfile
import unittest
from pathlib import Path

from microbench import repeat_mlp as mlp

ROOT = Path(__file__).resolve().parents[2]


class RepeatMlpTests(unittest.TestCase):
    def setUp(self):
        self.base = ROOT / "profiles/gb10.json"
        self.profile = json.loads(self.base.read_bytes())

    def raw(self, mode, value):
        memory = self.profile["measured"]["memory"]
        return {
            field: [{**row, "gbps_samples": [value] * mlp.SAMPLES} for row in memory[field]]
            for field in mlp.MODES[mode]
        }

    def test_geometry_and_full_sample_count_are_required(self):
        raw = self.raw("mlp", 20.0)
        mlp.point_maps(self.profile, "mlp", raw)
        point = raw[mlp.CONTIGUOUS[0]][0]
        point["gbps_samples"].pop()
        with self.assertRaisesRegex(ValueError, "finite positive samples"):
            mlp.point_maps(self.profile, "mlp", raw)
        point["bytes_in_flight"] += 1
        with self.assertRaisesRegex(ValueError, "geometry differs"):
            mlp.point_maps(self.profile, "mlp", raw)

    def test_gb10_gate_requires_temperature_clock_and_event_stability(self):
        row = {key: "Not Active" for key in mlp.THERMAL_EVENT_FIELDS}
        row.update({"temperature.gpu": "48", "clocks.current.sm": "2500"})
        trace = [{**row, "elapsed_seconds": second} for second in range(60)]
        windows = mlp.thermal_windows(trace, 60)
        self.assertEqual(mlp.thermal_stability(windows)["samples_per_window"], [30, 30])
        for patch in (
            {"temperature.gpu": "49"},
            {"clocks.current.sm": "2530"},
            {mlp.THERMAL_EVENT_FIELDS[-1]: "Active"},
            {mlp.THERMAL_EVENT_FIELDS[-1]: "[N/A]"},
        ):
            with self.subTest(patch=patch):
                recent = [{**sample, **patch} for sample in windows[1]]
                self.assertIsNone(mlp.thermal_stability((windows[0], recent)))

    def test_reduce_uses_independent_medians_and_existing_ceiling(self):
        snapshot = {key: "Not Active" for key in mlp.TELEMETRY}
        snapshot.update({"clocks.current.sm": "2000", "clocks.current.memory": "[N/A]"})
        paths = []
        source = {
            "source_revision": "fixture",
            "source_bundle_sha256": "fixture",
            "base_profile_sha256": mlp.digest(self.base),
            "binary_sha256": "fixture",
            "nvcc": "fixture",
            "target": "sm_121",
            "host": "fixture",
            "initial_gpu": {"uuid": "fixture", "driver_version": "fixture"},
            "clock_policy": "record_only",
            "telemetry": [snapshot],
            "command": ["fixture"],
            "started_utc": "2026-11-26T10:00:00+00:00",
            "finished_utc": "2026-11-26T10:01:00+00:00",
        }
        with tempfile.TemporaryDirectory() as temp:
            for mode in mlp.MODES:
                for number, value in enumerate((20.0, 23.0, 22.0, 90.0, 1.0), start=1):
                    path = Path(temp) / f"{mode}-{number}.json"
                    mlp.write_json(
                        path,
                        {**source, "mode": mode, "repeat": number, "raw": self.raw(mode, value)},
                    )
                    paths.append(path)
            updated, audit = mlp.reduce_runs(self.base, paths, "fixture")
            with self.assertRaisesRegex(ValueError, "Expected repeats"):
                mlp.reduce_runs(self.base, paths[:-1], "fixture")
        self.assertEqual(audit["previous_provenance"], self.profile["provenance"])
        self.assertEqual(
            updated["provenance"]["clock_policy"], self.profile["provenance"]["clock_policy"]
        )
        self.assertEqual(audit["measurement_telemetry"]["active_sm_clock_range_mhz"], [2000, 2000])
        memory = updated["measured"]["memory"]
        for field in (*mlp.CONTIGUOUS, mlp.STRIDED):
            self.assertEqual(memory[field][0]["bw_gbps"]["value"], 22.0)
            self.assertEqual(memory[field][0]["bw_gbps"]["rel_uncertainty"], 1.0)
            point = audit["fields"][field]["points"][0]
            self.assertEqual(point["repeat_range_gbps"], [1.0, 90.0])
            self.assertEqual(point["pooled_median_gbps"], 22.0)
        self.assertEqual(
            memory["dram_bw_read_gbps"], memory["dram_bw_by_active_sms"][-1]["bw_gbps"]
        )
        self.assertIsNone(audit["fields"][mlp.CONTIGUOUS[0]]["active_clock_range_mhz"])


if __name__ == "__main__":
    unittest.main()
