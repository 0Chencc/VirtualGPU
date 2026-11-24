"""CPU-only checks for the manifest, baseline, and scoring."""

import json
from pathlib import Path
import unittest

from virtualgpu.frontend.compiler import ARTIFACT_HASH_VERSION

from validation.baseline import gemm_work, latency_ns, predict
from validation.score import check_artifact, kendall_tau_b, median_ci95, observed_pair, predicted_pair, score
from validation.suite import manifest


class ManifestTest(unittest.TestCase):
    def test_frozen_manifest(self):
        frozen = json.loads(Path(__file__).with_name("suite.json").read_text())
        self.assertEqual(frozen, manifest())
        configs = {entry["config_id"] for entry in frozen["configs"]}
        splits = {"development": set(), "holdout": set()}
        for case in frozen["cases"]:
            self.assertTrue(set(case["config_ids"]) <= configs)
            splits[case["split"]].add(tuple(case["shape"]))
        self.assertFalse(splits["development"] & splits["holdout"])
        self.assertEqual(len(frozen["cases"][0]["config_ids"]), 36)


class BaselineTest(unittest.TestCase):
    def test_wave_tail_rounds_up(self):
        inputs = dict(resident_ctas_per_sm=2, sm_count=4, flops_per_cta=10,
                      bytes_per_cta=20, flops_per_second=1e9, bytes_per_second=1e9,
                      launch_ns=50)
        self.assertEqual(latency_ns(cta_count=8, **inputs), 210)
        self.assertEqual(latency_ns(cta_count=9, **inputs), 370)

    def test_masked_gemm(self):
        count, flops, size = gemm_work((17, 65, 33), (16, 64, 32))
        self.assertEqual(count, 4)
        self.assertEqual(flops, 2 * 16 * 64 * 64)
        self.assertEqual(size, 2 * (17 * 33 * 2 + 33 * 65 * 2 + 17 * 65) / 4)

    def test_profile_contract(self):
        path = Path(__file__).parents[1] / "schemas/examples/profile_example.json"
        profile = json.loads(path.read_text())
        measurement = {
            "case_id": "cube", "shape": [4096, 4096, 4096], "gpu": {"arch": "sm_89"},
            "configs": {
                "a": {"status": "ok", "num_warps": 4, "registers_per_thread": 96,
                      "shared_bytes": 49152, "tile": [128, 128, 32], "mma_shapes": ["16x8x16"]},
                "b": {"status": "rejected", "reason": "fixture"},
            },
        }
        a, b = predict(measurement, profile)
        self.assertEqual(a["profile_revision"], profile["provenance"]["profile_revision"])
        self.assertGreater(a["nominal_latency_ns"], 0)
        self.assertEqual(b["status"], "failed")


class ScoreTest(unittest.TestCase):
    def test_artifact_gate(self):
        toolchain = {
            "compiler": "triton", "compiler_version": "3.9.0", "assembler": "ptxas",
            "assembler_version": "12.9", "target_sm_compiled": "sm_89",
        }
        ir = {
            "metadata": {
                "config": {},
                "toolchain": dict(toolchain, artifact_hash_version=ARTIFACT_HASH_VERSION,
                                  ptx_sha256="matching", ttgir_sha256="aot"),
            },
            "launch": {"grid": [8, 1, 1], "dynamic_smem_bytes": 0},
            "resources": {"static_smem_bytes": 4096, "registers_per_thread": 96},
        }
        jit = {
            "artifact_hash_version": ARTIFACT_HASH_VERSION, "ptx_sha256": "matching",
            "ttgir_sha256": "jit",
            "grid": [8], "shared_bytes": 4096, "registers_per_thread": 96,
            "toolchain": dict(toolchain, enable_fp_fusion=True, ptx_options=None,
                              disable_ptxas_opt=False),
        }
        self.assertEqual(check_artifact(jit, ir),
                         {"status": "matched", "differences": [], "ttgir_match": False})
        jit["artifact_hash_version"] = "1"
        ir["metadata"]["toolchain"]["artifact_hash_version"] = "1"
        self.assertEqual(check_artifact(jit, ir)["differences"],
                         ["artifact_hash_version", "aot_artifact_hash_version"])
        jit["artifact_hash_version"] = ARTIFACT_HASH_VERSION
        ir["metadata"]["toolchain"]["artifact_hash_version"] = ARTIFACT_HASH_VERSION
        jit["ptx_sha256"] = "different"
        jit["registers_per_thread"] = 100
        self.assertEqual(check_artifact(jit, ir)["differences"],
                         ["ptx_sha256", "registers_per_thread"])
        jit["toolchain"]["enable_fp_fusion"] = False
        self.assertIn("enable_fp_fusion", check_artifact(jit, ir)["differences"])

    def test_measurement_intervals_and_gaps(self):
        self.assertEqual(median_ci95([[100, 100], [110, 110]]), (105, 105))
        blocks = {("session", index): 100 for index in range(3)}
        result = score({"a": blocks, "b": dict.fromkeys(blocks, 103)}, {})
        self.assertEqual(result["configs"]["a"]["measured_median_ci95_ns"], (100, 100))
        self.assertEqual(result["configs"]["a"]["relative_ci95_half_width"], 0)
        self.assertAlmostEqual(result["configs"]["b"]["gap_to_measured_best"], 0.03)

    def test_tau_b_with_ties(self):
        actual = {"a": 1, "b": 2, "c": 2}
        self.assertAlmostEqual(kendall_tau_b(actual, {"a": 1, "b": 2, "c": 3}), 2 / 6**0.5)
        self.assertEqual(kendall_tau_b(actual, {"a": 3, "b": 2, "c": 2}), -1)
        self.assertIsNone(kendall_tau_b(actual, {"a": 1}))

    def test_pair_rules(self):
        actual = {("session", block): 100 for block in range(8)}
        close = dict.fromkeys(actual, 103)
        faster = dict.fromkeys(actual, 90)
        self.assertEqual(observed_pair(actual, close, 100, 1)[0], "indistinguishable")
        self.assertEqual(observed_pair(actual, faster, 100, 1)[0], "b_faster")
        self.assertEqual(predicted_pair({"0": 4, "1": 2}, {"0": 3, "1": 3}), "indistinguishable")
        self.assertEqual(predicted_pair({"0": 3, "1": 2}, {"0": 3, "1": 3}), "indistinguishable")
        self.assertEqual(predicted_pair({"0": 4}, {"0": 3}), "b_faster")
        self.assertEqual(predicted_pair({"0": 4, "1": 4}, {"0": 3}), "unavailable")

    def test_shortlist_oracle_includes_abstentions(self):
        block = ("session", 0)
        timings = {"a": {block: 100}, "b": {block: 120}, "c": {block: 180}}
        records = {
            "a": {"status": "unsupported"},
            "b": {"status": "ok", "nominal_latency_ns": 40},
            "c": {"status": "ok", "nominal_latency_ns": 50},
        }
        result = score(timings, records, k=1, samples=10)
        self.assertEqual(result["shortlist"], ["b"])
        self.assertFalse(result["top_k_hit"])
        self.assertAlmostEqual(result["top_k_regret"], 0.2)
        self.assertEqual(result["kendall_tau_b"], 1)
        self.assertEqual(result["scored_configs"], 2)
        self.assertEqual(result["measured_configs"], 3)


if __name__ == "__main__":
    unittest.main()
