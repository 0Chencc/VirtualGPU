"""CPU-only checks for the manifest, baseline, and scoring."""

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from virtualgpu.frontend.compiler import ARTIFACT_HASH_VERSION

from validation.baseline import gemm_work, latency_ns, predict
from validation.experiment import core_records, development_gates, load_case, validate_core_builds
from validation.neusight import TileFeatures, fit, leave_one_case_out
from validation.provenance import read_core_revision, require_formal_build
from validation.score import check_artifact, kendall_tau_b, median_ci95, observed_pair, predicted_pair, score
from validation.suite import development_manifest, manifest


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

    def test_materialized_development_boundaries(self):
        suite = development_manifest({"specs": {"sm_count": 46},
                                      "provenance": {"profile_revision": "test"}})
        self.assertTrue(all(case["split"] == "development" for case in suite["cases"]))
        boundaries = [case for case in suite["cases"] if case["family"] == "sm-round"]
        self.assertEqual([case["shape"][1] // 64 for case in boundaries], [45, 46, 47])
        self.assertTrue(all(len(case["config_ids"]) == 36 for case in boundaries))


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
    def test_core_draws_include_every_shared_id(self):
        row = {"status": "ok", "nominal_latency_ns": 100, "draws": {"0": 100, "1": 110}}
        data = {"cases": {"case": {"cold": {"cfg": row}}}, "perturbation_draws": 1}
        self.assertEqual(core_records(data, "case", "cold", {"cfg": {}}), {"cfg": row})
        row["draws"] = {"0": 100}
        with self.assertRaisesRegex(ValueError, "invalid shared draw set"):
            core_records(data, "case", "cold", {"cfg": {}})

    def test_development_gates_include_regret_non_inferiority(self):
        cache = {
            "core": {"median_tau_b": 0.70, "mean_top_5_regret": 0.01},
            "roofline_wave": {"median_tau_b": 0.61, "mean_top_5_regret": 0.02},
            "neusight_style": {"median_tau_b": 0.65, "mean_top_5_regret": 0.005},
        }
        self.assertTrue(development_gates({"cold": cache})["cold"]["pass"])
        cache["neusight_style"]["mean_top_5_regret"] = 0.004
        self.assertFalse(development_gates({"cold": cache})["cold"]["pass"])

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

    def test_nominal_baseline_pair_margin(self):
        blocks = {("session", index): 100 for index in range(3)}
        records = {key: {"status": "ok", "nominal_latency_ns": value}
                   for key, value in (("a", 100), ("b", 104), ("c", 106))}
        result = score(dict.fromkeys(records, blocks), records, nominal_pair_margin=0.05)
        self.assertEqual(result["pair_confusion"],
                         {"indistinguishable": {"indistinguishable": 2, "a_faster": 1}})

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
        self.assertEqual(predicted_pair({"0": 100, "1": 100}, {"0": 106, "1": 104}), "indistinguishable")
        self.assertEqual(predicted_pair({"0": 100, "1": 100}, {"0": 105, "1": 110}), "a_faster")
        self.assertEqual(predicted_pair({"0": 100}, {"0": 106}, math.log(1.10)), "indistinguishable")
        self.assertEqual(predicted_pair({"0": 100}, {"0": 100}, 0), "indistinguishable")

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


class NeuSightTest(unittest.TestCase):
    def test_fit_uses_only_development_cases_other_than_scored_case(self):
        tile = TileFeatures((1.0, 2.0), 30.0, 3, 5.0)
        examples = [
            {"case_id": "a", "split": "development", "tile": tile, "measured_ns": 20.0},
            {"case_id": "b", "split": "development", "tile": tile, "measured_ns": 185.0},
        ]
        fits = leave_one_case_out(examples)
        self.assertAlmostEqual(fits["a"].predict_ns(tile), 185.0)
        self.assertAlmostEqual(fits["b"].predict_ns(tile), tile.kernel_floor_ns)
        with self.assertRaisesRegex(ValueError, "only development"):
            fit([dict(examples[0], split="holdout")])


class MeasurementTest(unittest.TestCase):
    def test_clean_cold_replacement_preserves_warm_labels(self):
        case = {"case_id": "shape", "shape": [1, 2, 3], "config_ids": ["cfg"]}
        report = {
            "case_id": "shape", "split": "development", "shape": [1, 2, 3],
            "manifest_sha256": "manifest", "session_id": "old",
            "gpu": {}, "software": {}, "kernel_sha256": "kernel",
            "configs": {"cfg": {"status": "ok"}},
        }
        protocol = {"clock_policy": "record_only", "timer": "graph", "l2_bytes": 4,
                    "eviction_bytes": 8, "blocks": 1}

        def sample(cache, ns):
            return dict(cache=cache, config_id="cfg", block=0, latency_ns=ns)

        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            cold = results / "replacement"
            aot = results / "aot" / "shape"
            cold.mkdir()
            aot.mkdir(parents=True)
            (aot / "cfg.json").write_text("{}")
            (results / "shape-session-1.json").write_text(json.dumps(dict(
                report, protocol=protocol, samples=[sample("warm", 100), sample("cold", 150)],
            )))
            (cold / "shape-session-1.json").write_text(json.dumps(dict(
                report, session_id="new",
                protocol=dict(protocol, cache_states=["cold"], cold_flush="read_cg",
                              eviction_bytes=16),
                samples=[sample("cold", 120)],
            )))
            with patch("validation.experiment.check_artifact", return_value={"status": "matched"}):
                with self.assertRaisesRegex(ValueError, "require read-only"):
                    load_case(case, results, "manifest", 1)
                data = load_case(case, results, "manifest", 1, cold)
        self.assertEqual(data.timings["warm"]["cfg"], {("old", 0): 100})
        self.assertEqual(data.timings["cold"]["cfg"], {("new", 0): 120})
        self.assertEqual(data.matched_config_sessions, 2)


class ProvenanceTest(unittest.TestCase):
    revision = "a" * 40

    def test_expected_revision_from_file_or_full_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "expected-revision.txt"
            path.write_text(self.revision + "\n")
            self.assertEqual(read_core_revision(str(path)), self.revision)
        self.assertEqual(read_core_revision(self.revision), self.revision)
        with self.assertRaises(ValueError):
            read_core_revision("abc")

    def test_all_formal_records_require_matching_clean_builds(self):
        clean = {"commit": self.revision, "dirty": False, "result_class": "formal"}
        require_formal_build(clean, self.revision, "candidate")
        rows = {"ok": {"status": "ok", "build": clean}, "failed": {"status": "failed", "build": clean}}
        bundle = {"cases": {"decode": {"warm": rows}}}
        validate_core_builds(bundle, self.revision)
        for bad in (None, dict(clean, dirty=True), dict(clean, result_class="diagnostic"),
                    dict(clean, commit="b" * 40)):
            with self.subTest(build=bad):
                rows["failed"]["build"] = bad
                with self.assertRaisesRegex(ValueError, "decode warm failed"):
                    validate_core_builds(bundle, self.revision)
        rows["failed"]["build"] = clean
        with self.assertRaisesRegex(ValueError, "Diagnostic prediction bundles"):
            validate_core_builds(dict(bundle, result_class="diagnostic"), self.revision)


if __name__ == "__main__":
    unittest.main()
