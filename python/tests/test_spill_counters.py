"""CPU-only tests of the post-hoc counter collection and attribution."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from microbench import spill_counters as spill


class Metric:
    def __init__(self, name, value=0, pcs=None):
        self.name = lambda: name
        self.values = value
        self.pcs = pcs

    def value(self, index=None):
        return self.values if index is None else self.values[index]

    def num_instances(self):
        return len(self.pcs) if self.pcs is not None else 0

    def has_correlation_ids(self):
        return self.pcs is not None

    def correlation_ids(self):
        return SimpleNamespace(as_uint64=lambda index: self.pcs[index])


def action(rows, local_sectors=(0, 0)):
    metrics = {
        spill.SOURCE_METRIC: Metric(
            spill.SOURCE_METRIC, [value for _, value, _ in rows], range(len(rows))
        ),
        **{
            key: Metric(key, value)
            for key, value in zip((*spill.SECTORS, *spill.WARP_METRICS), (*local_sectors, 0, 0))
        },
        "launch__grid_size": Metric("launch__grid_size", 2),
        "launch__block_size": Metric("launch__block_size", 128),
    }
    return SimpleNamespace(
        name=lambda: "test_kernel",
        metric_by_name=metrics.get,
        sass_by_pc=lambda pc: rows[pc][0],
        source_info=lambda pc: SimpleNamespace(
            file_name=lambda: "spill_attention_source.py", line=lambda: rows[pc][2]
        ),
    )


class SpillCounterTests(unittest.TestCase):
    def test_permission_failure_with_zero_subprocess_exit_does_not_publish(self):
        denied = subprocess.CompletedProcess(["ncu"], 0, "", "==ERROR== ERR_NVGPUCTRPERM")

        def measure(*_):
            with patch.object(spill.subprocess, "run", return_value=denied):
                return spill.run(["ncu"], {})

        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(spill, "POST_HOC", Path(folder) / "post-hoc"),
            patch.object(spill.shutil, "which", return_value="/tmp/ncu"),
            patch.object(spill, "measure", side_effect=measure),
            patch.object(spill.sys, "argv", ["spill_counters"]),
            patch.object(spill.sys, "stderr") as stderr,
        ):
            with self.assertRaises(SystemExit) as error:
                spill.main()
            self.assertEqual(error.exception.code, 1)
            self.assertIn("ERR_NVGPUCTRPERM", stderr.write.call_args.args[0])
            self.assertFalse(spill.POST_HOC.exists())

    def test_spill_uses_control_to_remove_frame_initialization(self):
        zero = action([("STL.64 [R1], R0", 8192, 0)], (0, 1024))
        measured = action(
            [("@P0 LDL R0, [R1]", 65536, 0), ("STL.64 [R1], R0", 8192, 0)], (8192, 1024)
        )
        row = spill.spill_row((128, 1, 2, 128, 2), [zero, measured])
        self.assertEqual(
            row["per_thread_iteration"], {"thread_local_load": 2, "thread_local_store": 0}
        )
        self.assertEqual(row["per_cta_iteration"][spill.SECTORS[0]], 32)
        self.assertEqual(row["per_cta_iteration"][spill.SECTORS[1]], 0)

    def test_attention_uses_distinct_source_ranges_and_counts_odd_tail(self):
        helper = "def _attention_tiles():\n    pass\n"
        kernel = "def fused_attention():\n    _attention_tiles()\n    _attention_tiles()\n    _attention_tiles()\n"
        src = spill.attention_source(helper, kernel)
        ranges = spill.scope_ranges(src)
        self.assertEqual(set(ranges), {"main", "diagonal"})
        main, diagonal = [ranges[key][0] for key in ("main", "diagonal")]
        rows = [
            ("@!P0 LDL.128 R1, [R2]", 16, main),
            ("STL [R1], R2", 24, diagonal),
            ("LDL R1, [R2]", 3, 999),
            ("LDG.E R1, [R2]", 99, main),
        ]
        case = {"batch": 1, "query_heads": 2, "seq_len": 97}
        config = {"config_id": "test", "block_m": 64, "block_n": 32, "num_warps": 4}
        measured = action(rows)
        row = spill.attention_row(case, config, measured, measured, ranges)
        self.assertEqual(row["loops"]["main"]["cta_iterations"], 4)
        self.assertEqual(row["loops"]["diagonal"]["cta_iterations"], 8)
        self.assertEqual(row["loops"]["main"]["per_cta_iteration"]["load"], 4)
        self.assertEqual(row["loops"]["diagonal"]["per_cta_iteration"]["store"], 3)
        self.assertEqual(row["outside_or_unattributed"], {"load": 3, "store": 0})
        with self.assertRaisesRegex(ValueError, "changed dynamic local instruction counts"):
            spill.attention_row(case, config, measured, action(rows[1:]), ranges)

    def test_target_follows_device_compute_capability(self):
        self.assertEqual(spill.target_arch("8.6"), "sm_86")
        self.assertEqual(spill.target_arch("12.1"), "sm_121")
        with self.assertRaises(ValueError):
            spill.target_arch("[N/A]")

    def test_atomic_publish_only_exposes_complete_json_and_keeps_existing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "post-hoc" / "counters.json"
            data = {"result_class": "post-hoc", "spill_reload_latency_by_count": [1]}
            spill.publish(path, data)
            with self.assertRaises(FileExistsError):
                spill.publish(path, {"partial": True})
            self.assertEqual(json.loads(path.read_bytes()), data)
            self.assertEqual(list(path.parent.iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
