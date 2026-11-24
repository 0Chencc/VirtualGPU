"""Tests for AOT compilation cache, Python batch API, and CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from virtualgpu.api import (
    extract_gemm_configs,
    predict_gemm_configs,
    predict_ir_dir,
    predict_kernel_ir,
)
from virtualgpu.cli import main as cli_main
from virtualgpu.frontend import compiler as compiler_mod
from virtualgpu.frontend.tests.test_triton_frontend import _matmul_kernel


REPO_ROOT = Path(__file__).resolve().parents[4]
SCHEMA_PATH = REPO_ROOT / "schemas" / "kernel_ir.schema.json"
PROFILE_EXAMPLE_PATH = REPO_ROOT / "schemas" / "examples" / "profile_example.json"


class _DummyNominalCore:
    """Mimics M1 virtualgpu._core with single-draw nominal predict()."""

    @staticmethod
    def predict(kernel_ir_json: str, profile_json: str, l2_state: str = "cold") -> dict[str, Any]:
        ir = json.loads(kernel_ir_json)
        if ir["resources"]["static_smem_bytes"] > 100_000:
            return {
                "status": "unsupported",
                "l2_state": l2_state,
                "dispatch_policy": "round_robin",
                "reason": "SmemExceedsLimit",
            }
        return {
            "status": "ok",
            "l2_state": l2_state,
            "dispatch_policy": "round_robin",
            "time_ns": 12345.0,
            "makespan_ns": 11000.0,
            "ctas_per_sm": 1,
            "occupancy_limiter": "Registers",
            "waves": 2,
            "tail_efficiency": 1.0,
            "load_bytes": 1024,
            "l2_hit_bytes": 512,
            "dram_bytes": 512,
            "bound_fraction": {"mma": 0.8, "dram": 0.2},
        }


class _DummyPerturbedCore:
    """Mimics M2 virtualgpu._core with perturbation draws support."""

    @staticmethod
    def predict(
        kernel_ir_json: str,
        profile_json: str,
        l2_state: str = "cold",
        draws: int = 1,
        seed: int = 0,
    ) -> dict[str, Any]:
        base = 10000.0 + seed
        return {
            "status": "ok",
            "l2_state": l2_state,
            "dispatch_policy": "round_robin",
            "time_ns": base,
            "draws": [base + 100.0 * i for i in range(draws)],
            "waves": 1,
        }


def test_aot_disk_cache_hits_and_skips_recompilation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_dir = tmp_path / "aot_cache"
    cfg = {
        "config_id": "c_cache",
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 3,
    }
    compiler_mod._MEMORY_CACHE.clear()
    res1 = extract_gemm_configs(
        _matmul_kernel,
        shape=(1024, 1024, 512),
        configs=[cfg],
        target_arch="sm_89",
        cache_dir=cache_dir,
        use_cache=True,
    )
    assert res1["c_cache"]["status"] == "ok"
    cached_files = list(cache_dir.glob("*.json"))
    assert len(cached_files) == 1

    # Clear memory cache and poison triton.compile to prove disk cache is used.
    compiler_mod._MEMORY_CACHE.clear()
    import triton

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("triton.compile should not be called on disk cache hit")

    monkeypatch.setattr(triton, "compile", _boom)
    res2 = extract_gemm_configs(
        _matmul_kernel,
        shape=(1024, 1024, 512),
        configs=[cfg],
        target_arch="sm_89",
        cache_dir=cache_dir,
        use_cache=True,
    )
    assert res2["c_cache"]["status"] == "ok"
    assert (
        res1["c_cache"]["ir"]["metadata"]["toolchain"]["ptx_sha256"]
        == res2["c_cache"]["ir"]["metadata"]["toolchain"]["ptx_sha256"]
    )


def test_predict_api_formats_nominal_perturbed_and_unsupported_records(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    ir_dir = tmp_path / "irs"
    configs = [
        {
            "config_id": "c_ok",
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": 3,
        },
        {
            "config_id": "c_big_smem",
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": 5,
        },
    ]
    records = predict_gemm_configs(
        _matmul_kernel,
        case_id="test-case",
        shape=(1024, 1024, 512),
        configs=configs,
        profile=PROFILE_EXAMPLE_PATH,
        l2_state="cold",
        cache_dir=cache_dir,
        ir_out_dir=ir_dir,
        core_module=_DummyNominalCore,
    )
    assert len(records) == 2
    by_id = {r["config_id"]: r for r in records}
    assert by_id["c_ok"]["status"] == "ok"
    assert by_id["c_ok"]["nominal_latency_ns"] == 12345.0
    assert by_id["c_ok"]["draws"] == {"0": 12345.0}
    assert by_id["c_ok"]["assumptions"]["l2_state"] == "cold"
    assert by_id["c_big_smem"]["status"] == "unsupported"
    assert by_id["c_big_smem"]["reason"] == "SmemExceedsLimit"

    # Perturbed core populates string-keyed draws {"0": ..., "1": ..., "2": ...}
    ir_ok = json.loads((ir_dir / "c_ok.json").read_text(encoding="utf-8"))
    rec_pert = predict_kernel_ir(
        ir_ok,
        PROFILE_EXAMPLE_PATH,
        case_id="test-case",
        config_id="c_ok",
        l2_state="warm",
        draws=3,
        seed=7,
        core_module=_DummyPerturbedCore,
    )
    assert rec_pert["status"] == "ok"
    assert rec_pert["draws"] == {"0": 10007.0, "1": 10107.0, "2": 10207.0}

    # Directory prediction helper produces matching records.
    dir_records = predict_ir_dir(
        ir_dir,
        case_id="test-case",
        profile=PROFILE_EXAMPLE_PATH,
        l2_state="cold",
        core_module=_DummyNominalCore,
    )
    assert {r["config_id"] for r in dir_records} == {"c_ok", "c_big_smem"}


def test_cli_extract_and_predict_ir_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    cfg_file = tmp_path / "configs.json"
    cfg_file.write_text(
        json.dumps([
            {
                "config_id": "c01",
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
                "num_warps": 4,
                "num_stages": 3,
            }
        ]),
        encoding="utf-8",
    )
    ir_dir = tmp_path / "irs"
    cache_dir = tmp_path / "cache"
    rc = cli_main([
        "extract",
        "--kernel", "virtualgpu.frontend.tests.test_triton_frontend:_matmul_kernel",
        "--shape", "512", "512", "256",
        "--configs-json", str(cfg_file),
        "--arch", "sm_89",
        "--out-dir", str(ir_dir),
        "--cache-dir", str(cache_dir),
    ])
    assert rc == 0
    ir_path = ir_dir / "c01.json"
    assert ir_path.is_file()
    jsonschema.validate(instance=json.loads(ir_path.read_text(encoding="utf-8")), schema=schema)

    # Patch virtualgpu._core for CPU-only CLI predict test when extension is not built.
    import virtualgpu

    monkeypatch.setattr(virtualgpu, "_core", _DummyNominalCore, raising=False)
    out_preds = tmp_path / "preds.json"
    rc2 = cli_main([
        "predict",
        "--ir-dir", str(ir_dir),
        "--case", "smoke-case",
        "--profile", str(PROFILE_EXAMPLE_PATH),
        "--cache", "cold",
        "--out", str(out_preds),
    ])
    assert rc2 == 0
    preds = json.loads(out_preds.read_text(encoding="utf-8"))
    assert len(preds) == 1
    assert preds[0]["case_id"] == "smoke-case"
    assert preds[0]["config_id"] == "c01"
    assert preds[0]["status"] == "ok"
    assert preds[0]["draws"] == {"0": 12345.0}
