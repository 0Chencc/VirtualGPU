"""GPU profile schema validation and loading utilities."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

DEFAULT_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "schemas" / "gpu_profile.schema.json"
)


class ProfileValidationError(ValueError):
    """Raised when a GPU profile fails schema or semantic validation."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        summary = "\n".join(f"- {err}" for err in errors)
        super().__init__(f"GPU profile validation failed ({len(errors)} error(s)):\n{summary}")


def load_schema(schema_path: Path | str | None = None) -> dict[str, Any]:
    if schema_path is None:
        try:
            from virtualgpu import _core

            return json.loads(_core.schema("gpu_profile"))
        except Exception:
            schema_path = DEFAULT_SCHEMA_PATH
    path = Path(schema_path)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _check_measured_value(mv: dict[str, Any], path: str, errors: list[str]) -> None:
    lo = mv.get("ci95_low")
    hi = mv.get("ci95_high")
    val = mv["value"]
    if lo is not None and hi is not None:
        if lo > hi:
            errors.append(f"{path}: ci95_low ({lo}) > ci95_high ({hi})")
        if not (lo <= val <= hi):
            errors.append(f"{path}: value ({val}) outside [ci95_low={lo}, ci95_high={hi}]")


def _walk_measured_values(obj: Any, path: str, errors: list[str]) -> None:
    if isinstance(obj, dict):
        if {"value", "rel_uncertainty", "unit", "method"}.issubset(obj.keys()):
            _check_measured_value(obj, path, errors)
            return
        for key, val in obj.items():
            _walk_measured_values(val, f"{path}.{key}", errors)
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            _walk_measured_values(item, f"{path}[{idx}]", errors)


def _check_monotonic_curve(
    points: list[dict[str, Any]],
    x_key: str,
    max_x: int,
    path: str,
    errors: list[str],
) -> None:
    prev = 0
    for idx, pt in enumerate(points):
        x = pt[x_key]
        if x <= prev:
            errors.append(f"{path}[{idx}].{x_key} ({x}) must be strictly increasing (prev={prev})")
        if x > max_x:
            errors.append(f"{path}[{idx}].{x_key} ({x}) exceeds hardware limit ({max_x})")
        prev = x


def _semantic_checks(profile: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    specs = profile["specs"]
    measured = profile["measured"]

    _walk_measured_values(measured, "measured", errors)

    sm_count = specs["sm_count"]
    mem = measured["memory"]
    _check_monotonic_curve(
        mem["dram_bw_by_active_sms"], "active_sms", sm_count, "measured.memory.dram_bw_by_active_sms", errors
    )
    _check_monotonic_curve(
        mem["l2_bw_by_active_sms"], "active_sms", sm_count, "measured.memory.l2_bw_by_active_sms", errors
    )

    for lat_key in ("dram_latency_by_bw", "l2_latency_by_bw"):
        if lat_key in mem:
            prev_bw = -1.0
            for idx, pt in enumerate(mem[lat_key]):
                bw = pt["achieved_bw_gbps"]
                if bw < prev_bw:
                    errors.append(
                        f"measured.memory.{lat_key}[{idx}].achieved_bw_gbps ({bw}) must be non-decreasing (prev={prev_bw})"
                    )
                prev_bw = bw

    if "l2_rescan_hit_rate_by_working_set" in mem:
        prev_ws = 0
        for idx, pt in enumerate(mem["l2_rescan_hit_rate_by_working_set"]):
            ws = pt["working_set_bytes"]
            if ws <= prev_ws:
                errors.append(
                    f"measured.memory.l2_rescan_hit_rate_by_working_set[{idx}].working_set_bytes ({ws}) "
                    f"must be strictly increasing (prev={prev_ws})"
                )
            hr = pt["estimated_hit_rate"]["value"]
            if not (0.0 <= hr <= 1.0):
                errors.append(
                    f"measured.memory.l2_rescan_hit_rate_by_working_set[{idx}].estimated_hit_rate.value ({hr}) "
                    f"must be within [0, 1]"
                )
            prev_ws = ws

    max_warps = specs["max_warps_per_sm"]
    for key, curve in measured["compute"]["mma_tflops_per_sm_by_warps"].items():
        _check_monotonic_curve(
            curve,
            "mma_warps_per_sm",
            max_warps,
            f"measured.compute.mma_tflops_per_sm_by_warps['{key}']",
            errors,
        )

    clock_policy = profile["provenance"]["clock_policy"]
    for field in ("observed_sm_clock_range_mhz", "observed_mem_clock_range_mhz"):
        rng = clock_policy.get(field)
        if rng is not None and rng[0] > rng[1]:
            errors.append(f"provenance.clock_policy.{field}: min ({rng[0]}) > max ({rng[1]})")

    dispatch = measured["cta_dispatch"]
    if dispatch["observed_policy"] not in dispatch["candidate_policies"]:
        errors.append(
            f"measured.cta_dispatch.observed_policy ({dispatch['observed_policy']!r}) "
            f"must appear in candidate_policies ({dispatch['candidate_policies']!r})"
        )

    return errors


def validate_profile(
    profile: dict[str, Any],
    schema_path: Path | str | None = None,
) -> None:
    """Validate a GPU profile dict against the JSON Schema and semantic invariants."""
    schema = load_schema(schema_path)
    validator = Draft202012Validator(schema)
    schema_errors = sorted(validator.iter_errors(profile), key=lambda e: list(e.absolute_path))
    if schema_errors:
        formatted = [
            f"{'.'.join(str(p) for p in err.absolute_path) or '<root>'}: {err.message}"
            for err in schema_errors
        ]
        raise ProfileValidationError(formatted)

    sem_errors = _semantic_checks(profile)
    if sem_errors:
        raise ProfileValidationError(sem_errors)


def load_profile(
    path: Path | str,
    schema_path: Path | str | None = None,
) -> dict[str, Any]:
    """Load and validate a GPU profile JSON file."""
    profile_path = Path(path)
    with profile_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    validate_profile(data, schema_path=schema_path)
    return data


def _main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m virtualgpu.profile <profile.json>", file=sys.stderr)
        return 2
    try:
        profile = load_profile(argv[1])
    except (OSError, json.JSONDecodeError, ProfileValidationError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    ident = profile["identity"]
    print(f"OK: {ident['sku']} ({ident['arch']}, schema {profile['schema_version']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
