"""Bracket a frozen GB10 MLP process with the registered P3a reference."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROTOCOL = "gb10-mlp-plateau-p3a-v1"
GPU_LOCK = Path("/tmp/virtualgpu-gpu.lock")


def require_lock():
    """Check the exclusive flock inherited from the capture command."""
    for fd in Path("/proc/self/fd").iterdir():
        try:
            if fd.resolve(strict=True) != GPU_LOCK.resolve(strict=True):
                continue
            info = (Path("/proc/self/fdinfo") / fd.name).read_text()
        except FileNotFoundError:
            continue
        if any(
            line.startswith("lock:") and "FLOCK  ADVISORY  WRITE" in line
            for line in info.splitlines()
        ):
            return
    raise RuntimeError(f"GB10 DRAM references require an inherited exclusive flock on {GPU_LOCK}")


def current_uuid():
    value = subprocess.check_output(
        ["nvidia-smi", "-i", "0", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    if not value.startswith("GPU-") or "\n" in value:
        raise ValueError("Expected one CUDA device UUID")
    return value


def freeze_options(device, modes):
    if (
        device["name"],
        device["major"],
        device["minor"],
        device["sm_count"],
        device["integrated"],
    ) != ("NVIDIA GB10", 12, 1, 48, True) or "mlp" not in modes:
        raise ValueError("DRAM references require GB10 and its full MLP sweep")
    rev = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"], text=True
    ).strip()
    return {"protocol": PROTOCOL, "device_uuid": current_uuid(), "source_revision": rev}


def block_command(mlp_command):
    return [sys.executable, str(Path(__file__).resolve()), *mlp_command]


def run_block(mlp_command):
    """Run all three processes under the caller's lock and one host trace."""
    samples = mlp_command[mlp_command.index("--samples") + 1]
    probe = [
        mlp_command[0],
        "--mode",
        "dram-probe",
        "--device",
        "0",
        "--samples",
        samples,
        "--warmup",
        "1",
    ]

    def run_probe():
        result = subprocess.run(probe, capture_output=True, text=True, check=False)
        if result.returncode:
            return {
                "error": result.stderr,
                "returncode": result.returncode,
                "stdout": result.stdout,
            }
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return {"error": "invalid probe JSON", "stdout": result.stdout}

    before = run_probe()
    workload = json.loads(subprocess.check_output(mlp_command, text=True))
    return {"before": before, "workload": workload, "after": run_probe()}


def probe_samples(raw):
    if (raw["mode"], raw["sm_count"], raw["samples"], raw["bytes"]) != (
        "dram-probe",
        48,
        15,
        256 * 1024**2,
    ):
        raise ValueError("DRAM reference probe differs from the registered protocol")
    (p3a,) = [point for point in raw["candidates"] if point["name"] == "P3a"]
    if (
        p3a["issue"],
        p3a["ctas_per_sm"],
        p3a["threads_per_cta"],
        p3a["loads_in_flight_per_thread"],
        p3a["bytes_in_flight_per_cta"],
    ) != ("ld_global_v4", 4, 256, 4, 16384) or p3a["occupancy"] < 4:
        raise ValueError("DRAM reference is not the registered P3a geometry")
    values = p3a["gbps_samples"]
    if len(values) != 15 or not all(
        type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values
    ):
        raise ValueError("Expected 15 finite positive P3a samples")
    utc = datetime.fromtimestamp(raw["unix_ms_end"] / 1000, timezone.utc).isoformat()
    return {"utc": utc, "gbps_samples": values}


def attach(profile, run_dir, manifest, digest, state):
    """Publish complete receipts and only pair rates from eligible blocks."""
    from virtualgpu.profile import (
        GB10_REFERENCE_GEOMETRIES,
        dram_mlp_geometry,
        dram_reference_value,
    )

    options = manifest.get("dram_reference")
    if options is None:
        if "dram_reference_manifest_sha256" in state:
            raise ValueError("Opt-in DRAM reference freeze is missing")
        return
    if options["protocol"] != PROTOCOL:
        raise ValueError("Unknown DRAM reference protocol")
    if state.get("dram_reference_manifest_sha256") != digest:
        raise ValueError("DRAM reference capture does not match its opt-in freeze")

    curve = profile["measured"]["memory"]["dram_bw_by_bytes_in_flight"]
    targets = {dram_mlp_geometry(p): p for p in curve}
    if not GB10_REFERENCE_GEOMETRIES <= targets.keys():
        raise ValueError("Profile is missing a registered GB10 plateau point")

    references = {}
    for rnd in state["rounds"]:
        result = rnd["results"].get("mlp")
        if result is None:
            continue
        receipt = result["dram_reference"]
        block_bytes = (run_dir / receipt["raw"]).read_bytes()
        if hashlib.sha256(block_bytes).hexdigest() != receipt["sha256"]:
            raise ValueError("DRAM reference block digest differs from capture state")
        block = json.loads(block_bytes)
        workload = json.loads((run_dir / result["raw"]).read_bytes())
        if block["workload"] != workload:
            raise ValueError("DRAM reference block must contain the recorded MLP process")
        if any("error" in block[side] for side in ("before", "after")):
            continue  # The raw receipt retains an incomplete pair for audit.
        before, after = (probe_samples(block[side]) for side in ("before", "after"))
        value, drift = dram_reference_value(before["gbps_samples"], after["gbps_samples"])
        status = "void" if result["status"] != "valid" else ("drift" if drift else "valid")
        block_id = rnd["round"]
        ref_id = f"{digest}:{block_id}"
        references[ref_id] = {
            "device": "NVIDIA GB10",
            "device_uuid": options["device_uuid"],
            "session_id": digest,
            "block_id": block_id,
            "probe": "dram-saturation-probe/P3a",
            "protocol": PROTOCOL,
            "issue_mode": "ld_global_v4",
            "active_sms": 48,
            "cache_preparation": "cold_read_only_l2_flush",
            "value_gbps": value,
            "before": before,
            "after": after,
            "source_revision": options["source_revision"],
            "binary_sha256": manifest["binary_sha256"],
            "raw_capture_sha256": receipt["sha256"],
            "status": status,
        }
        if status == "valid":
            for p in workload["dram_bw_by_bytes_in_flight"]:
                key = dram_mlp_geometry(p)
                if key in GB10_REFERENCE_GEOMETRIES:
                    targets[key].setdefault("reference_observations", []).append(
                        {
                            "reference_id": ref_id,
                            "value_gbps": statistics.median(p["gbps_samples"]),
                        }
                    )

    if not references:
        raise ValueError("Opt-in DRAM reference capture has no complete pairs")
    profile["references"] = references
    profile["provenance"]["capture_protocol"]["dram_reference_protocol"] = PROTOCOL


if __name__ == "__main__":
    print(json.dumps(run_block(sys.argv[1:])))
