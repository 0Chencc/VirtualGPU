"""Independent roofline and CTA-wave baseline."""

import argparse
import json
import math
from pathlib import Path


def latency_ns(*, cta_count, resident_ctas_per_sm, sm_count, flops_per_cta,
               bytes_per_cta, flops_per_second, bytes_per_second, launch_ns):
    capacity = sm_count * resident_ctas_per_sm
    if min(cta_count, capacity, flops_per_second, bytes_per_second) <= 0:
        raise ValueError("CTA counts and rates must be positive")
    wave_ns = 1e9 * capacity * max(
        flops_per_cta / flops_per_second, bytes_per_cta / bytes_per_second,
    )
    return launch_ns + math.ceil(cta_count / capacity) * wave_ns


def gemm_work(shape, tile, element_bytes=2):
    """Return grid size, padded FLOPs per CTA, and mean requested bytes."""
    m, n, k = shape
    bm, bn, bk = tile
    gm, gn, gk = (math.ceil(dim / block) for dim, block in zip(shape, tile, strict=True))
    cta_count = gm * gn
    flops_per_cta = 2 * bm * bn * bk * gk
    # Each A slice is read for every N tile, and each B slice for every M tile.
    total_elements = m * k * gn + k * n * gm + m * n
    return cta_count, flops_per_cta, element_bytes * total_elements / cta_count


def residency(config, specs):
    warps = config["num_warps"]
    threads = warps * specs["warp_size"]
    limit = min(specs["max_blocks_per_sm"], specs["max_threads_per_sm"] // threads,
                specs["max_warps_per_sm"] // warps)
    units = specs["register_alloc_unit_regs"]
    registers = config["registers_per_thread"]
    if specs["register_alloc_granularity"] == "warp":
        registers = math.ceil(registers * specs["warp_size"] / units) * units * warps
    else:
        registers = math.ceil(registers * threads / units) * units
    if registers:
        limit = min(limit, specs["registers_per_sm"] // registers)
    shared = config["shared_bytes"] + specs["reserved_shared_mem_per_cta_bytes"]
    shared_unit = specs["shared_mem_alloc_granularity_bytes"]
    if shared:
        full_shared = math.ceil(shared / shared_unit) * shared_unit
        limit = min(limit, specs["shared_mem_per_sm_bytes"] // full_shared)
    return limit


def predict(measurement, profile):
    specs = profile["specs"]
    arch = profile["identity"]["arch"]
    if measurement["gpu"]["arch"] != arch:
        raise ValueError("Profile and measured GPU architecture differ")
    measured_fields = profile["measured"]
    rates = measured_fields["compute"]["mma_tflops_per_sm_by_warps"]
    read_bw = measured_fields["memory"]["dram_bw_read_gbps"]["value"] * 1e9
    launch_ns = measured_fields["overhead"]["launch_overhead_us"]["value"] * 1e3
    revision = profile["provenance"]["profile_revision"]
    output = []
    for key, config in measurement["configs"].items():
        row = {"case_id": measurement["case_id"], "config_id": key, "profile_revision": revision}
        if config["status"] != "ok":
            output.append(dict(row, status="failed", reason=config["reason"]))
            continue
        shapes = config["mma_shapes"]
        curve = rates.get(f"mma.sync:f16:f16:f32:{shapes[0]}") if len(shapes) == 1 else None
        resident = residency(config, specs)
        if not curve or not resident:
            output.append(dict(row, status="unsupported", reason="Missing MMA rate or residency"))
            continue
        count, flops, size = gemm_work(measurement["shape"], config["tile"])
        output.append(dict(row, status="ok", nominal_latency_ns=latency_ns(
            cta_count=count, resident_ctas_per_sm=resident, sm_count=specs["sm_count"],
            flops_per_cta=flops, bytes_per_cta=size, bytes_per_second=read_bw,
            flops_per_second=specs["sm_count"] * 1e12 * max(
                point["tflops_per_sm"]["value"] for point in curve
            ),
            launch_ns=launch_ns,
        )))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("measurement", type=Path)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    measurement = json.loads(args.measurement.read_text())
    print(json.dumps(predict(measurement, profile), indent=2))


if __name__ == "__main__":
    main()
