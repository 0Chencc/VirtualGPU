"""Independent analytical work and NeuSight-style features for Phase 4 attention."""

from dataclasses import dataclass
import math

from validation.baseline import latency_ns, residency
from validation.neusight import TileFeatures


@dataclass(frozen=True)
class AttentionWork:
    ctas: int
    flops_per_cta: float
    bytes_per_cta: float
    working_set_bytes: int


def attention_work(case, config):
    batch = case["batch"]
    query_heads, kv_heads = case["query_heads"], case["kv_heads"]
    seq, dim = case["seq_len"], case["head_dim"]
    bm, bn = config["block_m"], config["block_n"]
    query_tiles = math.ceil(seq / bm)
    prefixes = [min(seq, (tile + 1) * bm) if case["causal"] else seq
                for tile in range(query_tiles)]
    key_tiles = sum(math.ceil(prefix / bn) for prefix in prefixes)
    ctas = batch * query_heads * query_tiles
    # Each query row transfers FP16 Q and O and one FP32 LSE.
    query_bytes = seq * (4 * dim + 4)
    kv_bytes = 4 * dim * sum(prefixes)
    working_set = batch * (query_heads * query_bytes + kv_heads * seq * 4 * dim)
    return AttentionWork(ctas, 4 * bm * bn * dim * key_tiles / query_tiles,
                         (query_bytes + kv_bytes) / query_tiles, working_set)


def target_rates(config, profile):
    specs, measured = profile["specs"], profile["measured"]
    shapes = config["mma_shapes"]
    rates = measured["compute"]["mma_tflops_per_sm_by_warps"]
    curve = rates.get(f"mma.sync:f16:f16:f32:{shapes[0]}") if len(shapes) == 1 else None
    resident = residency(config, specs)
    if not curve or resident <= 0:
        raise ValueError("Missing FP16 MMA rate or residency")
    flops = specs["sm_count"] * max(row["tflops_per_sm"]["value"] for row in curve) * 1e12
    dram = measured["memory"]["dram_bw_read_gbps"]["value"] * 1e9
    launch = measured["overhead"]["launch_overhead_us"]["value"] * 1e3
    return resident, flops, dram, launch


def roofline(case, config, profile):
    work = attention_work(case, config)
    resident, flops, dram, launch = target_rates(config, profile)
    return latency_ns(cta_count=work.ctas, resident_ctas_per_sm=resident,
                      sm_count=profile["specs"]["sm_count"],
                      flops_per_cta=work.flops_per_cta, bytes_per_cta=work.bytes_per_cta,
                      flops_per_second=flops, bytes_per_second=dram, launch_ns=launch)


def tile_features(case, config, profile, cache):
    work = attention_work(case, config)
    resident, flops, dram, launch = target_rates(config, profile)
    specs = profile["specs"]
    l2 = profile["measured"]["memory"]["l2_bw_read_gbps"]["value"] * 1e9
    capacity = specs["sm_count"] * resident
    waves = math.ceil(work.ctas / capacity)
    warm = cache == "warm"
    footprint = work.working_set_bytes
    l2_capacity = specs["l2_size_bytes"]
    compulsory = max(footprint - l2_capacity, 0) if warm else footprint
    compute_ns = work.ctas * work.flops_per_cta / flops * 1e9
    l2_ns = work.ctas * work.bytes_per_cta / l2 * 1e9
    dram_ns = compulsory / dram * 1e9
    features = (
        math.log2(work.flops_per_cta), math.log2(work.bytes_per_cta),
        math.log2(footprint / l2_capacity), work.ctas / (waves * capacity),
        math.log2(resident), math.log2(compute_ns / l2_ns),
        math.log2(config["block_m"]), math.log2(config["block_n"]),
        math.log2(case["head_dim"]), math.log2(case["query_heads"] / case["kv_heads"]),
        math.log2(config["num_warps"]), math.log2(config["num_stages"]), float(case["causal"]),
        float(warm),
    )
    return TileFeatures(features, max(compute_ns, l2_ns, dram_ns) / waves, waves, launch)
