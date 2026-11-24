"""A roofline-bounded, learned tile baseline for dense GEMM."""

from dataclasses import dataclass
import math

import numpy as np

from validation.baseline import gemm_work, residency


@dataclass(frozen=True)
class TileFeatures:
    values: tuple
    wave_floor_ns: float
    waves: int
    launch_ns: float

    @property
    def kernel_floor_ns(self):
        return self.launch_ns + self.waves * self.wave_floor_ns


def tile_features(shape, config, profile, cache):
    """Build target-independent features and a target-specific latency floor."""
    specs = profile["specs"]
    measured = profile["measured"]
    mma_shapes = config["mma_shapes"]
    rates = measured["compute"]["mma_tflops_per_sm_by_warps"]
    curve = rates.get(f"mma.sync:f16:f16:f32:{mma_shapes[0]}") if len(mma_shapes) == 1 else None
    capacity = specs["sm_count"] * residency(config, specs)
    if not curve or capacity <= 0:
        raise ValueError("Missing MMA rate or residency")
    count, flops, requested = gemm_work(shape, config["tile"])
    waves = math.ceil(count / capacity)
    memory = measured["memory"]
    dram_bw = memory["dram_bw_read_gbps"]["value"] * 1e9
    l2_bw = memory["l2_bw_read_gbps"]["value"] * 1e9
    l2_capacity = specs["l2_size_bytes"]
    fp_rate = specs["sm_count"] * max(row["tflops_per_sm"]["value"] for row in curve) * 1e12
    m, n, k = shape
    working_set = 2 * (m * k + k * n + m * n)
    warm = cache == "warm"
    compulsory = max(working_set - l2_capacity, 0) if warm else working_set
    # Aggregate the physical demand, then amortize it over the analytical waves.
    # Requested tile traffic is served at the L2 ceiling and only compulsory
    # traffic is charged to DRAM. The regressor learns extra latency above this floor.
    compute_ns = count * flops / fp_rate * 1e9
    l2_ns = count * requested / l2_bw * 1e9
    dram_ns = compulsory / dram_bw * 1e9
    wave_floor_ns = max(compute_ns, l2_ns, dram_ns) / waves
    bm, bn, bk = config["tile"]
    values = (
        math.log2(flops / 1024), math.log2(requested / 1024),
        math.log2(working_set / l2_capacity), count / (waves * capacity),
        math.log2(capacity / specs["sm_count"]), math.log2(compute_ns / l2_ns),
        math.log2(bm), math.log2(bn), math.log2(bk), math.log2(config["num_warps"]),
        math.log2(config["num_stages"]), math.log2(config["group_m"]), float(warm),
    )
    launch_ns = measured["overhead"]["launch_overhead_us"]["value"] * 1e3
    return TileFeatures(values, wave_floor_ns, waves, launch_ns)


@dataclass(frozen=True)
class EfficiencyFit:
    center: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: float

    def predict_ns(self, tile):
        normalized = (tile.values - self.center) / self.scale
        log_slowdown = max(0.0, self.intercept + normalized @ self.weights)
        return tile.launch_ns + tile.waves * tile.wave_floor_ns * math.exp(log_slowdown)

    def to_json(self):
        return {
            "center": self.center.tolist(), "scale": self.scale.tolist(),
            "weights": self.weights.tolist(), "intercept": self.intercept,
        }


def fit(training, penalty=1.0):
    """Fit log inverse efficiency using matched development labels only."""
    if not training or penalty <= 0:
        raise ValueError("Training examples and a positive ridge penalty are required")
    if any(row["split"] != "development" for row in training):
        raise ValueError("Training accepts only development labels")
    matrix = np.asarray([row["tile"].values for row in training], dtype=float)
    center = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale == 0] = 1.0
    normalized = (matrix - center) / scale
    target = []
    for row in training:
        tile = row["tile"]
        wave_ns = (row["measured_ns"] - tile.launch_ns) / tile.waves
        target.append(math.log(max(wave_ns / tile.wave_floor_ns, 1.0)))
    target = np.asarray(target)
    intercept = float(target.mean())
    weights = np.linalg.solve(
        normalized.T @ normalized + penalty * np.eye(matrix.shape[1]),
        normalized.T @ (target - intercept),
    )
    return EfficiencyFit(center, scale, weights, intercept)


def leave_one_case_out(examples):
    """Score each development case without its own timings entering the fit."""
    cases = sorted({row["case_id"] for row in examples})
    return {key: fit([row for row in examples if row["case_id"] != key]) for key in cases}
