"""Score predictions against one measured GEMM case and cache regime."""

import argparse
import itertools
import json
import math
from pathlib import Path
import random
import statistics


def sign(value):
    return (value > 0) - (value < 0)


def kendall_tau_b(measured, predicted):
    shared = sorted(measured.keys() & predicted.keys())
    concordance = measured_pairs = predicted_pairs = 0
    for a, b in itertools.combinations(shared, 2):
        observed = sign(measured[b] - measured[a])
        forecast = sign(predicted[b] - predicted[a])
        concordance += observed * forecast
        measured_pairs += bool(observed)
        predicted_pairs += bool(forecast)
    denominator = math.sqrt(measured_pairs * predicted_pairs)
    return concordance / denominator if denominator else None


def observed_pair(a, b, samples, seed):
    by_session = {}
    for session, block in sorted(a.keys() & b.keys()):
        key = session, block
        by_session.setdefault(session, []).append(math.log(a[key] / b[key]))
    if sum(map(len, by_session.values())) < 2:
        return "unavailable", None
    rng = random.Random(seed)
    estimates = sorted(
        statistics.median(value for values in by_session.values()
                          for value in rng.choices(values, k=len(values)))
        for _ in range(samples)
    )
    low, high = estimates[int(0.025 * (samples - 1))], estimates[math.ceil(0.975 * (samples - 1))]
    margin = math.log(1.05)
    if high < -margin:
        return "a_faster", (low, high)
    if low > margin:
        return "b_faster", (low, high)
    if low >= -margin and high <= margin:
        return "indistinguishable", (low, high)
    return "unresolved", (low, high)


def predicted_pair(a, b):
    # JSON uses string keys for the draw IDs.
    if a.keys() != b.keys() or "0" not in a:
        return "unavailable"
    directions = {sign(a[draw] - b[draw]) for draw in a}
    if directions == {-1}:
        return "a_faster"
    if directions == {1}:
        return "b_faster"
    return "indistinguishable"


def score(timings, records, k=5, samples=1000, seed=17):
    if not timings or k < 1 or samples < 2:
        raise ValueError("Require timings, positive k, and at least two bootstrap samples")
    measured = {key: statistics.median(values.values()) for key, values in timings.items()}
    predicted = {key: row["nominal_latency_ns"] for key, row in records.items()
                 if row["status"] == "ok" and key in measured}
    shortlist = sorted(predicted, key=lambda key: (predicted[key], key))[:k]
    oracle = min(measured.values())
    regret = min((measured[key] / oracle - 1 for key in shortlist), default=None)
    pairs = []
    confusion = {}
    for a, b in itertools.combinations(sorted(measured), 2):
        observed, interval = observed_pair(timings[a], timings[b], samples, seed)
        forecast = "unavailable"
        if a in predicted and b in predicted:
            forecast = predicted_pair(records[a].get("draws", {}), records[b].get("draws", {}))
        pairs.append({
            "a": a, "b": b, "observed": observed, "predicted": forecast,
            "observed_log_ratio_ci95": interval,
        })
        counts = confusion.setdefault(observed, {})
        counts[forecast] = counts.get(forecast, 0) + 1
    return {
        "kendall_tau_b": kendall_tau_b(measured, predicted),
        "k": k,
        "shortlist": shortlist,
        "top_k_hit": any(measured[key] == oracle for key in shortlist),
        "top_k_regret": regret,
        "scored_configs": len(predicted),
        "measured_configs": len(measured),
        "configs": {
            key: {
                "measured_median_ns": measured[key],
                "nominal_latency_ns": predicted.get(key),
                "prediction_status": records.get(key, {}).get("status", "missing"),
            }
            for key in sorted(measured)
        },
        "pair_confusion": confusion,
        "pairs": pairs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="JSON array of model or baseline records")
    parser.add_argument("measurements", type=Path, nargs="+", help="Raw runner session reports")
    parser.add_argument("--profile-revision", required=True)
    parser.add_argument("--cache", choices=("warm", "cold"), required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    sessions = [json.loads(path.read_text()) for path in args.measurements]
    case_id = sessions[0]["case_id"]
    identity = ("case_id", "gpu", "kernel_sha256", "manifest_sha256")
    timings = {}
    for session in sessions:
        if any(session[field] != sessions[0][field] for field in identity):
            parser.error("Measurements must use the same case, GPU, kernel, and manifest")
        protocol_fields = ("clock_policy", "timer", "l2_bytes", "eviction_bytes")
        if any(session["protocol"][field] != sessions[0]["protocol"][field]
               for field in protocol_fields):
            parser.error("Measurements must use the same timing and cache preparation protocol")
        for sample in session["samples"]:
            if sample["cache"] == args.cache:
                block = session["session_id"], sample["block"]
                per_config = timings.setdefault(sample["config_id"], {})
                if block in per_config:
                    parser.error("Duplicate timing sample")
                per_config[block] = sample["latency_ns"]
    predictions = {}
    for row in json.loads(args.predictions.read_text()):
        if row["case_id"] == case_id and row["profile_revision"] == args.profile_revision:
            key = row["config_id"]
            if key in predictions:
                parser.error("Duplicate prediction record")
            predictions[key] = row
    report = score(timings, predictions, args.top_k)
    report.update(case_id=case_id, cache=args.cache, profile_revision=args.profile_revision)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
