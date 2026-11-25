"""Score predictions against one measured GEMM case and cache regime."""

import argparse
import itertools
import json
import math
from pathlib import Path
import random
import stat
import statistics

from virtualgpu.frontend.compiler import ARTIFACT_HASH_VERSION

from validation.holdout import read_committed_audit, read_verified_report, require_release
from validation.provenance import read_core_revision, require_formal_build

PRACTICAL_MARGIN = 0.05
PRACTICAL_LOG_MARGIN = math.log(1 + PRACTICAL_MARGIN)


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


def median_ci95(groups, samples=1000, seed=17):
    rng = random.Random(seed)
    estimates = sorted(
        statistics.median(value for group in groups
                          for value in rng.choices(group, k=len(group)))
        for _ in range(samples)
    )
    last = samples - 1
    return estimates[int(0.025 * last)], estimates[math.ceil(0.975 * last)]


def observed_pair(a, b, samples, seed):
    by_session = {}
    for session, block in sorted(a.keys() & b.keys()):
        key = session, block
        by_session.setdefault(session, []).append(math.log(a[key] / b[key]))
    if sum(map(len, by_session.values())) < 2:
        return "unavailable", None
    low, high = median_ci95(list(by_session.values()), samples, seed)
    margin = PRACTICAL_LOG_MARGIN
    if high < -margin:
        return "a_faster", (low, high)
    if low > margin:
        return "b_faster", (low, high)
    if low >= -margin and high <= margin:
        return "indistinguishable", (low, high)
    return "unresolved", (low, high)


def predicted_pair(a, b, log_margin=PRACTICAL_LOG_MARGIN):
    # JSON uses string keys for the draw IDs.
    if a.keys() != b.keys() or "0" not in a:
        return "unavailable"
    margin_ratio = math.exp(log_margin)
    if all(a[draw] < b[draw] and a[draw] * margin_ratio <= b[draw] for draw in a):
        return "a_faster"
    if all(b[draw] < a[draw] and b[draw] * margin_ratio <= a[draw] for draw in a):
        return "b_faster"
    return "indistinguishable"


def check_artifact(jit, ir):
    """Check the emitted code and resource contract before using timings."""
    metadata = ir["metadata"]
    aot = metadata["toolchain"]
    toolchain = jit.get("toolchain", {})
    expected_options = metadata["config"]
    resources = ir["resources"]
    expected = {
        "artifact_hash_version": ARTIFACT_HASH_VERSION,
        "ptx_sha256": aot.get("ptx_sha256"),
        "grid": ir["launch"]["grid"],
        "registers_per_thread": resources["registers_per_thread"],
        "shared_bytes": resources["static_smem_bytes"] + ir["launch"]["dynamic_smem_bytes"],
        **{key: aot.get(key) for key in (
            "compiler", "compiler_version", "assembler", "assembler_version", "target_sm_compiled",
        )},
        "enable_fp_fusion": expected_options.get("enable_fp_fusion", True),
        "ptx_options": expected_options.get("ptx_options") or None,
        "disable_ptxas_opt": False,
    }
    actual = {**toolchain, **jit, "grid": [*jit["grid"], 1, 1][:3]}
    differences = [key for key, value in expected.items()
                   if key not in actual or actual[key] != value]
    if aot.get("artifact_hash_version") != ARTIFACT_HASH_VERSION:
        differences.append("aot_artifact_hash_version")
    if not aot.get("ptx_sha256"):
        differences.append("missing_aot_hash")
    ttgir = aot.get("ttgir_sha256")
    return {
        "status": "matched" if not differences else "mismatch",
        "differences": differences,
        "ttgir_match": bool(ttgir and ttgir == jit.get("ttgir_sha256")),
    }


def pair_observations(timings, samples=1000, seed=17):
    return {(a, b): observed_pair(timings[a], timings[b], samples, seed)
            for a, b in itertools.combinations(sorted(timings), 2)}


def score(timings, records, k=5, samples=1000, seed=17, nominal_pair_margin=None,
          observations=None):
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
    if observations is None:
        observations = pair_observations(timings, samples, seed)
    for (a, b), (observed, interval) in observations.items():
        forecast = "unavailable"
        if a in predicted and b in predicted:
            if nominal_pair_margin is None:
                forecast = predicted_pair(records[a].get("draws", {}), records[b].get("draws", {}))
            elif max(predicted[a], predicted[b]) <= min(predicted[a], predicted[b]) * (1 + nominal_pair_margin):
                forecast = "indistinguishable"
            else:
                forecast = "a_faster" if predicted[a] < predicted[b] else "b_faster"
        pairs.append({
            "a": a, "b": b, "observed": observed, "predicted": forecast,
            "observed_log_ratio_ci95": interval,
        })
        counts = confusion.setdefault(observed, {})
        counts[forecast] = counts.get(forecast, 0) + 1
    comparison = {}
    for key in sorted(measured):
        groups = {}
        for (session, _), latency in sorted(timings[key].items()):
            groups.setdefault(session, []).append(latency)
        median = measured[key]
        interval = median_ci95(list(groups.values()), samples, seed) if len(timings[key]) >= 2 else None
        comparison[key] = {
            "measured_median_ns": median,
            "measured_median_ci95_ns": interval,
            "relative_ci95_half_width": (interval[1] - interval[0]) / (2 * median) if interval else None,
            "gap_to_measured_best": median / oracle - 1,
            "nominal_latency_ns": predicted.get(key),
            "prediction_status": records.get(key, {}).get("status", "missing"),
        }
    return {
        "kendall_tau_b": kendall_tau_b(measured, predicted),
        "k": k,
        "shortlist": shortlist,
        "top_k_hit": any(measured[key] == oracle for key in shortlist),
        "top_k_regret": regret,
        "scored_configs": len(predicted),
        "measured_configs": len(measured),
        "configs": comparison,
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
    parser.add_argument("--sealed-audit", type=Path, help="Committed digest index for sealed holdout labels")
    parser.add_argument("--prediction-sha256", help="Independently archived prediction digest for holdout release")
    gate = parser.add_mutually_exclusive_group(required=True)
    gate.add_argument("--aot", action="append", default=[], metavar="CONFIG_ID=KERNEL_IR_JSON",
                      help="Require matching AOT output for every scored configuration")
    gate.add_argument("--unchecked", action="store_true",
                      help="Skip the AOT comparison for current-version JIT reports")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--core-revision", type=read_core_revision, help="Expected core SHA or revision file")
    mode.add_argument("--diagnostic", action="store_true", help="Allow unverified builds or baseline records")
    args = parser.parse_args()
    if args.core_revision and args.unchecked:
        parser.error("Formal scoring requires --aot")
    seal = None
    if args.sealed_audit:
        if not args.core_revision or not args.prediction_sha256:
            parser.error("Holdout scoring requires the phase1-final revision and archived prediction digest")
        try:
            input_predictions = require_release(args.core_revision, args.predictions, args.prediction_sha256)
            seal = read_committed_audit(args.sealed_audit)
        except ValueError as error:
            parser.error(str(error))
    else:
        if args.prediction_sha256 or any(stat.S_IMODE(path.stat().st_mode) == 0 for path in args.measurements):
            parser.error("Sealed labels require --sealed-audit")
        input_predictions = json.loads(args.predictions.read_text())
    if args.core_revision:
        try:
            for row in input_predictions:
                require_formal_build(row.get("build"), args.core_revision,
                                     f"{row['case_id']} {row['config_id']}")
        except ValueError as error:
            parser.error(str(error))
    artifacts = {}
    for entry in args.aot:
        key, sep, path = entry.partition("=")
        if not sep or not key or not path or key in artifacts:
            parser.error("Supply one --aot CONFIG_ID=KERNEL_IR_JSON per configuration")
        artifacts[key] = json.loads(Path(path).read_text())
    try:
        sessions = ([read_verified_report(path, seal) for path in args.measurements] if seal is not None
                    else [json.loads(path.read_text()) for path in args.measurements])
    except ValueError as error:
        parser.error(str(error))
    if any(session.get("split") == "holdout" for session in sessions) and seal is None:
        parser.error("Holdout scoring requires verified sealed reports")
    case_id = sessions[0]["case_id"]
    identity = ("case_id", "gpu", "kernel_sha256", "manifest_sha256")
    timings = {}
    artifact_checks = {}
    for session in sessions:
        if any(session[field] != sessions[0][field] for field in identity):
            parser.error("Measurements must use the same case, GPU, kernel, and manifest")
        protocol_fields = ("clock_policy", "timer", "l2_bytes", "eviction_bytes")
        if any(session["protocol"][field] != sessions[0]["protocol"][field]
               for field in protocol_fields):
            parser.error("Measurements must use the same timing and cache preparation protocol")
        for sample in session["samples"]:
            if sample["cache"] == args.cache:
                key = sample["config_id"]
                checks = artifact_checks.setdefault(session["session_id"], {})
                if key not in checks:
                    jit = session["configs"][key]
                    if key in artifacts:
                        checks[key] = check_artifact(jit, artifacts[key])
                    elif jit.get("artifact_hash_version") != ARTIFACT_HASH_VERSION:
                        checks[key] = {"status": "mismatch", "differences": ["artifact_hash_version"]}
                    elif not jit.get("ptx_sha256"):
                        checks[key] = {"status": "mismatch", "differences": ["missing_jit_hash"]}
                    else:
                        checks[key] = {"status": "unchecked" if args.unchecked else "missing_aot"}
                if checks[key]["status"] not in ("matched", "unchecked"):
                    continue
                block = session["session_id"], sample["block"]
                per_config = timings.setdefault(key, {})
                if block in per_config:
                    parser.error("Duplicate timing sample")
                per_config[block] = sample["latency_ns"]
    predictions = {}
    for row in input_predictions:
        if row["case_id"] == case_id and row["profile_revision"] == args.profile_revision:
            key = row["config_id"]
            if key in predictions:
                parser.error("Duplicate prediction record")
            predictions[key] = row
    if args.core_revision and not predictions:
        parser.error("No core predictions match the requested case and profile")
    report = score(timings, predictions, args.top_k) if timings else {"error": "No eligible timings"}
    report.update(case_id=case_id, cache=args.cache, profile_revision=args.profile_revision,
                  result_class="formal" if args.core_revision else "diagnostic",
                  expected_core_revision=args.core_revision,
                  artifact_gate="required" if artifacts else "unchecked",
                  artifact_checks=artifact_checks)
    print(json.dumps(report, indent=2))
    if not timings:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
