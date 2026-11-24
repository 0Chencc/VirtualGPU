"""Score the completed development sweep against independent baselines."""

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics

from validation.baseline import predict as roofline
from validation.neusight import fit, leave_one_case_out, tile_features
from validation.score import PRACTICAL_MARGIN, check_artifact, pair_observations, score

CACHE_STATES = ("warm", "cold")


@dataclass
class CaseMeasurements:
    artifact_rows: dict
    timings: dict
    statuses: dict
    source_report: dict
    matched_config_sessions: int


def load_case(case, results, manifest_sha, sessions, cold_results=None):
    key = case["case_id"]
    config_ids = set(case["config_ids"])
    rows, irs, statuses = {}, {}, {}
    timings = {cache: {} for cache in CACHE_STATES}
    session_ids = set()
    hardware = None
    protocols = {}
    sources = ((results, CACHE_STATES),) if cold_results is None else (
        (results, ("warm",)), (cold_results, ("cold",)),
    )
    files = ((directory / f"{key}-session-{index}.json", states)
             for directory, states in sources for index in range(1, sessions + 1))
    for path, states in files:
        report = json.loads(path.read_text())
        identity = (report["case_id"], report["split"], report["shape"], report["manifest_sha256"])
        if identity != (key, "development", case["shape"], manifest_sha):
            raise ValueError(f"{path}: report does not match the development manifest")
        if set(report["configs"]) != config_ids or report["session_id"] in session_ids:
            raise ValueError(f"{path}: incomplete or duplicate session")
        protocol = report["protocol"]
        current = report["gpu"], report["software"], report["kernel_sha256"]
        if hardware is not None and hardware != current:
            raise ValueError(f"{path}: runner environment changed across sessions")
        hardware = current
        if not set(states) <= set(protocol.get("cache_states", CACHE_STATES)):
            raise ValueError(f"{path}: missing requested cache states")
        run_protocol = tuple(protocol[field] for field in ("clock_policy", "timer", "l2_bytes", "blocks"))
        if "cold" in states:
            if protocol.get("cold_flush") != "read_cg" or protocol["eviction_bytes"] < 2 * protocol["l2_bytes"]:
                raise ValueError(f"{path}: cold timings require read-only L2 eviction")
            run_protocol += (protocol["eviction_bytes"],)
        if protocols.setdefault(states, run_protocol) != run_protocol:
            raise ValueError(f"{path}: runner protocol changed across sessions")
        session_ids.add(report["session_id"])
        for config_id, row in report["configs"].items():
            if statuses.setdefault(config_id, row["status"]) != row["status"]:
                raise ValueError(f"{path}: inconsistent status for {config_id}")
            if row["status"] != "ok":
                continue
            if config_id not in irs:
                ir_path = results / "aot" / key / f"{config_id}.json"
                irs[config_id] = json.loads(ir_path.read_text())
            verdict = check_artifact(row, irs[config_id])
            if verdict["status"] != "matched":
                raise ValueError(f"{path}: {config_id}: {verdict}")
            rows[config_id] = row
        observed = set()
        for sample in report["samples"]:
            cache, config_id, block = sample["cache"], sample["config_id"], sample["block"]
            if cache not in states:
                continue
            trial = (cache, config_id, block)
            if trial in observed:
                raise ValueError(f"{path}: duplicate trial {trial}")
            observed.add(trial)
            timings[cache].setdefault(config_id, {})[report["session_id"], block] = sample["latency_ns"]
        expected = {(cache, config_id, block) for cache in states for config_id in rows
                    for block in range(protocol["blocks"])}
        if observed != expected:
            raise ValueError(f"{path}: incomplete randomized blocks")
    return CaseMeasurements(rows, timings, dict(Counter(statuses.values())), report,
                            len(rows) * len(session_ids))


def build_examples(cases, measurements, configs, profile):
    examples = []
    for case in cases:
        key = case["case_id"]
        data = measurements[key]
        for cache, timings in data.timings.items():
            for config_id, samples in timings.items():
                try:
                    tile = tile_features(
                        case["shape"], {**configs[config_id], **data.artifact_rows[config_id]},
                        profile, cache,
                    )
                except ValueError:
                    continue
                examples.append({
                    "case_id": key, "cache": cache, "config_id": config_id,
                    "split": case["split"], "tile": tile,
                    "measured_ns": statistics.median(samples.values()),
                })
    return examples


def advantage(core, baseline):
    tau = (core["kendall_tau_b"], baseline["kendall_tau_b"])
    regret = (baseline["top_k_regret"], core["top_k_regret"])
    return {
        "tau_b_gain": tau[0] - tau[1] if None not in tau else None,
        "top_5_regret_reduction": regret[0] - regret[1] if None not in regret else None,
    }


def core_records(predictions, key, cache, timings):
    records = predictions["cases"][key][cache]
    if not timings.keys() <= records.keys():
        raise ValueError(f"{key} {cache}: missing core prediction statuses")
    draw_ids = {str(i) for i in range(predictions["perturbation_draws"] + 1)}
    for config_id, row in records.items():
        if row["status"] == "ok":
            draws = row["draws"]
            if (draws.keys() != draw_ids or draws["0"] != row["nominal_latency_ns"]
                    or any(not math.isfinite(value) or value <= 0 for value in draws.values())):
                raise ValueError(f"{key} {cache} {config_id}: invalid shared draw set")
    return records


def combined_pair_confusion(cases):
    counts = {}
    for case in cases:
        for observed, row in case["pair_confusion"].items():
            counts.setdefault(observed, Counter()).update(row)
    return counts


def summarize(reports, has_core):
    summary = {}
    methods = ("roofline_wave", "neusight_style", "core") if has_core else (
        "roofline_wave", "neusight_style",
    )
    for cache in CACHE_STATES:
        summary[cache] = {}
        for method in methods:
            cases = [entry["cache"][cache][method] for entry in reports.values()]
            tau = [row["kendall_tau_b"] for row in cases if row["kendall_tau_b"] is not None]
            regret = [row["top_k_regret"] for row in cases if row["top_k_regret"] is not None]
            summary[cache][method] = {
                "cases_with_tau_b": len(tau),
                "median_tau_b": statistics.median(tau) if tau else None,
                "cases_with_top_5_regret": len(regret),
                "mean_top_5_regret": statistics.mean(regret) if regret else None,
                "measured_configs": sum(row["measured_configs"] for row in cases),
                "scored_configs": sum(row["scored_configs"] for row in cases),
                "pair_confusion": combined_pair_confusion(cases),
            }
    return summary


def development_gates(summary):
    """Apply the criteria in docs/design.md separately to both cache regimes."""
    gates = {}
    for cache, methods in summary.items():
        core = methods["core"]
        tau, regret = core["median_tau_b"], core["mean_top_5_regret"]
        checks = {}
        for baseline in ("roofline_wave", "neusight_style"):
            other = methods[baseline]
            rank = other["median_tau_b"]
            shortlist = other["mean_top_5_regret"]
            gain = tau - rank if tau is not None and rank is not None else None
            excess = regret - shortlist if regret is not None and shortlist is not None else None
            checks[baseline] = {
                "median_tau_b_gain": gain,
                "mean_top_5_regret_excess": excess,
                "ranking_pass": gain is not None and gain >= 0.05 - 1e-12,
                "non_inferiority_pass": excess is not None and excess <= 0.005 + 1e-12,
            }
        absolute = regret is not None and regret <= 0.01 + 1e-12
        gates[cache] = {
            "absolute_regret_pass": absolute,
            "baselines": checks,
            "pass": absolute and all(row["ranking_pass"] and row["non_inferiority_pass"]
                                     for row in checks.values()),
        }
    return gates


def evaluate(suite, results, profile, sessions, manifest_sha, core_predictions=None, cold_results=None):
    cases = [case for case in suite["cases"] if case["split"] == "development"]
    configs = {cfg["config_id"]: cfg for cfg in suite["configs"]}
    revision = profile["provenance"]["profile_revision"]
    if core_predictions is not None and (
        core_predictions["profile_revision"] != revision
        or core_predictions["manifest_sha256"] != manifest_sha
        or core_predictions["perturbation_draws"] < 1
    ):
        raise ValueError("Core predictions must match the manifest and profile and include perturbations")
    measurements = {
        case["case_id"]: load_case(case, results, manifest_sha, sessions, cold_results)
        for case in cases
    }
    examples = build_examples(cases, measurements, configs, profile)
    fits = leave_one_case_out(examples)
    learned = {}
    for row in examples:
        key, cache = row["case_id"], row["cache"]
        learned.setdefault((key, cache), {})[row["config_id"]] = {
            "status": "ok", "nominal_latency_ns": fits[key].predict_ns(row["tile"]),
        }
    reports = {}
    for key, data in measurements.items():
        case_result = {
            "candidate_status_counts": data.statuses,
            "matched_config_sessions": data.matched_config_sessions,
            "cache": {},
        }
        baseline = {row["config_id"]: row for row in roofline(data.source_report, profile)}
        for cache, timings in data.timings.items():
            observations = pair_observations(timings)
            scores = {
                "roofline_wave": score(timings, baseline, nominal_pair_margin=PRACTICAL_MARGIN,
                                       observations=observations),
                "neusight_style": score(
                    timings, learned.get((key, cache), {}), nominal_pair_margin=PRACTICAL_MARGIN,
                    observations=observations,
                ),
            }
            if core_predictions is not None:
                scores["core"] = score(
                    timings, core_records(core_predictions, key, cache, timings),
                    observations=observations,
                )
                scores["core_advantage"] = {
                    name: advantage(scores["core"], scores[name])
                    for name in ("roofline_wave", "neusight_style")
                }
            case_result["cache"][cache] = scores
        reports[key] = case_result
    summary = summarize(reports, core_predictions is not None)
    return {
        "profile_revision": revision,
        "manifest_sha256": manifest_sha,
        "sessions_per_case": sessions,
        "cold_flush": "read_cg",
        "neusight_evaluation": "leave_one_development_case_out",
        "neusight_full_development_fit": fit(examples).to_json(),
        "baseline_pair_margin": PRACTICAL_MARGIN,
        "model_pair_margin": PRACTICAL_MARGIN,
        "core_perturbation_draws": (core_predictions["perturbation_draws"]
                                    if core_predictions is not None else None),
        "summary": summary,
        "development_gates": development_gates(summary) if core_predictions is not None else None,
        "cases": reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("profile", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--cold-results", type=Path, help="Directory of replacement read-only cold sessions")
    parser.add_argument("--core-predictions", type=Path,
                        help="Prediction records grouped by case, cache state, and config ID")
    args = parser.parse_args()
    if args.sessions < 1:
        parser.error("Use a positive session count")
    raw = args.manifest.read_bytes()
    profile = json.loads(args.profile.read_text())
    core = json.loads(args.core_predictions.read_text()) if args.core_predictions else None
    report = evaluate(json.loads(raw), args.results, profile, args.sessions,
                      hashlib.sha256(raw).hexdigest(), core, args.cold_results)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
