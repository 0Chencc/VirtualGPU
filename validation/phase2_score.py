"""Verify and score the sealed GB10 transfer cohorts."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import stat

from validation.experiment import development_gates, summarize
from validation.holdout import read_committed_audit, require_release, sha256
from validation.phase2_aot import REVISION, write_json
from validation.phase2_capture import verify_release
from validation.provenance import require_formal_build
from validation.score import PRACTICAL_MARGIN, check_artifact, pair_observations, score

CACHES = ("warm", "cold")
METHODS = ("core", "neusight_style", "roofline_wave")


def read_collection(directory, index, archive, audit):
    files = {}
    for cohort in archive["cohorts"]:
        path = directory / cohort / "manifest.json"
        raw = path.read_bytes()
        if sha256(raw) != index["files"][f"{cohort}/manifest.json"]:
            raise ValueError(f"{cohort}: manifest changed")
        cases = json.loads(raw)["cases"]
        if set(archive["cohorts"][cohort]) != {case["case_id"] for case in cases}:
            raise ValueError(f"{cohort}: prediction census differs from the manifest")
        for case in cases:
            case_id = case["case_id"]
            keys = set(case["config_ids"])
            forecasts = archive["cohorts"][cohort][case_id]
            methods = {"core"} if cohort == "diagnostic" else set(METHODS)
            if set(forecasts) != set(CACHES):
                raise ValueError(f"{case_id}: missing cache predictions")
            for cache, predictions in forecasts.items():
                if set(predictions) != methods or any(set(rows) != keys for rows in predictions.values()):
                    raise ValueError(f"{case_id}/{cache}: incomplete prediction census")
                for key, row in predictions["core"].items():
                    require_formal_build(row.get("build"), REVISION, f"{case_id}/{cache}/{key}")
            for session in (1, 2, 3):
                name = f"{cohort}/sealed/{case_id}-session-{session}.json"
                files[name] = (cohort, case, session)
    if set(files) != set(audit["reports"]):
        raise ValueError("Published predictions and label receipts differ")
    verified = {}
    # Verify every report before parsing any timing labels.
    for name in files:
        path = directory / name
        if path.is_symlink() or stat.S_IMODE(path.stat().st_mode) != 0:
            raise ValueError(f"{name}: report is not sealed")
        path.chmod(0o400)
        try:
            raw = path.read_bytes()
        finally:
            path.chmod(0)
        if sha256(raw) != audit["reports"][name]["sha256"]:
            raise ValueError(f"{name}: report digest mismatch")
        verified[name] = raw
    verified_utc = datetime.now(timezone.utc).isoformat()
    return files, verified, verified_utc


def collect_case_timings(case, sessions, artifact):
    timings = {cache: {} for cache in CACHES}
    statuses = Counter()
    complete = True
    for ordinal, report in sorted(sessions.items()):
        protocol = report["protocol"]
        if (report["shape"] != case["shape"] or report["case_id"] != case["case_id"]
                or report["split"] != case["split"] or protocol["order_seed"] != 100 + ordinal
                or protocol["cache_states"] != list(CACHES) or protocol["blocks"] != 10
                or protocol["thermal_warmup_seconds"] < 60 or protocol["cold_flush"] != "read_cg"
                or protocol["timer"] != "cuda_graph_single_launch"
                or set(report["configs"]) != set(case["config_ids"])):
            raise ValueError(f"{case['case_id']}/{ordinal}: timing protocol changed")
        eligible = set()
        for key, jit in report["configs"].items():
            statuses[jit["status"]] += 1
            if jit["status"] != "ok":
                continue
            matches = check_artifact(jit, artifact(case["case_id"], key))
            if matches["status"] != "matched" or not matches["ttgir_match"]:
                raise ValueError(f"{case['case_id']}/{ordinal}/{key}: AOT/JIT mismatch")
            eligible.add(key)
        blocks = report["complete_blocks"]
        if eligible and protocol["eviction_bytes"] < max(3 * protocol["l2_bytes"], 128 * 1024**2):
            raise ValueError(f"{case['case_id']}/{ordinal}: undersized cold eviction")
        complete = complete and blocks == 10
        observed = set()
        for row in report["samples"]:
            cache, key, block = row["cache"], row["config_id"], row["block"]
            trial = cache, key, block
            latency = row["latency_ns"]
            if trial in observed or not math.isfinite(latency) or latency <= 0:
                raise ValueError(f"{case['case_id']}/{ordinal}: invalid or duplicate sample")
            observed.add(trial)
            timings[cache].setdefault(key, {})[report["session_id"], block] = latency
        if observed != {(cache, key, block) for cache in CACHES for key in eligible for block in range(blocks)}:
            raise ValueError(f"{case['case_id']}/{ordinal}: incomplete randomized block")
    return timings, dict(statuses), complete


def evaluate(cohort, directory, index, archive, collection):
    def artifact(case_id, config_id):
        path = directory / cohort / "aot" / case_id / f"{config_id}.json"
        raw = path.read_bytes()
        if sha256(raw) != index["files"][path.relative_to(directory).as_posix()]:
            raise ValueError(f"{case_id}/{config_id}: artifact digest changed")
        return json.loads(raw)

    out = {}
    for case_id, (case, sessions) in collection.items():
        forecasts = archive["cohorts"][cohort][case_id]
        timings, statuses, complete = collect_case_timings(case, sessions, artifact)
        caches = {}
        for cache in CACHES:
            observed = pair_observations(timings[cache])
            scores = {}
            for method, records in forecasts[cache].items():
                if not timings[cache]:
                    scores[method] = {
                        "kendall_tau_b": None, "top_k_hit": False, "top_k_regret": None,
                        "scored_configs": 0, "measured_configs": 0, "pair_confusion": {},
                    }
                    continue
                row = score(timings[cache], records, observations=observed,
                            nominal_pair_margin=None if method == "core" else PRACTICAL_MARGIN)
                errors = [abs(item["nominal_latency_ns"] / item["measured_median_ns"] - 1)
                          for item in row["configs"].values() if item["nominal_latency_ns"] is not None]
                row["mean_absolute_relative_error"] = sum(errors) / len(errors) if errors else None
                scores[method] = row
            caches[cache] = scores
        out[case_id] = {"candidate_session_status_counts": statuses, "complete": complete, "cache": caches}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("seal", type=Path)
    parser.add_argument("amendment")
    args = parser.parse_args()
    output = args.directory / "transfer-scores.json"
    if output.exists():
        raise FileExistsError(output)
    index, policy, digest = verify_release(args.directory, args.amendment)
    archive = require_release(REVISION, args.directory / "predictions.json", digest)
    require_formal_build(archive["build"], REVISION, "GB10 prediction archive")
    if (set(archive["cohorts"]) != {"primary", "secondary", "diagnostic"}
            or (archive["draws"], archive["seed"], archive["pair_margin"]) != (32, 17, 0.05)):
        raise ValueError("Unexpected GB10 prediction archive coverage or perturbation settings")
    audit = read_committed_audit(args.seal)
    if audit["device"] != "GB10":
        raise ValueError("Expected a sealed GB10 collection")
    opened = datetime.now(timezone.utc).isoformat()
    files, verified, verified_utc = read_collection(args.directory, index, archive, audit)
    collection = {cohort: {} for cohort in archive["cohorts"]}
    uuid, session_ids = None, set()
    for name, (cohort, case, session) in files.items():
        report = json.loads(verified.pop(name))
        uuid = uuid or report["gpu"]["uuid"]
        if (report["timestamp_utc"] != audit["reports"][name]["timestamp_utc"]
                or report["frozen_revision"] != REVISION or report["cohort"] != cohort
                or report["source"]["commit"] != policy["harness_commit"]
                or report["prediction_archive_sha256"] != digest or report["gpu"]["arch"] != "sm_121"
                or report["gpu"]["uuid"] != uuid or report["gpu"]["driver"] != policy["driver_version"]
                or report["manifest_sha256"] != index["files"][f"{cohort}/manifest.json"]
                or report["session_id"] in session_ids):
            raise ValueError(f"{name}: inconsistent provenance")
        session_ids.add(report["session_id"])
        collection[cohort].setdefault(case["case_id"], (case, {}))[1][session] = report

    out = {name: {"cases": evaluate(name, args.directory, index, archive, cases)}
           for name, cases in collection.items()}
    for cohort in ("primary", "secondary"):
        entry = out[cohort]
        entry["summary"] = summarize(entry["cases"], True)
    primary = out["primary"]
    primary["gates"] = development_gates(primary["summary"])
    for cache, gate in primary["gates"].items():
        gate["complete"] = all(case["complete"] for case in primary["cases"].values())
        gate["all_cases_have_shortlists"] = all(
            case["cache"][cache]["core"]["top_k_regret"] is not None for case in primary["cases"].values()
        )
        gate["pass"] = gate["pass"] and gate["complete"] and gate["all_cases_have_shortlists"]
    primary["pass"] = all(gate["pass"] for gate in primary["gates"].values())

    write_json(output, {
        "cohorts": out, "profile_revision": archive["profile_revision"],
        "prediction_archive_sha256": digest, "seal_sha256": sha256(args.seal.read_bytes()),
        "release_started_utc": opened, "all_report_digests_verified_utc": verified_utc,
        "scored_utc": datetime.now(timezone.utc).isoformat(),
    })


if __name__ == "__main__":
    main()
