"""Verify and score the four frozen v0.2 holdout cohorts on CPU."""

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

from validation.experiment import development_gates, summarize
from validation.holdout import sha256
from validation.pair_precision import summarize as pair_precision
from validation.posthoc import read_sealed_bytes
from validation.provenance import REPO_ROOT, require_formal_build
from validation.score import PRACTICAL_MARGIN, check_artifact, pair_observations, score
from validation.v02_inputs import COHORTS, DRAWS, SEED, TAG, TARGETS, case_inputs, git

AMENDMENT = "validation/v02-freeze-amendment.json"
RETRY = "validation/v02-rtx4070-gemm-retry.json"
CACHES = ("warm", "cold")
METHODS = ("core", "roofline_wave", "neusight_style")
LOCATIONS = {
    "rtx4070-gemm": ("results/v02-rtx4070/gemm", "sealed"),
    "rtx4070-attention": ("results/attention-fde371d", "holdout-sealed"),
    "gb10-gemm": ("results/v02-gb10/gemm", "sealed"),
    "gb10-attention": ("results/v02-gb10/attention", "sealed"),
}
ATTENTION_SHAPE = ("batch", "query_heads", "kv_heads", "seq_len", "head_dim", "causal")


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_digest(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{label}: expected a SHA-256 digest")


def require_timestamp(value, label):
    if not isinstance(value, str) or datetime.fromisoformat(value).utcoffset() != timedelta(0):
        raise ValueError(f"{label}: expected a UTC timestamp")


def require_release(directory, amendment):
    revision = git("rev-parse", "--verify", f"refs/tags/{TAG}^{{commit}}").decode().strip()
    tag_ref = f"refs/tags/{TAG}"
    tag = git("rev-parse", "--verify", tag_ref).decode().strip()
    if git("ls-remote", "--exit-code", "origin", tag_ref).decode().split() != [tag, tag_ref]:
        raise ValueError(f"The local {TAG} tag differs from the published tag")
    approved = git("rev-parse", "--verify", f"{amendment}^{{commit}}").decode().strip()
    for ancestor, descendant in ((revision, approved), (approved, "origin/main")):
        subprocess.run(["git", "merge-base", "--is-ancestor", ancestor, descendant],
                       cwd=REPO_ROOT, check=True)
    if git("rev-parse", "HEAD").decode().strip() != revision or git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError(f"Scoring requires the clean {TAG} checkout")
    amendment_raw = git("show", f"{approved}:{AMENDMENT}")
    if git("show", f"origin/main:{AMENDMENT}") != amendment_raw:
        raise ValueError("The freeze amendment must be published on origin/main")
    evidence = json.loads(amendment_raw)
    local_evidence = json.loads((directory / "prediction-evidence.json").read_bytes())
    if evidence != local_evidence:
        raise ValueError("Local evidence differs from the published freeze amendment")
    raw = (directory / "predictions.json").read_bytes()
    if sha256(raw) != evidence["prediction_archive_sha256"]:
        raise ValueError("Prediction archive digest mismatch")
    archive = json.loads(raw)
    if {key: val for key, val in archive.items() if key != "cohorts"} != {
        key: val for key, val in evidence.items() if key != "prediction_archive_sha256"
    }:
        raise ValueError("Prediction provenance differs from the published freeze amendment")
    require_formal_build(archive["build"], revision, "Prediction archive")
    if (archive["result_class"], archive["tag"], archive["tag_object"]) != ("formal", TAG, tag):
        raise ValueError("Prediction archive does not identify the published freeze tag")
    if (archive["draws"], archive["seed"], archive["pair_margin"]) != (DRAWS, SEED, PRACTICAL_MARGIN):
        raise ValueError("Unexpected prediction draw or pair settings")
    if set(archive["cohorts"]) != COHORTS.keys() or set(archive["cohort_inputs"]) != COHORTS.keys():
        raise ValueError("The prediction archive must cover all four cohorts")
    for field in ("binary_sha256", "fit_sha256"):
        require_digest(evidence[field], field)
    require_timestamp(evidence["recorded_utc"], "Prediction archive")
    index_raw = (directory / "aot-index.json").read_bytes()
    if sha256(index_raw) != evidence["aot_index_sha256"]:
        raise ValueError("Frozen AOT index digest mismatch")
    return revision, approved, archive, json.loads(index_raw), evidence


def checked_path(root, relative):
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError("Report or artifact path must stay inside its archive")
    return path


def check_receipt(receipt, label, *, legacy=False):
    require_digest(receipt.get("sha256"), label)
    require_timestamp(receipt.get("timestamp_utc"), label)
    status = receipt.get("status", "complete" if legacy else None)
    blocks = receipt.get("complete_blocks", 10 if legacy else None)
    if status != "complete" or type(blocks) is not int or blocks != 10:
        raise ValueError(f"{label}: expected a complete 10-block receipt")


def retry_receipts(name, case, session, slot, seal, root, retry_raw):
    effective = seal["reports"][slot]
    history = seal.get("failed_reports", {}).get(slot)
    if name != "rtx4070-gemm" or not history or retry_raw is None:
        raise ValueError(f"{slot}: missing authorized retry history")
    if sha256(retry_raw) != effective.get("retry_receipt_sha256"):
        raise ValueError(f"{slot}: retry receipt digest mismatch")
    retry = json.loads(retry_raw)
    orig = [row for row in history if row["receipt"]["sha256"] == effective.get("repeat_of")]
    if len(orig) != 1 or len(history) != 1:
        raise ValueError(f"{slot}: retry must refer to the single failed attempt")
    failure = orig[0]["receipt"]
    require_digest(failure.get("sha256"), slot)
    require_timestamp(failure.get("timestamp_utc"), slot)
    if failure.get("status") != "protocol_failed":
        raise ValueError(f"{slot}: repeat_of must refer to a protocol failure")
    expected = {"case_id": case["case_id"], "session": session, "order_seed": 100 + session,
                "retry_attempt": 1, "original_report": slot, "retry_of_sha256": failure["sha256"],
                "original_receipt": failure, "retry_report": effective["report_path"],
                "capture_revision": seal["source"]["commit"], "manifest_sha256": seal["manifest_sha256"],
                "profile_sha256": seal["profile_sha256"], "wave_boundary_sha256": seal["wave_boundary_sha256"],
                "aot_sha256": seal["aot_sha256_by_case"][case["case_id"]],
                "original_seal": COHORTS[name],
                "retry_receipt": {field: effective[field]
                                  for field in ("sha256", "timestamp_utc", "status", "complete_blocks")}}
    if effective.get("order_seed") != 100 + session or any(retry.get(k) != v for k, v in expected.items()):
        raise ValueError(f"{slot}: retry receipt does not agree with the holdout seal")
    if datetime.fromisoformat(effective["timestamp_utc"]) <= datetime.fromisoformat(failure["timestamp_utc"]):
        raise ValueError(f"{slot}: retry must follow the failed capture")
    return checked_path(root, effective["report_path"]), (checked_path(root, orig[0]["report_path"]), failure)


def check_predictions(case, records, revision):
    keys = set(case["config_ids"])
    if set(records) != set(CACHES):
        raise ValueError("Prediction archive is missing a cache regime")
    for methods in records.values():
        if set(methods) != set(METHODS) or any(set(rows) != keys for rows in methods.values()):
            raise ValueError("Prediction archive is missing methods or candidate statuses")
        for method, forecasts in methods.items():
            for row in forecasts.values():
                if method == "core":
                    require_formal_build(row.get("build"), revision, "Core candidate")
                if row["status"] != "ok":
                    continue
                nominal = row["nominal_latency_ns"]
                if not math.isfinite(nominal) or nominal <= 0:
                    raise ValueError("Invalid nominal prediction")
                if method == "core":
                    draws = row["draws"]
                    if (set(draws) != {str(i) for i in range(DRAWS + 1)} or draws["0"] != nominal
                            or any(not math.isfinite(t) or t <= 0 for t in draws.values())):
                        raise ValueError("Invalid frozen core draws")


def prepare_collection(directory, root, revision, archive, index):
    if set(index) != COHORTS.keys():
        raise ValueError("Frozen AOT index must cover all four cohorts")
    collection = {}
    for name, (location, sealed_dir) in LOCATIONS.items():
        cases, _, seal, evidence = case_inputs(name, revision)
        device = name.split("-")[0]
        cohort = archive["cohort_inputs"][name]
        profile = git("show", f"{revision}:profiles/{device}.json")
        if (any(cohort.get(key) != value for key, value in evidence.items())
                or cohort["profile_sha256"] != sha256(profile)
                or cohort["profile_revision"] != json.loads(profile)["provenance"]["profile_revision"]):
            raise ValueError(f"{name}: frozen profile, manifest or seal differs from its archive")
        expected = {case["case_id"] for case in cases}
        if set(archive["cohorts"][name]) != expected or set(index[name]) != expected:
            raise ValueError(f"{name}: prediction or AOT case census differs from the manifest")
        histories = seal.get("failed_reports", {})
        retry_slots = {slot for slot, entry in seal["reports"].items() if "report_path" in entry}
        if set(histories) != retry_slots:
            raise ValueError(f"{name}: orphaned or missing failed report history")
        retry_raw = git("show", f"{revision}:{RETRY}") if retry_slots else None
        if retry_slots and git("show", f"origin/main:{RETRY}") != retry_raw:
            raise ValueError("The frozen retry receipt is not published")
        base = root / location
        collection[name] = {}
        for case in cases:
            case_id = case["case_id"]
            records = archive["cohorts"][name][case_id]
            check_predictions(case, records, revision)
            capture_raw = (base / "aot" / f"{case_id}.json").read_bytes()
            capture_digest = seal["aot_sha256_by_case"][case_id]
            capture = json.loads(capture_raw)
            if (sha256(capture_raw) != capture_digest or capture["case_id"] != case_id
                    or capture["manifest_sha256"] != seal["manifest_sha256"]
                    or set(capture["configs"]) != set(case["config_ids"])):
                raise ValueError(f"{case_id}: capture compiler metadata differs from the seal")
            frozen = index[name][case_id]
            if (frozen["capture_aot_sha256"] != capture_digest
                    or set(frozen["configs"]) != set(case["config_ids"])):
                raise ValueError(f"{case_id}: frozen IR index differs from the capture archive")
            for row in frozen["configs"].values():
                if row["status"] == "ok" and "ir" not in row:
                    raise ValueError(f"{case_id}: successful frozen IR record has no artifact")
                if "ir" in row and digest_file(checked_path(directory, row["ir"])) != row["ir_sha256"]:
                    raise ValueError(f"{case_id}: frozen IR artifact digest mismatch")
            reports, failed = {}, []
            for session in (1, 2, 3):
                slot = f"{case_id}-session-{session}.json"
                entry = seal["reports"][slot]
                check_receipt(entry, slot, legacy=name == "rtx4070-attention")
                path = base / sealed_dir / slot
                if slot in retry_slots:
                    path, failed_report = retry_receipts(name, case, session, slot, seal, root, retry_raw)
                    failed.append(failed_report)
                reports[session] = (path, entry)
            collection[name][case_id] = {
                "case": case, "capture": capture, "capture_sha256": capture_digest,
                "receipt_manifest": seal["manifest_sha256"],
                "capture_profile_sha256": seal.get("profile_sha256"), "forecasts": records,
                "reports": reports, "failed": failed,
            }
    return collection


def verify_reports(collection):
    verified = {}
    files = set()
    for name, cases in collection.items():
        verified[name] = {}
        for case_id, entry in cases.items():
            snapshots = {}
            sources = list(entry["reports"].items()) + [(None, row) for row in entry["failed"]]
            for session, (path, receipt) in sources:
                key = path.resolve()
                if key in files:
                    raise ValueError("A sealed report is referenced more than once")
                files.add(key)
                raw = read_sealed_bytes(path)
                if sha256(raw) != receipt["sha256"]:
                    raise ValueError(f"{path.name}: report digest mismatch")
                if session is not None:
                    snapshots[session] = raw
            verified[name][case_id] = snapshots
    return verified


def report_timings(name, case_id, entry, snapshots, sessions_seen, device_uuids):
    case = entry["case"]
    device, family = name.split("-")
    shape = case["shape"] if family == "gemm" else [case[key] for key in ATTENTION_SHAPE]
    config_ids = set(case["config_ids"])
    expected_toolchain = entry["capture"]["configs"]
    timings = {cache: {} for cache in CACHES}
    statuses = {key: {} for key in case["config_ids"]}
    legacy = name == "rtx4070-attention"
    if set(snapshots) != {1, 2, 3}:
        raise ValueError(f"{case_id}: expected all three case-sessions")
    for order, raw in sorted(snapshots.items()):
        report = json.loads(raw)
        _, receipt = entry["reports"][order]
        session = report["session_id"]
        gpu = report["gpu"]
        uuid = gpu["uuid"]
        if (report["timestamp_utc"] != receipt["timestamp_utc"] or report["case_id"] != case_id
                or report["shape"] != shape or report["split"] != "holdout"
                or report["manifest_sha256"] != entry["receipt_manifest"]
                or report["aot_sha256"] != entry["capture_sha256"]
                or report["kernel_key"] != entry["capture"]["kernel_key"]
                or report["source"] != entry["capture"]["source"]
                or (not legacy and report.get("profile_sha256") != entry["capture_profile_sha256"])
                or gpu["arch"] != TARGETS[device] or not uuid
                or uuid != device_uuids.setdefault(device, uuid) or session in sessions_seen):
            raise ValueError(f"{case_id}/{order}: inconsistent report provenance")
        sessions_seen.add(session)
        protocol = report["protocol"]
        if (protocol["order_seed"] != 100 + order or protocol["blocks"] != 10
                or protocol["cache_states"] != list(CACHES) or protocol["cold_flush"] != "read_cg"
                or protocol["timer"] != "cuda_graph_single_launch"
                or protocol["l2_bytes"] <= 0
                or protocol["eviction_bytes"] < max(3 * protocol["l2_bytes"], 128 * 1024**2)
                or (device == "gb10" and protocol["thermal_warmup_seconds"] < 60)):
            raise ValueError(f"{case_id}/{order}: capture protocol changed")
        if (report.get("complete_blocks", 10 if legacy else None) != 10
                or report.get("collection_status", "complete" if legacy else None) != "complete"):
            raise ValueError(f"{case_id}/{order}: expected ten completed blocks")
        if not report["configs"].keys() <= config_ids:
            raise ValueError(f"{case_id}/{order}: unexpected capture candidates")
        measured, eligible = set(), set()
        for key in case["config_ids"]:
            jit = report["configs"].get(key, {"status": "missing"})
            status = jit["status"]
            if status == "ok":
                measured.add(key)
                aot = expected_toolchain[key]
                if aot["status"] == "ok":
                    match = check_artifact(jit, aot)
                    if match["status"] == "matched" and match["ttgir_match"]:
                        eligible.add(key)
                if key not in eligible:
                    status = "artifact_mismatch"
            statuses[key][str(order)] = status
        expected_trials = {(cache, key, block) for cache in CACHES
                           for key in measured for block in range(10)}
        observed = set()
        for row in report["samples"]:
            cache, key, block = row["cache"], row["config_id"], row["block"]
            trial = cache, key, block
            value = row["latency_ns"]
            if (trial in observed or trial not in expected_trials or type(block) is not int
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{case_id}/{order}: invalid or duplicate timing sample")
            observed.add(trial)
            if key in eligible:
                timings[cache].setdefault(key, {})[session, block] = value
        # The original attention receipts predate the complete_blocks field.
        # The complete sample census still establishes all ten randomized blocks.
        if not expected_trials or observed != expected_trials:
            raise ValueError(f"{case_id}/{order}: incomplete randomized blocks")
    return timings, statuses


def scored_case(timings, forecasts, statuses):
    status_counts = Counter(status for sessions in statuses.values() for status in sessions.values())
    caches = {}
    for cache in CACHES:
        measured = timings[cache]
        observed = pair_observations(measured, samples=1000, seed=SEED)
        scores = {}
        for method in METHODS:
            rows = forecasts[cache][method]
            if measured:
                result = score(measured, rows, k=5, samples=1000, seed=SEED, observations=observed,
                               nominal_pair_margin=None if method == "core" else PRACTICAL_MARGIN)
                result.pop("pairs")
            else:
                result = {"kendall_tau_b": None, "top_k_hit": False, "top_k_regret": None,
                          "shortlist": [], "measured_configs": 0, "scored_configs": 0,
                          "configs": {}, "pair_confusion": {}}
            errors = [abs(row["nominal_latency_ns"] / row["measured_median_ns"] - 1)
                      for row in result["configs"].values() if row["nominal_latency_ns"] is not None]
            result.update(
                prediction_status_counts=dict(Counter(row["status"] for row in rows.values())),
                missing_or_invalid_predictions={key: row["status"] for key, row in rows.items()
                                                if row["status"] != "ok"},
                mean_absolute_relative_error=sum(errors) / len(errors) if errors else None,
            )
            scores[method] = result
        caches[cache] = scores
    unavailable = sorted(set(statuses) - timings["warm"].keys())
    return {"candidate_session_status_counts": dict(status_counts), "status_by_config": statuses,
            "unmeasured_configs": unavailable, "cache": caches}


def aggregate(cases):
    summary = summarize(cases, True)
    for cache, methods in summary.items():
        for method, row in methods.items():
            per_case = [case["cache"][cache][method] for case in cases.values()]
            apes = [score["mean_absolute_relative_error"] for score in per_case
                    if score["mean_absolute_relative_error"] is not None]
            row.update(
                undefined_tau_b=[key for key, case in cases.items()
                                 if case["cache"][cache][method]["kendall_tau_b"] is None],
                top_5_hits=sum(score["top_k_hit"] for score in per_case),
                cases_with_measurements=sum(score["measured_configs"] > 0 for score in per_case),
                mean_absolute_relative_error=sum(apes) / len(apes) if apes else None,
                directional=pair_precision(row["pair_confusion"]),
            )
    gates = development_gates(summary)
    for cache, gate in gates.items():
        gate["cases_missing_core_shortlist"] = [
            key for key, case in cases.items()
            if case["cache"][cache]["core"]["measured_configs"] and not case["cache"][cache]["core"]["shortlist"]
        ]
        gate["pass"] = gate["pass"] and not gate["cases_missing_core_shortlist"]
    return {"cases": cases, "summary": summary, "gates": gates,
            "pass": all(row["pass"] for row in gates.values())}


def evaluate(collection, snapshots):
    cohorts = {}
    sessions_seen, device_uuids = set(), {}
    for name, entries in collection.items():
        cases = {}
        for case_id, entry in entries.items():
            timings, statuses = report_timings(name, case_id, entry, snapshots[name][case_id],
                                               sessions_seen, device_uuids)
            scored = scored_case(timings, entry["forecasts"], statuses)
            scored["failed_capture_attempts"] = [receipt for _, receipt in entry["failed"]]
            cases[case_id] = scored
        cohorts[name] = aggregate(cases)
    devices = {device: all(cohorts[f"{device}-{family}"]["pass"] for family in ("gemm", "attention"))
               for device in TARGETS}
    return cohorts, devices


def decimal(value, percent=False):
    if value is None:
        return "n/a"
    return f"{value:.2%}" if percent else f"{value:.3f}"


def rate(numerator, denominator):
    return decimal(numerator / denominator if denominator else None, percent=True)


def render(report, scores_name):
    word = lambda value: "Pass" if value else "Fail"
    lines = [
        "# VirtualGPU v0.2 formal holdout evaluation", "",
        f"Frozen core `{report['core_revision']}`. Release amendment `{report['amendment_commit']}`.",
        f"All report digests verified at {report['all_report_digests_verified_utc']}.", "",
        "| Device | Decision |", "| --- | --- |",
        *(f"| {device} | {word(value)} |" for device, value in report["devices"].items()),
    ]
    for name, cohort in report["cohorts"].items():
        lines.extend([
            "", f"## {name}", "", f"Cohort decision: **{word(cohort['pass'])}**.", "",
            "| Cache | Method | Median tau-b | Defined ranks | Mean top-5 regret | Top-5 hits | Coverage | "
            "Direction agreement | Ordered coverage | MAPE |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for cache, methods in cohort["summary"].items():
            for method in METHODS:
                row = methods[method]
                pair = row["directional"]
                lines.append(
                    f"| {cache} | {method} | {decimal(row['median_tau_b'])} | "
                    f"{row['cases_with_tau_b']}/{len(cohort['cases'])} | "
                    f"{decimal(row['mean_top_5_regret'], True)} | "
                    f"{row['top_5_hits']}/{row['cases_with_measurements']} | "
                    f"{row['scored_configs']}/{row['measured_configs']} | "
                    f"{rate(pair['agree'], pair['ordered_calls'])} | "
                    f"{rate(pair['hardware_ordered_found'], pair['hardware_ordered'])} | "
                    f"{decimal(row['mean_absolute_relative_error'], True)} |"
                )
        lines.extend([
            "", "| Cache | Rank vs Roofline | Rank vs NeuSight | Absolute regret | Non-inferiority | Decision |",
            "| --- | --- | --- | --- | --- | --- |",
        ])
        for cache, gate in cohort["gates"].items():
            roof = gate["baselines"]["roofline_wave"]
            fit = gate["baselines"]["neusight_style"]
            criteria = (roof["ranking_pass"], fit["ranking_pass"], gate["absolute_regret_pass"],
                        roof["non_inferiority_pass"] and fit["non_inferiority_pass"], gate["pass"])
            lines.append(f"| {cache} | " + " | ".join(map(word, criteria)) + " |")
        lines.extend([
            "", "| Case | Capture status counts across sessions | Unmeasured configs |",
            "| --- | --- | --- |",
        ])
        for case_id, case in cohort["cases"].items():
            counts = ", ".join(f"{key}={val}" for key, val in sorted(case["candidate_session_status_counts"].items()))
            missing = ", ".join(case["unmeasured_configs"]) or "None"
            lines.append(f"| {case_id} | {counts} | {missing} |")
    lines.extend([
        "", f"[Full scores and candidate status audit]({scores_name}).",
        "The score file includes confusion matrices, undefined ranks, missing or invalid",
        "predictions, and each configuration's status in every session.", "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("amendment", help="Published freeze amendment Git commit")
    parser.add_argument("output", type=Path, help="New directory for the formal results")
    parser.add_argument("--measurement-root", type=Path, default=REPO_ROOT,
                        help="Checkout root containing the captured results directories")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    revision, amendment, archive, index, evidence = require_release(args.archive, args.amendment)
    collection = prepare_collection(args.archive, args.measurement_root, revision, archive, index)
    started = utc()
    snapshots = verify_reports(collection)
    verified = utc()
    cohorts, devices = evaluate(collection, snapshots)
    report = {
        "result_class": "formal", "core_revision": revision, "amendment_commit": amendment,
        "prediction_archive_sha256": evidence["prediction_archive_sha256"],
        "cohort_inputs": archive["cohort_inputs"], "release_started_utc": started,
        "all_report_digests_verified_utc": verified, "scored_utc": utc(),
        "cohorts": cohorts, "devices": devices,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{args.output.name}-", dir=args.output.parent) as staging:
        output = Path(staging)
        scores = output / "scores.json"
        scores.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (output / "report.md").write_text(render(report, scores.name), encoding="utf-8")
        print(f"Score SHA-256 {digest_file(scores)}")
        output.rename(args.output)


if __name__ == "__main__":
    main()
