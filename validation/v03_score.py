"""Score v0.3 only after verifying the published freeze and opening authorization."""

import argparse
from datetime import datetime
from pathlib import Path

from validation.holdout import sha256
from validation.provenance import require_formal_build
from validation.v02_score import (
    aggregate,
    check_predictions,
    checked_path,
    report_timings,
    scored_case,
    utc,
    verify_reports,
)
from validation.v03_capture import receipt_name, write_json
from validation.v03_plan import (
    DEVICES,
    FAMILIES,
    OPEN,
    PLAN,
    aot_case,
    checked_json,
    git,
    introduction,
    load_plan,
    published_main,
    read_published,
    require_ancestor,
    require_reviewed_code,
)

FREEZE = "validation/v03-freeze-amendment.json"
TAG = "v0.3-final"


def verified_freeze(main, release, release_sha, root):
    freeze, sha = read_published(main, FREEZE, "predictions_frozen")
    opened, _ = read_published(main, OPEN, "opening_approved")
    frozen_at, opened_at = introduction(main, FREEZE), introduction(main, OPEN)
    require_ancestor(introduction(main, PLAN), frozen_at)
    require_ancestor(frozen_at, opened_at)
    if (opened_at == frozen_at or opened["freeze_sha256"] != sha
            or datetime.fromisoformat(opened["timestamp_utc"]) < datetime.fromisoformat(freeze["timestamp_utc"])):
        raise ValueError("Opening authorization must follow and identify the published freeze")
    tag_ref = f"refs/tags/{TAG}"
    tag = git("rev-parse", "--verify", tag_ref).decode().strip()
    if git("ls-remote", "--exit-code", "origin", tag_ref).decode().split() != [tag, tag_ref]:
        raise ValueError("The local v0.3 tag differs from the published tag")
    revision = git("rev-parse", f"{tag_ref}^{{commit}}").decode().strip()
    require_ancestor(revision, frozen_at)
    head = require_reviewed_code(release["code_commit"], capture=False)
    if freeze["revision"] != revision or head != revision or freeze["release_sha256"] != release_sha:
        raise ValueError("Score only from the clean tagged checkout against its approved manifest release")
    from virtualgpu import _core

    from validation.v02_predict import require_imports_from_checkout

    require_imports_from_checkout(_core)
    require_formal_build(_core.build_info(), revision, "Local core")
    if set(freeze["profiles"]) != set(DEVICES) or set(freeze["fits"]) != set(FAMILIES):
        raise ValueError("The freeze must identify both profiles and both family fits")
    for device, digest in freeze["profiles"].items():
        if sha256(git("show", f"{revision}:profiles/{device}.json")) != digest:
            raise ValueError(f"{device}: frozen profile digest mismatch")
    for entry in freeze["fits"].values():
        checked_json(checked_path(root, entry["path"]), entry["sha256"])
    return freeze


def prepare_collection(freeze, inputs, root, aot_directory):
    archive = checked_json(checked_path(root, freeze["predictions"]["path"]), freeze["predictions"]["sha256"])
    names = {f"{device}-{family}" for device in DEVICES for family in FAMILIES}
    if set(archive) != names or set(freeze["seals"]) != names:
        raise ValueError("The freeze must include predictions and receipts for all four cohorts")
    collection, label_paths = {}, set()
    for name in sorted(names):
        device, family = name.split("-")
        seal_ref = freeze["seals"][name]
        seal = checked_json(checked_path(root, seal_ref["path"]), seal_ref["sha256"])
        if seal["reports"].keys() & seal["failures"].keys():
            raise ValueError(f"{name}: duplicate attempt across receipt categories")
        if (seal["pending"] or seal["release_sha256"] != freeze["release_sha256"]
                or (seal["device"], seal["family"]) != (device, family)):
            raise ValueError(f"{name}: missing or invalid cohort seal")
        cases = inputs[family]["devices"][device]["cases"]
        if set(archive[name]) != {case["case_id"] for case in cases}:
            raise ValueError(f"{name}: prediction case inventory differs from the manifest")
        entries = {}
        for case in cases:
            key = case["case_id"]
            forecast = archive[name][key]
            check_predictions(case, forecast, freeze["revision"])
            artifact = aot_case(aot_directory, inputs, device, family, case)
            effective, failed = {}, []
            for filename, row in (seal["reports"] | seal["failures"]).items():
                if row["case_id"] != key:
                    continue
                if (filename != receipt_name(key, row["session"], row["attempt"])
                        or row["aot_sha256"] != artifact["sha256"]
                        or row["manifest_sha256"] != inputs["aot"]["manifests"][family]
                        or row["release_sha256"] != freeze["release_sha256"]):
                    raise ValueError(f"{name}/{filename}: receipt identity mismatch")
                path = checked_path(root, f"{name}/sealed/{filename}")
                if "sha256" in row:
                    label_paths.add(path)
                if row["status"] == "complete" and filename in seal["reports"]:
                    if row["session"] in effective:
                        raise ValueError(f"{name}/{key}: duplicate effective session")
                    effective[row["session"]] = path, row
                elif row["status"] == "protocol_failed" and filename in seal["failures"]:
                    failed.append((path if "sha256" in row else None, row))
                else:
                    raise ValueError(f"{name}/{filename}: invalid receipt classification")
            if set(effective) != {1, 2, 3}:
                raise ValueError(f"{name}/{key}: all three valid sessions are required")
            first = effective[1][1]
            entries[key] = {
                "case": case, "reports": effective, "failed": failed, "forecasts": forecast,
                "capture_sha256": artifact["sha256"], "capture_profile_sha256": first["profile_sha256"],
                "receipt_manifest": first["manifest_sha256"],
                "capture": {"configs": artifact["configs"], "kernel_key": first["kernel_key"],
                            "source": {"commit": first["source_commit"], "python_dirty": False,
                                       "validation_dirty": False}},
            }
        if len(seal["reports"]) != 3 * len(cases) or any(
            row["case_id"] not in entries for row in seal["failures"].values()
        ):
            raise ValueError(f"{name}: extraneous sealed receipt")
        collection[name] = entries
    if set(root.glob("*-*/sealed/*.json")) != label_paths:
        raise ValueError("Sealed report inventory differs from the approved receipts")
    return collection


def score_collection(collection):
    # The existing verifier hashes every report before the first timing parse.
    on_disk = {
        name: {key: {**entry, "failed": [(p, r) for p, r in entry["failed"] if p is not None]}
               for key, entry in cases.items()}
        for name, cases in collection.items()
    }
    snapshots = verify_reports(on_disk)
    cohorts, sessions, devices = {}, set(), {}
    for name, entries in collection.items():
        cases = {}
        for key, entry in entries.items():
            timings, statuses, probes = report_timings(
                name, key, entry, snapshots[name][key], sessions, devices, legacy=False,
            )
            case = scored_case(timings, entry["forecasts"], statuses)
            case["cache_control_ns"] = probes
            case["failed_capture_attempts"] = [receipt for _, receipt in entry["failed"]]
            cases[key] = case
        cohorts[name] = aggregate(cases)
    return cohorts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("aot_directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    main_revision = published_main()
    release, release_sha, inputs = load_plan(main_revision)
    root = args.directory.resolve()
    freeze = verified_freeze(main_revision, release, release_sha, root)
    collection = prepare_collection(freeze, inputs, root, args.aot_directory)
    cohorts = score_collection(collection)
    decisions = {device: all(cohorts[f"{device}-{family}"]["pass"] for family in FAMILIES) for device in DEVICES}
    write_json(args.output, {"timestamp_utc": utc(), "revision": freeze["revision"],
                             "cohorts": cohorts, "devices": decisions})


if __name__ == "__main__":
    main()
