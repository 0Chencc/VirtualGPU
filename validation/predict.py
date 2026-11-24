"""Run the model on artifact-verified development candidates."""

import argparse
import hashlib
import json
from pathlib import Path

from virtualgpu import _core

from validation.experiment import CACHE_STATES, load_case
from validation.provenance import read_core_revision, require_formal_build, source_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("profile", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--cold-results", type=Path, help="Directory of replacement read-only cold sessions")
    parser.add_argument("--draws", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--core-revision", type=read_core_revision, help="Expected core SHA or revision file")
    mode.add_argument("--diagnostic", action="store_true", help="Allow unverified or dirty build provenance")
    args = parser.parse_args()
    if args.sessions < 1 or args.draws < 1:
        parser.error("Use positive session and perturbation draw counts")
    source = source_info()
    revision_expected = args.core_revision or source["commit"]
    build_info = getattr(_core, "build_info", lambda: None)()
    if not args.diagnostic:
        try:
            if source["commit"] != revision_expected or source["python_dirty"] or source["validation_dirty"]:
                raise ValueError("Formal prediction requires the requested HEAD and clean Python sources")
            require_formal_build(build_info, revision_expected, "Loaded binary")
        except ValueError as error:
            parser.error(str(error))

    raw = args.manifest.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    profile = args.profile.read_text()
    revision = json.loads(profile)["provenance"]["profile_revision"]
    predictions = {}
    for case in json.loads(raw)["cases"]:
        if case["split"] != "development":
            continue
        key = case["case_id"]
        data = load_case(case, args.results, digest, args.sessions, args.cold_results)
        ir_dir = args.results / "aot" / key
        predictions[key] = {}
        for cache in CACHE_STATES:
            records = {}
            for config_id in data.artifact_rows:
                ir = (ir_dir / f"{config_id}.json").read_text()
                try:
                    record = _core.predict(
                        ir, profile, l2_state=cache,
                        draws=args.draws, seed=args.seed,
                    )
                except ValueError as error:
                    record = {"status": "failed", "reason": str(error), "build": build_info}
                if args.diagnostic:
                    build = record.get("build") or {}
                    record["build"] = {**build, "result_class": "diagnostic"}
                else:
                    require_formal_build(record.get("build"), revision_expected, f"{key} {cache} {config_id}")
                if record["status"] == "ok":
                    record["nominal_latency_ns"] = record["time_ns"]
                records[config_id] = record
            predictions[key][cache] = records
            print(f"{key} {cache}: {len(records)} prediction records", flush=True)
    bundle = {
        "manifest_sha256": digest,
        "profile_revision": revision,
        "perturbation_draws": args.draws,
        "perturbation_seed": args.seed,
        "result_class": "diagnostic" if args.diagnostic else "formal",
        "source": source,
        "cases": predictions,
    }
    args.output.write_text(json.dumps(bundle, indent=2) + "\n")


if __name__ == "__main__":
    main()
