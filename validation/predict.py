"""Run the model on artifact-verified development candidates."""

import argparse
import hashlib
import json
from pathlib import Path

from virtualgpu import _core

from validation.experiment import CACHE_STATES, load_case


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
    args = parser.parse_args()
    if args.sessions < 1 or args.draws < 1:
        parser.error("Use positive session and perturbation draw counts")

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
                    record = {"status": "failed", "reason": str(error)}
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
        "cases": predictions,
    }
    args.output.write_text(json.dumps(bundle, indent=2) + "\n")


if __name__ == "__main__":
    main()
