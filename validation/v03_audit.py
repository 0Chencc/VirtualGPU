"""Publish v0.3 cohort receipts without opening any sealed timing report."""

import argparse
import json
from pathlib import Path

from validation.holdout import sha256
from validation.v02_score import require_digest, require_timestamp
from validation.v03_capture import check_sealed, receipt_name, utc, write_json
from validation.v03_plan import DEVICES, FAMILIES, load_plan, published_main, require_capture


def audit(directory, output, device, family):
    main = published_main()
    release, release_sha, inputs = load_plan(main)
    base = directory / f"{device}-{family}"
    cases = inputs[family]["devices"][device]["cases"]
    cohort = inputs["aot"]["cohorts"][f"{device}-{family}"]
    accepted, failed, consumed, sealed, pending = {}, {}, set(), set(), []
    for case in cases:
        case_id = case["case_id"]
        for session in (1, 2, 3):
            previous = None
            attempt = 1
            while True:
                name = receipt_name(case_id, session, attempt)
                path = base / "receipts" / name
                if not path.is_file():
                    pending.append(name)
                    break
                raw = path.read_bytes()
                row = json.loads(raw)
                approval, retry = require_capture(main, device, release_sha, case_id, session, attempt, previous)
                expected = {
                    "release_sha256": release_sha, "capture_approval_sha256": approval,
                    "retry_approval_sha256": retry, "repeat_of_receipt_sha256": previous,
                    "device": device, "family": family, "case_id": case_id,
                    "session": session, "attempt": attempt, "manifest_sha256": release["inputs"][family],
                    "aot_sha256": cohort["cases"][case_id]["aot_sha256"],
                    "profile_sha256": release["capture_profiles"][device],
                }
                if any(row.get(key) != value for key, value in expected.items()):
                    raise ValueError(f"{name}: receipt does not match the approved capture")
                require_timestamp(row["timestamp_utc"], name)
                consumed.add(name)
                if "sha256" in row:
                    require_digest(row["sha256"], name)
                    check_sealed(base / "sealed" / name)
                    sealed.add(name)
                if row["status"] == "complete" and row["complete_blocks"] == 10 and name in sealed:
                    accepted[name] = row
                    break
                if row["status"] != "protocol_failed":
                    raise ValueError(f"{name}: unexpected receipt status")
                failed[name] = row
                previous = sha256(raw)
                attempt += 1
    if ({p.name for p in (base / "receipts").glob("*.json")} != consumed
            or {p.name for p in (base / "sealed").glob("*.json")} != sealed):
        raise ValueError("Unknown, duplicate, or unreceipted attempt")
    write_json(output, {
        "device": device, "family": family, "timestamp_utc": utc(), "release_sha256": release_sha,
        "reports": accepted, "failures": failed, "pending": pending,
    })
    print(f"{device}-{family}: {len(accepted)} complete, {len(failed)} failed, {len(pending)} pending")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("device", choices=DEVICES)
    parser.add_argument("family", choices=FAMILIES)
    args = parser.parse_args()
    audit(args.directory, args.output, args.device, args.family)


if __name__ == "__main__":
    main()
