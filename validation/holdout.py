"""Capture workload holdout reports and verify their seal before scoring."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess


class ArtifactMismatch(ValueError):
    pass


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def verify_compiled(artifacts, directory):
    from validation.score import check_artifact

    if not any(row["status"] == "ok" for row in artifacts.values()):
        raise ArtifactMismatch("No eligible JIT artifacts")
    for config, jit in artifacts.items():
        if jit["status"] != "ok":
            continue
        aot = directory / f"{config}.json"
        if not aot.is_file() or check_artifact(jit, json.loads(aot.read_text()))["status"] != "matched":
            raise ArtifactMismatch(f"AOT/JIT mismatch for {config}")


def seal_report(path, report):
    data = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    receipt = {"sha256": sha256(data), "timestamp_utc": report["timestamp_utc"]}
    # Publish with no read permission from creation. The open descriptor can write.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        path.unlink()
        raise
    return receipt


def require_release(expected_revision, archive, archive_digest):
    """Check the freeze tag and archived predictions before accessing report bytes."""
    tag = subprocess.run(
        ["git", "rev-parse", "--verify", "refs/tags/phase1-final^{commit}"],
        capture_output=True, text=True,
    )
    if tag.returncode or tag.stdout.strip() != expected_revision:
        raise ValueError("Sealed labels require the phase1-final commit")
    data = archive.read_bytes()
    if sha256(data) != archive_digest:
        raise ValueError("Prediction archive digest does not match")
    return json.loads(data)


def read_committed_audit(path):
    from validation.provenance import REPO_ROOT

    relative = path.resolve().relative_to(REPO_ROOT)
    checked = subprocess.run(
        ["git", "show", f"HEAD:{relative.as_posix()}"], cwd=REPO_ROOT, capture_output=True,
    )
    data = path.read_bytes()
    if checked.returncode or checked.stdout != data:
        raise ValueError("Holdout scoring requires the committed seal index")
    return json.loads(data)


def read_verified_report(path, audit):
    files = audit["reports"]
    if path.name not in files or path.is_symlink() or stat.S_IMODE(path.stat().st_mode) != 0:
        raise ValueError(f"{path.name}: report is missing from the seal or has unexpected permissions")
    # Caller has already checked the tag and prediction archive. Restore the seal
    # even if the digest check or JSON parsing fails.
    path.chmod(0o400)
    try:
        data = path.read_bytes()
    finally:
        path.chmod(0)
    if sha256(data) != files[path.name]["sha256"]:
        raise ValueError(f"{path.name}: report digest does not match the seal")
    report = json.loads(data)
    if (report["split"] != "holdout" or report["manifest_sha256"] != audit["manifest_sha256"]
            or report["timestamp_utc"] != files[path.name]["timestamp_utc"]):
        raise ValueError(f"{path.name}: inconsistent holdout provenance")
    return report


def capture(directory, case_id, session):
    from types import SimpleNamespace

    from validation.runner import measure

    raw = (directory / "manifest.json").read_bytes()
    suite = json.loads(raw)
    case = next(c for c in suite["cases"] if c["case_id"] == case_id and c["split"] == "holdout")
    configs = [cfg for cfg in suite["configs"] if cfg["config_id"] in case["config_ids"]]
    args = SimpleNamespace(cache="both", blocks=10, seed=100 + session, warmup_seconds=2)

    def verify(rows):
        verify_compiled(rows, directory / "aot" / case_id)

    name = f"{case_id}-session-{session}.json"
    report_path = directory / name
    receipts = directory / "receipts"
    receipts.mkdir(exist_ok=True)
    receipt_path = receipts / name
    if report_path.exists() or receipt_path.exists():
        raise FileExistsError(name)
    report = measure(case, configs, suite["input_seed"], args, sha256(raw), verify_artifacts=verify)
    receipt = seal_report(report_path, report)
    with receipt_path.open("x") as output:
        json.dump(receipt, output, indent=2)
        output.write("\n")


def write_audit(directory, output):
    raw = (directory / "manifest.json").read_bytes()
    suite = json.loads(raw)
    names = {f"{case['case_id']}-session-{session}.json"
             for case in suite["cases"] for session in (1, 2, 3)}
    receipts = directory / "receipts"
    if names != {path.name for path in receipts.glob("*.json")}:
        raise ValueError("Incomplete sealed collection")
    files = {}
    for name in sorted(names):
        path = directory / name
        if path.is_symlink() or stat.S_IMODE(path.stat().st_mode) != 0:
            raise ValueError(f"{name}: raw report is not sealed")
        files[name] = json.loads((receipts / name).read_text())
    audit = {
        "schema_version": 1,
        "device": "RTX 4070",
        "manifest_sha256": sha256(raw),
        "manifest_materialization": suite["materialization"],
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reports": files,
    }
    with output.open("x") as stream:
        json.dump(audit, stream, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("capture")
    run.add_argument("directory", type=Path)
    run.add_argument("case")
    run.add_argument("session", type=int, choices=(1, 2, 3))
    audit = commands.add_parser("audit")
    audit.add_argument("directory", type=Path)
    audit.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "capture":
        try:
            capture(args.directory, args.case, args.session)
        except ArtifactMismatch:
            print(f"{args.case} session {args.session}: AOT/JIT FAIL", flush=True)
            raise SystemExit(2) from None
        print(f"{args.case} session {args.session}: AOT/JIT PASS", flush=True)
    else:
        write_audit(args.directory, args.output)


if __name__ == "__main__":
    main()
