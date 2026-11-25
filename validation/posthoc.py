"""Audited read access to formally published workload holdouts."""

from datetime import datetime, timezone
import fcntl
import json
import os
import stat
import subprocess

from validation.holdout import sha256
from validation.provenance import REPO_ROOT

NOTICE = "Post-hoc diagnostics do not change frozen conclusions and cannot enter any formal decision."
PUBLISHED_REPORTS = {
    "validation/holdout-seal.json": "validation/phase1-holdout-report.md",
    "validation/phase2-gb10-seal.json": "validation/phase2-gb10-report.md",
}


def published_seal(path):
    """Require a published release that identifies this exact seal digest."""
    if not path.resolve().is_relative_to(REPO_ROOT):
        raise ValueError("Post-hoc access requires a published workload holdout seal")
    seal_path = path.resolve().relative_to(REPO_ROOT).as_posix()
    report_path = PUBLISHED_REPORTS.get(seal_path)
    if report_path is None:
        raise ValueError("Post-hoc access is limited to the published Phase 1 and Phase 2 holdouts")

    def git(*args):
        result = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True)
        if result.returncode:
            raise ValueError("Post-hoc access requires a published seal and report on origin/main")
        return result.stdout

    revision = git("rev-parse", "--verify", "origin/main^{commit}").decode().strip()
    seal = git("show", f"{revision}:{seal_path}")
    report = git("show", f"{revision}:{report_path}")
    if seal != path.read_bytes() or f"`{sha256(seal)}`".encode() not in report:
        raise ValueError("Post-hoc seal does not match the published release")
    return seal_path, report_path, revision, json.loads(seal)


def read_sealed_bytes(path):
    # O_PATH pins the inode without opening it for data access. Restore mode
    # zero before reading any bytes through the short-lived read descriptor.
    handle = os.open(path, os.O_PATH | os.O_NOFOLLOW)
    try:
        info = os.fstat(handle)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0:
            raise ValueError(f"{path.name}: expected a sealed mode-zero report")
        reference = f"/proc/self/fd/{handle}"
        with_read_access = None
        try:
            os.chmod(reference, 0o400)
            with_read_access = os.open(reference, os.O_RDONLY)
        finally:
            try:
                os.chmod(reference, 0)
            except BaseException:
                if with_read_access is not None:
                    os.close(with_read_access)
                raise
        with os.fdopen(with_read_access, "rb") as reader:
            return reader.read()
    finally:
        os.close(handle)


def read_posthoc_reports(paths, release, revision, purpose, prediction_digest):
    seal_path, report_path, published_revision, seal = release
    gb10 = "manifest_sha256" not in seal
    names = [(f"{path.parent.parent.name}/{path.parent.name}/{path.name}" if gb10 else path.name)
             for path in paths]
    if len(set(names)) != len(names) or not set(names) <= seal["reports"].keys():
        raise ValueError("Post-hoc reports must be distinct entries in the published seal")

    audit = REPO_ROOT / "results" / "post-hoc-access.jsonl"
    audit.parent.mkdir(mode=0o700, exist_ok=True)
    fd = os.open(audit, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "result_class": "post-hoc",
            "prediction_revision": revision,
            "prediction_sha256": prediction_digest,
            "purpose": purpose,
            "published_revision": published_revision,
            "published_report": report_path,
            "seal": seal_path,
            "reports": names,
        }, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        verified = []
        for path, name in zip(paths, names, strict=True):
            data = read_sealed_bytes(path)
            if sha256(data) != seal["reports"][name]["sha256"]:
                raise ValueError(f"{name}: report digest does not match the published seal")
            verified.append(data)

    records = [json.loads(data) for data in verified]
    for name, record in zip(names, records, strict=True):
        if record["timestamp_utc"] != seal["reports"][name]["timestamp_utc"]:
            raise ValueError(f"{name}: inconsistent sealed report timestamp")
        if gb10:
            if record["gpu"]["arch"] != "sm_121" or record["cohort"] != name.split("/")[0]:
                raise ValueError(f"{name}: inconsistent GB10 holdout provenance")
        elif record["split"] != "holdout" or record["manifest_sha256"] != seal["manifest_sha256"]:
            raise ValueError(f"{name}: inconsistent RTX 4070 holdout provenance")
    return records


def mark_posthoc(report, purpose):
    report.update(result_class="post-hoc", title="Post-hoc workload diagnostic",
                  formal_eligible=False, notice=NOTICE, purpose=purpose)
    rows = [*report.get("configs", {}).values(), *report.get("pairs", ())]
    rows.extend(row for session in report["artifact_checks"].values() for row in session.values())
    for row in rows:
        row["result_class"] = "post-hoc"
