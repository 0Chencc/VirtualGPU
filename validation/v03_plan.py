"""Read only the published v0.3 plan and its separate execution approvals."""

import json
from pathlib import Path
from subprocess import check_output, run

from validation.holdout import sha256
from validation.provenance import COMMIT_SHA, REPO_ROOT
from validation.v02_score import require_timestamp

PLAN = "validation/v03-manifest-release.json"
OPEN = "validation/v03-label-release.json"
APPROVALS = "validation/v03-approvals"
DEVICES = {"gb10": "sm_121", "rtx4070": "sm_89"}
FAMILIES = ("gemm", "attention")
INPUTS = {
    "history": "validation/v03-holdout-history.json",
    "aot": "validation/v03-holdout-aot-audit.json",
    **{family: f"validation/v03-{family}-holdout.json" for family in FAMILIES},
}
CAPTURE_CODE = ("python", ":(glob)validation/*.py", "microbench", "schemas")
SCORE_CODE = tuple(f"validation/{name}.py" for name in (
    "v03_score", "v03_plan", "v03_capture", "v02_score", "v02_inputs", "v02_predict",
    "holdout", "posthoc", "provenance", "score", "experiment", "pair_precision",
))


def git(*args):
    return check_output(["git", *args], cwd=REPO_ROOT)


def published_main():
    revision = git("rev-parse", "refs/remotes/origin/main").decode().strip()
    remote = git("ls-remote", "--exit-code", "origin", "refs/heads/main").decode().split()
    if remote != [revision, "refs/heads/main"]:
        raise ValueError("Fetch origin. The cached main differs from published main.")
    return revision


def require_ancestor(ancestor, descendant):
    if not COMMIT_SHA.fullmatch(ancestor) or run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant], cwd=REPO_ROOT
    ).returncode:
        raise ValueError(f"{ancestor} must be an ancestor of {descendant}")


def read_published(revision, path, status):
    raw = git("show", f"{revision}:{path}")
    record = json.loads(raw)
    if record.get("status") != status:
        raise ValueError(f"{path}: expected {status}")
    require_timestamp(record.get("timestamp_utc"), path)
    return record, sha256(raw)


def introduction(revision, path):
    return git("log", "-1", "--format=%H", revision, "--", path).decode().strip()


def checked_json(path, digest):
    raw = path.read_bytes()
    if sha256(raw) != digest:
        raise ValueError(f"{path}: SHA-256 mismatch")
    return json.loads(raw)


def load_plan(main):
    release, digest = read_published(main, PLAN, "released")
    require_ancestor(release["code_commit"], introduction(main, PLAN))
    if set(release["inputs"]) != set(INPUTS) or set(release["capture_profiles"]) != set(DEVICES):
        raise ValueError("The release must bind all inputs and both capture policies")
    inputs = {key: checked_json(REPO_ROOT / path, release["inputs"][key]) for key, path in INPUTS.items()}
    aot = inputs["aot"]
    if (aot["history_sha256"] != release["inputs"]["history"]
            or aot["manifests"] != {family: release["inputs"][family] for family in FAMILIES}):
        raise ValueError("AOT audit is for a different inventory or set of manifests")
    return release, digest, inputs


def require_capture(main, device, release_sha, case_id=None, session=None, attempt=1, failed_sha=None):
    plan_commit = introduction(main, PLAN)
    path = f"{APPROVALS}/{device}.json"
    record, digest = read_published(main, path, "capture_approved")
    approval_commit = introduction(main, path)
    require_ancestor(plan_commit, approval_commit)
    if approval_commit == plan_commit or record["device"] != device or record["release_sha256"] != release_sha:
        raise ValueError("Approval must follow and identify the published release and device")
    if device == "rtx4070" and record.get("desktop_window_confirmed") is not True:
        raise ValueError("RTX 4070 requires an approved desktop window")
    if attempt == 1:
        return digest, None
    retry_path = f"{APPROVALS}/retries/{device}/{case_id}-session-{session}-attempt-{attempt}.json"
    retry, retry_sha = read_published(main, retry_path, "retry_approved")
    retry_commit = introduction(main, retry_path)
    require_ancestor(approval_commit, retry_commit)
    if retry_commit == approval_commit or any(
        retry.get(key) != expected for key, expected in {
            "device": device, "case_id": case_id, "session": session, "attempt": attempt,
            "capture_approval_sha256": digest, "failed_receipt_sha256": failed_sha,
        }.items()
    ):
        raise ValueError("Retry approval must identify the exact failed attempt")
    return digest, retry_sha


def require_reviewed_code(revision, *, capture):
    head = git("rev-parse", "HEAD").decode().strip()
    require_ancestor(revision, head)
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("Use a clean checkout")
    if git("diff", "--name-only", revision, head, "--", *(CAPTURE_CODE if capture else SCORE_CODE)):
        raise ValueError("Code differs from the reviewed revision")
    return head


def aot_case(directory, inputs, device, family, case):
    row = inputs["aot"]["cohorts"][f"{device}-{family}"]["cases"][case["case_id"]]
    name = row["aot_file"]
    if Path(name).name != name:
        raise ValueError("Invalid AOT filename")
    records = checked_json(directory / name, row["aot_sha256"])
    if records.keys() != set(case["config_ids"]):
        raise ValueError("AOT audit must cover every configuration")
    return {
        "wave": row.get("wave_reference"),
        "sha256": row["aot_sha256"],
        "configs": {
            key: {**value, "status": "ok"} if value["status"] == "fits_profile" else value
            for key, value in records.items()
        },
    }
