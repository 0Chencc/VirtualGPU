"""Git and binary provenance for validation artifacts."""

from pathlib import Path
import re
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


def source_info():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "python_dirty": bool(git("status", "--porcelain", "--untracked-files=all", "--", "python/")),
        "validation_dirty": bool(git("status", "--porcelain", "--untracked-files=all", "--", "validation/")),
    }


def read_core_revision(value):
    if not COMMIT_SHA.fullmatch(value):
        try:
            value = Path(value).read_text().strip()
        except OSError as error:
            raise ValueError("Provide a full core commit SHA or a file containing one") from error
    if not COMMIT_SHA.fullmatch(value):
        raise ValueError("Expected a 40-character core commit SHA")
    return value


def require_formal_build(build, revision, label):
    if not isinstance(build, dict):
        raise ValueError(f"{label}: missing core build metadata")
    if build.get("dirty") is not False or build.get("result_class") != "formal":
        raise ValueError(f"{label}: core build is dirty or classified as diagnostic")
    if not COMMIT_SHA.fullmatch(revision) or build.get("commit") != revision:
        raise ValueError(f"{label}: core build commit {build.get('commit')!r} does not match {revision}")
