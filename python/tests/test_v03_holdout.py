import json
from contextlib import nullcontext

import pytest

from validation import v03_audit, v03_capture, v03_plan, v03_score
from validation.holdout import sha256

CASE = {"case_id": "synthetic", "config_ids": ["config"]}
FAMILY = "gemm"
DEVICE = "gb10"
HASH = "a" * 64


def test_capture_authorization_follows_the_manifest_release(monkeypatch):
    plan_commit = "a" * 40
    approval_commit = "b" * 40
    ancestry = []

    def read(_main, path, status):
        assert (path, status) == ("validation/v03-approvals/gb10.json", "capture_approved")
        return {"device": "gb10", "release_sha256": HASH}, "receipt-digest"

    monkeypatch.setattr(v03_plan, "read_published", read)
    monkeypatch.setattr(v03_plan, "require_ancestor", lambda a, b: ancestry.append((a, b)))
    monkeypatch.setattr(v03_plan, "introduction", lambda _, path: plan_commit if path == v03_plan.PLAN else approval_commit)
    assert v03_plan.require_capture("main", "gb10", HASH) == ("receipt-digest", None)
    assert ancestry == [(plan_commit, approval_commit)]
    with pytest.raises(ValueError, match="must follow"):
        v03_plan.require_capture("main", "gb10", "wrong-digest")
    approval_commit = plan_commit
    with pytest.raises(ValueError, match="must follow"):
        v03_plan.require_capture("main", "gb10", HASH)


def test_missing_device_approval_never_reaches_lock_or_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(v03_capture, "published_main", lambda: "main")
    monkeypatch.setattr(v03_capture, "load_plan", lambda _: (
        {"code_commit": "code"}, HASH, {FAMILY: {"devices": {DEVICE: {"cases": [CASE]}}}},
    ))
    monkeypatch.setattr(v03_capture, "require_reviewed_code", lambda *args, **kwargs: "clean")
    monkeypatch.setattr(v03_capture, "require_capture", lambda *args: pytest.fail("missing approval"))
    monkeypatch.setattr(v03_capture, "require_lock", lambda: pytest.fail("reached lock"))
    with pytest.raises(pytest.fail.Exception, match="missing approval"):
        v03_capture.capture(tmp_path, tmp_path, DEVICE, FAMILY, CASE["case_id"], 1, 1)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("cold,expected", [(20, "complete"), (5, "protocol_failed")])
def test_sealing_and_audit_validate_the_probe_without_reading_labels(monkeypatch, tmp_path, cold, expected):
    release = {"code_commit": "code", "capture_profiles": {DEVICE: HASH}, "inputs": {FAMILY: HASH}}
    inputs = {
        FAMILY: {"devices": {DEVICE: {"cases": [CASE]}}, "configs": [{"config_id": "config"}]},
        "aot": {"cohorts": {f"{DEVICE}-{FAMILY}": {
            "profile_sha256": HASH, "cases": {CASE["case_id"]: {"aot_sha256": HASH}},
        }}},
    }
    for module in (v03_capture, v03_audit):
        monkeypatch.setattr(module, "published_main", lambda: "main")
        monkeypatch.setattr(module, "load_plan", lambda _: (release, HASH, inputs))
        monkeypatch.setattr(module, "require_capture", lambda *args: (HASH, None))
    monkeypatch.setattr(v03_capture, "require_reviewed_code", lambda *args, **kwargs: "clean")
    monkeypatch.setattr(v03_capture, "checked_json", lambda *args: {})
    monkeypatch.setattr(v03_capture, "aot_case", lambda *args: {"sha256": HASH})
    monkeypatch.setattr(v03_capture, "require_lock", lambda: None)
    monkeypatch.setattr(v03_capture.host_quiet, "wait_for_quiet_host", lambda *args: {})
    monkeypatch.setattr(v03_capture, "host_trace", lambda samples: nullcontext())
    monkeypatch.setattr(v03_capture, "run_session", lambda *args: ({
        "timestamp_utc": v03_capture.utc(),
        "complete_blocks": 10,
        "spark": {"after": {"compute_processes": {"status": 0, "stdout": ""}}},
        "configs": {"config": {"status": "ok"}},
        "cache_control_ns": {"warm": [10], "cold": [cold]},
        "samples": [{"latency_ns": 12345}],
    }, "kernel"))
    capture = v03_capture.capture
    if expected == "complete":
        capture(tmp_path, tmp_path, DEVICE, FAMILY, CASE["case_id"], 1, 1)
    else:
        with pytest.raises(SystemExit):
            capture(tmp_path, tmp_path, DEVICE, FAMILY, CASE["case_id"], 1, 1)

    label = tmp_path / "gb10-gemm" / "sealed" / "synthetic-session-1.json"
    v03_capture.check_sealed(label)
    out = tmp_path / "audit.json"
    v03_audit.audit(tmp_path, out, DEVICE, FAMILY)
    audit = json.loads(out.read_text())
    receipts = audit["reports"] if expected == "complete" else audit["failures"]
    assert receipts[label.name]["status"] == expected
    assert "12345" not in out.read_text()
    assert len(audit["pending"]) == (2 if expected == "complete" else 3)


def test_freeze_requires_separate_opening_approval_before_any_report_access(monkeypatch, tmp_path):
    monkeypatch.setattr(v03_score, "read_published", lambda _, path, status: (
        {"timestamp_utc": "2026-11-27T00:00:00Z", "freeze_sha256": HASH}, HASH,
    ))
    monkeypatch.setattr(v03_score, "introduction", lambda *args: "same-commit")
    monkeypatch.setattr(v03_score, "require_ancestor", lambda *args: None)
    monkeypatch.setattr(v03_score, "git", lambda *args: pytest.fail("opening gate was skipped"))
    with pytest.raises(ValueError, match="must follow"):
        v03_score.verified_freeze("main", {}, HASH, tmp_path)


def test_aot_resource_status_and_digest(tmp_path):
    file = tmp_path / "aot.json"
    file.write_text(json.dumps({"config": {"status": "fits_profile"}}))
    row = {"aot_file": file.name, "aot_sha256": sha256(file.read_bytes())}
    inputs = {"aot": {"cohorts": {"gb10-gemm": {"cases": {"synthetic": row}}}}}
    assert v03_plan.aot_case(tmp_path, inputs, DEVICE, FAMILY, CASE)["configs"]["config"]["status"] == "ok"
    file.write_text("{}")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        v03_plan.aot_case(tmp_path, inputs, DEVICE, FAMILY, CASE)
