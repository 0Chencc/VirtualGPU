"""Frozen v0.2 manifest and seal metadata shared by prediction and scoring."""

import json
import subprocess

from validation.holdout import sha256
from validation.provenance import REPO_ROOT


TAG = "v0.2-final"
DRAWS = 32
SEED = 17
COHORTS = {
    "rtx4070-gemm": "validation/v02-rtx4070-gemm-seal.json",
    "rtx4070-attention": "validation/attention-holdout-seal.json",
    "gb10-gemm": "validation/v02-gb10-gemm-seal.json",
    "gb10-attention": "validation/v02-gb10-attention-seal.json",
}
TARGETS = {"rtx4070": "sm_89", "gb10": "sm_121"}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT)


def case_inputs(name, revision):
    device, family = name.split("-")
    plan_path = f"validation/v02-{family}-holdout.json"
    plan_raw = git("show", f"{revision}:{plan_path}")
    plan = json.loads(plan_raw)
    selected = plan["devices"][device]
    seal_raw = git("show", f"{revision}:{COHORTS[name]}")
    if git("show", f"origin/main:{COHORTS[name]}") != seal_raw:
        raise ValueError(f"{name}: frozen seal differs from the published receipt")
    seal = json.loads(seal_raw)
    manifest_raw = plan_raw
    if name == "rtx4070-attention":
        manifest_raw = git("show", f"{revision}:{selected['manifest']}")
        if sha256(manifest_raw) != selected["manifest_sha256"] or sha256(seal_raw) != selected["seal_sha256"]:
            raise ValueError("The original attention holdout does not match preregistration")
        manifest = json.loads(manifest_raw)
        cases = [row for row in manifest["cases"] if row["case_id"] in selected["case_ids"]]
    else:
        manifest = plan
        cases = selected["cases"]
    expected = {f"{case['case_id']}-session-{i}.json" for case in cases for i in (1, 2, 3)}
    if (set(seal["reports"]) != expected or seal["manifest_sha256"] != sha256(manifest_raw)
            or set(seal["aot_sha256_by_case"]) != {case["case_id"] for case in cases}
            or seal.get("pending") or seal.get("protocol_failures_without_report")
            or any(row.get("status", "complete") != "complete" for row in seal["reports"].values())):
        raise ValueError(f"{name}: publish the complete holdout seal before archiving")
    if selected["target"] != TARGETS[device] or any(row["split"] != "holdout" for row in cases):
        raise ValueError(f"{name}: wrong target or split")
    evidence = {"plan_sha256": sha256(plan_raw), "manifest_sha256": sha256(manifest_raw),
                "seal_sha256": sha256(seal_raw)}
    return cases, {row["config_id"]: row for row in manifest["configs"]}, seal, evidence
