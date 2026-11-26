"""Analyze DRAM probe sessions against the frozen decision rule.

docs/results/analysis/v0.3-gb10-dram-probe-plan.md: the ratio (workload over
the block's paired probe) is adopted only if its across-session coefficient of
variation is at most half that of the absolute workload and the two 95%
intervals do not overlap. Blocks whose probes before and after differ by more
than 3% are flagged as drift; results are given with and without them.

Usage:
    python microbench/probe_analyze.py session-1.json ... --out analysis.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path

DRIFT_BOUND = 0.03
BOOTSTRAP = 2000


def cv(values):
    return st.pstdev(values) / st.mean(values)


def medians(entries, key="gbps_samples"):
    return {e["name"]: st.median(e[key]) for e in entries}


def workload_points(block):
    return {
        f"{p['issue_mode']}:{p['bytes_in_flight']}": st.median(p["gbps_samples"])
        for p in block["workload"]["points"]
    }


def session_rows(session):
    rows = []
    for block in session["blocks"]:
        before = medians(block["probe_before"]["candidates"])
        after = medians(block["probe_after"]["candidates"])
        probe = {n: (before[n] + after[n]) / 2 for n in before}
        drift = {n: abs(before[n] - after[n]) / probe[n] > DRIFT_BOUND for n in probe}
        rows.append({"probe": probe, "drift": drift, "work": workload_points(block)})
    return rows


def across_session_cv(sessions, value, rng):
    """CV of session medians, with a 95% interval from resampling blocks within sessions."""
    point = cv([st.median(value(r) for r in rows) for rows in sessions])
    boots = []
    for _ in range(BOOTSTRAP):
        meds = [st.median(value(rng.choice(rows)) for _ in rows) for rows in sessions]
        boots.append(cv(meds))
    boots.sort()
    return {"cv": point, "ci95": [boots[int(0.025 * BOOTSTRAP)], boots[int(0.975 * BOOTSTRAP) - 1]]}


def analyze(sessions, exclude_drift):
    rng = random.Random(17)
    names = sorted(sessions[0][0]["probe"])
    points = sorted(sessions[0][0]["work"])
    out = {}
    for name in names:
        kept = [[r for r in rows if not (exclude_drift and r["drift"][name])] for rows in sessions]
        if any(len(rows) < 2 for rows in kept):
            out[name] = {"error": "too few blocks after drift exclusion"}
            continue
        res = {
            "blocks_per_session": [len(rows) for rows in kept],
            "probe_session_medians": [st.median(r["probe"][name] for r in rows) for rows in kept],
            "points": {},
        }
        for point in points:
            absolute = across_session_cv(kept, lambda r, p=point: r["work"][p], rng)
            ratio = across_session_cv(
                kept, lambda r, p=point, n=name: r["work"][p] / r["probe"][n], rng
            )
            within_abs = st.mean(cv([r["work"][point] for r in rows]) for rows in kept)
            within_ratio = st.mean(
                cv([r["work"][point] / r["probe"][name] for r in rows]) for rows in kept
            )
            res["points"][point] = {
                "absolute": {**absolute, "within_session_cv_mean": within_abs},
                "ratio": {**ratio, "within_session_cv_mean": within_ratio},
                "ratio_adopted": ratio["cv"] <= 0.5 * absolute["cv"]
                and ratio["ci95"][1] < absolute["ci95"][0],
            }
        out[name] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    raw = [json.loads(p.read_text()) for p in args.sessions]
    sessions = [session_rows(s) for s in raw]
    out = {
        "post_hoc": True,
        "drift_bound": DRIFT_BOUND,
        "sessions": [
            {
                "file": p.name,
                "first_block_utc": s["blocks"][0]["state_before"]["utc"],
                "thermal_stability": s["thermal_stability"],
                "drift_blocks": {n: sum(r["drift"][n] for r in rows) for n in rows[0]["drift"]},
            }
            for p, s, rows in zip(args.sessions, raw, sessions)
        ],
        "all_blocks": analyze(sessions, exclude_drift=False),
        "without_drift_blocks": analyze(sessions, exclude_drift=True),
    }
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
