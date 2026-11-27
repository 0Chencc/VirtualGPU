"""Registered same-device check of the measured non-MMA issue rate.

The rule is in docs/results/analysis/v0.3-non-mma-same-device-check-registration.md. It reads
only a qualified 0.23.0 campaign's merged raws (`merged-mma.json`, `merged-sfu-shfl.json`) and
the profile built from it, runs the pure FFMA and the FFMA and LOP3 mix kernels through the
core, and prints the verdict of reading B with reading A and the per-MHz comparison beside it.

    python scripts/non_mma_check.py --profile gb10=GB10.json --run gb10=GB10_RUN_DIR \\
        [--profile rtx4070=RTX4070.json --run rtx4070=RTX4070_RUN_DIR] [--out OUT.json]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "microbench")]

import collect_profile  # noqa: E402
import recheck_mechanisms as rm  # noqa: E402

MISS_MAX = 0.10
FFMA_PER_ITERATION = 8  # fp32_fma_kernel: 8 fmaf per iteration
MIX_PER_ITERATION = 128  # alu_issue_kernel: 64 FFMA and 64 LOP3 per iteration


def missed(predicted, measured):
    """|predicted / measured - 1| strictly above 10%, both sides, without dividing."""
    return abs(predicted - measured) > MISS_MAX * measured


def per_mhz(rates, launch):
    """Median over rounds of each round's rate over the mean of its two clock probes."""
    clocks = [(b + a) / 2 for b, a in zip(launch["sm_mhz_before"], launch["sm_mhz_after"])]
    return st.median(r / c for r, c in zip(rates, clocks, strict=True))


def points(profile, mma, sfu):
    """The FFMA point and the mix curve as the collector reads them: warps, frozen steps,
    per-round rates and the launch record."""
    sm = profile["specs"]["sm_count"]
    fma = mma["fp32_fma"]
    ffma = {
        "kernel": "ffma",
        "warps": fma["warps_per_sm"],
        "steps": fma["launch"]["length"],
        "per_iteration": FFMA_PER_ITERATION,
        "rounds": collect_profile.ffma_issue_samples(mma["fp32_fma_tflops_samples"], sm),
        "launch": fma["launch"],
    }
    curve = [
        {
            "kernel": "mix",
            "warps": q["warps_per_sm"],
            "steps": q["launch"]["length"],
            "per_iteration": MIX_PER_ITERATION,
            "rounds": [float(x) for x in q["rate_gwarp_inst_per_sm_samples"]],
            "launch": q["launch"],
        }
        for q in sfu["alu_issue_rate_per_sm_gips"]
    ]
    peak = collect_profile.peak_point(sfu["alu_issue_rate_per_sm_gips"])
    mix = curve[sfu["alu_issue_rate_per_sm_gips"].index(peak)]
    return ffma, mix, curve


def ir(point, length, reading):
    """Reading A: the instructions as pipeline loop ops with no barrier. Reading B: one
    prologue op of all of them beside the anchor MMA, with an empty loop."""
    sm = point["sm_count"]
    if reading == "A":
        k = rm.kernel(grid=sm, warps=point["warps"], steps=length, alu=point["per_iteration"])
        k["resources"]["barriers"] = 0
        return k
    k = rm.kernel(grid=sm, warps=point["warps"], steps=1)
    k["body"]["prologue"]["ops"].append(rm.counted("alu", "alu", point["per_iteration"] * length))
    return k


def predicted_rates(profile_text, pts, reading):
    """Gwarp-inst/s per SM from the makespan difference of the frozen length and its half."""
    from virtualgpu import _core

    irs = []
    for p in pts:
        irs += [json.dumps(ir(p, n, reading)) for n in (p["steps"], p["steps"] // 2)]
    out = _core.predict_batch(irs, profile_text, l2_state="warm", draws=0, seed=17)
    for r in out:
        if r["status"] != "ok":
            raise ValueError(r["reason"])
    spans = [r["makespan_ns"] for r in out]
    return [
        p["warps"] * p["per_iteration"] * (p["steps"] - p["steps"] // 2) / (a - b)
        for p, a, b in zip(pts, spans[0::2], spans[1::2], strict=True)
    ]


def check_gpu(profile, profile_text, mma, sfu):
    field = profile["measured"]["compute"]["non_mma_issue_rate_per_sm_gips"]
    if "capture_protocol" not in profile["provenance"] or field.get("source") != "measured":
        return {"verdict": "not decided", "reason": "not a qualified 0.23.0 profile"}
    ffma, mix, curve = points(profile, mma, sfu)
    for p in [ffma, *curve]:
        p["sm_count"] = profile["specs"]["sm_count"]
        p["measured"] = st.median(p["rounds"])
    judged = [ffma, mix]
    pred = {r: predicted_rates(profile_text, judged, r) for r in ("A", "B")}
    clock = field["value"] / field["per_cycle"]["value"]
    rows = []
    for i, p in enumerate(judged):
        m = p["measured"]
        rows.append(
            {
                "kernel": p["kernel"],
                "warps_per_sm": p["warps"],
                "steps": p["steps"],
                "measured": m,
                "predicted_b": pred["B"][i],
                "miss_b": pred["B"][i] / m - 1,
                "rejects": missed(pred["B"][i], m),
                "predicted_a": pred["A"][i],
                "miss_a": pred["A"][i] / m - 1,
                "measured_per_mhz": per_mhz(p["rounds"], p["launch"]),
                "predicted_b_per_mhz": pred["B"][i] / clock,
            }
        )
    curve_b = predicted_rates(profile_text, curve, "B")
    return {
        "verdict": "reject" if any(r["rejects"] for r in rows) else "pass",
        "field": {k: field[k] for k in ("value", "method")} | {"clock_mhz": clock},
        "winner": "ffma" if field["method"] == collect_profile.FFMA_ISSUE_METHOD else "mix",
        "ratio_mix_over_ffma": mix["measured"] / ffma["measured"],
        "judged": rows,
        "mix_curve_b": [
            {"warps_per_sm": p["warps"], "measured": p["measured"], "miss_b": b / p["measured"] - 1}
            for p, b in zip(curve, curve_b, strict=True)
        ],
    }


def pairs(values):
    return dict(v.split("=", 1) for v in values)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", action="append", required=True, help="GPU=PROFILE.json")
    ap.add_argument("--run", action="append", required=True, help="GPU=RUN_DIR")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    profiles, runs = pairs(args.profile), pairs(args.run)
    result = {}
    for gpu, path in profiles.items():
        text = Path(path).read_text()
        run = Path(runs[gpu])
        result[gpu] = check_gpu(
            json.loads(text),
            text,
            json.loads((run / "merged-mma.json").read_text()),
            json.loads((run / "merged-sfu-shfl.json").read_text()),
        )
    out = json.dumps(result, indent=1, sort_keys=True)
    if args.out:
        args.out.write_text(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
