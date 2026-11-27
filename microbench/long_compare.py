"""Compare a mode's legacy-length and frozen-length launches from one session.

The standard long-launch mechanism check (docs/components/long-launch-capture.md,
section 3): run the scheduled path at the legacy lengths (`x1`) and at the frozen
lengths (`long`), alternating, and report the shift of every point.

    python microbench/long_compare.py x1 MODE PILOT.json X1_SCHEDULE.txt
    python microbench/long_compare.py shift MODE --x1 A.json C.json --long B.json D.json

`x1` writes a schedule of each point's first pilot trial, which is the legacy length.
`shift` matches points by identity, takes each point's median launch time over steps
(per-step time) in every file, and reports per point the long over x1 ratio of the
pooled medians minus one, the implied fixed cost per launch F of the 1/steps model,
and rules 1 and 2 of the capture rules for each file.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

try:
    import capture_posthoc
    import long_launch
except ImportError:  # imported as microbench.long_compare
    from microbench import capture_posthoc, long_launch


def x1_schedule(pilot):
    """The first sizing trial of every point, in emission order."""
    lengths = []
    for _, point in long_launch.points(pilot):
        trials = point.get("sizing_trials")
        lengths.append(trials[0]["steps"] if trials else point["steps"])
    return "".join(f"{n}\n" for n in lengths)


def per_step(mode, capture):
    rows = {}
    for path, point in long_launch.points(capture):
        key = json.dumps(long_launch.identity(mode, path, point), sort_keys=True)
        steps = point["steps"]
        rows[key] = (steps, st.median(point["launch_ns_samples"]) / steps)
    return rows


def shift(mode, x1_captures, long_captures):
    x1 = [per_step(mode, c) for c in x1_captures]
    lg = [per_step(mode, c) for c in long_captures]
    keys = list(x1[0])
    if any(set(r) != set(keys) for r in x1 + lg):
        raise ValueError(f"{mode}: captures differ in points")
    out = []
    for key in keys:
        s_x1 = x1[0][key][0]
        s_long = lg[0][key][0]
        t_x1 = st.median(r[key][1] for r in x1)
        t_long = st.median(r[key][1] for r in lg)
        row = {
            "point": json.loads(key),
            "steps_x1": s_x1,
            "steps_long": s_long,
            "per_step_ns_x1": t_x1,
            "per_step_ns_long": t_long,
            "shift": t_long / t_x1 - 1,
        }
        if s_x1 != s_long:
            row["fixed_cost_ns"] = (t_x1 - t_long) / (1 / s_x1 - 1 / s_long)
        out.append(row)
    shifts = sorted(r["shift"] for r in out)
    return {
        "mode": mode,
        "points": len(out),
        "shift_median": st.median(shifts),
        "shift_p05": shifts[int(0.05 * (len(shifts) - 1))],
        "shift_p95": shifts[int(0.95 * (len(shifts) - 1))],
        "rules_x1": [capture_posthoc.dispersion_check(c)["rules"] for c in x1_captures],
        "rules_long": [capture_posthoc.dispersion_check(c)["rules"] for c in long_captures],
        "rows": out,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    a = sub.add_parser("x1")
    a.add_argument("mode")
    a.add_argument("pilot", type=Path)
    a.add_argument("schedule", type=Path)
    b = sub.add_parser("shift")
    b.add_argument("mode")
    b.add_argument("--x1", type=Path, nargs="+", required=True)
    b.add_argument("--long", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    if args.action == "x1":
        if args.schedule.exists():
            raise FileExistsError(args.schedule)
        args.schedule.write_text(x1_schedule(json.loads(args.pilot.read_text())))
    else:
        load = [json.loads(p.read_text()) for p in args.x1]
        load_long = [json.loads(p.read_text()) for p in args.long]
        print(json.dumps(shift(args.mode, load, load_long), indent=1))


if __name__ == "__main__":
    main()
