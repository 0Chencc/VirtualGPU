"""Post-hoc: effect of the cross-wave L2 reuse change on the opened v0.2 holdout.

Reports only; nothing is tuned on it. Reruns the installed core on the archived
formal KernelIR of all four v0.2 cohorts with the frozen v0.2 profiles, nominal
profile, and compares with the archived nominal predictions.

Usage:
    python scripts/posthoc_l2_form_c_impact.py --archive <results/v02-formal> --out impact.json
"""

import argparse
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

from virtualgpu import _core

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validation.score import kendall_tau_b  # noqa: E402


def frozen_profile(device):
    return subprocess.check_output(
        ["git", "show", f"v0.2-final:profiles/{device}.json"], cwd=ROOT, text=True
    )


def case_scores(meas, pred):
    best = min(meas, key=meas.get)
    order = sorted(pred, key=pred.get)
    return {
        "tau_b": kendall_tau_b(meas, pred),
        "top5_regret_pct": 100 * (min(meas[c] for c in order[:5]) / meas[best] - 1),
        "best_rank": order.index(best) + 1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    scores = json.loads((ROOT / "validation" / "v02-holdout-scores.json").read_text())["cohorts"]
    archive = json.loads((args.archive / "predictions.json").read_text())["cohorts"]
    out = {"post_hoc": True, "build": _core.build_info(), "cohorts": {}}
    for cohort, body in scores.items():
        profile = frozen_profile(cohort.split("-")[0])
        res = {}
        for case, cdata in body["cases"].items():
            for cache, cc in cdata["cache"].items():
                configs = {
                    k: v["measured_median_ns"]
                    for k, v in cc["core"]["configs"].items()
                    if v.get("nominal_latency_ns") and v.get("measured_median_ns")
                }
                old = {
                    k: archive[cohort][case][cache]["core"][k]["nominal_latency_ns"]
                    for k in configs
                }
                new = {}
                for k in configs:
                    ir = (args.archive / "ir" / cohort / case / f"{k}.json").read_text()
                    r = _core.predict(ir, profile, l2_state=cache, draws=0, seed=17)
                    new[k] = r["time_ns"]
                changed = [abs(new[k] / old[k] - 1) for k in configs]
                res[f"{case}:{cache}"] = {
                    "archived": case_scores(configs, old),
                    "changed": case_scores(configs, new),
                    "max_abs_prediction_change": max(changed),
                    "configs_changed_over_1pct": sum(c > 0.01 for c in changed),
                    "configs": len(configs),
                }
        summary = {}
        for cache in ("warm", "cold"):
            rows = [v for k, v in res.items() if k.endswith(f":{cache}")]
            taus = {
                w: [r[w]["tau_b"] for r in rows if r[w]["tau_b"] is not None]
                for w in ("archived", "changed")
            }
            summary[cache] = {
                f"{w}_{name}": value
                for w in ("archived", "changed")
                for name, value in (
                    ("median_tau_b", st.median(taus[w])),
                    ("mean_top5_regret_pct", st.mean(r[w]["top5_regret_pct"] for r in rows)),
                )
            }
        out["cohorts"][cohort] = {"summary": summary, "cases": res}
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
