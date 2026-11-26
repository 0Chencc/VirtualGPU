"""Score the post-hoc GB10 captures against the frozen predictions.

Usage:
    python microbench/posthoc_gb10_score.py --captures <dir> --predictions <dir> --out scores.json
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
from pathlib import Path

# Profile revision -> git object holding it (the frozen predictions used these files).
PROFILES = {"0.21.0-posthoc": "v0.2-final", "0.22.0-posthoc": "78c664e"}


def profile_ceiling(git_rev):
    """dram_bw_by_active_sms at all SMs, the bandwidth cap the core applies."""
    text = subprocess.check_output(["git", "show", f"{git_rev}:profiles/gb10.json"], text=True)
    p = json.loads(text)
    sms = p["specs"]["sm_count"]
    curve = p["measured"]["memory"]["dram_bw_by_active_sms"]
    return next(q["bw_gbps"]["value"] for q in curve if q["active_sms"] == sms)


def med(xs):
    return st.median(xs)


def wave_rescan(raw, pred):
    keyed = {(p["ctas_per_sm"], p["rows_per_wave"], p["k_steps"]): p for p in pred["wave_rescan"]}
    rows = []
    for block in raw["blocks"]:
        cps = block["ctas_per_sm"]
        pts = block["points"]
        for x in pts:
            t1 = med(x.get("one_wave_up_ns_samples") or x["one_wave_down_ns_samples"])
            x["_t1"], x["_tw"] = t1, (med(x["four_wave_ns_samples"]) - t1) / 3
        for x in pts:
            ref = next(
                y
                for y in pts
                if y["rows_per_wave"] == x["rows_per_wave"]
                and y["reversed"] == x["reversed"]
                and y["footprint_ratio"] < 0.6
            )
            hit_b = ref["_tw"] / ref["footprint_bytes"]
            miss_b = x["_t1"] / x["footprint_bytes"]
            measured = (miss_b - x["_tw"] / x["footprint_bytes"]) / (miss_b - hit_b)
            p = keyed[(cps, x["rows_per_wave"], x["k_steps"])]
            f = p["rewave_hit_fraction"]
            forms = {
                "b": f["form_b"],
                "c": f["form_c_reversed" if x["reversed"] else "form_c_forward"],
            }
            if not x["reversed"]:
                forms["a"] = f["form_a"]
            rows.append(
                {
                    "ctas_per_sm": cps,
                    "rows_per_wave": x["rows_per_wave"],
                    "footprint_ratio": x["footprint_ratio"],
                    "reversed": bool(x["reversed"]),
                    "reference": x is ref,
                    "measured_hit": measured,
                    "predicted_hit": forms,
                }
            )
    scored = [r for r in rows if not r["reference"]]
    summary = {}
    for form in ("a", "b", "c"):
        for rev in (False, True):
            errs = [
                abs(r["predicted_hit"][form] - r["measured_hit"])
                for r in scored
                if r["reversed"] == rev and form in r["predicted_hit"]
            ]
            if errs:
                summary[f"{form}:{'reversed' if rev else 'forward'}"] = {
                    "points": len(errs),
                    "mean_abs_error": st.mean(errs),
                    "max_abs_error": max(errs),
                }
    return {"points": rows, "summary": summary}


def staged_mma(raw, pred):
    keyed = {
        (p["ctas_per_sm"], p["tokens"], p["stage_bytes"], p["mma_iters_per_stage"]): p
        for p in pred["staged_mma"]
    }
    rows = []
    for x in raw["points"]:
        p = keyed[(x["ctas_per_sm"], x["tokens"], x["stage_bytes"], x["mma_iters_per_stage"])]
        m = med(x["stage_turnaround_ns_samples"])
        rows.append(
            {
                **{
                    k: x[k] for k in ("ctas_per_sm", "tokens", "stage_bytes", "mma_iters_per_stage")
                },
                "measured_ns": m,
                "predicted_ns": p["stage_turnaround_ns"],
                "error": p["stage_turnaround_ns"] / m - 1,
            }
        )
    errs = [abs(r["error"]) for r in rows]
    return {
        "points": rows,
        "summary": {
            "points": len(rows),
            "mape": st.mean(errs),
            "median_abs_error": med(errs),
            "max_abs_error": max(errs),
        },
    }


def b_stream(raw, pred, profile_ceiling):
    keyed = {
        (
            p["tokens"],
            p["ctas_per_sm"],
            p["stage_bytes"],
            p["row_bytes"],
            p["mma_iters_per_stage"],
        ): p
        for p in pred["b_stream"]
    }
    rows = []
    for block in raw["blocks"]:
        probe = med(
            block["ceiling_before"]["gbps_samples"] + block["ceiling_after"]["gbps_samples"]
        )
        for x in block["points"]:
            p = keyed[
                (
                    block["tokens"],
                    x["ctas_per_sm"],
                    x["stage_bytes"],
                    x["row_bytes"],
                    x["mma_iters_per_stage"],
                )
            ]
            m = med(x["time_ns_samples"])
            pt = p["prediction"]["time_ns"]
            meas_frac = x["bytes"] / m / probe
            pred_frac = x["bytes"] / pt / profile_ceiling
            rows.append(
                {
                    "tokens": block["tokens"],
                    **{
                        k: x[k]
                        for k in (
                            "ctas_per_sm",
                            "stage_bytes",
                            "row_bytes",
                            "pattern",
                            "mma_iters_per_stage",
                        )
                    },
                    "probe_gbps": probe,
                    "measured_gbps": x["bytes"] / m,
                    "absolute_time_error": pt / m - 1,
                    "measured_fraction_of_probe": meas_frac,
                    "predicted_fraction_of_profile_ceiling": pred_frac,
                    "fraction_error": pred_frac - meas_frac,
                }
            )

    def summarize(key):
        errs = [abs(r[key]) for r in rows]
        return {
            "points": len(errs),
            "mean_abs": st.mean(errs),
            "median_abs": med(errs),
            "max_abs": max(errs),
        }

    return {
        "points": rows,
        "summary": {
            "absolute_time_error": summarize("absolute_time_error"),
            "fraction_error": summarize("fraction_error"),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cap = {
        m: json.loads((args.captures / f"{m}.json").read_text())
        for m in ("wave-rescan", "staged-mma", "b-stream")
    }
    out = {
        "post_hoc": True,
        "captures": {
            m: {
                "source_revision": c["source_revision"],
                "started_utc": c["started_utc"],
                "thermal_stability": c["thermal_stability"],
            }
            for m, c in cap.items()
        },
    }
    for rev, git_rev in PROFILES.items():
        pred = json.loads((args.predictions / f"predictions-profile-{rev}.json").read_text())
        ceiling = profile_ceiling(git_rev)
        out[rev] = {
            "wave_rescan": wave_rescan(cap["wave-rescan"]["raw"], pred),
            "staged_mma": staged_mma(cap["staged-mma"]["raw"], pred),
            "b_stream": b_stream(cap["b-stream"]["raw"], pred, ceiling),
        }
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
