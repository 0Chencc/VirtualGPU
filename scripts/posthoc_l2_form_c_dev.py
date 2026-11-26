"""Post-hoc: effect of the cross-wave L2 reuse change on the RTX 4070 development sets.

Reports only; nothing is tuned on it. Three steps, so the old and new cores can
run from different environments on identical KernelIR:

    ir       regenerate the attention development KernelIR (CPU, Triton frontend)
    predict  nominal predictions of the installed core, both cache states
    score    Kendall tau-b and top-5 regret of two prediction files
    phase2   nominal predictions on the GB10 Phase 2 AOT IR; its labels are only in
             sealed reports, which are not read, so this step reports no score

GEMM uses the M2 development AOT IR. Labels are the medians over all sessions
and blocks of the published development reports (clean-cold for GEMM cold).
"""

import argparse
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validation.score import kendall_tau_b  # noqa: E402

CACHES = ("warm", "cold")


def sets(results):
    m2 = results / "m2"
    attn = results / "attention-fde371d"
    return {
        "gemm": {
            "manifest": json.loads((m2 / "suite.json").read_text()),
            "warm": m2,
            "cold": m2 / "clean-cold",
        },
        "attention": {
            "manifest": json.loads((attn / "manifest.json").read_text()),
            "warm": attn / "development",
            "cold": attn / "development",
        },
    }


def dev_cases(manifest):
    return [c for c in manifest["cases"] if c["split"] == "development"]


def attention_ir(case, config):
    # Same call as frozen_ir in validation/v02_predict.py, without its CUDA imports.
    from virtualgpu.frontend import extract_attention_kernel_ir

    from validation import attention
    from validation.attention_runner import _AOT_POINTERS

    _, constants, options = attention.launch_spec(case, config)
    hq, hk, seq, dim, causal, scale, bm, bn = constants
    knobs = dict(
        num_warps=options["num_warps"],
        num_stages=options["num_stages"],
        BM=bm,
        BN=bn,
        SCALE_LOG2=scale,
    )
    return extract_attention_kernel_ir(
        attention.fused_attention,
        case["batch"],
        hq,
        seq,
        dim,
        knobs,
        "sm_89",
        H_kv=hk,
        causal=causal,
        call_args=(*_AOT_POINTERS, hq, hk, seq, dim),
        use_cache=False,
    )


def make_ir(args):

    manifest = sets(args.results)["attention"]["manifest"]
    configs = {c["config_id"]: c for c in manifest["configs"]}
    for case in dev_cases(manifest):
        out = args.ir / case["case_id"]
        out.mkdir(parents=True, exist_ok=True)
        for cid in case["config_ids"]:
            try:
                ir = attention_ir(case, configs[cid])
            except Exception as error:  # noqa: BLE001
                print(f"{case['case_id']} {cid}: {error}", file=sys.stderr)
                continue
            (out / f"{cid}.json").write_text(json.dumps(ir))


def ir_path(args, family, case_id, cid):
    if family == "gemm":
        return args.results / "m2" / "aot" / case_id / f"{cid}.json"
    return args.ir / case_id / f"{cid}.json"


def predict(args):
    from virtualgpu import _core

    profile = subprocess.check_output(
        ["git", "show", "v0.2-final:profiles/rtx4070.json"], cwd=ROOT, text=True
    )
    out = {"build": _core.build_info(), "cases": {}}
    for family, data in sets(args.results).items():
        for case in dev_cases(data["manifest"]):
            rows = {cache: {} for cache in CACHES}
            for cid in case["config_ids"]:
                path = ir_path(args, family, case["case_id"], cid)
                if not path.exists():
                    continue
                for cache in CACHES:
                    try:
                        r = _core.predict(
                            path.read_text(), profile, l2_state=cache, draws=0, seed=17
                        )
                    except ValueError:
                        continue
                    if r["status"] == "ok":
                        rows[cache][cid] = r["time_ns"]
            out["cases"][f"{family}:{case['case_id']}"] = rows
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


def labels(data, case, cache):
    samples, bad = {}, set()
    for path in sorted(data[cache].glob(f"{case['case_id']}-session-*.json")):
        report = json.loads(path.read_text())
        bad |= {cid for cid, row in report["configs"].items() if row["status"] != "ok"}
        for s in report["samples"]:
            if s["cache"] == cache:
                samples.setdefault(s["config_id"], []).append(s["latency_ns"])
    return {cid: st.median(v) for cid, v in samples.items() if cid not in bad}


def case_scores(meas, pred):
    keys = [k for k in meas if k in pred]
    meas, pred = {k: meas[k] for k in keys}, {k: pred[k] for k in keys}
    best = min(meas, key=meas.get)
    order = sorted(pred, key=pred.get)
    return {
        "configs": len(keys),
        "tau_b": kendall_tau_b(meas, pred) if len(keys) > 2 else None,
        "top5_regret_pct": 100 * (min(meas[c] for c in order[:5]) / meas[best] - 1),
    }


def score(args):
    old = json.loads(args.old.read_text())
    new = json.loads(args.new.read_text())
    out = {"post_hoc": True, "old_build": old["build"], "new_build": new["build"], "families": {}}
    for family, data in sets(args.results).items():
        fam = {}
        for case in dev_cases(data["manifest"]):
            key = f"{family}:{case['case_id']}"
            for cache in CACHES:
                meas = labels(data, case, cache)
                o, n = old["cases"][key][cache], new["cases"][key][cache]
                changed = [abs(n[k] / o[k] - 1) for k in o if k in n]
                fam[f"{case['case_id']}:{cache}"] = {
                    "old": case_scores(meas, o),
                    "new": case_scores(meas, n),
                    "max_abs_prediction_change": max(changed, default=0.0),
                }
        summary = {}
        for cache in CACHES:
            rows = [v for k, v in fam.items() if k.endswith(f":{cache}")]
            for w in ("old", "new"):
                taus = [r[w]["tau_b"] for r in rows if r[w]["tau_b"] is not None]
                summary[f"{cache}_{w}_median_tau_b"] = st.median(taus)
                summary[f"{cache}_{w}_mean_top5_regret_pct"] = st.mean(
                    r[w]["top5_regret_pct"] for r in rows
                )
        out["families"][family] = {"summary": summary, "cases": fam}
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


def phase2(args):
    from virtualgpu import _core

    profile = subprocess.check_output(
        ["git", "show", "v0.2-final:profiles/gb10.json"], cwd=ROOT, text=True
    )
    out = {"build": _core.build_info(), "predictions": {}}
    for cohort in ("primary", "secondary"):
        for path in sorted((args.results / "phase2-gb10" / cohort / "aot").glob("*/*.json")):
            if path.stem == "failures":
                continue
            for cache in CACHES:
                try:
                    r = _core.predict(path.read_text(), profile, l2_state=cache, draws=0, seed=17)
                except ValueError:
                    continue
                if r["status"] == "ok":
                    key = f"{cohort}:{path.parent.name}:{cache}:{path.stem}"
                    out["predictions"][key] = r["time_ns"]
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("ir", "predict", "score", "phase2"))
    ap.add_argument(
        "--results", type=Path, required=True, help="directory holding m2/ and attention-fde371d/"
    )
    ap.add_argument("--ir", type=Path, required=True, help="attention KernelIR directory")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--old", type=Path)
    ap.add_argument("--new", type=Path)
    args = ap.parse_args()
    {"ir": make_ir, "predict": predict, "score": score, "phase2": phase2}[args.step](args)


if __name__ == "__main__":
    main()
