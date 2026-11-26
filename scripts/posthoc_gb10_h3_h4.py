"""Post-hoc counterfactuals for v0.2 GB10 GEMM hypotheses H3 and H4.

Diagnostic only. Runs a core built from `v0.2-final` on the archived formal
KernelIR with one profile or IR field changed, nominal profile, no draws.
See docs/results/analysis/v0.3-gb10-h3-h4-posthoc.md.

Usage:
    python scripts/posthoc_gb10_h3_h4.py --archive <results/v02-formal> \
        --profile <gb10.json> [--out result.json]
"""

import argparse
import copy
import json
import sys
from collections import OrderedDict
from pathlib import Path

from virtualgpu import _core

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validation.score import kendall_tau_b  # noqa: E402

COHORT = "gb10-gemm"
CASES = [
    "v02-gb10-decode-4",
    "v02-gb10-decode-40",
    "v02-gb10-mask-mnk-1153",
    "v02-gb10-mask-mnk-641",
    "v02-gb10-prefill-1792",
    "v02-gb10-prefill-896",
    "v02-gb10-resident-wave-above",
    "v02-gb10-resident-wave-below",
    "v02-gb10-resident-wave-exact",
]
KIB = 1024


def load_scores():
    path = ROOT / "validation" / "v02-holdout-scores.json"
    return json.loads(path.read_text())["cohorts"][COHORT]["cases"]


def predict(ir, prof, cache):
    return _core.predict(json.dumps(ir), json.dumps(prof), l2_state=cache, draws=0, seed=17)


class Case:
    def __init__(self, archive, scores, name, cache):
        self.archive, self.name, self.cache = archive, name, cache
        self.configs = scores[name]["cache"][cache]["core"]["configs"]

    def ir(self, cfg):
        return json.loads((self.archive / "ir" / COHORT / self.name / f"{cfg}.json").read_text())

    def measured(self, cfg):
        return self.configs[cfg]["measured_median_ns"]

    def score(self, prof, ir_edit=None):
        pred, rows = {}, {}
        for cfg in self.configs:
            ir = self.ir(cfg)
            if ir_edit:
                ir_edit(ir)
            r = predict(ir, prof, self.cache)
            pred[cfg] = r["time_ns"]
            rows[cfg] = {"pred_ns": r["time_ns"], "dram_bytes": r["dram_bytes"]}
        meas = {c: self.measured(c) for c in self.configs}
        best = min(meas, key=meas.get)
        order = sorted(pred, key=pred.get)
        regret = min(meas[c] for c in order[:5]) / meas[best] - 1
        return {
            "best_rank": order.index(best) + 1,
            "tau_b": kendall_tau_b(meas, pred),
            "top5_regret_pct": 100 * regret,
            "configs": rows,
        }


def edited(base, fn):
    p = copy.deepcopy(base)
    fn(p)
    return p


def zero_rescan(p):
    for q in p["measured"]["memory"]["l2_rescan_hit_rate_by_working_set"]:
        q["estimated_hit_rate"]["value"] = 0.0


def scale_mlp(factor, lo=0, hi=float("inf"), mode=None):
    def fn(p):
        for q in p["measured"]["memory"]["dram_bw_by_bytes_in_flight"]:
            if lo <= q["bytes_in_flight"] < hi and mode in (None, q["issue_mode"]):
                q["bw_gbps"]["value"] *= factor

    return fn


def monotone_mlp(p):
    pts = p["measured"]["memory"]["dram_bw_by_bytes_in_flight"]
    for mode in {q["issue_mode"] for q in pts}:
        peak = 0.0
        for q in sorted(
            (q for q in pts if q["issue_mode"] == mode), key=lambda q: q["bytes_in_flight"]
        ):
            peak = max(peak, q["bw_gbps"]["value"])
            q["bw_gbps"]["value"] = peak


def wave_floor(case, cfg, slots, cap):
    """DRAM bytes if each wave reads its unique A and B panels once and L2 is LRU at wave grain."""
    ir = case.ir(cfg)
    m, k = ir["tensors"][0]["shape"]
    n = ir["tensors"][1]["shape"][1]
    c = ir["metadata"]["config"]
    tm, tn = -(-m // c["BM"]), -(-n // c["BN"])
    size = {"A": c["BM"] * k * 2, "B": c["BN"] * k * 2}
    lru, used, dram = OrderedDict(), 0, 0
    for w0 in range(0, tm * tn, slots):
        pids = range(w0, min(w0 + slots, tm * tn))
        for key in dict.fromkeys([("A", p // tn) for p in pids] + [("B", p % tn) for p in pids]):
            if key in lru:
                lru.move_to_end(key)
                continue
            dram += size[key[0]]
            lru[key] = size[key[0]]
            used += size[key[0]]
            while used > cap:
                used -= lru.popitem(last=False)[1]
    return dram


def truncated_grid(case, cfg, prof, n):
    ir = case.ir(cfg)
    ir["launch"]["grid"][0] = n
    for acc in ir["access_map"].values():
        acc["entries"] = acc["entries"][:n]
    r = predict(ir, prof, case.cache)
    return {
        k: r[k] for k in ("time_ns", "load_bytes", "l2_hit_bytes", "merged_bytes", "dram_bytes")
    }


def prefetch_distance(d):
    def fn(ir):
        loop = ir["body"]["pipeline_loop"]
        loop["prefetch_distance"], loop["num_stages"] = d, d + 1
        for op in loop["ops"]:
            if op["kind"] == "gmem_load":
                op["stage_offset"] = d

    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    base = json.loads(args.profile.read_text())
    scores = load_scores()
    out = {
        "post_hoc": True,
        "build": _core.build_info(),
        "profile_revision": base["provenance"]["profile_revision"],
        "h3": {},
        "h4": {},
    }

    no_reuse = edited(base, zero_rescan)
    for name in CASES:
        for cache in ("warm", "cold"):
            case = Case(args.archive, scores, name, cache)
            out["h3"][f"{name}:{cache}"] = {
                "archived_profile": case.score(base),
                "rescan_zero": case.score(no_reuse),
            }

    p896 = Case(args.archive, scores, "v02-gb10-prefill-896", "cold")
    arch = json.loads((args.archive / "predictions.json").read_text())
    core = arch["cohorts"][COHORT]["v02-gb10-prefill-896"]["cold"]["core"]
    cap = base["specs"]["l2_size_bytes"]
    out["h3"]["prefill-896:cold:wave_floor"] = {
        cfg: {
            "measured_ns": p896.measured(cfg),
            "floor_dram_bytes": wave_floor(p896, cfg, 48 * core[cfg]["ctas_per_sm"], cap),
            "model_dram_bytes": core[cfg]["dram_bytes"],
        }
        for cfg in p896.configs
    }
    s4 = "m128n128k64-w8-s4-g1"
    out["h3"]["prefill-896:cold:truncated_grid"] = {
        n: truncated_grid(p896, s4, base, n) for n in (48, 96, 144, 192, 336)
    }
    distances = {}
    for cfg in (c for c in p896.configs if "-s4-" in c):
        distances[cfg] = {}
        for d in (0, 1, 2):
            ir = p896.ir(cfg)
            prefetch_distance(d)(ir)
            distances[cfg][str(d)] = predict(ir, base, "cold")["time_ns"]
    out["h3"]["prefill-896:cold:prefetch_distance"] = distances

    h4 = {
        "archived_profile": lambda p: None,
        "mlp_x1.02": scale_mlp(1.02),
        "mlp_x1.05": scale_mlp(1.05),
        "mlp_monotone": monotone_mlp,
        "cp_async_lt_512k_x1.05": scale_mlp(1.05, 0, 512 * KIB, "cp_async"),
        "cp_async_512k_1.5m_x1.05": scale_mlp(1.05, 512 * KIB, 1600 * KIB, "cp_async"),
        "cp_async_2.3m_x1.05": scale_mlp(1.05, 2000 * KIB, 2400 * KIB, "cp_async"),
        "cp_async_ge_4.6m_x1.05": scale_mlp(1.05, 4000 * KIB, float("inf"), "cp_async"),
    }
    for name in ("v02-gb10-resident-wave-exact", "v02-gb10-resident-wave-below"):
        for cache in ("warm", "cold"):
            case = Case(args.archive, scores, name, cache)
            res = {}
            for label, fn in h4.items():
                r = case.score(edited(base, fn))
                res[label] = {
                    k: v for k, v in r.items() if k != "configs" or label == "archived_profile"
                }
            out["h4"][f"{name}:{cache}"] = res

    text = json.dumps(out, indent=1, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
