"""Collect post-hoc spill counters with NVIDIA Nsight Compute."""

import argparse
import ast
import csv
import hashlib
import importlib
import inspect
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from microbench.build import build, find_cuda_root

ROOT = Path(__file__).resolve().parents[1]
POST_HOC = ROOT / "results/post-hoc"
ATTENTION_MANIFEST = ROOT / "validation/attention-suite.json"
ATTENTION_CASE = "b2-q32-kv4-s768-d128-causal"
ATTENTION_CONFIGS = ("m64n32-w4-s2", "m128n64-w4-s2", "m128n128-w8-s2")
SECTORS = tuple(f"l1tex__t_sectors_pipe_lsu_mem_local_op_{op}.sum" for op in ("ld", "st"))
WARP_METRICS = ("sass__inst_executed_local_loads", "sass__inst_executed_local_stores")
SOURCE_METRIC = "thread_inst_executed_true"
METRICS = (*SECTORS, *WARP_METRICS, SOURCE_METRIC)
LAUNCH = re.compile(r"^SPILL_COUNTER_LAUNCH (\d+) ([01]) (\d+) (\d+) (\d+)$", re.MULTILINE)
LOCAL_OP = re.compile(r"^(?:@!?U?P\w+\s+)?(LDL|STL)(?:\.|\s|$)")


def run(args, env):
    completed = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True)
    message = completed.stdout + "\n" + completed.stderr
    if "ERR_NVGPUCTRPERM" in message:
        raise RuntimeError(
            "ERR_NVGPUCTRPERM: GPU performance counters are unavailable. "
            "Apply the Windows driver permission change and reboot before retrying."
        )
    if completed.returncode or re.search(r"^==ERROR==", message, re.MULTILINE):
        raise RuntimeError(f"{Path(args[0]).name} failed:\n{message.strip()}")
    return completed.stdout


def verify_metrics(ncu, env):
    groups = {"profiling": SECTORS, "source": (*WARP_METRICS, SOURCE_METRIC)}
    for collection, names in groups.items():
        table = run(
            [ncu, "--query-metrics-collection", collection, "--query-metrics-mode", "all"], env
        )
        available = {line.split()[0] for line in table.splitlines() if line.strip()}
        if missing := set(names) - available:
            raise RuntimeError(f"NCU does not expose these {collection} metrics: {sorted(missing)}")


def collect(ncu, report_api, scratch, name, kernel, metrics, program, env):
    output = scratch / f"{name}.ncu-rep"
    log = run(
        [
            ncu,
            "--kernel-name",
            f"regex:^{kernel}$",
            "--kernel-name-base",
            "function",
            "--clock-control",
            "none",
            "--cache-control",
            "none",
            "--metrics",
            ",".join(metrics),
            "--export",
            str(output),
            *program,
        ],
        env,
    )
    if not output.is_file():
        raise RuntimeError(f"NCU collected no report for {name}: {log.strip()}")
    report = report_api.load_report(str(output))
    ranges = [report.range_by_idx(index) for index in range(report.num_ranges())]
    actions = [
        group.action_by_idx(index) for group in ranges for index in range(group.num_actions())
    ]
    if not actions:
        raise RuntimeError(f"NCU collected no launches for {name}")
    return report, actions, log


def count(metric, index=None):
    value = metric.value() if index is None else metric.value(index)
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or value % 1:
        raise ValueError(f"{metric.name()}: invalid count {value!r}")
    return int(value)


def require_metric(action, name):
    metric = action.metric_by_name(name)
    if metric is None:
        raise ValueError(f"{action.name()}: missing NCU metric {name}")
    return metric


def counters(action):
    return {metric: count(require_metric(action, metric)) for metric in (*SECTORS, *WARP_METRICS)}


def lane_counts(action, scope_for_pc):
    metric = require_metric(action, SOURCE_METRIC)
    if not metric.has_correlation_ids() or not metric.num_instances():
        raise ValueError(f"{action.name()}: NCU supplied no instruction-level counters")
    ids = metric.correlation_ids()
    by_scope = {}
    for index in range(metric.num_instances()):
        pc = ids.as_uint64(index)
        opcode = action.sass_by_pc(pc)
        if not opcode:
            raise ValueError(f"{action.name()}: missing SASS for profiled PC {pc:#x}")
        match = LOCAL_OP.match(opcode.strip())
        if not match:
            continue
        scope = by_scope.setdefault(scope_for_pc(pc), Counter())
        scope["load" if match[1] == "LDL" else "store"] += count(metric, index)
    return by_scope


def spill_row(metadata, actions):
    frame, serial, reloads, steps, ctas = metadata
    if steps != 128 or len(actions) != 2:
        raise ValueError("Expected paired zero- and 128-step spill launches")
    totals = []
    for action in actions:
        dimensions = [
            count(require_metric(action, name))
            for name in ("launch__grid_size", "launch__block_size")
        ]
        if dimensions != [ctas, 128]:
            raise ValueError("Spill launch geometry differs from its metadata")
        lanes = lane_counts(action, lambda pc: "kernel").get("kernel", {})
        totals.append(
            counters(action) | {f"thread_local_{op}": lanes.get(op, 0) for op in ("load", "store")}
        )
    delta = {name: totals[1][name] - before for name, before in totals[0].items()}
    if min(delta.values()) < 0:
        raise ValueError(
            f"Negative spill count after subtracting the zero-step control: {metadata}"
        )
    return {
        "frame_bytes_per_thread": frame,
        "dependency_mode": "serial_chain" if serial else "batch_before_dep",
        "reload_count": reloads,
        "ctas": ctas,
        "threads_per_cta": 128,
        "steps": steps,
        "launch_totals": {"zero_step": totals[0], "measured": totals[1]},
        "per_cta_iteration": {name: value / (ctas * steps) for name, value in delta.items()},
        "per_thread_iteration": {
            name: delta[name] / (ctas * steps * 128)
            for name in ("thread_local_load", "thread_local_store")
        },
    }


def spill_rows(log, actions):
    launches = [tuple(map(int, match)) for match in LAUNCH.findall(log)]
    if len(launches) != 120 or len(actions) != len(launches) or len(set(launches)) != len(launches):
        raise ValueError("Expected 60 distinct zero-step and 60 matching measured spill launches")
    rows = []
    for index in range(0, len(launches), 2):
        control, measured = launches[index : index + 2]
        if control[:3] != measured[:3] or control[3] != 0 or control[4] != measured[4]:
            raise ValueError("Spill control order does not match its stepped launch")
        rows.append(spill_row(measured, actions[index : index + 2]))
    return rows


def attention_source(helper, kernel):
    main = helper.replace("_attention_tiles", "_attention_main_tiles")
    diagonal = helper.replace("_attention_tiles", "_attention_diagonal_tiles")
    call = "_attention_tiles("
    calls = kernel.split(call)
    if len(calls) != 4:
        raise ValueError("Expected three call sites in fused_attention")
    return (
        "import triton\nimport triton.language as tl\n\n"
        + main
        + "\n\n"
        + diagonal
        + "\n\n"
        + calls[0]
        + "_attention_main_tiles("
        + calls[1]
        + "_attention_diagonal_tiles("
        + calls[2]
        + "_attention_main_tiles("
        + calls[3]
    )


def attention_target(scratch, config_id, variant):
    # This entry point is run only under NCU. Keep GPU imports out of CPU tooling.
    import torch
    import triton
    from virtualgpu.frontend.compiler import hash_ptx

    from validation.attention import _attention_tiles, fused_attention, launch_spec

    suite = json.loads(ATTENTION_MANIFEST.read_bytes())
    case = next(case for case in suite["cases"] if case["case_id"] == ATTENTION_CASE)
    config = next(config for config in suite["configs"] if config["config_id"] == config_id)
    source = attention_source(
        inspect.getsource(_attention_tiles.fn), inspect.getsource(fused_attention.fn)
    )
    file = scratch / "spill_attention_source.py"
    file.write_text(source)
    sys.path.insert(0, str(scratch))
    # Each target starts a new process and imports the file just written above.
    annotated = importlib.import_module(file.stem).fused_attention
    grid, constants, options = launch_spec(case, config)
    batch, hq, hk, seq, dim = (
        case[k] for k in ("batch", "query_heads", "kv_heads", "seq_len", "head_dim")
    )
    q, k, v, out = [
        torch.zeros((batch, head, seq, dim), dtype=torch.float16, device="cuda")
        for head in (hq, hk, hk, hq)
    ]
    lse = torch.empty((batch, hq, seq), dtype=torch.float32, device="cuda")
    args = (q, k, v, out, lse, *constants)
    compiled = [
        kernel.run(*args, grid=grid, warmup=True, **options)
        for kernel in (fused_attention, annotated)
    ]

    launches = [kernel[grid] for kernel in compiled]

    def identity(kernel):
        return {
            "ptx_sha256": hash_ptx(kernel.asm["ptx"]),
            "registers_per_thread": kernel.n_regs,
            "shared_bytes": kernel.metadata.shared,
            "runtime_local_spill_count": kernel.n_spills,
        }

    if identity(compiled[0]) != identity(compiled[1]):
        raise ValueError("Attention source annotation changed PTX or launch resources")
    launches[("original", "annotated").index(variant)](*args)
    torch.cuda.synchronize()

    print(
        json.dumps(
            {"artifact": identity(compiled[0]), "grid": grid, "triton_version": triton.__version__}
        ),
        flush=True,
    )


def scope_ranges(source):
    return {
        part: (node.lineno, node.end_lineno)
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        for part in ("main", "diagonal")
        if node.name == f"_attention_{part}_tiles"
    }


def attention_iterations(case, config):
    bm, bn, seq = config["block_m"], config["block_n"], case["seq_len"]
    starters = range(0, seq, bm)
    copies = case["batch"] * case["query_heads"]
    return {
        "main": copies * sum(math.ceil(start / bn) for start in starters),
        "diagonal": copies * sum(math.ceil(min(bm, seq - start) / bn) for start in starters),
    }


def attention_row(case, config, action, original, line_ranges):
    def scope(pc):
        info = action.source_info(pc)
        if info is not None and Path(info.file_name()).name == "spill_attention_source.py":
            for name, (start, end) in line_ranges.items():
                if start <= info.line() <= end:
                    return name
        return "outside_or_unattributed"

    attributed = lane_counts(action, scope)
    control = lane_counts(original, lambda pc: "all").get("all", {})
    if any(
        sum(bucket[op] for bucket in attributed.values()) != control.get(op, 0)
        for op in ("load", "store")
    ):
        raise ValueError("Attention source annotation changed dynamic local instruction counts")
    iterations = attention_iterations(case, config)
    loops = {}
    for part, visits in iterations.items():
        counts = attributed.get(part, Counter())
        loops[part] = {
            "cta_iterations": visits,
            "thread_local_load": counts["load"],
            "thread_local_store": counts["store"],
            "per_cta_iteration": {op: counts[op] / visits for op in ("load", "store")},
            "per_thread_iteration": {
                op: counts[op] / (visits * config["num_warps"] * 32) for op in ("load", "store")
            },
        }
    other = attributed.get("outside_or_unattributed", Counter())
    return {
        "config_id": config["config_id"],
        "kernel_totals": counters(original),
        "loops": loops,
        "outside_or_unattributed": {op: other[op] for op in ("load", "store")},
    }


def profile_attention(ncu, api, scratch, env):
    suite = json.loads(ATTENTION_MANIFEST.read_bytes())
    case = next(case for case in suite["cases"] if case["case_id"] == ATTENTION_CASE)
    configs = {row["config_id"]: row for row in suite["configs"]}
    rows = []
    for config_id in ATTENTION_CONFIGS:
        program = [
            sys.executable,
            "-m",
            "microbench.spill_counters",
            "--attention-target",
            str(scratch),
            config_id,
        ]
        action_sets = []
        identity = None
        for variant in ("original", "annotated"):
            report, actions, log = collect(
                ncu,
                api,
                scratch,
                f"{config_id}-{variant}",
                "fused_attention",
                METRICS,
                [*program, variant],
                env,
            )
            records = [
                json.loads(line) for line in log.splitlines() if line.startswith('{"artifact":')
            ]
            if len(actions) != 1 or len(records) != 1 or (identity and identity != records[0]):
                raise ValueError(f"{config_id}: inconsistent attention launch identity")
            identity = records[0]
            action_sets.append((report, actions[0]))
        source = (scratch / "spill_attention_source.py").read_text()
        original, annotated = (item[1] for item in action_sets)
        config = configs[config_id]
        geometry = (math.prod(identity["grid"]), 32 * config["num_warps"])
        for action in (original, annotated):
            reported = tuple(
                count(require_metric(action, metric))
                for metric in ("launch__grid_size", "launch__block_size")
            )
            if reported != geometry:
                raise ValueError(f"{config_id}: attention launch geometry changed")
        row = attention_row(case, config, annotated, original, scope_ranges(source))
        rows.append({**row, **identity})
    return {"case": case, "configs": rows}


def publish(path, data):
    raw = json.dumps(data, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as scratch:
        staging = Path(scratch) / "complete.json"
        staging.write_text(raw)
        os.link(staging, path)


def measure(ncu, nvcc):
    env = dict(os.environ, LC_ALL="C")
    version = run([ncu, "--version"], env).strip()
    if not re.search(r"^Version 2026\.4\.", version, re.MULTILINE):
        raise ValueError("This collector requires Nsight Compute 2026.4")
    gpu = run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=uuid,name,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        env,
    )
    uuid, name, driver, capability = next(csv.reader(gpu.splitlines(), skipinitialspace=True))
    if capability != "8.9" or "RTX 4070" not in name:
        raise ValueError("This counter suite targets the Bifrost RTX 4070")
    env["CUDA_VISIBLE_DEVICES"] = uuid
    revision = run(["git", "rev-parse", "HEAD"], env).strip()
    if run(["git", "status", "--porcelain"], env):
        raise ValueError("Run the counter suite from a clean committed checkout")
    cuda = find_cuda_root(nvcc)
    if cuda is None:
        raise ValueError("nvcc not found in the virtual environment. Pass --nvcc /path/to/nvcc.")
    compiler = run([str(cuda / "bin/nvcc"), "--version"], env).strip()
    api_path = Path(ncu).resolve().parent / "extras/python"
    sys.path.insert(0, str(api_path))
    api = importlib.import_module("ncu_report")

    with tempfile.TemporaryDirectory(prefix="virtualgpu-spill-") as directory:
        scratch = Path(directory)
        binary = build("sm_89", scratch / "vgpu_bench", nvcc)
        probe, actions, _ = collect(
            ncu,
            api,
            scratch,
            "permission",
            "spill_reload_latency_kernel",
            SECTORS[:1],
            [str(binary), "--mode", "spill-counter-probe"],
            env,
        )
        if len(actions) != 1:
            raise ValueError("Performance counter permission probe expected one CUDA launch")
        count(require_metric(actions[0], SECTORS[0]))
        del actions, probe
        verify_metrics(ncu, env)
        report, actions, log = collect(
            ncu,
            api,
            scratch,
            "spill",
            "spill_reload_latency_kernel",
            METRICS,
            [str(binary), "--mode", "spill-counters"],
            env,
        )
        spill = spill_rows(log, actions)
        del actions, report
        env.update(TRITON_CACHE_DIR=str(scratch / "triton"), TRITON_DISABLE_LINE_INFO="0")
        attention = profile_attention(ncu, api, scratch, env)
        return {
            "schema_version": 1,
            "result_class": "post-hoc",
            "provenance": {
                "utc": utc(),
                "commit": revision,
                "gpu": name,
                "gpu_uuid": uuid,
                "driver": driver,
                "ncu": version,
                "nvcc": compiler,
                "metrics": list(METRICS),
                "target": "sm_89",
                "clock_control": "none",
                "cache_control": "none",
                "spill_binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "attention_manifest_sha256": hashlib.sha256(
                    ATTENTION_MANIFEST.read_bytes()
                ).hexdigest(),
            },
            "spill_reload_latency_by_count": spill,
            "causal_attention": attention,
        }


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--nvcc", type=Path)
    parser.add_argument("--attention-target", nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.attention_target:
        directory, config, variant = args.attention_target
        attention_target(Path(directory), config, variant)
        return 0
    try:
        ncu = shutil.which(args.ncu)
        if ncu is None:
            raise ValueError("Nsight Compute CLI was not found. Pass --ncu /path/to/ncu.")
        data = measure(ncu, args.nvcc)
        stamp = datetime.fromisoformat(data["provenance"]["utc"]).strftime("%Y%m%dT%H%M%S%fZ")
        output = POST_HOC / f"spill-counters-{stamp}.json"
        publish(output, data)
    except (OSError, ImportError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
