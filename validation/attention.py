"""FP16 fused attention forward for the Phase 4 RTX 4070 suite."""

from functools import partial
import math

import triton
import triton.language as tl

SHAPE_FIELDS = ("batch", "query_heads", "kv_heads", "seq_len", "head_dim", "causal")


@triton.jit
def _attention_tiles(q, k_base, v_base, rows, features, acc, normalizer, maximum,
                     start, stop, SEQ: tl.constexpr, D: tl.constexpr, BN: tl.constexpr,
                     SCALE_LOG2: tl.constexpr, MASK_CAUSAL: tl.constexpr):
    offsets = tl.arange(0, BN)
    for tile in tl.range(start, stop, BN):
        cols = tile + offsets
        addresses = cols[:, None] * D + features[None, :]
        k = tl.load(k_base + addresses, cols[:, None] < SEQ, other=0)
        v = tl.load(v_base + addresses, cols[:, None] < SEQ, other=0)
        logits = tl.dot(q, tl.trans(k)) * SCALE_LOG2
        valid = cols[None, :] < SEQ
        if MASK_CAUSAL:
            valid = valid & (rows[:, None] >= cols[None, :])
        logits = tl.where(valid, logits, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(logits, 1))
        correction = tl.math.exp2(maximum - new_max)
        weights = tl.math.exp2(logits - new_max[:, None])
        acc = tl.dot(weights.to(tl.float16), v, acc * correction[:, None])
        normalizer = normalizer * correction + tl.sum(weights, 1)
        maximum = new_max
    return acc, normalizer, maximum


@triton.jit
def fused_attention(Q, K, V, O, LSE, HQ: tl.constexpr, HK: tl.constexpr,
                    SEQ: tl.constexpr, D: tl.constexpr, CAUSAL: tl.constexpr,
                    SCALE_LOG2: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    query_block = tl.program_id(0)
    head_batch = tl.program_id(1)
    batch = head_batch // HQ
    query_head = head_batch % HQ
    kv_head = query_head // (HQ // HK)
    q_base = head_batch * SEQ * D
    kv_base = (batch * HK + kv_head) * SEQ * D
    first = query_block * BM
    rows = first + tl.arange(0, BM)
    features = tl.arange(0, D)
    addresses = rows[:, None] * D + features[None, :]
    q = tl.load(Q + q_base + addresses, rows[:, None] < SEQ, other=0)
    acc = tl.full((BM, D), 0, tl.float32)
    normalizer = tl.full((BM,), 0, tl.float32)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    if CAUSAL:
        acc, normalizer, maximum = _attention_tiles(
            q, K + kv_base, V + kv_base, rows, features, acc, normalizer, maximum,
            0, first, SEQ, D, BN, SCALE_LOG2, False,
        )
        acc, normalizer, maximum = _attention_tiles(
            q, K + kv_base, V + kv_base, rows, features, acc, normalizer, maximum,
            first, tl.minimum(first + BM, SEQ), SEQ, D, BN, SCALE_LOG2, True,
        )
    else:
        acc, normalizer, maximum = _attention_tiles(
            q, K + kv_base, V + kv_base, rows, features, acc, normalizer, maximum,
            0, SEQ, SEQ, D, BN, SCALE_LOG2, False,
        )
    output = acc / normalizer[:, None]
    tl.store(O + q_base + addresses, output.to(tl.float16), rows[:, None] < SEQ)
    tl.store(LSE + head_batch * SEQ + rows, maximum + tl.math.log2(normalizer), rows < SEQ)


def launch_spec(case, config):
    batch, hq, hk, seq, dim, causal = (case[key] for key in SHAPE_FIELDS)
    bm, bn = config["block_m"], config["block_n"]
    grid = (triton.cdiv(seq, bm), batch * hq, 1)
    constants = (hq, hk, seq, dim, causal, math.log2(math.e) / math.sqrt(dim), bm, bn)
    options = {"num_warps": config["num_warps"], "num_stages": config["num_stages"],
               "enable_fp_fusion": True}
    return grid, constants, options


def reference(q, k, v, causal):
    import torch

    batch, hq, seq, dim = q.shape
    hk = k.shape[1]
    output = torch.empty_like(q)
    lse = torch.empty((batch, hq, seq), dtype=torch.float32, device=q.device)
    q32 = q.float().reshape(batch, hk, hq // hk, seq, dim)
    kt32 = k.float().transpose(-1, -2).unsqueeze(2)
    v32 = v.float().unsqueeze(2)
    keys = torch.arange(seq, device=q.device)
    for start in range(0, seq, 64):
        end = min(start + 64, seq)
        logits = (q32[..., start:end, :] @ kt32) / math.sqrt(dim)
        if causal:
            logits.masked_fill_(keys[start:end, None] < keys[None, :], -torch.inf)
        out = torch.softmax(logits, dim=-1) @ v32
        output[:, :, start:end] = out.reshape(batch, hq, end - start, dim).half()
        lse[:, :, start:end] = torch.logsumexp(logits, dim=-1).reshape(batch, hq, end - start) * math.log2(math.e)
    return output, lse


def prepare_candidates(case, configs, seed, arch=89):
    import torch
    from triton.runtime.errors import PTXASError

    from validation.runner import jit_artifact

    batch, hq, hk, seq, dim, causal = (case[key] for key in SHAPE_FIELDS)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    inputs = [torch.randn((batch, heads, seq, dim), dtype=torch.float16, device="cuda",
                          generator=generator) for heads in (hq, hk, hk)]
    output = torch.empty_like(inputs[0])
    lse = torch.empty((batch, hq, seq), dtype=torch.float32, device="cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    expected, expected_lse = reference(*inputs, causal)
    operations, artifacts = {}, {}
    for config in configs:
        key = config["config_id"]
        grid, constants, options = launch_spec(case, config)
        args = (*inputs, output, lse, *constants)
        operation = partial(fused_attention[grid], *args, **options)
        try:
            compiled = fused_attention.run(*args, grid=grid, warmup=True, **options)
        except (triton.CompilationError, PTXASError) as error:
            artifacts[key] = {"status": "compile_failed", "reason": str(error)}
            continue
        row = {**jit_artifact(compiled, grid, arch), **options}
        artifacts[key] = row
        output.fill_(torch.nan)
        lse.fill_(torch.nan)
        try:
            operation()
        except (triton.OutOfResources, RuntimeError) as error:
            row.update(status="launch_failed", reason=str(error))
            continue
        torch.cuda.synchronize()
        row.update(registers_per_thread=compiled.n_regs, spill_count=compiled.n_spills)
        try:
            torch.testing.assert_close(output, expected, rtol=0.02, atol=0.003)
            torch.testing.assert_close(lse, expected_lse, rtol=0.0005, atol=0.002)
        except AssertionError as error:
            row.update(status="incorrect", reason=str(error))
            continue
        row["status"] = "ok"
        operations[key] = operation
    return operations, artifacts
