# Phase 4 attention suite proposal

[attention-suite.json](attention-suite.json) enumerates forward fused
scaled dot-product attention on RTX 4070 (`sm_89`). Inputs and output are
FP16 in contiguous BHSD layout. The kernel uses FP32 softmax reductions and
accumulators. The suite covers prefill forward attention, with equal query
and KV sequence lengths. Decode with one query token and a long KV cache
uses flash-decoding kernels and will have a separate suite.

GQA maps query head `h` to KV head `h // (query_heads / kv_heads)`
without expanding K or V. Causal cases include the diagonal. The runner
checks the FP16 output and FP32 base-2 log-sum-exp against a chunked FP32
reference before collecting labels.

## Candidate space and splits

The raw 36-point space is the CUDA forward search space in the
[Triton 3.9.0 fused-attention tutorial](https://github.com/triton-lang/triton/blob/v3.9.0/python/tutorials/06-fused-attention.py)
with block M in {64, 128}, block N in {32, 64, 128}, stages in {2, 3, 4},
and warps in {4, 8}. Eligibility is explicit for each case. Keep block M
at most the sequence length, block N at most the head dimension, and
block N at most block M in causal cases. The tutorial's SM 90 eight-warp
filter does not apply to the RTX 4070. Warp specialization is disabled.
The runner enumerates the selected configurations directly and records
hardware resource failures.

The development and workload-holdout splits each have eight cases and
228 eligible case/config combinations. Each split has two cases per
causality and grouping combination, covering both 64- and 128-element
heads. Development includes a 1,021-token noncausal GQA case and a
1,025-token causal MHA case so both modes exercise tail masking. The
holdout uses different odd lengths and includes longer aligned workloads
at 2,560, 3,072, and 4,096 tokens. Splits are by full workload shape
across all configs and both cache states. Only development timings may be
opened while developing the attention predictor or an attention baseline.

## Label protocol

Before collecting any labels, review the matrix and freeze the kernel
source, shape specialization, scale, compiler options, and runner version.
Use the same kernel and options for offline AOT and hardware JIT. Verify
hash protocol v2 PTX and TTGIR, target, compiler and assembler versions,
launch grid, registers, and shared bytes. Record compile, launch,
correctness, and artifact-verification failures separately.

Use the RTX 4070 lock and record-only clock telemetry. Collect warm and
clean-cold CUDA Graph single-launch timings in ten randomized blocks per
session, with three sessions and order seeds 101, 102, and 103. Use the
existing read-only L2 eviction buffer (at least 128 MiB and three times L2
capacity) and cache-control probe. Keep holdout reports and their receipts
separate from development. Authorize release only after the formal
predictors are frozen and the holdout predictions and label receipts are
archived with digests.

## Runner

Run from the repository root with the validation environment. Offline AOT
does not require a GPU. Use a complete per-case AOT archive for collection.

```sh
python -m validation.attention_runner aot validation/attention-suite.json \
  b1-q16-kv16-s512-d64-full results/attention/aot/b1-q16-kv16-s512-d64-full.json
```

After the suite is approved for measurement, collect a development session
with the existing GPU lock. Use a unique output path for each session.

```sh
flock /tmp/virtualgpu-gpu.lock python -m validation.attention_runner capture \
  validation/attention-suite.json b1-q16-kv16-s512-d64-full \
  results/attention/development/b1-q16-kv16-s512-d64-full-session-1.json \
  --aot results/attention/aot/b1-q16-kv16-s512-d64-full.json --session 1
```

Holdout capture requires `--sealed`. It writes a mode-zero report and a
separate digest receipt. AOT and JIT share the same kernel entry point.
The AOT archive includes the manifest hash, kernel key, hashes and
resource metadata for each configuration. Only candidates passing the
reference and artifact checks contribute timing samples.
