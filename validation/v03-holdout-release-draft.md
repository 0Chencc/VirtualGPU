# Draft v0.3 holdout manifest release amendment

Status: draft for independent review. No capture, retry, tag, or label
access has been approved. This amendment covers the workload design and
capture protocol. The later prediction freeze has its own amendment.

## Inventory and approved candidate design

The Ascend evaluation stays outside the v0.3 freeze. Its opened shapes
count toward the historical novelty inventory. The inventory includes
the previously audited NVIDIA workloads, the opened Ascend v2 holdout,
the manifest-recorded v1 diagnostic shapes and the documented Ascend
development exclusions. Assert the same inventory through the date of
manifest release and review any additional opened datasets before
publishing.

Apply the [registered replacement rule](v03-holdout-selection.md) at
`fa21449`. The RTX 4070 GEMM wave triplet changes K from 2816 to 2752.
The GB10 GEMM wave triplet changes K from 3072 to 3008. The GB10 aligned
non-causal D64 MHA attention length changes from 896 to 384. In the last
search 768, 1024, 640, 1152, 512 and 1280 were reserved or historical.
These repairs preserve GEMM per-axis tile remainders and the wave
triplets, as well as the attention configuration IDs, strata and
alignment. The two GEMM cohorts share no axis value. Every
candidate GEMM axis and attention length clears the updated history.
All 16 attention lengths are distinct.

The manifests list the exact cases and configuration IDs. Keep the
reviewed GEMM space of 36 configs for each case. Attention retains the
per-case query tile, key tile and causal filters in the generator. The
AOT audit records the static status and reason for every excluded
configuration.

| Cohort | Cases | Configuration rows | AOT fit | Over resource limit | Reference wave CTAs |
| --- | ---: | ---: | ---: | ---: | --- |
| RTX 4070 GEMM | 9 | 324 | 324 | 0 | 91 / 92 / 93 |
| GB10 GEMM | 9 | 324 | 324 | 0 | 95 / 96 / 97 |
| RTX 4070 attention | 8 | 228 | 188 | 40 | n/a |
| GB10 attention | 8 | 228 | 188 | 40 | n/a |

The GEMM reference `m64n128k32-w4-s4-g1` uses 128 registers per
thread, 36,864 B static shared memory and two resident CTAs per SM on
both devices. CPU AOT compiled all 1,104 points without accessing
either GPU. Use Triton 3.9.0 with assembler optimization, ptxas 12.9
(V12.9.86, sm_89) and ptxas-blackwell 13.4 (V13.4.59, sm_121a PTX).
The per-config records bind hash protocol v2 PTX and TTGIR, resources,
launch grid and options. Before timing, verify the JIT and AOT identity,
the actual wave geometry and hardware, and reference correctness.
Preserve static exclusions and every attempted configuration's status
and reason. The smoke checks in this draft use CPU fixtures only.

| Release input | SHA-256 |
| --- | --- |
| [GEMM manifest](v03-gemm-holdout.json) | `fa03257073857afb940eb477171db6bb8b91a93af4385c77445d41e21982608b` |
| [Attention manifest](v03-attention-holdout.json) | `aea8e303eb9a01943b7e762fc853d174c87721f70d104c33d530d247b942bb5d` |
| [Historical inventory](v03-holdout-history.json) | `c7fdec43f70289ab7c17432c312d30ef8bbb54ce3d69f33c1234ecf5dcc83e1f` |
| [CPU AOT audit index](v03-holdout-aot-audit.json) | `31a9dcdb889f17d8545524125445748a2a399eebf1594c8cb4336c75dc244bb7` |
| [GB10 capture policy](../profiles/gb10.json), driver 580.173.02 | `10625fa4832c864fcba4d81c9b99480e58f69c81644a9950083b702488442471` |
| [RTX 4070 capture policy](../profiles/rtx4070.json), driver 610.88 | `3bb13b7307eabb803465a05cbcd939b78a6252bb2c54f3dc761c069ac2a8ed91` |
| CPU AOT archive `results/v03-holdout-release-aot.tar.gz` | `f73d6ec8b94897fd2c38f3e33342c7aad58621072ee45d3ee3ed7bbe14873bcf` |

## Publication and execution authorization

The release lead publishes the reviewed plan on origin/main after
independent review and informs the project owner. The reviewed code
commit must be reachable from the plan commit. If the code is squashed
on merge, bind the published commit containing the reviewed code. This draft
reviews code at `1e11b8d`. The
payload at `validation/v03-manifest-release.json` has
`status: released`, `timestamp_utc`, `code_commit`, an `inputs`
object with the SHA-256 for each of `gemm`, `attention`, `history`
and `aot`, and a `capture_profiles` object mapping `gb10` and
`rtx4070` to the policy profile SHA-256 values above.

Each capture requires a subsequent origin/main commit containing
`validation/v03-approvals/<device>.json` with
`status: capture_approved`, `timestamp_utc`, `device` and the
`release_sha256`. The release lead may approve both GB10 cohorts on
Spark 1 as soon as the reviewed plan is published and the machine is
available. For RTX 4070, the project owner first grants the local
desktop window and the approval records
`desktop_window_confirmed: true`. Both sets of attention lengths
are decided when the plan is published. The GB10 measurements can
proceed while the RTX 4070 cohorts wait for their window.

Use a clean checkout with the approved capture code and files and
verify origin/main against the remote. Capture policies are the
digest-bound record-only profiles used in the resource audit. Their
driver requirements are the exact `provenance.driver_version`
values in the table above. Stop on a mismatch and return to review
before any capture under a different policy. The capture-policy
digest is distinct from the prediction-profile digest fixed later.
Qualified 0.23.0 profiles are required for the prediction freeze.
The sealed kernel timings may be collected earlier.

## Capture, seal and retry

Run `validation.v03_capture` with the extracted AOT directory,
device, family, case ID and session. Hold
`/tmp/virtualgpu-gpu.lock` for every CUDA operation and Spark CPU
heavy task. Apply the CUDA/host process checks, telemetry,
the GB10 integrated host and thermal gates or the RTX 4070 Windows
quiet gate. Check the Windows process list for game, streaming and
compute activity before starting a local session. Capture each case
in three sessions with ten randomized complete blocks per session
and seeds 101, 102 and 103. Measure warm and clean-cold single CUDA
Graph launches. Use read-only L2 eviction and require a finite,
positive cold read probe whose median exceeds the warm median.
Ten complete blocks and all declared configuration records are
required. See the [registered protocol](v03-preregistration.md#hardware-and-baselines).

Create raw reports in a private directory at mode 000. Publish their
SHA-256, UTC, status, block count and configuration-status census.
Keep the host trace, invalid rows and protocol-failure receipts.
`validation.v03_audit` aggregates receipts and checks the cohort's
completeness using file metadata and the public approval chain.
Keep the sealed files unopened.

After both approvals for the device, a GB10 case-session and its
receipt audit use the following entry points. The case ID comes from
the published manifest.

```sh
flock /tmp/virtualgpu-gpu.lock .venv/bin/python -m validation.v03_capture \
  results/v03-holdout-sealed results/v03-holdout-release-pinned/aot \
  gb10 gemm v03-gb10-wave-exact 1
.venv/bin/python -m validation.v03_audit results/v03-holdout-sealed \
  results/v03-gb10-gemm-seal.json gb10 gemm
```

Retry only an immutable, documented protocol failure. A separate
origin/main approval at
`validation/v03-approvals/retries/<device>/<case>-session-<n>-attempt-<k>.json`
must carry `status: retry_approved`, `timestamp_utc`, `device`,
`case_id`, `session`, `attempt`, `capture_approval_sha256` and
`failed_receipt_sha256`. Start with attempt 1, keep every failed
attempt intact, and use the same seed and approved inputs. Only a
failed predecessor may be retried. An invalid configuration inside
an otherwise valid session is retained as recorded.

## Freeze, scoring and opening

The [scoring implementation](v03_score.py) uses the reviewed v0.2
numerical routines for the registered v0.3 criteria. It requires
both families and cache regimes on each device, includes every
hardware-eligible measured candidate in the oracle, and reports
missing and invalid candidates, read probes, ranking and regret,
paired judgments and latency errors. The four cohort decisions give
two independent device decisions. The original decision criteria in
the [preregistration](v03-preregistration.md#decisions) apply.

The project owner authorizes the final tag, formal release and label
opening. Publish a `v0.3-final` tag and then
`validation/v03-freeze-amendment.json` with `status: predictions_frozen`,
`timestamp_utc`, `revision`, `release_sha256`, `profiles`
(the final profile digests), `fits` (the two frozen family fits),
`predictions` (all four cohort records), and `seals` (all four
cohort receipts). Archive references use relative `path` and
`sha256`. Preserve both baseline fitting inputs and the formal
build metadata. Archive the three methods' predictions and status
for every configuration and cache regime. A subsequent
`validation/v03-label-release.json` on origin/main records
`status: opening_approved`, `timestamp_utc` and `freeze_sha256`.
On the clean tagged checkout, verify all frozen inputs and every
sealed report digest before parsing timing labels and score once.
The capture approval gives no access to the labels.
