# M2 development scoring audit

The RTX 4070 development manifest has SHA-256
`cbfe410e95f327943f6f0cd59250f217b874cbb4654a3a161f7e52e3756698d4`.
Its label audit records 27 sessions, 255 case/config entries, 765 matching
AOT/JIT pairs, and 15,300 warm and cold timings. The frozen workload holdout
and DGX Spark GEMM labels remained sealed throughout these decisions.

## Shortlist criteria revision

Commit `298ea00` at 04:57:36 +08:00 on 2026-11-25 changed the shortlist
criterion to mean top-5 regret at most 1% and at most 0.5 percentage points
above each baseline. The median Kendall tau-b improvement requirement
remained 0.05 against each baseline. The cross-GPU experiment uses the same
criteria and trains NeuSight-style only on RTX 4070 labels.

The development baseline report for profile `0.3.0-m2` was finalized at
04:56:57 +08:00 and showed 0.46% mean warm top-5 regret for the learned
baseline. The criterion changed after seeing that result and before the
validation owner saw any full M2 development-grid model scores. The first
such score artifact, `results/m2/scores.json`, was written at
05:01:03 +08:00. This ordering applies to the full M2 ranking, shortlist,
and pair scores.

Before the M2 nominal preflight, the validation owner had already received
the M1 closeout summary from the coordinator. It included model-versus-GPU
timing differences on two cases reused in M2, m1-cube and decode-16. It
reported m1-cube cold and warm timing errors of +4.7% and +9.1%, approximately
-22% systematic error for decode-16, and warm underprediction of up to 52%.
The validation owner had no full M2 development-grid ranking or shortlist
scores at the time of the criterion change. The local nominal preflight
printed model status and simulator execution duration for one configuration
per case.

## Pair resolution revision

Commit `c45b773` at 05:00:29 +08:00 on 2026-11-25 fixed the practical
resolution band at `log(1.05)`. An ordered call requires the same direction
and at least that margin on every shared draw, including draw 0. The
coordinator reports that the model-pair evidence inspected for this change
was the three decode-16 pairs from development. All three were equivalent
on hardware and ordered by the previous draw-only rule. The 5% margin
comes from the existing hardware equivalence band. The holdout remained
sealed.

## Model rerun

`results/m2/scores.json` and `results/m2/scores-5pct.json` used core
`5c4863e` before the same-timestamp CTA ordering fix. They are provisional
development scores. The formal rerun uses the fixed core with the same
frozen labels and predeclared scoring rules.
