# Phase 2 RTX 4070 to DGX Spark transfer

Formal evaluation of the `phase1-final` model on the sealed GB10 profile `0.11.0-m3`.
Only the primary cohort determines the Phase 2 decision. The NeuSight-style fit uses the
frozen RTX 4070 development labels.

## Primary transfer on development workloads

| Cache | Method | Median Kendall tau-b | Mean top-5 regret | Coverage | Defined rank cases |
| --- | --- | ---: | ---: | ---: | ---: |
| Warm | VirtualGPU | 0.446 | 1.10% | 255/255 | 8/9 |
| Warm | NeuSight-style | 0.311 | 1.33% | 255/255 | 8/9 |
| Warm | Roofline + wave | 0.213 | 1.95% | 255/255 | 7/9 |
| Clean-cold | VirtualGPU | 0.597 | 0.16% | 255/255 | 8/9 |
| Clean-cold | NeuSight-style | 0.501 | 1.59% | 255/255 | 8/9 |
| Clean-cold | Roofline + wave | 0.247 | 1.02% | 255/255 | 7/9 |

Overall primary decision: **Fail**. Warm exceeded the 1% absolute regret
ceiling by 0.095 percentage points. Both ranking margins and both
non-inferiority checks passed on warm. Clean-cold passed all checks.

| Cache | Rank vs NeuSight | Rank vs Roofline | Absolute regret | Non-inferiority | Decision |
| --- | --- | --- | --- | --- | --- |
| Warm | Pass | Pass | Fail | Pass | Fail |
| Clean-cold | Pass | Pass | Pass | Pass | Pass |

| Case | Cache | Kendall tau-b (core / NeuSight / Roofline) | Top-5 regret (same order) | Equivalent | Unresolved | Measured |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| decode-1 | Warm | 0.573 / 0.446 / 0.151 | 1.04% / 1.04% / 1.49% | 188 | 98 | 36 |
| decode-1 | Clean-cold | 0.565 / 0.576 / 0.150 | 0.00% / 0.00% / 0.36% | 217 | 80 | 36 |
| decode-16 | Warm | 0.232 / 0.232 / -0.058 | 0.34% / 1.11% / 7.54% | 352 | 73 | 36 |
| decode-16 | Clean-cold | 0.235 / 0.251 / -0.041 | 0.49% / 0.95% / 8.84% | 372 | 80 | 36 |
| decode-32 | Warm | 0.501 / 0.390 / 0.156 | 1.34% / 1.34% / 0.00% | 179 | 80 | 36 |
| decode-32 | Clean-cold | 0.492 / 0.343 / 0.133 | 0.91% / 0.94% / 0.00% | 176 | 101 | 36 |
| mask-m | Warm | 0.690 / 0.672 / 0.468 | 0.00% / 0.00% / 0.00% | 85 | 37 | 36 |
| mask-m | Clean-cold | 0.698 / 0.114 / 0.466 | 0.00% / 5.43% / 0.00% | 79 | 32 | 36 |
| swizzle-320 | Warm | 1.000 / 1.000 / n/a | 0.00% / 0.00% / 0.00% | 0 | 0 | 2 |
| swizzle-320 | Clean-cold | 1.000 / 1.000 / n/a | 0.00% / 0.00% / 0.00% | 0 | 0 | 2 |
| m1-cube | Warm | n/a / n/a / n/a | 0.00% / 0.00% / 0.00% | 0 | 0 | 1 |
| m1-cube | Clean-cold | n/a / n/a / n/a | 0.00% / 0.00% / 0.00% | 0 | 0 | 1 |
| sm-round-below | Warm | 0.391 / 0.125 / 0.242 | 3.89% / 3.89% / 3.89% | 51 | 34 | 36 |
| sm-round-below | Clean-cold | 0.632 / 0.491 / 0.254 | 0.00% / 3.73% / 0.00% | 110 | 44 | 36 |
| sm-round-exact | Warm | 0.329 / 0.124 / 0.231 | 0.64% / 0.64% / 0.64% | 75 | 34 | 36 |
| sm-round-exact | Clean-cold | 0.609 / 0.511 / 0.261 | 0.00% / 1.88% / 0.00% | 111 | 51 | 36 |
| sm-round-above | Warm | 0.355 / 0.135 / 0.213 | 2.60% / 3.95% / 3.95% | 70 | 38 | 36 |
| sm-round-above | Clean-cold | 0.585 / 0.547 / 0.247 | 0.00% / 1.42% / 0.00% | 120 | 39 | 36 |

## Secondary transfer on unseen workloads

| Cache | Method | Median Kendall tau-b | Mean top-5 regret | Coverage | Defined rank cases |
| --- | --- | ---: | ---: | ---: | ---: |
| Warm | VirtualGPU | 0.762 | 0.14% | 324/324 | 9/9 |
| Warm | NeuSight-style | 0.432 | 36.84% | 324/324 | 9/9 |
| Warm | Roofline + wave | 0.523 | 4.12% | 324/324 | 9/9 |
| Clean-cold | VirtualGPU | 0.749 | 0.13% | 324/324 | 9/9 |
| Clean-cold | NeuSight-style | -0.194 | 45.20% | 324/324 | 9/9 |
| Clean-cold | Roofline + wave | 0.519 | 3.89% | 324/324 | 9/9 |

| Case | Cache | Kendall tau-b (core / NeuSight / Roofline) | Top-5 regret (same order) | Equivalent | Unresolved | Measured |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| prefill-256 | Warm | 0.765 / 0.635 / 0.620 | 0.00% / 0.84% / 9.67% | 41 | 35 | 36 |
| prefill-256 | Clean-cold | 0.775 / 0.204 / 0.620 | 0.00% / 35.44% / 12.65% | 30 | 28 | 36 |
| prefill-1024 | Warm | 0.799 / 0.606 / 0.647 | 0.00% / 8.02% / 0.00% | 12 | 42 | 36 |
| prefill-1024 | Clean-cold | 0.777 / 0.601 / 0.634 | 0.00% / 7.16% / 0.00% | 16 | 45 | 36 |
| prefill-2048 | Warm | 0.683 / 0.565 / 0.624 | 0.00% / 4.86% / 0.00% | 25 | 40 | 36 |
| prefill-2048 | Clean-cold | 0.689 / 0.619 / 0.661 | 0.00% / 2.24% / 0.00% | 25 | 35 | 36 |
| mask-mnk | Warm | 0.794 / 0.232 / 0.193 | 0.00% / 0.20% / 17.25% | 66 | 13 | 36 |
| mask-mnk | Clean-cold | 0.781 / 0.200 / 0.180 | 0.00% / 2.42% / 18.23% | 55 | 27 | 36 |
| resident-wave-below | Warm | 0.762 / -0.330 / 0.523 | 0.00% / 104.77% / 2.52% | 141 | 28 | 36 |
| resident-wave-below | Clean-cold | 0.746 / -0.308 / 0.503 | 0.00% / 104.30% / 1.50% | 118 | 39 | 36 |
| resident-wave-exact | Warm | 0.730 / -0.352 / 0.480 | 0.00% / 103.96% / 1.29% | 123 | 42 | 36 |
| resident-wave-exact | Clean-cold | 0.737 / -0.346 / 0.493 | 0.00% / 108.72% / 1.51% | 126 | 29 | 36 |
| resident-wave-above | Warm | 0.762 / -0.305 / 0.507 | 0.00% / 103.79% / 1.13% | 123 | 52 | 36 |
| resident-wave-above | Clean-cold | 0.749 / -0.307 / 0.497 | 0.00% / 106.03% / 1.13% | 119 | 52 | 36 |
| l2-working-set-below | Warm | 0.512 / 0.505 / 0.398 | 0.00% / 3.88% / 5.25% | 77 | 87 | 36 |
| l2-working-set-below | Clean-cold | 0.657 / -0.194 / 0.519 | 0.09% / 18.75% / 0.00% | 118 | 77 | 36 |
| l2-working-set-above | Warm | 0.732 / 0.432 / 0.527 | 1.22% / 1.22% / 0.00% | 112 | 58 | 36 |
| l2-working-set-above | Clean-cold | 0.757 / -0.308 / 0.583 | 1.07% / 21.73% / 0.00% | 98 | 64 | 36 |

## Descriptive pairwise precision

Agreement is the share of all predicted faster-than calls that the hardware
orders in the same direction. Hardware-equivalent and unresolved pairs stay
in the denominator. This metric does not enter the formal decision.

### Primary transfer

| Cache | Method | Ordered calls | Agree | Reversed | Hardware equivalent | Hardware unresolved | Hardware-ordered pairs ordered correctly |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| warm | roofline_wave | 3853 | 46.2% | 23.4% | 21.4% | 9.0% | 59.1% |
| warm | neusight_style | 3854 | 51.1% | 20.1% | 20.2% | 8.6% | 65.3% |
| warm | core | 1236 | 95.1% | 0.2% | 3.2% | 1.5% | 39.0% |
| cold | roofline_wave | 3853 | 43.2% | 21.2% | 25.8% | 9.8% | 59.4% |
| cold | neusight_style | 3777 | 50.9% | 15.1% | 24.3% | 9.7% | 68.6% |
| cold | core | 1592 | 95.4% | 0.1% | 2.9% | 1.6% | 54.3% |

### Secondary transfer

| Cache | Method | Ordered calls | Agree | Reversed | Hardware equivalent | Hardware unresolved | Hardware-ordered pairs ordered correctly |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| warm | roofline_wave | 4899 | 69.6% | 13.7% | 10.2% | 6.5% | 74.9% |
| warm | neusight_style | 5009 | 53.1% | 28.6% | 11.2% | 7.0% | 58.5% |
| warm | core | 3025 | 96.9% | 0.8% | 0.9% | 1.5% | 64.4% |
| cold | roofline_wave | 4899 | 70.4% | 13.2% | 10.1% | 6.3% | 75.5% |
| cold | neusight_style | 4943 | 41.3% | 40.6% | 11.2% | 6.8% | 44.7% |
| cold | core | 3098 | 97.6% | 0.2% | 0.9% | 1.2% | 66.2% |

## GB10-only TMA diagnostic

The three stage configurations passed AOT/JIT validation in all three sessions.
Warm hardware medians for stages 2 and 3 differed by 0.32%. Clean-cold hardware
medians for stages 3 and 4 were equal, about 8.5% lower than stage 2.

Candidate-session statuses: `{"ok": 9}`.

| Requested stages | Buffers | Prefetch distance | MMA | Cache | Median measured ns | Nominal predicted ns |
| ---: | ---: | ---: | --- | --- | ---: | ---: |
| 2 | 1 | 0 | mma.sync m16n8k16 | Warm | 20288 | 20090 |
| 2 | 1 | 0 | mma.sync m16n8k16 | Clean-cold | 24416 | 24330 |
| 3 | 2 | 1 | mma.sync m16n8k16 | Warm | 20352 | 19620 |
| 3 | 2 | 1 | mma.sync m16n8k16 | Clean-cold | 22336 | 20123 |
| 4 | 3 | 2 | mma.sync m16n8k16 | Warm | 22256 | 20082 |
| 4 | 3 | 2 | mma.sync m16n8k16 | Clean-cold | 22336 | 20309 |

## Release audit

The prediction freeze `eae13bd` and [seal receipt](phase2-gb10-seal.json)
`076b1da` were verified on `origin/main` before release. All 57 report
digests matched before scoring. The original 10 sessions used `d42128e`
and the remaining 47 used `fd5c947`. No timed blocks were rejected. The
[acquisition audit](phase2-acquisition-audit.md) records the idle-check
interruption. The snapshots on either side of the preceding session did
not show the compute PID observed at the interruption.

| Event | UTC |
| --- | --- |
| Predictions archived | 2026-11-25T10:29:57.674713+00:00 |
| Release started | 2026-11-25T13:02:20.744904+00:00 |
| All report digests verified | 2026-11-25T13:02:20.807458+00:00 |
| Scored | 2026-11-25T13:03:28.564642+00:00 |

| Artifact | SHA-256 |
| --- | --- |
| Frozen GB10 profile | `6494b25720a6ff3be84e1f8ea8a9b34df3990b3adf22240e864e0f71f10d36f4` |
| GB10 AOT index | `4cd542f206529442650638fe7fb148ca0ddf642d573998284e0c00329203f3d0` |
| Prediction archive | `8d1744a806d7b45a1e2e4b86dd74a43dce3c9eca599ca3ee43a808ffc172b635` |
| Report seal index | `d9a883e0335932673ae93692d65c19eb4ea87f67de9eac5a66e80d08be09651c` |
| Full scoring archive | `8e1b9c3aca5d4759957257632b126529102075b36b097d7ef6c37e6811609cb8` |
| Compact scores | `4caf9c0fab330b6d3a79357573da281408426ee199eee8726b0dafc016ef8a0d` |

The [score summary](phase2-gb10-scores.json) includes per-case metrics,
shortlists, coverage, and confusion matrices. The complete scoring archive
retains individual candidate and pair records.
