# VirtualGPU v0.2 formal holdout evaluation

Frozen core `e70824b9c3ea5d6818a5bef306ee21c9e5d2d74a`. Release amendment `6d007ac5074c79e7bdf8b6ee1e988867d38943c8`.
All report digests verified at 2026-11-26T03:12:17.861180+00:00.

| Device | Decision |
| --- | --- |
| rtx4070 | Fail |
| gb10 | Fail |

RTX 4070 GEMM passed in both cache regimes. RTX 4070 attention exceeded
the 1% absolute regret ceiling on warm and clean-cold, at 1.303% and
1.433%, respectively.

GB10 GEMM exceeded the same ceiling on warm and clean-cold, at 2.119%
and 1.998%. Clean-cold also exceeded the NeuSight-style regret by
0.598 percentage points against the 0.5-point non-inferiority margin.
GB10 attention passed on clean-cold. On warm, its median Kendall tau-b
gain over NeuSight-style was 0.04836 against the required 0.05.

## rtx4070-gemm

Cohort decision: **Pass**.

| Cache | Method | Median tau-b | Defined ranks | Mean top-5 regret | Top-5 hits | Coverage | Direction agreement | Ordered coverage | MAPE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| warm | core | 0.748 | 9/9 | 0.82% | 5/9 | 324/324 | 94.47% | 53.87% | 16.21% |
| warm | roofline_wave | 0.334 | 9/9 | 14.36% | 0/9 | 324/324 | 54.93% | 60.38% | 123.65% |
| warm | neusight_style | 0.599 | 9/9 | 1.00% | 6/9 | 324/324 | 72.85% | 78.55% | 31.60% |
| cold | core | 0.693 | 9/9 | 0.73% | 6/9 | 324/324 | 92.70% | 50.35% | 14.88% |
| cold | roofline_wave | 0.390 | 9/9 | 13.28% | 0/9 | 324/324 | 54.16% | 61.49% | 119.74% |
| cold | neusight_style | 0.397 | 9/9 | 0.85% | 6/9 | 324/324 | 66.20% | 65.22% | 23.19% |

| Cache | Rank vs Roofline | Rank vs NeuSight | Absolute regret | Non-inferiority | Decision |
| --- | --- | --- | --- | --- | --- |
| warm | Pass | Pass | Pass | Pass | Pass |
| cold | Pass | Pass | Pass | Pass | Pass |

| Case | Capture status counts across sessions | Unmeasured configs |
| --- | --- | --- |
| v02-rtx4070-decode-2 | ok=108 | None |
| v02-rtx4070-decode-24 | ok=108 | None |
| v02-rtx4070-mask-mnk-385 | ok=108 | None |
| v02-rtx4070-mask-mnk-769 | ok=108 | None |
| v02-rtx4070-resident-wave-below | ok=108 | None |
| v02-rtx4070-resident-wave-exact | ok=108 | None |
| v02-rtx4070-resident-wave-above | ok=108 | None |
| v02-rtx4070-prefill-768 | ok=108 | None |
| v02-rtx4070-prefill-1536 | ok=108 | None |

## rtx4070-attention

Cohort decision: **Fail**.

| Cache | Method | Median tau-b | Defined ranks | Mean top-5 regret | Top-5 hits | Coverage | Direction agreement | Ordered coverage | MAPE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| warm | core | 0.710 | 8/8 | 1.30% | 4/8 | 188/188 | 90.23% | 69.42% | 7.96% |
| warm | roofline_wave | 0.175 | 8/8 | 10.60% | 0/8 | 188/188 | 60.66% | 44.74% | 25.26% |
| warm | neusight_style | 0.544 | 8/8 | 2.91% | 4/8 | 188/188 | 71.54% | 75.55% | 13.50% |
| cold | core | 0.711 | 8/8 | 1.43% | 4/8 | 188/188 | 89.44% | 70.87% | 8.68% |
| cold | roofline_wave | 0.175 | 8/8 | 10.34% | 0/8 | 188/188 | 60.34% | 44.77% | 25.25% |
| cold | neusight_style | 0.539 | 8/8 | 2.71% | 5/8 | 188/188 | 71.65% | 76.10% | 13.49% |

| Cache | Rank vs Roofline | Rank vs NeuSight | Absolute regret | Non-inferiority | Decision |
| --- | --- | --- | --- | --- | --- |
| warm | Pass | Pass | Fail | Pass | Fail |
| cold | Pass | Pass | Fail | Pass | Fail |

| Case | Capture status counts across sessions | Unmeasured configs |
| --- | --- | --- |
| b1-q12-kv12-s641-d64-full | ok=72 | None |
| b3-q16-kv16-s2560-d128-full | launch_failed=36, ok=72 | m128n128-w4-s3, m128n128-w4-s4, m128n128-w8-s3, m128n128-w8-s4, m128n64-w4-s4, m128n64-w8-s4, m64n128-w4-s3, m64n128-w4-s4, m64n128-w8-s3, m64n128-w8-s4, m64n64-w4-s4, m64n64-w8-s4 |
| b2-q16-kv16-s1153-d64-causal | ok=72 | None |
| b1-q24-kv24-s4096-d128-causal | launch_failed=24, ok=66 | m128n128-w4-s3, m128n128-w4-s4, m128n128-w8-s3, m128n128-w8-s4, m128n64-w4-s4, m128n64-w8-s4, m64n64-w4-s4, m64n64-w8-s4 |
| b2-q24-kv6-s1283-d64-full | ok=72 | None |
| b1-q32-kv4-s3072-d128-full | launch_failed=36, ok=72 | m128n128-w4-s3, m128n128-w4-s4, m128n128-w8-s3, m128n128-w8-s4, m128n64-w4-s4, m128n64-w8-s4, m64n128-w4-s3, m64n128-w4-s4, m64n128-w8-s3, m64n128-w8-s4, m64n64-w4-s4, m64n64-w8-s4 |
| b2-q32-kv8-s1667-d64-causal | ok=72 | None |
| b1-q40-kv5-s2177-d128-causal | launch_failed=24, ok=66 | m128n128-w4-s3, m128n128-w4-s4, m128n128-w8-s3, m128n128-w8-s4, m128n64-w4-s4, m128n64-w8-s4, m64n64-w4-s4, m64n64-w8-s4 |

## gb10-gemm

Cohort decision: **Fail**.

| Cache | Method | Median tau-b | Defined ranks | Mean top-5 regret | Top-5 hits | Coverage | Direction agreement | Ordered coverage | MAPE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| warm | core | 0.683 | 9/9 | 2.12% | 3/9 | 324/324 | 93.68% | 68.91% | 26.43% |
| warm | roofline_wave | 0.508 | 9/9 | 4.90% | 2/9 | 324/324 | 62.69% | 69.95% | 127.75% |
| warm | neusight_style | 0.222 | 9/9 | 1.93% | 1/9 | 324/324 | 58.56% | 54.68% | 37.39% |
| cold | core | 0.689 | 9/9 | 2.00% | 3/9 | 324/324 | 93.65% | 68.22% | 26.34% |
| cold | roofline_wave | 0.508 | 9/9 | 4.62% | 2/9 | 324/324 | 63.54% | 70.19% | 128.60% |
| cold | neusight_style | 0.221 | 9/9 | 1.40% | 1/9 | 324/324 | 59.00% | 54.55% | 53.53% |

| Cache | Rank vs Roofline | Rank vs NeuSight | Absolute regret | Non-inferiority | Decision |
| --- | --- | --- | --- | --- | --- |
| warm | Pass | Pass | Fail | Pass | Fail |
| cold | Pass | Pass | Fail | Fail | Fail |

| Case | Capture status counts across sessions | Unmeasured configs |
| --- | --- | --- |
| v02-gb10-decode-4 | ok=108 | None |
| v02-gb10-decode-40 | ok=108 | None |
| v02-gb10-mask-mnk-641 | ok=108 | None |
| v02-gb10-mask-mnk-1153 | ok=108 | None |
| v02-gb10-resident-wave-below | ok=108 | None |
| v02-gb10-resident-wave-exact | ok=108 | None |
| v02-gb10-resident-wave-above | ok=108 | None |
| v02-gb10-prefill-896 | ok=108 | None |
| v02-gb10-prefill-1792 | ok=108 | None |

## gb10-attention

Cohort decision: **Fail**.

| Cache | Method | Median tau-b | Defined ranks | Mean top-5 regret | Top-5 hits | Coverage | Direction agreement | Ordered coverage | MAPE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| warm | core | 0.670 | 8/8 | 0.82% | 5/8 | 188/188 | 90.55% | 70.51% | 10.32% |
| warm | roofline_wave | 0.274 | 8/8 | 9.84% | 0/8 | 188/188 | 63.38% | 44.37% | 166.99% |
| warm | neusight_style | 0.622 | 8/8 | 2.82% | 2/8 | 188/188 | 75.42% | 78.32% | 25.93% |
| cold | core | 0.703 | 8/8 | 0.33% | 6/8 | 188/188 | 86.79% | 73.40% | 9.76% |
| cold | roofline_wave | 0.278 | 8/8 | 10.47% | 0/8 | 188/188 | 63.70% | 44.22% | 163.68% |
| cold | neusight_style | 0.636 | 8/8 | 2.53% | 2/8 | 188/188 | 76.13% | 78.23% | 24.33% |

| Cache | Rank vs Roofline | Rank vs NeuSight | Absolute regret | Non-inferiority | Decision |
| --- | --- | --- | --- | --- | --- |
| warm | Pass | Fail | Pass | Pass | Fail |
| cold | Pass | Pass | Pass | Pass | Pass |

| Case | Capture status counts across sessions | Unmeasured configs |
| --- | --- | --- |
| b1-q20-kv20-s897-d64-full | ok=72 | None |
| b2-q12-kv12-s2304-d128-full | launch_failed=36, ok=72 | m128n128-w4-s3, m128n128-w4-s4, m128n128-w8-s3, m128n128-w8-s4, m128n64-w4-s4, m128n64-w8-s4, m64n128-w4-s3, m64n128-w4-s4, m64n128-w8-s3, m64n128-w8-s4, m64n64-w4-s4, m64n64-w8-s4 |
| b2-q20-kv20-s1345-d64-causal | ok=72 | None |
| b1-q28-kv28-s4608-d128-causal | launch_failed=24, ok=66 | m128n128-w4-s3, m128n128-w4-s4, m128n128-w8-s3, m128n128-w8-s4, m128n64-w4-s4, m128n64-w8-s4, m64n64-w4-s4, m64n64-w8-s4 |
| b2-q28-kv7-s1409-d64-full | ok=72 | None |
| b1-q36-kv6-s2816-d128-full | launch_failed=36, ok=72 | m128n128-w4-s3, m128n128-w4-s4, m128n128-w8-s3, m128n128-w8-s4, m128n64-w4-s4, m128n64-w8-s4, m64n128-w4-s3, m64n128-w4-s4, m64n128-w8-s3, m64n128-w8-s4, m64n64-w4-s4, m64n64-w8-s4 |
| b1-q48-kv8-s1857-d64-causal | ok=72 | None |
| b2-q32-kv8-s2433-d128-causal | launch_failed=24, ok=66 | m128n128-w4-s3, m128n128-w4-s4, m128n128-w8-s3, m128n128-w8-s4, m128n64-w4-s4, m128n64-w8-s4, m64n64-w4-s4, m64n64-w8-s4 |

[Full scores and candidate status audit](v02-holdout-scores.json).
The score file includes confusion matrices, undefined ranks, missing or invalid
predictions, and each configuration's status in every session.

## Per-case results

Triples show VirtualGPU, NeuSight-style, and roofline + wave in that order.
Equivalent and unresolved are hardware pair verdicts within the scoring protocol.

### rtx4070-gemm

| Case | Cache | Kendall tau-b | Top-5 regret | Equivalent | Unresolved | Measured |
| --- | --- | --- | --- | ---: | ---: | ---: |
| v02-rtx4070-decode-2 | warm | 0.891 / 0.632 / -0.314 | 0.00% / 0.00% / 20.00% | 115 | 79 | 36 |
| v02-rtx4070-decode-2 | cold | 0.858 / 0.232 / -0.298 | 0.00% / 0.00% / 9.66% | 190 | 103 | 36 |
| v02-rtx4070-decode-24 | warm | 0.600 / 0.323 / -0.182 | 0.94% / 2.07% / 16.38% | 211 | 36 | 36 |
| v02-rtx4070-decode-24 | cold | 0.631 / 0.124 / -0.137 | 0.76% / 1.89% / 12.31% | 216 | 27 | 36 |
| v02-rtx4070-mask-mnk-385 | warm | 0.873 / 0.468 / 0.201 | 0.00% / 0.00% / 51.13% | 85 | 22 | 36 |
| v02-rtx4070-mask-mnk-385 | cold | 0.809 / 0.383 / 0.167 | 0.16% / 0.16% / 46.95% | 100 | 17 | 36 |
| v02-rtx4070-mask-mnk-769 | warm | 0.657 / 0.406 / 0.223 | 0.00% / 0.00% / 10.22% | 44 | 58 | 36 |
| v02-rtx4070-mask-mnk-769 | cold | 0.639 / 0.397 / 0.186 | 0.00% / 0.00% / 8.95% | 39 | 65 | 36 |
| v02-rtx4070-resident-wave-below | warm | 0.787 / 0.598 / 0.377 | 0.38% / 0.00% / 7.52% | 128 | 43 | 36 |
| v02-rtx4070-resident-wave-below | cold | 0.787 / 0.404 / 0.398 | 0.00% / 0.00% / 9.30% | 118 | 79 | 36 |
| v02-rtx4070-resident-wave-exact | warm | 0.798 / 0.609 / 0.347 | 0.75% / 0.00% / 8.21% | 123 | 30 | 36 |
| v02-rtx4070-resident-wave-exact | cold | 0.817 / 0.411 / 0.390 | 0.00% / 0.00% / 8.46% | 118 | 31 | 36 |
| v02-rtx4070-resident-wave-above | warm | 0.748 / 0.599 / 0.334 | 0.00% / 1.82% / 4.00% | 46 | 38 | 36 |
| v02-rtx4070-resident-wave-above | cold | 0.540 / 0.314 / 0.402 | 0.00% / 0.00% / 10.37% | 41 | 46 | 36 |
| v02-rtx4070-prefill-768 | warm | 0.677 / 0.850 / 0.583 | 0.00% / 0.00% / 8.17% | 94 | 19 | 36 |
| v02-rtx4070-prefill-768 | cold | 0.693 / 0.844 / 0.574 | 0.00% / 0.00% / 8.92% | 86 | 31 | 36 |
| v02-rtx4070-prefill-1536 | warm | 0.616 / 0.600 / 0.709 | 5.29% / 5.13% / 3.58% | 66 | 56 | 36 |
| v02-rtx4070-prefill-1536 | cold | 0.562 / 0.571 / 0.689 | 5.66% / 5.56% / 4.58% | 42 | 83 | 36 |

### rtx4070-attention

| Case | Cache | Kendall tau-b | Top-5 regret | Equivalent | Unresolved | Measured |
| --- | --- | --- | --- | ---: | ---: | ---: |
| b1-q12-kv12-s641-d64-full | warm | 0.721 / 0.550 / -0.023 | 0.00% / 0.00% / 21.87% | 38 | 4 | 24 |
| b1-q12-kv12-s641-d64-full | cold | 0.739 / 0.545 / -0.037 | 0.00% / 0.00% / 20.59% | 40 | 10 | 24 |
| b3-q16-kv16-s2560-d128-full | warm | 0.717 / 0.623 / 0.566 | 3.56% / 0.00% / 6.80% | 28 | 14 | 24 |
| b3-q16-kv16-s2560-d128-full | cold | 0.715 / 0.606 / 0.557 | 3.74% / 0.00% / 6.60% | 27 | 13 | 24 |
| b2-q16-kv16-s1153-d64-causal | warm | 0.745 / 0.511 / -0.118 | 0.00% / 1.55% / 21.71% | 55 | 10 | 24 |
| b2-q16-kv16-s1153-d64-causal | cold | 0.740 / 0.528 / -0.118 | 0.00% / 0.00% / 21.21% | 45 | 18 | 24 |
| b1-q24-kv24-s4096-d128-causal | warm | 0.685 / 0.538 / 0.559 | 1.70% / 0.00% / 3.40% | 42 | 4 | 22 |
| b1-q24-kv24-s4096-d128-causal | cold | 0.680 / 0.532 / 0.558 | 2.38% / 0.00% / 4.06% | 35 | 13 | 22 |
| b2-q24-kv6-s1283-d64-full | warm | 0.478 / 0.457 / -0.045 | 0.42% / 9.60% / 8.21% | 53 | 2 | 24 |
| b2-q24-kv6-s1283-d64-full | cold | 0.464 / 0.457 / -0.045 | 0.28% / 9.67% / 8.29% | 53 | 2 | 24 |
| b1-q32-kv4-s3072-d128-full | warm | 0.759 / 0.621 / 0.567 | 4.75% / 0.00% / 6.17% | 30 | 14 | 24 |
| b1-q32-kv4-s3072-d128-full | cold | 0.722 / 0.613 / 0.562 | 5.07% / 0.00% / 6.12% | 19 | 26 | 24 |
| b2-q32-kv8-s1667-d64-causal | warm | 0.703 / 0.638 / -0.032 | 0.00% / 11.31% / 15.85% | 46 | 8 | 24 |
| b2-q32-kv8-s1667-d64-causal | cold | 0.708 / 0.635 / -0.023 | 0.00% / 11.67% / 15.53% | 43 | 12 | 24 |
| b1-q40-kv5-s2177-d128-causal | warm | 0.550 / 0.325 / 0.372 | 0.00% / 0.81% / 0.81% | 39 | 1 | 22 |
| b1-q40-kv5-s2177-d128-causal | cold | 0.541 / 0.316 / 0.372 | 0.00% / 0.34% / 0.34% | 38 | 4 | 22 |

### gb10-gemm

| Case | Cache | Kendall tau-b | Top-5 regret | Equivalent | Unresolved | Measured |
| --- | --- | --- | --- | ---: | ---: | ---: |
| v02-gb10-decode-4 | warm | 0.604 / 0.222 / -0.075 | 0.92% / 1.38% / 6.60% | 250 | 28 | 36 |
| v02-gb10-decode-4 | cold | 0.621 / 0.204 / -0.041 | 0.95% / 0.95% / 5.96% | 252 | 31 | 36 |
| v02-gb10-decode-40 | warm | 0.819 / 0.156 / 0.438 | 0.09% / 0.09% / 0.67% | 153 | 30 | 36 |
| v02-gb10-decode-40 | cold | 0.816 / 0.159 / 0.445 | 0.02% / 0.02% / 0.91% | 153 | 23 | 36 |
| v02-gb10-mask-mnk-641 | warm | 0.811 / 0.292 / 0.265 | 0.00% / 0.25% / 30.52% | 61 | 2 | 36 |
| v02-gb10-mask-mnk-641 | cold | 0.850 / 0.248 / 0.226 | 0.00% / 0.00% / 31.26% | 62 | 5 | 36 |
| v02-gb10-mask-mnk-1153 | warm | 0.588 / 0.454 / 0.508 | 3.57% / 0.00% / 0.60% | 66 | 6 | 36 |
| v02-gb10-mask-mnk-1153 | cold | 0.588 / 0.460 / 0.508 | 4.03% / 0.79% / 0.00% | 62 | 11 | 36 |
| v02-gb10-resident-wave-below | warm | 0.385 / 0.005 / 0.563 | 4.56% / 2.76% / 0.00% | 196 | 24 | 36 |
| v02-gb10-resident-wave-below | cold | 0.380 / 0.016 / 0.549 | 5.57% / 2.38% / 0.42% | 158 | 40 | 36 |
| v02-gb10-resident-wave-exact | warm | 0.309 / 0.000 / 0.607 | 4.67% / 2.73% / 0.08% | 201 | 31 | 36 |
| v02-gb10-resident-wave-exact | cold | 0.309 / 0.011 / 0.580 | 5.52% / 2.40% / 0.39% | 184 | 30 | 36 |
| v02-gb10-resident-wave-above | warm | 0.824 / 0.221 / 0.354 | 0.00% / 0.32% / 0.32% | 136 | 15 | 36 |
| v02-gb10-resident-wave-above | cold | 0.822 / 0.221 / 0.358 | 0.00% / 0.03% / 0.77% | 128 | 14 | 36 |
| v02-gb10-prefill-896 | warm | 0.698 / 0.689 / 0.544 | 5.27% / 5.27% / 5.27% | 48 | 24 | 36 |
| v02-gb10-prefill-896 | cold | 0.705 / 0.692 / 0.544 | 1.89% / 1.89% / 1.89% | 51 | 23 | 36 |
| v02-gb10-prefill-1792 | warm | 0.683 / 0.660 / 0.697 | 0.00% / 4.56% / 0.00% | 32 | 44 | 36 |
| v02-gb10-prefill-1792 | cold | 0.689 / 0.679 / 0.703 | 0.00% / 4.15% / 0.00% | 23 | 53 | 36 |

### gb10-attention

| Case | Cache | Kendall tau-b | Top-5 regret | Equivalent | Unresolved | Measured |
| --- | --- | --- | --- | ---: | ---: | ---: |
| b1-q20-kv20-s897-d64-full | warm | 0.358 / 0.650 / -0.045 | 2.87% / 8.64% / 11.51% | 40 | 7 | 24 |
| b1-q20-kv20-s897-d64-full | cold | 0.601 / 0.696 / 0.022 | 0.23% / 9.65% / 13.33% | 43 | 8 | 24 |
| b2-q12-kv12-s2304-d128-full | warm | 0.768 / 0.572 / 0.527 | 0.00% / 2.19% / 12.85% | 33 | 7 | 24 |
| b2-q12-kv12-s2304-d128-full | cold | 0.775 / 0.580 / 0.527 | 0.00% / 1.96% / 13.05% | 34 | 5 | 24 |
| b2-q20-kv20-s1345-d64-causal | warm | 0.717 / 0.703 / 0.031 | 0.00% / 3.81% / 10.91% | 36 | 19 | 24 |
| b2-q20-kv20-s1345-d64-causal | cold | 0.715 / 0.679 / 0.031 | 0.00% / 2.40% / 12.67% | 33 | 22 | 24 |
| b1-q28-kv28-s4608-d128-causal | warm | 0.775 / 0.532 / 0.477 | 0.00% / 2.45% / 4.64% | 25 | 8 | 22 |
| b1-q28-kv28-s4608-d128-causal | cold | 0.792 / 0.550 / 0.477 | 0.00% / 1.91% / 4.84% | 25 | 8 | 22 |
| b2-q28-kv7-s1409-d64-full | warm | 0.616 / 0.725 / 0.208 | 0.00% / 3.28% / 5.40% | 41 | 16 | 24 |
| b2-q28-kv7-s1409-d64-full | cold | 0.696 / 0.717 / 0.217 | 0.00% / 2.19% / 5.04% | 35 | 8 | 24 |
| b1-q36-kv6-s2816-d128-full | warm | 0.710 / 0.594 / 0.486 | 0.00% / 2.15% / 10.86% | 32 | 4 | 24 |
| b1-q36-kv6-s2816-d128-full | cold | 0.710 / 0.594 / 0.486 | 0.00% / 2.14% / 11.11% | 32 | 3 | 24 |
| b1-q48-kv8-s1857-d64-causal | warm | 0.630 / 0.696 / 0.092 | 1.03% / 0.00% / 17.69% | 42 | 10 | 24 |
| b1-q48-kv8-s1857-d64-causal | cold | 0.667 / 0.717 / 0.117 | 0.00% / 0.00% / 19.19% | 40 | 10 | 24 |
| b2-q32-kv8-s2433-d128-causal | warm | 0.610 / 0.359 / 0.339 | 2.67% / 0.00% / 4.84% | 28 | 3 | 22 |
| b2-q32-kv8-s2433-d128-causal | cold | 0.619 / 0.359 / 0.339 | 2.42% / 0.00% / 4.55% | 28 | 2 | 22 |

## Candidate and cache-control audit

Capture counts are configuration-session outcomes. Prediction counts are
per method and cache state, with one entry per manifest candidate.

| Cohort | Capture statuses | Configs with no eligible timings | Warm prediction statuses | Cold prediction statuses |
| --- | --- | ---: | --- | --- |
| rtx4070-gemm | ok=972 | 0 | core ok=324<br>neusight_style ok=324<br>roofline_wave ok=324 | core ok=324<br>neusight_style ok=324<br>roofline_wave ok=324 |
| rtx4070-attention | launch_failed=120, ok=564 | 40 | core ok=188, unsupported=40<br>neusight_style ok=188, unsupported=40<br>roofline_wave ok=188, unsupported=40 | core ok=188, unsupported=40<br>neusight_style ok=188, unsupported=40<br>roofline_wave ok=188, unsupported=40 |
| gb10-gemm | ok=972 | 0 | core ok=324<br>neusight_style ok=324<br>roofline_wave ok=324 | core ok=324<br>neusight_style ok=324<br>roofline_wave ok=324 |
| gb10-attention | launch_failed=120, ok=564 | 40 | core ok=188, unsupported=40<br>neusight_style ok=188, unsupported=40<br>roofline_wave ok=188, unsupported=40 | core ok=188, unsupported=40<br>neusight_style ok=188, unsupported=40<br>roofline_wave ok=188, unsupported=40 |

The score JSON lists unmeasured configurations and missing or invalid
predictions by case and configuration ID. The score uses the effective
RTX 4070 prefill-1536 session 2 retry and verifies its failed predecessor
against the published seal.

| Cohort | Sessions passing probe | Median warm session median (ns) | Median clean-cold session median (ns) | Clean-cold / warm ratio range |
| --- | ---: | ---: | ---: | --- |
| rtx4070-gemm | 27/27 | 15360.0 | 44032.0 | 2.0976–2.8667 |
| rtx4070-attention | 24/24 | 16384.0 | 44032.0 | 2.6061–2.8667 |
| gb10-gemm | 27/27 | 10048.0 | 50016.0 | 4.7262–5.4693 |
| gb10-attention | 24/24 | 10048.0 | 50080.0 | 4.1673–5.1863 |

Each accepted session had finite positive probe samples and a clean-cold
median greater than the warm median. The score JSON includes each probe's
samples and medians.

## Release evidence

- Formal label release began at `2026-11-26T03:12:17.779948+00:00`.
- All report hashes were verified at `2026-11-26T03:12:17.861180+00:00`.
- Scoring finished at `2026-11-26T03:14:24.444547+00:00`.
- Prediction archive SHA-256 `74caeeeb72380b0961b26db2f9ec63ebf1974b1153266b0c84bd2ff404f06d93`.
- Frozen fit SHA-256 `9bbd965c362aed8f55dbb01224e76aef5073074ef726753a4ad9b045e860bcd4`.
- Frozen AOT index SHA-256 `8d752498cb7a015428a960dd1a07a10e003cc84a3891bf135776287c5512e0fd`.
- Published score file SHA-256 `6557eebc3a01db01c846428b35b79f17400403fbc59e911d30f8a9b38b3bad48`.

The preregistered device decisions require both families to pass in both
cache regimes. Direction agreement and latency error are descriptive.
