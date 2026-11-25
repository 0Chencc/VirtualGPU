# Phase 1 RTX 4070 workload holdout

Formal frozen evaluation using phase1-final and profile 0.11.0-m3.
All three prediction sets were archived before the sealed workload timings were opened.
The NeuSight-style fit uses 4070 development labels only. GB10 labels remain sealed.

| Cache | Method | Median Kendall tau-b | Mean top-5 regret | Coverage |
| --- | --- | ---: | ---: | ---: |
| warm | VirtualGPU | 0.787 | 3.11% | 324/324 |
| warm | NeuSight-style | 0.387 | 1.19% | 324/324 |
| warm | Roofline + wave | 0.512 | 9.30% | 324/324 |
| cold | VirtualGPU | 0.750 | 3.02% | 324/324 |
| cold | NeuSight-style | 0.476 | 1.24% | 324/324 |
| cold | Roofline + wave | 0.514 | 9.25% | 324/324 |

## Formal thresholds

| Cache | Rank vs roofline | Rank vs NeuSight-style | Absolute regret | Non-inferiority | Overall |
| --- | --- | --- | --- | --- | --- |
| warm | Pass | Pass | Fail | Fail | Fail |
| cold | Pass | Pass | Fail | Fail | Fail |

Overall Phase 1 decision: **Fail**. The rank criterion passed in both cache
states. Mean top-5 regret exceeded the 1% absolute cap and the 0.5 percentage
point non-inferiority margin against NeuSight-style in both states. The
largest loss came from `mask-mnk`, with 24.47% warm and 22.19% clean-cold
top-5 regret.

Of 5,670 hardware pairs per cache state, 619 warm (10.92%) and 599 clean-cold
(10.56%) were equivalent within the 5% resolution band. The paired bootstrap
left 224 warm and 203 clean-cold pairs unresolved.

## Per-case metrics

Scores are shown in the order VirtualGPU, NeuSight-style, and roofline + wave.

| Case | Cache | Candidates | Kendall tau-b | Top-5 regret | Equivalent pairs | Unresolved pairs |
| --- | --- | ---: | --- | --- | ---: | ---: |
| prefill-256 | warm | 36 | 0.660 / 0.761 / 0.512 | 0.68% / 0.68% / 28.38% | 66 | 7 |
| prefill-256 | cold | 36 | 0.697 / 0.818 / 0.514 | 2.70% / 2.70% / 28.38% | 67 | 12 |
| prefill-1024 | warm | 36 | 0.728 / 0.387 / 0.731 | 0.00% / 7.99% / 0.00% | 63 | 57 |
| prefill-1024 | cold | 36 | 0.734 / 0.488 / 0.728 | 0.00% / 7.38% / 0.00% | 58 | 55 |
| prefill-2048 | warm | 36 | 0.689 / 0.460 / 0.701 | 2.82% / 0.00% / 6.77% | 59 | 40 |
| prefill-2048 | cold | 36 | 0.686 / 0.476 / 0.705 | 2.31% / 0.00% / 6.94% | 63 | 34 |
| mask-mnk | warm | 36 | 0.359 / 0.308 / 0.050 | 24.47% / 0.21% / 24.47% | 63 | 27 |
| mask-mnk | cold | 36 | 0.336 / 0.354 / -0.005 | 22.19% / 0.00% / 22.19% | 72 | 5 |
| resident-wave-below | warm | 36 | 0.798 / 0.193 / 0.495 | 0.00% / 0.39% / 0.39% | 73 | 27 |
| resident-wave-below | cold | 36 | 0.792 / 0.103 / 0.484 | 0.00% / 0.00% / 0.50% | 70 | 27 |
| resident-wave-exact | warm | 36 | 0.787 / 0.171 / 0.546 | 0.00% / 0.49% / 0.69% | 82 | 23 |
| resident-wave-exact | cold | 36 | 0.798 / 0.095 / 0.550 | 0.00% / 0.30% / 0.60% | 70 | 19 |
| resident-wave-above | warm | 36 | 0.802 / 0.173 / 0.529 | 0.00% / 0.97% / 0.58% | 74 | 19 |
| resident-wave-above | cold | 36 | 0.804 / 0.011 / 0.516 | 0.00% / 0.79% / 0.39% | 75 | 18 |
| l2-working-set-below | warm | 36 | 0.800 / 0.656 / 0.472 | 0.00% / 0.00% / 11.27% | 77 | 14 |
| l2-working-set-below | cold | 36 | 0.750 / 0.747 / 0.465 | 0.00% / 0.00% / 12.68% | 64 | 14 |
| l2-working-set-above | warm | 36 | 0.799 / 0.706 / 0.485 | 0.00% / 0.00% / 11.11% | 62 | 10 |
| l2-working-set-above | cold | 36 | 0.781 / 0.763 / 0.473 | 0.00% / 0.00% / 11.59% | 60 | 19 |

The [score summary](phase1-holdout-scores.json) includes per-case scores,
shortlists, coverage, and pair confusion matrices. The full local scoring
archive also contains per-config and per-pair details.

## Release evidence

The annotated `phase1-final` tag object `ca04403df570d46fd83bd0a3be900417ea7d66ee`
points to commit `129a6f55d3dcbf5c261a8397296400f91f291fe1`. The [freeze amendment](phase2-preregistration.md)
was published as `92439e0ab73e2c670109c56d510fa9b9f0615b6b` before opening.
Release began at 2026-11-25T09:42:31.523183+00:00. All 27
measurement report digests matched the [seal receipt](holdout-seal.json) at
2026-11-25T09:42:31.560984+00:00, before any timings were parsed.
All 324 configurations then passed AOT/JIT verification across the three
sessions.

| Artifact | Recorded (UTC) | SHA-256 |
| --- | --- | --- |
| Formal core binary | 2026-11-25T09:37:47.692479+00:00 | `f94d6732bf02b9ae69c8eb95ddb82c329f7403490b017aa69040fe399d975e0f` |
| RTX 4070 profile 0.11.0-m3 | 2026-11-25T09:37:47.692479+00:00 | `8ddab1c37b975aaf5390d160654ae34fd8ae04301ad64df1022c26896f3a0aa8` |
| Frozen NeuSight-style fit | 2026-11-25T09:38:13.038675+00:00 | `20f7b3869b8e7e79fba98a05d91d38c6c305a3a3bdfe475bd7838111a264ed72` |
| Fit training input index | 2026-11-25T09:38:13.038675+00:00 | `80df250aebfd782cbcfca9bbb539cdb9edbeaf5b0b0682274913ced53598b754` |
| Prediction archive | 2026-11-25T09:41:19.024605+00:00 | `e6869636ce5f838398bd8f859bd06a671ae468f147c99f9bfabb410a027c4852` |
| AOT match index | 2026-11-25T09:41:19.024605+00:00 | `3585245d009ec9afc6c2064c4d339bf744fb167b783bf8f1b50e3c070c8a177a` |
| Workload holdout manifest | 2026-11-25T09:41:19.024605+00:00 | `b6c8fd089543ec8444e820993acc962f25eb3e662f9d47d3fe3bb79dbf03eaf1` |
| Seal receipt | 2026-11-25T09:42:31.560984+00:00 | `14df3e947b38ea7f7cd9cda9be0ad7bcad33213b75506379ef9e47575f293ce9` |
| Pre-registration at publication | 2026-11-25T09:42:20.333395+00:00 | `0bef18245e9296ec8f4e5df7836e0475dc13d33f8d2fd7e43af173942337da30` |
| Full local scoring archive | 2026-11-25T09:43:09.389572+00:00 | `99dec3a67164c6f00d7373d5cfcff578435f66bf0d359c8c04bbaf3782e032ec` |
| Original local report | 2026-11-25T09:43:09.389572+00:00 | `f91c909bdecd068f5d5df48b201ecd51181323835b4e1029330fbabb66c6550e` |

The committed score summary has SHA-256 `acb2a7a9fa2aab719f4722540466e16e70713e88e9b4a2988d5f560387fe473f`.
