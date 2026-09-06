# Independent statistical review: PQN sampler diagnostic

Date: 2026-09-06

This is a read-only evidence review of the completed revision-3 campaign. It is
an adaptive exploratory diagnostic, not a promotion test or an algorithm
ranking. The reviewed artifacts are
`runs/pqn_coverage_final_20260906/{protocol.json,training_runs.json,evaluation.json,analysis.json}`
and `verification/final_artifact_validation.json` beneath that run directory.

## Independent recomputation

I recomputed the primary estimand directly from all 204 raw evaluation rows: 17
policy conditions times 12 fixed SIMD worlds. For each training seed, I averaged
the 12 paired world-level mass-integral differences, full epoch minus legacy.
The five training-seed effects were:

| Training seed | Initial | Legacy | Full epoch | Epoch - legacy |
| ---: | ---: | ---: | ---: | ---: |
| 401 | 3.7670 | 6.8724 | 3.4857 | -3.3868 |
| 502 | 1.7558 | 5.8444 | 1.1012 | -4.7433 |
| 603 | 2.9493 | 2.1029 | 1.1253 | -0.9776 |
| 704 | 2.8103 | 1.8458 | 3.8462 | +2.0003 |
| 805 | 1.3709 | 1.1000 | 1.3602 | +0.2602 |

With training seed as the independent unit (`n=5`, `df=4`), the mean paired
effect was **-1.3694 mass-integral units**, with a nominal two-sided 95% Student-t
interval of **[-4.7480, +2.0092]**. Two seed pairs favored full epoch and three
favored legacy. This reproduces `analysis.json`; the prespecified diagnostic
success criterion was not met.

The descriptive secondary effects also crossed zero:

- Legacy minus paired initialization: +1.0225, 95% CI [-1.9453, +3.9903].
- Full epoch minus paired initialization: -0.3470, 95% CI [-1.6352, +0.9413].

Neither recipe therefore demonstrated reliable improvement over its paired
initialization in this small screen. The primary interval permits both material
harm and moderate benefit; it is not evidence of equivalence and does not prove
that full epoch is worse.

## Mechanical result and causal scope

The sampler change worked mechanically. Legacy used 68.132% of eligible rows at
least once, while full epoch used 100%. Across the five runs per arm, legacy made
2,903,004 transition draws and 12,024 optimizer steps; full epoch made 2,503,528
draws and 11,237 optimizer steps. Full epoch exposed about 800,000 more unique
transitions, but legacy performed about 399,000 more draws and 787 more optimizer
steps.

The performance estimate consequently belongs to the complete sampler-recipe
change at an approximately equal hero-transition target. It does not isolate
unique coverage as the sole causal mechanism. The result shows that increasing
coverage to 100% under this recipe did not demonstrate a mass-integral benefit;
it does not show that discarded rows are irrelevant or that PQN is unsuitable.

The inference is conditional on these five training seeds, the 12 fixed SIMD
worlds, the 1,000-frame horizon, mechanics v2, and five SIMD greedy-food
opponents. The scripted reference means (greedy 28.0759 and random-safe 3.0478)
are descriptive context, not promotion thresholds or prespecified comparisons.

## Health and artifact checks

All ten training runs completed within the numerical and resource guards.
Maximum observed absolute Q was below 14, versus the 1,000 alarm. A few small
rollouts were unanimous in their raw actions; the largest collapse streak was
three updates containing only 144 eligible rows, below the 2,048-row emergency
threshold. It is accurate to say the runs were healthy under the specified
tripwires, but not that no local unanimous-action rollout occurred.

The independent final validator passed: it verified 87 frozen scientific source
files, all ten checkpoint tensor hashes, the five invariant paired
configurations, all optimizer batch counts (including `ceil(eligible / 256)` for
every full-epoch update), the exact training order, and all 204 raw evaluation
outcomes.

## Decision

Keep the legacy sampler as the default and keep PQN experimental. This campaign
provides no evidence for switching the default sampler or promoting PQN. The
supported conclusion is a failure to demonstrate benefit from the full-epoch
recipe in this diagnostic, not proof that full-epoch sampling or PQN is worse.
No additional run is required to close this campaign.
