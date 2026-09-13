# H1-A remote-heading decision-relevance report

Status: **the one frozen corpus completed, the independent raw-artifact audit
passed, and the decision is `INCONCLUSIVE_NOT_ADVANCED`.** The
probe found no holdout evidence that the controlled remote-heading rotation
requires different near-best hero actions under its finite, tape-controlled H16
return.
It therefore does not advance an information diagnostic, a history/recurrence
candidate, or a model change.

This report records the execution of the frozen
[H1-A protocol](PROTOCOL.md). That protocol explains why an observation alias
alone does not establish a different best action, and why these controlled
finite returns are neither optimal Q-values nor a measure of natural alias
prevalence.

## Frozen execution record

The single scientific execution used source revision
`e174fe48573f1af4bee6976624b1160e136eed76`. Its
[terminal receipt](/Users/josenunez/Projects/ml/snake-dqn-artifacts/observation-value-20260912/executions/h1a-20260912-v1/run/terminal.json),
[raw rows](/Users/josenunez/Projects/ml/snake-dqn-artifacts/observation-value-20260912/executions/h1a-20260912-v1/run/raw.jsonl),
[input manifest](/Users/josenunez/Projects/ml/snake-dqn-artifacts/observation-value-20260912/executions/h1a-20260912-v1/manifest.json),
[prelaunch approval](/Users/josenunez/Projects/ml/snake-dqn-artifacts/observation-value-20260912/executions/h1a-20260912-v1/root-prelaunch-approval.json),
and [supervision receipt](/Users/josenunez/Projects/ml/snake-dqn-artifacts/observation-value-20260912/supervision/h1a-corpus-v1/receipt.json)
are the authoritative detailed record. The
[root execution closure](/Users/josenunez/Projects/ml/snake-dqn-artifacts/observation-value-20260912/root-execution-closure.json)
records one scientific attempt, 113 retained execution files (3,953,064 bytes),
an unchanged source closure, and clean legacy and prior-integration checkouts.
The independent [raw-artifact audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/observation-value-20260912/independent-oracles/raw-audit-v3.md)
passed and its [machine-readable receipt](/Users/josenunez/Projects/ml/snake-dqn-artifacts/observation-value-20260912/independent-oracles/raw-audit-v3.json)
is SHA-256 `c3e177e7070636a7f719a807431d91d46b79151ae30f8d91ad88267fa02a5e56`.

The closure binds terminal receipt SHA-256
`98ac6372681b1321a23a7ed488418a68590e5f3f3b0aa88ee99e772a0f4d87f9`
and parent receipt SHA-256
`b7ba25e0d6ab4aa545655061bef9b9e037eceb7aa62cb12568d299f5031208b3`.
The root closure itself is SHA-256
`94d8a6248ca6daab1d8bd1a61f30d87c57429c5d7499135dac209429673954b2`.

An initial typo affected only the independent hand-world oracle's v1 preflight
([retained record](/Users/josenunez/Projects/ml/snake-dqn-artifacts/observation-value-20260912/independent-oracles/core-oracle-v1.json)).
The corrected v2 oracle passed before the corpus run. The preflight failure is
retained as operational history; it is not a corpus CLI failure or a scientific
experiment attempt, and it did not alter the frozen inputs or completed corpus.

## Result

| Scope | Retained pairs | Eligible worlds | H16 common-action regret | Disjoint near-best worlds | Decision use |
| --- | ---: | ---: | ---: | ---: | --- |
| Development | 9 | 7 of 8 | 0.0 throughout | 0 | Descriptive only |
| Holdout | 27 | 15 of 16 | 0.0 throughout; LCB 0.0 | 0 | `INCONCLUSIVE_NOT_ADVANCED` |
| Total | 36 | 22 of 24 | 0.0 throughout | 0 | No advancement |

The holdout corpus met the minimum eligible-world count (15, where 8 was
required), but no holdout world had disjoint near-best sets and the one-sided
world-clustered bootstrap lower confidence bound was 0.0. It therefore failed
the predeclared requirements of at least three disjoint holdout worlds and a
strict LCB greater than 0.01. One development world and one holdout world had
no retained pair and remain uncertain rather than zero-regret observations.

The independent audit recomputed zero regret, zero disjoint pairs, and zero
directional order disagreements at every recorded horizon; H16 alone had
decision authority.

| Horizon (total steps) | Pairs | Mean / max regret | Disjoint pairs | Directional disagreements | Holdout LCB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 36 | 0.0 / 0.0 | 0 | 0 | 0.0 |
| 8 | 36 | 0.0 / 0.0 | 0 | 0 | 0.0 |
| 16 (primary) | 36 | 0.0 / 0.0 | 0 | 0 | 0.0 |
| 32 | 36 | 0.0 / 0.0 | 0 | 0 | 0.0 |

The result is conditional on the protocol's geometric twin construction,
candidate selection, action tape, and finite continuations. In this corpus it
found zero common-action regret and no need to choose different near-best
actions; a zero disjoint-set count alone does not prove that the near-best sets
were identical. It supplies no evidence to add remote-heading tokens to
`raster31v3`, fit a GRU or other history model, or change the existing learner
network. It also does not show that the observation is globally adequate, that
natural aliases are absent, or that a learned policy would take the same action.
The Apex champion remains the incumbent; the earlier terminal B5 throughput
screen and its non-promotion status are unchanged.

The retained pairs still demonstrate the intended observational boundary: the
[supplemental raw evidence](/Users/josenunez/Projects/ml/snake-dqn-artifacts/observation-value-20260912/root-supplemental-evidence.json)
records 51 next-observation digest differences across 113 matched-action
comparisons, spanning 20 of the 36 pairs. No first-step reward or done flag
differed. This confirms controlled current-observation aliasing for those
pairs: a hidden-heading perturbation can change a later observation in this
corpus, while this probe measured no required action change. The supplemental
check found no current-observation or full-state digest reused across the
development/holdout split. These facts are not a natural alias-prevalence
estimate.

## Corpus health and verification

The worker completed in 23.916450 seconds; the supervising parent completed in
26.074647 seconds with confirmed exit 0. It ran one CPU process with one native
thread, peaked at 254,853,120 bytes (about 243.05 MiB) RSS, and remained within
the 2 GiB RSS and 900-second parent limits.

| Recorded work | Agent slots | Valid transition agent-slots |
| --- | ---: | ---: |
| Natural collection | 36,864 | 35,503 |
| Counterfactual branches | 39,852 | 33,481 |
| Total | 76,716 | 68,984 |

The 36 rejected candidates are also retained: 32 lacked a lowest-ID alive
single-segment enemy, 2 changed the heading-twin observation or mask, and 2 had
a dead hero. The raw audit recomputed every counterfactual count, but did not
replay the full natural collection; its 35,503 natural-valid figure is the
terminal's observed counter, not an independently replayed trajectory total.

The focused integration union passed 58 tests in 2.49 seconds; its
[log](/Users/josenunez/Projects/ml/snake-dqn-artifacts/observation-value-20260912/integration-union.log)
covers the evaluator, statistics, and diagnostic CLI. An independent
[hand-world oracle](/Users/josenunez/Projects/ml/snake-dqn-artifacts/observation-value-20260912/independent-oracles/core-oracle-v2.json)
also passed at source `e4fcfa415fea9bccbc1c2e9a7b6a27b3aba9c1ec`, checking
identical current observations, matched opponent tape, finite-return arithmetic,
row-0 hero-only overwrite, unavailable-action handling, and source immutability.

## Next state

The independent raw-artifact audit ran under one CPU thread in 1.698 seconds
and exactly replayed two representative retained pairs. No follow-on
diagnostic, larger model, retry, or quality evaluation is authorized by this
result. A later proposal would need a separate frozen design and fresh evidence;
it cannot be inferred from this small finite geometric screen.
