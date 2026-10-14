# First-lineage parallel preview — 2026-09-24

Status: **COMPLETE_AUDITED_EXPLORATORY_MIXED_RESULT**. The independent saved-output audit passed. This CPU-only preview ran alongside second-lineage MPS training under a separately reviewed concurrency exception requested by the user. The original controller, scientific protocol, source, locks and remaining training dose were unchanged.

The fixed first completed lineage2026096101 final31,005-rollout model (global update35,613) was compared with its mark4,608 parent on32 new paired solo worlds and32 new paired mixed-opponent worlds, H3000. No favorable endpoint or world selection occurred.

| Profile | Metric | Parent | Candidate | Candidate − parent | Paired wins / ties / losses |
|---|---|---:|---:|---:|---:|
| solo | Full-horizon living mass | 81.2239 | 66.1762 | -15.0477 | 17 / 0 / 15 |
| solo | Mean food | 231.5312 | 494.7500 | +263.2188 | 30 / 0 / 2 |
| solo | Mean frames alive | 1619.9062 | 2238.1562 | +618.2500 | 21 / 0 / 11 |
| solo | Fraction reachingH3000 | 0.0312 | 0.1562 | +0.1250 | 5 / 26 / 1 |
| mixed | Full-horizon living mass | 32.3387 | 48.7820 | +16.4433 | 20 / 1 / 11 |
| mixed | Mean food | 145.4062 | 391.3750 | +245.9688 | 30 / 1 / 1 |
| mixed | Mean frames alive | 953.7812 | 1696.6250 | +742.8438 | 25 / 1 / 6 |
| mixed | Fraction reachingH3000 | 0.0000 | 0.0312 | +0.0312 | 1 / 31 / 0 |

The mixed-opponent result is encouraging: food, time alive and the full-horizon living-mass score improved. Solo food and time alive improved, but living mass declined. Boost use increased sharply: mean full-horizon boost fractions were0.000979→0.385438 in solo and0.001042→0.288875 in mixed. Boost consumes body length, but its causal contribution to the observed mass tradeoff was not tested.

Only5/32 candidate solo games and1/32 mixed games reachedH3000; parent counts were1/32 and0/32. The result does not establish dependable episode survival, all-lineage reliability, package success or promotion. All three fixed lineages and the original held evaluation remain required.

## Accounting

All three preview receipts exited naturally with no guard breach. Root reconciled the audit’s own completion receipt once; no audit or gameplay was repeated. The preview used128 episodes,208,392 native/hero frames,11,103 batched model calls (355,296 evaluated rows including inactive lanes),424,380 opponent transitions,two checkpoint loads andzero optimizer updates. Guarded runtime was415.039 seconds. Minimum available RAM was30.56GiB; peak combined primary/preview RSS3.07GiB; peak preview RSS0.58GiB; maximum sampled global CPU50.1%.

The actual callback/runtime, source hashes, parent/candidate identities, per-world results, exclusion boundaries, resource receipts and review are bound in the artifact closeout. The saved-output audit parsed the new compact frame evidence only, using the old completed metadata solely for identity and seed exclusion. Opponent transition counts are hash-bound and bounded, not independently reconstructed from hero-only traces.

## Artifacts

- [Closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/task-aligned-first-lineage-preview-20260924/closeout.json)
- [Audited paired comparison](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/task-aligned-first-lineage-preview-20260924/run/stages/audit/output/comparison.json)
- [Independent audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/task-aligned-first-lineage-preview-20260924/run/stages/audit/output/report.json)
- [Frozen preview intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/task-aligned-first-lineage-preview-20260924/intent.json)
- [Implementation review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/task-aligned-first-lineage-preview-20260924/implementation-review.json)

No publication or push occurred for this preview. Its window is closed. Do not rerun its evaluation or audit.
