# Four-hour campaign handoff — 2026-09-13

## Decision

Do not extend the unchanged raster/PQN recipe. The two learning-curve arms each
reached the 150,000-transition budget with valid initial, 50k, and final
checkpoints. The frozen 12-call paired evaluator completed with
`INCONCLUSIVE_NOT_ADVANCED`: both training seeds were below both 50k and
initial checkpoints on the predeclared overall mass-integral comparisons, and
every corresponding opponent-mix mean was negative.

This is descriptive evidence from two training seeds, not a confidence interval
or promotion result. Apex `vector61` remains the incumbent. The repaired paired
evaluation is the policy-quality and promotion authority.

The original fixed-opponent pair remains incomplete: seed 1 stopped on its
finite-behavior tripwire and seed 2 hit its worker wall. The later operational
follow-up only established execution controls; it does not repair that causal
comparison.

## Current source qualification

The current clean detached qualified source is the diagnostic-observer successor at
`7a5274284be50432705adbaf2f8fe2067b74fbbd`. Its own full non-slow suite passed
with 2,749 tests. The prior product baseline is
`/Users/josenunez/Projects/ml/snake-dqn-four-hour-qualified-source` at
`002052329f3feae697d64d7d0e792cb7a3ee1e71`; it contains the reviewed circular
spawn fix and PQN archive-manifest rollback fix.

Focused evidence is complete:

- Capacity CLI regression slice: 139 passed in 12.77 s.
- Circular-arena successor: 32 passed in 0.52 s.
- Archive rollback successor: 80 passed in 3.20 s.
- Static formatting, import, lint, and review checks passed for the focused
  changes.

The new user-facing archive control is optional: `--archive-checkpoints` writes
per-invocation checkpoint records under `checkpoints/run_*/events.jsonl`.
Checkpoint publication is create-only and manifest replacement is atomic; a
caught hash or manifest failure removes the newly created file. This is not a
claim of an atomic two-file transaction across SIGKILL or power loss. Focused
coverage verified start, cadence, and final checkpoint preservation. Native first-run
and immutable-resume smokes are complete; the mutable-input resume attempt is retained
separately as a source-drift partial result.

The prior product baseline's full non-slow suite completed: 2,745 passed, 5
skipped, and 3 deselected in 94.04 seconds, with a natural reaped exit and no
drift. This qualifies the product source at the non-slow-suite boundary. The first native
archive smoke and its immutable same-target resume both completed naturally under separate
receipts. The mutable `latest_pqn.pth` resume also returned 0, but the supervisor correctly
classified it as partial because that output path changed its input hash. The public capacity
CLI then completed two 5,000-frame scripted calls with the fresh bounded body-storage policy.
It used one mix and has no strict-evaluation or promotion authority.
The frozen learning source `44acc33`, frozen capacity evaluator source `175dd21`,
and original main checkout remain separate historical authorities; a later
product source cannot change their past experiment results.

The current observer source `7a5274284be50432705adbaf2f8fe2067b74fbbd` has focused review,
static checks, four focused tests, and its own passing full non-slow suite. It is a
diagnostic-evaluator successor; its current-source qualification does not change past frozen
evaluation evidence. `0020523` remains the prior product baseline and archive-CLI evidence source.

## What the campaign established

- The completed curve weakens the case for longer training under the unchanged
  recipe.
- A capacity-repaired eight-cell operational evaluation and the curve evaluator
  both did not advance their diagnostics. Neither result promotes a candidate.
- The reflection probe shows output-coordinate right preference and no world-x
  reflection equivariance on its constructed paired inputs; its legal masks do not
  force those retained-right decisions. Because the reflected west-heading inputs are
  outside the absolute-east reset-start support, asymmetric experienced-state coverage
  remains a plausible contributor. This is not a universal-collapse or quality finding.
- The historical VC1 resource-only replay completed its old two-seed condition.
  It showed that disposable heads can fit the fixed visible-food training corpus;
  it does not establish observation sufficiency, RL learnability, or an encoder
  recommendation.
- Packed tactical paint has matched performance-profile gains, while padded
  batches lower RSS but fail strict numerical equivalence. Neither is a policy
  result.

Read the receipt-backed [experiment ledger](EXPERIMENT_LEDGER.md) for every
failure, setup issue, cap, successor, resource observation, and artifact hash.
Read the current research reports before using their diagnoses:
[learning curve](</Users/josenunez/Projects/ml/snake-dqn-artifacts/four-hour-20260913/research/learning-curve-results.md>),
[reflection](</Users/josenunez/Projects/ml/snake-dqn-artifacts/four-hour-20260913/research/reflection-results.md>),
[VC1 completion](</Users/josenunez/Projects/ml/snake-dqn-artifacts/four-hour-20260913/research/vc1-resource-completion-results.md>), and
[objective diagnosis](</Users/josenunez/Projects/ml/snake-dqn-artifacts/four-hour-20260913/research/objective-diagnosis.md>).

## Critical unknowns

The evidence does not yet identify the cause of the learned right preference.
The next causal analysis must distinguish training targets, sampled data,
legal-action selection, Q(lambda) targets, TD residuals, and per-action loss
contribution. The completed objective review confirms a non-equivalence/alignment gap between
the intended discounted training return and the 5,000-frame undiscounted mass integral. Whether
that gap caused the observed minimum-mass trajectories remains a hypothesis; PBRS invariance
also depends on true termination or an exact transformed bootstrap, and finite Q(lambda)
boundaries can retain residual effects.

No result establishes a better observation encoder, recurrence, replay
algorithm, self-play schedule, padding mode, or flip augmentation. Legacy
`PQNTrainer.flip_augment` is not a physical reflection because it leaves
absolute world-x unchanged; do not enable it as a remedy.

## Next bounded wave

1. **Qualification owner:** retain the completed qualified-source suite, archive first-run,
   immutable resume, and capacity-CLI receipts. Treat the mutable-resume drift as a historical
   partial attempt, not a failed policy. Any broader public-interface qualification needs a new,
   separately reviewed scope and receipt.
2. **Diagnosis owner:** the completed objective trace bound four fixed-world initial/final
   capacity cells, reduced exactly to the prior references, and retained observed trajectories.
   Its final checkpoints chose relative right on 19,447 of 19,460 valid decisions (99.933%) for
   seed 1 and 10,681 of 10,711 (99.720%) for seed 2, with no boost actions or executions. This
   fills the prior trajectory-evidence gap but remains diagnostic correlation, not reward
   causality. The older objective-trace/v1 packet remains frozen and unrun. The training-side
   TD trace remains unrun; it must retain legal left/straight/right opportunities, selected
   actions, Q(lambda) targets, TD residuals, and per-action loss contributions before a later
   causal intervention is selected.
3. **Experiment-design owner:** turn the diagnosis into a single-cause,
   separately frozen proposal. It must use fresh training seeds and the repaired
   paired evaluator. Do not spend a new 50k learning budget on the unchanged
   recipe.
4. **Promotion owner:** keep Apex as incumbent until a candidate passes the
   shared paired promotion gate. A diagnostic or source-qualification pass is
   not promotion evidence.

## Resource and record policy

Only one supervised compute-heavy process may run at once. The four-hour scientific sequence
used 6 of 8 policy-learning arms, 35 of 40 strict calls (17 historical, 12 curve, 2 capacity
CLI, and 4 objective trace), and 2,755.377469873754 of 4,800 supervised parent seconds in the
scientific sequence. This sum is process time, not campaign elapsed time; setup-only work,
profiling, tests, and side
diagnostics are accounted for separately. Before any future
numerical work, reserve the slot and record source closure, manifest/config
digest, seed plan, CPU/MPS limits, wall/RSS/reserve caps, expected checkpoint and
evaluation closure, and successor relationship. Retain natural exits, caps,
setup failures, and failed tests as new ledger rows; never rewrite them as
successes.

The numerical window is closed. The unchanged-recipe learning sequence and the separately bound
objective trace are complete. Any later work is a new bounded wave by default.
