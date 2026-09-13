# Four-hour RL work window — 2026-09-13

This directory is the operational record for the bounded four-hour Snake DQN
work window. It separates prepared work from executed work so that creating a
driver, manifest, or review assignment is never reported as an experiment
result.

## Research outcome

Keep Apex as incumbent and stop extending the tested PQN recipe unchanged. Both independent
150k training arms performed below their own 50k and initial checkpoints in the declared
full-horizon mass comparisons. The later trace explains the observed play: both 50k trained
policies overwhelmingly turned right and settled into tiny four-position loops. This establishes
the behavior on the measured worlds; it does not establish its training cause.

| Training seed | Mean mass change, 150k minus 50k | Mean mass change, 150k minus initial |
| --- | ---: | ---: |
| Seed 1 | -1.143075 | -0.492600 |
| Seed 2 | -6.426900 | -0.642600 |

These are descriptive effects from two independent training seeds, using four paired evaluation
worlds in each of two opponent mixes. They do not provide a population confidence interval,
Apex-versus-PQN comparison, or promotion result.

The separately reproduced 50k trajectories contained 30,128 right-family choices in 30,171
valid decisions. Food events fell from 543 initially to five after training, and boost events
fell from 1,269 to zero. Every final trajectory contained an exact period-four head-position
loop; final policies visited only 4-25 cells compared with 333-1,716 for initial checkpoints.
Read the [independently checked trajectory report](</Users/josenunez/Projects/ml/snake-dqn-artifacts/four-hour-20260913/research/objective-trace-results.md>) for per-world details.

The implementation delivered faster corrected tactical painting, explicit bounded evaluator
storage, optional immutable checkpoint archives, safe snapped GameState circular spawns, and
a detached evaluation-frame observer. The final product source `7a527428` passed 2,749 tests,
with five skips and three slow tests deselected. Public archive/resume and evaluator CLI smokes
passed on the prior product baseline `0020523`; exact source boundaries remain in the ledger.

Start the next work from [NEXT_WAVES.md](NEXT_WAVES.md): retain training targets, TD residuals,
and per-action legal opportunities, then test a single opt-in objective change. Keep optimizer,
replay, architecture, and observation experiments separate.

The [learning-curve v2 plot](</Users/josenunez/Projects/ml/snake-dqn-artifacts/four-hour-20260913/analysis/learning-curve/learning-curve-mass-integral-v2.png>) renders mean mass by checkpoint and opponent mix from paired-world data and has a visually approved footer. Its render supervisor completed naturally in 0.4204230420291424 s with 84,803,584-byte peak RSS and no drift; the image SHA-256 is `7dcab2e51f8305c631334c846577419ddc903374c9b6f85d62b2aa67da205309`.

The campaign starts at `2026-09-13T02:47:41+00:00` and ends at
`2026-09-13T06:47:41+00:00` (`2026-09-12 19:47:41` through `23:47:41` PDT).
No new numerical work may start after `2026-09-13T06:32:41+00:00`; the last
15 minutes are reserved for terminal checks, artifact closure, and handoff.

The campaign uses source
`/Users/josenunez/Projects/ml/snake-dqn-four-hour-source` at base revision
`44acc33b1f74d1cfe8a4d8e438fc5420eaf0baf0`. The integration worktree is
`/Users/josenunez/Projects/ml/snake-dqn-four-hour`. The original main checkout
is recorded as unchanged at
`950d583b9eb0c3e316edd0f651954d133334ac1a` when the campaign began.

Read [EXPERIMENT_LEDGER.md](EXPERIMENT_LEDGER.md) for factual run status and
[HANDOFF.md](HANDOFF.md) for the current next action. The authoritative mutable
campaign metadata is
[`campaign.json`](</Users/josenunez/Projects/ml/snake-dqn-artifacts/four-hour-20260913/campaign.json>).
Terminal receipts and frozen manifests take precedence over planning metadata.

## Decision boundary

The Apex `vector61` checkpoint remains the incumbent. The first learning study
uses the qualified, opt-in watch-aligned PQN decision phase in a fixed-opponent
setting. A completed study can provide diagnostic evidence only; promotion
continues to require the shared paired tournament gate and its prescribed
opponent mixes.

## Resource policy

Only one numerical job may run at once. Training uses at most two native CPU
threads, a 4 GiB RSS ceiling, an 8 GiB MPS-driver ceiling, and a 6 GiB
available-memory reserve. The initial available-memory observation was
35,496,427,520 bytes. Non-numerical research, artifact audit, documentation,
and code review may proceed in parallel when they do not contend with the
numerical run.

## Current campaign snapshot

The initial two-seed, fixed-opponent study stopped incomplete: one arm hit its
finite behavior tripwire and the other its worker grace-wall boundary, so it
produced no final checkpoints or evaluation. A separately frozen operational
follow-up then completed both 50,000-transition arms with exact coverage,
retained checkpoints, strict-profile loads, and changed weights. That is
engineering qualification, not a policy-quality result.

The first full follow-up evaluation stopped after five completed calls when its
sixth scheduled call exceeded the evaluation world's declared capacity. Its
partial record is not an outcome. A paired reproduction of that capacity
boundary completed. The full eight-cell recovery evaluation then completed with
opposing seed deltas and was `INCONCLUSIVE_NOT_ADVANCED`; it did not advance
the candidate. Both learning-curve arms then completed their 150,000-transition
budgets with retained checkpoints and validated closure. Their frozen evaluator
then completed all twelve calls and was `INCONCLUSIVE_NOT_ADVANCED`: both
training seeds had negative final-minus-50k and final-minus-initial overall
deltas, and every corresponding mix mean was negative. This stops more training
under the unchanged recipe and does not promote a candidate. The bounded
resource-only VC1 replay also completed naturally and only closes its historical
operational condition. The separate reflection probe then showed output-coordinate
right preference and no world-x equivariance on constructed paired inputs, while
legal masks did not force the retained-right choices. Its reflected west-heading
counterparts were outside the east-reset training support, so asymmetric experience
remains plausible; see [HANDOFF.md](HANDOFF.md) before treating it as a result.

The matched packed-paint profile is qualified performance-only evidence:
control-wall time improved 11.206% on CPU and 23.165% on MPS with exact trace
checks. The padding comparator showed lower peak RSS for padded batches
(1,239.859375 MiB versus 1,831.359375 MiB unpadded) while taking 10.57% longer,
and it failed strict numeric equivalence at three checkpoints. Neither result
establishes learning quality or promotion eligibility.

The current diagnostic-observer source at `7a5274284be50432705adbaf2f8fe2067b74fbbd`
passed its own non-slow suite (2,749 passed, 5 skipped, 3 deselected). The prior
product baseline at `002052329f3feae697d64d7d0e792cb7a3ee1e71` passed its fresh
non-slow suite (2,745 passed, 5 skipped, 3 deselected). Its optional archive CLI
completed bounded first-run and immutable-resume smokes; the mutable-output resume
is retained separately as a source-drift partial attempt. The public SIMD capacity
CLI also completed its bounded two-call, one-mix engineering smoke. These are
product and interface qualifications, not policy-quality or promotion evidence.
The artifact-only objective trace completed its bound four-call diagnostic parity
check with unchanged inputs. It supplies trajectory evidence only and has no reward-
causality, policy-quality, or promotion authority.

The [experiment ledger](EXPERIMENT_LEDGER.md) is the attempt ledger for all
completed, partial, failed, prepared, and active lanes. It records each
receipt-backed outcome and the boundaries on its interpretation.
