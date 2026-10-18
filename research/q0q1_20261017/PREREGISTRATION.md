# DRAFT_PENDING_OWNER_REVIEW — Q0/Q1 B8 preservation factorial (2026-10-17)

This is a prospective registration for Q0 and Q1.  It implements the local plan
in `docs/research/specs/after_t23_plan_2026-10-17.md` and adopts
`docs/research/advice/2026-10-17_after_t23_league.md` sections 2--4 as controlling
rules.  The RL and compute advice add implementation and capacity detail only;
where they differ, the league advice controls.  No result, audit, runtime,
qualification, training, or evaluation is claimed here.  This document does not
authorize real training, Colab/pod dispatch, tmux operations, serving changes, or
release decisions.

## Question, endpoint, and comparator

T23's 8M R0 recipe lost fixed-opponent mass against B8.  Q0 asks whether that
result and B8 initialization are measured faithfully.  Q1 then tests whether the
restarted schedule package and/or three-learner collection damage a competent B8
policy.  The primary endpoint is higher **terminal-hero H5000 full-horizon mass
integral** than released bare B8 on fixed opponents, with dead frames contributing
zero.  Survival, food, boost, action, Q, loss, and return measures are guards or
diagnostics unless stated otherwise.

The comparator is released B8 SHA-256
`3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`, with no
veto.  Q1 begins from its weights only: online and target are initialized
identically, with fresh Adam and empty replay and label rings.  It is never an
exact resume.

## Shared governance and budget

Q0/Q1 is a finite dependency graph: Q0a and Q0b must pass their stated acceptance
checks before Q1 is admitted.  Q0 is capped at **15 units**, including all setup,
qualification, copies, cleanup, CPU work, and the two GPU packets.  Q0a is capped
at **8 CPU-runtime-hours**, **480** new development H5000 episodes, and **4,096**
held-out observations.  Q0b has one L4 and one A100 packet, each at most **60
minutes**, plus at most one hour of available Mac correctness checks.

The combined Q0+Q1 envelope is at most **150 Colab units**, **24 GPU-runtime
hours**, and **40 CPU-runtime-hours**.  Stop at the first applicable cap.  Reserve
20% of each envelope before dispatch, so a complete study must have a measured
forecast within 120 units, 19.2 GPU-hours, and 32 CPU-hours.  All startup,
qualification, copies, evaluation, failed attempts, and cleanup are charged.
Record the observed CPU debit rate and include it in units; it is unknown at
registration and must not be recorded as zero.

Q1 scientific training is capped at **60 units plus its admitted CPU evaluation**;
the aggregate envelopes remain stricter if reached first.  At most **three Colab
runtimes** may be concurrent.  Begin with at most two L4 online learners and one
CPU worker; independent runtimes can parallelize seed work, but each runtime hosts
one learner and its actors/rings.  A100 is used only for Q0b's required comparison
unless the complete post-Q0 forecast proves an allowed placement; an L4 and A100
may not be used to confound Q1 arms.

Planning inputs are not live quotations: L4 is supplied as 2 units/hour and about
7.6k N=3 learner decisions/second; A100 is supplied as 6.8 units/hour and has
about 7.7k measured N=3 decisions/second (about 14k is a conditional projection).
N=1 has no qualified price or throughput at registration; use one third of N=3
throughput only as a conservative pre-Q0 budgeting sensitivity, never as measured
capacity.  An A100 must demonstrate whole-identical-job value under the admission
envelope; the supplied rate ratio requires more than 3.4× L4 useful work/hour to
cost fewer units.  Missing measured pricing, failed target-runtime parity, an
incomplete timing matrix, or a runtime that cannot hold the common qualified
configuration blocks admission rather than changing one slow cell.

Strict gates and serving qualification remain Mac-only.  Colab rows are
development evidence and cannot prove universal Mac/x86 identity.

## Q0a — bounded descriptive audit

### Inputs and acceptance checks

Hash-check the released B8 initializer, all five T23 finals, and every available
2M/4M snapshot.  Missing snapshots stay missing; no checkpoint is reconstructed
to make a curve.  Zero-update loading must reproduce B8 actions on 4,096 fresh
observations spanning short and long bodies, crowded and open states, exact Q
ties, boost eligibility, and all scripted/frozen/mixed contexts.

Fail closed on checkpoint fallback, wrong wrapper identity, mask disagreement,
or frozen-seat replay contamination. The ordinary all-false advisory mask
uses the established normal-only fallback; test agreement explicitly and
count/exclude trapped rows from the retained natural fidelity corpus. This
is distinct from prohibited checkpoint substitution.  Frozen controls must accumulate neither
gradients nor optimizer state.  The audit must also independently verify current
collector/five-step target semantics, death versus segment truncation, and no
cross-respawn bootstrap with scalar fixtures.

If all ten 2M/4M snapshots exist, evaluate each snapshot and B8 on 16 scripted,
8 frozen, and 8 mixed fresh terminal-hero H5000 worlds per training seed:
`5 × 32 × (2 snapshots + B8) = 480` mass-decomposition policy episodes.  The
B8 fidelity observation capture reuses those same 160 worlds; it creates no
second development-world allocation.  Its 160 B8 executions and its at-most-4096
captured observations are reported separately from the 480 policy-episode cap.
The completion receipt must make that reuse and both accounting denominators
auditable before Q1; a nominal 640 independent development-episode allocation is
not registered.  These observations are retrospective diagnosis, not a
replacement endpoint and cannot rescue T23's existing 8M KILL.

### Descriptive report

For every available artifact, report full-horizon mass and cumulative mass at
500/1000/2500/5000; alive fraction; no-death survival; food mass per alive
decision and whole horizon; length distribution; death cause; greedy and executed
boost with eligible denominators; B8 action disagreement; Q gaps/scales; TD and
anchor loss; training-return versus H5000-mass ranking; and scheduled values.
Stratify by length, including `<700` and `>=800`, but do not promote these strata
to efficacy endpoints.  Conditional-alive mass is diagnostic and carries
survivor-selection bias.  Lower food intake with unchanged survival supports a
productivity-loss description but does not identify a causal loss term.

The reward audit reports the actual mass/reward and target terms.  It must not
misstate gamma as merely rescaling a pure length reward: terminal potential, death
handling, discounted accumulation, and the five-step target remain material.
Q0a produces no efficacy GO.

## Q0b — implementation correctness and capacity qualification

### Packages that must exist before Q1

The trainer exposes the following explicit, default-preserving interfaces:

```text
--schedule-package restart|mature        # restart remains the default
--collection-profile legacy|task-aligned-n1  # legacy remains the default
--eps-start FLOAT --eps-end FLOAT --anchor-start FLOAT
```

`restart` retains T23 R0 behavior.  `mature` applies gamma `.998`, epsilon `.01`,
and anchor `.1` from learner decision `t=0` through 8M, with gamma used in reward
construction, stored discounts, five-step returns, and bootstrap targets.  It
uses explicit constant start/end values and a positive epsilon span; no decay
denominator is set to zero.  Tests at `t=0`, `2,000,000`, and `8,000,000` assert
gamma, epsilon, anchor, reward gamma, and target discounts exactly.

`legacy` is the T23 N=3 collector.  `task-aligned-n1` is a separate profile, never
the old N=1 default: one learner hero is in the historical gate hero slot; five
fixed opponents and wrappers/mechanics are constructed exactly as the historical
scripted, frozen, or mixed gate mix; and the mix is selected with equal probability
at each whole-world reset.  It has its own `q1-task-aligned-n1-20261017...`
namespace.  It does not use gate worlds, does not draw five uniform policies from
the six-policy pool, and logs both reset allocation and actual transition exposure.
On hero death it finalizes the terminal transition and resets the whole world.
At H5000, a surviving world truncates under the already registered bootstrap
semantics.  N=3 retains T23 segment/respawn semantics.  Discounted TD still only
surrogates uniform H5000 mass; this collection package does not claim to remove
that objective mismatch.

### Required Q0b oracle and parity suite

Before timing, pass independent scalar or old-path oracles for:

1. learner-seat ownership and counter accounting;
2. death/respawn n-step isolation, including terminal N=1 flush and H5000
   truncation bootstrap;
3. fixed-policy action identity, legal/advisory mask resolution, exact Q tie
   behavior, and frozen-policy immutability (no gradient or optimizer state);
4. exact historical scripted/frozen/mixed roster, hero slot, wrapper, mechanics,
   equal reset allocation, and new-namespace construction; and
5. full-horizon evaluator accounting with mass zero after terminal hero death.

Both short-body and long-body fixtures must pass outside the timed windows.  T22
and T24 target-runtime parity must pass for the candidate runtime; Mac CPU fixture
parity alone is insufficient for CUDA qualification.  The old/default path must
remain covered and defaults must remain unchanged.

The implementation names these checks in `tests/test_q1_trainer.py`:
`test_mature_schedule_is_constant_at_registered_q1_checkpoints`,
`test_restart_schedule_defaults_are_unchanged_and_explicit_starts_work`,
`test_task_collection_mix_draws_are_stable_isolated_and_uniform_by_probability`,
`test_task_collection_terminal_and_horizon_boundaries_are_distinct`,
`test_terminal_nstep_flush_prevents_death_respawn_credit_leakage`,
`test_horizon_truncation_keeps_the_registered_bootstrap_not_a_terminal_discount`,
`test_task_reward_gamma_changes_the_live_simulator_contract_at_schedule_boundaries`,
and `test_task_reset_replays_constructor_food_draw_path_for_fresh_gate_seed`.
`tests/test_q1_qualification.py` binds the timing, lag, counter-debt, target-CUDA,
and draft-admission requirements.  These are implementation acceptance tests, not
evidence that a runtime packet has passed.

### Capacity packet and freeze rule

On one L4 and one A100, after setup and completed compilation/replay readiness,
time every schedule-by-collector cell: N=1/N=3 × restart/mature.  Each cell has
at least 120 seconds warm-up followed by two 3-minute scored windows: 24 scored
minutes per hardware packet.  Setup, warm-up, and cleanup must fit in 60 minutes;
if they do not, write `INCOMPLETE` and do not shorten a scored window.

The packet records useful learner decisions, world frames, emitted replay rows,
finalized updates, target/policy-lag mean and p95, age at ingestion, long-body
exposure, memory, setup/export/cleanup time, observed GPU and CPU price, clean
exit, ring overrun/dropped-transition counts, and final counter reconciliation.
Require finite updates, no dropped transitions/ring overruns, no unexplained
same-collector schedule lag increase, and no more than 10% throughput decline
between scored windows.  Preserve fp32, batch-scale 2, raw local rings where
memory permits, queue depth 2, and 256 emitted replay rows/update.

Before Q1, select a bounded candidate actor/world/queue configuration and freeze
one common actor count within N=1 across C/D and within N=3 across A/B.  The same
schedule pair must not receive different actor counts.  If a runtime lacks CPU or
RAM for its package's frozen configuration, it is ineligible.  The Q1 manifest
binds the measurement, selected configuration, target device/software stack, and
observed pricing; no configuration is inferred from an unmeasured projection.

## Q1 factorial

### Invariants

All 20 runs retain the B8 network and observations, masks, mass reward definition,
kill scale zero, five-step returns, optimizer LR `1e-4`, fp32, target cadence,
replay and anchor batch sizes, and v8 label generation.  The established bs2
accounting is 768 TD replay rows plus 256 anchor rows/update and one update per
256 ingested transitions.  There is no reward shaping, recurrence, BF16, larger
network, LR search, PFSP, opponent publication, checkpoint selection, or changed
evaluation-mix weighting.  The final manifest records source/configuration,
initializer and opponent hashes, hardware, software, namespace, seed, counters,
and receipt identity.

Use five fresh collision-audited seed blocks, paired across A--D by block and
initializer: `261701`, `261703`, `261704`, `261705`, and `261707`.  Seed `261702`
was rejected by the pre-data collision audit and is not a Q1 seed.  The fresh bank
namespace is `q1-mature-n1n3-h5000-terminal-hero/20261017/v1`.  Training and
evaluation namespaces are disjoint and collision-audited against prior training,
labels, snapshots, development, and strict banks.  The authoritative training
namespace is `q1-t23-n3-20261017/v1` for both A/B and
`q1-task-aligned-n1-20261017/v1` for both C/D.  Arms use separate create-only
outputs and share only the paired exogenous seed block; they do not get invented
per-arm training namespaces.  Shared exogenous seeds do not make trajectories
identical after actions diverge.  The five blocks, conditional on B8, are the
independent experimental units; actor slots, descendants, frames, seats, and
checkpoints are not extra replicates.

Each paired four-arm seed block stays on the same qualified GPU type and software
stack.  Prefer one GPU type for the study.  If preassigned heterogeneous blocks
are unavoidable, report the blocks and do not pool a hardware-specific arm into a
recipe effect.  Run independent seed jobs in parallel only across allowed
runtimes; do not oversubscribe several learners to one small CPU allocation.

| Arm | Schedule package | Collection package |
|---|---|---|
| A | T23 R0: gamma `.99` through 2M then `.998` at 8M; epsilon `.05` to `.01` at 5M; anchor `1` through 8M | N=3 immutable six-policy pool with T23 seat and segment sampling |
| B | Mature: gamma `.998`, epsilon `.01`, anchor `.1` from `t=0` through 8M | Same N=3 package as A |
| C | Restarted package as A | Task-aligned N=1 terminal-hero package, equal scripted/frozen/mixed reset probabilities |
| D | Mature package as B | Task-aligned N=1 package as C |

Each run ends at 8,000,000 actual learner decisions.  Frozen-seat decisions never
count.  Emit replay rows and finalized optimizer updates separately and reconcile
them with learner decisions; expected update tokens are about 31,250/run before
warm-up/tail accounting and about 625,000 across 20 runs.  N=3 is expected to use
about 2.67M all-live world frames/run and N=1 about 8M, so equal learner data is
not equal simulator work.

### Registered counter overshoot and artifacts

The multi-actor lane increments its learner-decision counter only after accepting
an actor-frame message, but the learner drains up to one message from **each** of
`K` actors before the outer-loop budget condition is checked again.  For frozen
Q1 values `K` actors, `W` worlds, and `N` learner seats, the accepted decision
overshoot is bounded by `K × W × N − 1`, not by one actor message.  Register `K`,
`W`, `N`, this bound, target, actual final learner count, and overshoot
(`actual − 8,000,000`) in every receipt.  Unconsumed queued messages do not count
as learner decisions.  Emitted replay rows and their target/update debt are a
separate reconciliation: record emitted rows, completed updates, pending update
debt, and frozen/pool decisions without using any of them to relax the learner
decision endpoint.  A receipt exceeding the registered counter bound or failing
that reconciliation is `INVALID/INCOMPLETE`; do not trim rows or call it exact
8M.

Checkpoint cadence is every 500,000 learner decisions, retaining checkpoints at
0.5M, 2M, 4M, and 8M as required.  All checkpoint values, including the other
500k multiples, are diagnostics only.  Only the 8M final decides gameplay
efficacy.  A healthy arm is not stopped for a poor 2M curve; no intermediate
checkpoint may replace the final.

### Tripwires and run completeness

Kill the affected arm for nonfinite loss, Q values, or gradients; invalid actions;
more than 90% one-action share in a complete 100k learner-action window; greedy
boost over 2% overall in such a window; or greedy boost over 1% below length 700
in such a window.  Record zero-boost windows and eligible support, but impose no
new low-boost minimum quota.  A scientific tripwire has no replacement seed.

An operational interruption, time/unit/cap exhaustion, corrupt state, missing
identity/coverage, failed parity, nonzero operational exit, or failed counter
reconciliation is `INVALID/INCOMPLETE`.  Preserve the failed attempt and do not
automatically rerun, replace a seed, add a training extension, or substitute an
intermediate checkpoint.

## Q1 evaluation and analysis

Create five disjoint, collision-audited, fresh terminal-hero H5000 world banks,
one per seed block: 64 scripted, 32 frozen, and 32 mixed worlds.  Within a bank,
only the hero changes; A, B, C, D, and B8 each use the same world seed, ordered
opponent roster, seats, wrapper, profile, and mechanics.  The 128-world B8 row is
reused only in that exact bank and manifest.  Total evaluation is therefore
`5 seeds × 128 worlds × 5 identities = 3,200` episodes.  Evaluation uses one CPU
thread per world and parallelizes only immutable shards.  CPU Colab evaluation
uses the pinned evaluator, immutable shard receipts, and a per-runtime T23-style
Mac-reference parity fixture.  The Mac uses the `q1_parity` CLI's `--reference`
mode to produce the reference; each evaluation runtime uses its
`--verify-reference` mode.  The four-world parity probe is a smoke only.

The analyzer accepts a `q1-analysis-input/v1` object containing the committed
bank, its collision audit, and complete immutable shard receipts.  It does not
accept an unbound list of evaluation rows.  Each receipt must bind bank/profile,
source, runtime parity/environment, B8 identity, candidate training provenance,
and exact shard coverage.

For arm `j` and seed `s`, let `G[j,s]` be the equal-weight mean of the three paired
mix mass differences from B8.  Report per-mix/per-seed effects, mean `G`, and a
one-sided 90% Student-t lower and upper limit across the five seed effects
(`df=4`).  Use the seed effects, not frames or seats, as `n`.  Also report a
hierarchical world-within-seed bootstrap sensitivity, retaining paired world
structure; it is a sensitivity analysis, not an increase in independent seed
count.

Report the predeclared package contrasts B−A and D−C (schedule), C−A and D−B
(collection), and `(D−C)−(B−A)` (interaction).  These test packages, not single
constituents: collection changes seats, opponents, and terminal-world handling;
schedule changes gamma, epsilon, and anchor pressure.  Report every arm and
contrast.  The development intervals are selection screens, not familywise-
controlled discovery claims or strict confidence guarantees.

## Q1 analysis and ordered decisions

Apply this order independently to every arm:

1. **INVALID/INCOMPLETE:** missing coverage or identity, failed parity, corrupt
   state, or an operational cap.  Preserve evidence.  There is no replacement
   seed or automatic rerun.
2. **KILL:** a training tripwire; pooled alive-fraction difference versus B8 below
   `-.03` on any mix; any seed's scripted alive difference below `-.15`; pooled
   no-death episode fraction below `.50` on any mix; or `upper90(G) <= 0`.
3. **GO to fresh Mac Phase R preparation:** `lower90(G) > 0`, mean `G >= 20` mass,
   scripted mass point estimate `>= 0`, at least 4/5 positive seed effects, and
   every guard passes.
4. **INCONCLUSIVE:** every other valid result.  There is no optional world
   extension, automatic training extension, threshold revision, or recipe
   replacement in Q1.

A distinct preservation certificate can admit only Q2.  It requires every guard,
`lower90(G) > -10`, and scripted mass point estimate `>= -10`.  It can coexist
with efficacy KILL and does not make an inferior recipe equally strong or permit
extension of that static recipe.

Choose at most one recipe.  If any arm receives efficacy GO, rank GO arms by
`lower90(G)`, then mean `G`, then fixed identifier order **D, C, B, A**.  Advance
that arm's five 8M finals to fresh Mac Phase R preparation; do not demand 20M or
a league first.  Otherwise, rank preservation-qualified arms by the same order
to choose Q2's parent.  If none qualifies, close this same-observation fine-tuning
route for this budget.  No result opens a sequence of LR or anchor sweeps.

Fresh Mac evidence remains required before any strict conclusion.  Q1 does not
change the released B8 checkpoint or serving behavior.


## Executable packet and receipt accounting clarification

Q0's fidelity observations are captured during its 160 B8 development worlds,
then consumed by the dependent initialization audit; no additional gameplay
worlds are collected for that audit. The 4,096 retained natural observations
must cover all registered strata; executable exact-tie and normal-only fallback
fixtures are separately reported. Missing natural coverage is INCOMPLETE.

For Q1, accepted learner decisions D and emitted replay rows R reconcile exactly
as `D - R = dropped_truncation_rows + pending_nstep_rows`, using each actor's last
accepted FIFO prefix. Optimizer updates U and debt reconcile as `R = 256*U + debt`.
An identity, counter, operational cap, or parity failure precedes efficacy.
Each arm requires all five complete seed blocks; a documented scientific
tripwire ends that arm without a replacement or a requirement to produce its
absent final. Unaffected complete arms retain their own decisions; contrasts
with an unavailable arm are reported unavailable. Operational failures remain
INCOMPLETE, including attempts retained without a final checkpoint.

Each L4 job has a 90-minute script timeout and a 5,100-second trainer cap,
allowing five minutes for export; queue setup is capped at 30 minutes. This
bounds a whole attempt to the additive two-hour RL-advice limit. CPU pricing,
100 per-job four-world parity probes, startup, copies and analysis enter the
measured global forecast before admission; they are not free work. Q1 evaluation
uses one CPU thread per world, with parallelism across immutable shards.
