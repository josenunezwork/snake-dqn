# X0 follow-up waves: representative timing and per-environment reset

Date: 2026-09-08  
Status: **PLANNED FOLLOW-UP**. These packets were not executed in the repair campaign.
The resource budgets and advancement thresholds below are proposed experiment choices.

## Frozen boundary and evidence

The implementation source reviewed here is the clean integration commit
`7cb66a9437f63b89cd5dc8731b5f8577ce1c9765`. The current X0 campaign and every file
under `x0-corrected-pqn/` remain immutable. No follow-up may start until the root has
confirmed that all X0 learner processes have exited, frozen every terminal or partial
outcome, and recorded hashes for the source, manifests, telemetry, checkpoints, and
receipts. A follow-up uses a new dated artifact root and new seed namespaces; it never
extends, replaces, or relabels an X0 arm.

At this document's handoff, the root reports that all five X0 arms closed as retained
`wall_stop` outcomes and that the final analysis was independently accepted. The durable closure is recorded in `x0-root-closure.json`; all five outcomes are bound
by `x0-final-analysis.json` and its independent review. These files live under
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/rl-repair-20260908/`.

The retained evidence establishes the problem boundary:

- `x0-episode-utilization.md` shows a functioning batch barrier: completed environments
  are frozen until all 16 environments finish or reach the shared 5,000-frame cap.
  Screen 1 reached its first batch reset on telemetry row 314; only then did named frozen
  snapshots receive real rollout exposure. Its first 73 updates ran at 1,513.5 useful
  hero transitions per update-wall-second, while rows 74-491 ran at 859.7.
- `x0-autoreset-research.md` establishes that collision death and environment completion
  have different target semantics. Collision death is `done=True` and reward-only.
  A surviving snake on a population-floor or frame-cap boundary remains `done=False` and
  bootstraps from the final post-step observation and resolved action mask.
- The source implements those facts in `PQNTrainer._rollout`,
  `PQNTrainer._compute_targets`, `PQNTrainer._episode_over`, and
  `pqn_target_contract` in `src/training/pqn_trainer.py`. `BatchSim.reset()` in
  `src/simd_env/batch_sim.py` is currently all-environment only. Corrected-v3 owns one
  batch-wide `OpponentLease` from `src/training/pqn_selfplay.py` and releases it at the
  shared reset boundary.

These are two separate follow-ups. Packet A calibrates the accepted source without code
changes. Packet B is an opt-in semantic experiment. A timing result cannot authorize B,
and neither packet changes the current X0 result or promotes a model.

## Wave map and ownership

| Wave | Packet | Sole writer boundary | Depends on | Exit gate |
| --- | --- | --- | --- | --- |
| 0 | X0 closure | Root-owned frozen X0 manifests and receipts only | Current X0 processes | No live learner; all five outcomes and hashes accounted for |
| 1A | A1 calibration protocol/runner | New `x0-timing-calibration-v1/` artifact tree only | Wave 0 | Independent manifest review; no repository diff |
| 2A | A2 serial calibration execution | New arm subdirectories under that tree only | A1 | Both arms satisfy lifecycle evidence or are retained as inconclusive |
| 3A | A3 analysis/review | New `x0-timing-calibration-v1/review/` only | A2 | Recomputed phase totals and cap formula agree with raw telemetry |
| 1B | B0 reset contract | New `docs/experiments/pqn_per_env_autoreset_v1/contract.md` only | Wave 0 | Architecture review accepts the versioned semantics below |
| 2B | B1 simulator API | `src/simd_env/batch_sim.py`, `tests/test_simd_reset_contract.py` | B0 | Masked-reset and continuing-lane invariance tests pass |
| 2B | B2 assignment/action RNG API | `src/training/pqn_selfplay.py`, new `tests/test_pqn_selfplay_per_env.py` | B0 | Deterministic per-environment assignment/action oracles pass |
| 3B | B3 trainer/config integration | `src/training/pqn_trainer.py`, `src/scripts/train_pqn.py`, `src/core/config_loader.py`, `src/core/game_config.py`, `tests/test_config_pqn_block.py`, `tests/test_pqn_episode_contract.py`, `tests/test_pqn_contract_regressions.py`, `tests/test_pqn_resume_state.py`, `tests/test_train_pqn.py`, and new `tests/test_pqn_per_env_autoreset.py` | B1, B2 | Target, lease, seed, CLI, checkpoint, and compatibility gates pass |
| 4B | B4 independent verification | New `tests/test_pqn_per_env_autoreset_integration.py` and new `packages/autoreset-v1/` evidence only | Integrated B1-B3 | Focused union, slow parity, failure cleanup, and review accepted |
| 5B | B5 matched screen | New `x0-autoreset-matched-v1/` artifact tree only | B4 and a separately frozen execution protocol | Predeclared pairs complete or remain inconclusive; no seed replacement |

B1 and B2 are the only parallel write packets. B3 starts after both APIs are frozen. A1
and read-only B0 review may overlap after X0 closes. Source implementation may occur in an
isolated worktree while A runs against `7cb66a9`, but no test, build, evaluation, or other
compute may share the machine with an MPS calibration arm. One builder owns each listed
file; the root serially integrates reviewed commits.

## Packet A: representative timing calibration

### A1. Frozen protocol

Create `x0-timing-calibration-v1/` as a new, create-only artifact root. Its hashed
protocol and manifest must freeze:

- exact source `7cb66a9437f63b89cd5dc8731b5f8577ce1c9765` and the source-closure digest;
- the accepted X0 corrected-v3 world, observation, target, action-mask, optimizer, and
  **legacy sampler** settings: 16 environments, 6 snakes, rollout length 16, population
  floor, 5,000-frame cap, snapshot capacity 10, add interval 50, and hero fraction 0.8;
- two fresh training seeds derived through `SeedContext.stream_seed()` under new names
  such as `x0-followup/timing-v1/arm-1` and `arm-2`; neither may equal an X0 seed;
- one MPS learner at a time, two numerical CPU threads, 4-GiB RSS, 8-GiB MPS-driver,
  6-GiB available-system-memory reserve, and no concurrent tests or evaluation;
- the initial checkpoint hash, runner hash, complete resolved config and field sources,
  command, environment, device, and dependency hashes for each arm.

The runner is an external experiment driver, not a repository patch. It drives the public
`PQNTrainer.update()` loop and writes one atomic telemetry record per completed update.
It may stop only on the lifecycle rule below, a resource tripwire, an exception, or a hard
cap. Arm directories and terminal receipts are create-only; an existing path, source
drift, missing contract digest, or checkpoint mismatch fails before world construction.
No score is read for stopping. Partial logs and the last completed telemetry row remain
visible after failure.

### A2. Lifecycle-complete arms

Run the two arms serially. Each arm stops gracefully at the first completed update where
all three conditions are true:

1. `episode_reset_count >= 1`;
2. at least 64 completed updates have been recorded after the first reset; and
3. cumulative post-reset exposure to identities of the form
   `snapshot:<policy-id>:<content-sha256>` is at least 25,000 slot-transitions.

The hard stop is the first of 650,000 useful hero transitions or 15 minutes. The existing
memory and numerical tripwires also stop the arm. Missing any lifecycle condition before a
hard stop is an accepted **inconclusive calibration outcome**, not permission to extend the
arm or substitute a seed. A graceful lifecycle stop writes a final checkpoint and terminal
receipt; a forced stop records that neither is complete if that is the observed fact.

The analysis partitions raw update rows without moving a boundary:

- **early:** start through the row immediately before the first nonzero
  `completed_episodes` value;
- **sparse tail:** that first completion row through the row immediately before
  `episode_reset_count` first increments;
- **post-reset frozen:** rows at or after that increment, with exposure split by each exact
  snapshot identity rather than inferred from pool size.

For each phase and whole arm, report summed useful hero transitions, valid transitions,
rollout capacity, optimizer steps, sampled draws, unique sampled rows, update wall seconds,
useful transitions per wall minute, valid/capacity, hero/capacity, reset row/frame, exact
policy exposure, episode completions, peak RSS, peak MPS allocation, and minimum available
system memory. `pool_size` alone is not exposure evidence.

### A3. Baseline and decision rule

Recompute the same phase metrics from every frozen X0 arm with enough telemetry. This is a
descriptive source-matched baseline because the calibration seeds are new; do not present
it as a paired model-quality comparison. The first-episode early phase must be reported
separately so it cannot dominate the projected rate.

Only if both calibration arms meet every lifecycle condition may they set a future 500k
resource cap. For arm `i`, compute

`projected_i = process_elapsed_seconds_i / useful_hero_steps_i * 500000`.

Use `ceil(1.25 * max(projected_1, projected_2))`, bounded above by the plan's 30-minute
absolute ceiling. Use monitored process wall time, including startup and checkpoint work; report summed
update time separately. This is a conservative empirical budget, not a guarantee for every
future seed or policy trajectory. If the formula exceeds 30 minutes, either arm is
incomplete, hashes differ, or phase totals do not recompute, Packet A is inconclusive
and supplies no replacement cap. It never changes the
current X0 cap retroactively.

## Packet B: opt-in per-environment autoreset experiment

### B0. Contract and decisive design choice

Add two closed, checkpointed settings while retaining existing defaults:

- `episode_reset_mode`: `batch_barrier_v1` (default) or
  `per_env_autoreset_v1` (experimental);
- `episode_seed_mode`: `continuous_env_rng_v1` (default and legacy behavior) or
  `derived_env_episode_v1` (required by the matched experiment).

Do not add another algorithm recipe. `recipe=corrected-v3`, `obs_spec=raster31v3`, reward
v2, and mechanics v2 remain fixed. The selected reset and seed modes enter the runtime,
target, sampler, checkpoint, resolved-config, and manifest descriptors and their digests.
Version the new descriptor schema. An explicit compatibility adapter may identify the
known legacy reset/seed behavior from the old schema without rewriting its saved descriptor
bytes or digest. New metadata records the effective mode and adapter version. A continuation
request with conflicting semantics fails before weights, optimizer, world, or RNG state is
applied; unknown legacy schemas do not acquire fabricated provenance.

`per_env_autoreset_v1` uses an **explicit masked reset at the next rollout boundary**:

1. During a rollout, the transition that reaches collision death, population floor, or
   frame cap is stored exactly as today. A completed environment is frozen for the rest of
   that rollout, for at most `rollout_len - 1` steps.
2. Target computation finishes before any completed lane is reset. Actual snake death keeps
   `done=True` and `G_t = reward_t`. A surviving snake at floor/cap keeps `done=False` and
   bootstraps from the frozen final post-step observation and resolved mask. Invalid rows do
   not carry Q(lambda), so no return can enter the next episode.
3. At the next `_rollout()` entry, close leases only for completed environments, increment
   only their episode IDs, reset only their simulator lanes, derive their new seed streams,
   assign their next episode policies, and acquire their new leases. Continuing lanes retain
   byte-identical world/RNG state, episode IDs, policies, and pins.

This boundary is preferred over same-step reset because the current rollout already keeps
the final state available for its masked bootstrap. Same-step reset would require a second
final-observation channel and creates a larger target-risk surface. Waiting to the next
rollout sacrifices at most 15 lane steps under the proposed configuration while removing
the thousands of idle steps caused by the batch barrier.

Population floor is not relabeled as death. A true-terminal floor objective is outside this
experiment and would require a new reward and target contract.

### B1. Masked simulator reset

Refactor `BatchSim.reset()` around a private single-environment reset helper and add a
validated masked entry point, for example `reset_envs(env_mask, seeds=None)`. Its contract:

- mask shape is exactly `(E,)`; supplied seed count matches selected lanes and every seed is
  an unsigned 64-bit integer;
- only selected slices and episode-scoped containers are cleared: bodies/ring pointers,
  length/alive/direction, boost/starvation/respawn/reward baselines, food/corpse sets, frame,
  per-step rewards/dones/validity/kill/death outputs, traversal state, and action masks;
- a supplied seed replaces only that lane's `EnvRng`. It uses soft-reset draw order and does
  not replay `FoodManager`'s constructor-only discarded food draws;
- the existing no-argument `reset()` stays byte-for-byte compatible, including its one-time
  constructor draw behavior.

Required B1 tests compare all continuing-lane arrays, ordered food lists, sets, object-array
kill metadata, masks, frames, and serialized RNG state before and after another lane resets.
They cover one lane, a noncontiguous mask, all lanes, invalid masks/seeds, simultaneous floor
and cap, and a seeded reset replay. Existing SIMD reset and slow live-parity tests remain
green; any continuing-lane byte or RNG drift blocks B3.

### B2. Assignment, action RNG, and lease seam

Use stable C0 derivation directly, never Python `hash()`:

- world: `derive_seed(run_seed, f"pqn/world/env/{e}/episode/{k}")`;
- policy assignment: `derive_seed(run_seed, f"pqn/assignment/env/{e}/episode/{k}")`;
- hero exploration/action: `derive_seed(run_seed, f"pqn/action/env/{e}/episode/{k}")`.

Episode 0 is constructed with its derived world seed; later episodes replace only their own
world RNG immediately before masked soft reset. Policy assignment for `(e, k)` uses only its
assignment stream. Extend `batched_act()` with precomputed per-slot exploration decisions and
valid-action draws, generated from each environment's action stream, while preserving the
current grouped network forwards. The legacy single-RNG call remains the default path.

B2 tests prove that resetting or changing the length of environment 0 does not alter policy
IDs or exploration draws for continuing environment 1, and that the same `(run,e,k)` tuple
replays exactly. Test all-hero, mixed, and no-resident-snapshot cases. `OpponentLease.close()`
remains idempotent; no global pool API needs weaker eviction rules.

### B3. Trainer, telemetry, checkpoint, and cleanup

Replace the single corrected-v3 lease with one lease per environment only in the experimental
mode. `_episode_policy_ids` remains an `(E,S)` array, while episode IDs, reset counts, episode
seeds, and lease ownership are per environment. Acting may read from the shared pinned pool;
the leases guarantee that every referenced identity remains resident. On a completed lane,
capture its assignment/identity in telemetry, close its old lease, reset and reassign it at
the next rollout boundary. A `PQNTrainer.close()` path and `train_pqn.py` `finally` block close
all outstanding leases after success, exception, or resource stop. Pin underflow, a missing
assigned snapshot, or a nonzero pin count after close is a hard failure.

Version telemetry rather than silently reinterpreting the existing documentation mismatch.
The new schema records `episode_reset_mode`, `episode_seed_mode`, per-environment episode IDs
and reset counts, exact reset mask for the update, newly completed environment count, seed
identities, assignment identities, per-identity exposure, useful transitions, capacity, and
optimizer sampling counts. The existing default mode keeps its serialized fields and values.

Checkpoint behavior stays honest:

- inference loading remains independent of reset mode;
- old checkpoints remain valid for inference and explicit weights-only initialization;
- weights-only initialization starts fresh episode state and records the parent checkpoint
  SHA-256 plus the new run/seed/reset lineage;
- exact PQN resume remains unsupported;
- optimizer continuation for `per_env_autoreset_v1` fails closed until simulator state,
  per-environment RNGs, episode assignments, pool contents, leases, and MPS RNG can all be
  restored equivalently. The bounded matched screen therefore does not resume an arm.

This rejection is safer than serializing only counters while silently creating fresh worlds.
A future exact-resume packet would need interrupted-versus-uninterrupted equivalence tests and
is not part of autoreset v1.

### B4. Acceptance tests and compatibility gate

The independent verifier must demonstrate all of the following on the integrated source:

1. One environment reaches population floor while peers continue, is reset at the next
   rollout boundary, and contributes no fabricated reset transition.
2. Several environments completing together, all environments completing, and a lane reaching
   the frame cap reset exactly the selected lanes.
3. Collision death on the floor-causing step emits `done=True` and its reward-only target once.
   A surviving hero on the same floor/cap boundary emits `done=False` and its target uses the
   final pre-reset observation and resolved action mask.
4. Q(lambda) carry never crosses an episode ID. The first transition after reset cannot affect
   any target from the prior episode.
5. Reset observations and masks are valid, the population floor clears, and episode progress
   restarts within its declared normalization.
6. Policy identity is stable for the whole `(environment, episode)`; only reset lanes refresh.
   Their old pins release exactly once, continuing pins remain, admission cannot evict a pinned
   identity, and success/failure cleanup leaves zero pins.
7. The derived seed hierarchy reproduces the same `(environment, episode)` world, assignment,
   and actions. A reset in one lane leaves another lane's state and RNG byte-identical.
8. Default `batch_barrier_v1 + continuous_env_rng_v1` matches `7cb66a9` golden trajectories,
   targets, existing telemetry fields, and CLI resolution. Separately validate the new descriptor
   fields and the explicit legacy compatibility adapter; do not require new descriptor bytes to
   equal the old schema or rewrite old digests. Invalid or conflicting mode values fail before
   world construction. Cross-mode continuation fails closed.
9. Focused tests, `git diff --check`, relevant formatting/lint, the affected test union, and the
   existing slow SIMD/live parity cases pass under the runbook's serialized resource lanes.

No benchmark starts on a red gate. A correctness failure reopens its owning packet; it is not
worked around by loosening an oracle or changing the default.

### B5. Matched causal screen

Freeze a new protocol after B4. Use three fresh paired root seeds. For each root seed, run the
barrier arm and autoreset arm serially from the same initial checkpoint bytes, with identical
corrected-v3 settings except `episode_reset_mode`. Both arms use
`derived_env_episode_v1`, `sgd_epochs=1`, no flip augmentation, and the same immutable initial
checkpoint snapshot pre-admitted to a one-entry opponent pool. No learned snapshots are added
during the screen. This gives corresponding `(environment, episode)` worlds and assignments
the same seeds, exercises real frozen-policy leases, and prevents update-index pool admission
from becoming a second causal difference.

Stop each arm at the first completed update at or above 200,000 useful hero transitions.
Record overshoot, sampled draws, unique rows, optimizer steps, epsilon odometer, active fraction,
valid hero/capacity, completed episodes, episode-length distribution, reset counts, exact frozen
exposure, update and process wall time, peak memory, and numerical tripwires. The hard limit is
12 minutes or the existing resource caps. An incomplete pair remains visible and is not replaced.
Run order alternates by pair to reduce thermal/order bias. No arms run concurrently.

The historical X0 barrier results are reported separately because they used continuous RNG,
legacy sampling, and a dynamic pool. The causal comparison is only the new paired barrier arm
against its autoreset mate.

Autoreset v1 advances beyond a bounded experimental branch only if:

- all three pairs complete without semantic, numerical, lineage, lease, or resource failure;
- autoreset improves useful hero transitions per process-wall-minute by at least 15% in every
  pair, with the paired median and all raw values reported; and
- its optimizer draws per useful hero transition match the barrier arm within 1%, with any
  update-count difference disclosed.

Otherwise stop autoreset work and retain the result. Passing this threshold establishes a
worthwhile utilization improvement only. A separately frozen held-out policy evaluation would
still be required before making a learning-quality claim, and REL remains the only promotion
authority.

## Rollout, rollback, and conditional branches

`batch_barrier_v1 + continuous_env_rng_v1` remains the product and training default throughout.
Autoreset runs use a distinct config, contract digest, artifact namespace, and checkpoint
lineage. Rollback is disabling the experimental mode or reverting its isolated commits; old
checkpoints and the accepted X0 source remain readable. Never merge autoreset telemetry with X0
as one treatment and never copy an experimental checkpoint into a champion path.

PPO/X2 work remains conditional on the existing L0/L1/G1 dependencies and a separately frozen
algorithm-comparison protocol. H1 state-information probes remain conditional on evidence that
state information, rather than reset utilization, is the question worth resolving. Packet A or
B may improve cost estimates for those branches, but neither automatically starts them.

## Remaining risks

- Rollout-boundary autoreset still wastes up to 15 lane steps and may yield less improvement
  than same-step reset; the smaller target-risk surface is the decisive tradeoff for v1.
- Independent resets change the episode/start-state mixture even at equal valid-transition
  budgets. Throughput evidence cannot by itself establish sample efficiency or policy quality.
- A one-snapshot matched screen does not qualify dynamic learned-pool admission under asynchronous
  episode lifetimes. The lease oracles qualify safety; a later learning study must separately
  freeze any dynamic admission schedule to the useful-transition odometer.
- Exact environment continuation is deliberately unavailable. A wall or process stop leaves an
  incomplete arm rather than a falsely equivalent resume.
