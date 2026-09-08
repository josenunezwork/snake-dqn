# RL repair execution — 2026-09-08

Status: implementation in progress. The accepted plan is preserved at
`docs/plans/rl_repair_2026-09-06/`; its PLANNED fields describe the delivered
plan snapshot, not execution status.

- Plan base: `5d5e85d6da1be3af48845e233d826c9cdcec77ed`.
- Integration branch: `codex/rl-contract-repair`.
- Main at start: `950d583b9eb0c3e316edd0f651954d133334ac1a`, clean.
- Durable campaign ledger:
  `/Users/josenunez/Projects/ml/snake-dqn-artifacts/rl-repair-20260908/campaign.json`.
- Package worktrees:
  `/Users/josenunez/Projects/ml/snake-dqn-repair-worktrees/<package>/`.

The root integrates reviewed commits serially, records test evidence on the
integrated source, and starts only packages with accepted dependencies. Local
artifact paths below are accompanied by SHA-256 manifests in the campaign
folder. No package promotes a model or publishes main automatically.

## Resource baseline

Apple M5 Pro, 18 logical CPUs, 64 GiB unified memory; Torch 2.9.1 with MPS
available. The launch probe observed about 19.9 GiB available memory. This is a
point-in-time value. Keep at most two one-thread small CPU checks, or one
exclusive full-suite/evaluation/corpus/MPS job. MPS training uses two native
threads with the plan's 4 GiB RSS, 8 GiB driver and 6 GiB available-memory caps.

## Accepted packages

C0, A0, N0, S0 and E0 are accepted. The combined regression gate on
`572f089` passed **486 tests, 1 checkpoint-dependent skip** in 28.55s.
A subsequent import-lint comment correction leaves the tournament AST
unchanged; changed-file Black, isort and flake8 checks passed afterward.
An initial integration run caught the new `provided_fields` frozen set failing
JSON receipt serialization; `572f089` fixes the explicit receipt projection
and adds a regression. Failed and passing logs remain under `integration/`.

A1 and ENV are also accepted on integrated source `8b11f924`:

- The Apex union passed **323 tests in 12.12s**, covering the new coordinator
  lifecycle paths together with replay generation/epoch safety and the learner.
  The finalizer captures state while its owners are usable, attempts every
  cleanup step independently, and publishes the terminal checkpoint afterward.
  Wall-time enforcement is explicitly cooperative; resolved budgets and observed
  overshoot are recorded. It does not promise to preempt a hung learner step.
- The simulator union passed **52 tests in 143.92s**, including all three slow
  parity batteries (20 seeds × 10,000 frames each), repeated live resets, and
  inactive-world semantics. The exclusive run used about one CPU core; observed
  RSS was 345,440 KiB. This is a sampled resource observation, not peak memory.

OBS is accepted after an integrated **117 passed, 3 slow deselected** in 31.96s
on `cb3d03d1`. The slow cases had already passed in the ENV gate above. A later
import-only ordering fix gives the clean dispatch base `60747fb5`. The raster
tests compare a reconstructible v2 fixture from the pre-change producer and
independent v3 paint-order oracles, including complete enemy-slot permutations.
Legacy adapter defaults remain intact; v3 callers must supply semantic
normalization values explicitly. The Torch v3 surface currently delegates to
the canonical NumPy renderer; no CUDA throughput claim is made.

OFF and P1 are accepted and integrated through `64530856`. Their combined
regression run passed **668 tests in 14.83s**. OFF preserves full replay dataset
provenance across repeated copy-first migrations, validates both 58- and 61-D
rows, and labels the stored masks as advisory. Its complete live-world descriptor
preserves actual scaled pixel geometry without weakening the aligned SIMD contract.

P1 now freezes the actual world, reward, target, sampler and optimizer recipes.
Continuation rejects changed semantic descriptors even if their digests are
recomputed; it also validates actual Adam state. Weights-only loading requires
fresh optimizer, counters, pool and RNG state. Target and self-play descriptors
explicitly identify the remaining pre-P2 behavior rather than claiming it fixed.

S1 is accepted at source `0d88879`, integrated at `03336522`. Its final package
union passed **229 Python tests, 5 conditional skips**, with **32 frontend tests**
and a production build. The integrated P1/S1 union passed **271 tests, 2 skips**.
A checkpoint serialized on that same integrated source loaded in Watch and Play
with nondefault normalization values and passed the executable serving spot check.
These are initialization, serialization and serving checks; no model was trained.
The detached serving manifest describes manual reset, no live episode horizon,
the external evaluator's separate 5,000-frame horizon, and the live world's
unbounded storage adapter. Real GameState oracles cover pre-human-move observation
bytes, sparse IDs, mixed policies, reorder, deaths and respawns. Raw checkpoint
world, runtime and model-head descriptors must be complete before construction;
tests remove and re-sign each field to prove defaults cannot fabricate source truth.

The P1-WORLD amendment at `f85e4ea` serializes all 23 effective-world fields,
including inactive circle geometry for a rectangular world. Its focused gate
passed 96 tests. This is a provenance completion, with no numerical recipe change.

E1 preparation identified a further nontraining food-replenishment mismatch:
actual GameState maintains ambient food after each eater, while the previous
SIMD path spawned unconditionally. ENV-WATCH is accepted at `eee93701`: the
same actual-live oracle produced four expected failures on the preceding source,
then the integrated union passed 53 tests with 3 slow cases deselected. E1 will
inherit this amendment.

A frontend collection failure also exposed eight required source files present
on the main checkout disk but hidden from Git by a broad `lib/` ignore pattern.
Commit `0e441cdb` preserves their exact local bytes and adds a narrow exception.
The source manifest records their origin and hashes. With this prerequisite,
the S1 frontend passed 166 tests and its production build. Main was not edited.

Independent preparation produced a 70-digit mpmath Student-t reference fixture,
cross-checked with higher-precision density quadrature and analytic cases, plus
exact rational Holm boundary cases. The binary64 extension uses 80-digit exact-input references, with 422 independent
assertions. A bounded pure-Python numerical prototype passed 12 tests and independent
review against these fixtures, including direct quadrature and exact Holm boundaries.
These are E2 implementation oracles, not evidence that the inferential gate exists.

The campaign ledger records exact package/integration/test SHAs and durable
evidence manifests. No repaired-model training or model-quality qualification
has run yet in this execution campaign.

## Active integration wave

P2, A2 and E1 are in implementation and independent review. No consumer package
may treat a green helper test as acceptance of an unfinished runtime path.

- P2's immutable opponent pool passed source review and 39 focused tests. The
  trainer remains under repair for episode completion accounting, fixed-opponent
  exploration, truthful source descriptors and pristine continuation state.
- A2 is split into recipe/local policy, actor/replay/learner, entrypoints and
  persistent replay storage. An independent test owner exercises actual transport
  and load paths. Mask modes must survive every consumer, not merely serialization.
- E1 has profile/metric/anchor helpers and live/SIMD wiring. A separate verifier
  is adding controlled actual-runtime action and event oracles; source review and
  immutable input closure are still required.

The campaign's `cpu_test_slot.py` now enforces at most two concurrent one-thread
focused test processes. Owned-file repair/rerun attempts are continuously authorized
through that wrapper. Full-suite, evaluation, corpus and MPS work remains exclusive.
The recorded event stream has observed at most two concurrent test children.

Two ambiguous A2 card phrases are resolved explicitly in the execution ledger:
50/100 are local/distributed TD-target clipping values, while gradient clipping
must record its actual separate value. Historical checkpoints remain loadable for
inference or weights-only starts; restoring optimizer/counters requires verified
continuation, with no unverified optimizer-continuation escape hatch.
