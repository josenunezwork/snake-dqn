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

P1 (effective PQN config, seeding and explicit resume modes) and S1 (inference
identity, masks and serving contracts) are closing independent review findings
from `60747fb5`. Review found a corrected-v3 self-resume mismatch and live
serving normalization drift despite passing focused tests. The repair requires
a real producer-to-resume and producer-to-serving round trip. A detached serving
receipt binds the exact checkpoint bytes to the named deployment profile;
training checkpoints do not embed a self-referential artifact hash.

OFF remains in independent review. Its live pixel-world descriptor preserves
actual scaled geometry rather than rounding it to fit the separate aligned
SIMD contract. Further review exposed the replay quality codec accepting only
58 features although the incumbent uses 61. Both widths require explicit
validation and tests before the package is accepted.

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
exact rational Holm boundary cases. These are E2 implementation oracles, not
evidence that the new statistical gate has been implemented.

The campaign ledger records exact package/integration/test SHAs and durable
evidence manifests. No repaired-model training or model-quality qualification
has run yet in this execution campaign.
