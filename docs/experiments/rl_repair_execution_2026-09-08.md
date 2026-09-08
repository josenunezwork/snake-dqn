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

A1 remains isolated for process-budget/finalization review fixes. ENV, OBS and
OFF may proceed because their C0 dependency is accepted. The campaign ledger
records exact package/integration/test SHAs and durable evidence manifests.
