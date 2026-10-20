# T34 ExIt-B8 fit preparation

Implements P4, P5 and the fit-only P6 entrypoint from
[`2026-10-19_exit_fit_prep.md`](../../docs/research/specs/2026-10-19_exit_fit_prep.md),
following the binding
[`FINAL_PLAN.md`](../../docs/research/final_run_20261019/FINAL_PLAN.md).

This package prepares S1b (B8 architecture, primary) and S1a (full-arena tower).
The owner runs the full T3 sweep. Preparation does not authorize training rounds,
Colab allocation, promotion, or release. No TD loss is used.

## Boundaries and dependencies

- The sibling `research/t34_exit_b8/` package owns teacher exports and replay
  regeneration. This fit package owns its adapter boundary and local fake. The
  fake establishes interface and training-loop behavior; it does not establish
  real teacher replay identity or the production survival probes.
- Production observations must be regenerated from all six action traces each
  epoch. Verify each world's recorded hashes before using its rows, drop and
  report divergent worlds, and stop if a chunk's drop fraction exceeds 2%.
- Selection uses held-out decisive agreement and a preregistered 32-world
  survival probe. Behavior tripwires use full H5000 trajectories across all
  length deciles. Gate worlds must never enter fit or selection data.
- The architecture and live adapter are research code. The released B8 default,
  serving pins, checkpoints, and rollback configuration are not altered.
- Public input URLs, hashes, the production regenerator and probe binding are
  required before a production fit manifest can be admitted.

## Local operating limits

All preparation checks run on the Mac with at most two math threads, using the
existing primary checkout's Python environment. Run compute checks sequentially.
Do not start or alter tmux, Phase 0, S1, Colab sessions, pods, or paid services.
The future supervised L4 runtime must explicitly use CUDA.

Exact operator commands and measured acceptance results are recorded below once
the integrated preparation checks finish.

## Acceptance status

| Check | Current status |
| --- | --- |
| P4 harness and A4 preservation | Implementation in progress |
| T3 3x3 tau/margin sweep tool and tiny smoke | Implementation in progress; full sweep reserved for owner |
| P5 tower and A3 10,000-state zero-init identity | Implementation in progress |
| Independent live/sim full-sight bytes | Implementation in progress |
| A6 fit entrypoint, two worlds / 200 updates | Implementation in progress |
| Production replay and survival-probe integration | Depends on sibling engine and preregistered probe |
| Public-release input publication | Owner dependency; URLs unavailable at start |
| Live L4 execution, cross-runtime A2, gate and release | Outside this preparation task |
