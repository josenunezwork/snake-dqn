# ENV — Repair BatchSim reset and inactive-world semantics

**Status:** PLANNED. **Earliest wave:** 1. **Role:** builder. **Kind:** implementation.

**Dependencies:** C0

Match repeated live resets and stop finished environments from affecting future episodes.

Read [decisions and wave plan](../README.md), [agent runbook](../agent_runbook.md), and [dispatch.json](../dispatch.json) before starting. You are not alone: edit only this package's claimed files and inherit accepted dependency commits.

## Exclusive write ownership

- `src/simd_env/batch_sim.py`
- `src/simd_env/parity.py`
- `src/simd_env/__init__.py`
- `tests/test_simd_env.py`
- `tests/test_simd_parity.py`
- `tests/test_simd_reset_contract.py` — new

## Implementation / execution steps

1. Consume discarded constructor food RNG draws once; later reset() must follow live reset ordering. Retain the current batch-wide reset schedule.
2. Expose separate legal/advisory masks and the C0 resolved mask. Keep get_action_mask() as the legacy advisory API. Alive normal actions are domain-legal; boost requires the configured length threshold; dead rows are false.
3. Add active_env_mask to step(). Inactive environment state, frame, food, timers, masks and RNG are unchanged. Return no new transition/event for inactive rows and make validity explicit.
4. Expose existing per-step food/boost/death/kill event facts needed by E1 without changing event ordering or reward arithmetic. Do not introduce asynchronous resets or new gameplay rules.

## Acceptance evidence

- Seeds 41 and 314159 match across at least three resets and subsequent actions against the live reference.
- Inactive-world byte/RNG snapshots remain identical while another environment advances; post-floor food cannot affect the next initialization.
- Legal/advisory subsets and boost thresholds pass controlled/random fixtures. V1/v2 ordered mechanics fixtures remain unchanged; run the affected slow parity battery once at integration.

Focused target files below include new files this package creates. Commands and flags described in the steps are implementation requirements, not claims that they already exist.

```bash
SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q \
  tests/test_simd_reset_contract.py \
  tests/test_simd_env.py \
  tests/test_simd_parity.py
```

## Compatibility and rollback

Keep the legacy default step call functional. Reset semantics get a new contract ID; old checkpoints remain inference-loadable, not falsely exact-resumable.

## Handoff

Write command/result logs, source/config/checkpoint hashes, changed files, commit SHA, unresolved risks and interface changes to `<EVIDENCE_ROOT>/packages/env/`. The root resolves EVIDENCE_ROOT to an absolute durable path outside disposable worktrees before dispatch. Obtain a root-issued test/compute slot before running commands. Return absolute artifact paths and hashes; root verifies the durable manifest before accepting the dependency or cleaning a worktree. A passing narrow test does not waive an upstream gate or promote a model.

Audit source: [expanded state/system review](../../../state_and_system_review_2026-09-06.md).
