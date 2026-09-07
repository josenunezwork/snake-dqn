# OFF — Require offline replay provenance and consistent episode endings

**Status:** PLANNED. **Earliest wave:** 1. **Role:** data-engineer. **Kind:** implementation.

**Dependencies:** C0

Prevent silent mixing of unverifiable replay and honor the online episode boundary.

Read [decisions and wave plan](../README.md), [agent runbook](../agent_runbook.md), and [dispatch.json](../dispatch.json) before starting. You are not alone: edit only this package's claimed files and inherit accepted dependency commits.

## Exclusive write ownership

- `src/scripts/generate_experiences.py`
- `src/scripts/offline_train.py`
- `src/data/replay_quality.py`
- `src/data/memory_db_handler.py`
- `tests/test_generate_experiences.py`
- `tests/test_offline_train.py`
- `tests/test_replay_quality_pure.py`
- `tests/test_memory_db_handler.py`
- `tests/test_audit_replay.py`
- `src/data/replay_contract.py` — new
- `src/scripts/migrate_replay_metadata.py` — new
- `tests/test_replay_contract.py` — new
- `tests/test_replay_metadata_migration.py` — new

## Implementation / execution steps

1. Use C0 contracts/seeding; record observation/action/mask semantics, mechanics/arena/world, gamma/n-step/reward, effective seed and generator identity. Empty/incomplete metadata fails normal append/load.
2. Provide an explicit unverified-legacy path with missing-field and fallback-mask counts permanently attached to output provenance. Unknown gamma/n-step cannot be reconstructed from row shape.
3. Both single and parallel generators seed all RNGs and stop at mechanics-v2 population floor. Do not continue lone-survivor replay after that boundary.
4. Add dry-run and copy-first metadata migration using SQLite backup API (or a verified offline exclusive checkpoint including WAL). Fsync and atomically publish the destination copy; never mutate the only source. Preserve experience rows. Supplied known facts cannot turn unknown target semantics into verified facts. Do not migrate user databases in this package.

## Acceptance evidence

- Append/load reject mismatched mechanics, arena, target horizon and mask schema. Legacy opt-in is visible in resulting metadata.
- Controlled single/worker generation stops at the floor and repeats a short same-seed trace.
- Temporary SQLite fixtures include committed WAL data and an active reader. Backup retains every committed row; dry-run leaves source files unchanged; migration failure leaves source and prior valid destination intact. Unknown semantics remain unverified.

Focused target files below include new files this package creates. Commands and flags described in the steps are implementation requirements, not claims that they already exist.

```bash
SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q \
  tests/test_replay_contract.py \
  tests/test_replay_metadata_migration.py \
  tests/test_generate_experiences.py \
  tests/test_offline_train.py \
  tests/test_replay_quality_pure.py \
  tests/test_memory_db_handler.py
```

## Compatibility and rollback

Existing databases are retained. Evaluation of old checkpoints remains supported; legacy replay use is explicit and never silently upgraded.

## Handoff

Write command/result logs, source/config/checkpoint hashes, changed files, commit SHA, unresolved risks and interface changes to `<EVIDENCE_ROOT>/packages/off/`. The root resolves EVIDENCE_ROOT to an absolute durable path outside disposable worktrees before dispatch. Obtain a root-issued test/compute slot before running commands. Return absolute artifact paths and hashes; root verifies the durable manifest before accepting the dependency or cleaning a worktree. A passing narrow test does not waive an upstream gate or promote a model.

Audit source: [expanded state/system review](../../../state_and_system_review_2026-09-06.md).
