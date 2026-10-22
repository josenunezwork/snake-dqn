# M1 mechanical repair and pre-cell stop reconciliation

Outcome: untimed smoke PASS; no timed cells run. The frozen M1 bars, fixtures,
allocation, cumulative 14,400-second cap and single completion attempt remain
unchanged. This sibling supersedes only the startup/recovery instructions in
`M1_HANDOFF.md`; that reviewed file and prior receipts remain immutable.

## Root cause and API audit

Frozen `3708063b9c8e8513997121a4c028cf867365c605` accessed
`snake.auto_respawn` directly in `new_watch`. Ordinary `AISnake` instances do
not define it. Real Watch uses `GameSession.step()` →
`GameState.update(train_mode=False, learn=False, allow_respawn=True)`, whose
per-snake opt-out is `getattr(snake, "auto_respawn", True)`. M1 now uses that
same default and checks Watch mode, playing state, six snakes and speed 24.

The untimed `--smoke` invokes the same real T15 mixed `GameSession` constructor
as M1: B8 s3 against five FRP3 s12 + v8. T15's existing constructor scope only
overrides roster count and suppresses release telemetry; game, policies,
step/snapshot and multiprocessing are real. Path B remains disabled. Seven
real step/snapshot calls exercise frame progression and `last_error`; forced
hero/opponent death with timer 1 exercises actual respawn and selection-cache
invalidation. One frozen-800 fixture is submitted to a real spawn W1/K1 pool.
The worker exercises `l1.DeviceOperator`'s real `_world`, policy module/device
attributes, `_roots` cache, geometry/root hashing, restore, exact-stop/native
operator, guarded rollout methods, IPC decoding, counters and thread reports.
No additional invalid attribute assumption was found in this M1 worker path.
No game, worker pool or fixture execution is mocked by the smoke entrypoint.
The older L1 diagnostic's separate paced simulator loop is outside M1.

The smoke acquires no lease and writes no cell, latency distribution, bar or
timing receipt. It discards worker diagnostic clocks and prints only functional
evidence. It is an explicit independent launcher mode, never an implicit warmup
inside a timed cell. Run it in a separate process before timing.

## Stop and recovery verdict

Attempt-01 refused before an attempt directory/cell was created (foreign load).
Attempt-02 is the first charged attempt: INCOMPLETE, zero cells/exports,
mechanical failure, 0.8207511249929667 seconds charged. In frozen source the
failing `new_watch()` call precedes multiprocessing context/pool construction,
host-monitor start and contention construction. It owned none of those resources.
The coordinator PID 99091 is no longer present. The no-surviving-worker proof
rests on frozen code ordering, not an inference from a current process screen.

Historical `verified_stop=false` was conservative bookkeeping: `main` marked
the cell active before Watch construction. The receipt is preserved unchanged.
Future stop status tracks ownership/cleanup: true before resource creation,
false before pool construction, true after successful cooperative shutdown and
monitor cleanup. Partial setup is inside cleanup coverage. Cleanup exceptions
leave false and therefore refuse ordinary recovery.

The independently reviewed `M1_PRECELL_RECONCILIATION_20261022.json` admits
only this exact historical failure through `--reconciled-precell-stop`. Runtime
validation pins original receipt, attempt, log and frozen M1 source digests,
requires zero cells/exports and original attempt 1, binds the new namespace,
and retains its debit. The next timed receipt is attempt 2; that receipt cannot
authorize another completion. The operator dispatches this reconciliation once:
there is no durable registry preventing reuse of an original recovery proof.
The ordinary `--previous-attempt` path still requires the
same namespace and a verified stop. No completed cell crosses namespaces.

Remaining allowance: **14,399.179248875007 seconds**, including setup, failures,
startup and drain. The independent Codex-V review and reconciliation are
create-only exports beside this handoff. No original receipt is relabelled.

## Exact operator sequence

Use the published commit from the final operator message to create a new frozen
worktree; do not change `/Users/josenunez/Projects/ml/snake-dqn-m1run` at 3708063b.
The launcher supplies the primary venv and absolute checkpoint locations. Example
below uses the fixed implementation worktree after publication; it must remain
clean and unchanged for timing. A detached worktree at that same published SHA
has the same source namespace.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-m1
EXP=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp
FIXTURES="$EXP/bl-l2/ruler-fixtures-08-final/ruler-fixtures.json"
REFERENCE="$EXP/bl-l2/y1-native-mac/ruler-receipt.json"
PROOF="$PWD/research/bl_k1/M1_PRECELL_RECONCILIATION_20261022.json"
LEDGER="$EXP/bl-m1-ledger-03.json"

research/bl_k1/run_m1.sh --smoke --fixtures "$FIXTURES" --reference "$REFERENCE"
# Empty, create-only ledger; use a fresh numbered name if it already exists.
research/bl_k1/run_m1.sh --fixtures "$FIXTURES" --reference "$REFERENCE" \
  --make-ledger "$LEDGER"
research/bl_k1/run_m1.sh --check --fixtures "$FIXTURES" --reference "$REFERENCE" \
  --ledger "$LEDGER" --reconciled-precell-stop "$PROOF"
# Claude only, once Mac is idle and other lease holders are finished:
research/bl_k1/run_m1.sh --run --fixtures "$FIXTURES" --reference "$REFERENCE" \
  --ledger "$LEDGER" --reconciled-precell-stop "$PROOF" \
  --out "$EXP/bl-m1-attempt-03"
```

The implementer ran smoke, tests, style and recovery `--check` only. Claude runs
the final timed command. No serving, checkpoint, pin, tmux or original frozen
worktree changes were made.
