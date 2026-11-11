# Round 32 End-to-End Rehearsal and Round 33 Operations (`r32-r33-ops`)

**Dates:** 2026-10-23 to 2026-10-24
**Verdict:** `R32` synthetic consumer trace, ten-label R9 oracle, and 15-path lifecycle matrix passed (`PASS` for synthetic subcontracts, `RETAIN_FIXED_DESIGN` for statistics, while strict validation, registry admission, and release readiness stayed `BLOCKED` and main held for a same-pin canary); `R33` operations verified (`64` passed) and canceled its CPU pilot on zero demand.

## What was tried
- Built `research/r32_rehearsal`: `consumer-02` exported-byte trace (factory payload → label ledger → three-update fit → fused exports → fresh-process serving selectors), `ORACLE.json` (hand-calculated cases for all ten ordered R9 labels plus boundary ties), and `matrix.py` (`3` positive resume/stop paths and `12` fault injections).
- Built `STATISTICS.md` / `statistics.py` (`R32-5`): prospective comparison of the equal `128`-world/mix design against cost-aware fixed allocation and a single futility look at `64`/mix (`28` units, shared `288 min / 0.384 u` ceiling).
- Delivered `research/r33_ops` and `research/r33_verification`: `tools/ops/task_journal.py` (`100` deterministic SQLite crash/restart schedules), `package_spec.py`, `proof_dependencies.py`, and a `15`-node 3-shard Linux contract suite.

## What happened
- **`R32` synthetic subcontracts (`PASS` synthetic / `BLOCKED` consumer readiness):** Combined light verification passed (`24` passed, `76` deselected in `25.35 s`); `consumer-02`, the ten-label `ORACLE.json`, and all `3 + 12` lifecycle paths passed while production `F4` admission stayed `BLOCKED` pending an `ego2s`/T34 strict-request and registry adapter.
- **`R32` canary fix (`c1ab2750`):** First Colab CPU canary failed `INCOMPLETE` (`136.26 s` / `0.003028 u`) on `FileExistsError` because the supervisor pre-creates the empty `output` directory.
- **`R32` statistics fix (`RETAIN_FIXED_DESIGN`, main `HOLD`):** Passing `heavy_tail_df` (`5`) to `Generator.standard_t` (`4f11670a`) passed `28/28` local units (`9.10 s`), holding Colab main for a same-pin `4f11670a` canary (`16.84 min / 0.02245 u` prior series debit).
- **`R33` operations (`Implementation verified; CPU pilot canceled`):** Passed `64` tests (`1.06 s`), `15` Mac preflight checks (`3` shards, `3.58 s`), `15` hosted Linux CI nodes (`40057728587`), and `100` crash/restart schedules; handoff-time SLA targets stayed `INSUFFICIENT_LT10` and `A1-PROF` stayed `OWNER_HANDOFF_UNACKNOWLEDGED / NATIVE_PENDING`.

## Why
The Colab CLI supervisor pre-creates the attempt `output` directory before launching the remote worker (breaking guards that demand a non-existent path), `R32-5` kept `RETAIN_FIXED_DESIGN` because published data could not calibrate joint variance or costs, and `c1ab2750` canary proof could not admit `4f11670a` under the same-pin rule.

## What to reuse / what not to repeat
- Reuse `tools/ops/task_journal.py` and the `R32` fault-injection matrix (`12` sidecar, truncation, namespace, and snapshot faults) to rehearse consumer lifecycles offline before real data exist.
- Make output-directory guards accept an existing empty directory while still refusing files, symlinks, or non-empty trees, and never treat a canary from an earlier commit pin as valid admission for a modified commit.

## Full record
Original: `git show 90b3db94:docs/research/history/r32-r33-ops.md` | Primary sources: [r32_rehearsal/README.md](../../../research/r32_rehearsal/README.md), [STATISTICS.md](../../../research/r32_rehearsal/STATISTICS.md), [r33_ops/HANDOFF.md](../../../research/r33_ops/HANDOFF.md)
