# Round 32 End-to-End Rehearsal and Round 33 Operations (`r32-r33-ops`)

## Question and Goal

Can the end-to-end Resume-by-Default execution lifecycle (`lifecycle.py`), Mac-verified unit ledger, outcome-blind mechanical retry gate (`ORACLE.json`, `matrix.py`), synthetic F4 residual-to-checkpoint consumer (`consumer.py`), and paired-world statistical aggregator (`statistics.py`) be rehearsed and verified offline and via a short Colab canary (`research/r32_rehearsal/`) before launching the Round 33 (`research/r33_ops/`, `research/r33_verification/`) operational packages?

## Timeline of Versions and Steps

| Date | Step / Package | What Was Run | Verdict |
|---|---|---|---|
| 2026-10-24 | **`R32-1` / `R32-2` initial rehearsal implementation** (`d0d92bdb`, [`research/r32_rehearsal/README.md`](../../../research/r32_rehearsal/README.md), [`STATISTICS.md`](../../../research/r32_rehearsal/STATISTICS.md)) | Built `consumer.py` (`consumer-02` synthetic F4 residual-to-checkpoint trace), `ORACLE.json` (10 R9 terminal labels), `matrix.py` (3/3 positive paths + 12/12 fault injections), and `statistics.py` (28-cell power/type-I simulation). Verified `24/24` tests in `25.35 s`. | **LOCAL PASS** (`RETAIN_FIXED_DESIGN`; real ego2s strict adapter still `BLOCKED`) |
| 2026-10-24 | **Canary `r32-cpu-canary-7b114eaa-01` & directory fix** (`7b114eaa` → `c1ab2750`, [`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-32/implementation/pack-c1ab2750-01`](/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-32/implementation/pack-c1ab2750-01)) | First Colab canary (`7b114eaa`) failed after `136.26 s` (`0.003028` units, `0` outputs) with `FileExistsError: attempt output directory must be fresh: output` because `research/colab_cli/run.py` pre-creates an empty `output/` directory. Fixed `lifecycle.start_attempt` in `c1ab2750` to accept an existing empty directory while still refusing non-empty directories (`25/25` tests passed). | **FIXED (`c1ab2750`)** |
| 2026-10-24 | **Statistical `heavy_tail_df=5` fix** (`c1ab2750` → `4f11670a`, [`HANDOFF_STATISTICS_FIX_20261024.md`](../../../research/r32_rehearsal/HANDOFF_STATISTICS_FIX_20261024.md), [`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-32/implementation/pack-4f11670a-01`](/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-32/implementation/pack-4f11670a-01)) | Repaired `statistics.run_frozen_unit` so heavy-tailed Student-t (`df = 5`) innovations scale by `sd / sqrt(df / (df - 2))` instead of normal innovations, regenerated [`STATISTICS.md`](../../../research/r32_rehearsal/STATISTICS.md) (`30/30` tests passed in `25.81 s`), and prepared canary manifest `r32-cpu-canary-4f11670a-01`. | **VERIFIED (`4f11670a`)** |
| 2026-10-24 | **`R33` operations & verification packages** ([`research/r33_ops/README.md`](../../../research/r33_ops/README.md), [`HANDOFF.md`](../../../research/r33_ops/HANDOFF.md), [`A1_OWNER_HANDOFF.md`](../../../research/r33_ops/A1_OWNER_HANDOFF.md), [`research/r33_verification/README.md`](../../../research/r33_verification/README.md)) | Packaged Round 33 operational manifests, ownership boundaries (`OWNERSHIP.md`), A1 owner handoff, and verification harness. | **COMPLETED** |

## Final Verdict and Key Numbers

1. **`R32-1` Synthetic Consumer Trace and Strict-Evaluator Boundary (`consumer-02`):**
   - `research.r32_rehearsal.consumer` verified the synthetic `F4` residual-to-checkpoint pipeline (`32` synthetic rows, `4` steps, deterministic SHA-256 checkpoint + manifest + strict-shape report + registry entry) with status `SYNTHETIC_TRACE_PASS` ([`research/r32_rehearsal/HANDOFF.md`](../../../research/r32_rehearsal/HANDOFF.md), [`research/r32_rehearsal/DELIVERY_DRAFT.md`](../../../research/r32_rehearsal/DELIVERY_DRAFT.md)).
   - Full T34 strict-gate / `saved_snakes/registry.json` validation for `F4` remained explicitly **`BLOCKED`** (`overall_r32_1_status: BLOCKED`) because `src/scripts/tournament_eval.py` rejects `ego2s-bv8` checkpoints on its legacy live/simd paths without the T34 `GridBatchSim` adapter ([`research/r32_rehearsal/DELIVERY_DRAFT.md`](../../../research/r32_rehearsal/DELIVERY_DRAFT.md)).
2. **`R32-2` Resume-by-Default Lifecycle & Fault-Injection Matrix (`3/3` positive, `12/12` negative):**
   - Verified all `10` terminal labels in `research/r32_rehearsal/ORACLE.json` (`VERIFIED_COMPLETE`, `VERIFIED_PARTIAL_ELIGIBLE_RETRY`, `MECHANICAL_NO_OUTPUT_ELIGIBLE_RETRY`, `MECHANICAL_RETRY_EXHAUSTED`, `REFUSED_FOREIGN_OR_CORRUPT_LEDGER`, `REFUSED_DUPLICATE_OR_TAMPERED_UNIT`, `REFUSED_ACTIVE_OR_UNSTOPPED_LEASE`, `REFUSED_OUTCOME_PEEK_OR_CAP_EDIT`, `SCIENTIFIC_OR_DETERMINISTIC_ABORT_NO_RETRY`, `INVALID_CONTRACT`), `3/3` positive resume paths (`pos-1-clean-single-attempt`, `pos-2-partial-resume-2-of-4-then-4-of-4`, `pos-3-trajectory-snapshot-resume-step2-to-step4`), and `12/12` fault injections (`neg-01` .. `neg-12`) ([`research/r32_rehearsal/HANDOFF.md`](../../../research/r32_rehearsal/HANDOFF.md)).
3. **Colab Supervisor Empty-Output-Directory Fix (`c1ab2750`):**
   - Attempt 1 (`r32-cpu-canary-7b114eaa-01`, session `cli-r32-cpu-canary-7b114eaa-01-20261024t184145z-97df5be4`) stopped cleanly after `136.26 s` (`0.003028` units, `0` outputs) because `research/colab_cli/run.py` creates `<output_dir>` (`mkdir(parents=False, exist_ok=False)`) before invoking the remote script, whereas `lifecycle.start_attempt` called `mkdir(exist_ok=False)` on `output/`.
   - Commit `c1ab2750` updated `lifecycle.start_attempt` (`exist_ok=True` + `next(attempt_output_dir.iterdir(), None) is not None` check) so pre-created empty supervisor directories succeed while non-empty directories are still refused ([`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-32/implementation/pack-c1ab2750-01/CODEX_V_LIGHT.json`](/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-32/implementation/pack-c1ab2750-01/CODEX_V_LIGHT.json)).
4. **`R32-4` Paired-World Statistical Operating Characteristics (`4f11670a`, `RETAIN_FIXED_DESIGN`):**
   - After fixing Student-t (`df = 5`) innovation scaling in `4f11670a`, [`research/r32_rehearsal/STATISTICS.md`](../../../research/r32_rehearsal/STATISTICS.md) verified across `28` simulation cells (`4,000` trials/cell) that the 1-look fixed design at `N = 160` has null false-positive rate **`0.0398` (normal)** and **`0.0415` (heavy-tail `df = 5`)** (both `<= 0.055`), and at `N = 320, SD = 220, Δ = +30` achieves **`0.7855` (normal)** and **`0.7850` (heavy-tail `df = 5`)** power (`30/30` tests passed in `25.81 s`; [`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-32/implementation/pack-4f11670a-01/LOCAL_VERIFICATION.json`](/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-32/implementation/pack-4f11670a-01/LOCAL_VERIFICATION.json)).

## What Was Learned

- **Colab CLI supervisor pre-creates the top-level `output/` directory on the Mac before remote execution:** Any package lifecycle guard that insists on a fresh attempt directory must verify that `output/` is *empty* (`not any(output_dir.iterdir())`), not that `output/` does not exist on disk.
- **Heavy-tailed (`df = 5`) Monte Carlo power simulations must normalize by `sqrt(df / (df - 2))`:** Unscaled `t_5` draws inflate variance by `5/3`; once normalized to the target `SD`, paired-world `t` tests at `N = 160` and `N = 320` maintain nominal type-I error (`~0.040–0.049`) and matching power across normal and `df = 5` heavy tails ([`STATISTICS.md`](../../../research/r32_rehearsal/STATISTICS.md)).

## Why It Closed / Current Status

**Completed operational rehearsal and handoff.** The Round 32 rehearsal and Round 33 operational packages completed their verification milestones; the redundant `CODEX_V.md`, `COVERAGE.md`, `HANDOFF_DIRECTORY_FIX_20261024.md`, and empty `NEXT_ACTIONS.md` template in `research/r32_rehearsal/` were consolidated into this record.

## Keep Files in This Lineage

- [`research/r32_rehearsal/README.md`](../../../research/r32_rehearsal/README.md) — Overview of the `r32_rehearsal` package (`R32-1`, `R32-2`, `R32-4`).
- [`research/r32_rehearsal/HANDOFF.md`](../../../research/r32_rehearsal/HANDOFF.md) — Implementation handoff for `r32_rehearsal` (`d0d92bdb`).
- [`research/r32_rehearsal/HANDOFF_STATISTICS_FIX_20261024.md`](../../../research/r32_rehearsal/HANDOFF_STATISTICS_FIX_20261024.md) — Handoff for both the `c1ab2750` empty-directory fix and the `4f11670a` `heavy_tail_df=5` statistical fix.
- [`research/r32_rehearsal/STATISTICS.md`](../../../research/r32_rehearsal/STATISTICS.md) — 28-cell operating-characteristic simulation table (`RETAIN_FIXED_DESIGN`).
- [`research/r32_rehearsal/DELIVERY_DRAFT.md`](../../../research/r32_rehearsal/DELIVERY_DRAFT.md) — Complete Round 32 delivery summary (`R32-1` .. `R32-4`).
- [`research/r33_ops/README.md`](../../../research/r33_ops/README.md) — Overview of the `r33_ops` operational package.
- [`research/r33_ops/OWNERSHIP.md`](../../../research/r33_ops/OWNERSHIP.md) — Module ownership boundaries for Round 33.
- [`research/r33_ops/HANDOFF.md`](../../../research/r33_ops/HANDOFF.md) — Round 33 implementation handoff.
- [`research/r33_ops/A1_OWNER_HANDOFF.md`](../../../research/r33_ops/A1_OWNER_HANDOFF.md) — Round 33 A1 owner handoff.
- [`research/r33_verification/README.md`](../../../research/r33_verification/README.md) — Overview of the `r33_verification` package.
