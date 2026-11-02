# September 2026 RL Contract-Repair Plan & `X0` PQN Follow-Up (`rl-repair-pqn`)

## Question & Goal

Following the 2026-09-06 state & system audit (assessment commit `6986fd3`, product source `c1cf7e8`), how can the `raster31` / `BatchSim` / `PQN` stack be repaired through bounded, file-disjoint agent work packages (`Waves 0–10`) and evaluated in a bounded 5-arm `X0` corrected-PQN development screen?

---

## Timeline of Versions & Steps

| Date | Wave / Package | What Was Specified / Run | Verdict |
|---|---|---|---|
| 2026-09-06 | `docs/plans/rl_repair_2026-09-06/` (`Waves 0–10`, 24 packages) | Decomposed the RL contract repair into 24 bounded agent packages (`C0`, `N0`, `A0`, `A1`, `S0`, `E0`, `ENV`, `OBS`, `OFF`, `P1`, `S1`, `P2`, `A2`, `E1`, `E2`, `H0`, `G0`, `X0`, `L0`, `L1`, `G1`, `X2`, `H1`, `REL`) with an authoritative `dispatch.json` (IDs, dependencies, exact file ownership) and a `validate_plan.py` dependency/ownership/path/card checker. Based on assessment commit `6986fd3` (product source `c1cf7e8`, clean main `950d583`). | **PLANNED** (plan snapshot: "planned, not started"). |
| 2026-09-08 | `Waves 0–5` (`C0` -> `G0`) on integration branch `codex/rl-contract-repair` | Implemented and accepted the repair packages (e.g. combined Apex/PQN/serving union `1,241` tests at `A2`); `G0-R3` qualified clean source `7cb66a9` for a bounded development screen (`2,519` non-slow passes, `6` skips, `3` slow deselected; `3` slow parity passes; one MPS shakedown of `100,901` hero steps) (`docs/experiments/rl_repair_execution_2026-09-08.md`, removed in consolidation; see git c4c4ab41). | **G0-R3 QUALIFIED** for development screening (`strict_authority: false`, `release_action_authorized: false`). |
| 2026-09-08 | `Wave 6` (`X0` 5-arm `corrected-v3` PQN screen at `7cb66a9`) | Ran five `corrected-v3` arms (`500,000` requested hero transitions each, one MPS learner, frozen per-arm wall cap `435.61 s`; 16 envs × 6 snakes, rollout length 16) (`rl-repair-20260908/x0-final-analysis.json` as cited in the execution report; that artifact directory is not present in snake-dqn-artifacts). | **`EXECUTED_INCONCLUSIVE`** — all five arms retained as `wall_stop` (`404,516`–`455,227` last recorded hero steps; no final checkpoint or evaluation); no model-quality estimate. Under `batch_barrier_v1`, screen 1 ran `1,513.5` useful hero transitions per update-wall-second over its first 73 updates and `859.7` afterwards. |
| 2026-09-08 to 2026-09-12 | `rl_repair_2026-09-08-followup.md` (Packets `A` & `B`) | Packet A (`x0-timing-calibration-v1`): two lifecycle-complete calibration arms on `7cb66a9`. Packet B (`per_env_autoreset_v1`): opt-in per-environment autoreset (`B0`–`B4` accepted) and a frozen three-pair matched throughput screen `B5` (`/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909/`; `docs/experiments/pqn_followup_2026-09-12.md`, removed in consolidation; see git c4c4ab41). | **A: timing evidence only** (future 500k cap `822 s`); **B5 v2: `INCONCLUSIVE_STOP`** — stopped after pair 1 because the second child process could not be confirmed exited; no accepted pair or advance decision. |

---

## Final Verdict & Key Numbers

### 1. 24-Package Contract-Repair Architecture (`docs/plans/rl_repair_2026-09-06/README.md`)
- **Initial implementation decisions:** a corrected `raster31v3` descriptor/digest keeping the raster topology, tensor shapes and all 26 scalar slots; `ActionMaskSet(legal, advisory)` resolving per row to `legal & advisory` when nonempty, otherwise legal; actual death uses reward only while truncation bootstraps; batch-wide reset kept initially (asynchronous reset deferred); reward-v2 only; a named `promotion-v2-watch-rect` deployment profile ([`docs/plans/rl_repair_2026-09-06/README.md`](../../plans/rl_repair_2026-09-06/README.md), [`packages/c0.md`](../../plans/rl_repair_2026-09-06/packages/c0.md)).
- **Wave structure:** Waves 0–4 implement the repair, G0 (Wave 5) qualifies it, and Waves 6–10 (`X0`, `L0`/`L1` PPO, `H1`, `G1`, `X2`, `REL`) are conditional experiment/algorithm work.

### 2. Wave 6 `X0` Corrected PQN Screen (`docs/experiments/rl_repair_execution_2026-09-08.md`, removed in consolidation; see git c4c4ab41)
- **All 5 `X0` arms hit the frozen `435.61 s` wall cap** (observed detection `435.77–436.51 s`); the `2,132,589` aggregate hero steps is a lower bound from final telemetry rows. With zero complete arms, checkpoint pairs or evaluations, the five-seed interval is null. All checked loss/Q fields were finite; peak process RSS `1.452 GiB`, peak MPS-driver allocation `1.113 GiB`.
- **Batch-barrier utilization:** completed environments are frozen until all 16 environments finish or reach the shared 5,000-frame cap; screen 1 reached its first batch reset on telemetry row 314. Its first 73 updates ran at `1,513.5` useful hero transitions per update-wall-second, rows 74–491 at `859.7` ([`docs/plans/rl_repair_2026-09-08-followup.md`](../../plans/rl_repair_2026-09-08-followup.md)).
- **Packets A & B:** Packet A's predeclared process-wall formula gave a future 500,000-useful-hero-transition cap of `822 s` (timing evidence only). Packet B's `B5 v2` ended `INCONCLUSIVE_STOP` with four arms unrun and no retry; runtime defaults remain `batch_barrier_v1` with `per_env_autoreset_v1` opt-in.

---

## What Was Learned & Why It Closed

1. **File-disjoint, machine-checked wave specs made a 24-package repair integrable:** packages were accepted and serially integrated by the root with recorded test evidence, establishing the legal-vs-advisory mask separation (`ActionMaskSet`) and death-vs-truncation bootstrap rules.
2. **Whole-batch reset barriers starve long-tail survival rollouts:** frozen dead environments cut screen 1's useful throughput from `1,513.5` to `859.7` hero transitions per update-wall-second after the early phase, and the short G0 timing projection did not represent those later phases — motivating Packet A's lifecycle-complete calibration and Packet B's opt-in per-environment autoreset.
3. **Status — CLOSED (September 2026):** The campaign ledger marks `X0` `EXECUTED_INCONCLUSIVE` and `H1`, `L0`, `L1`, `G1`, `X2`, `REL` `DEFERRED_CONDITIONAL`; the follow-up B5 screen ended `INCONCLUSIVE_STOP`. No model was promoted and `main`'s incumbent was unchanged.

---

## Kept Files of the Lineage

- [`docs/plans/rl_repair_2026-09-06/README.md`](../../plans/rl_repair_2026-09-06/README.md)
- [`docs/plans/rl_repair_2026-09-06/agent_runbook.md`](../../plans/rl_repair_2026-09-06/agent_runbook.md)
- [`docs/plans/rl_repair_2026-09-06/packages/c0.md`](../../plans/rl_repair_2026-09-06/packages/c0.md)
- [`docs/plans/rl_repair_2026-09-06/packages/n0.md`](../../plans/rl_repair_2026-09-06/packages/n0.md)
- [`docs/plans/rl_repair_2026-09-06/packages/a0.md`](../../plans/rl_repair_2026-09-06/packages/a0.md)
- [`docs/plans/rl_repair_2026-09-06/packages/a1.md`](../../plans/rl_repair_2026-09-06/packages/a1.md)
- [`docs/plans/rl_repair_2026-09-06/packages/a2.md`](../../plans/rl_repair_2026-09-06/packages/a2.md)
- [`docs/plans/rl_repair_2026-09-06/packages/s0.md`](../../plans/rl_repair_2026-09-06/packages/s0.md)
- [`docs/plans/rl_repair_2026-09-06/packages/s1.md`](../../plans/rl_repair_2026-09-06/packages/s1.md)
- [`docs/plans/rl_repair_2026-09-06/packages/e0.md`](../../plans/rl_repair_2026-09-06/packages/e0.md)
- [`docs/plans/rl_repair_2026-09-06/packages/e1.md`](../../plans/rl_repair_2026-09-06/packages/e1.md)
- [`docs/plans/rl_repair_2026-09-06/packages/e2.md`](../../plans/rl_repair_2026-09-06/packages/e2.md)
- [`docs/plans/rl_repair_2026-09-06/packages/env.md`](../../plans/rl_repair_2026-09-06/packages/env.md)
- [`docs/plans/rl_repair_2026-09-06/packages/obs.md`](../../plans/rl_repair_2026-09-06/packages/obs.md)
- [`docs/plans/rl_repair_2026-09-06/packages/off.md`](../../plans/rl_repair_2026-09-06/packages/off.md)
- [`docs/plans/rl_repair_2026-09-06/packages/p1.md`](../../plans/rl_repair_2026-09-06/packages/p1.md)
- [`docs/plans/rl_repair_2026-09-06/packages/p2.md`](../../plans/rl_repair_2026-09-06/packages/p2.md)
- [`docs/plans/rl_repair_2026-09-06/packages/h0.md`](../../plans/rl_repair_2026-09-06/packages/h0.md)
- [`docs/plans/rl_repair_2026-09-06/packages/h1.md`](../../plans/rl_repair_2026-09-06/packages/h1.md)
- [`docs/plans/rl_repair_2026-09-06/packages/g0.md`](../../plans/rl_repair_2026-09-06/packages/g0.md)
- [`docs/plans/rl_repair_2026-09-06/packages/g1.md`](../../plans/rl_repair_2026-09-06/packages/g1.md)
- [`docs/plans/rl_repair_2026-09-06/packages/x0.md`](../../plans/rl_repair_2026-09-06/packages/x0.md)
- [`docs/plans/rl_repair_2026-09-06/packages/l0.md`](../../plans/rl_repair_2026-09-06/packages/l0.md)
- [`docs/plans/rl_repair_2026-09-06/packages/l1.md`](../../plans/rl_repair_2026-09-06/packages/l1.md)
- [`docs/plans/rl_repair_2026-09-06/packages/x2.md`](../../plans/rl_repair_2026-09-06/packages/x2.md)
- [`docs/plans/rl_repair_2026-09-06/packages/rel.md`](../../plans/rl_repair_2026-09-06/packages/rel.md)
- [`docs/plans/rl_repair_2026-09-08-followup.md`](../../plans/rl_repair_2026-09-08-followup.md)
