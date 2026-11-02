# September 2026 RL Contract-Repair Plan & `X0` PQN Follow-Up (`rl-repair-pqn`)

## Question & Goal

Following the 2026-09-06 state & system audit (`c1cf7e8` / `6986fd3`) that halted naive PQN training after uncovering observation-contract bugs, terminal-reward masking defects, action-mask mismatches, and an uncalibrated promotion gate, how can the `raster31` / `BatchSim` / `PQN` stack be repaired through machine-validated, non-overlapping agent work packages (`Waves 0–10`) and evaluated in a clean 5-seed `X0` learning screen?

---

## Timeline of Versions & Steps

| Date | Wave / Package | What Was Specified / Run | Verdict |
|---|---|---|---|
| 2026-09-06 to 2026-09-07 | `docs/plans/rl_repair_2026-09-06/` (`Waves 0–10`, 24 packages) | Decomposed the RL contract repair into 24 bounded agent packages (`C0`, `N0`, `A0`, `A1`, `S0`, `E0`, `ENV`, `OBS`, `OFF`, `P1`, `S1`, `P2`, `A2`, `E1`, `E2`, `H0`, `G0`, `X0`, `L0`, `L1`, `G1`, `X2`, `H1`, `REL`) with machine-readable YAML headers, file-scope isolation, and `validate_plan.py` verification (`validation.json`). | **`validate_plan.py` PASS** (`2026-09-07`, `0` file overlaps, `0` DAG cycles, all 24 package specs complete). |
| 2026-09-07 | `Waves 0–5` (`C0` -> `G0`, commits `e7fbaf8a`..`b1c6d27a`) | Implemented `raster31v3` schema & finite-value checks (`C0`, `OBS`), `ActionMaskSet(legal, advisory)` per-row resolution (`A0`–`A2`), `BatchSim` lifecycle/kill-credit parity (`N0`, `ENV`), `reward_v2` terminal-death vs truncation-bootstrap semantics (`P1`, `P2`), and `promotion-v2-watch-rect` evaluation (`E0`–`E2`, `H0`, `G0`). | **Waves `0`–`5` merged (`b1c6d27a`)** — all unit/contract gates passed; ready for Wave 6 (`X0`). |
| 2026-09-08 | `Wave 6` (`X0` 5-seed corrected `PQN` screen at `7cb66a94`) | Ran 5 corrected `PQN` seeds (`500,000` hero transitions target, `16` envs $\times$ `6` snakes, `rollout_len=24`, `30 min` wall cap) plus post-hoc root-cause closure (`/Users/josenunez/Projects/ml/snake-dqn-artifacts/rl-repair-20260908/x0-root-closure.json`, `x0-final-analysis.json`). | **`INCOMPLETE` (`wall_stop` on all 5 arms)** — `batch_barrier_v1` starved completed envs until all 16 envs finished (`1,513.5 -> 859.7` hero transitions/s after row 73); positive early learning signal (`+39.67` survival frames, `p = 0.0085`). |
| 2026-09-08 to 2026-09-12 | `rl_repair_2026-09-08-followup.md` (Packets `A` & `B`, `pqn_followup_2026-09-12.md`) | Specified Packet A (`x0-timing-calibration-v1`: wall-clock vs `alive_hero_Fraction` instrumentation) and Packet B (`per_env_autoreset_v1`: independent per-env reset at rollout boundaries `B0`–`B5` while zeroing GAE bootstrap across reset boundaries). | **Executed (`/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909/`)** — diagnosed that PQN bootstrap and solo-food learning dynamics remained sample-inefficient, motivating the `solo_food_*` diagnostic series. |

---

## Final Verdict & Key Numbers

### 1. 24-Package Contract-Repair Architecture (`docs/plans/rl_repair_2026-09-06/README.md`)
- **Wave 0 (`C0`):**Froze `raster31v3` (`tactical (9,31,31)`, `strategic (3,25,25)`, `scalars (26,)`), `ActionMaskSet(legal, advisory)`, `reward_v2`, and `promotion-v2-watch-rect` contract identifiers in `src/contracts/rl_repair.py` ([`docs/plans/rl_repair_2026-09-06/packages/c0.md`](../../plans/rl_repair_2026-09-06/packages/c0.md)).
- **Waves 1–3 (`N0`, `A0`–`A2`, `S0`, `E0`, `ENV`, `OBS`, `OFF`, `P1`, `S1`):**
  - Replaced hero-only mask reuse with per-row legal (`180°` reversal + boost eligibility) and advisory safety mask resolution across all snake rows ([`a0.md`](../../plans/rl_repair_2026-09-06/packages/a0.md), [`a1.md`](../../plans/rl_repair_2026-09-06/packages/a1.md), [`a2.md`](../../plans/rl_repair_2026-09-06/packages/a2.md)).
  - Repaired `BatchSim` body-segment lifecycle, food-respawn parity, and head-to-body kill attribution (`N0`, `ENV`), and enforced strict `done_death` (bootstrap `= 0`) vs `done_trunc` (bootstrap `= \gamma V(s_{t+1})`) in `PQNTrainer` (`P1`, `P2`) ([`env.md`](../../plans/rl_repair_2026-09-06/packages/env.md), [`p1.md`](../../plans/rl_repair_2026-09-06/packages/p1.md), [`p2.md`](../../plans/rl_repair_2026-09-06/packages/p2.md)).
- **Waves 4–5 (`P2`, `A2`, `E1`, `E2`, `H0`, `G0`):** Built the paired `promotion-v2-watch-rect` tournament gate (`145x83` arena, `6` snakes, paired world seeds, alive-frame and whole-horizon mass metrics) and baseline opponent suite (`scripted:greedy_food`, `scripted:random_safe`, frozen checkpoints) ([`e1.md`](../../plans/rl_repair_2026-09-06/packages/e1.md), [`e2.md`](../../plans/rl_repair_2026-09-06/packages/e2.md), [`g0.md`](../../plans/rl_repair_2026-09-06/packages/g0.md)).

### 2. Wave 6 `X0` Corrected PQN Screen & Root-Cause Closure (`docs/plans/rl_repair_2026-09-08-followup.md`)
- **All 5 `X0` Arms Hit `wall_stop` (`30 min`) Due to `batch_barrier_v1` Starvation:**
  - Under `batch_barrier_v1`, any environment whose hero snake died remained frozen (`done=True`) until all `16` environments terminated or reached the `5,000`-frame horizon (`/Users/josenunez/Projects/ml/snake-dqn-artifacts/rl-repair-20260908/x0-root-closure.json`).
  - In Screen 1, the first full batch reset did not occur until update `314`; from log row `1` to row `73`, useful throughput averaged **`1,513.5` hero transitions/s** (`118.9 s` elapsed for `180,000` transitions), whereas from row `73` to `wall_stop` (`1,681.5 s`), useful throughput fell **`1.76x` to `859.7` hero transitions/s** ([`docs/plans/rl_repair_2026-09-08-followup.md`](../../plans/rl_repair_2026-09-08-followup.md)).
- **Early Learning Signal Before Starvation (`x0-final-analysis.json`):** Across the completed portion of `X0`, corrected `PQN` showed a statistically significant survival improvement of **`+39.67` frames (`p = 0.0085`)** and `+0.47` food/episode (`/Users/josenunez/Projects/ml/snake-dqn-artifacts/rl-repair-20260908/x0-final-analysis.json`, [`docs/plans/rl_repair_2026-09-08-followup.md`](../../plans/rl_repair_2026-09-08-followup.md)).
- **Packets A & B (`per_env_autoreset_v1`):** Packet A calibrated per-update timing (`dt_rollout_ms`, `dt_learn_ms`, `alive_hero_fraction`), and Packet B replaced `batch_barrier_v1` with `per_env_autoreset_v1` at rollout boundaries while preserving `done_trunc` bootstrap semantics (`B0`–`B5` in [`docs/plans/rl_repair_2026-09-08-followup.md`](../../plans/rl_repair_2026-09-08-followup.md), executed in `/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909/`).

---

## What Was Learned & Why It Closed

1. **Multi-agent RL contract repairs benefit from file-disjoint wave specs:** Decomposing the 2026-09-06 audit fixes into 24 machine-validated packages (`Waves 0–5`) eliminated cross-agent merge conflicts and permanently established the repo's legal vs advisory mask separation (`ActionMaskSet`) and terminal-death vs horizon-truncation bootstrap rules.
2. **Synchronous `Q(\lambda)` rollouts cannot use whole-batch reset barriers on long-tail survival tasks:** Freezing dead environments until the longest-surviving environment hits `5,000` frames wastes nearly half of all rollout steps once any snake learns to survive (`1,513.5 -> 859.7` hero transitions/s), requiring rollout-boundary `per_env_autoreset_v1`.
3. **Status — CLOSED (September 2026):** Waves `0`–`6` and the `X0` follow-up (`Packets A & B`) completed in September 2026; Waves `7`–`10` (`L0`/`L1` PPO comparison, `G1`, `X2`, `H1`, `REL`) were superseded when the `solo_food_*` and `task_aligned_*` series moved the project to the `GridBatchSim` + `ego2s-bv8` distilled-DQN main line (`B8`).

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
