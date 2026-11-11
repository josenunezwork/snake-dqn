# September 2026 RL Contract-Repair Plan & `X0` PQN Follow-Up (`rl-repair-pqn`)

**Dates:** 2026-09-06 to 2026-09-12
**Verdict:** Integrated `Waves 0–5` (`G0-R3` qualified `7cb66a9` across `2,519` non-slow + `3` slow parity tests), while Wave 6 `X0` (`5` arms `wall_stop` at `435.61 s`) ended `EXECUTED_INCONCLUSIVE`, follow-up `B5 v2` ended `INCONCLUSIVE_STOP`, and Waves `7–10` closed `DEFERRED_CONDITIONAL`.

## What was tried
- Decomposed the post-audit RL repair into `24` file-disjoint agent work packages (`Waves 0–10`: `C0`..`REL`) governed by `dispatch.json` and `validate_plan.py`.
- Implemented and integrated `Waves 0–5` (`C0` -> `G0` on `codex/rl-contract-repair`, `7cb66a9`): `raster31v3` (`26` scalar slots), `ActionMaskSet(legal, advisory)`, reward-only death vs bootstrapped truncation, `reward-v2`, and `promotion-v2-watch-rect`.
- Ran Wave 6 `X0`: `5` `corrected-v3` MPS PQN arms (`16` envs × `6` snakes, rollout `16`, `500,000` requested hero transitions/arm, frozen `435.61 s` wall cap).
- Executed follow-up Packet A (`x0-timing-calibration-v1` full-lifecycle timing) and Packet B (`B0`–`B4` opt-in `per_env_autoreset_v1` + `B5 v2` `3`-pair matched throughput screen).

## What happened
- **`Waves 0–5` (`G0-R3 QUALIFIED`):** Passed `2,519` non-slow tests (`6` skips), `3` slow parity tests, and a `100,901`-step MPS shakedown (`strict_authority: false`).
- **Wave 6 `X0` (`EXECUTED_INCONCLUSIVE`):** All `5` arms hit the `435.61 s` wall cap at `404,516`–`455,227` hero steps (`2,132,589` lower-bound total; finite losses, `1.452 GiB` RSS, `1.113 GiB` MPS), yielding zero final checkpoints or evaluations as screen `1` useful throughput fell from **`1,513.5` (updates `1–73`) to `859.7` (updates `74–491`)** hero transitions/s.
- **Packets A & B (`822 s` timing; `B5 v2` `INCONCLUSIVE_STOP`):** Packet A calibrated an **`822 s`** `500,000`-step wall cap; Packet B (`per_env_autoreset_v1`) stopped after pair `1` when the second child's exit code went unconfirmed (`4` arms unrun; `batch_barrier_v1` remained default).

## Why
Under `batch_barrier_v1`, finished environments sit frozen until all `16` lanes die or reach `5,000` frames (first reset at row `314`), cutting useful hero throughput (`1,513.5 -> 859.7`/s) after the early all-alive phase that `G0`'s short shakedown had extrapolated from.

## What to reuse / what not to repeat
- Reuse `ActionMaskSet(legal, advisory)` (`legal & advisory` fallback to `legal`), death-vs-truncation bootstrap rules, `promotion-v2-watch-rect`, and `validate_plan.py`-checked file-disjoint package specs.
- Never project multi-episode rollout wall caps from a short pre-reset shakedown when whole-batch barriers (`batch_barrier_v1`) leave dead lanes idle.

## Full record
Original: `git show 90b3db94:docs/research/history/rl-repair-pqn.md` | Primary sources: [rl_repair_2026-09-06/README.md](../../plans/rl_repair_2026-09-06/README.md), [rl_repair_2026-09-06/agent_runbook.md](../../plans/rl_repair_2026-09-06/agent_runbook.md), [rl_repair_2026-09-08-followup.md](../../plans/rl_repair_2026-09-08-followup.md)
