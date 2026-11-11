# Early Solo & Ambient-Food PQN Series (2026-09-13 to 2026-09-21)

**Dates:** 2026-09-13 to 2026-09-21
**Verdict:** Closed without promoting a cold-start PQN checkpoint (`INDEPENDENT_SOLO_AMBIENT_SIGNAL_NOT_ESTABLISHED`): replay attribution exposed `98.5466%` boost-trail self-eating, cold-start ambient PQN remained seed-fragile (`0`, `22.625`, `110.75` food at `200k`), `1`-opponent `CX` broke paired retention (`70` of `76` head-on deaths), and `E128/T4` (`1,284` steps/s) was selected on M5 Pro.

## What was tried
- Tested `50%` `scripted:greedy_food` exploration advice in `6`-snake PQN (`scripted_advice`, `NOT_ADVANCED`), then built opt-in `1`-snake `S1` (`sole_snake_death_or_frame_cap_v1`, `441` tests) to isolate solo foraging over `5,000` frames (`solo_food_survival`).
- Ran exact food-source attribution (`ambient` vs own boost trail, `solo_food_source_trace`), fresh `2`-seed ambient replication (`1701–02`), and a `1`-pellet relocation `Q`-response probe (`168` rows).
- Swept `ambient_food_reward_coefficient = 0.1` vs `0.0` at `50k` and `200k` steps (`3` seeds `1901–03`, `C2`), `24`-transition TD/clone probes, telemetry exposure (`4,703` rows), and `eps_end = 0.10` (`solo_exploration_floor`).
- Synthesized the solo catalog through `CW` (`H768`) and `CX`–`DA` (`1`-opponent) and benchmarked `7` M5 Pro `(envs, threads)` PQN layouts (`pqn_mac_capacity_2026-09-21`).

## What happened
- **Boost-trail artifact & replication failure:** Apparent `50k` solo food growth (`2.500 -> 1,100.875` on `1601`) was **`98.5466%` (`8,679 / 8,807`) own boost-trail re-eating** (`128` ambient). On fresh worlds, `1701` hit `83.625` ambient contacts while `1702` fell to `0.000` (`0/7` worlds switched greedy action when adjacent food moved).
- **`200k` `C2` curve & `eps_end = 0.10` (`NO ADVANCEMENT`):** `200k` ambient means split at **`0` (`1901`), `22.625` (`1902`), and `110.75` (`1903`)** (`18/24` checks), even though failed `1901` hit `2,756` training ambient contacts (`13.765/1k`) and `24/24` positive adjacent-food TD residuals (`7/12`–`11/12` decode `< 12/12`); `eps_end = 0.10` ended at `1.375, 5.000, 2.375`.
- **Warm-start `H768` vs `1`-opponent ceiling & Mac sizing:** BC-warm-started `CW` passed all `37` `H768` conditions in `3/3` seeds (`122.51, 116.85, 135.34` food; `353, 364, 383` of `384` alive), whereas `1`-opponent `CX` failed paired retention (`70` of `76` `CY` deaths were head collisions). M5 Pro `E128/T4` hit **`1,284` hero transitions/s** (`+38.0%` over `E16/T2`'s `930`/s, `0.913 GiB` RSS).

## Why
Counting unattributed food contacts rewarded boosting and eating one's own dropped trail (`98.5466%`); for cold-start PQN's seed-fragile ambient foraging (`0` on `1901` despite `13.765/1k` contacts and positive TD residuals), **cause was not established** among exploration, coverage, or forgetting.

## What to reuse / what not to repeat
- Always separate `ambient` food from own boost-trail pickups in solo diagnostics, and use `E128/T4` (`1,284` transitions/s) for M5 Pro `BatchSim` PQN runs.
- Do not train cold-start replay-free PQN from scratch expecting seed-uniform foraging (`0` vs `110.75` at `200k`), or assume solo `H768` policies avoid enemy heads (`70/76` head-on deaths).

## Full record
Original: `git show 90b3db94:docs/research/history/solo-ambient-early.md` | Primary sources: [SOLO_FOOD_SERIES.md](../../experiments/SOLO_FOOD_SERIES.md)
