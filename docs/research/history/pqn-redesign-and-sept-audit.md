# July–September 2026 PQN Redesign Blueprint, State-System Audit, and RL Portfolio Reviews

**Dates:** 2026-07-05 to 2026-09-12
**Verdict:** Engineering-qualified the repaired `raster31v3` / `BatchSim` / PQN contracts (`G0-R3`, opt-in `per_env_autoreset_v1`, opt-in `watch_pre_move_v1`) while keeping `vector61` Apex frozen after the sampler screen (`-1.3694`), `X0` (`wall_stop`), `B5 v2` (`INCONCLUSIVE_STOP`), and `H1-A` (`0.0` LCB) produced no learning promotion.

## What was tried
- Drafted the `raster31v2` PQN blueprint (`31×31` `9`-plane tactical + `25×25` `3`-channel strategic rasters + `26` ego scalars, `~1.3M` params, `BatchSim`) and audited git history showing the June `2026-06-21`..`06-28` CNN was never trained (`zero` checkpoints).
- Tested `epoch` (`100.0%` rows) vs `legacy` (`68.132%`) PQN minibatch sampling across `5` seeds × `500,000` hero steps on `12` `1,000`-frame SIMD worlds (`pqn_sampler_2026-09-06`).
- Audited `100` source/config files across `vector61` and `raster31v2` (`state_system_audit_2026-09-06`) and executed the RL contract repair (`C0`–`H0`, `G0-R3` on `7cb66a9`) plus the `5`-arm `X0` screen (`500,000` requested steps/arm, `435.61 s` cap).
- Built opt-in `per_env_autoreset_v1` (`B0–B5`, `822 s` calibrated `500,000`-step cap), tested single-segment remote-heading aliasing on `24` worlds (`H1-A`), and added opt-in `watch_pre_move_v1` (`BatchSim.step_with_policy()`) after `LP1` exposed a frame-`0 -> 1` pre-observation shift.

## What happened
- **State-system audit (`vector61` kept frozen):** Training `free_space_v2.yaml` on `0.2`-scale boards vs `1.0` full arenas shifted **`44/61` feature distributions (max Kolmogorov–Smirnov gap `0.943`)**, alongside boost/hunger aliases, mask/paint-order mismatches, and `train_pqn.py --no-self-play` co-adapting `6` live heroes.
- **Sampler (`NOT ESTABLISHED`), `X0` (`EXECUTED_INCONCLUSIVE`), & `B5 v2` (`INCONCLUSIVE_STOP`):** `epoch - legacy` averaged `-1.3694` mass (`95%` CI `[-4.748, +2.009]`). All `5` `X0` arms hit the `435.61 s` cap (`404,516`–`455,227` steps as rate fell `1,513.5 -> 859.7`/s); `B5 v2` pair `1` ran `-2.948%` (`54,177.10` vs `52,580.11`/min) before `NoSuchProcess` (`child_exit_unconfirmed`).
- **`H1-A` (`INCONCLUSIVE_NOT_ADVANCED`), `watch_pre_move_v1` (`ENGINEERING_ACCEPTED`), & `VC1`:** `H1-A` found `0.0` regret LCB across `36` pairs (`51/113` next-obs diffs); `watch_pre_move_v1` passed `2,684` tests; `VC1` hit its `28.425 s` resource cap on seed `2` (`0.5926` seed-`1` held accuracy).

## Why
Encoding `vector61` features in world-frame coordinates caused severe `0.2 -> 1.0` scale drift (`0.943` max KS across `44/61` dims), while short all-hero timing calibrations (`1,513.5` vs `859.7` steps/s under whole-batch barriers) under-sized `X0`'s `435.61 s` wall cap before any arm could finish.

## What to reuse / what not to repeat
- Keep spatial observations heading-aligned and scale-invariant, and calibrate training wall caps across full late-episode survival phases (`822 s` vs `435.61 s`).
- Never cite "the June 2026 CNN lost" (it had zero checkpoints), and use `5` scripted foes + hero slot `0` rather than `--no-self-play` (`6` co-adapting heroes) for fixed-opponent PQN controls.

## Full record
Original: `git show 90b3db94:docs/research/history/pqn-redesign-and-sept-audit.md` | Primary sources: [ml_redesign_blueprint_2026-07.md](../../ml_redesign_blueprint_2026-07.md), [state_and_system_review_2026-09-06.md](../../state_and_system_review_2026-09-06.md), [pqn_decision_phase_2026-09-12.md](../../pqn_decision_phase_2026-09-12.md)
