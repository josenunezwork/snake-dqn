# RL Runs

<a id="pqn-redesign-and-sept-audit"></a>
## July–September 2026 PQN Redesign Blueprint, State-System Audit, and RL Portfolio Reviews

**Dates:** 2026-07-05 to 2026-09-12
**Verdict:** Engineering-qualified the repaired `raster31v3` / `BatchSim` / PQN contracts (`G0-R3`, opt-in `per_env_autoreset_v1`, opt-in `watch_pre_move_v1`) while keeping `vector61` Apex frozen after the sampler screen (`-1.3694`), `X0` (`wall_stop`), `B5 v2` (`INCONCLUSIVE_STOP`), and `H1-A` (`0.0` LCB) produced no learning promotion.

### What was tried
- Drafted the `raster31v2` PQN blueprint (`31×31` `9`-plane tactical + `25×25` `3`-channel strategic rasters + `26` ego scalars, `~1.3M` params, `BatchSim`) and audited git history showing the June `2026-06-21`..`06-28` CNN was never trained (`zero` checkpoints).
- Tested `epoch` (`100.0%` rows) vs `legacy` (`68.132%`) PQN minibatch sampling across `5` seeds × `500,000` hero steps on `12` `1,000`-frame SIMD worlds (`pqn_sampler_2026-09-06`).
- Audited `100` source/config files across `vector61` and `raster31v2` (`state_system_audit_2026-09-06`) and executed the RL contract repair (`C0`–`H0`, `G0-R3` on `7cb66a9`) plus the `5`-arm `X0` screen (`500,000` requested steps/arm, `435.61 s` cap).
- Built opt-in `per_env_autoreset_v1` (`B0–B5`, `822 s` calibrated `500,000`-step cap), tested single-segment remote-heading aliasing on `24` worlds (`H1-A`), and added opt-in `watch_pre_move_v1` (`BatchSim.step_with_policy()`) after `LP1` exposed a frame-`0 -> 1` pre-observation shift.

### What happened
- **State-system audit (`vector61` kept frozen):** Training `free_space_v2.yaml` on `0.2`-scale boards vs `1.0` full arenas shifted **`44/61` feature distributions (max Kolmogorov–Smirnov gap `0.943`)**, alongside boost/hunger aliases, mask/paint-order mismatches, and `train_pqn.py --no-self-play` co-adapting `6` live heroes.
- **Sampler (`NOT ESTABLISHED`), `X0` (`EXECUTED_INCONCLUSIVE`), & `B5 v2` (`INCONCLUSIVE_STOP`):** `epoch - legacy` averaged `-1.3694` mass (`95%` CI `[-4.748, +2.009]`). All `5` `X0` arms hit the `435.61 s` cap (`404,516`–`455,227` steps as rate fell `1,513.5 -> 859.7`/s); `B5 v2` pair `1` ran `-2.948%` (`54,177.10` vs `52,580.11`/min) before `NoSuchProcess` (`child_exit_unconfirmed`).
- **`H1-A` (`INCONCLUSIVE_NOT_ADVANCED`), `watch_pre_move_v1` (`ENGINEERING_ACCEPTED`), & `VC1`:** `H1-A` found `0.0` regret LCB across `36` pairs (`51/113` next-obs diffs); `watch_pre_move_v1` passed `2,684` tests; `VC1` hit its `28.425 s` resource cap on seed `2` (`0.5926` seed-`1` held accuracy).

### Why
Encoding `vector61` features in world-frame coordinates caused severe `0.2 -> 1.0` scale drift (`0.943` max KS across `44/61` dims), while short all-hero timing calibrations (`1,513.5` vs `859.7` steps/s under whole-batch barriers) under-sized `X0`'s `435.61 s` wall cap before any arm could finish.

### What to reuse / what not to repeat
- Keep spatial observations heading-aligned and scale-invariant, and calibrate training wall caps across full late-episode survival phases (`822 s` vs `435.61 s`).
- Never cite "the June 2026 CNN lost" (it had zero checkpoints), and use `5` scripted foes + hero slot `0` rather than `--no-self-play` (`6` co-adapting heroes) for fixed-opponent PQN controls.

### Full record
Original: `git show 90b3db94:docs/research/history/pqn-redesign-and-sept-audit.md` | Primary sources: [ml_redesign_blueprint_2026-07.md](../../ml_redesign_blueprint_2026-07.md), [state_and_system_review_2026-09-06.md](../../state_and_system_review_2026-09-06.md), [pqn_decision_phase_2026-09-12.md](../../pqn_decision_phase_2026-09-12.md)

<a id="rl-repair-pqn"></a>
## September 2026 RL Contract-Repair Plan & `X0` PQN Follow-Up (`rl-repair-pqn`)

**Dates:** 2026-09-06 to 2026-09-12
**Verdict:** Integrated `Waves 0–5` (`G0-R3` qualified `7cb66a9` across `2,519` non-slow + `3` slow parity tests), while Wave 6 `X0` (`5` arms `wall_stop` at `435.61 s`) ended `EXECUTED_INCONCLUSIVE`, follow-up `B5 v2` ended `INCONCLUSIVE_STOP`, and Waves `7–10` closed `DEFERRED_CONDITIONAL`.

### What was tried
- Decomposed the post-audit RL repair into `24` file-disjoint agent work packages (`Waves 0–10`: `C0`..`REL`) governed by `dispatch.json` and `validate_plan.py`.
- Implemented and integrated `Waves 0–5` (`C0` -> `G0` on `codex/rl-contract-repair`, `7cb66a9`): `raster31v3` (`26` scalar slots), `ActionMaskSet(legal, advisory)`, reward-only death vs bootstrapped truncation, `reward-v2`, and `promotion-v2-watch-rect`.
- Ran Wave 6 `X0`: `5` `corrected-v3` MPS PQN arms (`16` envs × `6` snakes, rollout `16`, `500,000` requested hero transitions/arm, frozen `435.61 s` wall cap).
- Executed follow-up Packet A (`x0-timing-calibration-v1` full-lifecycle timing) and Packet B (`B0`–`B4` opt-in `per_env_autoreset_v1` + `B5 v2` `3`-pair matched throughput screen).

### What happened
- **`Waves 0–5` (`G0-R3 QUALIFIED`):** Passed `2,519` non-slow tests (`6` skips), `3` slow parity tests, and a `100,901`-step MPS shakedown (`strict_authority: false`).
- **Wave 6 `X0` (`EXECUTED_INCONCLUSIVE`):** All `5` arms hit the `435.61 s` wall cap at `404,516`–`455,227` hero steps (`2,132,589` lower-bound total; finite losses, `1.452 GiB` RSS, `1.113 GiB` MPS), yielding zero final checkpoints or evaluations as screen `1` useful throughput fell from **`1,513.5` (updates `1–73`) to `859.7` (updates `74–491`)** hero transitions/s.
- **Packets A & B (`822 s` timing; `B5 v2` `INCONCLUSIVE_STOP`):** Packet A calibrated an **`822 s`** `500,000`-step wall cap; Packet B (`per_env_autoreset_v1`) stopped after pair `1` when the second child's exit code went unconfirmed (`4` arms unrun; `batch_barrier_v1` remained default).

### Why
Under `batch_barrier_v1`, finished environments sit frozen until all `16` lanes die or reach `5,000` frames (first reset at row `314`), cutting useful hero throughput (`1,513.5 -> 859.7`/s) after the early all-alive phase that `G0`'s short shakedown had extrapolated from.

### What to reuse / what not to repeat
- Reuse `ActionMaskSet(legal, advisory)` (`legal & advisory` fallback to `legal`), death-vs-truncation bootstrap rules, `promotion-v2-watch-rect`, and `validate_plan.py`-checked file-disjoint package specs.
- Never project multi-episode rollout wall caps from a short pre-reset shakedown when whole-batch barriers (`batch_barrier_v1`) leave dead lanes idle.

### Full record
Original: `git show 90b3db94:docs/research/history/rl-repair-pqn.md` | Primary sources: [rl_repair_2026-09-06/README.md](../../plans/rl_repair_2026-09-06/README.md), [rl_repair_2026-09-06/agent_runbook.md](../../plans/rl_repair_2026-09-06/agent_runbook.md), [rl_repair_2026-09-08-followup.md](../../plans/rl_repair_2026-09-08-followup.md)

<a id="four-hour-2026-09-13"></a>
## 2026-09-13 Four-Hour Improvement Program and Same-Day Follow-Up Studies

**Dates:** 2026-09-13
**Verdict:** No `raster31v3` PQN candidate advanced (`vector61` Apex remained incumbent): `50k -> 150k` training regressed both seeds (`-1.143075`, `-6.426900`) and all four single-factor tweaks reversed on seed `2`, while packed raster painting (`-23.165%` MPS), MPS `2.361×`–`2.534×` qualification, `CanonicalWatchScriptedPolicy`, and `hero_food_contact_events` landed permanently.

### What was tried
- Ran `L1` (`50k` steps) and `N2` (`150k` steps, `E16/S6/T16`, `watch_pre_move_v1` vs `random_safe` / `greedy_food`) after fixing `max_capacity = 400 -> 5,002` (`C2`), plus `O1` (`80,000`-frame trajectory trace), `X1` (`1,152`-row off-policy audit), and `R1` (`9,216`-row reflection probe).
- Profiled and optimized tactical raster painting (`I1` packed `uint32` `np.maximum.at`), benchmarked MPS vs CPU2 (`compute_probe`), and compared padded vs unpadded batches (`P2`).
- Tested four 2-seed single-factor PQN interventions: living-mass reward `beta = 0.003` (`mass_objective`), `{N, E, S, W}` reset headings (`direction_coverage`), Adam `lr = 1e-4` vs `5e-4` (`learning_rate`), and `eps_end = 0.20` vs `0.02` (`exploration_floor`).
- Re-evaluated `12` control checkpoints on `8` common worlds (`common_world_retention`), integrated `CanonicalWatchScriptedPolicy` (`scripted-anchor/v1`), and added `hero_food_contact_events` telemetry (`633cf6f`).

### What happened
- **`N2` curve (`INCONCLUSIVE_NOT_ADVANCED`):** Both `150k` arms fell below `50k` (`-1.143075`, `-6.426900`) and `initial` (`-0.492600`, `-0.642600`), though `common_world_retention` showed late drift was heterogeneous (`+4.6574` / `+1.8489` on seed `1342`) and caught a `prepared frame 1 -> anchor frame 0` scripted-clock offset.
- **`O1` period-`4` loop diagnosis:** `50k` policies picked relative right on **`30,128 / 30,171` (`99.86%`)** decisions, dropping food `543 -> 5` and boost `1,269 -> 0` while locking into exact period-`4` loops (`4–25` cells).
- **Single-factor reversals (`NOT_ADVANCED`):** Every tweak gained on seed `1` then reversed on seed `2`: `beta = 0.003` (`+10.40505` / `+0.56000` -> `-0.58615` / `-0.20370`, stuck at alive mass `1.0`), reset headings (`+7.28540` / `+7.98765` -> `-0.36315` / `-0.38565`), `lr = 1e-4` (`+1.31100` / `+1.57175` -> `-13.53755` / `-5.05230`), and `eps_end = 0.20` (`8/12` checks, `-0.5610` random-safe).
- **Engineering speedups (`INTEGRATED`):** Packed `uint32` `np.maximum.at` cut update wall **`-11.206%` CPU** and **`-23.165%` MPS**, and MPS ran **`2.361×`–`2.534×`** faster than CPU2 (`1.08 GiB` peak).

### Why
Under `gamma = 0.997` and sparse events (or unconditional `beta = 0.003` at `mass = 1.0`), tight period-`4` turning loops avoid collision penalties while forfeiting growth, and single-seed 6-snake PQN gains routinely reverse on a second seed.

### What to reuse / what not to repeat
- Keep packed `uint32` `np.maximum.at` raster painting, `CanonicalWatchScriptedPolicy` clock parity, `--simd-body-storage-policy`, and `hero_food_contact_events`.
- Never promote single-seed PQN gains without multi-seed, multi-mix replication (`4/4` single-seed positives reversed on seed `2`).

### Full record
Original: `git show 90b3db94:docs/research/history/four-hour-2026-09-13.md` | Primary sources: [pqn_decision_phase_2026-09-12.md](../../pqn_decision_phase_2026-09-12.md), [models-and-training-history.md](../../notes/models-and-training-history.md), [rollout_policies.py](../../../src/training/rollout_policies.py)

<a id="solo-food-microbenchmark"></a>
## Solo-Food Microbenchmark Series (Studies P–W, 2026-09-13)

**Dates:** 2026-09-13
**Verdict:** Behavior cloning (`P`: `3/3` PASS) and return regression + `0.1` teacher CE (`V`: `mixed`, `3/3` behavior PASS at `100%`, `2/3` scalar fit PASS) solved the `16`-frame length-`1` food microbenchmark where online PQN (`1/3`–`2/3`) and unranked return regression (`U`: `0/2`) failed, while zero-shot `H64` native transfer failed (`W`: `transfer_not_reliable`, `0/3` PASS).

### What was tried
- Built a `16`-frame, length-`1` solo food-seeking microbenchmark (`BatchSim`, `raster31v3`, `RasterDuelingNetwork`, `144` train + `288` new-world + `288` new-placement tasks at distances `3, 5, 7`) requiring `>= 95%` train and `>= 90%` held-out success across `3/3` seeds.
- Compared `500`-update supervised behavior cloning (`BC`) against `50,000`-step online PQN at `lambda = 0.65` (`Study P`), `lambda = 1.0` (`Study Q`), and halved late-half learning rate from `25k` to `50k` (`Study R`).
- Probed failed `Q`/`R` endpoints with zero-SGD first-action counterfactual rollouts (`Study S`, `12,960` branches) and plain-gradient directional pressure (`Study T`).
- Tested offline `SmoothL1` chosen-action return regression alone (`Study U`, `576`-row teacher archive) vs adding `0.1 * masked teacher CE` (`Study V`), then evaluated `Study V`'s `3` `checkpoint_500.pth` models zero-shot on `96` `64`-frame replenishing native solo lanes (`Study W`).

### What happened
- **BC vs online PQN (`P`–`R`):** BC passed **`3/3` seeds** (`100.00%` train/new-world, `98.96%`–`100.00%` new-placement), whereas online PQN passed `1/3` in `P` (`2301` collapsed to `33.33%`), `1/3` in `Q` (`2301` hit `100%` at `12.5k` then fell to `55.21%`–`56.25%`), and `2/3` in `R` (`2303` collapsed to `32.64%`–`33.33%`).
- **Counterfactuals (`S`, `T`):** Forcing the teacher's first move repaired **`95.36%`–`100.00%`** of failed `Q`/`R` blocks (`S`), while `T` showed opposite gradient signs (`+0.104`..`+0.130` vs `-0.373`..`-0.367`).
- **Offline return vs ranking (`U`, `V`):** Chosen-action return regression alone (`U`, `INCOMPLETE_BUDGET_EXHAUSTED`) hit `0.008–0.009` train MAE yet failed gameplay in **`0/2` evaluated seeds** (`21.53%–55.21%`), whereas adding `0.1` teacher CE (`V`) gave **`100% / 100% / 100%` greedy success in `3/3` seeds** (`2/3` scalar fit PASS).
- **Zero-shot `H64` native transfer (`W`, `transfer_not_reliable`):** Collected only **`2.208–2.990` ambient food vs Teacher `12.635`** (`0/3` PASS against the `>= 75%` = `9.476` bar; `2.667–4.135` vs `45.854` at frame `256`, with `2.396` and `6.958` own-trail eating).

### Why
Regressing only `Q(s, a_chosen)` toward scalar return leaves unchosen actions unconstrained (`Study U` vs `V`), while `16`-frame length-`1` single-pellet training leaves policies out-of-distribution once food replenishes, frame progress exceeds `16`, and growth (`L >= 4`) unlocks boost (`Study W` makes no single-factor attribution).

### What to reuse / what not to repeat
- Always pair offline scalar value/return targets with an explicit action-ranking or cross-entropy margin over unchosen actions (`Study V` `100%` vs `Study U` `21.53%–55.21%`).
- Train and evaluate on growing, replenishing native trajectories rather than `16`-frame length-`1` single-pellet tasks (`0/3` transfer in `Study W`).

### Full record
Original: `git show 90b3db94:docs/research/history/solo-food-microbenchmark.md` | Primary sources: [TASK_ALIGNED_AND_APEX_SERIES.md](../../experiments/TASK_ALIGNED_AND_APEX_SERIES.md)

<a id="solo-ambient-early"></a>
## Early Solo & Ambient-Food PQN Series (2026-09-13 to 2026-09-21)

**Dates:** 2026-09-13 to 2026-09-21
**Verdict:** Closed without promoting a cold-start PQN checkpoint (`INDEPENDENT_SOLO_AMBIENT_SIGNAL_NOT_ESTABLISHED`): replay attribution exposed `98.5466%` boost-trail self-eating, cold-start ambient PQN remained seed-fragile (`0`, `22.625`, `110.75` food at `200k`), `1`-opponent `CX` broke paired retention (`70` of `76` head-on deaths), and `E128/T4` (`1,284` steps/s) was selected on M5 Pro.

### What was tried
- Tested `50%` `scripted:greedy_food` exploration advice in `6`-snake PQN (`scripted_advice`, `NOT_ADVANCED`), then built opt-in `1`-snake `S1` (`sole_snake_death_or_frame_cap_v1`, `441` tests) to isolate solo foraging over `5,000` frames (`solo_food_survival`).
- Ran exact food-source attribution (`ambient` vs own boost trail, `solo_food_source_trace`), fresh `2`-seed ambient replication (`1701–02`), and a `1`-pellet relocation `Q`-response probe (`168` rows).
- Swept `ambient_food_reward_coefficient = 0.1` vs `0.0` at `50k` and `200k` steps (`3` seeds `1901–03`, `C2`), `24`-transition TD/clone probes, telemetry exposure (`4,703` rows), and `eps_end = 0.10` (`solo_exploration_floor`).
- Synthesized the solo catalog through `CW` (`H768`) and `CX`–`DA` (`1`-opponent) and benchmarked `7` M5 Pro `(envs, threads)` PQN layouts (`pqn_mac_capacity_2026-09-21`).

### What happened
- **Boost-trail artifact & replication failure:** Apparent `50k` solo food growth (`2.500 -> 1,100.875` on `1601`) was **`98.5466%` (`8,679 / 8,807`) own boost-trail re-eating** (`128` ambient). On fresh worlds, `1701` hit `83.625` ambient contacts while `1702` fell to `0.000` (`0/7` worlds switched greedy action when adjacent food moved).
- **`200k` `C2` curve & `eps_end = 0.10` (`NO ADVANCEMENT`):** `200k` ambient means split at **`0` (`1901`), `22.625` (`1902`), and `110.75` (`1903`)** (`18/24` checks), even though failed `1901` hit `2,756` training ambient contacts (`13.765/1k`) and `24/24` positive adjacent-food TD residuals (`7/12`–`11/12` decode `< 12/12`); `eps_end = 0.10` ended at `1.375, 5.000, 2.375`.
- **Warm-start `H768` vs `1`-opponent ceiling & Mac sizing:** BC-warm-started `CW` passed all `37` `H768` conditions in `3/3` seeds (`122.51, 116.85, 135.34` food; `353, 364, 383` of `384` alive), whereas `1`-opponent `CX` failed paired retention (`70` of `76` `CY` deaths were head collisions). M5 Pro `E128/T4` hit **`1,284` hero transitions/s** (`+38.0%` over `E16/T2`'s `930`/s, `0.913 GiB` RSS).

### Why
Counting unattributed food contacts rewarded boosting and eating one's own dropped trail (`98.5466%`); for cold-start PQN's seed-fragile ambient foraging (`0` on `1901` despite `13.765/1k` contacts and positive TD residuals), **cause was not established** among exploration, coverage, or forgetting.

### What to reuse / what not to repeat
- Always separate `ambient` food from own boost-trail pickups in solo diagnostics, and use `E128/T4` (`1,284` transitions/s) for M5 Pro `BatchSim` PQN runs.
- Do not train cold-start replay-free PQN from scratch expecting seed-uniform foraging (`0` vs `110.75` at `200k`), or assume solo `H768` policies avoid enemy heads (`70/76` head-on deaths).

### Full record
Original: `git show 90b3db94:docs/research/history/solo-ambient-early.md` | Primary sources: [SOLO_FOOD_SERIES.md](../../experiments/SOLO_FOOD_SERIES.md)

<a id="solo-food-pqn-handoff"></a>
## Solo-Food PQN Warmstart, Imitation-to-RL Handoff, Horizon Extension (H64–H768), and One-Opponent Studies (AN–DA and Enemy-Dose)

**Dates:** 2026-09-14 to 2026-09-21
**Verdict:** Across `37` studies (`AN`–`DA` + `enemy-dose`), Q-output scaling (`0.1`), an annealed `CE=1` behavior anchor, own-body time-to-vacate (`TTL`) planes, unclipped `half_mse` loss, and explicit body-length scalars unlocked `3/3`-seed solo PQN competence through `768`-step `H768` (`CW`: `37/37` checks), while `1`-opponent `S2` paired learning passed in only `1/3` seeds (`CZ`/`enemy-dose`).

### What was tried
- Tested `2,598`-param PointNet vs `1,435,751`-param food-only raster BC (`AN`), `32` unanchored PQN updates (`AO`), `8`-update value-offset (`BV`), action-margin pressure (`BW`), Q-head scale `tau = 0.1` (`BX`), `CE=1` behavior anchors annealed over updates `9–32` (`AP`, `BY`, `BZ`, `CA`), food reward `1.0` vs `0.1` (`CB`), and `1,024`–`2,048` PQN updates (`CC`, `CG`).
- Extended solo horizons (`H128` -> `H256` -> `H384` -> `H512` -> `H768`) on `384`-lane (`32`-world) fresh banks by adding own-body `TTL` (`CK`), auditing `lambda` / Huber terminal clipping (`CL`–`CN`), switching to unclipped `half_mse` (`0.5 * MSE`, `CO`–`CU`), and exposing `length / 150` + `log1p(length) / log1p(150)` (`CV`, `CW`).
- Evaluated zero-shot `S2` (`1` `GreedyFood` opponent at `H256`, `CX`/`CY`) and trained `512` + `1,024` `S2` updates (`4096 -> 5632`) with tactical enemy channels `1–3` (`CZ`, `DA`, `enemy-dose`).

### What happened
- **BC-to-PQN handoff (`AN`–`CG`):** Unanchored `32`-update PQN collapsed `H64` food (`11.44–11.74 -> 0.88–7.74` in `AO`; `90.5–94.8` margin-eroding mass in `BW`), whereas `tau = 0.1` + annealed `CE=1` + `food01 = 0.1` (`1.0` fell to `5.29–5.81` in `CB`) recovered at mark `1024` (`11.73–12.15` in `CC`) and passed `17/17` `H128` checks in **`3/3` seeds** (`CG`: `357–370 / 384` alive).
- **`H256` -> `H384` (`CI`–`CR`):** Food-only PQN self-trapped at `H256` (`258–310 / 384`, `607/607` `DEATH_SELF` in `CI`/`CJ`); adding own-body `TTL` + unclipped `half_mse` passed **`3/3` seeds at `H256` (`CO`: `368 / 376 / 374`) and `H384` (`CR`: `348 / 366 / 381`)**.
- **`H512` -> `H768` (`CV`–`CW`, `3/3` PASS):** Adding explicit length scalars passed **`3/3` seeds at `H512` (`CV`: `380 / 380 / 384`) and `H768` (`CW`: `353 / 364 / 383` alive, `122.51 / 116.85 / 135.34` food vs teacher `94.52`, `37/37` checks)**.
- **`1`-opponent `S2` (`CX`–`DA`, `1/3` paired PASS):** `70/76` zero-shot `S2` deaths were head collisions (`CY`); enemy channels `1–3` passed all paired `S2` checks in `1/3` seeds (`6102`/`7202`: `373 -> 376 / 384`), with `19–32` deaths/run (`~0.02%` of `131k` rows) carrying `48.1–51.8%` of squared TD residual (`DA`).

### Why
Uncalibrated BC logits suffer greedy-action margin erosion under initial TD updates unless scaled (`0.1`) and anchored, Smooth-L1 (`Huber`) clipped `100%` of rare (`0.08–0.10%`) terminal self-trap residuals, and saturated grids needed own-body `TTL` and global length to time clearance (`DA` reports `S2` residual shares without claiming an optimizer mechanism).

### What to reuse / what not to repeat
- Calibrate Q scale (`0.1`) with a temporary annealed `CE` anchor and unclipped squared loss when warm-starting TD from supervised policies.
- Include per-cell own-body clearance time (`TTL`) and explicit body-length scalars in raster observations for long-horizon (`H256`–`H768`) survival.

### Full record
Original: `git show 90b3db94:docs/research/history/solo-food-pqn-handoff.md` | Primary sources: [SOLO_FOOD_SERIES.md](../../experiments/SOLO_FOOD_SERIES.md), [behavioral-horizons.png](../../experiments/solo_food_pqn_h768_length_confirmation_2026-09-21/behavioral-horizons.png), [margin-pressure.png](../../experiments/solo_food_pqn_margin_pressure_2026-09-20/margin-pressure.png)

<a id="task-aligned-challenger"></a>
## Task-Aligned CZ `enemy_visible` Challenger & 6102 Continuation Series (2026-09-24 to 2026-09-26)

**Dates:** 2026-09-24 to 2026-09-26
**Verdict:** Retired (`2026-09-26`): task-aligned PQN continuation (`raster31v3`, `(9, 31, 31)` tactical grid) lifted `1/3` lineages (`2026096102`, `+37.9` to `+39.876` `H3000` mass over its parent), further continuation regressed solo mass (`-35.152`), and `6102` trailed incumbent `vector61` Apex by `-30.2` to `-66.4` mass at `H5000` while requiring `N = 8,676 > 120` pilot worlds.

### What was tried
- Continued all `3` CZ `enemy_visible` update-`4,608` PQN `raster31v3` parents (`2026096101`, `2026096102`, `2026096103`) for `31,005` mixed `[solo, s2, s2, s6, s6]` `H1024` rollouts (`23,776,196` hero transitions to update `35,613`, `Q scale 0.1`), evaluated on `32` paired `H3000` worlds × `3` profiles.
- Continued nominated challenger `2026096102` another `31,005` rollouts (`35,613 -> 66,618`, `7,933,551` transitions) against update `35,613`.
- Qualified the `6102` live serving bridge across `v1`–`v5` (`63`-hex literal fix, `(9, 31, 31)` tuple/JSON list fix, `monotonic` heartbeat fix, pytest `9.0.1` `subTest` count `82` vs `75` audit fix -> `AUDIT_ONLY_PASS`).
- Ran strict challenge pilot `pilot-v1` (`16` Apex dev + `16` paired `H5000` worlds/mix on `promotion-v2-watch-rect`, `720,000` frames) against `champion_a5_freespace_20260621.pth`.

### What happened
- **3-lineage `H3000` screen (`INCONCLUSIVE_SCREEN`, `6102` nominated):** Only `2026096102` cleared every gain and solo-retention bar (**`+37.946` (`+37.9`) solo, `+39.876` `food_pressure`, `+38.104` `mixed` mass**); `2026096101` (`-2.093` solo) and `2026096103` (`-17.199` solo, `+4.343` mixed) missed retention/gain bars.
- **Further `6102` continuation (`NEGATIVE_CONTINUATION_SCREEN`):** From `35,613 -> 66,618`, solo living mass fell **`-35.152`** (`78.750 -> 43.598`; `-5.067` `food_pressure`, `-7.299` `mixed`) even as food (`470.9 -> 660.1`), alive fraction (`0.806 -> 0.946`), and boost (`0.362 -> 0.646`) rose.
- **`H5000` strict pilot vs Apex (`PILOT_STOP_INFEASIBLE_AT_FIXED_MDE` / `INVALID_STOP_AUDIT_FAILED`, RETIRED):** Unaudited `6102` trailed Apex across all `3` mixes (**`-30.2` frozen, `-66.4` scripted, `-34.9` mixed** mass, winning `5/16` worlds each), and `10%`-MDE variance (`SD 55.8–118.3`) required **`N = 8,676 > 120`** worlds.

### Why
Task-aligned PQN on `raster31v3` remained far below `vector61` Apex at `H5000` (`-30.2` to `-66.4`), and `10%`-MDE gating on noisy `H5000` trajectories (`1–3` noise-to-mean ratio) needed `552–8,676` worlds/mix; for `66,618`'s `-35.152` mass drop alongside `0.555–0.646` boost, boost was not intervened on so **cause was not established**.

### What to reuse / what not to repeat
- Size `H5000` confirmation gates for thousands of fast vectorized episodes (`N = 552–8,676` at `10%` MDE) rather than `120` live-sim worlds.
- Normalize JSON round-trip shapes (`list` vs `tuple`), monotonic heartbeats, and pytest `subTest` counts before running strict auditors (`v1`–`v5` bridge stops).

### Full record
Original: `git show 90b3db94:docs/research/history/task-aligned-challenger.md` | Primary sources: [README.md](../../experiments/task_aligned_cz_integration_2026-09-24/README.md), [README.md](../../experiments/task_aligned_continuation_2026-09-24/README.md), [experiment_portfolio_review_2026-09-26.md](../experiment_portfolio_review_2026-09-26.md)

<a id="m3-m7-redesign"></a>
## October Spatial Redesign (GridBatchSim + ego2s-bv8 M1–M7) and T1–T11 / Track-4 Rollout & Return-Bank Series

**Dates:** 2026-10-07 to 2026-10-31
**Verdict:** Bare `B8` (`ego2s-bv8`, `~1.64M` params, no runtime veto) cleared T7 Phase R `STRICT_PASS` (`+138.67` H5000 mass over `frp3-s12 + v8`, `5/5` seeds `+`, Tier 2 `-0.016 >= -0.05`) and shipped `b8-s3`, while every post-B8 late-death fine-tune, feature probe, and privileged critic (`M6`, `T2–T6`, `T8–T11`, `t4_gate1`) closed `NULL`, `KILL`, or diagnostic-only.

### What was tried
- Built cell-exact `GridBatchSim` (`72/72` identical H5000 episodes, `27,923` scripted hero steps/s) and distilled `2.08M` `frp3-s12 + v8` decisions into `Ego2sNet`: `M2` (`12` scalars) vs `M2b` (`ego2s-b`, adding `enemy_next` + `18` scalars = `30` scalars).
- Trained multi-actor Double DQN in `M3-B` / `M3-C` (`gamma 0.99 -> 0.995`, `v2` kill bonus) and repaired the recipe in `M4-A` / `M4-B` (`ego2s-bv8`: `7×31×31` + `4×37×37` ego-rasters + `60` scalars incl. `30` `v8` safety scalars, `mass` reward `kappa=1.0`, `kill_scale=0`, `gamma -> 0.998`, `50/25/25` scripted/frozen/mixed, `v8margin` + on-policy `v8` labels, `5` seeds × `20M`).
- Evaluated `5`-seed `B8` on T7 fresh Phase R (`1,920` fresh worlds = `5,760` H10000 episodes) against `frp3-s12 + v8` and Tier 2 `champion + v8`.
- Tested post-B8 late-death (`L >= 800`) fixes: `M6` shaping (`TIGHT`, `DEATHW`), `T2/T2b` Huber deltas (`{10, 30, 100}`), `T3`/`T4a`/`T4c` `D=300` rollout-margin banks (`16,327` rows), `T4d–T4f` `D=300` return regression (`19,976` states), `T6` `RESTART` snapshots (`10,885` states), `T5/T8/T9/T10` input/horizon probes, `T11` `LOCAL30`, and `t4_gate1` privileged full-arena critic (`384` worlds).

### What happened
- **Distillation & `M4-B` (`B8`) -> T7 `STRICT_PASS`:** `M2` failed (`-29.5` mass), whereas `M2b` passed (`+52.4` vs bare `frp3-s12`, `LB90 +34.8`). `M3-C` collapsed to `1–11%` survival (`177.4`/`172.9`/`184.6` vs `434.9`), while `M4-B` (`B8`) passed T7 `STRICT_PASS` at **`+138.67` (`+138.666`) H5000 mass** (`LB90 +129.48`, `+93.75` H10000, Tier 2 `-0.0156`), shipping `b8-s3` (`+149.65`).
- **Margin & shaping fine-tunes (`NULL`/`KILL`):** `M6` was `NULL` (`-24` / `+10` mass); `T2/T2b` Huber sweeps gained `+0.072`..`+0.125` scripted survival but lost `14–202` mass; `T3` rollout margins (`55.86%` boost labels) lost `81–88` (`-88` / `-88` / `-81`) scripted mass; `T4a`/`T4c` (`8.58%` nonboost bank + `10,885` outside counters, `3.81%` drift) ended `KILL` (`-0.027` / `-0.014` survival).
- **Return & restart fine-tunes (`KILL`):** `T4e` `RETURN` burned **`34%` boost** (`+31.99 Q` boost target gap), `T4f`/`T11` tripped `2%`/`1%` boost guards, and `T6` `RESTART` gave `0.000` survival change.
- **Horizon probes & `t4_gate1` (`KILL_T4`):** `T1` found `46/49` (`93.9%`) late deaths escapable within `300` steps (`T3` dense `38/64 -> 54/64`, `608.1 -> 700.0` mass), yet `60`-scalar AUC fell `0.9121` (`H=30`) -> `0.7261` (`D=300`; `0.726–0.769` in `T5`/`T8`, `49–65`-step pocket lag in `T9`), and `t4_gate1` passed `0/8` bars (`+0.0266` EV, `0.5617` doom AUC).

### Why
Feeding `30` `v8` safety scalars + `enemy_next` into `Ego2sNet` with living-mass reward (`kappa=1.0`), zero kill bonus, `gamma=0.998`, and `50/25/25` mixes cured `M3-C`'s kill-lunge trap (`1–11%` survival), whereas `L >= 800` traps form over `50–100` multi-snake steps (`0.726–0.769` D300 AUC), making margin nudges spill caution and truncated `D=300` returns hide post-`300` boost burn (`+31.99 Q`).

### What to reuse / what not to repeat
- Reuse the `M4-B` `B8` training recipe: multi-scale ego-rasters + teacher safety cues (`ego2s-bv8`), pure living-mass reward (`kill_scale=0`), `gamma=0.998`, and `50/25/25` opponent mixes (`train_m3c.py`, `GridBatchSim`).
- Never combine explicit kill bonuses with short discount factors (`gamma=0.995`), and never regress truncated `D=300` rollout returns without pricing post-horizon segment loss (`34%` boost collapse).

### Full record
Original: `git show 90b3db94:docs/research/history/m3-m7-redesign.md` | Primary sources: [t7_phase_r_result_2026-10-16.md](../t7_phase_r_result_2026-10-16.md), [m4a_design_2026-10-12.md](../m4a_design_2026-10-12.md), [redesign_campaign_retrospective_2026-10-17.md](../redesign_campaign_retrospective_2026-10-17.md)

<a id="t16-t24-perf-league"></a>
## Colab GPU Throughput, Actor/Pool Optimization, and the T23 League Screen (`t16-t24-perf-league`, T16–T24)

**Dates:** 2026-10-17 to 2026-10-18
**Verdict:** Host-gather (`g3`), Numba `v8` frozen-pool (`T22`, `-85.1%`), and shared body-map (`T24`, `-26.5%`) speedups reached `3.0x` Mac on A100 (`15,927` tr/s) and made Colab L4 `~1.8x` more unit-efficient than A100 (`~3.4x` in post-T23 plan), while the `5`-seed × `8M` `N = 3` static-pool fine-tune (`T23`) closed `KILL` at `-80.8` H5000 mass vs `b8-s3` (`0/5` seeds positive).

### What was tried
- Benchmarked Mac MPS (`B5`) vs Colab A100-40GB/80GB and L4 on `Ego2sNet` (`1.64M` params, `256` transitions/update) against a `4x` Mac speed target (`84.375` upd/s / `21,600` tr/s), adding `3` nogil host-gather threads (`g3`) and raw mmap rings (`T16`, `T17`, `T17b`).
- Built archive-only RunPod H100 `prepare.py` (`$4/h` ceiling, `$8` cap, self-delete credential check; dry-run only, no result recorded, `T18`) and `--learner-seats {1,3,6}` over a 6-policy static pool (`T19`).
- Replaced Python `v8` opponent wrappers with Numba `v8` kernels (`T22`) and reused retained-body release maps across `6` actions (`T24`).
- Ran `T23` (`R0`): `5` seeds (`7101`–`7105`) × `8M` learner decisions from `b8-s3` (`bs2`, `fp32`, `N = 3` static pool) on A100-80GB, evaluated on `640` paired H5000 worlds (`1,280` episodes).

### What happened
- **Mac MPS ceilings (`T16`–`T17b`):** Mac MPS ran at `5,459` self-play and `5,214` `N = 3` tr/s (`20.4–21.3` upd/s), while MPS `bf16` regressed `~3x` (`47.81–50.53` -> `135.55–149.78 ms`/update) and `reduce-overhead` compile fell back to eager.
- **A100 learner ceilings:** On A100-40GB, `g3` + raw rings lifted self-play `11,500 -> 15,927` tr/s (`2.1x -> 3.0x` Mac, `62.2` upd/s), with isolated fp32 learner capping at `78` upd/s (`82.7` on A100-80GB, below `84.375` `4x` bar; `bf16` reversed to `47.6` upd/s in-run).
- **Pool & L4 efficiency (`T22`, `T24`):** `T22` cut non-B8 host pool work **`14.458 -> 2.154 ms`/frame (`-85.1%`)** (`13,578` `a4` / `14,833` `a6` projected A100 tr/s), and `T24` cut `L = 1200` `v8` **`22.155 -> 16.280 ms`/frame (`-26.5%`)**. On L4 (`12` vCPUs, `~2` u/h), `N = 3` hit `7,625` tr/s (`~3,800` vs A100 `~2,100` tr/s/unit = **`~1.8x`**, `~3.4x` in post-T23 plan).
- **`T23` league screen (`KILL`):** Mean equal-mix H5000 mass fell **`-80.8`** vs `b8-s3` (`90% CI [-96.1, -65.5]`, `0/5` seeds positive: `-60` to `-111`; scripted `-101`, frozen `-61`, mixed `-80`) with survival flat or higher (`0.76` vs `0.68` frozen), cancelling the long `Lane A` league run.

### Why
Python `v8` wrappers (`709 s` of `1,063 s` actor time) and single-thread gather (`19–23 ms`/update) bottlenecked throughput until compiled in Numba/nogil; for `T23`'s `-80.8` loss, the post-T23 plan named restarting B8's schedule (`gamma .99`, `eps .05`, `anchor 1.0`) with fresh Adam/empty replay as the leading **untested** suspect (cause not isolated).

### What to reuse / what not to repeat
- Keep `g3` nogil gather threads, Numba `v8` frozen-pool dispatch (`-85.1%`), and shared `v8` body-map templates (`-26.5%`), and price throughput per Colab unit (`L4` `~1.8x`–`3.4x` better than `A100` on host-heavy `N = 3`).
- Never use `bf16` on Mac MPS (`~3x` slower) or count raw actor steps/s without normalizing by `256`-transition learner updates/s, and do not repeat `T23`'s restarted-schedule `N = 3` static-pool fine-tune (`-80.8` mass).

### Full record
Original: `git show 90b3db94:docs/research/history/t16-t24-perf-league.md` | Primary sources: [RESULTS_T23.md](../../../research/t23_r0_screen_20261017/RESULTS_T23.md), [PREREGISTRATION_T23.md](../../../research/t23_r0_screen_20261017/PREREGISTRATION_T23.md), [t17b_20261017/results/](../../../research/t17b_20261017/results/)

<a id="q0-q1"></a>
## Post-T23 Qualification & Schedule/Collection Factorial (Q0 / Q1, 2026-10-17 to 2026-10-18)

**Dates:** 2026-10-17 to 2026-10-18
**Verdict:** `Q0` (`Q0a` mass/initializer audits + `Q0b` capacity qualification) `PASSED` and `QUALIFIED` both A100 and L4, while the `160M`-decision `Q1` `2 × 2` schedule × collection factorial closed `REFUSED` / `UNRUN` before dispatch when allocation-inclusive runtime accounting (`362.55 u`) exceeded Amendment A5's final caps.

### What was tried
- Ran `Q0a` Mass-Loss Audit (`480` H5000 episodes across `5` seeds × `2M`/`4M` checkpoints × `3` pools × `16` fresh worlds) to find when and how T23's `-80.78` (`-80.7846`) mass drop emerged.
- Ran `Q0a` Initializer Fidelity Audit (`698,905` candidates across `9/9` length × local-`31 × 31`-openness strata under Amendment A2, `4,096` retained rows) and `Q0b` capacity/lag qualification on A100 and L4 (`N=1` and `N=3`, Amendment A3).
- Registered `Q1` (`4` arms × `5` seeds × `8M` = `160M` decisions crossing `restart` vs `mature` `gamma=0.998, eps=0.01, anchor=0.1` with `legacy` `N=3` vs `task-aligned-n1` `N=1`) and audited whole-runtime Colab allocation lifetimes (`allocation_accounting.py`) against Amendment A5's final caps (`36` usable / `45` nominal GPU-h, `32/40` CPU-h, `168.6304/203.6304 u`).

### What happened
- **`Q0a` audits (`COMPLETE`):** T23's mass loss was already present at **`2M` decisions** (`scripted` `538.1 → 425.1`, `frozen` `524.1 → 470.0`, `mixed` `572.6 → 461.1`), driven by a **`12–17%` food/alive-frame drop** (`0.286 → 0.252`, `0.241 → 0.203`, `0.237 → 0.204`) with deaths/episode flat (`0.30 → 0.20` frozen); initializer check had `0` action diffs across `4,096` rows (`1,860` open).
- **`Q0b` capacity (`QUALIFIED`):** At `N=1`, L4 (`2,396.9 / 2,387.8 dec/s`, `2.0 u/h` rate) matched A100 (`2,380.7 / 2,430.8 dec/s`, `5.3 u/h` rate) at `~38%` unit burn; at `N=3`, A100 hit `13,508.1 / 13,585.5 dec/s` vs L4 `7,555.1 / 7,487.2 dec/s` (`~1.8×`).
- **`Q1` admission (`REFUSED`, `0` manifests):** Active `Q0` timers summed to **`13.39` (`13.394976`) GPU-h** (`68.63 u`), whereas whole-lifetime accounting exposed up to `~76` GPU-h of idle queue-worker exposure (`89.5528` GPU-h / `317.2435 u` conservative `Q0` upper bound), pushing `Q0 + Q1` upper bounds to **`105.63` GPU-h, `51.30` CPU-h, and `362.55 u`** past every A5 cap.

### Why
Idle legacy notebook queue workers sitting unclaimed between retries dominated runtime exposure (`89.55` GPU-h upper bound vs `13.39` GPU-h active `Q0` work); for T23's `2M` `12–17%` foraging drop, `Q0a` called the pattern consistent with, not proof of, schedule-restart shock (`Q1` never ran).

### What to reuse / what not to repeat
- Account for full Colab session allocation lifetimes (replaced immediately after by lease-fenced `colab_cli` sessions with verified stops) rather than only active job claim-to-terminal timers (`13.39` vs `89.55` GPU-h).
- Run `N=1` CPU-env-bound RL jobs on L4 (`~2,392 dec/s` at `2.0 u/h`) rather than A100 (`~2,406 dec/s` at `5.3 u/h`), and define crowding strata in the `31 × 31` local crop rather than `alive <= 3` global snakes (`0` hits under 6-snake respawn).

### Full record
Original: `git show 90b3db94:docs/research/history/q0-q1.md` | Primary sources: [Q0_MASS_AUDIT.md](../../../research/q0q1_20261017/results/Q0_MASS_AUDIT.md), [Q0B_A3_REQUALIFICATION_20261018.md](../../../research/q0q1_20261017/Q0B_A3_REQUALIFICATION_20261018.md), [Q1_A5_ADMISSION_20261018.md](../../../research/q0q1_20261017/Q1_A5_ADMISSION_20261018.md)

<a id="s1-mature"></a>
## S1 Mature-Schedule League vs Hero Fine-Tuning (`s1-mature`)

**Dates:** 2026-10-18 to 2026-10-19
**Verdict:** Both mature-schedule fine-tuning arms closed `KILL` on one Mac evaluation look (`-12.5` and `-14.9` pooled H5000 mass vs bare `b8-s3` across `5` paired seeds × `8M` decisions, vs `>= 20` and `LB90 > 0` GO bars; failure closes these packages only and cannot isolate schedule, collection, or optimizer-reset causality).

### What was tried
- Trained two `5`-seed × `8,000,000`-decision mature-schedule (`gamma = .998`, `epsilon = .01`, `anchor = .1` from step zero, fresh Adam, empty replay) fine-tunes from `b8-s3` on Colab L4: `N3` (`6` actors / `3` static-pool learner seats) and `N1` (`3` actors / `1` terminal-hero seat).
- Repaired a `48`-decision actor-ingestion barrier overshoot (`12,048` vs `12,000` decisions, bound `47`) in `train_m3c.py` by halting ingestion while a barrier is pending and returning drained packets to per-actor FIFOs (`0` overshoot).
- Audited full-capacity replay snapshot serialization (`500,000` replay + `300,000` label rows at `12,449 B`/row) and switched to uninterrupted final-only weight exports (`final.pth` + `target.pth`, `64 MiB` per-attempt cap) before running `10` L4 jobs and one `640`-world (`1,920`-episode) Mac look.

### What happened
- **Gameplay evaluation (`KILL` both arms):** While losing much less than T23's `-80.8`, both `N3` and `N1` regressed against bare `b8-s3` at **`-12.5` and `-14.9` pooled H5000 mass** (`64` scripted + `32` frozen + `32` mixed worlds/seed; audit PASS), missing `LB90(G) > 0` and mean `G >= 20`.
- **Snapshot size vs final-only export:** Full `torch.save` arrays measured **`11,178,149,683 B` (`N3`)** and **`10,490,632,471 B` (`N1`)**/snapshot (`867,602,770,220 B` across `4` snapshots × `10` jobs), whereas uninterrupted `30/60` min (`N3`) and `68/136` min (`N1`) runs exporting `final.pth` (`6,566,741 B`) + `target.pth` (`6,562,677 B`) shrank the archive to **`12,180,957 B` (`11.62 MiB`)**.
- **Preflight Mac eval timing:** Evaluating `10` disjoint H5000 worlds took `67.21 s` summed episode time (`2` 1-thread workers) / `63.68 s` serial (`154 s` command wall) with identical result bytes.

### Why
Per `PREREGISTRATION.md`, `S1` was designed without a restarted-schedule or warm-buffer control, so **cause was not established** between mature schedule, `N3` vs `N1` collection, fresh Adam/empty replay reset, or their interaction.

### What to reuse / what not to repeat
- Drain in-flight actor packets into learner-local FIFOs while a step barrier is pending so asynchronous multi-actor trainers hit exact decision targets (`0` overshoot).
- Price full replay-buffer checkpoint sizes (`~10.5–11.2 GB`/snapshot) before dispatch; for short (`30–68 min`) L4 runs under `~1 h`, export compact terminal weights (`~12.2 MB`) rather than hundreds of gigabytes of raw replay arrays.
- Do not repeat `8M`-step fresh-Adam / empty-replay DQN fine-tunes from `b8-s3` under static `N3` or `N1` pools (`-12.5` / `-14.9` mass).

### Full record
Original: `git show 90b3db94:docs/research/history/s1-mature.md` | Primary sources: [s1_mature_20261018/README.md](../../../research/s1_mature_20261018/README.md), [s1_mature_20261018/PREREGISTRATION.md](../../../research/s1_mature_20261018/PREREGISTRATION.md), [CAMPAIGN_LOG.md](../final_run_20261019/CAMPAIGN_LOG.md)

<a id="bl-ppo-p1"></a>
## Big-Lever E4/E2 PPO Pilot, WP-3 P1 100M PPO Endpoint, Y3 Critic Qualification, and A3 Learner Ceiling

**Dates:** 2026-10-20 to 2026-10-23
**Verdict:** Power gate `E9` blocked the two-arm `E2` pilot (`.27045`–`.30700` vs `.60` required), `E8` passed late-body critic qualification (`EV = 0.5867` via float64 Helmert least-squares, certifying value-head init only), `A3` kept `P1` on L4 (`0.98 s` learner vs `~9.3–9.6 s` host-CPU collection), and `Y3` stopped at `HOLD_MISSING_OPERATIONAL_PROOF`.

### What was tried
- Ran `E9` operating-characteristic simulations (`20,000` and `100,000` replicates using T23 paired H5000 mass SDs `364.683` scripted, `262.732` frozen, `212.360` mixed) to test whether two-arm `E2` (`GRU256` `R` vs feedforward `F`) met `>= .60` joint funding power at `+40` mass and `<= .10` null pass.
- Registered single-endpoint `WP-3 P1` (`100,000,000` hero transitions from `b8-s3`, `128` worlds × `128` steps, `12` L4 CPU collectors), gating actor unfreeze on `E8` held-out late-body (`L >= 800`) PopArt critic qualification (`EV >= .1` AND `MSE < baseline` using frozen B8 trunk + remaining-horizon clock).
- Qualified `Z-9` public-catalog HTTPS checkpoint staging (`2,409 B` catalog vs `30,720,004 B` direct uploads) and `A3` (`16,384`-transition 2-epoch PPO learner ceiling `<= 1.6384 s` on A100-40GB vs L4).

### What happened
- **`E9` power gate (`E2 BLOCKED`):** Full funding probability at `+40` was `.27045` (`96/32/32`) and `.30700` (`128/48/48`) vs `.60` required (null pass `.05050` / `.06160`); none of three `Z-4` replacements reached `.60` (best `.36216` at `192/64/64`).
- **`Z-9` upload & `A3` learner (`keep P1 on L4`):** Direct 6-checkpoint upload (`~690.6 s`) exceeded `300 s`, whereas the `2,409 B` catalog + HTTPS fetch took `65.41 s` (`242.11 s` worst case). Baseline `A3` missed `1.6384 s` (`1.87 s` A100 / `1.81 s` L4); optimized learner passed only on L4 (`0.98 s`; A100 `0.41 s` `REJECT_UNEXPLAINED_GRADIENT_CHANGE`) under `~9.3–9.6 s` CPU collection.
- **`E8` critic (`PASS`):** Online one-pass Adam (`source-02`) failed; offline `D1` converged to `EV = 0.5867 >= .10` and `MSE = 307,567.7 < 561,561.6` (`162` rows), and CPU float64 least-squares on a `256 × 255` Helmert basis (`rank 257`, `source-latebody-06`) passed critic qualification (`427,911,156 B` origin snapshot, certifying value-head init only).
- **`Y3` continuation (`HOLD_MISSING_OPERATIONAL_PROOF`):** Continuation stopped at `49,152` actor transitions (`3` updates) before the first `900 s` periodic capture, leaving no actor operational snapshot (repair `-05` `PREPARED_HELD`).

### Why
Cross-episode H5000 variance (`SD` `212.360`–`364.683`) limits 96–192-world-per-mix two-arm screens to `.27–.36` joint power at `+40`; in `E8`, frozen B8 features plus remaining clock linearly predict late returns (`EV 0.5867`), which one-pass Adam under-fit until replaced by exact float64 least squares; in `A3`, host-CPU rollout collection (`~9.3–9.6 s`) dwarfed GPU learning (`0.98 s`).

### What to reuse / what not to repeat
- Fit linear PopArt value warm-starts (frozen trunk + remaining-horizon clock) by exact float64 least squares (`EV 0.5867`) rather than a single under-converged Adam pass.
- Stage multi-checkpoint rosters via a tiny hash-pinned catalog (`2,409 B`) + HTTPS fetch (`bl_upload`) instead of uploading `30 MB` over CLI RPC (`~690.6 s` vs `300 s` cap).
- Keep CPU-collection-bound PPO runs (`~9.3–9.6 s` collection vs `0.98 s` learner) on L4 rather than A100, and check multi-arm gate power (`E9`) before dispatch.

### Full record
Original: `git show 90b3db94:docs/research/history/bl-ppo-p1.md` | Primary sources: [REGISTRATION_P1.md](../../../research/bl_ppo/REGISTRATION_P1.md), [E9_RULE.md](../../../research/bl_ppo/E9_RULE.md), [a3_p1_ceiling/README.md](../../../research/a3_p1_ceiling/README.md)

<a id="bl-prefix-e6-e7-e10"></a>
## Big-Lever Prefix Escape Dividend (`E6`), Frontier-Existence (`E7`), and Population Bridge (`E10`)

**Dates:** 2026-10-21
**Verdict:** Built, Mac-preflighted, and held pre-dispatch (`NOT_DISPATCHABLE` / `NOT_DISPATCHED`, `0` scientific Colab units spent) because upstream `E2` `R`/`F` candidate policies were blocked by the `E9` power gate and never produced.

### What was tried
- Registered `E6` (`REGISTRATION_E6.md`): substitute `64` candidate (`R` or `F`) decisions starting `100` decisions before late `b8-s3` deaths (`frame >= 2000`, `L >= 700`) before `b8-s3` resumes across `128` death worlds (`44` scripted, `42` frozen, `42` mixed) and `128` length-matched (`±100` cells) ordinary worlds.
- Registered `E7` (`REGISTRATION_E7.md`): roll out `8` stochastic `b8-s3` prefixes (`5%` non-greedy softmax temperature) from `100`, `150`, and `200` decisions before death across `128` death worlds and `2` independent confirmation RNG streams (`>= +20` mass on `>= 20%` of worlds -> `TRANSFER_ADMISSIBLE` vs `FRONTIER_SCARCE`).
- Registered `E10` (`REGISTRATION_E10.md`): two-phase population bridge `theta = mean_m(q_m * mu_m)` testing whether `UB90(U)` reaches `+40` (`FINITE_MENU_INSUFFICIENT` if `UB90(U) < 40`, minimum rescue fraction `r >= 0.163`, score ceiling `B = 2501.5`).
- Audited `11,224` high-range `+2**40` seeds, built an ordinal-label adapter around `strict_promotion.materialize_rosters`, and ran 2-worker full-H5000 Mac preflights (`production_preflight.py`).

### What happened
- **Seed audit & roster adapter (`PASS`):** Shifting derived seeds by `+2**40` (`bl-prefix-e6e7-development/20261021/v2`) passed with `0` collisions across `11,224` seeds (`3,998` unburned exploration seeds >= `3,072` needed after retiring `2` operational timing seeds), and `54` package tests passed with exact `300`-frame restoration parity.
- **Full-H5000 runtime sizing (`CAP_NOTE.md`):** Across 2 Mac workers, `E6` measured `0.0032107433 s/step` (`37.76M` max world-steps -> `520.156944 min` / `17.338565 u` expected on 12-worker L4 at `3×` slowdown, needing a `43.4 u` cap vs original `4 u`), and `E7` measured `0.0031417799 s/step` (`24.96M` max world-steps -> `341.745108 min` / `11.391504 u`, needing `28.5 u` vs original `20 u`).
- **Dispatch gate (`NOT_DISPATCHED`):** Held with `resource_readiness.ready = true` and `bl_prefix.ready = false` (`0` scientific units) because `E9` blocked `E2`, leaving no `R`/`F` endpoints or critic receipts.

### Why
Rolling prefix branches all the way to `H = 5,000` across `8` prefixes and `2` confirmation streams requires `24.96M–37.76M` world-steps (`5.7–8.7` L4 hours vs initial `4 u` / `20 u` estimates extrapolated from tiny fixtures), and `E6` scientifically depended on `E2` candidate checkpoints that `E9` prevented from running.

### What to reuse / what not to repeat
- Shift new development seed families into high 64-bit ranges (`+2**40`) with an ordinal-index roster wrapper around `strict_promotion.materialize_rosters` to avoid 32-bit historical seed collisions without editing frozen evaluation code.
- Multiply conditioned late-death rescue gains by natural root prevalence `q_m` and debit ordinary-world collateral `C` (`E10` two-phase bridge) rather than quoting conditioned root lifts as population H5000 gains.
- Size H5000 continuation caps from full-horizon `production_preflight.py` fixtures (`24.96M–37.76M` steps), not short-horizon smokes.

### Full record
Original: `git show 90b3db94:docs/research/history/bl-prefix-e6-e7-e10.md` | Primary sources: [bl_prefix/README.md](../../../research/bl_prefix/README.md), [REGISTRATION_E10.md](../../../research/bl_prefix/REGISTRATION_E10.md), [CAP_NOTE.md](../../../research/bl_prefix/CAP_NOTE.md)

<a id="bl-e16"></a>
## Big-Lever E16 Execution-Ranking Harness and Round-07 X6 Campaign Governance

**Dates:** 2026-10-21
**Verdict:** Closed at `BUILD_VERIFIED / NOT_DISPATCHABLE` (`UNUSED` under `PLAN_v3`, `0` Colab runs) after passing Mac continuation-parity preflights and a `24,876`-seed freshness audit while upstream `E2-F`, `E3`, and critic prerequisites remained unmet.

### What was tried
- Built `E16` (`research/bl_e16`, `2M` PopArt critic + `4M` PPO actor transitions from `E2-F`) comparing Arm `A` (natural inputs) vs Arm `G` (7-plane `_RasterTower(7,16,64,(6,10))` E3 geometry + optional `GRU(43, 64)` over `128` steps) across `288` paired H5000 worlds (`96/mix`, `LB90(J) > 0`, `point G >= 10`, `>= 5/100` rescues/mix, `>= 90%` tail retention, `UB90 <= 10`).
- Built Round-07 `X6` machine-checked governance ledgers (`admission.py`): campaign look ledger (`LOOK_LEDGER.md`), append-only world-burn manifest (`BURN_MANIFEST.md`), 4-receipt `E13` parity rule (`PARITY_RULE.md`), and 3-slot `alpha=0.05` Holm final confirmation family (`FINAL_FAMILY_REGISTRATION.md`).
- Audited `24,876` 48-bit high signed-int64 seeds against `1,167,085` historical identities (`120,177` files) and ran Mac serial/parallel and boundary-restore parity preflights (`preflight-01`..`03`).

### What happened
- **Seed audit (`PASS`):** Verified `0` collisions across all `24,876` seeds (`288` eval, `9` timing, `3` preflight, `12,288` critic, `12,288` actor) against `1,167,085` historical identities, `7` FRP exclusions, and `46` alias records.
- **Mac preflight (`BUILD_VERIFIED / NOT_DISPATCHABLE`):** Completed `480` active worldsteps (`3` worlds × `4` controllers × `40` frames) in `2.7720 s` serial (`173.16 worldsteps/s`) and `3.1048 s` parallel (`154.60 worldsteps/s`), with bit-exact next-3-transition, weight, and Adam parity across boundary restores for Arms `A` and `G` (`40 u / 20 h` ceiling, `5.667 h` / `11.334 u` forecast).
- **Lifecycle status (`UNUSED`):** Never dispatched because upstream prerequisites (`E2-F` parent, `E12` critic qualification, `E3` `GO` + `E14` `PASS`, and `E4` throughput rulers) were never satisfied before `PLAN_v3` stopped the line.

### Why
`E9` blocked upstream `E2` and `E3` never delivered a `GO` receipt, so `admission.py`'s hash-bound prerequisite checks kept `E16` non-dispatchable by design; during Mac preflight, unpickling `GridBatchSim` initially broke NumPy interior/padded grid view aliases until load-time view restoration and 32-row Monte Carlo critic-spool flushing were added.

### What to reuse / what not to repeat
- Restore NumPy interior/padded view aliases after unpickling `GridBatchSim` snapshots and flush pending Monte Carlo critic spools at checkpoint boundaries to guarantee bit-exact resume parity.
- Reuse 48-bit high signed-int64 seed derivations (`0` collisions vs `1,167,085` historical seeds) and hash-bound prerequisite gates (`admission.py`, `LOOK_LEDGER.md`, `BURN_MANIFEST.md`) so stubbed or unvetted arms cannot launch.

### Full record
Original: `git show 90b3db94:docs/research/history/bl-e16.md` | Primary sources: [bl_e16/README.md](../../../research/bl_e16/README.md), [PREFLIGHT.md](../../../research/bl_e16/PREFLIGHT.md), [REGISTRATION.md](../../../research/bl_e16/REGISTRATION.md)

<a id="train01"></a>
## GPU-Resident Long-Horizon PPO on A100 (`train01`, Packets 05–16)

**Dates:** 2026-10-27 to 2026-10-30
**Verdict:** Closed with zero strict-gate nominations after `384` GPU-resident PPO updates (`100,663,296` actor rows) from `b8-s3` peaked at `+14.23` and finished at `+8.09` equal-mix H5000 mass against the `>= +40` material-gain bar (insufficient evidence to advance, not a proof of zero benefit).

### What was tried
- Fine-tuned a fully trainable `b8-s3`-initialized `ego2s` actor with Proximal Policy Optimization (`PPO`), a frozen-B8 Kullback-Leibler (`KL`) anchor, and a separate frame-clocked PopArt return-normalized critic (`E360` complete-life preparation: `270` fit + `90` holdout worlds).
- Ran `384` updates (`E1024` GPU-resident worlds × `T256` / `256`-step rollouts = `100,663,296` actor rows across equal `scripted`/`frozen`/`mixed` opponent mixes) on an A100 GPU under Resume-by-Default, evaluating checkpoints on a `96`-world-per-mix paired H5000 development proxy against a `>= +40` pooled mass bar.
- Repaired four engineering blockers across packets `05`–`15`: NumPy slice-view memory retention, `30`-s `git fetch` staging RPC timeouts, first-update `+inf` KL divergence at calibrated temperature `T = 0.008439731318504158` (`T ≈ 0.00844`), and `300 s` upload timeouts on `1.33 GB` resume snapshots.

### What happened
- **Critic prep & trajectory (`COMPLETE`):** Critic preparation passed (`held_out_late_ev = 0.4404 >= 0.10`), and packet `15` completed all `384` updates (`100,663,296` actor rows in `13,745.224 s` on completion attempt c2).
- **Gameplay screen (`Closed, no nomination`):** Endpoint (update `384`) equal-mix paired H5000 mass vs bare `b8-s3` was `+8.09` (`CI95 [-18.66, +34.85]`; `scripted` `+26.32`, `frozen` `+2.85`, `mixed` `-4.90`), peaking at `+14.23` (`[-11.86, +40.32]`; `+41.16`/`+4.60`/`-3.06`) at update `320` after `-0.14`, `-7.99`, `-2.10`, `-3.81` at updates `64`/`128`/`192`/`256`.
- **Decision hedge:** Both endpoint and peak missed `>= +40`, so zero checkpoints crossed to the Mac strict gate; per closeout, this single-trajectory screen is insufficient evidence to advance, not a proof of zero benefit.
- **Engineering fixes:** Copying `12,443 B` slices cut the `160,000`-row buffer upper bound from `47.78 GB` (`298,632 B` E24 views) to `1.99 GB`; `background-sparse-v1` (`191` files, `925,149 B`) installed source in `0.025–0.035 s` vs `32.379 s` `git fetch`; float64 `log_softmax` stopped `+inf` KL; and `xz -6` shrank `1,327,620,931 B` (`1.33 GB`) snapshots `~151x` to `8,804,064 B` (`8.8 MB`).

### Why
Dividing a B8 advantage gap (`≈ 0.877`) by sharp temperature `T ≈ 0.00844` pushed legal float32 log-probabilities below `-103.97` (`<= -103.972084`), where float32 `exp()` underflows to zero and categorical KL returns `+inf`. For the `+8.09` plateau (`mixed` `-4.90`), deeper learning saturation had its cause not established in this single-trajectory screen.

### What to reuse / what not to repeat
- Compute categorical `log_softmax`, KL divergence, and entropy in `float64` whenever temperature scaling (`T ≈ 0.00844`) can push legal logits past float32 `exp()` underflow (`-103.97`).
- Clone observation rows before buffering so slices do not pin full batch tensors (`47.78 GB` vs `1.99 GB`), and compress multi-gigabyte resume snapshots with `xz -6` (`1.33 GB → 8.8 MB`) plus sparse source archives to stay under `300 s` / `30`-s staging caps.
- Do not continue single-seed frozen-input `b8-s3` PPO trajectories expecting large `mixed`-field gains after `100,663,296` transitions plateaued near `+8.09`.

### Full record
Original: `git show 90b3db94:docs/research/history/train01.md` | Primary sources: [TRAIN01.md](../../../research/sandbox_a_resident/TRAIN01.md), [TRAIN01_14.md](../../../research/sandbox_a_resident/TRAIN01_14.md), [TRAIN01_15.md](../../../research/sandbox_a_resident/TRAIN01_15.md)
