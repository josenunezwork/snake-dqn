# Native Solo Behavioral Cloning, Spacious Teacher, and Learner-State Aggregation (`solo-native-imitation`, Series `X`–`AM`)

## Question and Goal

Can the unchanged `RasterDuelingNetwork` (`raster31v3` input) learn reliable 1-snake food-seeking and survival in the native `145 x 83` simulator (source frozen at `56e0e92434ae510a84ebaf6f58cf41c3e4421b04` for the first BC screen) via behavioral cloning (`BC`) of scripted teachers (`GreedyFood` and flood-fill `SpaceTeacher`) and learner-state aggregation across 64- to 256-frame horizons?

## Timeline of Versions and Steps

All 16 studies in this September 13–14, 2026 series have artifacts under `/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/`; their per-study READMEs were removed in consolidation (see git c4c4ab41, `docs/experiments/<slug>/README.md`). Letter codes are given only where the source READMEs use them; the held-out `W` benchmark (`6,144` teacher-state rows, eight world clusters of `12` heading/ray poses) predates this lineage.

| Date | Code / Slug | Horizon & Design | Key Result | Verdict |
|---|---|---|---|---|
| 2026-09-13 | **`X`** (`solo_native_food_bc`) | 64-frame BC, `3,072` rows from `48` lanes across four training worlds, `3` seeds x `500` updates, Adam lr `0.0005` | Train fit `100%`; held-out fit `83.84%` / `84.64%` / `85.77%` (gate `>= 95%`); final food `9.125` / `8.990` / `9.573` vs W teacher `12.635`; only seed `2026092703` passed gameplay. | **FAIL (joint 0/3)** |
| 2026-09-13 | **`Y`** (`solo_native_food_diversity`) | Same `3,072` rows reallocated across `16` training worlds | Train fit `100%`; held-out fit `85.319%` / `86.393%` / `85.368%`; food `9.844` / `10.198` / `9.260`; seed `2026092801` update-250 gameplay missing. | **`INCOMPLETE`** |
| 2026-09-13 | — (`solo_food_observation_probe`) | Observation-only food-distance rule on saved examples (no training) | Recovered the teacher action on all **`12,288`** saved examples (X train `3,072`, Y train `3,072`, held-out `6,144`); the six learned models' `836`–`993` held-out errors all fell on decoder-correct rows. | **OBSERVATION RECOVERS TEACHER ACTION** |
| 2026-09-13 | **`AA`** (`solo_native_food_input_ablation`) | Unchanged network restricted to food-geometry inputs, `3` fresh seeds | Train fit `100%`; held-out fit `88.623%` / `87.402%` / `87.093%` (failed); gameplay pass 3/3 (food `11.115` / `10.448` / `10.812`, `96/96` alive). | **FAIL (`heldout_fit_fail`)** |
| 2026-09-13 | — (`solo_native_food_fresh_confirmation`) | AA's three fixed policies on eight untouched 64-frame worlds | All three passed all six gameplay conditions (food `10.20` / `10.83` / `10.54`; `96`, `94`, `96` of `96` alive). | **`FRESH_NATIVE64_BEHAVIOR_CONFIRMED`** |
| 2026-09-14 | — (`solo_native_food_128`) | Same policies, 128 frames, eight new worlds | All six gates passed (food `21.04` / `19.01` / `19.60`; `90`, `91`, `91` of `96` alive). | **`NATIVE128_BEHAVIOR_CONFIRMED`** |
| 2026-09-14 | **`AD`** (`solo_native_food_256`) | Same policies, 256 frames | Food `32.72` / `36.90` / `36.23` passed the food/cell/paired/retention checks, but endpoint alive only `55`, `79`, `66` of `96`; every seed failed time-alive and endpoint-alive. | **`NATIVE256_BEHAVIOR_NOT_RELIABLE`** |
| 2026-09-14 | — (`solo_native_death_replay`) | Exact replay of the 256-frame screen's recorded deaths | All **`100`** deaths (teacher `12`; AA seeds `41` / `17` / `30`) were legal fallbacks taken when the advisory-safe set was empty; `0` advisory-safe fatal actions. | **NO ADVISORY-SAFE FATAL ACTION** |
| 2026-09-14 | — (`solo_native_food_space_teacher`) | Greedy food + existing `SnakeStateMixin._get_free_space_features` spacious veto, eight fresh 256-frame worlds | Spacious teacher `96/96` alive, food `46.375` vs original greedy teacher `67/96` (`29` self-deaths), `41.729`; food `+4.6458`, 95% CI `[0.6910, 8.6007]`. | **PASSED ALL ELIGIBILITY CONDITIONS** (teacher calibration) |
| 2026-09-14 | **`AG`** (`solo_native_space_bc`) | 256-frame BC of the spacious teacher (`3,072` rows, `122` disagreement rows; `3` seeds) | Train fit `100%`; held-out natural fit `85.4%`–`86.8%`, disagreement fit `25.0%`–`58.3%`; alive **`71/96`, `81/96`, `56/96`**; food `30.69`–`35.80` vs teacher `47.64`; seed `2026093001` midpoint evaluation stopped at the watchdog. | **`INCOMPLETE / INCONCLUSIVE`** |
| 2026-09-14 | **`AH`** (`solo_native_space_policy_replay`) | Spacious-teacher labels on AG's three completed final trajectories | Students missed **`73.5%`–`86.5%`** of behavior-changing veto rows before `F`; `60%`–`70%` of death lanes had a missed veto in the preceding 16 decisions (veto error `97.3%`–`100%` vs ordinary `21.8%`–`36.7%`). | **`ADVANCE_TARGETED_AUGMENTATION`** |
| 2026-09-14 | **`AI`** (`solo_native_space_aggregation`) | One learner-state aggregation round (`1,536` original + `1,536` learner-state rows vs control on AG's `3,072` rows) | Mixture beat control in all `3` pairs (endpoint **`55→70`, `66→73`, `58→71` of `96`**; food `36.88` / `38.98` / `35.16`) but missed the absolute survival gate (`>= 87/96`); endpoint mean `+12.15 pp`, 95% CI `[-1.44, +25.75]`; held-out disagreement fit `44.7%`–`51.1%`. | **FAIL (reliability gate)** |
| 2026-09-14 | **`AJ`** (`solo_native_space_refresh`) | Fixed refresh replacing AI's learner half with `1,536` newly captured learner-state rows | Endpoint beat control in all `3` pairs (**`61→71`, `65→72`, `63→65` of `96`**; `+6.60 pp`, 95% CI `[-7.77, +20.96]`); food `36.88` / `35.77` / `34.50` vs teacher `43.68`; held-out disagreement fit `20.0%`–`44.0%`. | **FAIL (reliability gate)** |
| 2026-09-14 | **`AK`** (`solo_space_food_attribution`) | Saved-artifact decoder audit (`6,167` held-out + `3 x 1,536` learner rows = `10,775`) | Decoder recovered the old greedy action on **`10,775 / 10,775`** rows; `3,554` exact duplicates, none conflicting; all `825` / `922` / `863` ordinary held-out errors were decoder-recoverable. | **`ERRORS_ON_RECOVERABLE_FOOD_DECISIONS`** |
| 2026-09-14 | **`AL`** (`solo_native_food_view`) | Training-only food-view auxiliary, `3` fresh seeds x `500` updates | Held-out natural fit near `83%`–`84%`; food-view endpoint survival worse than control in all `3` seeds (**`50 vs 61`, `73 vs 77`, `64 vs 77` of `96`**; mean `-9.72 pp`, 95% CI `[-24.62, +5.17]`, interval includes zero). | **FAIL (non-regression gates)** |
| 2026-09-14 | **`AM`** (`solo_food_cross_view`) | Food-only vs full Q inference on AL's saved step-500 checkpoints (`11/12` archives recovered) | On the two available food-view seeds, food-only raised ordinary held-out accuracy **`+2.25 pp`** (`83.76% → 86.01%`) and **`+2.13 pp`** (`84.27% → 86.40%`); the README states this neither establishes a three-seed result nor explains survival. | **`INCOMPLETE_REQUIRED_COUNTERFACTUAL_MISSING`** |

## Final Verdict and Key Numbers

1. **Short-horizon (64–128 frame) native behavior was reachable, held-out teacher fit was not:** No BC arm reached the held-out fit gate (best `88.623%` in AA vs `>= 95%`), but AA's food-geometry policies passed every gameplay gate on fresh 64-frame and 128-frame worlds (`90`–`96` of `96` alive).
2. **Food direction is recoverable from the observation:** a deterministic decoder matched the teacher on `12,288 / 12,288` saved examples and on `10,775 / 10,775` AK rows, and learned errors fell on decoder-recoverable rows; the READMEs explicitly do not claim this explains poor generalization or deaths.
3. **256-frame survival was the bottleneck:** AD's policies collected food but ended with `55`–`79 / 96` alive; all `100` replayed deaths occurred with no advisory-safe action left; the spacious teacher fixed the teacher side (`96/96`), but imitating it (AG) and two aggregation rounds (AI, AJ) improved survival over matched controls only directionally (CIs include zero) and never met the `87/96` gate; a food-view auxiliary (AL) did not help.

## What Was Learned

- **Deaths are decided before the last safe move disappears:** all `100` replayed 256-frame deaths had an empty advisory-safe set, and AG students frequently missed spacious-teacher turns while a safe action still existed (AH); the sources treat this as an association, not proof that an alternate action would rescue a game.
- **Rare veto decisions did not generalize:** with learner-state aggregation (AI, AJ) and a food-view auxiliary (AL), held-out natural fit stayed around `84%`–`87%` and held-out disagreement fit `20%`–`51%`. (The later `ego2s-bv8` redesign is described in [`docs/notes/redesign-and-b8.md`](../../notes/redesign-and-b8.md); these READMEs do not themselves claim it as the motivation.)

## Why It Closed / Current Status

**Closed diagnostic lineage (2026-09-14).** Apex remained the incumbent throughout; none of these studies was promotion evidence. The solo-food series continued with later codes (e.g. `AO` PQN warm-start, `AP` behavior anchor) catalogued in [`docs/experiments/SOLO_FOOD_SERIES.md`](../../experiments/SOLO_FOOD_SERIES.md).

## Keep Files in This Lineage

All 16 per-experiment `README.md` files in this lineage had action `CONSOLIDATE` and are replaced by this record. Related kept overview documents:
- [`docs/experiments/README.md`](../../experiments/README.md) — Index of the September 2026 `docs/experiments/` series.
- [`docs/experiments/SOLO_FOOD_SERIES.md`](../../experiments/SOLO_FOOD_SERIES.md) — Catalog of the `solo_food_*` / `solo_native_*` studies.
- [`docs/notes/redesign-and-b8.md`](../../notes/redesign-and-b8.md) — Compendium chapter connecting the September raster/imitation findings to the October `ego2s-bv8` (`B8`) redesign.
