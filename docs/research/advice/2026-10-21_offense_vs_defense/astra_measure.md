**Keep defense on the delivery path; test incremental corpse harvesting in parallel before pivoting to offense.**
The quoted premise is materially wrong: mechanics v2, death drops, boost pellets and kill credit already exist in the benchmark and default served game.
In 36 measured games B8 received 332 kill credits; death drops supplied 39.71% of mass eaten, while boost occupied 0.1252% of acted frames.
That establishes the current source of growth, not the largest remaining causal improvement.
T3's roughly +92 is a development result pooled over two B8 checkpoints, not a defensive upper bound; its B8 s3 row improved by +122.812.
My next experiment would rank safe continuations by mass, with rescue search as the matched control and unchanged rules.

# Offense versus defense: measurement persona

2026-10-21 · independent Astra advice · source snapshot `61b8f751506dafff5fdee69d529309bd147d5232`.

This document contains a local descriptive diagnostic plus prospective advice. It is not a registration, a candidate comparison, a promotion result, or authorization to launch the proposed follow-ups. No sealed Round 0 directory was opened. No cloud job, tmux session, existing process, serving configuration, checkpoint or repository code was changed.

**What I actually measured**

**Measured: 36 terminal-hero H5000 games, 12 per mix.** All numbers below are descriptive sample values, not estimates of a new controller's improvement. Means are per game unless stated otherwise.

| Metric | Scripted | Frozen | Mixed |
|---|---:|---:|---:|
| Hero kills | 18.083 | 4.083 | 5.500 |
| Other deaths | 75.250 | 22.167 | 32.833 |
| Other deaths near hero head | 2.667 | 1.250 | 1.083 |
| Other deaths near hero body | 34.500 | 6.917 | 9.667 |
| Mass eaten: ambient | 505.917 | 576.333 | 663.750 |
| Mass eaten: death drops | 753.833 | 317.750 | 300.667 |
| Mass eaten: boost trails | 0.000 | 178.417 | 159.000 |
| Executed boost frames | 7.833 | 5.250 | 4.083 |
| Gross mass burned by boosting | 2.333 | 1.417 | 1.167 |
| Food eaten on boost frames | 0.833 | 0.667 | 0.083 |
| Net mass change on boost frames | -1.500 | -0.750 | -1.083 |
| Final live mass (dead = 0) | 805.000 | 711.333 | 983.250 |
| Final stored length (including death lengths) | 1258.417 | 1072.083 | 1123.250 |
| H5000 mass integral | 637.093 | 495.057 | 552.801 |
| Boost share of acted frames | 0.1730% | 0.1194% | 0.0854% |
| Death-food share of all mass eaten | 59.84% | 29.63% | 26.76% |
| Survived to H5000 | 7/12 | 7/12 | 10/12 |

Across all 36 games: **332 credited kills; 20,952 ambient, 16,467 death-drop and 4,049 boost-trail mass eaten**. Death drops supplied 39.71% of gross acquired mass. There were 206 boosted frames in 164,514 acted frames (0.1252%), burning 59 mass. Food eaten on boosted frames was 19, so the observed boost-frame balance was -40 mass. Acted frames include a terminal death frame; the denominator is not post-step living frames.

The economy is mix-dependent: death food dominates this scripted sample (59.84%), but supplies only 29.63%/26.76% against frozen/mixed opponents. Across the full sample, ambient food supplies 50.53% and boost trails 9.76%. Native corpse-class totals would combine deaths and trails into 49.47%, overstating death harvesting if read as corpse-only gain. B8's 4,049 trail-food pickups cannot mostly be its own: a per-game bound from its burn counts puts at least **4,018** of them as opponent-origin. This is an arithmetic lower bound, not donor tracking. B8 already benefits from other snakes' boosting while scarcely boosting itself.

The distinction matters: **gross food-source gain** counts +1 for each pellet eaten, including food eaten on a fatal transition. It is not a decomposition of mass integral or of surviving final mass. `final_mass` is zero after hero death; `final_stored_length` preserves length at death for the accounting identity. Every game checks:

`initial length + ambient gain + death-drop gain + boost-trail gain − boost burn = final stored length`.

“Near head” means the dying opponent's post-movement head is within Manhattan distance 10 cells (100 px) of the hero's post-movement head. “Near body” means within 2 cells of any hero segment. These counts can include credited kills; they must not be added to kills, and neither proximity nor kill credit establishes deliberate hunting. All opponent events stop at hero death or H5000. Opponents respawn, so kills/game can exceed five (`src/simd_env/eval_engine.py:524-548`).

**Method, provenance and checks**

- One CPU simulation process, invoked with `nice -n 10`, all five numerical thread variables set to 1, CPU device, shared project venv. The runtime reported niceness 15 because the shell already had positive niceness. A short source-check fixture was the only concurrent numerical process. AC and open lid were checked; each main launch had a 3,300-second alarm.
- Fixed descriptive sample: 12 worlds per scripted/frozen/mixed, H5000, hero slot 0 terminal, opponents respawning; 36 worlds, no outcome-driven extension. Seeds and ordered rosters were written before gameplay to `run02/plan.json`. Namespace: `astra-offense-measure-20261021/run02/diagnostic-only`. Concrete seeds had no hits under this checkout's `docs`, `research`, and `src` roots. External artifact history and sealed banks were deliberately not scanned; **this is not a fully collision-audited gate bank**. These worlds are consumed diagnostic worlds, unsuitable for future gating.
- The checkpoint was the serving registry's `b8_s3_t7_20261016.pth`, verified as SHA-256 `3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`; unset/blank `SNAKE_SERVE_CHECKPOINT` selects B8 s3 in repository defaults (`web/backend/served_checkpoint.py:70-102,394-402`). This is a code/default verification, not inspection of a running server's environment.
- World construction uses the same `bank_engine.planned_rosters` / `build_world` path used by T34 (`research/t34_pathb/bank.py:133-146`; `research/t3_rollout_labels_20261015/bank_engine.py:367-460`). Bare B8 uses rowwise CPU inference and the advisory action mask. Runtime hooks execute in their normal order (`bank_engine.py:487-514`). Checkpoint bytes for the frozen pool and deployment config are verified by `grid_h5000_identity._context` (`research/redesign_scope_20261007/grid_h5000_identity.py:258-281`).
- The historical benchmark's scripted mix is five greedy-food anchors; frozen rotates champion_a5, best_apex_fs, best_apex_stage1_fs and best_apex_pre_fs; mixed alternates checkpoint slots 1/3/5 and random-safe slots 2/4. This is the **benchmark mixed roster**, distinct from product Watch's FRP3+v8 opponents (`research/apex_safety_20260926/dev_screen.py:103-113,358-379`; `src/evaluation/strict_promotion.py:697-708`; `docs/research/final_run_20261019/SUCCESS_CONTRACT.md:36-40`).
- Actual configuration in every world: mechanics v2, six snakes, 1450×830, 10 px cells, initial/max ambient food 300, boost minimum length 5, one segment burned per three boost frames, opponent respawn timer 1. Full config is in every world JSON. The optional mass-integral column follows the evaluator's post-step alive mass, divided by all 5000 frames (`src/simd_env/eval_engine.py:1059-1061,1134`).
- Native food telemetry distinguishes ambient from corpse-class, but **combines death drops and boost trails** (`src/simd_env/batch_sim.py:251-259`; `src/simd_env/grid_sim.py:510-517,875-896`). Scratch-only instance wrappers tag actual successful additions with origin, preserve provenance on deduplication, remove FIFO evictions, and attribute the hero's first eaten traversed cell. Bulk death-drop bookkeeping executes in side state; the original dynamics method executes exactly once. The script asserts equality with native food/corpse sets after every bulk drop. No model inputs or simulator rules are changed.
- **36/36 mass-balance checks passed; instrumented versus uninstrumented replays matched every action and physical-state digest for the first world of each mix (12,999 total replayed frames).** Three worlds are a bounded instrumentation check, not a universal parity proof. Five synthetic source checks also passed: FIFO overflow, deduplicated trail/death overlap, boost-caused eviction, ambient/death/trail consumption against native class counters, and a consumed boost cell reinserted as death food. A read-only independent review found no blocking attribution issue. Successful run wall time, including three replay controls: **495.423 s (8.26 min)**. No outcome-based extension occurred.
- The first launch (`run01`) stopped at an incorrect v1 assumption before any gameplay. Its plan/log and `measure_v1_aborted.py` are retained. The corrected run asserts v2. This was a configuration correction before outcome collection, not a discarded outcome sample.

**The rules materially change the offense argument**

Corpses exist in both versions. V1 attempts every other physical body segment and adds it to the ambient class; v2 attempts every in-arena physical segment and adds it to the corpse class (`src/simd_env/grid_sim.py:875-896`; live equivalent `src/game/game_state.py:601-630`). “100%” describes attempted segment density, not guaranteed recoverable mass. Existing food deduplicates the insert; out-of-arena cells do not drop. Physical body length can differ from logical length after growth.

At v2, **death food and boost trails share a global FIFO cap of `max(10, floor(0.5 × max_food))` = 150 pellets here**. New pellets evict the oldest corpse-class pellets. The cap exists in both GridBatchSim and live FoodManager, using the same constants (`src/core/mechanics_constants.py:24-64`; `src/simd_env/grid_sim.py:435-507`; `src/game/food_manager.py:148-159`). A length-800 death therefore does not leave 800 edible mass on the board. An isolated corpse can leave at most 150 pellets immediately after its drop, less after overlap/competition. It cannot by itself double a hero already longer than 150 under these rules. Consumption is also at most one pellet per snake per frame, even on a two-cell boost (`src/simd_env/grid_sim.py:703-750`).

Measured death-drop accounting (includes the hero's terminal death in the “all” rows; source JSON records each death separately):

| Death subset | Events | Mean logical length | Mean attempted cells | Mean accepted inserts | Mean new pellets retained immediately after drop |
|---|---:|---:|---:|---:|---:|
| All deaths | 1575 | 49.222 | 49.180 | 48.424 | 39.362 |
| Opponent deaths only | 1563 | 42.274 | 42.232 | 41.476 | 38.512 |
| Deaths at logical length ≥800 | 9 | 1018.000 | 1018.000 | 1017.222 | 150.000 |

Largest observed dying length: **1454**, with 1454 physical segments, 1454 attempted in-arena cells, 1453 accepted inserts during the drop, and **150 newly inserted pellets still present immediately afterward**. Accepted inserts count transient additions that can be evicted later in the same drop; they are not all edible afterward. Retention here is per drop, before any later death in the same frame.

Boosting moves a second cell when eligible and charges one segment every three executed boost frames; the burned tail becomes a corpse-class pellet in v2 (`src/game/snake.py:155-177`; `src/simd_env/batch_sim.py:701-739`; `src/simd_env/grid_sim.py:510-517`). Gross burn and the **observed net balance on boosting frames** (all food eaten on those frames minus burn) are measured above. This is not causal boost ROI: a no-boost counterfactual was not run, delayed gains are not assigned to earlier boosting, and boost-origin food can come from any snake. “Boost only pays for kills/corpse races” is therefore not established; reaching ambient food or escaping death can also have value.

Kill credit is implemented: head-to-body collisions credit the body owner; v2 size-resolved head-ons credit the winner (`src/simd_env/grid_sim.py:831-873`). Credit is an event, not a direct transfer of the victim's full length. Growth comes from actually eating the resulting food. Large bodies can receive passive collision credit; this diagnostic does not distinguish encirclement, active cut-off, accidental contact, or scavenging after another opponent's kill.

Default serving also uses v2: `_mechanics_v2_enabled` defaults on, ego2s maps to the 61-D-world configuration, and the live game drops trail/corpse food (`web/backend/session.py:323-343,755-760`; `src/game/game_state.py:369-378,601-630`). The result should **not** be numerically transplanted to continuous-respawn product Watch: its roster and episode semantics differ, and I did not run a live Watch session or verify an existing server's overrides.

**Which parts of the quoted claim survive checking?**

| Claim | Finding |
|---|---|
| Mechanics v2 was never implemented | False. The measured runtime uses it, and live code implements it. No rules rewrite is needed to make death food or boost trails exist. |
| B8 has roughly zero kills / 89% boost abuse | Not supported for current B8; see the new measurements. The older “whole lineage kills are ~0” note is `docs/h100_training_recipe.md:154-156`. The question's `docs/slither_training_log_20260615.md` is absent in this checkout. I could not recover the 89% claim. T3 CTRL's documented final greedy boost rate was **0.0899%**, a different diagnostic (`docs/research/t3_screen_result_2026-10-15.md:15-19`). |
| Current T34 is principally defensive | Correct as an action-selection description. `teacher_v2._rank_outcomes` keeps a surviving bare action first, then surviving alternatives by Q, then dead actions by alive steps; recorded mass is not the ordering key (`research/t34_exit_b8/teacher_v2.py:253-287`). Path B similarly keeps bare when it reaches the horizon (`research/t34_pathb/engine_v2.py:941-966`). |
| Its upside is capped at roughly +92 | False. The dense development result is **700.005−608.054 = +91.951 pooled over s0 and s3**. The served **s3 row is 686.662−563.850 = +122.812**. These are measured effects of one teacher on reused scripted worlds, not optimal-defense ceilings (`docs/research/t3_ceiling_test_2026-10-15.md:113-129`; `docs/research/specs/2026-10-20_t34_nogo_branch_plan.md:30-35`). |
| B8 has a late enclosure problem | Supported by the historical census, with scope limits. B8 s3 had 15/32 deaths, median death frame 3283 and median death length 972. Pooled hazard increased with late-game frame/crowding, not monotonically with length; 19/49 late enclosure onsets had a one-step escape (`docs/research/panel_mac_answers_2026-10-14.md:79-119`). That is not evidence that killing more is the larger remaining lever. |
| Exact search can exploit opponents perfectly | Exact continuation is useful against these fixed, fully captured simulator policies/RNG states. It does not establish optimal search, accessible future randomness in deployment, robustness to changed opponents, or delivery before a live deadline. Preserve the campaign's information-access, stale-result and paced-efficacy requirements (`SUCCESS_CONTRACT.md:50-54`; `CAMPAIGN_LOG.md:84-90`, both under `docs/research/final_run_20261019/`). |

**What I would do next, in order — prospective proposals**

My belief is that **preserving a large B8's harvesting lifetime is the fastest credible delivery route**, because there is already a positive rescue mechanism result. That is a judgment about evidence and time to delivery, not a measured defense-versus-offense ranking. The source census makes corpse harvesting a real mechanism worth testing; it does not reveal how much of it B8 is leaving uncollected. Conversely, many kills already occurring does not prove offensive opportunity is exhausted.

| Priority / dependency | Work and decision enabled | Bounds and proposed stopping rule |
|---|---|---|
| 0 — completed here | Correct the factual premise; keep these diagnostic worlds out of later gates. | Local only; measured receipt below. No deployment change. |
| 1 — alongside the adopted delivery path | Continue the existing Path B deliverability/retention work under its own registration. Do not substitute these data for any registered outcome. | Existing campaign caps and timing bars; this document adds no spend or launch. Direction reset explicitly retains paced evaluation, sealed outcomes and Mac strict/serving qualification (`CAMPAIGN_LOG.md:84-90`). |
| 2 — new mechanism probe, before training | Compare the exact bare-first, then Q-ranked-survivors ordering with **mass-ranked surviving branches** at 96 prospectively sampled roots, 32/mix, from separate fresh trajectories. Include roots where bare already survives, so harvesting can be tested rather than only rescue. Record all legal first-action continuations (at most six), H300, identical B8 continuation and cloned opponent/RNG state. Measure mass integral, survival, deaths caused, and actually eaten source food. | At most 172,800 branch frames plus a collection cap of 180,000 frames; one L4 supervisor lane, ≤30 min and ≤1 unit including setup/export, with 0.2 unit reserved. At the owner-supplied planning rate 2 u/h, 0.5 h = 1 u; actual debit also binds. Stop at the first cap, incomplete stays incomplete. Not launched here. |
| 3 — only if probe finds repeatable safe harvest advantage | Fresh paired H5000 study of bare B8, the exact rescue controller, and rescue plus mass-ranked harvesting; equal trigger/branch budgets for the two search arms. Proposed initial bank: 16 worlds/mix (48 paired worlds, 144 episodes). Keep mechanics, checkpoint and rosters fixed. Then continuous-respawn mixed Watch6 under real deadlines if it advances. | One L4 lane, ≤4 runtime-hours/8 units total including setup/export and 20% reserve. A separate disjoint timing fixture must show the full frozen bank fits before admission; otherwise reject this envelope before data. No post-look extension. Mac timing/strict/serving work remains separate. |

For step 2, a proposed **mechanism-admission** rule is: INVALID on provenance/parity/nonfinite/illegal failures; NO-GO for this one-action H300 mechanism if fewer than 12/96 roots yield at least +10 mean branch mass with unchanged H300 survival relative to the exact bare-first/Q-ranked choice; otherwise admit step 3. The 12/96 and +10 are prospective engineering thresholds, not confidence claims or proof of H5000 improvement. Correlated roots do not become independent worlds. Failure would not close multi-step encirclement or longer-horizon offense.

For step 3, freeze the full statistic, information access and guard thresholds before collecting data. Use paired worlds, stratified by mix, and an equal-mix H5000 mass contrast. A proposed ordered decision is INVALID/INCOMPLETE first; KILL for any active survival-guard failure or nonpositive upper 90% bound for the increment over rescue; GO only if the one-sided lower 90% bound for that increment is positive, mean increment ≥+20, and the candidate also meets the active campaign bars versus B8; otherwise INCONCLUSIVE. The product and release decision still requires the contract's paired Watch metrics, real paced delivery, Mac strict gate and owner release OK (`SUCCESS_CONTRACT.md:27-57`; current amendments and P7 govern, not this sketch).

The current survival-first ranking cannot test this hypothesis just by logging its mass column. A later implementer must add the separate mass-ranking arm and bind trigger schedule, branch budget and tie rule as candidate identity. Use full-horizon mass with dead frames zero; do not pay a kill bonus or replace mass with kills. Better harvesting may occur without extra kills, and extra kills may merely redistribute existing corpse food.

First proposed new-wave envelope: ≤9 units, 1.8 reserved inside that ceiling, all conditional on registration and measured admission. This is a ceiling, **not a runtime prediction**: today's instrumented bare-rollout rate does not price search branches or L4 throughput. No RunPod, rule change, reward sweep or distillation run is justified by the census. If the unchanged-rule harvest arm produces a material paired gain over rescue that survives paced mixed Watch6, I would promote offense/harvesting to the critical path. If it only increases kills, loses survival, or loses gain at real deadlines, I would retain rescue. Changing the 150-pellet cap is a separate product/rules decision and requires a new B8 baseline; it cannot establish a win on the existing benchmark.

**Per-game measurements and reproducibility receipt**

S/F/M = scripted/frozen/mixed. H/B = near-head/near-body other deaths. Boost tuple = executed frames / gross burn / net mass change during boosted frames. Final tuple = final live mass / stored length at horizon or death. Seeds fully identify each row together with the ordered roster in `plan.json`.

| ID/mix | Seed | Acted frames | Kills | Near H/B | Ambient | Death food | Trail food | Boost frames/burn/net | Final live/stored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 00/S | 332506711 | 5000 | 21 | 2/49 | 576 | 833 | 0 | 3/1/-1 | 1409/1409 |
| 01/F | 1750569712 | 2999 | 2 | 1/4 | 414 | 274 | 74 | 5/1/-1 | 0/762 |
| 02/M | 1503521921 | 5000 | 1 | 0/4 | 744 | 292 | 101 | 0/0/+0 | 1138/1138 |
| 03/S | 1670748077 | 5000 | 18 | 1/37 | 582 | 812 | 0 | 3/1/-1 | 1394/1394 |
| 04/F | 846129436 | 3742 | 4 | 2/4 | 424 | 210 | 158 | 15/5/-3 | 0/788 |
| 05/M | 3540007376 | 5000 | 6 | 1/8 | 650 | 294 | 194 | 0/0/+0 | 1139/1139 |
| 06/S | 2852538658 | 3197 | 14 | 3/21 | 354 | 712 | 0 | 20/6/-5 | 0/1061 |
| 07/F | 2242775669 | 5000 | 4 | 1/9 | 589 | 461 | 197 | 0/0/+0 | 1248/1248 |
| 08/M | 2333935125 | 5000 | 5 | 1/11 | 724 | 311 | 147 | 3/1/-1 | 1182/1182 |
| 09/S | 379562576 | 3132 | 19 | 3/32 | 421 | 513 | 0 | 1/0/+0 | 0/935 |
| 10/F | 3715323823 | 3567 | 2 | 1/5 | 534 | 194 | 105 | 4/1/+0 | 0/833 |
| 11/M | 3528443450 | 5000 | 5 | 1/11 | 676 | 207 | 171 | 4/1/+0 | 1054/1054 |
| 12/S | 3668494020 | 4156 | 14 | 1/26 | 469 | 589 | 0 | 8/2/-1 | 0/1057 |
| 13/F | 1208674173 | 5000 | 6 | 2/12 | 648 | 336 | 213 | 2/0/+0 | 1198/1198 |
| 14/M | 1167525437 | 5000 | 6 | 0/9 | 738 | 302 | 154 | 1/0/+0 | 1195/1195 |
| 15/S | 3722153341 | 4800 | 15 | 5/32 | 538 | 917 | 0 | 7/2/+0 | 0/1454 |
| 16/F | 82983394 | 5000 | 6 | 0/10 | 696 | 341 | 200 | 0/0/+0 | 1238/1238 |
| 17/M | 2057243203 | 5000 | 6 | 1/8 | 698 | 314 | 212 | 1/0/+0 | 1225/1225 |
| 18/S | 1197459185 | 5000 | 26 | 4/43 | 500 | 811 | 0 | 6/2/-1 | 1310/1310 |
| 19/F | 428831390 | 5000 | 2 | 0/6 | 635 | 414 | 214 | 2/0/+0 | 1264/1264 |
| 20/M | 1047100385 | 5000 | 8 | 0/11 | 756 | 338 | 145 | 1/0/+0 | 1240/1240 |
| 21/S | 756760947 | 5000 | 18 | 2/41 | 537 | 624 | 0 | 34/11/-7 | 1151/1151 |
| 22/F | 3532892949 | 5000 | 4 | 1/5 | 730 | 171 | 200 | 2/0/+0 | 1102/1102 |
| 23/M | 2090538996 | 5000 | 9 | 3/12 | 643 | 462 | 141 | 6/2/-2 | 1245/1245 |
| 24/S | 1887256897 | 5000 | 16 | 1/32 | 617 | 841 | 0 | 1/0/+0 | 1459/1459 |
| 25/F | 2262487131 | 5000 | 8 | 2/10 | 620 | 352 | 273 | 1/0/+0 | 1246/1246 |
| 26/M | 667289486 | 3234 | 2 | 3/4 | 509 | 147 | 84 | 10/3/-3 | 0/738 |
| 27/S | 3785949982 | 5000 | 15 | 3/29 | 426 | 1005 | 0 | 3/1/-1 | 1431/1431 |
| 28/F | 3579960408 | 4239 | 5 | 1/9 | 531 | 444 | 144 | 31/10/-5 | 0/1110 |
| 29/M | 3983358801 | 4174 | 4 | 0/10 | 556 | 203 | 189 | 23/7/-7 | 0/942 |
| 30/S | 3847368358 | 4039 | 13 | 2/24 | 465 | 470 | 0 | 6/2/-2 | 0/934 |
| 31/F | 3223081272 | 5000 | 5 | 3/8 | 641 | 383 | 215 | 0/0/+0 | 1240/1240 |
| 32/M | 1658449498 | 5000 | 7 | 3/17 | 593 | 412 | 247 | 0/0/+0 | 1253/1253 |
| 33/S | 2203117564 | 5000 | 28 | 5/48 | 586 | 919 | 0 | 2/0/+1 | 1506/1506 |
| 34/F | 588691372 | 3235 | 1 | 1/1 | 454 | 233 | 148 | 1/0/+0 | 0/836 |
| 35/M | 6677491 | 5000 | 7 | 0/11 | 678 | 326 | 123 | 0/0/+0 | 1128/1128 |

Artifacts (outside Git):

- Main create-only receipt: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/astra-offense/measure/run02/receipt.json` — SHA-256 `005f44530d00009771193b88db03fb58022ddf7a9e38be1b561efbdba3c89e91`.
- Accounting/hash check: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/astra-offense/measure/run02/audit.json` — SHA-256 `ca363ecd8d5fde89f1899049a65b6badf917fc88a05754adf416dcce3df51382`.
- Per-game CSV: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/astra-offense/measure/run02/per-game.csv` — SHA-256 `588605c0d3daadfecc8e3544a45610792226bee372b64c57ae5472d0a8a10d06`; full per-game/death-event JSONs are `world-00.json` through `world-35.json` in the same directory, hash-bound by the main receipt.
- Instrumentation: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/astra-offense/measure/measure.py` — SHA-256 `77ae10740a6d0bd158a49cb4a86ec2dae16f46e9829ecc157dccd6ea9200da9c`.
- Synthetic attribution tests: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/astra-offense/measure/check_sources.py`, receipt `/Users/josenunez/Projects/ml/snake-dqn-artifacts/astra-offense/measure/source-checks.json` — receipt SHA-256 `fd956dbcff9dc2182a8dfdb6aa3676ee009b7855d5939134c5ffd8326b99bdf2`.
- Stdout/progress: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/astra-offense/measure/run02.log`; pre-game failed configuration assertion retained in `run01.log` and `run01/plan.json`.

Exact measurement command (already completed; outputs are create-only, so do not rerun into `run02`):

```bash
nice -n 10 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
  PYTHONDONTWRITEBYTECODE=1 \
  NUMBA_CACHE_DIR=/Users/josenunez/Projects/ml/snake-dqn-artifacts/astra-offense/measure/numba-cache \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
  /Users/josenunez/Projects/ml/snake-dqn-artifacts/astra-offense/measure/measure.py
```

The source-check command used the same interpreter, niceness and environment, replacing `measure.py` with `check_sources.py`. `finalize.py` verifies all receipt hashes, sample completeness, replay checks and mass balances, then produces the CSV and this document's tables. Research outputs remain local; only this advice document is committed/pushed to the advisor branch. Nothing is merged to main.
