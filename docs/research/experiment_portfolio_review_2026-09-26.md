# Snake RL research portfolio review, Sep 6–26 2026 (192 experiments)

> Generated 2026-09-26 by a multi-agent review of the 192 experiment write-ups (8 family readers, a synthesis, then a skeptical correction pass). Numbers come from the write-ups, not from re-derived raw data. Treat derived tallies as approximate.

Sources: the eight family reports and the strict-pilot README. Claims attributed to the July redesign blueprint (the Phase-0 plan and throughput targets) come from the blueprint, not from the family reports. All other numbers come from the write-ups. Where I derived a number (verdict tallies, compute sums, rough t-values, sign tests), I say so.

---

## 1. Bottom line

- **Nothing reached the deployed agent.** Apex/vector61 is still the incumbent and no default changed. In this period, the only learned challenger played against Apex at the deployment profile was 6102. It lost in all three mixes at H5000:
  - mean mass −30.2, −66.4 and −34.9;
  - 5 of 16 world wins in each mix.
  
  That pilot is unaudited, with n=16 per mix.
- **The most important finding is about measurement.** At H5000:
  - The paired per-world standard deviation (SD) of the mass difference is 56–118, while a 10% effect is only 3.9–7.3.
  - A strict gate at that effect size would need 552–8,676 worlds per mix.
  - The per-world noise-to-mean ratio was 1–3.
  - Deployment-profile power was first measured on the program's last day (Sep 26). A pre-run feasibility estimate from proxies had already predicted a stop.
  - Per the blueprint, not the family reports, this gate analysis was meant to come first.
- **The solidly replicated results are failure modes, not improvements:**
  - PQN is unstable across seeds, and single interventions reverse on a fresh seed.
  - A naive handoff from behavior cloning (BC) to TD fine-tuning quickly destroys the imitated skill. It replicated in the raster-PQN and Apex-vector learners. Anchoring, lower LR or continued dose mitigated it, and the best RL lineages (CW, 6102) descend from a BC→PQN handoff.
  - In solo food imitation, training fit is perfect but held-out fit plateaus at about 84–90%, even though the needed information is recoverable from the observation.
  - Solo deaths at every horizon are almost all self-traps that finish after the safe-action set is already empty.
- **The strongest positive result is structural safety.** A tiny learned safety head on privileged reachability features (BL→BM→BN→BO) raised solo survival to 96/96 at H128, H256 and H512. At H512 the parent had 30/21/21 of 96. Food also rose.
  - The capped flood-fill oracle it imitates gave 96/96 at H128 (BD, BE).
  - A flood-fill-vetoed teacher gave 96/96 at H256 (AF).
  - None of this was tested with opponents, at H5000, or on top of Apex.
- **The bookkeeping is rigorous but the experiments are underpowered.**
  - Most studies used 2–3 seeds, often correlated lineages, with 4–32 worlds or single 48–96-case banks.
  - Gates were all-seed conjunctions.
  - Most of the 74 "negative" verdicts mean "not established", not "no effect". About a dozen do show significant harm.
- **Process cost evidently exceeded science compute, though elapsed time was not measured.**
  - Reported science time is roughly 31 h over 20 days, summed over six of the eight families.
  - 13 studies were classified invalid, and 12 of those failed on tooling or rules, not science.
  - Qualifying a serving bridge for 6102 took five admissions over about 1.5 days. A roughly 1-hour pilot then retired the candidate.
- **The program narrowed to proxies, and where transfer was tested it failed.** It moved to 16-frame microbenchmarks, H4 scripted encounters and solo arenas:
  - The solo lineage lost food and survival when one opponent was added (CX).
  - 6102's +38 mass over its parent at H3000 still left it below Apex at H5000 in the unaudited pilot.
  - The solo safety results were never tested at deployment.

---

## 2. Timeline and arc

| Family | #exp | Dates | Central question | Headline result |
|---|---:|---|---|---|
| foundations_and_early_screens | 23 | 09-06 → ~09-21 | Can corrected raster31v3/PQN learn reliably (fixed opponents, then solo), and does any single knob help? | Every learning screen ended NOT_ADVANCED, INCONCLUSIVE or NOT_ESTABLISHED. Seed variance exceeds intervention effects. Degenerate right-turn period-4 loops. |
| solo_food_and_native (0913–14) | 30 | 09-13 → 09-14 | Can the raster net learn solo food by PQN, and failing that by BC, and extend to survival? | PQN unstable even at H16 (1/3–2/3 seeds). BC collects food to H128 (3/3). H256 fails on self-trap deaths. |
| solo_food (0919–20) | 28 | 09-19 → 09-20 | Can a learned add-on fix the frozen food parent's self-traps? | Shared safety head: 96/96 survival at H128/H256/H512 in 3/3 seeds, with food up at H256/H512 (H512 CI about +18 to +29). Later food-routing fixes (BR–BU) all failed. |
| solo_food_pqn | 35 | 09-14, 09-20 → 09-21 | Can PQN warm-started from BC keep and extend solo food/survival, then handle one opponent? | Unanchored handoff collapses. Cumulative lineage passes H768 solo 3/3 (CW). No single intervention established. One opponent hurts 3/3. |
| controlled_encounter_a | 21 | 09-21 → 09-22 | Can a learner master scripted H4/H16 threat encounters by imitation, then native TD? | Teacher labels alias on hidden state. Held fit about 70–96%. One package gain (global teacher, food) fails absolute gates. TD fixes all fail. |
| controlled_encounter_b | 21 | 09-21 → 09-22 | Same bank: label design, breadth, regularization, native-TD adaptation | Survival-only gate is gameable by fixed rules. Broad data helps food and held fit, not survival. Native TD at lr 5e-4 degrades the parent. Line ENDED. |
| apex | 25 | 09-22 → 09-24 | Can the canonical Apex learner learn and extend solo food to H256, and what fixes residual failures? | TD from scratch learns H8/H16 3/3. BC warm start collapses. Second-food loops. H256 self-traps. Width/breadth: no reliable benefit. |
| task_aligned | 9 | 09-24 → 09-26 | Does a mixed-experience PQN continuation beat its parent, and can 6102 beat Apex at H5000? | 6102 +38 over parent at H3000 (the only lineage passing both gain and solo retention). Further continuation hurts solo mass. Bridge qualification needed 5 attempts. 6102 loses to Apex; line retired. |

**How the questions evolved.**

1. **Sep 6–13: full problem, weak learner.**
   - The program started aimed at the real goal: make the raster31v3/PQN stack good enough to challenge Apex.
   - The 09-06 audit found bounded contract counterexamples. The 09-12 phase witness found a trainer-vs-serving decision-phase mismatch, which was repaired opt-in.
   - Fixed-opponent PQN then produced degenerate policies: 99.9% right turns, period-4 loops, food 543→5.
   - Several single-knob screens (living-mass, reset heading, LR, solo replication) gave one good seed and one reversed seed. Others (scripted advice, ambient bonus at 50k) lost outright.
   - The program shrank the problem and went solo.
2. **Sep 13–14: solo, and a further retreat to imitation.**
   - PQN on a 16-frame, one-pellet task typically learned and then collapsed (1/3–2/3 seeds). BC solved it, so the program switched to imitation.
   - Food generalized to H128, but survival broke at H256 on self-traps.
   - That led to a safety teacher (flood-fill veto), DAgger rounds and representation probes. The probes pointed to generalization failure, not missing information.
3. **Sep 19–20: the one clean construction result.**
   - The program froze the food parent and learned only a safety correction.
   - The CNN residual memorized its training data but did not generalize.
   - A tiny shared head on privileged reachability features, with a ranking hinge and 1,500 updates, passed at every tested horizon up to H512.
   - Follow-up attempts to fix the parent's remaining food-routing errors all failed.
4. **Sep 20–21: back to RL on top of BC (solo_food_pqn).**
   - The handoff collapse was contained with a CE anchor and Q-scaling. The anchor was then released, and unchanged PQN recovered after a dip.
   - The lineage accumulated about 5.6k updates and several unestablished changes (half-MSE, length scalars) and reached H768 solo reliability. Which factor drove this is untested.
   - Adding one GreedyFood opponent broke retention.
5. **Sep 21–22: the opponent problem cut down to 4-frame encounters.**
   - Much of the study count went to teacher-label pathologies: aliasing, hidden future RNG, horizon regret and null label interventions.
   - Native TD traded crossing losses for frontal rescues.
   - Both encounter lines stopped, with no next step selected.
6. **Sep 22–24: pivot to the incumbent's own learner in solo.**
   - Findings paralleled the raster work: BC→TD collapse, period-4 loops (here after the first food), and self-traps at H256 despite free-space features.
   - In 14 of 15 pre-death states, freshly trained small-arena Apex models chose a 0.03125 free-space action when a 1.0 alternative was available.
7. **Sep 24–26: jump back to the deployment problem.**
   - The CZ lineages were continued with mixed opponents. 6102 was nominated on H3000 gains over its parent in an INCONCLUSIVE package screen.
   - Bridge qualification then ran on the full governance stack.
   - An unaudited strict pilot showed that 6102 loses and that the gate cannot resolve 10% effects.

**Why the arc looks like this:** most families ended by nominating a narrower diagnostic or sub-task. That produced a funnel toward ever-smaller proxies (H4, H16, solo). The final leap back to H5000 tested a candidate chosen by gains over a parent, not by distance to the incumbent.

---

## 3. What is actually established (strongest evidence first)

1. **Self-trap deaths dominate solo mortality.** Every solo death census found self-collision (one wall death in CT). Wherever a census checked the fatal move, it was a fallback taken when no advisory-safe action remained, never a move the mask called safe:
   - Raster BC: AE 100/100 at H256, AR 38/38 at H128.
   - PQN lineage: CE 54/54, CJ 607/607, CT 97/98 self-collision (1 wall), all fallback. CQ found all self-collision; fallback was not reported.
   - Apex: terminal_mask 15/15.
   
   CU examined 98 H512 windows. In every one, the safe set first emptied at the fatal step, and 63 of 98 had multiple safe directions one step earlier.
   - Evidence: audited exact replays, several learners, many horizons.
   - Caveat: with an opponent, deaths are mostly head collisions with a non-empty advisory set (CY: 70/76).
2. **Flood-fill-style safety fixes solo survival without costing food.**
   - SpaceTeacher at H256 (AF): 96/96 vs 67/96, food +4.65 [0.69, 8.60].
   - Parent-consistent oracle at H128: BD (reused worlds) and BE (fresh worlds) both 96/96 in 3/3, with positive mass CIs on fresh worlds. BD and BE were not independently audited.
   - Learned shared head: BL reached 97–99% held intervention fit. BN at H256: 96/96 vs parent 60/59/69, with positive food CIs. BO at H512: 96/96 vs 30/21/21, food +18 to +29. At H128 the food CIs mostly include zero.
   
   This is strong within scope: solo, food reachable, maximum snake length about 105, privileged features. BM's gameplay archives are byte-identical to BL's, so BM confirms determinism, not generalization. No new-training-seed gameplay confirmation exists.
3. **A naive BC→TD handoff destroys the imitated skill quickly.**
   - Raster PQN: AO 3/3 with wholly negative CIs; AP native arm 3/3; BV and BX controls 3/3; CA no-CE 2/3.
   - Apex: canonical_retention, where 3/3 BC parents collapsed (held food 9/15/62 of 96) while from-scratch controls reached 96/96.
   - Encounter bank: native_td_learning at lr 5e-4.
   
   It is mitigable:
   - A teacher-CE anchor preserves the skill (AP, BY, CA reanchor) but never improves on the parent.
   - Continued unchanged PQN recovered past mark 64 in 3/3 (CC).
   - lr 1e-4 largely avoided the encounter degradation.
   
   It replicated across two network families.
4. **Solo food imitation fits training data but not held-out data, and the cause there is generalization, not observability.**
   - Training fit is 100%. Held fit is about 84–90% (native_food_bc, AA, AJ, BR, BT, BU).
   - A hand-coded decoder recovers the teacher action on 100% of 12,288 rows (observation_probe) and 10,775 rows (AK), and every learned error lands on a decoder-correct row.
   - None of these closed the gap: world diversity (native_food_diversity, BT), coordinate planes (BU), a point model (AN), an auxiliary view loss (AL).
   - The CNN safety residual's held veto fit was far lower: 11–64%, not helped by more worlds (AYr1).
   - Encounters differ. Held fit was about 70–96%, labels partly alias on hidden state (item 9), and world breadth did raise held fit (about 70%→86%, and 92–96% with the global teacher).
5. **PQN seed variance swamps single-intervention effects.** Paired reversals:
   - living-mass reward: +10.4 then −0.59
   - reset heading: +7.3 then −0.36
   - LR 1e-4: +1.31 then −13.54
   - solo seed 1701 food 83.6 vs seed 1702 food 0
   - microbenchmark recipes: 1/3, 1/3, 2/3
   
   A "survive but stop eating" collapse recurs even at lineage doses above 1M transitions:
   - CL 5602: 1.75 food at 384/384 alive.
   - CO 5801: 3.31 food at 383 alive.
   - CN 5701: midpoint food 0.60.
   
   This is strong as a pattern; the cause is unidentified.
6. **Short-horizon solo food seeking is learnable by several routes.**
   - BC food-geometry policies passed fresh-world confirmation at H64 and H128 in 3/3 seeds.
   - The Apex network learned H8 by supervised imitation (research_transfer). The canonical Apex TD learner from scratch reached 96/96 at H8 (canonical_retention controls, descriptive) and transferred zero-shot to H16 (fresh_h16), all 3/3.
   - The cumulative PQN lineage passed H768 in 3/3 (CW: endpoints 353–383 of 384, food 117–135). CW is outcome-selected at every step from a single BC ancestry, so its effective n is 3 correlated lineages.
7. **6102 is probably worse than Apex at H5000, and the current gate cannot resolve 10% effects.**
   - From the README's paired SDs and means with n=16, a back-of-envelope t is about −2.1 (frozen), −2.2 (scripted) and −2.5 (mixed).
   - 5 of 16 wins per mix is not significant by a sign test (two-sided p ≈ 0.21).
   - These are my derivations on unaudited data.
   - The pilot can flag a large deficit; it cannot detect a modest gain.
8. **Survival-only gates are gameable.**
   - Fixed turn-priority rules pass the encounter survival package 192/192 (static_controls).
   - Short-horizon solo survival sat at the random or mask ceiling up to H64 (Apex banks, AO), which hid the collapse in food.
9. **Teacher labels carry hidden-state pathologies.**
   - Exact-input alias conflicts rejected two corpora.
   - One audited case (3419) flips under future-RNG changes alone.
   - H4 targets contain horizon regret on 87 decisions across 79 witness games.
   - The GreedyFood teacher fails its own calibration at H256 and longer (CH: 333/384; CW: 127/384 at H768).
   - Prevalence is unmeasured.
10. **Recurrence was not supported.** The remote-heading alias exists but was not action-relevant in the H1-A probe (0 regret over 36 pairs).
11. **The training objective and the gate differ.** Analytically, the discounted training return is not equivalent to the H5000 undiscounted mass-integral gate. Whether this caused any failure is unproven.
12. **Engineering facts** (well measured, low stakes):
    - MPS is 2.36–2.53× faster than CPU for whole raster-PQN updates.
    - E128/T4 gives 1.38× throughput.
    - Packed paint is 11–23% faster.
    - The decision-phase mismatch was real and was repaired opt-in.

---

## 4. What did not work, and recurring failure modes

**Dead ends, grouped:**
- **Single-knob PQN tuning:**
  - At 50k–200k: sampler coverage, living-mass reward, reset heading, scripted advice, LR, epsilon floor (multi-snake and solo), ambient bonus, budget escalation, λ=1, late-half LR.
  - In the lineage: λ=.95, selector cut, Huber vs half-MSE, training-cap horizon (CG, CR), and food coefficient 1.0 (CB, significantly harmful).
- **Handoff repairs:** value offset (BV), Q-scale alone (BX 1/3), anchor release for beyond-parent gains (BY, CA).
- **Imitation generalization levers:**
  - world diversity or breadth in solo (native_food_diversity, AYr1, BT)
  - full-context residual (AZ)
  - reachability scalars in the CNN (BA, BB)
  - auxiliary view loss (AL)
  - point model (AN)
  - CoordConv (BU)
  - L2 (l2_learning)
  - DAgger refresh beyond one round (AJ)
  - own-policy coverage (BR)
- **Encounter TD fixes:** crossing rehearsal, direction anchor, NLL coefficient (gradient-incompatible), ambient reward overlay, unchanged-dose and full-cycle continuation.
- **Apex-learner levers:** replay mixing, uniform replay and width/breadth scaling were rejected or null. H256 collection was invalid on fresh worlds and showed no gain on the existing bank. ε=0.1 is a near-miss, not a dead end: it gave the largest H64 gains and was rejected on one-food and one-bank retention misses.
- **Input access in PQN** (not established, but carried forward): body channel (CK 1/3), length scalars (CV 1/3; retained in the CW lineage), enemy channels (CZ 1/3; the visible arm produced 6102), extra enemy dose (0/3).
- **Diagnostic only, never tried as an intervention:** first-transition gradient weighting (native_feedback). Its support pattern failed.

**Recurring scientific failure modes:**
1. **Acquire-then-lose dynamics.** Skills appear at an intermediate checkpoint and disappear or dip by the final one:
   - four_hour N2
   - solo seeds 1702 and 1901
   - microbenchmark seed 2301 (95%→33%)
   - CC's dip at mark 128 (later recovered)
   - midpoint swings in CV
   
   Gates on a single final checkpoint therefore measure a noisy point on an unstable trajectory.
2. **Degenerate periodic loops:** relative-right period-4 loops (four_hour) and period-4 loops after the first food (Apex growth census). The mechanism is untested. Exact-coverage absence was shown to be uninformative, the finite-return screen found no discordance, and most between-food teacher disagreements were tie-equivalent moves.
3. **Safety information available but not reflected in choices.**
   - Freshly trained small-arena Apex models picked 0.03125-space actions when a 1.0 option existed (14/15; association only).
   - Raster learners missed teacher vetoes in 60–70% of death lanes (AH).
   - Generic networks did not learn escape-preserving turns from the amounts of data used here. A narrow inductive bias on privileged features did (BH→BK→BL).
4. **Gains that do not transfer across regimes:**
   - H64 survival hid a food collapse.
   - Solo skill lost 21–29 endpoints with one opponent (CX).
   - H3000 gains over the parent did not carry to H5000 vs Apex.
   - Train-world competence (91–94% of teacher food) did not survive held worlds.
5. **Food/boost vs mass tradeoff.** task_aligned raised food, survival and boost (boost fraction ≈0.001–0.04 → 0.29–0.65) but did not consistently raise mass:
   - Solo mass fell in the 6101 preview and for 6103.
   - The 6102 continuation lost −35.2 [−45.8, −24.5].
   - By contrast, 6102 itself gained +37.9 over its parent.
   
   Boost as the cause was never tested.
6. **Teacher-dependence.** A lot of effort went into repairing oracle labels: clearance tie-breaks, RNG aliasing, early-food ties, common-path and receding teachers. The receding teacher cost 2,209 s and produced labels identical to the raw ones.

---

## 5. Process assessment

**Verdict counts** (my tally of the family classifications, rechecked):

| Family | n | pos | neg | null | diag | infra | invalid | not audited |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| foundations | 23 | 0 | 12 | 1 | 3 | 7 | 0 | 4 |
| solo native 0913–14 | 30 | 4 | 11 | 3 | 9 | 0 | 3 | 3 |
| solo food 0919–20 | 28 | 7 | 11 | 1 | 9 | 0 | 0 | 12 |
| solo_food_pqn | 35 | 2 | 17 | 5 | 10 | 0 | 1 | 1 |
| controlled_encounter_a | 21 | 1 | 6 | 1 | 7 | 6 | 0 | 3 |
| controlled_encounter_b | 21 | 2 | 8 | 4 | 5 | 2 | 0 | 2 |
| apex | 25 | 2 | 7 | 3 | 8 | 0 | 5 | 2 |
| task_aligned | 9 | 0 | 2 | 2 | 0 | 1 | 4 | 5 |
| **Total** | **192** | **18 (9%)** | **74 (39%)** | **20 (10%)** | **51 (27%)** | **16 (8%)** | **13 (7%)** | **32 (17%)** |

- **Positives are fewer than they look.** About 7 of the 18 are not independent learned-gameplay gains:
  - teacher or oracle calibrations (AF, BD, BE);
  - diagnostics (horizon_diagnostic, one_action_rescue);
  - a training-fit admission (BK);
  - a byte-identical repeat of BL (BM).
  
  solo_food_survival is a weak learned gain: survival was at ceiling from initialization, and one seed's food was 98.5% its own trail, though both seeds improved ambient food. None of the 18 compares against Apex.
- **Audit coverage is 83%, but mostly bookkeeping-level.**
  - Audits typically check hashes, receipts and arithmetic, often without decoding raw arrays or replaying natively. Some reused the frozen decision reducer.
  - Audits themselves failed on auditor bugs, e.g. in anchor_retention, bridge v4, native_pretrap and the strict pilot.
  - The 0919–20 family's 12 unaudited studies include its key oracle screens BD and BE.
- **Invalid runs: 13, of which 12 were tooling or rule failures.**
  - Examples: a 63-character SHA literal, a JSON tuple/list round-trip, heartbeat field names, subTest row counts, watchdogs, a KeyError.
  - Freshness overlaps: 11 of 3,072 rows; 1 of 127,506 inputs; 48 raw geometries discovered after training finished.
  - Only CH stopped for a scientific reason: the teacher failed its own calibration.
  - Two invalids left partial or provisional data (horizon_continuation, native_pretrap).
  - Zero-recovery windows turned each small defect into a whole new admission.
- **Split between science and infrastructure, by count:**
  - about 58% comparative learning or evaluation studies (pos+neg+null)
  - 27% saved-data diagnostics, many running in 1–60 s
  - 15% infrastructure or invalid
  
  **By compute,** reported science time is roughly 31 h on one Mac over 20 days (my sum):
  - task_aligned ≈ 15 h
  - solo_food_pqn ≈ 7.6 h
  - CE-B ≈ 2.8 h
  - CE-A ≈ 2.4 h
  - 0919–20 ≈ 2.0 h
  - apex ≈ 1.5 h (training and evaluation studies only)
  - The other two families were not summed; their runs were seconds to minutes each.
  
  Elapsed time went mostly to design, freeze, qualification, audit and write-up, which the write-ups do not measure.
  
  **Training doses:**
  - 50k–200k transitions per arm in most foundations screens (500k per arm in the 09-06 sampler)
  - about 1.4M per lineage in solo_food_pqn
  - about 7.9M per lineage in the task_aligned CZ continuation, with another 7.93M for the 6102 continuation
  
  The write-ups repeatedly flag the gap to Apex's historical training dose as a confound.
- **Horizon mismatch.** Deployment is H5000 against mixed opponents.
  - Outside foundations, only 3 of 169 experiments ran at H5000: solo_food_survival (solo, survival at ceiling), bridge admission-v4 (interface only) and the strict pilot. Only the pilot measured performance against Apex.
  - Foundations' four solo 5,000-frame screens used a one-snake diagnostic profile with survival at the mask ceiling.
  - The two encounter families (42 experiments) evaluated at H4/H16.
  - Other solo work stopped at H768.
  - task_aligned screened at H3000 in solo and scripted-opponent profiles.
- **Sample size vs noise.**
  - Seeds: n=2–3 training seeds per comparison.
    - solo_food_pqn and task_aligned: correlated lineages descended from 3 BC parents (3401–3403).
    - Encounter families: warm-start lineages.
    - 0919–20: 3 frozen parent cohorts.
    - Apex: fresh initializations.
  - Evaluation units per comparison: 4–8 worlds (df≤7), 32 worlds (df 31), single 48–96-case banks (Apex), or 16 worlds (pilot).
  - Gates: strict all-seed conjunctions. At an illustrative 80% per-seed pass rate, a true effect passes all three only about 51% of the time.
  - Near misses: many rejections turned on a single lane or food (CD 86 vs 87; CF 342 vs 348; second_food 47→46; ε-0.1 food 149 vs 150).
  - Selection: cohort and arm choices were outcome-informed at nearly every step (CK, CP, CS, CV, CX, 6102). This is disclosed.
  - Net effect: the gates protect against false promotion but leave "negative" nearly uninformative.
  - Evaluation noise is large even before any intervention. Initial-checkpoint means varied 1.46 vs 6.93 within one study. The pilot's scripted-mix difference (−66.4) exceeds that mix's development-bank Apex mean (38.8). Whether this reflects bank-to-bank or episode-to-episode variance is unknown, because the README does not say whether pilot and development worlds coincide.
- **What was done well:** pre-registration, preserving failed attempts instead of silently rerunning, honest negative labeling, a kill criterion that actually fired, and consistent disclosure of confounds.

---

## 6. Has anything improved the deployed agent?

No. Apex/vector61 is unchanged. No checkpoint was promoted, and no serving or default behavior changed.

The engineering changes that landed are on local main. None changed defaults, and the foundations report says none were pushed:
- food-contact telemetry
- opt-in scripted-anchor adapter
- solo lifecycle and fenced solo evaluator
- opt-in living-mass and ambient coefficients
- opt-in watch_pre_move_v1
- packed paint and four_hour product fixes

The only candidate that faced Apex lost (unaudited).

Knowledge relevant to the deployed agent:
- Freshly trained small-arena Apex-architecture models self-trap at H256, and just before death they choose low-free-space actions when roomier ones exist (14/15, association). These were not the champion checkpoint.
- The 09-06 audit found counterexamples in the incumbent's stack. Their effect on the champion is unmeasured:
  - stale Apex replay-slot priorities;
  - serving row, mechanics and mask mismatches;
  - vector61 feature drift between arena scales 0.2 and 1.0 (44 of 61 dims), which also limits how far the small-arena Apex results transfer.
- A flood-fill safety layer solves solo self-traps up to H512 with snakes up to length about 105.
- The promotion gate cannot currently confirm a modest improvement.

None of these has been applied to the incumbent.

---

## 7. Recommendations

**Highest-leverage next moves:**

1. **Calibrate the promotion instrument before training anything else.**
   - Run an A/A test (Apex vs Apex) at H5000 to measure null variance and gate resolution directly. The blueprint also proposes an ordering check: champion > scripted anchor > random-safe.
   - Choose a variance-reduced design on development data only, then pre-register it. Candidates:
     - several episodes per world;
     - covariate adjustment of the paired Δ on a pre-measured world covariate (pairing already removes the incumbent's own per-world score);
     - a pre-registered change of estimand (log or rank mass, survival time).
   - Add a sequential stop for clear losers.
   - Evidence: the pilot's 8,676-world requirement and its 1–3 noise-to-mean ratio. With n=16 it could only flag deficits of roughly 30–66 (t ≈ −2.1 to −2.5). Modest effects are undetectable.
2. **Size, then test, a serving-time safety layer on the incumbent, with no training.**
   - Step 1: take a death census of Apex's H5000 pilot or development episodes to measure the self-trap share. This is cheap if the episodes were archived with replayable actions. With opponents, most deaths may be head collisions with a non-empty safe set (CY 70/76), in which case the achievable effect is small.
   - Step 2: if self-traps are a material share, compare Apex against Apex plus a capped flood-fill veto at H5000 across all three mixes.
   - Evidence: the self-trap censuses (§3.1), the 96/96 oracle and head results (§3.2), and the pre-trap choices of fresh Apex-architecture models.
3. **Buy throughput before more learning science.**
   - Build the vectorized simulator. The blueprint targets ≥40k steps/s locally and ≥100k/s on a rental.
   - My derivation puts the pilot at about 188 frames/s end-to-end (720k frames in 3,828 s).
   - A strict gate needs roughly 500× more evaluation worlds than the pilot, and learning results are seed-variance-limited.
4. **Redesign screen statistics.**
   - Use at least 5 independent seeds from fresh initializations, not re-seeded descendants of three BC parents.
   - Pool effects across seeds with a random-effects CI instead of all-seed point conjunctions.
   - Evaluate several checkpoints per run, or the area under the learning curve, instead of a single final checkpoint, given the acquire-then-lose dynamics.
   - Require one cheap development-only H5000 check against Apex before any lineage gets multi-day investment. That needs a lightweight harness that can serve the candidate without full bridge governance. With one, 6102's deficit would plausibly have shown in about an hour.
5. **Tier the governance.**
   - Keep full freeze and independent audit for promotion-relevant runs.
   - Use a lightweight path for diagnostics under a minute.
   - Check bank freshness and overlaps with tooling at construction time, not at admission.
   - Allow one mechanical-defect recovery without a new admission.
   - Evidence: 12 of 13 invalid studies failed on tooling or rules.

**Start doing:**
- If mass is the objective, test the boost/mass tradeoff causally. For example, run a boost-restricted or boost-cost ablation of 6102 or its continuation, or the blueprint's mechanics-v2 changes. Do not tune around the observed food-up, mass-down pattern. Note that own-trail food already exists in the current simulator (source_trace).

**Stop doing:**
- Naive BC→TD handoffs: unanchored, lr 5e-4, Q-scale 1. These collapsed in at least 7 studies across two network families. Anchored, low-LR or longer-dose handoffs are not ruled out, and the best RL lineages came from one.
- Single-knob PQN tuning until seed variance is controlled.
- H4/H16 controlled-encounter teacher and label engineering. Both lines have stopped, and survival-only gates there are gameable.
- Diagnostic chains whose only output is the next diagnostic.
- Nominating challengers on gains over a weak parent instead of distance to Apex.
- Serving-bridge qualification for any candidate that has not first shown a promising development-only H5000 screen against Apex.