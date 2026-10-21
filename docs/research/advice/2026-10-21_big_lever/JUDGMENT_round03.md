# Round 3 judgment — big-lever research rounds

Judge: Fable (round-3 judge), 2026-10-21 ~01:30 PT. Inputs: `../CHARTER.md`, `../round-01/JUDGMENT.md`,
`../round-02/JUDGMENT.md`, `../LIVE_FACTS.md` (23:45 PT entry; still no E1–E9 results), `../QUEUE.md`, `QUESTION.md`,
`astra_01..10.md`. Repo verified read-only at main `8db79d23` (Sol's frozen packages: `research/bl_prefix/REGISTRATION_E6.md`,
`REGISTRATION_E7.md` (c6d6bb5f), `research/bl_info_value/PROTOCOL.md` + `PREFLIGHT.md` (smoke complete), `research/bl_ppo/INTENT.md`
(intent 3bedc5de, E8 amendments not yet applied, `exp/bl-e2/source-authority.json` NOT_DISPATCHED), `research/gpu_sim_spike/E1_PREFLIGHT.md`
(undispatched)). `exp/bl-e6e7/` is empty: E6/E7 have had no gameplay, so reporting additions are still pre-data. Sealed study not
opened. No jobs run.

## (a) Bottom line

1. The round closed the arithmetic the question asked for, and all ten advisors closed it the same way. The exact per-world
   ledger is `ΔM = [Σ both-alive (L_cand − L_B8) + Σ cand-only L_cand − Σ B8-only L_B8] / 5000`, pooled as the equal-weight mean
   of three mix means (`SUCCESS_CONTRACT.md:27-30`, verified). The planning approximation is `G_m ≈ q_m · ρ · K − C_m` with
   `K = L(5000 − d)/5000`. At B8 s3's scripted census median death (frame 3,283, L 972 — `panel_mac_answers_2026-10-14.md:79-86`,
   verified) K ≈ 340; with 90 % retained tail (306) and 8–10 collateral, **+40 needs ≈ 15–17 additional full-horizon survivors per
   100 natural worlds in every mix** (02: 13.2 at C=5; 03: 16.1; 04: 14.7–16.3; 05: 14.7; 06: 17.0; 07: 15.7; 08: 14.7–18.4;
   09: 16.3; 10: 11.8 at C=0). Scripted-only improvement needs ~45–50 per 100 scripted worlds (02, 07, 08, 09); T3's entire +122.8
   reproduced on scripted alone is +40.9 pooled (10). Deaths after frame ~4,200 are worth ≤ 150–190 each and need 25–40 rescues/100.
2. **The E6 observable is already in endpoint-mass units** (REGISTRATION_E6: score = remaining mass-sum/5000). Its population value
   is `q · EPD`, where q is the natural fraction of worlds that enter the eligible stratum (frame ≥ 2,000, L ≥ 700). At q ≈ .4 and
   EPD +10 the whole-world contribution is ≈ +4 before collateral — **an order of magnitude short of +40** (01, 02, 03, 05, 07, 08,
   09, 10 independently). Overlapping ancestor roots of one death cannot be summed; prefix→B8 continuation hides the candidate's
   own re-death.
3. **Round 2 was wrong on two clauses and I withdraw them.** (i) "LB90 > 0 with ordinary ≥ −2 ⇒ needs scale" is too permissive:
   the frozen E6 label NEEDS_SCALE (REGISTRATION_E6 §decision) stands as written, but it admits *replication planning and the
   population bridge (E10)*, never a replication dispatch. (ii) "Ending 36 decisions before death means the dividend cannot be
   bought by caution" is false (01, 03, 09, 10): a 64-decision cautious prefix can shrink growth before the handback; E6's
   secondary (candidate to H5000) and both-alive growth inside the window are the caution check and must be reported.
4. The collateral scale on record is larger than the rescue scale: T3 LABELS lost 55.33 of 75.15 on common-alive frames and
   shared survivors finished 443–460 cells smaller (`t3_diagnostics:181-185`, verified); T23 −80.8 with survival unchanged; T6
   shared-survivor length −23.7..−28 (`t6_screen_result:21-23`, verified) ≈ −10..−14 mass over 2,500 shared frames. A 20-cell
   persistent deficit costs 10 endpoint mass (03). Therefore every endpoint readout must carry the three ledger terms by B8
   death-frame × length stratum, and the mass-ledger is the explanation of any G, never a derived rescue count.
5. **Prevalence q_m is free.** E6's bank is collected "in deterministic seed order, rotating scripted/frozen/mixed" until 44/42/42
   eligible deaths (REGISTRATION_E6 §bank), bounded at 4,096 worlds. If the collection ledger (worlds screened per mix, death frame
   and length of every death, survivors) is retained, q_m and the B8 death-stratum distribution fall out at 0 units. 10's oracle
   envelope `U_i = max(0, max over 24 prefixes ΔM_i)` (ineligible worlds = 0) is computable from E7's per-prefix training outcomes
   plus that ledger — also 0 units. **E10** (below) registers both before E6/E7 data; `UB90(U) < 40` is the single pre-scale
   measurement the red team asked for and it closes the *B8-local single-prefix* rescue thesis, not lever 1.
6. Units are not the binding constraint for replication (08, arithmetic verified): 100M hero transitions at 1–4k/s is 7–28 L4-h
   (14–56 u); 2 × 100M fits in a day of allocation. Round 2's "100M per seed is a week object" is withdrawn. The calendar is set by
   build, critic qualification, evaluation and the Mac gate. The engine (lever 2) is therefore needed for 10^9 and for
   search-in-the-loop branch rates, not for replication.
7. T35's two-deviation beam is a live fact the search advisor missed: pooled KILL, 26/26 EXTEND events NOT_RESCUED, zero novel
   rescues (`CAMPAIGN_LOG.md:73-76`, `research/t35_two_deviation/RESULT_EXTEND.md`). 06's 256 two-deviation roots are not funded;
   one-deviation current-policy Q targets remain the lever-4 form, after E1.
8. Shortlist order unchanged; lever 1 down to 35 % / 15 % (transfer gap quantified, no measured q); engine unchanged at 30 % but
   re-scoped; representation 30/20 with geometry ahead of memory (01, 07 agree) and a Mac serving-cost probe added (E14);
   search 20/15 minus two-deviation; restart archive 10 % with 04's 80/10/10 + final natural-only phase as the registered form.

## (b) Per-advisor verdicts

| ID | Persona | Verdict | Why (verified facts in **bold**) |
|---|---|---|---|
| 01 | Expert player / body-routing | **keep** | Concrete behaviour: outward turn before the trail crosses the return corridor (300–150), choose the pocket side whose exit releases first (150–50), hold the exit against food (last 50). **T1: survivor alternative within 300 decisions in 46/49 late deaths but within 100 in only 25/49; sealers 13 opponent / 13 hero** (`t1_commitment_study:35-48,75-76`, verified) — so ~half the late deaths have no 100-decision-local alternative. Its 17 rescues/100 at V=289, C=8 is the round's central number. Geometry before memory; served search unproven. Correct that E6 can be bought by caution. |
| 02 | Large-scale on-policy RL | **keep** | Population bridge with 192 natural worlds/mix per endpoint, three continuation modes (prefix→B8, prefix→candidate, candidate-from-reset), exposure floor ≥ 2M natural L ≥ 800 transitions/seed (reporting floor), 32M pre-declared funding stop. **Equal-mix weighting regardless of 96/32/32** verified. Its E6 cost worry is handled by the registration's fail-closed forecast (REGISTRATION_E6 §limits). Right that PPO ratios do not correct restart occupancy. |
| 03 | Plasticity / forgetting | **keep** | **E2 unfreezes the encoder (INTENT: LR 1e-5 encoder)**, **fused width 256 (`ego2s_network.py:58-60`)**, **T4c rate is CTRL-relative at Q-gap > 0.3 on L ≥ 700 states (`t4c_prereg:66-71`)** — all verified. The exchange-rate measurement (before/after one predeclared optimizer block on the same roots) and the trigger (ordinary EPD UB90 < −2 or common-alive loss < −10 ⇒ retention arms A/B/C) replace Round 2's "4 % drift + null dividend". Withdraws its own five-seed scale call. |
| 04 | Exploration / restart | **keep** | **T6: frozen −40.14 vs CTRL, shared survivors −24..−28, restarted L ≥ 800 death rate .32 vs .20** verified. Prevalence-weighted decomposition `p·f·τ·b − c`; eight attempts per root are correlated; E7 at 5 % cannot certify E2's 1 % discovery rate; the "horizon collapse" reading confounds intervention timing with reach. Registered archive form adopted: 80/10/10, half of frontier starts before the first deviation, back up 32 decisions only after ≥ 70 % productive continuations, final 2M natural-only. Retracts 5B. |
| 05 | Credit assignment / critic | **keep** | `q·p·e·K − C`; EPD already /5000, never multiply by p or K again. Critic *scale* qualification (EV ≥ .3 per mix × L × horizon cell, MSE ≤ .8 baseline, GAE bias within ±5 on early roots, refit against the current stochastic actor) is distinct from E6's EV ≥ .1 preparation gate — adopted as E12. Opportunity budget: 1,805 eligible worlds/mix at σ=200 (arithmetic checked); 100M at .2 eligibility ≈ 1,667/mix. Defers RUDDER. |
| 06 | Search-in-the-loop | **partly** | NEI identity (finite-horizon performance-difference lemma on *new*-policy states) is the right gate predictor; mass/1000 vs /5000 scaling note correct (INTENT reward). But **T35 two-deviation beam: pooled KILL, W_novel 0** (`CAMPAIGN_LOG.md:76`) — its 144-template two-deviation search is already refuted at this budget; and its 20k branch world-steps/s precondition is unmeasured (E1 pending, `RESULT.md:6`). One-deviation targets after E1 only. |
| 07 | Representation / latency | **keep** | **E3 arm B has no current-step opponent tokens; C adds all six seats' head/heading/mass/alive/action history** (`PROTOCOL.md` §Models, verified) — C−B mixes temporal and current information; E3b control stands. **Vector61 nearest-enemy trend is stateful (`vector61_featurizer.py:43-51`)** verified. Function-preserving actor (`Q_B8/T + W z`, W=0) with a separate richer critic is the right insertion. Spatial cost (~18.7M MAC for an 8→16 stem on 145×83) must be measured against **p95 ≤ 11.688 ms** (`SUCCESS_CONTRACT.md:46-47`) — E14. |
| 08 | Distributed systems / engine | **keep** | **Cap 3 is our supervisor's own refusal (`colab_cli/supervisor.py:818`)**; **`ego_device.py:163-181` now has pointer jumping with early exit every 8 rounds (worst-case bound V−1)** — Round 1's "runs 12,035 iterations" is stale, the three-kernel brief still stands; **T16 4,062–9,138 tr/s learner-bound; QUALIFICATION 64 MiB/43.16 s, no exact resume** verified. Capacity table arithmetic checked (2k/s ⇒ 13.9 h/100M; 0.97B/week/runtime at 80 % duty). Four admissibility layers adopted into the parity rule file. Right that Round 2's "100M = a week" was wrong. |
| 09 | Evaluation / statistics | **keep** | Stratified variance `Σ s²/(9 n_m)`; survival SE ≈ .023 at 384 pairs (checked); +.03 true clears the +.03 point bar ~50 % regardless of N; E7's 26/128 passes the point rule with Wilson LB90 ≈ .16; joint family α (.05 E2 route + .05 E6 route, Holm R/F) adopted into E9; nomination rule for two artifacts; platform-transfer audit `B = ΔMac − ΔColab` with ±10/±20 margins adopted as E13; Dwork adaptive-reuse warning for E6/E7 worlds mined into training. **`reference.py:117-122` never reruns policies** verified. |
| 10 | Red team | **keep** | The oracle envelope `U` (hindsight max over 24 B8-local 64-decision prefixes, perfect triggering, zero ordinary damage, ineligible = 0) is the right single pre-scale measurement; `.3×10 − .7×2 = +1.6` and `122.8/3 = 40.9` checked. Its 60-u fresh-census version is unnecessary: U is derivable from E7 + the E6 collection ledger (E10). Correctly scoped — it bounds single-prefix finite-menu strategies, not trained multi-intervention policies. Its NO-GO branch (E3 → richer-input 20M pair) is adopted as the week's pivot. |

Factual errors found: 06 omitted the T35 two-deviation KILL (changes its proposal 1); Round 1's flood-loop statement is stale
(08). No advisor arithmetic error that changes a conclusion. Corrections to the Round 3 fact sheet: (i) T1's "46/49 within 300"
are D300-surviving alternatives, not full-life rescues, and only 25/49 exist within 100 decisions; (ii) E6's EPD is already in
endpoint-mass units; (iii) q_m is recoverable from E6's own collection ledger.

## (c) Disagreements ruled

1. **Replication size — 02 (2 × 32M stage) vs 08/09 (2 × 100M) vs 05 (2 × 120M opportunity budget).** 08/09 win on cost (100M
   is hours, not days) and 02 wins on decision logic. Ruling: 2 × 100M (120M if E4 ≥ 2k/s and the 5 % eligibility case holds),
   with a **pre-registered futility-only look at 32M** (tripwires, survival band, UB90(G) ≤ 0, negative candidate-owned early
   dividend ⇒ STOP; no efficacy claim at the look). Endpoint-only efficacy with 09's nomination rule. 05's 1,805-world floor is a
   reporting requirement (exposure count per mix), not a size driver, unless the E10 ledger shows q < .1.
2. **What a positive E6 licenses — all ten vs Round 2.** Round 2 withdrawn (bottom line 3). NEEDS_SCALE ⇒ E10 bridge + E12
   critic qualification + replication *registration*; dispatch of 2 × 100M requires E10 not FINITE_MENU_INSUFFICIENT or a
   registered non-B8-local design change (archive, search targets, representation) in the replication recipe.
3. **Where prevalence comes from — 10 (fresh 256/mix), 04 (512/mix census), 02 (192/mix per endpoint), 05 (512/mix).** The
   judge's ruling: the E6 collection ledger gives q_m and the death-stratum distribution now at 0 u; per-endpoint prevalence
   and the full ledger come from E2's existing 96/32/32 screen bank (B8 rows already there, `exp/bl-e2/screen-bank-v2`) with E8's
   ledger amendment. A larger fresh natural bank (384/128/128) belongs to the replication stage (E15), not the pilot.
4. **Two-deviation search (06) vs T35.** T35 wins; two-deviation templates are closed at this budget. One-deviation
   current-policy Q targets (06's auxiliary arm) stay the lever-4 form after E1 CONTINUE; NEI on new-policy states is its gate
   predictor.
5. **Geometry vs memory (01, 07) and whether either must be served.** No disagreement: geometry first, history only if E3 C−B ≥ 10 %
   after the E3b control; a served planner is not on the table (Path B smoke STOP, LIVE_FACTS). The serving cost of a geometry
   encoder is unmeasured and must be probed on the Mac before any training commits to it (E14).
6. **Exploration 1 % vs 5 % (02, 04, 05, 09 unanimous).** Any change is a separately named treatment; E7's FRONTIER_SCARCE /
   TRANSFER_ADMISSIBLE labels describe 5 % discoverability under B8, not E2's learner; report realised departures per prefix.
7. **Retention trigger — 03 vs Round 2.** 03 wins: trigger on ordinary EPD UB90 < −2 or whole-world common-alive loss < −10 at any
   endpoint, regardless of disagreement rate; arms A (unanchored) / B (cumulative KL ≤ .005 per cell) / C (B + frozen trunk,
   16.8k-param residual) are the registered follow-on, not a pilot change.
8. **"Needs the engine for 100M" — Round 2 vs 08.** 08 wins; engine re-scoped to 10^9 + branch rollouts. E1's rule and brief
   are unchanged (the realistic 1,024–4,096-world long-body benchmark remains the admission test).

## (d) Running shortlist (carried from Round 2, updated)

| # | Lever | Conf. push / beats B8 at bar | Cost to first signal | Most informative next experiment |
|---|---|---|---|---|
| 1 | **Long-horizon on-policy RL from B8** (exact H5000 return, horizon critic, PopArt, calibrated warm start, measured drift, natural resets) — transfer target now explicit: ≥ 15–17 added survivors/100 natural worlds per mix at ≤ 10 collateral | 35 % / 15 % (down 5/5: quantified gap, no measured q) | E2 20 u + E6 4 u + E10 0 u + E12 16 u; replication 2 × 100M ≈ 60–120 u | E2 (E8-amended) + E6, then **E10 population bridge** (q_m, U envelope, both continuations) before any replication registration |
| 1b | Retention treatment (03's A/B/C: cumulative anchor, frozen-trunk residual) — sub-arm, conditional | — | 36 u, Sol 1.5 d | Only on the 03 trigger at an E2 endpoint |
| 2 | **GPU-resident exact engine** — re-scoped: needed for 10^9 and lever-4 branch rates, not for 2 × 100M | 30 % reaches ≥ 20k long-body L4 world-steps/s this week (unchanged) | 20 u + Sol 2 d + 4–7 d qualify | E1 as briefed (three contracts, realistic benchmark; code already has early exit, not a bound) |
| 3 | **Representation: full-res geometry (+ history) student**, function-preserving insertion (`Q_B8/T + W z`) | 30 % / 20 % (unchanged; geometry ahead of memory) | 16 u E3 + 0 u E14 (Mac) + 30 u richer-input 20M pair on E3 GO | E3 as frozen; **E14 serving-cost probe** decides servability; E3b current-token control only if C > B |
| 4 | **Search-in-the-loop** (one-deviation current-policy Q̂ targets, NEI on new-policy states) | 20 % / 15 % (two-deviation form closed by T35) | blocked on E1 rate; 80 u pilot | After E1 CONTINUE and E6/E10: 06's 3-arm pilot without two-deviation roots |
| 5 | Learned world model | 5 % / 5 % | — | Only if E1 STOPs and lever 1 shows a dividend |
| 6 | League / history curriculum | 5 % / 5 % | — | Not funded |
| 7 | **Restart archive from confirmed prefixes** (80/10/10, half pre-deviation starts, 32-decision back-up on ≥ 70 % productive, final natural-only phase) — add-on to lever 1 | 10 % as add-on (unchanged) | E7 20 u; transfer arm 70 u | E7 → E10 frontier map; transfer arm only if TRANSFER_ADMISSIBLE **and** U envelope is not FINITE_MENU_INSUFFICIENT |

## (e) Still unresolved

- **Measured q_m** (eligible-death prevalence per mix) and the B8 death-stratum distribution: every transfer number this round is
  conditional on it. E10 supplies it from E6's ledger.
- **Whether any B8-local 64-decision prefix menu can reach +40 even with hindsight** (U envelope). If UB90(U) < 40, lever 1's
  recipe must contain a non-B8-local discovery mechanism before 100M, and Round 4 must decide which.
- E4 rate, assignment cap (E5), and whether T34's lanes leave two slots this week.
- Serving cost of a full-resolution geometry encoder on the Watch6 path (E14) — lever 3 is unservable until measured.
- Parity doctrine still unwritten as a rule file: 09's three layers + 08's four (engine identity on injected candidates, policy
  agreement on decisive roots, durability/exact resume, closed-loop Mac evidence) — needed before any 100M tranche.
- Adaptive reuse: E6/E7 worlds mined into an archive are burned for evaluation (Dwork); the replication bank must be fresh.
- Whether E7's 5 %-exploration discoverability says anything about a 1 %-exploration learner over 100M (nobody can bound it).

## (f) Experiments Claude can run now (≤ 20 u each; dispatch via Sol between rounds; report in LIVE_FACTS.md)

Queue status (from `../QUEUE.md`): E1, E2 (E8 pending), E3, E6/E7 are built or being built; E8/E9 are queued amendments. Below,
**refine** means a pre-data addition to a built package; nothing re-proposes them.

**E10 — Population bridge and oracle envelope (0 u; Sol 0.5 day; register before E6/E7 gameplay — `exp/bl-e6e7/` is empty,
so this is pre-data).** Additions to E6/E7 *reporting* only; frozen decision labels untouched.
(1) Retain and export the E6 collection ledger: per mix, worlds screened, every death's frame and last length, survivors, and
the eligible-death count; report `q_m` = eligible deaths / worlds screened with a one-sided 90 % Wilson interval, and the
death-frame × length histogram (bins 2000–2999/3000–3999/4000–4999 × 700–999/1000–1199/≥ 1200).
(2) E7 exports every one of the 24 per-root training-RNG remaining-mass deltas vs paired B8 (not only the selected prefix).
(3) Compute 10's envelope `U_i = max(0, max_24 ΔM_i)` on training RNG (ineligible worlds = 0; ineligible count from the ledger),
equal-mix mean with world-clustered one-sided UB90; also the confirmed-only variant using E7's two confirmation streams.
(4) Report the prevalence-weighted dividend `q_m × EPD_m` from E6, per mix, and E6's secondary rows (candidate greedy to H5000)
as the caution check with both-alive growth inside the 64-decision window.
New registered label (one look, with E7): **FINITE_MENU_INSUFFICIENT** iff training-RNG `UB90(U) < 40` — the B8-local
single-prefix rescue thesis cannot reach the gate even with perfect selection; any replication recipe must then include a
registered non-B8-local discovery mechanism (archive, search targets, or representation) and Round 4 decides which. It does
not kill lever 1. Cost: inside E6/E7 caps.

**E8 — E2 pre-data amendments (queued; refine, 0 u).** Add to the queued list: (8) occupancy log per mix by length
(< 400/400–799/800–1199/≥ 1200) × frame (< 2000/2000–3499/≥ 3500) and the natural L ≥ 800 exposure count per seed (02's 2M floor
reported, not enforced); (9) endpoint ledger (both-alive / candidate-only / B8-only) stratified by B8 death-frame × length bin
with rescue and re-death counts, candidate growth slope at L ≥ 800, and per-mix survival guards; (10) realised non-greedy
departures by length decile (not just KL). Everything else as queued (critic horizon scalar, rollout-cut bootstrap, PopArt refit
+ EV ≥ .1 gate, drift by decile, Holm LB90 > 0, boost floor flag, ledger).

**E9 — Operating-characteristics simulation (queued; refine, 0 u, Sol 0.5 day).** Add 09's joint family: one-sided α = .05 to
the E2 efficacy route (Holm R/F) and .05 to the E6 early-dividend route (Holm R/F); simulate the joint funding probability at
true +40 (require ≥ .60) and the family null-pass (≤ .10) with shared B8 rows; keep the 128/48/48 escalation rule. Report E7's
Wilson LB90 alongside its point rule.

**E12 — Critic scale qualification (≤ 16 u; Sol 1 day; runs with E6 on each E2 endpoint).** On world-disjoint current-policy
trajectories: EV, signed bias and MSE per mix × L (800–999 / ≥ 1000) × remaining horizon (≤ 1000 / > 1000) vs a cross-fitted
time-and-length baseline; require EV ≥ .3 and MSE ≤ .8 × baseline in supported cells, GAE (λ = .995) mean bias within ±5 endpoint
mass on the E6 early roots vs complete returns; unnormalised advantage variance and non-greedy tails at L ≥ 800. Failure =
"critic not scale-qualified" (preparation), blocks replication registration, routes to lever 3. Kill: accounting errors ⇒ INVALID.

**E14 — Serving-cost probe for a geometry encoder (Mac only, 0 u; Sol 1 day; needs a quiet Mac window — after the Round 0
chunk).** Full Watch6 path (observation, masks, rasterisation, unchanged opponents, serialisation) with a stub actor =
B8 + an 8-plane 145×83 encoder (8→16 3×3 stem, 16-ch depthwise/pointwise block, multiscale pooling, hero-neighbourhood readout)
and, separately, + five 64-unit opponent GRUs updated once per frame. Paced registered probe; report update p95/p99, long-body
and contention tails vs B8. Kill: full-path p95 > 11.688 ms or any overrun ⇒ that encoder is unservable; compress and re-probe
before E3's winner is allowed into any training arm. No gameplay evidence; latency only.

**E3 — run as frozen.** The Round 2 amendment (current-step opponent tokens on arm B) is not applied: PROTOCOL is frozen at
d85d75af and the smoke reconstruction ran; register **E3b** (B + current tokens, same split) only if E3 reports C − B ≥ 10 %.
Report early-root regret is unavailable in the T3 bank (decisive/counterweight only); an ancestor stratum can come from E6/E7
roots later (E3c, not now).

**E1 — unchanged** (three contracts, realistic benchmark, frozen rule; note the current `ego_device` already early-exits, so the
union-find contract is about the worst case and the per-world BFS about count caps, not about the iteration count per se).

**E13 — Platform-transfer audit (queued, not now; ≤ 20 u + Mac time; needs a nominee).** Nominee and B8 closed-loop on a fresh
development bank 384/128/128 on both platforms; `B = ΔMac − ΔColab`, 90 % intervals inside ±10 pooled / ±20 per mix; decisive
divergences reported. Precondition for treating any Colab number as predictive of the strict print.

**E15 — Replication registration (0 u, Sol 0.5 day; after E2/E6/E10/E12).** 2 × 100M (120M if E4 ≥ 2k/s), natural resets,
equal-mix episodes, 1 % exploration logged, exposure count per mix; pre-registered 32M futility-only look; endpoint-only
efficacy on a fresh 384/128/128 bank with 09's nomination rule (both Holm-positive + guard A; ≥ one at +40 and scripted +.03;
nominate the greater lower bound); archive arm (if admitted) carries the final 25 % natural-only phase; two seeds = replication,
five = recipe.

**Dispatch order (refined from QUEUE):** D0 rerun → E4 → E5 → **E6 + E7 together (shared bank; E10 readouts registered first)**
→ E1 → E3 → E2 (after E8 + E9 + E4 receipt) → E12 with E6 on E2 endpoints → E14 in the first quiet Mac window. E6 remains
registered before any E2 endpoint look. Budget this inter-round: E1 20 + E2 20 + E3 16 + E4 2 + E5 1 + E6 4 + E7 20 + E12 16
≈ 99 u; E10/E14/E8/E9 are 0 u.

Deferred: 06's one-deviation search pilot (after E1 CONTINUE + E6/E10); 03's arms A/B/C (on trigger); 04's transfer arm
(after E7 TRANSFER_ADMISSIBLE and E10 not FINITE_MENU_INSUFFICIENT); 10's richer-input 20M pair (on E3 GO + E14 pass); any
A100/H100 use (unanimous: no measured deadline benefit on the 12-vCPU host).
