# Fable judgment — 30-day Colab portfolio (2026-10-20, Mac clock 13:40 PT)

Independent judge; read-only. Sources: the eight Astra proposals on main, SUCCESS_CONTRACT / P7 / DESIGN_V2 / CAMPAIGN_LOG, research/t34_*, snake-dqn-artifacts receipts and live supervisor polls. Probes ≤ 60 s, 1 thread, nothing touched.

## 1. Fact check

**Decisive new fact no proposal has.** A5 #2 is running now (jobs `t34-exit-b8-v2-a5-s{0,1}-a3`, 2 workers × 5 numba threads, 256 resident roots, cap 1,500 s). Both lanes log a flat **~30–31 executed row-steps/s per lane** from t=230 s to t=1,160 s, **0/256 worlds done**. At that rate the D=300 shadow fill alone (76,800 rows) takes 43 min; no world can finish inside the cap. A5 #2 will read **STOP** under Addendum B (≥ 50 worlds/h). Scale: the Mac serial path is 6.6 ms/row (M3) ≈ 150 rows/s per thread; the Mac batching profile ran ~55 rows/s per process; one L4 lane is therefore ≈ 0.2–0.5 of one Mac process, and all three lanes ≈ 0.3 of the 6-worker Mac. Every proposal prices "800 teacher worlds = 32 u" from the unmet 50 worlds/h floor; at the measured engine the honest price is ~25–35k rows/world → **≈ 1.6–2 u per teacher world**, i.e. ~400 u per 800-world round. The engine is host-bound Python-per-world; A100/G4 cannot help (all eight correctly say so).

Also verified: L4 observed 1.54 u/h (both A5 receipts; price at 2.0) — gpuops, search, distill correct. Account cap 3 assignments (QUALIFICATION) — all correct. T3 dense +91.95 pooled, 38→54/64, s3 subset 17→26/32, +122.8 — evalstats, product, distill, skeptic correct. M2 39/75 registered NO-GO, 39/42 conditional — all correct. T23 −80.8, S1 −12.5/−14.9, T3 99.66 %/−75.1/−55.3, T4c KILL, T5 separability FAIL — correct. Unpaced envelope 42.533 u = 362.5 s × 176 / 3600 × 2 × 1.2 — correct, but it is a **Mac** rate; an L4 core is 2.4–3.7× slower, so the arm is ~100–150 u on L4 unless worlds are run in parallel processes per lane. FINAL_PLAN cap ≈ 270 u — skeptic correct. **distill's loss.py counterexample reproduces exactly** (0.550 / 0.792 / 0.298 / 2.126): KL(prev‖student, τ=0.05) is applied to decisive rows and dominates the 0.1-margin hinge, so a rescue that needs a > ~0.1 Q flip is not the loss minimum; sample_weight scales both terms and cannot fix the ratio. This is the one genuinely new, testable mechanism in the eight documents. gpuops correctly read SHADOW_BATCHING: Mac throughput fell 19.1 % (0.809×); the campaign log's "Mac −19 %" is wall-time wording that reads as a gain — it was not.

Wrong or unsupported: strategist "Round 0 costs 9.6 u" (teacher-only, unmet floor; honest price ≈ 120–160 u). strategist lane 2 (4,096 endpoint roots, ~49M branch frames) ≈ 450 lane-h ≈ 900 u at the measured engine — infeasible; it is also the T4d–f return-regression family with a longer horizon. product lane 4 uses H20000 sessions (not a contract horizon) and under-prices assisted sessions ~2×. selfplay lane 3 needs a FINAL_PLAN stop-list exception and labels are the bottleneck, not allocation. search lane 3 (learned ordering) revives the T4 regression family that T5 separability FAILed. Fit duration is unmeasured everywhere (FITCACHE_VERIFICATION is a 16-row smoke; CUDA path "implemented but not qualified").

## 2. Scores (decision value/unit, feasibility, contract, honesty) and rank

| Rank | Proposal | Score | Why |
|---|---|---|---|
| 1 | skeptic | 7.5 | Tightest accounting (270 cap, 35.44+7.09, 298 u honestly uncommitted); bounded diagnostic-then-one-successor logic; refuses idle targets |
| 2 | gpuops | 7.0 | Best ops detail, caught the batching slowdown and the 5.6 % hero-forward ceiling; lane 5 (275 u) speculative |
| 3 | evalstats | 7.0 | Two real contributions: GO power ≤ 50 % at true +40; Mac confirmation ≈ 134 paced h per finalist; lane 4 too large |
| 4 | distill | 6.5 | Only new mechanism (loss counterexample, reproduced); lane C (500 u, 6,400 worlds) impossible at the measured engine |
| 5 | strategist | 6.0 | Sound frame; lane 2 infeasible and derivative; lane 3 duplicates skeptic/selfplay residual arm |
| 6 | search | 6.0 | Coherent Claim B successors with equal branch budgets; lane 3 weak; relies on unpriced searched-world rates |
| 7 | selfplay | 5.0 | Lanes 2/4 fine but duplicative; lane 3 low value |
| 8 | product | 4.5 | Lanes 4/5 premature (Watch8/12/Play before Watch6 exists), mispriced |

Common blind spot: none inspected the live lanes; all treated 50 worlds/h as "unverified", not as ~100× off. All respected the contract (Mac-only paced/strict/serving, bare wins ties) and were honest about history.

## 3. Merged portfolio (priced 2.0 u/h; observed 1.54 gives ~25 % headroom)

Units are **not** the scarce resource (prepaid, expiring, no other use); scarce are quiet-Mac hours, Sol days and the 3-assignment cap. So bounded decisive studies are worth running on Colab even at the 10× price, but open-ended label farming is not until the engine is fixed.

| # | Lane / question | HW | Units / wall | Start | Kill | Pri |
|---|---|---|---|---|---|---|
| L0 | **Engine placement**: can an L4 lane exceed 150 rows/s? Rerun the A5 bench with 10 workers × 1 thread (per-world serial engine → parallel across worlds), 1 CPU-runtime probe (cpu_count + same bench), 2-world unpaced cost smoke on L4 | L4 ×2, CPU ×1 | ≤ 10 u / 1 day | now (A5 #2 ends ~14:00) | none; it is a measurement | 1 |
| L1 | **T34 deciders** (sequence intact): A5 result → Addendum C (pre-data) replacing the 50/h floor with a priced admission (Round 0 ≤ 180 u, ≤ 4 days) → **Round 0** 240 paired worlds → **unpaced retention arm** re-priced from the L0 smoke (≤ 150 u) | L4 ×3 | 120–160 u Round 0 (~30 h on 3 lanes at the measured rate; ~6 h if L0 gives 5×); 60–150 u unpaced | Round 0 Oct 21 after ratification; unpaced Oct 23 | registered rules; INCOMPLETE on cap | 1 |
| L2 | **Claim A chain**: R1/R2 labels (800 worlds each) — on Colab only if L0 reaches ≥ 300 rows/s/lane (≈ 50 u/round), else on the Mac after retention (~20 h, free); CUDA fit qualification (≤ 10 u); ~10 cached fits (≈ 3 u each, measure first); bare-student 96/32/32 paired screens (~1.6M rows ≈ 30 u each) | L4 | 160–310 u | Round 0 GO | P7 tripwires; R2 fails → close | 2 |
| L3 | **Projection fix** (distill B + skeptic/strategist/selfplay residual arm, merged): 2×2 = {frozen loss vs KL-anchor-only-on-nondecisive + correction on rescue rows} × {trainable B8 vs frozen trunk + zero-init residual}, 1 seed each, +2 seeds on the winner; same labels, cache, tripwires; screens as L2 | L4 | ~140 u | R1 labels exist | no arm ≥ 70 % held-out decisive agreement with ≤ 5 % all-length disagreement, or all screens mass UB90 ≤ 0 | 2 |
| L4 | **Search successor offline** (search/gpuops/evalstats lane 2): at most two frozen successors vs v2 vs bare on 96 fresh paired worlds; which one depends on retention attribution (deadline loss → cheaper schedule/branch reuse; no-rescue → two-deviation beam at equal branch budget) | L4 | ~150 u (searched world ≈ 2 u) | paced retention verdict + attribution (~Oct 25) | < 10 % extra rescues of the failure cohort or > 5 % lost rescues; mass gain vs v2 UB90 ≤ +20 | 3 |
| L5 | **Independent census / replication** (gpuops 4, evalstats 5, skeptic 5): one frozen finalist replicated on 128/mix fresh paired worlds + failure census (no rescue / stale / re-death / spillover); without a finalist, the census alone | L4 | 100–150 u | Nov 3 | fixed sample; no extension | 3 |
| R | Reserve, released weekly to registered follow-ups only | — | ~150 u | — | — | — |

Total ≈ 950–1,100 u. Fits and screens are the only GPU-bound Colab work; they are where an L4 beats the Mac.

**Week by week (≤ 3 assignments, no idle allocations).**
- Oct 20–26 (~300 u): L0 probes (Oct 20–21, 2–3 short jobs); Round 0 on 3 lanes (Oct 21–23); unpaced arm on 3 lanes (Oct 23–25); CUDA fit qualification on a freed lane (Oct 25–26). Mac: Phase 0 → quiet timing gate → pre-check → paced retention (done ~Oct 24).
- Oct 27–Nov 2 (~300 u): on Round 0 GO, R1 labels (Colab if L0 passed, else Mac), fits + R1 screens on 2 lanes; L3 2×2 fits on lane 3. Retention GO → Mac strict gate for Claim B at once (contract §5). Retention attribution → L4 pilot.
- Nov 3–9 (~300 u): R2 labels/fits/screens; L4 successor screens; freeze finalist(s) Nov 9.
- Nov 10–16 (~150 u): L5 replication/census on Colab; Mac strict + serving qualification (Watch6 only).
- Nov 17–20 (~100 u): reserve on registered follow-ups; closure report. Unused units are reported, not burned.

## 4. Not funded

strategist lane 2 (endpoint labels at 4,096 roots; infeasible and T4-derivative); product lanes 4–5 (H20000 Watch sessions; Watch8/12/Play before a Watch6 candidate exists); selfplay lane 3 (population allocation); search lane 3 (learned ordering; T5 FAIL family); distill lane C at 6,400 worlds (keep its 2×2 idea only if the engine reaches ≥ 300 rows/s); A100/G4/H100 for anything host-bound; Colab "strict" or confirmation evidence (Mac only); T23/league/PPO/reward grids; any allocation held idle waiting on the Mac or Sol.

## 5. First three lanes this week

1. **L0 engine placement (today).** After A5 #2 terminates (~14:00), dispatch one L4 bench with `T34_WORKERS=10, T34_THREADS=1` (same bank, 1,500 s cap, ≤ 1 u) and one CPU-runtime bench (≤ 0.2 u); run the 2-world unpaced cost smoke on L4 with 6 parallel worlds (≤ 1 u). Record rows/s per lane and s/world. Sol: if workers×1 helps, make worker count a manifest knob; no algorithm change.
2. **L1 Round 0 (Oct 21).** Operator writes Addendum C pre-data: A5 STOP acknowledged; admission by price (≤ 180 u, ≤ 4 days wall, chunk = floor(2.5 h × measured paired worlds/h)); ratify Round 0 with the L0-measured paired rate; dispatch 3 lanes. The Mac stays on Phase 0 → retention.
3. **L1 unpaced arm (Oct 23).** Amend the 42.533 u envelope pre-data from the L0 smoke (parallel worlds per lane; ≤ 150 u); dispatch on 3 lanes after Round 0 chunks clear. Meanwhile qualify the CUDA fit path on a spare lane so L2/L3 fits can start the day R1 labels exist.
