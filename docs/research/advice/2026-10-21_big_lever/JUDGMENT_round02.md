# Round 2 judgment — big-lever research rounds

Judge: Fable (round-2 judge), 2026-10-21 ~00:45 PT. Inputs: `../CHARTER.md`, `../round-01/JUDGMENT.md`,
`../LIVE_FACTS.md` (23:45 PT entry; no E1–E5 results yet), `QUESTION.md`, `astra_01..10.md`. Repo verified read-only
at main `6f62f1e8` (Sol's frozen E1/E2/E3 packages: `research/gpu_sim_spike/KILL_RULE.md`, `research/bl_ppo/INTENT.md`,
`research/bl_info_value/PROTOCOL.md`; E2 `source-authority.json` says NOT_DISPATCHED). Sealed study not opened. No jobs.

## (a) Bottom line

1. The round answered its question with a near-unanimous mechanism statement (02, 03, 05, 06, 07, 08, 09, 10 agree on
   every clause): the learner must optimise the **exact finite-horizon H5000 alive-mass return (γ=1, zero bootstrap at
   death and at frame 5000, bootstrap at rollout cuts)** through a **separately refitted critic that knows the remaining
   horizon**, from a **calibrated B8 warm start with a measured cumulative drift bound on ordinary states**, with **fixed
   opponents and natural resets**, and with exact-sim counterfactuals used first as a **measurement**, not a label.
2. Discounting was not the killer: at γ=.998 a consequence 100 decisions away still carries 0.819 weight (03, 05); what
   γ=.998 did lose is 77 % of the post-decision-3,200 tail (05), and replay DQN carried the rest through five-step
   bootstrapping (`train_m3c.py:184` N_STEP=5, verified) under a v8 anchor. Round 1's "no new return machinery" was
   wrong (02, 05 right): complete-life return accounting, finite-horizon value normalisation and a horizon-aware critic
   are the engineering centre of lever 1.
3. **E2 as frozen has two gaps** that would make a null uninterpretable: `research/bl_ppo/INTENT.md` gives the critic no
   remaining-time input (B8's 60 scalars carry no episode clock — verified `src/simd_env/ego_raster.py:62`,
   `ego_raster_b.py:14-30`), and is silent on bootstrapping at rollout cuts. E2 is NOT_DISPATCHED, so these are
   pre-data amendments (E8 below), not retrospective changes.
4. The round's main product is the **single observable** six advisors converged on independently (02 K, 03, 04 prefix
   advantage, 05 D, 06 E, 08 escape dividend): the **early-prefix escape dividend** — paired remaining-H5000 mass from
   roots ~100 decisions before B8 enclosure deaths, candidate steering the next 32–64 decisions then B8 resuming,
   versus uninterrupted B8, with matched ordinary roots as counterweight. UB90 ≤ 0 on the early stratum with an
   adequately fitted critic = *wrong mechanism*; LB90 > 0 with pooled whole-world mass still inconclusive = *needs
   scale*. It costs ≤4 u and must be registered before any E2 endpoint is inspected (E6).
5. Exploration is the under-specified half of the mechanism (04 right, 02/08 concede): 1 % non-greedy is ~50 random
   departures per surviving life, and the probability of a specific two-step alternative at specified decisions is
   ~4e-6. Whether an earlier reachable frontier exists for B8-local exploration is testable for ≤20 u (E7) and
   decides whether a restart archive (Go-Explore/Backplay style, not T6's fixed death-window TD) joins lever 1.
6. Throughput reality (08, 01, 10): without a device engine, CPU collection buys ~1–4k hero transitions/s/runtime;
   100M per seed is a week object, 10^9 is not. The engine (lever 2) remains the only route to 10^9 and 01 has lowered
   its own expectations (20–60k L4 world-steps/s, "conjecture"); 01's three-contract kernel plan replaces Round 1's
   "label propagation" suggestion, which is wrong for serpentine corridors.
7. Red team (10) is partly right: a +10 pilot GO buys *replication*, never 10^9; at illustrative SE≈18 a null arm clears
   +10 ~29 % of the time (09, arithmetic checked). E2's GO must add LB90 > 0 with Holm across the two arms (E8).
8. Shortlist unchanged in order, reweighted: lever 1 (40 % push / 20 % beats B8 at bar), engine 30 %, representation
   30/20, search-in-the-loop 20/15 (06's design is now the right form), new lever 7 restart archive 10 % as add-on.

## (b) Per-advisor verdicts

| ID | Persona | Verdict | Why (verified facts in **bold**) |
|---|---|---|---|
| 01 | GPU-native sim engineer | **keep** | Only advisor who re-read the kernels: **`v8_device.py:143` loop bound `min(h*w, _AREA_MAX)`; `_CAP_MIN/_CAP_MAX/_AREA_MAX = 32/160/4096` bound *count*, not radius (`:23,319,404`); `ego_device.py:144` H×W loop; CPU release test marks `seen` only after the rel-TTL check (`v8_nb.py:149-160`); reach-time keeps 24 synchronous rounds with tail-wait semantics (`ego_raster.py:43-48`)** — all verified. Three distinct reachability contracts (tiled union-find for uncapped region size; per-world sparse-frontier BFS for v2/v5/v7; untouched reach-time) is the correct E1 brief. Its realistic-accounting demand (1,024–4,096 worlds, L 800–1,500, real five opponents incl. `device_policy.py:153-168` loop) supersedes the spike's serial-root benchmark. Lowered its own Round 1 numbers honestly. |
| 02 | Large-scale on-policy RL | **keep** | Fullest recipe: critic-only remaining time, ≥2M critic-refit transitions with held-out late-body EV ≥ .1, decaying KL-to-initial coefficient, 256-step unrolls + 32 burn-in, equal-mix episode weighting. Factual catch verified: **E2's boost tripwire 3×B8 = 3×0.12 % = 0.36 % sits below the contract's 0.5 % floor (`SUCCESS_CONTRACT.md:32`)** — a tension to log, not a bug. Its K diagnostic = the round's observable. Correct that an inconclusive 8M pilot cannot separate sample hunger from wrong mechanism — hence E6/E7. |
| 03 | Plasticity / forgetting forensics | **keep** | Corrects the Round 2 fact sheet: **T4c −9.9 and T6 −40.1 are vs paired CTRL, not vs B8 s3** (`t4c_screen_result:11-15`, `t6_screen_result:11-15`, verified; both still far above incumbent). **T4c consequential drift 3.81 % vs B8's 0.99 %; decisive agreement at gap 100–300 only 39.46 % despite 99.82 % in-window** (verified) — the sharpest evidence that fine-tunes change many ordinary decisions while missing the target exception. **T6 fatal-Q preference moved only −0.0096; shared-survivor length −24..−28** (verified). Rightly rejects "loss of plasticity" as diagnosis without a matched learning-speed test; per-update KL .01 is not a cumulative anchor. Its arm C (frozen trunk + 17k-param residual) is the cheapest containment arm for a follow-on. |
| 04 | Exploration / restart distribution | **keep** | The round's missing half. **T6 spec: restart-frac 0.2 from 10,885 death-window states L≥700 (`t6_restart_arm_spec:6-19`); restarted L≥800 death rates 0.31–0.34 vs fresh 0.18–0.21; boost < 0.45 %** (verified) — T6 raised hard-state exposure without transfer, not a boost collapse. **T9: no clean early separator; survivors occupy pockets too (`t9_divergence:7-20`)** verified, so "reward open space" is unjustified. Its discovery experiment (restore 100/150/200 decisions pre-death, 8 stochastic 64-decision B8 prefixes, confirm under independent RNG) is the cheapest test of whether a 1 %-exploration learner can ever find the alternative. Its archive proposal is lever 7 (add-on). Caveat: a Go-Explore archive changes training occupancy and PPO ratios do not correct it — it must be a separately named arm. |
| 05 | Long-horizon credit assignment | **keep** | Best arithmetic of the round: γ=.998 weights the 1,700-frame tail at decision 3,200 as 395.7 frame-equivalents (23.3 %) yet weights a 100-decision-away event at 0.819 — so "γ too small to see the pocket" is false, "γ undervalues the tail" is true. **Implementation trap verified: `train_m3c.py` mass reward is `κ(1−γ)Φ(L)` (`:10-16, 307-310`); at γ=1 the term vanishes** — Sol must build E2's reward from the endpoint definition, never from that code path. PopArt for the fresh critic, λ=1 vs .995 as a follow-on arm, RUDDER only as variance reduction. Its D observable is the counterfactual-branch form of E6. |
| 06 | Search-in-the-loop | **partly** | Now the right form for lever 4: current-policy continuation Q̂ₖ(s,a) refreshed every 2M transitions, policy target q ∝ πₖ exp(Â/10), critic-only arm to isolate calibration, "net executable improvement" E measured before/after an update on unseen roots. **D0 KILL is mechanical, no registered decision (`CAMPAIGN_LOG.md:97`); T3 LABELS 99.64 % bank agreement with 67.27 % trajectory disagreement, 55.33/75.15 of the loss on common-alive frames (`t3_diagnostics:27,141,181-185`)** verified. Not funded now: its 60-u pilot needs a measured 20k branch world-steps/s that does not exist, and it would contaminate the clean R/F comparison (02 right). Deferred behind E1 and E6. |
| 07 | Representation / partial observability | **keep** | E3 refinement adopted: give arm B the same *current* opponent tokens as C so C's gain is temporal only; report early-root regret separately; the ≤1 % unnecessary-switch rule is 40× the 0.025 % intervention prevalence and needs ~300 independent negatives — a screen, not a safety bound. **Serving caches one observation/forward per frame (`web/backend/ego2s_policy.py:7-21`); Watch6 update p95 ≤ 11.688 ms (`SUCCESS_CONTRACT.md:42-49`)** verified. Correctly withdraws its Round 1 eight-iteration planning module as an unjustified latency commitment. |
| 08 | Distributed RL systems | **keep** | Placement and accounting are right and verified: **12 vCPU = 6 physical cores; A100 hero DQN 4,062–9,138 tr/s, learner-bound by a8 (`t16_colab_smoke02:3-17`)**; **CLI qualification excludes exact continuation and measured only a 64 MiB export (`colab_cli/QUALIFICATION.md:56-68`)**. Budget 1–4k useful hero/s/runtime (2k central); 1B at 2k/s = 139 L4-h = 214–278 u — affordable in units, not in calendar at three slots. Its escape-dividend bank (128 pre-death + 128 matched ordinary roots) is E6's bank spec. Right that Round 1's 20k floor omits learner duty cycle. |
| 09 | Evaluation / statistics skeptic | **keep** | Withdraws PQN-first; defines the three admission layers (engine fidelity under identical events / artifact admission with closed-loop Mac gameplay since **`reference.py:117-122` never reruns policies** / release evidence). Operating characteristics: equal-mix G, SE≈18 under paired SD 200 → null arm > +10 ~29 %, two arms ~50 % (correlated by shared B8 rows); Holm two-arm one-look needs G ≳ 29.6; power 30/72/95 % at +20/+40/+60. **P7 unit-of-inference table and Holm rule (`P7_STATISTICS.md:10-24, 38-39`)** verified. Its demand for a simulated joint OC before data is E9. |
| 10 | Red team: B8 is the ceiling | **partly** | Honest that the ceiling is a conjecture; **T10 rejects a universal ceiling (`t10_certificate_horizon:29-38`) and T8's 294k classifier is not a capacity sweep** verified; **T23 all five seeds negative, survival unchanged/better (`RESULTS_T23.md:7-10`)** verified. Its per-world mass-ledger identity (both-alive growth + candidate-only − B8-only) is adopted as a mandatory E2 readout. Where it is wrong: demanding G ≥ 40 from a feedforward 8M endpoint as the refutation bar makes the pilot unfalsifiable in the other direction; a positive early-prefix dividend plus a positive G at +10 with LB90 > 0 is enough to keep funding, and replication (not 10^9) is what it buys. Its "if E2 fails, make E3 the week's decision" is the correct NO-GO branch. |

Factual errors found: none that change a conclusion. Corrections to the Round 2 fact sheet: (i) T4c/T6 deltas are vs
CTRL (03); (ii) D0 is a mechanical failure awaiting rerun, not a registered KILL (06); (iii) E2's boost tripwire is
below the contract floor (02); (iv) E2's critic lacks a horizon input and a rollout-cut rule (judge, verified).

## (c) Disagreements ruled

1. **Cumulative B8 anchor (03, 02 decaying) vs "do not silently add an imitation penalty" (09, 10).** Both right at
   different times. For E2 (two arms, frozen): add *measurement* only — KL(B8‖π) and greedy disagreement by length
   decile on a fixed ordinary-state bank every 1M transitions — no new loss. If E2 shows drift > ~4 % consequential
   disagreement (T4c's level) with a null or negative dividend, 03's arms B/C become the registered follow-on, named
   as a separate treatment. 03 is right that per-update KL .01 cannot bound cumulative displacement; 09 is right that
   an unmeasured permanent anchor would reintroduce the S1/T4c failure mode under another name.
2. **Restart curricula: 04 vs 02/03/05/07/08 ("no death-window restarts").** Rule: not in E2 (keeps the R/F comparison
   clean), but 04's *discovery* test runs now (E7) because it answers a question every other advisor assumed: whether
   B8-local stochastic exploration can reach a better prefix at all. T6 refutes fixed-fatal-state TD restarts, not
   successful-prefix backward curricula; 04 is right about that distinction.
3. **Label propagation (Round 1) vs tiled union-find / per-world BFS (01).** 01 wins: neighbour-min propagation
   needs rounds ≈ path length, which in a serpentine long-body corridor is thousands; the H×W bound is the symptom,
   not the cause. E1's brief is amended to 01's three contracts and 01's benchmark workload.
4. **Search-in-the-loop now (06) vs after E1+E6 (02, 03, 05).** Defer. 06's form is the right one for lever 4 and its E
   observable is the correct target, but it needs a measured branch-rollout rate and would be a third treatment inside
   a two-arm pilot. Round 1's ruling stands: search enters inside the loop, after the engine qualifies.
5. **What a +10 E2 GO licenses: 09/10 vs Round 1's "GO to scale".** 09/10 win. GO = replicate (two fresh seeds × 100M,
   endpoint-only) and run E6 on them; 10^9 waits for the engine and for a dividend that survives replication.
6. **"Wrong mechanism" criterion.** 02, 04, 05, 06, 08 propose equivalent early-root dividends; 10 proposes whole-world
   G with the mass ledger. Rule: E6's prefix dividend is the mechanism test (it ends ≥36 decisions before the fatal
   event, so it cannot be bought by caution); 10's ledger is the mandatory *explanation* of whatever G is. Both go on
   every endpoint. A dividend requires a critic that passed its held-out EV check (02) — otherwise the result is
   INCOMPLETE, not "wrong mechanism".
7. **A100/H100.** Unanimous: same 12-vCPU host, no measured deadline benefit; stays L4 until a device engine moves the
   bottleneck (consistent with the A100 verdict in memory).

## (d) Running shortlist (carried from Round 1, updated)

| # | Lever | Conf. push / beats B8 at bar | Cost to first signal | Most informative next experiment |
|---|---|---|---|---|
| 1 | **Long-horizon on-policy RL from B8** — now specified: exact H5000 return, horizon-aware refitted critic (PopArt), calibrated warm start, cumulative-drift measurement, fixed opponents, natural resets | 40 % / 20 % (down 5: two E2 gaps found, noisy +10 bar) | E2 20 u + E6 4 u; replication 2×100M ≈ 60 u | E2 (amended per E8) + **E6 early-prefix escape dividend** on both endpoints |
| 2 | **GPU-resident exact engine** (enabler for 10^9 and lever 4) | 30 % reaches ≥20k long-body L4 world-steps/s this week (down 5: 01's own revision) | 20 u + Sol 2 days (three kernels) + 4–7 days qualify | E1 with 01's kernel plan and realistic benchmark (1,024–4,096 worlds, L 800–1,500, five real opponents) |
| 3 | **Representation: full-res geometry (+ history) student** | 30 % / 20 % (unchanged) | 16 u, Sol 1–2 days | E3 as frozen, with 07's opponent-token control on arm B if still pre-reconstruction |
| 4 | **Search-in-the-loop** (06's refreshed current-policy Q̂ targets, E observable) | 20 % / 15 % (up 5: better form) | blocked on E1 rate; 60 u pilot | After E1 CONTINUE and E6: 06's 3-arm pilot (PPO / +critic / +policy target) |
| 5 | Learned world model | 5 % / 5 % (down) | — | Only if E1 STOPs and lever 1 shows a dividend |
| 6 | League / history curriculum | 5 % / 5 % | — | Not funded |
| 7 | **Restart archive from confirmed early prefixes** (Go-Explore/Backplay form; add-on to lever 1) — new | 10 % as add-on | E7 20 u, Sol 1 day | **E7 frontier-existence test**; transfer arm only if ≥20 % of worlds yield a confirmed prefix |

## (e) Still unresolved

- Whether an earlier reachable frontier exists for B8-local exploration (E7), and at what non-greedy rate the pilot
  should explore — 1 % was chosen for warm-start fidelity, not for discovery.
- Whether E2's critic can reach held-out late-body EV ≥ .1 at all on B8 inputs; a failure there is "critic
  preparation", not "wrong mechanism", and would push E3's result forward.
- Collector throughput (E4) and the assignment cap (E5): everything from 8M vs 20M sizing to the week's slot plan
  depends on them.
- Transfer from a positive early-prefix dividend to pooled H5000 mass across the three mixes: nobody quantified how
  many rescues, at what collateral cost on ordinary roots, +40 requires (05's arithmetic: saving ~12 % of comparable
  worlds at L≈1,000 from frame 3,300 ≈ +40 if growth is preserved) — Round 3's question.
- The parity doctrine for GPU-trained artifacts is now specified in three layers (09) but not yet written into a rule
  file; needed before any 100M tranche.
- Whether the owner can raise the three-assignment cap (self-imposed per LIVE_FACTS; test pending).

## (f) Experiments Claude can run now (≤20 u each; dispatch via Sol between rounds; report in LIVE_FACTS.md)

**E1 (being built) — amend the brief, keep the rule.** Sol implements 01's three contracts: (i) uncapped ego region
size by tiled connected-component union-find (local labels → boundary unions → pointer jumping → integer size
reduction, convergence-terminated with a visible watchdog); (ii) v2/v5/v7 count-capped searches as per-world
sparse-frontier BFS (one block per world/slot/action, atomic dedupe, device-side termination at empty frontier or
cap, rejected release cells stay unseen, scratch sized for queued cells); (iii) reach-time untouched (24 rounds,
tail-wait semantics). Benchmark 256/1,024/4,096 worlds, L 800–1,500, all three mixes, five real opponents, resets
and collection writes, three warmed repeats, p95 tick. Rule unchanged (`KILL_RULE.md`: ≤0.1× cost, ≥99 % agreement,
0 unexplained, decisive-root audit, <20k long-body world-steps/s ⇒ STOP). Cap 15 u + 5 u, 48 h.

**E2 (being built) — pre-data amendments (E8, 0 u, Sol 0.5 day; commit before any collection).** (1) Critic receives
remaining horizon (5000−t)/5000 as a critic-only scalar; actor receives nothing new. (2) Rollout cuts bootstrap V;
death and frame 5000 bootstrap zero; the reward is built from the endpoint definition (alive mass/1000 per frame),
never from `train_m3c.py`'s `κ(1−γ)Φ` path. (3) Critic refit on ≥2M transitions under the frozen stochastic initial
actor, PopArt-normalised targets, held-out late-body (L ≥ 800) explained variance ≥ .1 and lower error than a
time/length baseline before actor unfreeze; failure = INCOMPLETE (critic), logged. (4) Log every 1M transitions:
KL(B8‖π) and greedy disagreement by length decile on a fixed 10k ordinary-state bank; stochastic vs greedy mass;
long-body exposure (decisions at L ≥ 800). (5) GO additionally requires LB90 > 0 after Holm across the two arms;
record R−F descriptively. (6) Boost: keep the 3×B8 tripwire; flag (not kill) endpoints below the 0.5 % contract
floor. (7) Mandatory mass-ledger decomposition per 10's identity on every endpoint. Everything else in `INTENT.md`
stands (8M default, E4-bound sizing, 20 u / 10 L4-h cap, one look).

**E6 — Early-prefix escape dividend (the mechanism observable). ≤4 u, Sol 0.5–1 day; register before any E2 endpoint
is inspected.** Bank: from a fresh collision-audited development namespace (never the strict banks, never the sealed
study), play B8 to collect 128 worlds with a terminal hero death at frame ≥ 2,000 and L ≥ 700, and 128 mix-, length-
and time-matched ordinary roots from surviving worlds; roots at 100 decisions before the recorded death (ordinary:
matched frame). Snapshot full sim/opponent/RNG state and the actual 128-decision prefix (reuse the T4b snapshot
tooling, `docs/research/specs/t4b_snapshot_restart_spec_2026-10-16.md`). Protocol per root, per controller
(R, F, B8): candidate greedy for 64 decisions, then B8 greedy to the original H5000; two shared future RNG streams;
recurrent state reconstructed from the actual prefix; opponents react. Secondary: candidate greedy to H5000.
Observable: EPD = equal-stratum mean paired (candidate − B8) remaining mass-sum / 5000, world-clustered one-sided
90 % bounds; ordinary stratum reported as the collateral guard. Cost: 256 roots × 3 × 2 = 1,536 continuations ≤
7.7 M world-steps on L4 CPU (12 processes) ≈ 2 allocation-h; cap 4 u. Decision (frozen, one look, after the critic
EV check): early-stratum UB90 ≤ 0 ⇒ **wrong mechanism for this recipe** (both arms) — the week pivots to E3's
branch; LB90 > 0 with ordinary stratum ≥ −2 ⇒ **needs scale** is admissible even if E2's whole-world G is
INCONCLUSIVE; anything else INCONCLUSIVE. Missing snapshots/prefixes ⇒ INCOMPLETE.

**E7 — Frontier-existence test (does B8-local exploration reach a better prefix?). ≤20 u, Sol 1 day.** From the same
128 late-death worlds, restore 100 / 150 / 200 decisions before death. Per root, 8 stochastic-B8 prefixes (temperature
giving 5 % non-greedy — deliberately above E2's 1 %) of 64 decisions, then greedy B8 to H5000, all six legal actions
allowed, opponents reacting; select the best prefix per root on training RNG, then confirm it under two independent
future RNG streams. Observable: fraction of source worlds with a *confirmed* prefix ≥ +20 remaining mass-sum/5000 and
no loss of terminal survival vs paired B8. Cost ≤ 15.4 M + 4.6 M world-steps; at an assumed 1k world-steps/s/process
× 12 on L4 CPU ≈ 1.5–2 h; cap 10 L4-h / 20 u, first cap wins. Decision: < 20 % of worlds ⇒ the 1 %-exploration pilot
is unlikely to discover rescues on its own: register a raised-exploration or archive arm (lever 7) *before* any 100M
tranche, and prioritise lever 4 (search supplies the frontier). ≥ 20 % ⇒ lever 7 transfer arm is admissible (80/10/10
occupancy, separately named). Report the 100/150/200 strata separately; the horizon at which the fraction collapses is
the mechanism's effective reach.

**E9 — Operating-characteristics simulation (Mac, 0 u, Sol 0.5 day; publish before E2 data).** Using paired SDs from
the T23 and S1 screens (not the sealed study), simulate the joint pass/kill/inconclusive probabilities of E2's
amended rule (two arms, shared B8 rows, Holm, survival band v2, scripted no-death ≥ 0) at true G ∈ {0, +20, +40, +60,
+80} and survival Δ ∈ {−.05, 0, +.03, +.06}. If the null-pass probability of the family exceeds 10 % or power at +40
is below 60 %, raise the screen to 128/48/48 worlds before dispatch (cost inside E2's cap at the E4-measured
episode rate, else INCOMPLETE and say so).

**E3 (being built) — keep as frozen; one pre-data amendment if reconstruction has not started:** arm B receives the
same current-step opponent tokens as arm C (position, heading, displacement, length, boost, life mask) so that C − B
measures temporal information only; report early-root (ancestor) regret separately from decisive-root regret. If
reconstruction has started, leave E3 untouched and register this as E3b.

**E4, E5 — unchanged and prerequisite** (collector ruler ≤2 u; assignment-cap test ≤1 u). E2 sizing binds to E4's
immutable receipt per `INTENT.md` §4.

Deferred: 06's 3-arm search-in-the-loop pilot (after E1 CONTINUE + E6); 03's anchor arms B/C (after E2 drift
readout); 05's λ=1 vs .995 and RUDDER arms (after E6 sign); 04's transfer arm (after E7 ≥ 20 %); any A100/H100 use.

Budget this inter-round: E1 20 + E2 20 + E3 16 + E4 2 + E5 1 + E6 4 + E7 20 ≈ 83 u, within three slots over ~2 days.
