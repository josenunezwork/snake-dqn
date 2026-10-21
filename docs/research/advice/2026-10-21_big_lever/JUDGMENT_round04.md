# Round 4 judgment — big-lever research rounds

Judge: Fable (round-4 judge), 2026-10-21 ~02:30 PT. Inputs: `../CHARTER.md`, `../round-01..03/JUDGMENT.md`, `../LIVE_FACTS.md`
(23:58 PT entry is the newest; still no E1–E14 scientific result), `../QUEUE.md`, `QUESTION.md`, `astra_01..10.md` (advisors
read snapshot `f77e0305`). Repo verified read-only at main `d2454013` (four commits after the advisors' snapshot: BL prefix seed
allocation, native CPU collectors + recoverable L4 PPO entrypoints). Sealed study not opened. No jobs run.

Repo state that matters for this judgment (verified): (i) **E8 has not landed** — `research/bl_ppo/model.py:101` still
`self.critic = nn.Linear(256, 1)` on the core, no remaining-horizon input, no PopArt; `INTENT.md` unchanged at 3bedc5de;
`exp/bl-e2/source-authority.json` NOT_DISPATCHED. QUEUE's "after bl-e2 finishes" trigger has now fired (db44a13c). (ii) **E6/E7
have had no gameplay**: allocation rounds 0–5 were REJECTED (`exp/bl-e6e7/rejected-round-{0..5}.json`: seed collisions, then an
unsafe inventory path); the frozen allocator now tries whole rounds 1–100 (`research/bl_prefix/ALLOCATION_ADDENDUM.md:1-16`);
`SEED_AUDIT_PREFLIGHT.json` status PASS on candidate 8. So **E10 is still pre-data** and must be registered before the bank is
played. (iii) No `FINITE_MENU` / envelope text exists anywhere in `research/bl_prefix/` — E10 is not yet written.

## (a) Bottom line

1. **The round converged on one measurement, and it is not any of E6/E7/E10/E3/E12.** All ten advisors independently refuse to
   rank mechanisms on dividends, envelopes, regret or explained variance (09's table: each of those "cannot establish an
   executable policy rescue rate"). The ranking statistic is **natural-start, candidate-owns-the-whole-life, paired H5000 mass
   on fresh worlds** — equal-mix `G` vs B8 (and, for arms grown from a common parent, the incremental `J` = post-block minus
   parent) — explained by the three-term ledger, a *productive rescue* count (B8 dies, candidate survives to H5000, candidate's
   mean length over B8's dead tail ≥ 90 % of B8's last living length), newly caused deaths, and collateral defined as negative
   common-alive mass plus B8-only mass averaged over **all** natural worlds. Nine of ten proposed the same ≤20-u job under
   different names; I register it once as **E16** (§f) with named arms, a common parent and simultaneous bounds.
2. **The arithmetic is now consistent and tighter than Round 3 said.** With tail value 306 and C = 10, +40 needs
   `(40+10)/306 = 16.3` added full-horizon survivors per 100 natural worlds (04, 08, 10); at 15 rescues the collateral budget is
   **5.9**, not 10 (02: 15 × 306 = 45.9). At q = .4 that is 41 % of eligible deaths rescued and executed (10), 58 % of eligible
   worlds needing a *productive* prefix at 70 % transfer (03), or `f·τ ≥ .375` (06). E7's 20 % threshold is an *investigation*
   threshold: even granting every confirmation a full rescue it supplies ≈ +14.5 (10). Nobody disputes this.
3. **Round 3 was wrong on three clauses, withdrawn here.** (i) `q × EPD` is unsafe: the registered EPD is the *equal-weight
   early/ordinary average* (`REGISTRATION_E6.md` §observable, verified) — the bridge must use `q_m × mean(early)_m` with ordinary
   collateral measured on natural worlds, not on matched survivors (03, 09, 10). (ii) "FINITE_MENU_INSUFFICIENT may be escaped
   by a registered non-B8-local design change" is too permissive — a *name* supplies no frontier (03, 09, 10); replaced in §c3.
   (iii) E7's confirmation (`+20 AND no loss of terminal survival in each stream`, `REGISTRATION_E7.md` §confirm, verified)
   admits delay-only prefixes; E10 must count *productive* confirmations separately (03).
4. **Opponent-sealed deaths are not a lever.** All 13 resolved opponent sealers are `scripted:greedy_food`
   (`t1_commitment_study:75-76`), a deterministic state rule with fixed tie-breaks (`src/simd_env/fast_anchors.py:3-15`,
   verified) — history adds no information about them (07, 10); perfect opponent-only rescue at q = .4 is ≤ 10.6/100 ≈ +24.5
   after collateral (05, 07, 10 agree). Current-token geometry is the right form; history only after E3b.
5. **Curricula cannot raise natural prevalence.** An 80/20 replay mixture enriches eligibility ≤ 1.3× at q = .4 (06); T6 raised
   L ≥ 800 death exposure 1.65× (.1965 → .3241 per 1,000, `t6_screen_result:36-37` verified) with zero scripted survival gain.
   PLR-style replay survives only as an optional arm of lever 7, never as a lever.
6. **Pure exploration is demoted from discovery mechanism to backbone.** At 1 % non-greedy a specific two-step alternative
   anywhere in a 100-decision window has probability ≈ 4e-4; over 8k eligible lives (100M) that is ≈ 3.2 occurrences before
   continuation or learning (02, arithmetic checked). PPO remains the learner; the frontier must be supplied by geometry
   (E3 → E14 → E16-G), by current-policy one-deviation targets (after E1), or by a current-policy archive (after E10 ≥ 40).
7. **Two build-level facts change the queue.** The E3 student and the E14 stub are different architectures (E3: 7 geometry
   planes + one joint `GRU(43, 64)`, `students.py:83-87`, `reconstruct.py:69-81`; E14 as briefed: 8 planes + five 64-unit GRUs)
   — E14 must probe the *E3* architecture or it qualifies nothing (05). The spike's `dense_decision` continues B8, ranks
   survival first and divides mass by depth (`rollout.py:11-19, 60-66`, verified) — the search-target operator is a rebuild,
   not a flag (04). E1's 20k floor counts scheduled lanes including dead/illegal branches (`KILL_RULE.md:12-21`), not useful
   hero transitions (08).
8. Shortlist: representation moves to the top as the conditional favourite of 02, 05, 08, 09, 10 (35 % / 20 %); long-horizon
   PPO 30 / 12 as backbone; engine 30 % unchanged; search 20 / 12; archive 8 %; opponent-aware new at 5 % add-on; world model and
   league 3 %. The week's bet is decided by E16, not by E2's own GO.

## (b) Per-advisor verdicts

| ID | Persona | Verdict | Why (verified facts in **bold**) |
|---|---|---|---|
| 01 | Expert player / body-routing | **partly** | Honest that no measured fraction exists; its 28–48 % repairable prior (hero 50–70 %, opponent 20–40 %) is labelled a planning range. Correct that **E7 confirms action tapes with fallbacks and learns no trigger** (`REGISTRATION_E7.md` §confirm); correct that geometry exposes exits but cannot create trajectories. Its proposal 2 (fit residual actors on E7's successful prefixes, then natural-start evaluation) is a reasonable *arm* but conditions on E7 successes that may not exist; folded into E16-G/E16-R. Opponent correction description (turn toward the mouth while arrival margin is positive, else abandon early) is the clearest behavioural spec of the round. |
| 02 | Large-scale on-policy RL | **keep** | Withdraws PPO's first place as the discovery mechanism on its own arithmetic (3.2 two-step occurrences per 8k eligible lives at 1 %). **Collateral budget 5.9 at 15 rescues; .0039 mass per batch update over 1,526 updates** (checked). **`model.py:101` critic has no clock** verified — right that E8 is unimplemented. Its fixed-update-block design (`J` after equal budgets from a common parent, unavailable arms *unranked*) is the fairest form of E16 and is adopted; its 1M-per-arm sizing is too small (§c2). |
| 03 | Exploration / restart | **keep** | Best supply arithmetic: `r = q f τ`, required f = .58 at q = .4, τ = .7 (table checked). **E7 confirmation admits delay-only prefixes** verified; **E6 EPD is the balanced early/ordinary average** verified. Right that an archive replaying the same menu cannot escape U; retracts its own large archive budget; its 80/10/10 transfer assay (natural control vs archive, 8M + 2M each) becomes E16-R, admissible only under U ≥ 40 and productive coverage. |
| 04 | Search-in-the-loop | **keep** | **`rollout.py` continues B8, ranks survival first, divides mass by depth; `device_policy.py:137-138` calls B8** verified — the target operator is new code. NEI on new-policy occupancy as the ranking scalar is the same statistic as E16's `G`. Its two-refresh probe (512 roots/cycle, 81.92M branch steps = 1.14 h at 20k/s vs 11.4 h at 2k/s) is the right ≤20-u form and correctly infeasible on CPU — E16-S waits for E1 CONTINUE. Right to retire the 80-u pilot. |
| 05 | Representation / latency | **keep** | **E3/E14 architecture mismatch** verified (7 planes + joint GRU vs 8 planes + five GRUs) — this is the round's most consequential catch. **100 D300 mass = 6 endpoint mass**; opponent-only ≤ 10.6/100 (checked). Function-preserving `Q_B8/T + Wz` with a separate rich critic, identity checks, and "regret is not rescue probability" are all adopted. Its 2 × 4M geometry-vs-control pair is E16-G. |
| 06 | Curriculum / UED | **partly** | Honest negative: **enrichment ≤ .8 + .2/q = 1.3× at q = .4**; **T6 1.65× exposure with zero survival gain** verified. Its constrained PLR (replay whole worlds from frame 0, no teleport, 1 % per-source cap, final 25 % natural-only) is cheaper than snapshots and is kept as an optional arm E16-P of lever 7; PAIRED/ACCEL correctly rejected. Not a lever: it cannot change natural prevalence. |
| 07 | Opponent modelling | **partly** | Corrects the round's framing: **13 opponent sealers are deterministic greedy_food** (`fast_anchors.py:3-15` verified), so memory cannot add information about them; its own forecast is 3–6/100 as an add-on, **q ≈ .59 needed even at perfect rescue** (checked). G-vs-H design (shared opponent GRU, auxiliary forecasting heads, leave-one-frozen-policy-out) is sound but deferred behind E3b; the sealer-stratified E10 ledger (0 u) is adopted. LIAM is the right precedent, LOLA correctly rejected. |
| 08 | Distributed systems / engine | **keep** | **KILL_RULE floor counts scheduled lanes incl. dead/illegal** verified; **cap 3 is `supervisor.py:818`**; **A100 host 12 vCPU, 4,062 tr/s learner-bound** verified. Compute-shape table is correct and its three-arm common-parent assay (2M each, `S_j = LB90[mean of three mix ΔM]`) merges into E16. Four-receipt `PARITY_RULE.md` (engine / policy / durability / artifact) adopted as E13's rule text. Right that 100M on CPU is hours; wrong that CPU can reach 10^9 this week (278 h at 1k/s in two slots is not a week). |
| 09 | Evaluation / statistics | **keep** | Three corrections adopted: **E6 quota stopping (44/42/42) makes screened N outcome-dependent** — add a stopping-aware (confidence-sequence) sensitivity bound beside the registered Wilson; **U's zero on ineligible worlds is valid only for an abstaining controller; missing eligible outcomes are missing, not zero**; adaptive reuse (mined E6/E7 worlds are burned for evaluation). Its "different readouts cannot share a leaderboard" table is the round's organising principle; its one arms-and-looks contract (family α = .10, simultaneous bounds, E2/E6/E3 nominate rather than add OR-routes) is adopted for E16. |
| 10 | Red team | **keep** | Calendar-and-transfer forecast, explicitly falsifiable. **T3-dense +122.8 on 32 reused scripted worlds; LABELS −55.33 common-alive** verified; `.4 × .2 × 306 − 10 = 14.5` checked. Its per-mechanism "measurement that proves it" list reduces to E16 plus the funnel E10 → E3/E12 → E14; its productive-rescue definition is adopted verbatim. Concedes 15/100 *per mix* is a planning assumption, the pooled contract is the gate — agreed. |

Factual errors found: none that change a conclusion. Corrections to the Round 4 fact sheet: (i) EPD is the balanced early/ordinary
average, so `q × EPD` was wrong (03, 09, 10); (ii) E7 confirmation admits delay-only prefixes; (iii) E8 is not implemented at HEAD
despite "complete" PPO packaging (02, judge); (iv) E3 and E14 describe different encoders (05); (v) the spike's rollout is a
B8-continuation survival-first ranker, not the search-target operator (04).

## (c) Disagreements ruled

1. **What admits a 100M tranche — 04/05/08 ("G ≥ 40 at the pilot") vs 02/09/10 ("LB90 > 0 on an executable statistic, +40 is
   unfalsifiable at ≤ 20 u").** 02/09/10 win: at paired SD ≈ 200 and 96 worlds/mix, SE ≈ 18; demanding point +40 from a 4M block
   makes every mechanism KILL by construction. Ruling (E16 admission, frozen before data): Holm-adjusted **LB90(J) > 0** for the
   arm, **point G ≥ +10 vs B8**, **productive rescues ≥ 5/100 point in each mix** (one third of the target from ~4 % of the
   budget) with tail retention ≥ 90 %, **collateral UB90 ≤ 10**, survival guard A, E12-qualified critic. UB90(G) < 40 kills the
   *artifact* at the bar (10), not the mechanism. Anything else is INCONCLUSIVE and buys no 100M.
2. **Common parent + equal block (02, 08) vs per-mechanism pilots with own critics (03, 05, 06, 07).** Common parent wins for
   comparability (same B8 warm start, same critic preparation, same exploration, same bank); but 02's 16 × 65,536 ≈ 1M per arm
   is below any rescue-event signal. Ruling: **2M shared critic preparation + 4M actor transitions per arm**, control + ≤ 2
   treatments inside 20 u at the E4-measured rate (≥ 2k/s); one treatment if E4 < 2k/s. Treatments that need their own critic
   (geometry) refit on the same 2M under their own inputs; that is part of the treatment.
3. **Does "archive" escape FINITE_MENU_INSUFFICIENT? — Round 3 vs 03/09/10.** Round 3 withdrawn. New rule: under
   `UB90(U) < 40` the B8-menu archive (E16-R) is **omitted**; the admissible frontier suppliers are geometry (E16-G, after E3 GO +
   E14 pass on the E3 architecture) and current-policy one-deviation targets (E16-S, after E1 CONTINUE); any later archive must be
   built from *current-policy* confirmed prefixes and show its own measured envelope first. Under `UB90(U) ≥ 40`, E16-R is
   admitted only with E7 TRANSFER_ADMISSIBLE **and** productive confirmed coverage consistent with `r ≥ .163` at τ ≤ 1 (03's
   τ-requirement test: if `(40 + C)/confirmed gross supply > 1`, reject).
4. **History / opponent modelling now (07) vs after E3b (05, 10).** 05/10 win: the only identified opponent sealers are
   deterministic; current tokens are in E16-G's input; history is funded only if E3 reports C − B ≥ 10 % and E3b confirms it is
   temporal. 07's sealer-stratified ledger is adopted into E10 at 0 u.
5. **Curriculum as a lever (06's own framing is modest) vs "not funded" (10).** Not a lever; E16-P optional under lever 7, and
   only with the final 25 % natural-only phase and the 1 % per-source cap.
6. **Engine necessity (08 vs Round 3).** Round 3 stands: the engine is for 10^9 and for E16-S branch rates (81.92M branch
   steps per two-refresh probe is 11.4 h on CPU — outside any 20-u cap). 08's correction that E1's floor is not a hero rate is
   adopted into E1's reporting: also report useful completed branch world-steps/s and roots/hour.
7. **Statistics of the bridge (09 vs the registered E6/E7 displays).** Keep the registered Wilson/Student-t displays untouched
   (frozen); add 09's stopping-aware sensitivity bound and coverage table as *additional* E10 rows. Decision labels unchanged.
8. **Per-mix 15/100 vs pooled gate (10, 09).** Agreed: 16.3/100 is a symmetric planning scenario; the contract's gate is pooled
   mass with per-mix survival guards (`SUCCESS_CONTRACT.md:27-30`). E16 reports both and admits on pooled `J`/`G`.

## (d) Running shortlist (carried from Round 3, updated)

| # | Lever | Conf. push / beats B8 at bar | Cost to first signal | Most informative next experiment |
|---|---|---|---|---|
| 3→1 | **Representation: full-res geometry (+ current opponent tokens) as a function-preserving residual actor with a separate rich critic** — the conditional favourite of 02, 05, 08, 09, 10 | 35 % / 20 % (up 5 on push: every advisor routes here under U < 40; execution still unmeasured) | E3 16 u + E14 0 u (on the E3 architecture) + E16-G inside E16's 20 u | E3 → **E14 amended** → **E16-G** (geometry residual vs matched control from the common parent) |
| 1→2 | **Long-horizon on-policy RL from B8** — now the *backbone* (exact H5000 return, horizon critic, PopArt, calibrated warm start, measured drift), not the discovery mechanism | 30 % / 12 % (down 5/3: 1 % exploration cannot find two-step corrections at 100M) | E8 0 u → E4 2 u → E2 20 u → E12 16 u | **Land E8** (verified missing), then E2 + E12; E2's F endpoint is E16's common parent |
| 2→3 | **GPU-resident exact engine** (10^9 and E16-S branch rates) | 30 % reaches ≥ 20k scheduled long-body world-steps/s this week (unchanged) | 20 u, dispatch pending | E1 as briefed + 08's useful-rate reporting; search-target operator rebuild (04) only on CONTINUE |
| 4 | **Search-in-the-loop**, one-deviation current-policy Q̂ targets on new-policy states | 20 % / 12 % (down 3: operator is a rebuild) | 20 u two-refresh probe (E16-S) after E1 | E16-S: PPO vs PPO + refreshed targets, 2 × 2M cycles, 512 roots/cycle |
| 7→5 | **Restart archive from confirmed prefixes** (80/10/10, final 25 % natural-only; PLR replay variant E16-P) | 8 % as add-on (down 2: supply arithmetic needs 58 % productive coverage at q = .4) | E7 20 u (shared bank) + E16-R inside 20 u | E10 productive coverage + U; E16-R only under U ≥ 40 and TRANSFER_ADMISSIBLE |
| new 6 | **Opponent-conditioned actor / forecast heads** (07's G-vs-H) | 5 % as add-on | 20 u, after E3b | Only if E3 C − B ≥ 10 % and E3b attributes it to history |
| 5→7 | Learned world model | 3 % / 3 % | — | Only if E1 STOPs and E16-G shows a dividend |
| 6→8 | League / adaptive opponents | 3 % / 3 % | — | Not funded; curriculum folded into lever 5 |
| 1b | Retention sub-arm (03-R3's A/B/C) | conditional | 36 u | On the ordinary-collateral trigger at any E16 arm (UB90 < −2 or common-alive loss < −10) |

## (e) Still unresolved

- **The discovery-density question is unmeasured for the learner that will actually run**: E7 measures 5 % discoverability
  under B8; E2 logs can measure realised non-greedy *runs* (≥ 2 consecutive departures at L ≥ 800) per life — added to E8 (§f).
- **Measured q_m, productive coverage and U** — every number in §a2–3 is conditional on E10, which is unwritten and whose bank
  allocation has been rejected six times.
- **Whether E16's 4M slope predicts a 100M endpoint at all.** Nobody can bound this; no B8-warm-start learning curve exists on
  record that was positive at any block. This is Round 5's question.
- **Serving cost of the E3 architecture** (E14 amended) — lever 1 is unservable until measured; a Mac quiet window is needed.
- **Parity rule file** (08's four receipts + 09's three layers) still unwritten; required before any 100M tranche (E13).
- **E4 rate, E5 cap, T34 slot usage** — E16 sizing (two treatments vs one) binds to E4.
- Adaptive reuse: any world used by E6/E7/E16 training is burned; the replication bank (E15) and strict banks must be fresh.

## (f) Experiments Claude can run now (≤ 20 u each; dispatch via Sol between rounds; report in LIVE_FACTS.md)

Queue status (from `../QUEUE.md` and the repo at d2454013): E1 built, undispatched; E2 built **without E8**; E3 built; E6/E7 built,
bank allocation retrying; E8/E9/E10/E12/E14 queued, none landed. Items below **refine** queued packages or add one new one (E16).

**E8 — E2 pre-data amendments (queued; now unblocked; 0 u; Sol 0.5 day; verified not implemented at `model.py:101`).** As
queued (critic horizon scalar, rollout-cut bootstrap, PopArt refit + held-out L ≥ 800 EV ≥ .1, drift by decile, Holm LB90 > 0,
boost-floor flag, ledger, occupancy log, endpoint ledger by death-frame × length, realised departures by decile) **plus (11)**:
per 1M transitions, count realised non-greedy *runs* of length ≥ 2 and ≥ 3 consecutive decisions at L ≥ 800 per surviving life
(02's sequence density), and the mass outcome of lives containing one. E2 must not collect before E8 is committed.

**E10 — Population bridge and envelope (queued; 0 u; Sol 0.5 day; register before the E6/E7 bank plays — still pre-data).**
As queued (collection ledger, 24 per-root outcomes, U with FINITE_MENU_INSUFFICIENT iff UB90(U) < 40) with four amendments:
(a) the bridge uses `q_m × mean(early stratum)_m`, never the balanced EPD; (b) report **productive** confirmations separately
(candidate survives H5000 and tail ≥ 90 % of B8's last length) and the τ-requirement `(40 + C)/confirmed gross supply`;
(c) sealer-stratified rows (hero / opponent / none / unresolved, reusing T1's method) per mix; (d) 09's coverage table
(eligible outcomes missing ≠ 0) and a stopping-aware confidence-sequence sensitivity bound beside the registered Wilson display.
Frozen E6/E7 labels untouched.

**E14 — Serving-cost probe (queued; 0 u; Mac quiet window; amended).** Probe the **E3 student's exact deployment architecture**
(7 geometry planes, `_RasterTower(7,16,64,(6,10))`, joint `GRU(43,64)` recomputed per frame from a rolling 128-step window) as a
residual on B8 (`Q_B8/T + Wz`), not the 8-plane/five-GRU stub; and a second variant without the GRU. Kill unchanged: full-path
p95 > 11.688 ms or any overrun ⇒ unservable; compress and re-probe before E16-G is allowed into training.

**E16 — Common execution-ranking harness (NEW; ≤ 20 u; Sol 1–1.5 days; after E2 + E12 produce a critic-qualified F endpoint).**
Parent: E2's F endpoint, nominated *before* its efficacy is read. Arms, each 2M critic preparation + 4M actor transitions under the
E8-amended objective, same exploration, optimizer, opponents, equal mixes, natural resets: **E16-A** control (natural PPO);
**E16-G** geometry residual (E3-GO architecture, E14 pass, rich critic refit on its own inputs); **E16-R** archive 80/10/10 (only
under U ≥ 40 + TRANSFER_ADMISSIBLE + τ ≤ 1); **E16-S** PPO + one-deviation current-policy targets, 2 × 2M refresh (only after E1
CONTINUE); **E16-P** PLR whole-world replay (optional, lever 5). Admit ≤ 2 treatments per dispatch at E4 ≥ 2k/s, 1 otherwise;
unavailable arms are UNRANKED, never zero. Freeze endpoints; evaluate parent, B8 and all arms greedy to H5000 on **96 fresh
collision-audited natural worlds per mix**, one look. Primary: Holm/simultaneous world-clustered one-sided bounds on `J` (arm −
parent) and `G` (arm − B8), variance `Σ s²/(9 n_m)`; mandatory explanation rows: three-term ledger by B8 death-frame × length,
productive rescues/100 per mix, newly caused deaths, tail retention, collateral (negative common-alive + B8-only over all worlds).
Cost: 2 + 3 × 4 = 14M hero transitions ≈ 1.9 h at 2k/s; evaluation ≤ 5 controllers × 288 × 5,000 = 7.2M world-steps, allowed only
with a measured disjoint full-policy episode rate ≥ 800 world-steps/s aggregate; 2 h setup/export; cap **10 L4 allocation-h / 20 u,
4 u reserve, first cap wins**; forecast overrun ⇒ NOT_DISPATCHABLE (no bank shrinking). Decision (frozen): integrity/cap/critic
failure ⇒ INCOMPLETE; tripwires, survival band v2 or common-alive loss < −10 ⇒ KILL that arm; **100M admission** per §c1
(LB90(J) > 0 Holm, point G ≥ +10, productive rescues ≥ 5/100 point in each mix, retention ≥ 90 %, collateral UB90 ≤ 10, guard A);
UB90(G) < 40 ⇒ artifact rejected at the bar; otherwise INCONCLUSIVE. Training worlds, archive sources and E6/E7 roots never enter
the evaluation bank.

**E1 — unchanged rule; add reporting (0 u).** Alongside scheduled world-steps/s, report useful completed branch world-steps/s,
roots/hour, and clone/restore cost at L 800–1,500 with five reacting opponents (08, 04). On CONTINUE, Sol's next job is the
search-target operator (current-policy continuation, absolute H5000 horizon, mass ranking) — a rebuild of `rollout.py`, not a flag.

**E12 — unchanged** (critic scale qualification on each E2 endpoint; its PASS on F is E16's precondition).

**E13 — Parity rule file (0 u; Sol 0.5 day + independent check; write before any 100M).** Adopt 08's four receipts (engine identity
on injected candidates; policy agreement with first-divergence explanation on ordinary and decisive roots; durability = exact resume
equality on-platform, real-size exports; artifact = hash-bound actor + fresh closed-loop Mac development evidence + `ΔMac − ΔColab`
within ±10 pooled / ±20 per mix) as `research/bl_ppo/PARITY_RULE.md`; 09's family-α contract (one-sided .10 across all admission
claims; looks spend separately) goes in the same file.

**E3, E4, E5, E6, E7, E9, E15 — unchanged.** E9 additionally simulates E16's admission rule (null-pass ≤ .10 across ≤ 2 treatments,
power at true J = +20).

**Dispatch order (refined):** E8 (commit) → D0 rerun → E4 → E5 → E10 (register) → E6 + E7 (first passing allocation round) → E1 →
E3 → E2 → E12 → E14 (first quiet Mac window, on the E3 architecture) → **E16** (arms admitted by E10/E3/E14/E1 outcomes) → E13 →
E15. Budget this inter-round: E1 20 + E2 20 + E3 16 + E4 2 + E5 1 + E6 4 + E7 20 + E12 16 + E16 20 ≈ 119 u; E8/E9/E10/E13/E14 are
0 u. Deferred: 07's G-vs-H history pilot (after E3b); 03-R3's retention arms (on trigger); 06's PAIRED/ACCEL (never this week);
any A100/H100 use (unanimous again: no measured deadline benefit on the 12-vCPU host).
