# Round 6 judgment — big-lever research rounds

Judge: Fable (round-6 judge), 2026-10-21 ~04:30 PT. Inputs: `../CHARTER.md`, `../round-01..05/JUDGMENT.md`, `../LIVE_FACTS.md`
(newest entry **00:40 PT: Round 0 CLOSED — integrity KILL stands, data sealed, no outcome read; re-registration only if a lane
commits to consume it**; still no L0/L1/L2/E1–E17 scientific result), `../QUEUE.md`, `QUESTION.md`, `astra_01..10.md` (advisors read
snapshot `264aff40`). Repo verified read-only at main `e70f4df0` (three commits after the snapshot: E16 audited execution/resume
wiring, BL prefix Mac proof). The smoke artifact `t34-pathb-smoke-20261021/` (80 `n1-*.json`, 80 `lad-*.json`) and the dense
ceiling receipts `t3-rollout-labels-20261015/dense/ceiling/` (64 worlds) were read; two zero-unit computations below are mine.
Sealed study not opened. No jobs run.

Repo state that matters (verified at `e70f4df0`): **E8 still not landed** (`research/bl_ppo/model.py:101` `self.critic = nn.Linear(256, 1)`;
E16's own `RichCritic` has the horizon clock and PopArt, `research/bl_e16/model.py:65-113`, but that does not qualify E2's parent).
**`research/bl_served_search/` does not exist on main** (Sol is building L2/L0 there per the launcher's note); `exp/bl-l2/allocation-01/`
holds only a bank allocation. **L2 cannot be run with `ceiling.py --trigger-mode dense` as Round 5 wrote**: the harness loads census
baselines (`ceiling.py:58-67`), caps `--worlds` at 32 and `--workers` at 3 (`:263-269`), `batched_rollout` raises on any non-fast-anchor
opponent (`engine.py:99-104` "supports only stateless fast-anchor opponents"), and `build_replay` asserts no Vector61 runtime is
controlled (`research/t1_commitment_20261015/replay.py:185-189`). E1b is prepared, undispatched (`exp/bl-e1b/DISPATCH.txt`).
`research/bl_e16/REGISTRATION.md:86-89` still says `100M_ADMISSIBLE` (06, 07 right: it must be read as 32M admission per Round 5 §c1).

## (a) Bottom line

1. **All ten advisors answer the question the same way, and the record confirms them: the mechanism's unmeasured frozen/mixed value
   binds first (L2), and the served form that failed was not "too slow" but mis-targeted.** My recount of the 75 N=1 T3 records
   (`recount`, this judgment): the first fully evaluated improving candidate arrived inside the *last-decision* lead in **57/75**
   (22/34 rescue-capable, 35/41 delay-only; **0/5 M3**; 50/68 excluding the 7 deliveries) — 03's numbers, verified; Round 5's 53/68
   was a mis-stated denominator — but **in 0 of 75 did it complete before its own target frame**: median own-target slack 0.625 s
   (15 frames; min 0.125 s, max 9 s) against 5–12 s of compute; median compression needed **10.0×**, minimum 1.03×. The chronological
   doom→LAD scanner searches the frames just ahead of the live frame, whose deadlines are ~0.6 s away. Round 5's "a k=2 engine moves
   53 → 58/68" is withdrawn: at k=2 the delivered count stays near zero because the targeted roots have no slack at any k below ~10.
2. **The T3 rule is a certificate rule, and that changes the serving arithmetic (02 right; quantified here for the first time).**
   The operator maximises H300 survival with ties kept at B8 (`ceiling.py:140-143`, verified), so a root whose bare continuation
   survives H300 can never change an action, and the shared greedy shadow certifies it for free (`engine.py:47-50` already does this
   for one root). From the 64 dense receipts: **440 of 36,996 batches (1.19 %) had the bare continuation die within H300**; all 75
   changes are among them (365 had no better alternative); they occur in **22/64 worlds**, and six worlds hold **64–76 consecutive**
   such roots (a 320–380-decision doom window, ≈ 15 s at 24 fps). Steady-state demand is therefore the shadow advance only;
   the real demand is a **burst of 4.8 roots/s for up to ~15 s**, nonmonotone in rescue availability. Round 5's "4.8 batches/s
   continuous = 6.34× one worker" is withdrawn as the necessary condition.
3. **What is necessary instead (my ruling, from the smoke's own records):** (i) the shadow must detect doom early enough — in **7/34**
   rescue intervals the historical change frame was already behind the live frame at detection, and the D300-survival LAD lead ranges
   **−45..433 frames (median 309)**; cold shadow fill cost up to **9.2 s** (median 1.9 s); (ii) per-root latency S must leave reachable
   roots: the D300-rescue LAD is reachable in **29/34 at S = 1 s, 27/34 at S ≤ 5 s, 21/34 at 8 s, 17/34 at 12 s** (today's one-thread
   latency), and the exact H500 improving LAD in 73/75 at 1 s, 61/75 at 5 s, 29/75 at 12 s, **1/75 at 18 s**; (iii) burst capacity W/S
   must cover the reachable window — at S = 5 s the window holds a median **38.5 periodic roots** (24 at S = 8 s, 4.5 at S = 12 s) and
   closes in ≤ 12 s, so searching half of it needs ≥ 2.4 roots/s sustained for a burst: **W = 4 ⇒ S ≤ 1.67 s; W = 2 ⇒ S ≤ 0.83 s;
   W = 1 ⇒ S ≤ 0.42 s** with zero overruns. The binding constraint is therefore *root targeting + burst coverage + shadow lead*,
   in that order, and the engine's per-batch latency matters only through those three.
4. **The owner's pending clause (one extra CPU thread → registered W ≤ 4 worker processes) decides which of those is reachable this
   week.** Under **YES**: a W ≤ 4 process pipeline needs S ≤ 1.67 s per 6-action H300 batch on Watch6 opponents — T3's measured 1.32 s
   was on scripted opponents and the Phase 0 batched engine measured 1.0–1.7 s CPU / 0.7–1.2 s MPS on a different N8/H128 fixture, so
   this is plausible and must be measured (L1′). Under **NO**: W = 1 needs S ≤ 0.42 s, ≈ 3× under T3 and 2.5–4× under Phase 0's
   batched CPU numbers with no measured basis; the lever is then dead on this Mac this week unless L1′'s one-thread cell lands below
   0.42 s. The amendment must also register **K outstanding requests** (01, 05): both numbers are part of the Claim B identity.
   The owner should be told plainly that "no" closes the lever they asked about, not merely slows it.
5. **L2 is the first dispatch but is neither built nor priced.** 08 and 10 are right: it is a one-Sol-day runner (fresh banks, isolated
   per-branch opponent runtime — Vector61 carry, v8 state, observation history — and a parity receipt under `research/bl_e16/PARITY_RULE.md`
   §1–2 against the Mac reference on fixed roots) and its cost is 92,480 batches at an unmeasured aggregate rate R: 7.4 / 13.8 / 22.4
   L4 allocation-h at R = 4 / 2 / 1.2 batches/s (08's table, checked). Round 5's 8 h / 16 u is withdrawn; the envelope is repriced to
   **≤ 32 allocation-h / 64 u incl. 20 % reserve, first cap wins, NOT_DISPATCHABLE on forecast overrun**, with a disjoint long-body
   timing fixture before dispatch. Its KILL label is renamed **PACKAGE_KILL** (04, 05, 08, 09, 10 unanimous: an unpaced controller is not
   a mathematical ceiling on a deadline-skipping one) but remains the funding gate: no paced package without frozen and mixed LB90 > 0.
6. **Arithmetic corrections to Round 5 (all verified):** the comparator is s3, so "zero frozen/mixed gain" gives **+40.9 pooled, not +31**
   (06, 08, 10) — a margin of 0.94 mass, i.e. scripted-only transfer would need 97.7 % retention (10); E17 evaluates **8 × 288 = 2,304
   episodes, not 864** (07) — 15.6 u + 4 reserve at 900 world-steps/s, over cap at 800/s; L2's 16 u was never a forecast. None reverses
   a ruling; all change dispatch arithmetic.
7. **Identity and evaluation for a wall-clock controller are now specified (05, 09, 10 agree):** the candidate is planner + scheduler +
   deadlines + fallback + load envelope; "delivered" = consumed by the tick before its frozen cutoff for the exact session/life/frame/
   checkpoint/shadow epoch; three oracles (semantic, availability-trace replay, fresh paced); every miss counts; one final α = .05 family
   across one bare and one assisted artifact (Holm .025/.05), a reserved third slot for planner-over-RL-winner; six zero-overrun
   sessions still leave a 39 % upper bound on session overrun probability (09, checked). Adopted as the rule for Rounds 7–10.
8. **The RL/representation line is untouched by all this (07, 09):** E8 → E16 → E17 under Round 5's funding rules; E16's `100M_ADMISSIBLE`
   label needs a prospective erratum (32M admission), not a byte edit. 07's continuation-supplier assay (an RL endpoint as the planner's
   rollout policy, same B8 fallback, 8 u) is the only new RL item and is conditional on L2/L1′. **Shortlist:** served search stays first
   but drops to 35 % / 20 % (unbuilt L2, bursty demand, pending clause); representation 30 / 18; PPO backbone 25 / 10; engine 30 %;
   search-in-the-loop 15 / 10; archive 6; opponent 4; world model / league 2.

## (b) Per-advisor verdicts

| ID | Persona | Verdict | Why (verified facts in **bold**) |
|---|---|---|---|
| 01 | Apple-silicon latency / systems | **keep** | Capacity arithmetic right (`ηW/S ≥ λ` ⇒ S ≤ 167/333/500/667 ms at W = 1–4 for *continuous* 4.8/s; K = ⌈600/208⌉+1 = 4). **`v8_nb.py:98` `nogil=True`; `app.py:37` one intra-op thread (measured ~3× oversubscription)** verified. Its layout (1 Watch6 lane, 1 shadow/coordinator, 3 search workers, 1 headroom on six P-cores; E-cores for housekeeping; QoS is guidance, not pinning) and its L1 reporting list are adopted. Correction: its 500 ms target is sized to continuous demand; the certificate rule (§a2) makes the burst-coverage cells (§f L1′) the ones that decide. |
| 02 | Real-time search serving | **keep** | The round's key insight: **`engine.py:47-50` returns bare if its continuation survives; `ceiling.py:140-143` ties keep B8** — so "4.8 roots/s is the peak decision demand, not 4.8 expensive batches/s" is exactly right, and my dense-receipt census (1.19 % of batches, bursts of up to 76 roots) quantifies it. **Phase 0 two-worker speedup 1.51×** (`SPEED_REPORT.md:34-40`) verified. Reachable-root EDF with horizon checkpoints 32→96→192→300, bounded K, descendant invalidation, and ≥ 28/34 as the delivery bar are adopted. Its `dτ ≥ .681` (checked) is a planning identity, labelled as such. |
| 03 | Smoke forensics (adversarial) | **keep** | Every number re-derived from the records: **57/75, 22/34, 35/41, 0/5, 50/68**; **`b8_s3_12-c00`: detection at live 3057, LAD 3490, first improvement 11.734 s later belongs to frame 3068 (0.458 s ahead) ⇒ 25.6× compression**; **4180/4185 consecutive changes (`s3_8-c02/c03`)**; **M3-frozen-0: LAD 2143, actions 0/1 survive H500, bare 2 dies in 3 steps, 7.152 s maintenance, no completed improving evaluation**; **`lad-*.json` hold only kind/lad/outcomes/probes/rule** — all verified. Its structural point (L0 cannot replay a LAD-first schedule from these records) is correct and supersedes Round 5's L0; my own-deadline table is what L0 can still say. |
| 04 | Expert player / body-routing | **keep** | **Selector maximises alive transitions, not mass (`ceiling.py:140-143`)**; **mixed rosters alternate checkpoint-pool and random_safe slots (`dev_screen.py:358-378`)**, not "half greedy_food" — verified. Its L2 readout (exposure, baseline death-frame × length strata, closure attribution by T1's rule, timing vs closure, productive vs D300-only vs re-death, exact ledger with no summed root dividends) is adopted as L2's mandatory rows. Illustrative +10/mix at q=.25, f=.30, tail 240, C=8 (checked) is labelled a sensitivity input. |
| 05 | Evaluation / trust skeptic | **keep** | The identity test (semantic oracle, availability-trace oracle, fresh paced runs with no best-of-N), the "delivered" definition, the fallback-reason partition and the one-family α rule are adopted verbatim (§a7). **`controller_v2.py:137-172` already separates accepted results from changed actions and fences after intervention** — consistent with my reading of `engine_v2.py:769-783` (late candidates rejected inside the engine, so controller late discards are 0 while 475 evaluations were late). SE = 11.79 at 96/mix (checked). |
| 06 | Representation / latency | **keep** | **8,640 branch rows/s vs 24 live rows (360×); 1 ms per six-row step = 1.44 core-s/s; two workers ⇒ ≤ 417 ms/batch, 1.39 ms/step** (checked). **E3 stem has two stride-2 convs then pooling (`students.py:23-35`)**; **`REGISTRATION.md:86-89` `100M_ADMISSIBLE`** verified. Five-cell E14 (B8 / F / F+G / B8+planner / F+G+planner), executing the encoder at W = 0, is adopted as E14's cell list, planner cells after L2/L1′. Geometry-inside-rollouts is correctly separated as its own expensive hypothesis. |
| 07 | Large-scale on-policy RL | **keep** | **E16 `RichCritic` has the remaining-horizon clock and PopArt (`bl_e16/model.py:65-113,128-138`); E2 does not (`bl_ppo/model.py:101`)** verified. **E17 invoice: 8 × 288 = 2,304 episodes** — Round 5's 864 was wrong; 15.56 u + 4 reserve at 900/s, over the usable allowance at 800/s. Its continuation-supplier assay (search with C vs B8 continuation, same B8 fallback to H5000, 48 source worlds, ≤ 8 u) becomes E20, conditional. Right that a better bare policy need not improve search and vice versa. |
| 08 | Distributed systems / Colab | **keep** | The round's most consequential build fact: **`replay.py:185-189` asserts fast-anchor opponents and uncontrolled Vector61 runtime; `engine.py:99-104` raises on stateful opponents; `ceiling.py` caps 32 worlds / 3 workers and loads census baselines** — L2 is a rebuild, not a flag. **Cap 3 at `supervisor.py:817-818`** verified. Cost table (7.4 / 13.8 / 22.4 h; R ≥ 4.76 batches/s needed to fit 16 u) checked. Its one-day runner + half-day parity audit (4 u) and the 32 h / 64 u envelope are adopted; its conditional 384-pair confirmation (144 u) is deferred to the plan rounds. |
| 09 | Sequential design / calibration | **keep** | OC table checked (SE = 16.67 at 64/64/32 with SD 200; KILL 86.8 % at 0, 10.0 % at +40; GO 29.7 % at +40 — a reason to call INCONCLUSIVE likely and to size the confirmation, not to loosen the rule). Staged graph `L0 ∥ L2 → L1 → paced smoke → development qualification → strict` with DELIVERY_READY as the smoke's positive label, **39.3 % one-sided UB after six zero-overrun sessions** (checked), the [.10, .30) pause branch, and proper scoring of E17 forecasts are adopted. Right that search KILL does not calibrate RL forecasts. |
| 10 | Red team: served search is a mirage | **keep** | **122.812 on 32 reused scripted worlds (`t3_ceiling_test:119-129`); 40.937 pooled if frozen/mixed are zero; 97.71 % retention needed; S0 long-body mixed fixture 67 % in opponent veto; 384 H5000 runs = 22.2 Mac-h** — all verified/checked. Its decisive measurement (paced equal-mix H5000 gain on 32 fresh worlds/mix, twice, every fallback charged; UB90(G_paced) < 40 closes) is adopted as L4, after 09's smoke. Its narrower position — stop the week's systems investment when the registered delivered controller fails, protect the bare-net line — is the ruling. |

Factual errors found: none in the advisors that change a conclusion. **Corrections to Round 5 (judge's own, verified):** (i) 53/68 → 57/75
inside the last-decision lead, 0/75 inside own-frame deadlines; (ii) "k=2 moves 53→58" withdrawn (median compression needed 10×);
(iii) "4.8 batches/s continuous is necessary" withdrawn — the T3 rule certifies 98.8 % of roots for free; the demand is a ≤ 15-s burst;
(iv) "+31 if frozen/mixed are zero" → +40.9 (s3 row); (v) E17 864 → 2,304 episodes; (vi) L2 "unchanged `ceiling.py` run, 16 u" →
rebuild + reprice; (vii) L0's LAD-first replay is structurally INCOMPLETE from the retained records.

## (c) Disagreements ruled

1. **Periodic (02, 03, 04, 10) vs doom-triggered (smoke form) vs anytime/EDF (01, 02).** Under the T3 rule these are one controller:
   the periodic clock plus a shadow certificate *is* a doom trigger, and the only live design choice is the **root-targeting rule inside
   the burst**. Ruling — the "reachable-root schedule": on shadow doom at frame d with live frame ℓ and measured per-root latency S
   (p99), search periodic roots r ∈ [ℓ + S·24 + margin, d) in EDF order across W processes, horizon checkpoints 32→96→192→300, bounded K,
   publish only exact H300 winners, invalidate all descendants on any executed deviation/respawn/epoch change, bare B8 on every miss.
   Chronological-from-live scanning (the smoke) is retired this week (02, 03, 05, 10). Delay-only improvements are published only when
   no surviving alternative exists in the searched set (03's preemption mechanism is logged, not fixed, this week).
2. **Is 4.8 batches/s necessary? (01, 03, 06, 08, 09, 10 assumed yes; 02 no).** 02 wins on the receipts (§a2). L1's kill rule is
   rewritten around burst coverage (§f L1′); W = 2 "covering 4.8/s on paper" (Round 5) is withdrawn in both directions — it is neither
   needed continuously nor sufficient in a burst without S ≤ 0.83 s.
3. **L0: time-divided replay (Round 5) vs forensic bookkeeping only (03, 02, 05, 07, 10).** 03 wins; L0 becomes L0′ — the own-deadline
   table, the LAD-reachability curves (computed here) and the dense-receipt demand census, formalised by Sol at 0 u. No k× replay.
4. **Is L2 an upper bound on paced search? (04, 05, 08, 09, 10: no).** Agreed: label renamed PACKAGE_KILL; but it stays the funding
   gate, because "skipping harmful interventions helps" is a conjecture with no mechanism on record, and the paced controller's misses
   are not chosen by value. A paced controller may be re-proposed after a PACKAGE_KILL only with its own registered unpaced-vs-paced
   evidence.
5. **L2 cost and buildability (08, 10 vs Round 5).** 08/10 win (§a5). A100/H100: unanimous no for the eleventh time; CPU-bound work.
6. **Geometry inside rollouts now (06, partly) vs feedforward continuation first (07).** 07 wins for this week; 06's planner cells enter
   E14 only after L2 GO + L1′ pass; E20 tests an RL endpoint as continuation before any geometry-in-rollout cell.
7. **Family of claims (05) vs development/final separation (09).** Both adopted: development prints (E16, E15 looks, L2, L3, L4) fund;
   the final family (α = .05, one bare + one assisted + one reserved planner-over-net slot, Holm, intersection-union per artifact) is
   registered before any strict bank is touched. Bare wins an unresolved tie; owner OK unchanged.
8. **Mac paced calendar (05: 12/mix × 2; 09: 16/mix + 6 sessions; 10: 32/mix × 2 = 32 Mac-h).** Staged: 09's smoke first (≤ 9 Mac-h,
   DELIVERY_READY), then 10's development efficacy (32 Mac-h), then a fresh strict bank. That is ≈ 2 Mac-days of exclusive paced time
   on days 3–6 if everything passes; nothing else paced runs then. The RL line's strict gates compete for the same hours — Round 7's
   question.
9. **The owner's clause.** Ruled in §a4; the amendment text is drafted in §f (W ≤ 4 processes *and* K ≤ 8 outstanding, one math thread
   each, QoS declared, both identity-bound). The judge cannot grant it; the owner's answer selects which L1′ cells count.

## (d) Running shortlist (carried from Round 5, updated)

| # | Lever | Conf. push / beats B8 at bar | Cost to first signal | Most informative next experiment |
|---|---|---|---|---|
| 1 | **Served exact-sim search on this Mac** — T3 rule as a certificate controller + reachable-root burst schedule, W worker processes, K-deep speculation, Claim B identity (planner + scheduler + deadlines + fallback) | 35 % / 20 % (down 5/5: L2 unbuilt and dearer; demand bursty; 7/34 dooms detected too late; clause pending) | L2′ 64 u + 4 u audit + Sol 1.5 d; L1′ 0 u (quiet Mac); L3/L4 ≈ 41 Mac-h | **L2′** (frozen/mixed value; PACKAGE_KILL at pooled UB90 < 40) then **L1′ burst-coverage probe** (S per batch on Watch6 opponents; roots/s at W ∈ {1..4}) |
| 1c | RL endpoint as planner continuation (07's E20) — sub-arm | conditional | 8 u | only after L2′ GO + L1′ pass + a pre-nominated endpoint |
| 2 | **Representation: full-res geometry residual on the F parent** | 30 % / 18 % (unchanged) | E3 16 u + E14 (five cells) + E16-G | E3 → E14 five-cell → E16-G |
| 3 | **Long-horizon on-policy RL from B8** (backbone) | 25 % / 10 % (unchanged) | E8 0 u → E4 → E2 → E12 → E16 → E17 (2,304 episodes) | **Land E8** (still missing at `model.py:101`) |
| 4 | **GPU-resident exact engine** (Colab only) | 30 % (unchanged) | E1b 15 + 5 u, prepared | dispatch E1b as briefed |
| 5 | Search-in-the-loop (one-deviation current-policy targets) | 15 % / 10 % | after E1b | E16-S |
| 6 | Restart archive / PLR | 6 % | E7 + E10 | E10 U envelope |
| 7 | Opponent-conditioned add-on | 4 % | after E3b | — |
| 8 | World model; league | 2 % / 2 % | — | not funded |
| 1b | Retention sub-arm | conditional | 36 u | on the collateral trigger |

## (e) Still unresolved

- **T3-dense's frozen/mixed gain** — still the single number that decides lever 1; L2′ is now a build (runner with isolated stateful
  opponent carry + parity receipt) and a priced dispatch, not a flag.
- **S per 6-action H300 batch on Watch6 opponents (FRP3 + v8) from a live-shaped world**, on one thread, CPU-batched, and MPS — never
  measured (T3's 1.32 s is scripted; Phase 0's 1.0–1.7 s is N8/H128). Everything in §a3–4 keys off it.
- **Shadow lead / detection latency**: 7/34 rescue intervals were already behind the live frame at detection with cold fill up to
  9.2 s; a warm, continuously advanced shadow at ≥ 300 frames lead has not been measured on the serving path.
- **Whether the reachable-root schedule finds a *rescuing* root when the last improving root is a delay** (66/75 exact LADs are
  delays): the per-root outcome map inside doom windows exists only in the dense receipts (scripted); L2′ must export it for
  frozen/mixed.
- **The owner's clause** (W and K) — pre-data identity components; the judge cannot grant them.
- **Mac paced-hour budget**: search qualification (≈ 41 Mac-h) and the RL line's strict gates share one calendar — Round 7.
- E8, E16 `100M_ADMISSIBLE` erratum, E17 invoice, E13 family file — unlanded; Round 0 closed with no consumer, so no lane may assume
  the teacher's offline ceiling.

## (f) Experiments Claude can run now (≤ 20 u each unless stated; dispatch via Sol between rounds; report in LIVE_FACTS.md)

Mac served-search line (development evidence; nothing touches a registered study):

**L0′ — Own-deadline forensics and demand census (0 u; Sol 0.25 day; replaces L0).** Publish in `research/bl_served_search/`:
(1) per-interval own-target slack (first improving candidate's frame − detection live frame) vs compute time, with the fixed denominators
75 / 34 / 41 / 5; (2) LAD-reachability curves vs S ∈ {1, 2, 3, 5, 8, 12, 18} s for the exact H500 LAD and the D300-survival LAD (this
judgment's numbers are the expected output: 61/75 and 27/34 at 5 s; 29/75 and 17/34 at 12 s; 1/75 at 18 s); (3) from the dense
receipts, per-batch bare-dies flags: 440/36,996, per-world counts, consecutive-run lengths, and for each run whether/when a surviving
alternative appeared. No k× replay (INCOMPLETE by structure). Kill: none — this is bookkeeping that sizes L1′.

**L2′ — Dense ceiling on fresh frozen/mixed worlds, rebuilt and repriced (≤ 64 u incl. 20 % reserve; Sol 1 day runner + Codex 0.5 day
parity; first dispatch after the parity audit).** Runner: fresh collision-audited development banks 64 frozen / 64 mixed / 32 scripted
(s3 only, never strict banks, never E6/E7/E16 worlds), unchanged T3 operator (every 5th decision from first L ≥ 600, 6 advisory actions,
H300 = min(300, 5000 − t), survival-max, ties → B8 → Q → index, B8 greedy continuation, eps 0, no veto), **independent per-branch
opponent runtime** (Vector61 carry, v8 state, observation history cloned per branch; `replay.py:185-189` and `engine.py:99-104`
replaced, not bypassed), deterministic shard merges. Parity audit (≤ 4 u): 18 mix × length fixtures + 96 preselected roots incl. near
ties and frozen carry; branch scores, chosen actions, opponent actions and first-divergence state vs the Mac reference per
`PARITY_RULE.md` §1–2; any unexplained semantic mismatch blocks dispatch. Pricing: time three disjoint long-body fixtures per mix on
the runtime; forecast = 92,480 batches / R + 1 h; dispatch only if forecast ≤ 80 % of the envelope; else NOT_DISPATCHABLE (no bank
shrinking). Readout: 04's strata (exposure, baseline death-frame × length, closure attribution, timing vs closure, productive /
D300-only / re-death, exact three-term ledger) + per-batch bare-dies and alternative-survives flags exported. Decision (frozen, one
look): pooled-equivalent UB90 < 40 ⇒ **PACKAGE_KILL** (this unpaced design; funds no Mac systems work this week); frozen LB90 > 0 and
mixed LB90 > 0 and pooled point ≥ 40 ⇒ **FUND_SYSTEMS**; else INCONCLUSIVE (09's OC: ≈ 60 % at a true +40 — plan for it).

**L1′ — Burst-coverage latency probe on this Mac (0 u; quiet window ≥ 2 h, AC, lid open; Sol 1 day; after L2′ FUND_SYSTEMS, or in
parallel if the Mac is idle, since it is timing-only).** Part A, per-batch latency S: one live-shaped world (L 800 / 1,200 / 1,500,
Watch6 FRP3 + v8, S2 native v8 on), one 6-action × H300 batch, engines (i) T3 as run, (ii) Phase 0 batched CPU one thread, (iii) same with
B8 forward on MPS (synchronised timing); report exclusive stage times (sim, observation, hero v8, opponent v8, Vector61/FRP3, B8 forward,
clone/digest/IPC) p50/p95/p99 and the per-root actions vs the reference (identical or the cell is INVALID). Part B, burst coverage with
the Watch6 server paced at 24 fps: W ∈ {1, 2, 3, 4} worker *processes* (spawn, one math thread each, QoS as 01's layout), a scripted
12-s burst of 60 pending roots spaced 5 decisions apart ≥ S ahead of live; report roots completed in 12 s, deadline-valid fraction,
queue depth K needed (⌈S_p99/0.208⌉ + 1), update p95/p99, overruns, refill after a forced invalidation. Kill: no W ≤ 4 cell delivers
≥ 2.4 valid roots/s over the burst with zero overruns and update p95 ≤ 11.688 ms ⇒ continuous served search is unservable on this Mac
this week (report the shortfall). **Under an owner "no" on the clause, only the W = 1 cells count** (bar: S ≤ 0.42 s).

**L3 — First paced smoke (0 u; ≤ 9 Mac-h; Sol 1–2 days; after L2′ FUND_SYSTEMS + L1′ pass + the clause amendment).** 09's design:
freeze one controller (reachable-root schedule, W, K, QoS, warm-up, load profile, fallback rule); 16 fresh paired worlds/mix at 24 fps +
six paired 10-min Watch6 sessions; availability-trace replay must reproduce executed actions and scores; predeclared fixtures exercise
stale rejection and deadline fallback. Labels: identity failure ⇒ INVALID; timing-bar breach ⇒ STOP; UB95(G_paced) < 40 ⇒ STOP;
timing + identity pass + ≥ 1 exercised valid intervention ⇒ **DELIVERY_READY**; none exercised ⇒ INCONCLUSIVE.

**L4 — Paced development efficacy (0 u; 32 Mac-h; after DELIVERY_READY).** 10's design: 32 fresh worlds/mix, candidate and B8, each
twice at absolute 24-fps deadlines, randomised order, world-clustered; equal-mix paired gain, survival/boost guards, three-term ledger,
every miss charged. UB90(G_paced) < 40 ⇒ artifact closed; point ≥ 40 and LB90 > 0 and guards and zero overruns ⇒ nominate for a fresh
strict bank (separate, Mac-only, owner OK unchanged).

**Contract amendment text for the owner (0 u; identity component, pre-data):** "Path B may use **W ≤ 4 worker processes** (one math
thread each, QoS declared) and **K ≤ 8 outstanding requests**; W, K, the root-targeting rule and the fallback rule are frozen before any
efficacy test and are part of the candidate's identity; all other §4 clauses (never blocks a tick, deadline miss → bare, misses count,
visible labelling, rollback pins B8 s3) unchanged."

Colab RL / representation line (unchanged rules; corrections only):

**E8 — still unlanded; still blocks E2 (0 u).** **E16 erratum 02 (0 u):** `100M_ADMISSIBLE` reads as admission to the 2 × 32M
futility-only tranche per Round 5 §c1; prospective note, no byte edit. **E17 invoice (0 u):** 2,304 episodes; require a measured
disjoint full-policy rate ≥ 900 world-steps/s or NOT_DISPATCHABLE. **E14 (0 u, Mac quiet window):** 06's five cells with the encoder
executed at W = 0; planner cells deferred until L1′. **E20 — continuation-supplier assay (≤ 8 u; Sol 0.5–1 day; conditional on L2′ GO +
L1′ pass + a pre-nominated endpoint C):** 48 fresh source worlds (16/mix), prespecified root rule, identical 6-action H300 searches with
B8 vs C continuation, chosen action executed then the same B8 fallback to H5000; paired remaining mass/5000; UB90 ≤ 0 ⇒ supplier rejected.
**E1b, E3, E4, E5, E6/E7 (+E10), E12, E13, E15, E16 (+rows), E17, E19 — unchanged.**

**Dispatch order:** L0′ (now) → L2′ parity audit → **L2′** → E8 (commit) → D0 rerun → E4 → E5 → E10 → E6 + E7 → L1′ (first quiet window;
cells per the owner's answer) → E1b → E3 → E2 → E12 → E14 → E16 → E17 → L3 → L4 → E20 → E13/E15 → 32M tranche. Budget this inter-round:
L2′ 64 + audit 4 + E1b 20 + E2 20 + E3 16 + E4 2 + E5 1 + E6 4 + E7 20 + E12 16 + E16 20 + E17 20 + E20 8 ≈ 235 u; L0′/L1′/L3/L4/E8/E14 and
the amendments are 0 u. Mac paced hours reserved if the search line passes: ≈ 41 h (L3 + L4) on days 3–6. A100/H100: no.
