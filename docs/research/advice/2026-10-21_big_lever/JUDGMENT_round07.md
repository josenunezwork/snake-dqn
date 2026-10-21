# Round 7 judgment — big-lever research rounds

Judge: Fable (round-7 judge), 2026-10-21 ~05:30 PT. Inputs: `../CHARTER.md`, `../round-01..06/JUDGMENT.md`, `../LIVE_FACTS.md`
(newest entry 00:40 PT; no owner answer on the W/K clause; no L0′/L1′/L2′/E1–E17 scientific result), `../QUEUE.md`, `QUESTION.md`,
`astra_01..10.md` (advisors read snapshot `95d40cd8`, which is also main HEAD now). Repo verified read-only at `95d40cd8`. I also read
`exp/bl-l2/RESULT_L0.md`, `exp/bl-l2/manifests-03/l2-plan.json`, `exp/bl-l2/final-preflight.json` and listed Sol's worktree
`../snake-dqn-bl-l2` (light commands). Sealed study not opened. No jobs run.

Repo state that matters (verified): **E8 still not landed** (`research/bl_ppo/model.py:101` `self.critic = nn.Linear(256, 1)`).
**L2 is built but NOT dispatchable**: Sol's worktree already carries the stateful-opponent rebuild (`ceiling.py:147-184` clones the
Vector61 runtime incl. v8 memory per branch — so Round 6's "rebuild" is partly done), but `l2-plan.json` prints `dispatchable=false`:
conservative forecast **529 min against the still-frozen 384-min usable / 16-u / 480-min envelope**, "no L4 full-body all-mix rate
measured", and the final preflight fixtures fired **0 search batches** ("H500 fixtures do not reach L600"). `PREREGISTRATION_L2.md:43-59`
still says `KILL_SERVED_SEARCH_PUSH` / 16 u; `readiness.py:38-39` still enforces ≤ 384 min. **E16 is BUILD_VERIFIED / NOT_DISPATCHABLE**
(`bl_e16/PREFLIGHT.md:3`) and its admission rule cannot pass at any plausible rate: `admission.py:40-43` requires the 2×-estimate timeout
≤ 480 min, i.e. forecast ≤ 4 h, while one treatment at 2k/s and 900 eval steps/s costs ≈ 5.4 h. **The E2 screen is a Mac CPU job**
(`bl_ppo/PREFLIGHT.md:37-44`: ≈ 8.5 Mac CPU-h, one thread). **Round 5's L0 printed INCOMPLETE** exactly as Round 6 predicted
(`exp/bl-l2/RESULT_L0.md`: records hold no per-evaluation timeline); the L0′ bookkeeping version has not been published.

## (a) Bottom line

1. **The calendar collision the question posed is mostly an artefact of the question's own number, and I withdraw it.** All ten advisors
   correctly recomputed a fully paced 640-pair bank as **74.1 h** (33.3 h is 96/mix × 2 arms — the question mislabelled it). But nine of
   ten then applied 74.1 h to Line R's bare nominee, and that is wrong on the record: **a bare checkpoint's strict gate is unpaced.** T7
   (the B8 s3 STRICT_PASS) ran **5,760 H10000 episodes + 45 H5000 controls in 6.83 h** on ≤ 6 single-thread Mac workers
   (`docs/research/t7_phase_r_result_2026-10-16.md:11`); the bare-unpaced amendment shows paced and unpaced bare outcomes are
   byte-identical and the paired estimand unchanged (`2026-10-20_t34_retention_amendment_bare_unpaced.md:96-110`); and 384/128/128 is the
   **E13 transfer bank**, not a sized strict design (`bl_e16/REGISTRATION_ERRATUM_01.md:11-15`; P7 §1 sizes from development variance,
   ≥ 16 worlds/mix). So **Line R's strict gate ≈ 7–10 unpaced CPU-h; only Line S's candidate arm is paced** (≈ 22 h at T7's 384 worlds,
   ≈ 37 h at 640, with the bare comparator unpaced under the same registered logic). Both lines fit one Mac week *if* Line S is ready by
   Friday; what binds is Sol builder-days, the pending clause and L2′'s dispatchability — not Mac hours.
2. **Line S's critical path is blocked at its first stage and nobody can price it yet.** L2′ cannot dispatch under the frozen 16-u envelope
   (verified above), and no long-body all-mix batch on Watch6 opponents has ever been timed (0 batches in the preflight fixtures). Round 6's
   64-u/32-h repricing must be registered *prospectively* (label → PACKAGE_KILL, cap → 64 u) and a timing fixture that actually fires
   batches must run before dispatch (X0/X1 below). Earliest honest L2′ print: **Fri Oct 23 evening PT**. Therefore the day-3 decision is
   **Fri Oct 23 23:59 PT** (10), not Friday 08:00 (01, 04, 05, 07, 08).
3. **Line R's chain is blocked at its first stage too, twice.** E8 is still absent; and E16 as frozen is undispatchable by its own admission
   arithmetic (forecast must be ≤ 4 h). Both need 0-u pre-data fixes on day 1 with a *separate* builder (07 right: Round 6's serial dispatch
   list wrongly queued E8 behind L2′). The E2 screen's 8.5 Mac CPU-h is a hidden Mac load (07) and must be scheduled in a night window or
   moved to L4 under the E16 evaluation precedent (pre-data amendment).
4. **Trust doctrine is now settled and unanimous (02–09):** a Colab-trained bare net files E13's four receipts (engine identity, policy
   first-divergence audit, exact on-platform resume + real-size export, `ΔMac − ΔColab` 90 % interval inside ±10 pooled / ±20 per mix,
   `PARITY_RULE.md:55-131` verified) before its strict bank; a wall-clock controller files three oracles (semantic; availability-trace
   replay reproducing executed actions and scores; fresh paced, no best-of-N) with the identity bound to planner + scheduler + W/K +
   deadlines + fallback + load envelope; planner-over-net inherits both. Batching, placement and declared float order may differ;
   chosen actions, scores, mechanics, opponents, H5000 accounting and identity fields may not. None of these receipts exists yet; the
   Mac side of E13 is ≈ 1 unpaced CPU-h at T7's episode rate (08's 14 Mac-h assumed a slow runner), so it is not a calendar item.
5. **Release qualification for Claim B (04, verified):** a skill STRICT_PASS survives a product failure as evidence — T12 printed
   SERVING_FAIL / HOLD after T7 passed (`t12/results/REVIEW.md:1-21`) and T7's gate was not revisited — but a timing repair that changes
   deadlines/scheduler/fallback is a *new* Claim B artifact needing fresh paced efficacy; T12's "fix the serving defect only" precedent
   does not transfer automatically (`t12/protocol.md:119-124`). The relative boost floor is Path-B-only; every new registration must decide
   its own boost clause pre-data (`bare_unpaced.md:207-251`).
6. **Round 6 corrected on four points:** (i) the final family with a reserved third slot is **Holm .05/3, .05/2, .05** (02, 04–09; Round 6's
   .025/.05 was the two-slot shorthand) — ruling in §c4; (ii) "half the window at 2.4 roots/s" is inconsistent with its own 12-s window
   (19.25/12 = 1.6/s; 2.4/s is an 8-s window — 02); (iii) **W = 4 at S = 1.67 s needs K = 9 > 8** (01, arithmetic checked:
   ⌈1.667/0.2083⌉ + 1 = 9), so under the drafted clause W = 4 requires S_p99 ≤ 1.458 s; (iv) under an owner "no", the contract's
   "≤ 1 outstanding request" also binds: **W = 1 and K = 1** — no speculation at all (01, 02, 10).
7. **Day 7 has four honest outcomes, not three** (unanimous): qualified / scientific KILL (a registered rule fired) / INCONCLUSIVE (a
   registered one-look printed it) / NOT_READY (calendar or receipts). No relabelling; a line excluded from the week's paced reservation
   keeps its scientific status.
8. **Shortlist:** served search stays first but drops to **30 % / 18 %** (undispatchable L2′, zero timed batches, clause pending — offset by
   the rebuild being partly done); representation 30 / 18 (E16 needs a cap amendment); PPO backbone **22 / 9** (E8 absent on day 1 of the
   week); engine 30 %; search-in-the-loop 15 / 10; archive 6; opponent 4; world model / league 2.

## (b) Per-advisor verdicts

| ID | Persona | Verdict | Why (verified facts in **bold**) |
|---|---|---|---|
| 01 | Apple-silicon systems | **keep** | **K = ⌈S_p99/0.2083⌉+1 ⇒ W = 4 at 1.667 s needs K = 9; S_p99 ≤ 1.458 s under K ≤ 8** (checked). **Under NO the contract's "≤ 1 outstanding request" binds (`SUCCESS_CONTRACT.md:50-54`)** verified. Cold 12-s ceiling table (`W·⌊12/S⌋ ≥ 29`) is a correct lower bound; η = .8 is an allowance, labelled. W = 3 layout (one headroom P-core) adopted as the default finalist. Its "no day-7 guarantee" is the unanimous view. |
| 02 | Real-time search serving | **keep** | The round's sharpest correction: **19.25 roots / 12 s = 1.6/s, not 2.4/s** — Round 6's bar implies an 8-s window. Controller spec (absolute per-root cutoff with two-tick margin, EDF, 32→96→192→300 checkpoints, K = queued+running+unconsumed, lead cap 432 decisions, delay published only if every action at that root dies) is the right frozen identity and is adopted as L3's controller text. **`PREREGISTRATION_L2.md:43-60` still 16 u / universal KILL** verified. Withdraws its own 28/34 bar — correct. |
| 03 | Paced identity / replay | **keep** | **`controller_v2.py:137-172` fences; `:64-110` bounded records; `test_t34_pathb_trace_v2.py:102-115` four-frame trace** verified. Three-oracle package (events/record/replay/semantic/receipt, independent reducer, hash-chained journal, fixture list incl. cutoff ±1 ns, K saturation, worker crash, respawn) is adopted as the Sol spec for `research/bl_paced_identity/`. Divergence doctrine (INVALID_IDENTITY seals the whole evaluation; verifier-only repair may recover raw evidence; action-affecting repair = new artifact) adopted. Its 74.1-h objection applies to Claim B only (§a1). |
| 04 | Release / serving qualification | **keep** | The only advisor who saw that bare nominees can be evaluated unpaced (**clock-independence amendment `:96-116`** verified) and that **the relative boost floor is Path-B-only (`:207-251`)**. **T12 SERVING_FAIL/HOLD after T7; T7 gate not revisited; repair needs a new producer version** verified. Owner packet list (hashes, clause table, rollback matrix, pin schema beyond `served_checkpoint.py:172-217`, seven-day monitoring) adopted. 40-h "only under amendment" strict figure is for Claim B; its "105.7 h after day 3" overstates R. |
| 05 | Evaluation / trust skeptic | **keep** | Final-family registration text (three slots, composite p = max of clause p-values, Holm .05/3-.05/2-.05, unavailable slot p = 1, no reset after failure, bare wins ties) adopted verbatim as the X6 draft. Burn manifest (append-only, ancestors/descendants, prefixes do not refresh a world) adopted. **`readiness.py:31-39` ≤ 384 min; L2 label/cap stale** verified. Pipeline-power arithmetic (three independent +40 point filters ≈ 12.5 % at a true +40) is the reason the plan must minimise *serial* point filters — carried to Round 8. |
| 06 | Representation | **keep** | **E16 `admission.py:32-43`: timeout ≤ 480 ⇒ forecast ≤ 4 h ⇒ the 5.67-h example is NOT_DISPATCHABLE** verified — and worse: one treatment at 2k/s is ≈ 5.4 h, so E16 cannot dispatch as frozen at any plausible rate. **384/128/128 is the E13 transfer bank (`ERRATUM_01:11-15`)** verified. Five-cell E14 split (bare cells now; planner cells after L2′/L1′) adopted; "geometry-in-training creates no third artifact; changing the served net/continuation/fallback does" is the right third-slot rule. |
| 07 | Large-scale on-policy RL | **keep** | **E8 amendment written, critic still clockless (`model.py:101`)**; **E2 screen ≈ 8.5 Mac CPU-h, one thread (`PREFLIGHT.md:37-44`)** — the hidden Mac dependency nobody else saw — both verified. Right that Round 6's serial list wrongly queued E8 behind L2′. Colab calendar (367 u ceiling incl. 160-u E15 envelope) is a ceiling, labelled. "A bare nominee exists late Sunday at the earliest" is consistent with E16 → E17 → 32M; with E16's cap fix it may slip to Monday. |
| 08 | Distributed systems / parity | **partly** | Receipt list (18 engine fixtures; 96 frozen roots, *same actor across paths*; drained-barrier durability per Chandy–Lamport; 128/mix transfer = 768 episodes/platform, SD(d) = 100 ⇒ half-widths 8.4/14.5) adopted as the E13 execution spec. **`QUALIFICATION.md:65-68` excludes exact continuation; `run.py:380-399` target binds identity** verified. Wrong on the Mac price: 14 Mac-h per transfer audit assumes ≤ 65 s/episode single-threaded; T7's harness did ≈ 843 H10000 episodes/h on six workers ⇒ ≈ 1 h. Its 581-u planning ceiling is fine as a ceiling. |
| 09 | Sequential design | **keep** | Look ledger (E16 one look at .10; E17 forecast vs persistence; 32M futility only, zero α; 100M = E15's .05 family; final Mac family .05 composite) adopted as the X6 ledger. D5 (Sun 08:00) check = integrity and remaining-hours only, no interim look — adopted. UNCALIBRATED ≠ STOP; pause at the exact checkpoint if the balanced design cannot be funded/scheduled — adopted. DEADLINE_INFEASIBLE label adopted as the NOT_READY sub-label. Applies 74.1 h to R (wrong, §a1). |
| 10 | Calendar red team | **partly** | Right on structure: admit by measured finish time; day-3 table (adopted, §c6); "drop" = calendar exclusion, not KILL; hidden locks (clause silence permits nothing, `docs/STATUS.md:22` Mac occupancy, user play, thermals). **L2 preparation exists but retains the old label/16-u envelope; readiness enforces 384 min** verified. Wrong on the premise: R's strict gate is unpaced (T7 6.83 h), so "41 + 2×74.1 = 189 h" double-counts; the real serial Mac load is ≈ 80–95 paced h for S + ≈ 17 paced h + ≈ 10–20 unpaced CPU-h for R. |

Factual errors found: the shared 74.1-h-for-both-lines premise (all except 04), traceable to the question's own mislabelled 33 h; 08's
14 Mac-h E13 price. Corrections to the Round 7 fact sheet (judge's own): (i) "a fresh 384/128/128 paced bank ≈ 33 Mac-h per controller
pair" was wrong twice — 33 h is 96/mix paced, and bare gates are unpaced; (ii) L2′ is not "unbuilt" — the stateful clone exists in Sol's
worktree — but it is undispatchable and unpriced; (iii) E16 cannot dispatch as frozen; (iv) L0 printed INCOMPLETE (expected), L0′ unpublished.

## (c) Disagreements ruled

1. **Can the week hold both lines? (10, 06, 09, 02, 03: no; 01, 07, 08: one reservation; 04: conditional).** Reframed by §a1. The Mac is
   serial for anything that loads the CPU, so the ledger is: Line S ≈ 9 (L3) + 32 (L4) + 22–37 (candidate-arm strict, bare unpaced) +
   ≈ 11 (product sessions, 32 paired 10-min) + ≈ 6 (serving regressions) = **80–95 paced h**; Line R ≈ 7–10 (unpaced strict, T7 harness) +
   ≈ 1 (E13 Mac half) + 8.5 (E2 screen, CPU) + ≈ 11 + ≈ 6 (product/serving, paced) = **≈ 35 h, of which ≈ 17 paced**; quiet windows L1′ 3 h
   + E14 3 h. Total ≈ 120–135 h against Fri 00:00 → Tue 23:00 = 119 h plus Wed–Thu nights. Ruling: **one paced-qualification reservation at
   a time (S's, if admitted Fri 23:59); R's unpaced work runs in the gaps before it and R's paced product/serving block runs after S's
   block or in S's place.** No line is "dropped"; S loses only its paced reservation if not admitted. 10's finish-time admission rule
   (`slack = exclusive hours − remaining serial work`) is adopted as the day-3 and day-5 instrument.
2. **Day-3 timestamp — Fri 08:00 (01, 04, 05, 07, 08) vs Fri 23:59 (10).** 10 wins: L2′ needs a prospective envelope amendment, a batch-firing
   timing fixture, parity audit and ≤ 32 allocation-h; it cannot print before Friday evening. Decision at **Fri Oct 23 23:59 PT**; the
   L1′ cells and the clause answer must be in LIVE_FACTS by then or S is deferred (not killed).
3. **Is the 2.4 roots/s bar right? (02: no, 1.6/s).** 02 is right that Round 6's inputs give 1.6/s at a 12-s window. Ruling: L0′ (the
   bookkeeping version, still unpublished) must publish the per-doom reachable-window duration and root count from the 34 rescue-capable
   intervals; the admission bar is `roots_needed / window_seconds` from that table. Until it lands, **2.4/s stays as the conservative screen
   and 1.6/s is reported beside it**; L1′ is timing-only, so every cell's roots/s is recorded and can be re-read against the published bar
   without re-running. Under NO: W = 1, K = 1 ⇒ S ≤ 0.42 s (2.4) / 0.63 s (1.6).
4. **Third claim slot — Round 6's .025/.05 vs three-slot Holm (02, 04, 05, 06, 07, 08, 09).** The advisors win. Ruling: the final family is
   **frozen when the first strict bank is opened** and contains exactly the artifacts that have a frozen nominee at that moment (bare R;
   assisted-B8; planner-over-R); Holm at .05/m over those m slots; a slot without a nominee at freeze is **not** in the family (it cannot be
   added later); 05's composite-p text and the bare-wins-ties rule apply. This is stricter than Round 6 if three nominees exist and equal
   to it if two do.
5. **W/K clause arithmetic (01) vs Round 6's cells.** 01 wins: under K ≤ 8 the W = 4 cell's bar is **S_p99 ≤ 1.458 s**, W = 3 ⇒ ≤ 1.25 s
   mean / K = 7, W = 2 ⇒ 0.83 s. K is counted as queued + running + completed-but-unconsumed (02); lookahead distance is not K. The
   contract amendment text (Round 6 §f) is amended to say so.
6. **The day-3 table (10) vs "S keeps priority if ready" (01, 04, 07, 08).** Same rule, different wording; 10's table is adopted with one
   addition: **FUND_SYSTEMS + authorised L1′ PASS + S finish-time invoice fits ⇒ S holds the paced block; R runs unpaced work before it and
   paced product/serving after it** (not "R loses the week"). INCONCLUSIVE/INVALID/pending at 23:59 ⇒ S deferred, R takes the block.
7. **E13 cost (08: 26 u + 28 Mac-h) vs the T7 rate.** Colab half as 08 (≈ 2.6–3.4 u per audit); Mac half ≈ 1 unpaced h per audit, not 14.
   Two audits (initial + final SHA) ≈ 7 u + 2 Mac-h. Adopted.
8. **Serial dispatch (Round 6) vs separate builders (07, 06, 08).** Separate builders win: Sol-R lands E8 + the E16 cap amendment on day 1;
   Sol-S lands the L2′ envelope amendment + timing fixture; a third owner builds `bl_paced_identity` (03's spec); Codex audits. Round 6's
   single ordered list is replaced by three parallel lanes with one Mac serial ledger.
9. **A100/H100.** Unanimous no for the twelfth time; nothing this week is GPU-bound.

## (d) Running shortlist (carried from Round 6, updated)

| # | Lever | Conf. push / beats B8 at bar | Cost to first signal | Most informative next experiment |
|---|---|---|---|---|
| 1 | **Served exact-sim search on this Mac** (T3 certificate controller, reachable-root EDF burst, W/K-bound Claim B identity, three oracles) | 30 % / 18 % (down 5/2: L2′ undispatchable + unpriced, 0 batches timed, clause pending; partly offset by the stateful clone existing) | X0 + X1 ≤ 4 u → L2′ ≤ 64 u + 4 u audit; L1′ 3 Mac-h; L3 9 + L4 32 paced h | **X0/X1 (make L2′ dispatchable and priced) → L2′ → L1′** (bars per §c3/§c5) |
| 1c | RL endpoint as planner continuation (E20) | conditional | 8 u | after L2′ FUND_SYSTEMS + L1′ pass + a nominee |
| 2 | **Representation: geometry residual on the F parent** | 30 % / 18 % (unchanged) | E3 16 u → E14 bare cells (3 Mac-h) → E16-G (needs X3 cap fix) | E3 → E14 → E16-G |
| 3 | **Long-horizon on-policy RL from B8** (backbone) | 22 % / 9 % (down 3/1: E8 absent on day 1; E16 undispatchable as frozen) | E8 0 u → E4 2 u → E2 20 u (+ 8.5 Mac CPU-h or L4 screen) → E12 → E16 (X3) → E17 | **Land E8 + X3 today** |
| 4 | **GPU-resident exact engine** (Colab only) | 30 % (unchanged) | E1b 20 u, prepared | dispatch E1b |
| 5 | Search-in-the-loop | 15 % / 10 % | after E1b | E16-S |
| 6 | Restart archive / PLR | 6 % | E7 + E10 | E10 envelope |
| 7 | Opponent-conditioned add-on | 4 % | after E3b | — |
| 8 | World model; league | 2 % / 2 % | — | not funded |
| 1b | Retention sub-arm | conditional | 36 u | on the collateral trigger |

## (e) Still unresolved

- **The owner's W/K clause** — still the single fact that decides which L1′ cells count; silence = W 1 / K 1.
- **S per 6-action H300 batch on Watch6 opponents from a long body** — never measured; L2′ cannot be priced and L1′ cannot be read without it.
- **L0′ per-doom window table** (sets the roots/s bar: 1.6 vs 2.4); the Round-5 L0 is INCOMPLETE as predicted.
- **T3-dense's frozen/mixed gain** (L2′) — still the number that decides lever 1.
- **E16's cap** — as frozen it cannot dispatch; the fix is a pre-data amendment the funding judge must register.
- **Strict-bank sizing for Claim B** (P7: from L4's development variance) and whether the bare comparator may run unpaced in that study
  (needs the bare-unpaced logic registered for the new study, with its own boost clause).
- **Mac occupancy by the running campaign** (Phase 0 status, `docs/STATUS.md:22`) — quiet windows assume it; must be checked, not assumed.
- E8, E13 receipts, `bl_paced_identity`, final family text, burn manifest — all unwritten on main.

## (f) Experiments Claude can run now (≤ 20 u each unless stated; dispatch via Sol between rounds; report in LIVE_FACTS.md)

**X0 — Make L2′ dispatchable (0 u; Sol 0.25 day; today).** Prospective amendment to `PREREGISTRATION_L2.md` + `readiness.py`: label
`KILL_SERVED_SEARCH_PUSH` → `PACKAGE_KILL` (same rule), envelope → ≤ 32 L4 allocation-h / 64 u incl. 20 % reserve, usable ≤ 1,536 min,
timeout ≥ 2× estimate; everything else (banks 64/64/32, operator, one look, no bank shrinking) unchanged. Kill: none — bookkeeping;
without it L2′ is NOT_DISPATCHABLE by its own readiness check.

**X1 — L2′ batch-rate ruler on L4 (≤ 4 u; after X0; before the parity audit).** Three disjoint long-body fixtures per mix (L 800 / 1,200 /
1,500 at frames ≥ 2,500, Watch6 FRP3 + v8, S2 native v8 on) that **actually fire ≥ 30 dense batches each** (the preflight fixtures fired 0);
12 processes, one math thread each; report completed batches/worker-s per mix and the aggregate R. Forecast = 92,480/R + 1 h; dispatch
L2′ only if ≤ 80 % of 32 h. Kill: R < 0.8 batches/s aggregate ⇒ L2′ NOT_DISPATCHABLE this week; report the shortfall (this also prices
L1′'s per-root S on the same opponents, CPU-only).

**X2 — L1′ as Round 6 §f, with the §c3/§c5 bars (0 u; quiet Mac window 3 h, AC, lid open; after Phase 0 occupancy is confirmed idle).**
Cells per the clause answer (pending ⇒ run all, count W = 1/K = 1 only). Report per cell: S p50/p95/p99 by stage, roots completed in a
12-s and a 15-s burst, deadline-valid fraction at the 2.4/s and 1.6/s bars, K actually outstanding (queued + running + unconsumed), update
p95/p99, overruns, refill after forced invalidation; actions vs reference identical or INVALID. Kill: no authorised cell meets the published
bar with zero overruns and p95 ≤ 11.688 ms ⇒ served search unservable this week; releases L3/L4/strict/product reservations.

**X3 — E16 cap amendment (0 u; Sol 0.25 day; pre-data, needs the funding judge).** Raise E16's envelope to 20 L4 allocation-h / 40 u incl.
4 u reserve with the same 2×-timeout rule (so forecasts ≤ 8 h admit), one treatment at E4 < 2k/s, two otherwise. Also register that the E2
screen may run on L4 under E16's evaluation precedent if no Mac CPU night window exists (development evidence only). Kill: forecast > 8 h ⇒
NOT_DISPATCHABLE; no bank shrinking.

**X4 — E8 lands today (0 u; separate builder).** Unchanged content (Rounds 2–4); verified still missing at `model.py:101`.

**X5 — Strict-runner and E13 timing receipt for a bare nominee (0 u; ≤ 2 unpaced Mac CPU-h; any non-paced window).** Run the T7 harness
(≤ 6 single-thread workers) on 48 disjoint development worlds (16/mix) H5000 with B8 s3 and the F parent (or B8 twice if no parent exists);
report episodes/h and project 192/96/96 and 384/128/128 banks; same for 768 E13 transfer episodes. Kill: none — this replaces the 33/74-h
dispute with a measured number and sizes Line R's Mac slot.

**X6 — Final family + burn manifest + look ledger (0 u; Sol 0.5 day; before any strict bank).** 05's registration text with the §c4 freeze
rule, 09's look ledger, the append-only exclusion manifest, and each artifact's boost clause decided pre-data (recommend the relative floor
`0.5 × s_bare ≤ s ≤ .10` for both artifacts, with the funding judge's OK). Kill: unregistered look or family ⇒ the next claim is blocked.

**X7 — `research/bl_paced_identity/` (0 u; 24 Sol-h + 8 Codex-h over ~2 days; before L3).** 03's spec verbatim (events/record/replay/
semantic/receipt; independent reducer; hash-chained journal; fixture list). Kill: missing fields or failed oracle ⇒ L3 blocked, its 9 h released.

**X8 — L0′ bookkeeping (0 u; Sol 0.25 day; replaces the INCOMPLETE L0).** Own-deadline table, LAD-reachability curves, dense-receipt demand
census **and the per-doom reachable-window duration/root count** that fixes the roots/s bar (§c3).

**Contract amendment text for the owner (revised):** "Path B may use **W ≤ 4 worker processes** (one math thread each, QoS declared) and
**K ≤ 8 outstanding requests** (queued + running + completed-but-unconsumed); W, K, the root-targeting rule, horizon checkpoints and the
fallback rule are frozen before any efficacy test and are part of the candidate's identity; all other §4 clauses unchanged. Silence or 'no'
means W = 1, K = 1." The owner should be told that under 'no' the lever needs S ≤ 0.42–0.63 s on one thread, which has no measured basis.

**Dispatch order (three parallel lanes, one Mac ledger):** Lane S: X0 → X1 → L2′ parity audit → L2′ (print by Fri eve) ∥ X8 → X2 (first quiet
window) → X7 → [day-3 decision Fri 23:59] → L3 → L4 → Claim B strict (candidate paced, bare unpaced) → product/serving. Lane R: X4 → X3 →
D0 → E4 → E5 → E10 → E6+E7 → E3 → E2 (+ Mac night screen or L4) → E12 → E14 bare cells → E16 → E17 → 32M → E13 → unpaced strict → product/
serving. Lane C (0 u): X5, X6. E1b as a slot allows. Budget this inter-round: X1 4 + L2′ 64 + audit 4 + E1b 20 + E2 20 + E3 16 + E4 2 +
E5 1 + E6 4 + E7 20 + E12 16 + E16 40 (after X3) + E17 20 + E13 7 ≈ 238 u; everything else 0 u. A100/H100: no.
