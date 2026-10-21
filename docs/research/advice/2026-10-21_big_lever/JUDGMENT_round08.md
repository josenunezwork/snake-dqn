# Round 8 judgment — big-lever research rounds (plan round 1: the draft under stress)

Judge: Fable (round-8 judge), 2026-10-21 ~06:30 PT. Inputs: `../CHARTER.md`, `../round-01..07/JUDGMENT.md`, `../LIVE_FACTS.md`
(newest entry 00:40 PT; **no owner answer on the W/K clause — both branches kept**), `../QUEUE.md`, `QUESTION.md`, `astra_01..10.md`
(advisors read snapshot `57483b2f`). Repo verified read-only at main `10c305a0` (six commits after the snapshot). Sealed study not
opened. No jobs run. Light commands only (`git`, `sed`, `grep`, `tmux ls`, `colab usage`).

Repo and account state that matters (verified, all newer than what the advisors saw):
- **E8 has landed** (`bec9f6a7`, 01:04 PT): `research/bl_ppo/model.py:149` `self.critic = PopArtValueHead(257)` with a critic-only
  `remaining_horizon` input (`:194-223`), `refit.py`, `diagnostics.py`, `monitor.py`, `E8_AMENDMENT.md` (OWNER_AUTHORIZED_PRE_DATA).
  Advisors 02, 06, 08 wrote "E8 still absent" — true at their snapshot, stale now. X4 is done; E2 remains blocked only by the frozen E9
  admission (commit message, `E9_RULE.md`).
- **E9 is published with measured SDs**: T23 paired mass SDs 364.7 scripted / 262.7 frozen / 212.4 mixed ⇒ **SE 23.46 at 96/32/32,
  19.48 at 128/48/48** (`E9_RULE.md` §Mac simulation). Round 5's 11.8 (SD 200) and Round 4's 18 are both withdrawn; every power
  statement below uses the measured SDs.
- **Colab cap 3 is measured, not self-imposed** (`CAMPAIGN_LOG.md:100`: 4th assignment refused). **All three slots are occupied now**
  (tmux `bl-e3`, `bl-e1b`, `t34-o1b`; `colab usage` 06:20 PT: 1,142.44 u, 4.62 u/h, 3 active). L2′ has no slot until one frees.
- **Claim A D0 = registered KILL** (`CAMPAIGN_LOG.md:100`); Round 0 CLOSED (LIVE_FACTS 00:40). No Mac replay is owed; Phase 0
  (`phase0-selection`) is the only Mac occupant and was forecast to end Wed 10:00–13:00 PT.
- **X0 has not landed**: `research/bl_served_search/readiness.py:40` still `0 < 2*estimate <= timeout <= 384`; `manifests.py:66-95`
  `forecast()` still prints NOT_DISPATCHABLE at 16 u with conservative 529 min. **E16 executes only A/G** (`bl_e16/admission.py:13`
  `IMPLEMENTED = {"A","G"}`, `:131` arms must be 2–3 and all implemented, `:195-227` E3 **and** E14 receipts mandatory for every plan).
- Supervisor rules (`colab_cli/supervisor.py:192,201`): `timeout ≥ 2 × estimate`; `timeout × rate/60 ≤ unit_cap`. CLI qualification
  excludes exact continuation (`QUALIFICATION.md:65-68`).

## (a) Bottom line

1. **Ten of ten found the same break, and it is real: the draft's Stage 2→4 seam.** S admitted Fri 23:59 opens its first strict bank
   at **Sun Oct 25 17:00** (Sat 00:00 + L3 9 h + L4 32 h); Round 7 §c4 freezes the final family at that event; R's bare nominee is
   "Sun night–Mon at the earliest" and is therefore **excluded by the plan's own rule** (01, 04, 05, 06, 07, 08, 10). Independently,
   the serial Mac ledger is **95 h** from Sat 00:00 to Tue 23:00 against **80–95 h for S alone** and **105–123 h for both** (01, 04,
   05, 10; arithmetic checked). The draft's "each nominee" and Round 7's "both lines fit one Mac week" are **withdrawn**.
2. **Line R cannot produce a strict-eligible nominee this week at all**, so the collision resolves itself once stated honestly. The
   R chain after E8 is E4/E9 → E2 (two 8M arms ≈ 1.65 h each at 2k/s) → E6/E7 (needs E2 endpoints and critic receipts, `REGISTRATION_E6.md
   §decision`) → E12 → E16 (needs E3 GO + E14 PASS, both pending) → E17 (7.78 h) → 2 × 32M → 100M decision → balanced 4 × 100M
   (55.6 allocation-h; 27.8 wall-h on two slots) → endpoint screens → E13. Earliest 100M endpoint **Mon–Tue**; nominee Tue (06, 07, 08).
   A 4M or 32M checkpoint promoted to strict by calendar pressure is exactly what Round 5 §c1 forbids. **Ruling: R's day-7 deliverable
   is decisions + an executable, funded E15 replication registration + a durability receipt — NOT_READY for release, by design.**
3. **Therefore this week's release track is S alone, conditional, and the final family has m ≤ 1.** If S is admitted Fri 23:59 the
   family at its strict opening is {Claim B}; if S is deferred, **no strict bank opens this week** and next week's family can hold
   both. No bounded wait for R (§c2): waiting past Mon 08:00 pushes S's product/serving past Tue 23:00 at any strict size ≥ 264 worlds.
4. **L2′'s admission arithmetic was wrong in Round 7 and the plan's Thursday is too late.** With X0's 64 u / 20 % reserve, the usable
   allowance is 51.2 u = 1,536 timeout-min at 2 u/h; the supervisor's 2× rule makes the **estimate ≤ 12.8 h ⇒ R ≥ 2.18 batches/s**
   aggregate (02, 03, 08, 09; `92,480/R + 1 h`). Round 7's "R < 0.8 ⇒ kill" and "dispatch if forecast ≤ 80 % of 32 h" are withdrawn
   (0.8/s is 33 h, over even the hard cap). Latest dispatch for a Fri 23:59 print is **Thu 22:23** under a 25.6-h timeout (03, 09).
   The conservative `forecast()` (3× Mac slowdown on scripted batches) is 8.8 h ⇒ implied R 3.0/s — fits — but frozen/mixed batches
   with v8 opponents (67 % of worker time in the S0 mixed fixture) are unpriced; a 2× dearer batch makes L2′ NOT_DISPATCHABLE at 64 u.
   **Pre-data change (§f Y0): chunk L2′ into four 40-world immutable chunks (mixes interleaved, Round 0's pattern)** — each ≤ 8 h
   timeout, dispatchable on any free slot, parallel on two, no 25-h assignment, and 08's unverified 24-h Colab session limit becomes moot.
5. **L2′ is powered for a T3-sized effect, not for +40.** With measured SDs the L2′ pooled SE is **25.7** (not 16.7), frozen SE 32.9,
   mixed 26.5: FUND_SYSTEMS needs frozen mean ≥ 42.6, mixed ≥ 34.3 and pooled ≥ 40. At a true +40 in every mix that joint event is
   ≈ 20–30 %; at a true +80 (T3's +92 scale) ≈ 80 %. INCONCLUSIVE is the modal outcome at +40 and the plan must say so (09's
   "≈ 60 %" in Round 6 was SD-200 arithmetic). The bank stays 64/64/32 — no shrinking, no growing.
6. **X1 cannot price the Mac's per-root S** (03): L2's batch timer brackets clone → rollout only (`ceiling.py:181-190`), on a different
   host, no MPS, aggregate over 12 workers. Round 7's "X1 also prices L1′'s S" is withdrawn. One shared fixture/operator harness with
   **separate platform receipts** (03's spec) serves both; L1′ needs its own Mac window — and since Phase 0 ends Wed and Round 0 owes no
   replay, **L1′ can run Wed evening**, before L2′, giving the owner's clause a measured basis by Thursday.
7. **Stage 0 is over-committed as drafted** (02): X7 was priced as two builders (12 h recorder + 12 h independent reducer + 8 h Codex),
   the L3 controller (EDF, W/K, checkpoints, invalidation) has **no owner and no allowance** (16–24 Sol-h), and three lanes × two days
   = 48 h against 70–90 h of work. Ruling: Stage 0's milestone is **dispatch readiness** (X0-chunked, X1, X3, X6-state-machine, X9),
   not "everything lands"; X7/L3 get a named production owner and a separate replay owner and are due before L3, not Thu 20:00.
8. **Two trust gaps move into Stage 0.** (i) Continuation: the 32M pause → 100M is an experiment fork until exact resume is proved on
   the real trainer (08; `QUALIFICATION.md:65-68`, `run.py:380-399`) — a ≤ 8-u durability acceptance before any 32M dispatch (Y3).
   (ii) The owner packet, Claim B pin schema, visible label and rollback matrix are defined **before L3**, populated later (10; the
   registry binds only checkpoint + veto, `served_checkpoint.py:133-149,172-217`). E17's 2× timeout liability is 31 u > its 20-u cap
   (08) — same defect as E16's, same fix (Y4). E15's 160-u envelope cannot fund balanced 4 × 100M plus two looks (111 u training at 2k/s;
   222 u at 1k/s — 06, 07, 08) — repriced in Y8.
9. **Round 7 corrected on five points:** both-lines-fit (§a1); L2′ R bar and dispatch time (§a4); X1 → Mac S transfer (§a6); the
   X0–X8 list omitted X7's second builder, the L3 owner, durability, the packet schema and a treatment-availability receipt (07: with
   no G there is no executable E16, and "PPO alone" is refused by `admission.py:131`); SE 11.8 (now 23.5 measured).
10. **Shortlist:** served search **28 % / 16 %** (L2′ unchunked/undispatchable, slots full, powered only for T3-scale); representation
    **28 / 16** (E3 running; E14 needs a Mac window; E16 is A/G-only); PPO backbone **22 / 9** (E8 landed; no nominee this week by
    construction); engine 30 % (E1b running); search-in-the-loop 15 / 10; archive 6; opponent 4; world model / league 2.

## (b) Per-advisor verdicts

| ID | Persona | Verdict | Why (verified facts in **bold**) |
|---|---|---|---|
| 01 | Calendar red team | **keep** | Withdraws its 74-h error cleanly, then finds the real one: **S alone 80–95 h vs 95 h available; both 105–123 h** (checked). Branch table (strict Sun 17:00 → Mon 15:00 / Tue 06:00; product; serving; "zero slack at 37 h") adopted as §f Y10's ledger. Right that R's late work cannot be pulled into Wed–Fri. **`readiness.py:40` timeout rule ⇒ R ≥ 2.177** (checked). Its "latest S start Fri 16:00 at 95 h" is the correct consequence: the 37-h strict case is inadmissible this week. |
| 02 | Builder-capacity red team | **keep** | The only advisor who priced the work: X7 = two builders (**`round-07/astra_03.md:33`**), L3 controller unowned, 70–90 Sol-h vs 48 available. Collision map verified: **E16 imports `PPOPolicy` (`bl_e16/model.py:18-19`)**, **X3 touches `admission.py:23-56` + `budget.py:9-40`**, **X0/X1 share `manifests.py:67-95` + `readiness.py:31-41`**. Dispatch-readiness milestone and "freeze events first, build against a fake clock" adopted. Factual: "E8 still a spec" was true at its snapshot; landed since. |
| 03 | Apple-silicon systems | **keep** | **L2 timer brackets clone → rollout only (`ceiling.py:181-190`)**; **Phase 0 MPS numbers were N8/H128 (`PROFILE.md`)** verified. Correctly kills Round 7's X1 → Mac S transfer. The W=1/K=1 bar is submit-to-consumption C, not S: **S ≤ 333/542 ms** with the two-tick margin, and the 15-tick alignment example (S = 400 ms ⇒ 1.6 roots/s) is right. Two separate 3-h windows; 26 % chance of seeing a true p99 in 30 samples (checked). Shared harness / separate receipts adopted as Y1/Y2. |
| 04 | Release / serving qualification | **keep** | **T7 6.828 h is wall on ≤ 6 workers, not a receipt for a new actor** (`t7_phase_r_result:11`); **product 10.67 h = 32 × 2 × 10 min before overhead**; **regression "6 h" has no runtime receipt (T12/T15 enumerate checks)** verified. Sizing: ≈ 88 worlds/mix at SD 200 for 90 % power at +40 under Holm .025 — with measured SDs this is larger; L4 variance sizes it. Dated checklist and `SKILL_PASS / PRODUCT_FAIL / RELEASE_HOLD` disposition adopted (Y10). Right that family freeze precedes R's arrival. |
| 05 | Evaluation / trust skeptic | **keep** | Family state machine (nominee frozen / bank unopened; S deferred then readmitted; "opens" = first strict evaluative episode incl. comparator-only) adopted verbatim into X6 (Y6). **`REGISTRATION.md:86-94` still `100M_ADMISSIBLE`** verified — needs a prospective authority map, not an operator reading. Point-filter stacking (three +40 point filters ≈ 12.5 % at true +40, illustrative) is why I weaken 09's L4 clause (§c5). Laundering vector named correctly: a repaired scheduler inheriting a STRICT_PASS. |
| 06 | Large-scale on-policy RL | **keep** | Best R calendar: **E16 5.44 h / E17 7.78 h / balanced 4 × 100M 55.6 allocation-h = 27.8 wall-h on two slots**; 160 u leaves 8.44 h for evaluation after 111 u of training (checked). "Promise an executable replication registration, not a nominee" is the ruling (§a2). **`run.py:380-399` binds target** verified. Its 192-u illustrative envelope is adopted as the E15 ceiling to be ratified (Y8). Stale on E8 (landed). |
| 07 | Representation | **keep** | The round's sharpest code fact: **`IMPLEMENTED = {"A","G"}`, arms 2–3 all implemented, E3 + E14 receipts mandatory (`admission.py:13,131,195-227`)** — so "drop G, run PPO alone" is refused by the harness. **E3 forecast 2.97 h / 5.9 u (`PREFLIGHT.md:40-51`)** verified; E3 is dispatched now. "No admitted G ⇒ no E16/E17/E15 dispatch under this plan; a replacement treatment needs its own registration" adopted (§c7). Right that representation GO may select no deployable architecture (`PROTOCOL.md:72-76`). |
| 08 | Distributed systems / parity | **keep** | **`supervisor.py:192` 2× rule; `:201` cap rule; `QUALIFICATION.md:65-68` no exact continuation; `bl_ppo/PREFLIGHT.md` resume restored a completed endpoint only** verified. E17 timeout liability **31.1 u > 20 u** (checked) ⇒ Y4. Durability acceptance as a Stage-0 entry requirement (Y3) and slot-reservation table adopted. Its Colab 24-h session claim is external and **unverified here**; chunking (Y0) removes the dependency either way. E13 timeout liability ⇒ 18 u envelope (Y8). |
| 09 | Expert player / Line S adversary | **keep** | **T3 s3 563.85 → 686.66, 17/32 → 26/32 on reused census worlds (`t3_ceiling_test:119-125`)**; **T1 46/49 within 300, 25/49 within 100, 13/13/16/7 sealers** verified. Right that L3/L4's pooled rules cannot establish cross-mix transfer under pacing and that a scripted-only paced artifact can legitimately reach strict. Its L4 clause is adopted in weakened form as a *budget* status (§c5). Ledger fields (productive rescue, D300-only, re-death, newly caused deaths, sealer strata, per-mix denominators) adopted into L2′/L3/L4 exports (Y7). 97.71 % retention arithmetic re-checked. |
| 10 | Owner-trust red team | **keep** | **Registry binds checkpoint + veto only (`served_checkpoint.py:133-149,172-217`)**; **monitoring window = 7 days AND 100k decisions, monitor reports only (`b8_s3_release_erratum:49-98`)**; **contract: one extra thread, ≤ 1 outstanding (`SUCCESS_CONTRACT.md:50-54`)** verified. X9 (packet schema, "Assisted hero — [controller ID]" label with per-tick fallback disclosure, authority receipt, rollback matrix incl. drain/invalidate, monitoring contract) adopted before L3 (Y5). Right that silence on W/K is "not authorised", not KILL. |

Factual errors found: none that change a conclusion. Three advisor statements are stale rather than wrong (E8 absent — 02, 06, 08;
"cap 3 self-imposed" — none repeated it). Corrections to the Round 8 fact sheet (judge's own): (i) "both lines fit" withdrawn; (ii)
L2′ R bar is 2.18/s and dispatch-by is Thu 22:23, not "Thu night"; (iii) X1 does not price Mac S; (iv) SE is 23.5 measured; (v) E8 landed.

## (c) Disagreements ruled

1. **Two release tracks (draft) vs one (01, 04, 05, 06, 07, 08, 10 — unanimous among those who computed it).** One. The week has at
   most one paced-qualification block and R cannot nominate before Tue. Ruling: **S is the conditional release track; R is the research
   track whose day-7 product is decisions + E15 registration + durability + E13-ready export.** Nothing is "dropped"; R keeps every
   scientific label; next week's family may hold both.
2. **Bounded wait for R before S opens strict (05 "lawful but invoiced", 07 "register either") vs freeze at S's opening (01, 04, 06, 10).**
   No wait. R's nominee does not exist by Mon; a wait to Mon 08:00 ends S's chain at Tue 23:00 with zero slack at the *smallest*
   admissible strict bank. If S is deferred Fri 23:59, no strict bank opens this week (§a3) — that is the only way both lines reach a
   future family intact.
3. **L2′ admission bar — Round 7 (0.8/s), 01/03 (1.04/s if the timeout rule were waived), 02/03/08/09 (2.18/s).** 2.18/s under the
   frozen 2× rule and the study's own usable allowance (reserve excluded, as `readiness.py` and `PREREGISTRATION_L2.md:53-58` already
   do at 16 u). The supervisor's cap check uses `unit_cap` (`supervisor.py:201`), so the study rule is the binding one. Ruling adds
   **chunking** (Y0): four 40-world chunks, each its own manifest with timeout ≤ 480 min (estimate ≤ 4 h ⇒ same 2.18/s aggregate bar at
   12 workers, or 1.09/s if two chunks run on two slots), deterministic merge, one look after all four. A chunk that times out is
   INCONCLUSIVE for the whole study (no partial-bank verdict), as Round 0 ruled for its chunks.
4. **Does X1 price Mac S? (Round 7 yes; 03 no).** 03. Y1 is one harness, two receipts; L1′ (Y2) is the only Mac latency evidence.
5. **09's strict-budget clause (L4 must show frozen LB90 > 0 and mixed LB90 > 0) vs 05's point-filter stacking.** Both partly. At 32
   worlds/mix with measured SDs, per-mix LB90 > 0 at L4 needs frozen mean ≥ 60 and mixed ≥ 49 — that converts L4 into a ~25 % filter
   at a true +40. Ruling: L4 **reports** per-mix LB90 and the full ledger; strict admission additionally requires **frozen point > 0
   and mixed point > 0** (paced) with no per-mix UB90 < 0; failure prints **TRANSFER_UNRESOLVED** — a budget status that releases the
   strict/product reservation and preserves L4's scientific label. Prospective; registered in L4's text before L3 data.
6. **E17 cap (08: amend to 40 u / 20 h) vs leave at 20 u.** Amend, pre-data, same form as X3 (Y4). The alternative — splitting E17
   into per-arm jobs — does not reduce aggregate timeout liability (08) and adds exact-resume dependencies E17 cannot yet meet.
7. **No G ⇒ what? (07: explicit stop branch; draft: implicit "PPO alone").** 07, by code. If E3 prints no retained architecture or E14
   fails, E16/E17/E15 are NOT_READY this week; E2/E6/E7/E12 decisions stand on their own; any A-only or other-treatment study is a new
   registration with its own invoice. E3 is running now; E14's Mac window (3 h) is booked Thu (Y2).
8. **Stage-0 scope (02) vs the draft's X0–X8 "all by Thu 20:00".** 02. Milestone = dispatch readiness; owners: Sol-S (X0-chunked, X1,
   X8), Sol-R (X3, E17 cap, E9/E2 admission, durability Y3), Sol-C (X7 recorder + L3 controller production seams), Codex (independent
   replay reducer, X6/X9 audits). D0 is closed (KILL); E1b and E3 are already dispatched; O1 is not on the path.
9. **Day-3 timestamp.** Fri 23:59 stays, but the instrument changes: the admission record (Y10) carries L2′ chunk prints, authorised L1′
   cells, X7 oracle status, X9 readiness, **finish-time invoice with a 2-h packet allowance and 6-h recovery reserve** (01), and the
   37-h strict case is refused (S's strict bank is sized from L4 variance; at 95 h the latest S start is Fri 16:00, so only the
   ≤ 80-h invoice is admissible from Sat 00:00).
10. **A100/H100.** Unanimous no for the thirteenth time. Nothing on either line is GPU-bound; the Mac and the builders bind.

## (d) Running shortlist (carried from Round 7, updated)

| # | Lever | Conf. push / beats B8 at bar | Cost to first signal | Most informative next experiment |
|---|---|---|---|---|
| 1 | **Served exact-sim search on this Mac** (T3 certificate controller, reachable-root EDF burst, W/K-bound Claim B identity, three oracles) — the week's only possible release-track artifact | 28 % / 16 % (down 2/2: L2′ needs chunking + 2.18/s; slots full; powered for +80 not +40; clause pending) | Y0 + Y1 ≤ 8 u → L2′ ≤ 64 u (4 chunks) → L1′ 3 Mac-h → L3 9 + L4 32 paced h | **Y1/Y2 (shared harness; L1′ Wed–Thu) ∥ Y0 → L2′ chunks** (print by Fri eve) |
| 1c | RL endpoint as planner continuation (E20) | conditional | 8 u | after L2′ FUND_SYSTEMS + L1′ pass + a nominee (next week) |
| 2 | **Representation: geometry residual on the F parent** | 28 % / 16 % (down 2/2: E16 is A/G-only and needs E3 GO + E14 PASS; no nominee this week) | E3 (running, 16 u) → E14 3 Mac-h (Thu) → E16 40 u (X3) | E3 print → E14 → E16-G |
| 3 | **Long-horizon on-policy RL from B8** (backbone) | 22 % / 9 % (unchanged: E8 landed, but the chain cannot nominate before Tue) | E9 → E4 → E2 20 u → E6/E7 24 u → E12 16 u → E16 → E17 40 u (Y4) → 32M (after Y3) | **E2 dispatch when a slot frees**; Y3 durability before any 32M |
| 4 | **GPU-resident exact engine** (Colab only) | 30 % (E1b running) | — | read E1b |
| 5 | Search-in-the-loop | 15 % / 10 % | after E1b | E16-S (next week) |
| 6 | Restart archive / PLR | 6 % | E7 + E10 | E10 envelope |
| 7 | Opponent-conditioned add-on | 4 % | after E3b | — |
| 8 | World model; league | 2 % / 2 % | — | not funded |
| 1b | Retention sub-arm | conditional | 36 u | on the collateral trigger |

## (e) Still unresolved

- **The owner's W/K clause** — still pending; both branches kept. L1′ (Y2) runs all cells Wed–Thu and records roots/s per cell, so
  the answer can be read against measured numbers without re-running; silence at Fri 23:59 = W 1 / K 1 (bar S ≤ 333/542 ms per 03).
- **S per batch on Watch6 opponents, on this Mac** (L1′) and **R on L4** (X1) — the two numbers that decide whether L2′ dispatches
  and whether any W cell can pass; both unmeasured; Phase 0 ends Wed so the Mac window exists.
- **L0′ per-doom window table** (sets 1.6 vs 2.4 roots/s) — still unpublished.
- **E3's print** (running) and **E14 on the E3 architecture** — decide whether E16 has any executable treatment this week.
- **Whether a 25-h Colab assignment is even possible** (08, unverified) — chunking makes it irrelevant for L2′; E17's 20-h cap and the
  balanced 100M waves still assume ≥ 14-h sessions and must be chunked or paused at exact checkpoints (Y3).
- **Strict-bank sizing for Claim B** from L4 variance (P7 §1) with measured SDs; at SD ≈ 263 frozen the bank is larger than T7's 384.
- **Mac occupancy**: Phase 0 ETA Wed 10:00–13:00 (CAMPAIGN_LOG 19:25 entry) — must be confirmed idle, not assumed, before Y2.
- X0 (chunked), X3, X6 state machine, X7, X9, E17 cap, E15 repricing — all unlanded.

## (f) Experiments Claude can run now (≤ 20 u each unless stated; dispatch via Sol between rounds; report in LIVE_FACTS.md)

**Y0 — X0 corrected and chunked (0 u; Sol-S 0.5 day; today).** Prospective amendment to `PREREGISTRATION_L2.md`, `manifests.py`,
`readiness.py`: label → PACKAGE_KILL; envelope → ≤ 64 u incl. 20 % reserve; **four immutable chunks of 40 worlds** (16 frozen / 16 mixed
/ 8 scripted each, deterministic seed order, mixes interleaved), each its own manifest with `timeout ≤ 480 min` and `timeout ≥ 2 ×
estimate`, 12 processes one math thread each, deterministic ordered merge; one look after all four chunks are terminal; a chunk timeout
or fault ⇒ study INCONCLUSIVE (no partial-bank verdict, no retry without re-registration). Admission: per-chunk conservative estimate
≤ 240 min ⇒ aggregate **R ≥ 2.18 batches/s** on X1's fixtures; else NOT_DISPATCHABLE (report the shortfall). Kill: none — bookkeeping.

**Y1 — X1 as a shared harness with separate platform receipts (≤ 4 u; Sol-S 1 day + Codex 0.5 day; after Y0, before parity).** 03's
spec: frozen root bytes (bodies, food, RNG, opponent carry, roster/checkpoint hashes, legal actions, H300 clip, tie rule, S2 on) for
disjoint long-body fixtures at L 800 / 1,200 / 1,500 in **all three mixes**, each firing **≥ 30 batches per fixture**; exclusive stage
timers (sim, observation, hero v8, opponent v8, Vector61/FRP3, B8 forward, clone/digest) plus submit→visible and submit→consumed; the
same operator and expected outputs on L4 (12 workers, aggregate R) and on the Mac (L1′, Y2). Semantic outputs must match across
platforms or the fixture is INVALID; timing is platform-labelled and never transferred. Kill: R < 2.18/s aggregate on L4 ⇒ L2′
NOT_DISPATCHABLE at 64 u this week (the funding judge may reprice the envelope prospectively; the bank never shrinks).

**Y2 — Two exclusive Mac windows (0 u; Wed evening + Thu; AC, lid open; after Phase 0 is confirmed idle).** Window A (3 h, Wed): **L1′**
as Round 6 §f / Round 7 X2 with 03's allocation (20 min controls, 60 paired CPU/MPS root measurements, 60 × 12/15-s bursts incl. forced
invalidation, 40 min receipts); all W ∈ {1..4} cells run, each cell's roots/s and K recorded; **authorised cells per the clause answer;
silence ⇒ W 1 / K 1 with bar S_submit→consumed ≤ 333 ms (2.4/s) / 542 ms (1.6/s)**. Window B (3 h, Thu): **E14 bare cells** on the E3
architecture (B8 / F / F+G × three lengths × two 5-min sessions, encoder executed at W = 0, plus feature-cost and cold-state checks);
planner cells deferred. Kill: no authorised L1′ cell meets the published bar with zero overruns and p95 ≤ 11.688 ms ⇒ served search
unservable this week (releases L3/L4/strict/product); E14 p95 > 11.688 ms or any overrun ⇒ G unservable, E16-G not admitted.

**Y3 — Exact-continuation durability acceptance for the PPO trainer (≤ 8 u incl. 1.6 reserve; Sol-R 1 day + Codex 0.5 day; before
any 32M dispatch and before E17).** 08's spec: fixed `target_transitions = 100M` with a separate `pause_at = 32M`, immutable checkpoint,
explicit restart; uninterrupted vs exported/re-uploaded continuation on matched L4 software through ≥ 3 optimizer updates and ≥ 1 episode
boundary; compare optimizer/PopArt state, RNG, counters, worlds, opponents, pending rollouts and the next 1,000 actions; bound the
real trainer and architecture (`PARITY_RULE.md` §3). Kill: any unexplained state mismatch ⇒ no 32M pause is registered; E17 and E15
must then be single-session jobs or NOT_READY.

**Y4 — E17 cap amendment (0 u; Sol-R 0.25 day; pre-data; funding judge).** Envelope → 40 u incl. 8 reserve / 20 allocation-h, same
2× rule (forecast ≤ 8 h admits; the corrected invoice is 7.78 h). Require the measured disjoint full-policy rate ≥ 900 world-steps/s
and an E16 export receipt before admission. Kill: forecast > 8 h ⇒ NOT_DISPATCHABLE; no bank shrinking.

**Y5 — X9: owner packet, Claim B pin schema, label and rollback matrix before L3 (0 u; Sol-C 8 h + Codex 4 h; Wed–Thu).** 10's
spec: packet front page + evidence index (each clause: threshold, observed value/interval, PASS/FAIL/PENDING, artifact digest, receipt
hash, audit command); pin schema binding planner + scheduler + W/K + deadlines + fallback + load envelope + UI build (beyond
`served_checkpoint.py:172-217`); first-screen label "Assisted hero — [controller ID]" with per-tick fallback disclosure; rollback matrix
(assisted Watch6; bare candidate; B8 Watch8/12/Play; missing/corrupt pin; stale/wrong-life; restart; explicit rollback that disables
assistance, drains/invalidates pending responses and shows the resulting identity); monitoring contract with candidate-specific
baselines, windows, named operator; statuses qualified / owner-authorised / deployed / monitoring-incomplete kept separate. Kill: missing
schema at L3 ⇒ L3 blocked (its 9 h released).

**Y6 — X6 as an executable family state machine + whole-pipeline OC on measured SDs (0 u; Sol 1 day + Codex 0.5 day; before any
strict bank).** 05's rules: "opens" = first strict evaluative episode launched (comparator-only counts); members = artifacts with a frozen
nominee at that event; late nominees excluded; frozen-but-unavailable members keep p = 1; S deferred then readmitted only with the same
identity and a new finish-time invoice; composite p = max of clause p-values; Holm .05/m; bare wins ties; no reset after failure.
Prospective authority map for `REGISTRATION.md:86-94` (`100M_ADMISSIBLE` = 32M-tranche admission). OC simulation (100k replicates)
of **L2′ (4 chunks) → L4 (+ §c5 clause) → strict → product** and **E16 → E17 → 32M → balanced 100M** using E9's SDs (365/263/212),
session clustering, shared controls, calendar censoring; report conditional admissions, unconditional qualification, false
qualification, missed winners, units and finish dates separately. Kill: family error > .05 beyond MC error ⇒ registration blocked.

**Y7 — Ledger fields and TRANSFER_UNRESOLVED on L2′/L3/L4 exports (0 u; Sol-S 0.5–1 day + Codex 0.25 day; before L2′ dispatch).**
09's world-level fields: productive rescue (B8 dies, candidate reaches H5000, tail length ≥ 90 % of B8's last living length), D300-only
escape + re-death, delay-only changes, newly caused deaths, common-alive growth loss, natural-world denominators, B8 death-frame × length,
sealer strata (hero / opponent / none / unresolved) where reconstruction supports it, proposed / completed / consumed-before-deadline
interventions by mix; Codex reconstructs mass from trajectories. L4 strict-admission clause per §c5 (frozen point > 0 ∧ mixed point > 0,
no per-mix UB90 < 0; else TRANSFER_UNRESOLVED, budget status). Kill: unreconciled totals ⇒ admission blocked.

**Y8 — E15 and E13 repricing (0 u; Sol-R 0.5 day; funding judge; before any 100M).** E15 envelope → **192 u incl. 20 % reserve at
2k/s** (balanced 4 × 100M = 111 u + two looks on the registered interim bank + exports), chunked per Y3's pause points; at a measured
rate < 1.5k/s ⇒ pause/defer, no bank shrinkage; credit prior funded tranches explicitly. E13 envelope → **18 u incl. 3.6 reserve** for two
audits (2× timeout liability), 128/mix = 768 episodes/platform fixed (`ERRATUM_01.md:11-13` still names 384/128/128 — correct it).

**Y9 — Stage-0 dispatch-readiness milestone with owners (0 u; Codex 2 h; today).** Thu 12:00 code cutoff for Y0, Y1, X3, Y4, Y6, Y7,
Y9; Thu 12:00–20:00 independent review + immutable manifests. Sol-C owns X7 recorder + L3 controller production seams (events frozen
Wed; fake clock/transport); Codex owns the independent replay reducer; neither is due Thu 20:00 — both due before L3 (Sat 00:00 at the
earliest). Kill: missing review, stale source hash or contradictory admission bounds ⇒ NOT_READY for that item; no dependent dispatch.

**Y10 — Fri 23:59 admission record and serial Mac ledger (0 u; Codex 2 h; template today, filled Fri).** 01's finish-time invoice:
`slack = exclusive Mac hours − serial remaining execution − 2 h packet − 6 h recovery`; S admissible only with L2′ FUND_SYSTEMS ∧
authorised L1′ PASS ∧ X7 oracles ∧ Y5 schema ∧ slack ≥ 0 at the **L4-variance-sized** strict bank (the 37-h case is refused). Sun 08:00 =
integrity + hours only. Day 7 = one of qualified / scientific KILL / INCONCLUSIVE / NOT_READY per line, with R's registration packet
(E15 + Y3 receipt + E13-ready export) as its stated deliverable.

**Unchanged:** E2 (dispatch when a slot frees, E9 admission as frozen), E4, E5 (done: cap 3), E6/E7 (+E10), E12, E1b (running), E3
(running), E16 (X3), E20, E19, contract amendment text (Round 7 §f; silence ⇒ W 1 / K 1).

**Dispatch order (three builder lanes, one Mac ledger, three Colab slots currently full):** Colab — as slots free: L2′ chunks 1–4 (after
Y0 + Y1 + parity; two chunks in parallel when two slots are free) ∥ E4 → E2 → E6+E7 → E12 → E16 → E17 (after Y3/Y4) → 32M. Mac — Phase 0
ends → Y2-A (L1′, Wed eve) → X5 (2 h) → Y2-B (E14, Thu) → [Fri 23:59] → L3 → L4 → Claim B strict (L4-sized) → product → serving → packet.
Builders — Sol-S: Y0, Y1, Y7, X8; Sol-R: X3, Y3, Y4, Y8, E2 admission; Sol-C: X7 + L3 controller, Y5; Codex: Y6, Y9, Y10, replay reducer,
all audits. Budget this inter-round: Y1 4 + L2′ 64 + parity 4 + Y3 8 + E2 20 + E6/E7 24 + E12 16 + E16 40 + E17 40 (Y4) ≈ 220 u plus
E3/E1b already running; A100/H100: no.
