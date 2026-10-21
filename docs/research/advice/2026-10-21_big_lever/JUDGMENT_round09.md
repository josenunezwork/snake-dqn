# Round 9 judgment — big-lever research rounds (plan round 2: the near-final plan under adversarial review)

Judge: Fable (round-9 judge), 2026-10-21 **~03:00 PT** (wall clock `date` = 02:50 PDT; the "~06:30 PT" self-stamps in Rounds 7–8
were wrong — file mtimes and LIVE_FACTS entries put Round 8 at ~01:13 PT and this round's answers at ~02:45 PT; the calendar below
uses the real clock: **all of Wednesday is still ahead**). Inputs: `../CHARTER.md`, `../round-01..08/JUDGMENT.md`, `../LIVE_FACTS.md`
(newest entries 01:10 / 02:45 / 02:55 PT), `../QUEUE.md`, `QUESTION.md`, `astra_01..10.md` (advisors read snapshot `6e05c52f`).
Repo verified read-only at main `88adb3e2` (two commits after the snapshot: `bc5cfffd` = L2′ AMENDMENT_08 prose; `88adb3e2` = E1b
CUDA dtype fix). Sealed study not opened. No jobs, Colab or tmux.

Facts newer than the advisors' snapshot, folded in as authoritative (LIVE_FACTS 01:10–02:55 PT + repo):
- **O1 = GO** (35/48 fresh worlds have a surviving single-action H500 branch ≥ +5 from corpse intake; frozen 9 / mixed 13 / scripted
  13; `CAMPAIGN_LOG.md:101`). Offensive harvesting via exact-sim search is real against these opponents. **O2** (bare vs rescue-only
  D vs D+O, 96 fresh paired worlds, ≤ 24 u) is the registered next step, Sol building (`research/t34_o1_harvest/PREREGISTRATION_O1.md:85-93`:
  O1 GO authorizes only preparation of a separately registered O2). It is an *unpaced offline* value test; serving it is Line S.
- **E9 = BLOCKED** (`research/bl_ppo/E9_RULE.md:43-61`: joint funding at a true +40 is .270 / .307 at 96/32/32 / 128/48/48 vs the
  required .60; null-pass .051/.062 OK; `selection.dispatch_admitted = false`; "the frozen options stop at 128/48/48; no further N,
  threshold, arm or guard change is authorized by this result"). **E2 cannot be dispatched by any slot freeing.**
- **E8 landed** (`research/bl_ppo/model.py:149` PopArt critic with remaining-horizon input; 101 tests).
- **E1b FAILED mechanically** on L4 (CUDA lacks bool `scatter_reduce` in `v8_device._paint`; CPU preflight missed it); the fix is
  `88adb3e2` ("enforce dtype contracts"); relaunch pending. **E3 FAILED mechanically** (57.8 MB upload hit the supervisor's upload-time
  cap; Sol moving reconstruction onto the runtime); relaunch pending. **E4 dispatched 02:46.** O1 census-02 SUCCEEDED.
- **Claim A D0 = registered KILL.** Colab cap **3 measured** (4th refused; `supervisor.py:818`).
- **Y0 landed as prose only**: `research/bl_served_search/AMENDMENT_08.md` (OWNER_AUTHORIZED_PRE_DATA; four 40-world chunks, 16 u each
  incl. 3.2 reserve, ≤ 384 timeout-min ⇒ estimate ≤ 192 min per chunk, aggregate R ≥ 2.18/s, one look, chunk fault ⇒ INCONCLUSIVE,
  Y1 nine-fixture ruler, Y7 ledger + TRANSFER_UNRESOLVED clause, X8 L0′). **But the code is unchanged**: `readiness.py:41` still
  `0 < 2*estimate <= timeout <= 384` against a *single* 160-world plan, and `manifests.py:66-95` `forecast()` still prints
  `NOT_DISPATCHABLE … unit_cap 16`. L2′ is dispatchable by neither the old nor the new rule until the chunk-aware code lands.
- **Owner's W ≤ 4 / K ≤ 8 answer: still pending.** Both branches kept; silence ⇒ W 1 / K 1.

## (a) Bottom line

1. **The round did its job: ten of ten refused a clause, and the refusals cluster on four seams, each verified against code.**
   (i) **Stage 2 demands Sunday's evidence on Friday** (01, 02, 03, 07, 09 independently): "X7 three oracles PASS" is the *L3*
   inventory — `receipt.py:160-163` requires exactly 108 fresh Darwin arm-runs, 16 paired worlds/mix and 6 paired sessions — and
   "slack ≥ 0 at the L4-sized strict bank" needs L4's paced variance, which exists Sun 17:00 at the earliest. X7's own receipt already
   exposes the right Friday instrument: `l3_identity_prerequisites = PASS` is computed from `base_ready` independently of the paced
   oracle (`receipt.py:346-354, 412-417`); its build receipt is explicitly PRECHECK_ONLY, paced NOT_RUN (`PREFLIGHT.md:36-40`).
   **Ruling: Stage 2 becomes three dated admissions** — Fri 23:59 *provisional reservation*, post-L3 *identity gate*, post-L4
   *pre-strict gate* (§c1, text in §f Z-A). (ii) **The serial ledger has ≤ 29 h for strict, not "an L4-sized bank"** (01, 02, 07, arithmetic
   checked: 95 − 9 − 32 − 10.67 − 6 − 2 − 6 = 29.3 h); a 384-world candidate-paced strict (22.2 h) leaves 7.1 h for bare controls, replay
   and audits; 640 worlds (37 h) is −8 h; X7's reserved **8 unpaced replay + 2 semantic hours** (`README.md:138-141`, `PREFLIGHT.md:94-98`)
   fall inside the 29 h unless consumed before Friday. **Ruling: the Friday invoice carries a strict-time *ceiling*; the post-L4 gate sizes
   from variance and defers (NOT_READY) if the sized bank exceeds the ceiling — the bank is never shrunk to fit.** (iii) **Stage 3′/4 promise
   deliverables that the record already forbids** (06, 08, 09): E9 is BLOCKED so "E2 when a slot frees" is not executable; E3 may print GO
   with `selected_architecture = null` (`PROTOCOL.md:72-76`); E16 executes A/G only and refuses A alone (`admission.py:13, 165-169`);
   192 u cannot fund balanced 4 × 100M under the supervisor's 2× liability (222 u at 2k/s; `supervisor.py:192, 201`). **Ruling: R's day-7
   deliverable is a receipt-backed disposition with explicit blockers; "funded E15" is withdrawn; E15's envelope is re-ratified at 320 u
   (08) before any tranche.** (iv) **Stage 4's four scalar labels lose the joint outcome** SKILL_PASS / PRODUCT_FAIL / RELEASE_HOLD
   (07, 09; T12 precedent `t12/protocol.md:119-124` verified). **Ruling: day-7 prints a tuple per line — scientific verdict / readiness /
   release status — with 09's terminal reducer.**
2. **Two defects nobody in Round 8 saw, both in the paced-identity package, both pre-data fixes (04):** (a) **the clock seam** — one
   `cutoff_ns` per request is checked at visibility (`controller.py:60`) *and* at consumption (`:268-277`), consumption is accepted only
   when `target_frame == frame` (`:189-233`), and the two-tick *publication* margin lives in the not-yet-built candidate adapter
   (`README.md:31-33`). If the adapter sets `cutoff = T_r − 2 ticks` every on-time result is rejected at consumption; if it sets
   `cutoff = D_cons` nothing enforces the publication margin. Not a proven bug (no adapter exists) — a spec ambiguity that must be
   closed with two cutoffs and a fake-clock four-case test before any real timing. (b) **the W/K guard** — `events.py:144-157` refuses
   `workers > 1 or max_pending > 1` without `owner_multi_worker=True`, and `transport.py:66-71` forces `SNAKE_DQN_DEVICE=cpu`. Round 8's
   "run all W cells Wednesday, interpret later" cannot execute: the expanded cells need a recorded **probe-only** authorization (a
   smaller decision than the serving clause, which I recommend putting to the owner today), and the MPS cell needs an explicit device
   adapter. Silence ⇒ expanded cells are NOT_MEASURED, not FAIL.
3. **The plan is powered for a T3-scale effect, and PLAN.md must say so in its first paragraph.** With measured SDs (365/263/212):
   L2′ pooled SE 25.7 (FUND_SYSTEMS ≈ 20–30 % at a true +40, ≈ 80 % at +80); L4 at 32/mix SE **29.3** (05, 07; checked), one-sided 90 %
   half-width 37.6, so "point ≥ 40 ∧ LB90 > 0" at L4 passes ≈ 50 % at a true +40 and ≈ 90 % at +80; strict adds another ≈ 50 % / 90 %
   point filter. **Unconditional qualification this week at a true +40 ≈ 5–8 %; at a true +80 ≈ 60–65 %.** The week can qualify a
   T3-sized served-search gain; it cannot distinguish +40 from 0 on any single stage. That is a design property to be stated, not a
   reason to loosen any bar (05 right, Round 8 §c5 stands).
4. **Y0 is half-landed and the Wednesday L1′ is a target, not a promise.** The prose amendment (`AMENDMENT_08.md`) is correct and
   already contains 08's liability arithmetic (384 timeout-min per chunk ⇒ estimate ≤ 192 min; "simultaneous constraints") — Round 8's
   "1.09/s if two chunks run on two slots" is **withdrawn** (08: parallel slots cut elapsed time, not aggregate timeout liability).
   The code (`readiness.py`, `manifests.py`) still enforces the single-plan 16-u envelope; L2′ cannot dispatch until Sol-S lands the
   chunk-aware checks (Z-0). L1′ needs Y1's Mac adapter (03: "Y1 due Thursday does not establish readiness for Wednesday evening");
   earliest honest window = the first quiet window *after* Y1's Mac half is accepted and Phase 0 is confirmed idle — Wed night if Sol-S
   lands it today, else Thu.
5. **Line R is blocked at its root this week and the plan must say what unblocks it.** E9's BLOCKED print leaves no registered E2
   design; the queue's "E2 when a slot frees" is dead text. A **prospective E9 replacement** (Z-4: three candidate designs simulated
   under the existing E9 machinery, the funding judge admits the first meeting .60/.10) is the only path to any E2, hence to E6/E7
   endpoints, E12, E16 and beyond. E3's mechanical failure and E14's Thu window are the second gate. **Earliest executable E16 is next
   week** under any outcome. R's day 7 = disposition packet (E9′ ruling, E4 receipt, E3 print, E14 receipt, Y3 receipt, E15 envelope
   ratification, burn manifest) — NOT_READY for release by design, as Round 8 ruled; now also NOT_READY for E16 by code.
6. **Week 2 (10):** "both lines may enter one family" is lawful only if no strict family opened in week 1. **Ruling:** no alpha split
   (10's .025/.025 would enlarge the strict bank past the 29-h ceiling and is unnecessary when each artifact gets exactly one family and
   no retry); instead the handoff carries FAMILY_UNOPENED / FAMILY_OPENED with the frozen membership, and week-2's family is registered
   prospectively from artifacts **without a prior strict look** — an S that opened keeps its result and is not retested; an S deferred
   unopened re-enters with the same identity (§c6).
7. **O1 GO changes the shortlist's ceiling, not this week's plan.** Harvest branches are found by the same exact-sim search Line S
   would serve, but under a *different operator* (the T3 certificate rule changes an action only when the bare continuation dies
   within H300; a harvest operator changes actions on living roots). O2 is unpaced and offline; a harvest-aware served controller is a
   new Claim B identity for week 2 at the earliest. Recorded as lever 1's upside and as a week-2 item (§d).
8. **Shortlist:** served search **25 % / 14 %** (down 3/2: Y0 code unlanded, clock seam, W/K guard, L1′ slips a day; offset by O1 GO as
   upside); representation **25 / 14** (E3 mechanical failure; E14 Thu at risk); PPO backbone **18 / 8** (E9 BLOCKED, no executable E2);
   engine **28 %** (E1b failed mechanically, fix landed); search-in-the-loop 15 / 10; **offense via search (O2) new 5 % add-on**;
   archive 6; opponent 4; world model / league 2.

## (b) Per-advisor verdicts

| ID | Persona | Verdict | Why (verified facts in **bold**) |
|---|---|---|---|
| 01 | Operator at 03:00 | **keep** | The round's anchor: **`check_paced` requires 108 runs / 16 pairs per mix / 6 sessions (`receipt.py:160-163, 250-261`); `base_ready` vs `full_ready` (`:346-354`); `l3_identity_prerequisites` exported independently (`:412-417`); PREFLIGHT PRECHECK_ONLY, paced NOT_RUN (`:36-40`)** — all verified. Three-receipt admission (Friday provisional → post-L3 → pre-strict) with create-only JSON and a Codex rehearsal of seven receipt-only cases is adopted as Stage 2's text. `95 − 9 − 32 − 11 − 6 − 2 − 6 = 29 h` checked. Right that a tired operator resolves a contradiction by copying a historical bank size — the exact laundering vector. |
| 02 | Calendar red team | **keep** | Same seam, plus the two numbers that bite: **8 unpaced replay hours reserved after Friday ⇒ slack −1 h** (`PREFLIGHT.md:94-98` verified); **four 1-h chunk overheads vs AMENDMENT_08's 900 s per chunk** (checked: the amendment prices 3,600 s total, so 02's 31.6 timeout-h is the four-separate-hours case, not the registered one). Right that parallel slots reduce elapsed time, not liability. "Monday product failure" → "whenever product adjudication completes" adopted. Dated table adopted as Y10's ledger. |
| 03 | Builder-capacity red team | **keep** | **X3 implemented as 40 u / 20 h with a separate funding receipt required (`REGISTRATION_AMENDMENT_02.md:17-40`)**; **X7 scaffolding landed, root selection / cutoff margin belong to the unbuilt adapter (`README.md:27-35`)**; **E9 BLOCKED (`E9_RULE.md:56-61`)**; **`run.py:435` binds target, CLI has `--resume` but no pause target (`:943-951`)** — all verified. Item-specific Stage 0 gates (Thu 12:00 for Y0/Y1/X3/Y4/Y6/Y7/Y9; Fri 12:00/20:00 for X7/adapter/Y5), one writer + different named reviewer per item, Codex-I / Codex-V split, "a slip blocks dependents, never buys reduced evidence" — adopted. Its capacity table is labelled planning, accepted as such. |
| 04 | Apple-silicon latency | **keep** | The round's two code catches: **single `cutoff_ns` checked at visibility (`controller.py:60`) and consumption (`:268-277`), consumption only at `target_frame == frame` (`:189-233`)** and **W>1 or K>1 refused without owner authorization (`events.py:144-157`); transport forces CPU (`transport.py:66-71`)** — verified. `1/2.4 − 2/24 = 333 ms`, `1/1.6 − 2/24 = 542 ms` checked; its K=8 tail cap `(8−1)×5/24 = 1.458 s` matches Round 7 §c5. W/K cell matrix with per-stratum tails, frozen offered-root trace, ≥ 29/20 (12 s) and ≥ 36/24 (15 s) valid consumptions, "no consumption ⇒ cannot pass", and the `exp/bl-l1/<window>/<attempt>/` evidence packet are adopted as L1′'s text. Conjecture (W = 3 > W = 4) labelled. |
| 05 | Evaluation / trust skeptic | **keep** | **L4 SE 29.31 at 32/mix; half-width 37.56** checked; its three algebraic rows (mass met but frozen ≤ 0 ⇒ TRANSFER_UNRESOLVED; transfer met but mass INCONCLUSIVE; 1/1/121 passes both) show the arrow "L4 → strict" is underdetermined. Joint receipt (`data_status / mass_status / transfer_status / strict_admission / release_status`, precedence CLOSED_BELOW_BAR → BAR_MET → INCONCLUSIVE) adopted verbatim. Correctly withdraws its own `100M_ADMISSIBLE` objection (**`AMENDMENT_02.md:93-99` now maps it to 32M preparation**, verified). Right that "no per-mix UB90 < 0" protects only scripted once frozen/mixed points are > 0. |
| 06 | Large-scale on-policy RL | **keep** | **E9 .270/.307 < .60 (`E9_RULE.md:43-55`)**; **E3 GO with no architecture (`PROTOCOL.md:72-76`)**; **E16 A/G only, A alone refused (`admission.py:13, 165-169`)** — verified. Its print-licence table (E2 GO ⇒ planning only; E6 NEEDS_SCALE ⇒ bridge; E12 PASS ⇒ critic only; E17 ⇒ another short test, never a 100M probability) is the correct reading of Rounds 3–5 and is carried into PLAN. 192-u arithmetic (62.3 h / 124.5 u at 2k/s leaves 29 u; 161.6 u at 1.5k/s) checked; "funded" withdrawn. Stale on E8 only at its snapshot. |
| 07 | Release / serving qualification | **keep** | **T12: SERVING_FAIL / HOLD, fix only the serving defect under a new producer version, T7 not revisited (`protocol.md:119-124`)**; **pin validator binds checkpoint + veto + strict files only (`served_checkpoint.py:133-149`)**; **X7 BUILD_VERIFIED_NOT_QUALIFIED** — verified. Serial arithmetic `29.333 h; 384 worlds 22.222 h leaves 7.111 h; 640 = 37.037 h` checked. Dated acceptance contract (Thu 20:00 freeze → Fri provisional → Sun 08:00 integrity → post-L4 sizing → strict → product → Tue 21:00 seal) and the two owner signature texts (release authorization vs acknowledgment) adopted as Stage 4's text. |
| 08 | Distributed systems / Colab parity | **keep** | **`supervisor.py:192` timeout ≥ 2× estimate; `:201` timeout × rate ≤ cap; `:817-830` account check reserves active caps only** — verified. Liability table (4 × 100M = 222 u at 2k/s; 336M remaining = 187 u > 153.6; E17 31.1 u fits 40 − 8; E13 13.5 u fits 18 − 3.6) checked. **"1.09/s on two slots" is wrong** — ruled (§c3). 320-u E15 envelope with 64 reserve adopted as the ratification target. Typed E13 index (`execution_platform / verification_host / artifact_sha / evidence_scope`) adopted. Slot-assignment rule (record terminal export before assigning a freed slot) adopted. |
| 09 | Print interpreter | **keep** | **E3 `selected_architecture = null` (`fit.py:245-257`, `PROTOCOL.md:72-76`); E16 binds E3/E14 architecture identity (`admission.py:236-267`); readiness still enforces the old envelope** — verified. The 108-cell table is correct and collapses cleanly because L1′/E3 are independent across S/R: S routes {K, I, N, T, U, A} × R routes {G, H, C}; **"currently stop at E9 in every G cell"** is the sharpest line of the round. Terminal reducer (registered stop ⇒ scientific KILL scoped to the study; mechanical ⇒ NOT_READY; completed strict FAIL ⇒ KILL of that artifact; product/serving fail ⇒ RELEASE_HOLD) adopted verbatim. Owner silence = W1/K1 "does not kill search" — agreed. |
| 10 | Owner-trust red team | **partly** | Right that the week-2 sentence is unconditional and that comparator-only execution opens the family (**`FINAL_FAMILY_REGISTRATION.md:10-18`** verified); right on the six-record sealed index and on "E13-ready ≠ E13 PASS" (**`PARITY_RULE.md:111-131`**). Its alpha split (.025 week-1 singleton + .025 reserved) is **not adopted**: it would enlarge the strict bank past the 29-h ceiling, and the error it guards against (a second qualification attempt of the *same* artifact) is already excluded by the no-retry rule; distinct artifacts in prospectively registered families on fresh banks are sequential registrations, not an alpha reset (§c6). `1 − .95² = 9.75 %` is correct for two attempts of one artifact, which the plan forbids. |

Factual errors found: none in the advisors that change a conclusion. Stale-at-snapshot only: Y0 prose (`bc5cfffd`) landed after all ten
wrote (03, 08, 09 said "prose alone has not implemented Y0" — now true in the other direction: prose yes, code no). Corrections to the
Round 9 fact sheet (judge's own): (i) Round 8's "1.09/s if two chunks run on two slots" withdrawn; (ii) Round 8's "L1′ Wed evening,
all cells" is not executable as written (W/K guard, Y1 dependency); (iii) Rounds 7–8's time stamps were ~3.5 h fast — the calendar
has a full Wednesday; (iv) the strict stage must be stated as a ≤ 29-h ceiling, not "L4-sized"; (v) "E2 when a slot frees" is dead text.

## (c) Disagreements ruled

1. **Stage 2 — one Friday predicate (Round 8 Y10) vs three dated admissions (01, 02, 03, 07, 09 — unanimous among those who read
   `receipt.py`).** Three admissions. **Fri 23:59 provisional reservation** requires: all four L2′ chunks terminal and valid with the
   governing aggregate FUND_SYSTEMS; the authorised L1′ cell PASS (silence ⇒ W1/K1); production X7 `l3_identity_prerequisites = PASS`;
   Y5 schema + L3/L4 registrations frozen; Y6 reviewed; X8 window table published; provisional invoice with a **strict-time ceiling**
   `C_strict = 95 − 9 − 32 − 10.67 − 6 − 2 − 6 − (post-Friday replay + semantic hours)` ≥ the smallest admissible bank's paced hours.
   **Post-L3 gate** (before L4): DELIVERY_READY + complete three-oracle PASS incl. replay of every registered attempt; anything else blocks
   L4 and releases downstream hours. **Pre-strict gate** (after L4 seal + audit): 05's joint tuple, deterministic sizing from L4 paced
   variance (historical SDs as a labelled sensitivity), `slack = (Tue 23:00 − now) − remaining serial work − 2 − 6 ≥ 0`, sized bank
   ≤ `C_strict`; else NOT_READY (defer), never a smaller N. Comparator-only evaluation opens the family {Claim B}.
2. **Stage 0 — blanket Thu 20:00 (draft) vs item-specific gates (03).** 03. Thu 12:00 code cutoff / 20:00 acceptance for Y0 (code), Y1,
   X3 acceptance, Y4, Y6, Y7, Y9; **X7 production adapter, replay adapter, clock-seam fix and Y5 are exempt: integrated source frozen
   Fri 12:00, prerequisites + Y5 acceptance Fri 20:00**; Y3 before E17/32M; Y8 before any E13/100M; X8 before Friday admission. Each item:
   one writer, named different reviewer, files, acceptance command, receipt path/hash; Codex-I builds Y6/replay, **Codex-V** verifies.
   A missed gate blocks dependents only and never extends the week's reservation.
3. **L2′ chunk arithmetic — Round 8 ("1.09/s on two slots") vs 08/02.** 08. Liability is per chunk: 384 timeout-min ⇒ estimate ≤ 192 min
   ⇒ 23,120 batches + 900 s overhead in 192 min ⇒ **R ≥ 2.18/s per occupied 12-worker runtime, regardless of how many run in parallel**
   (`AMENDMENT_08.md` already states this; Round 8 §c3's parenthetical is withdrawn). Two chunks in parallel cut wall time to ≈ 2 × 3.2 h
   plus merge; they do not lower the bar. Y1's ruler decides dispatchability; R < 2.18 ⇒ NOT_DISPATCHABLE (the funding judge may
   reprice the envelope prospectively; the bank never shrinks).
4. **L1′ — "all cells Wed, interpret later" (Round 8) vs the W/K guard (04).** 04, by code. K1 cells (W1/K1, CPU) run in the first
   qualified window; **expanded cells (W 1–4 / K 8, and the MPS backend) run only with a recorded owner *probe-only* authorization**
   (`owner_multi_worker=True` for timing probes, explicitly not a serving-clause answer) and a verified MPS device adapter; otherwise
   NOT_MEASURED. I recommend Claude put the probe-only question to the owner today as a separate, smaller decision; the serving clause
   (W ≤ 4 / K ≤ 8 in the Claim B identity) stays pending and still selects which measured rows count on Friday. Timestamps:
   `T_r` (target tick), `D_pub = T_r − 2 ticks`, `D_cons = T_r + 1 tick`; S = visibility − admission, C = consumption − admission; the
   bars 333/542 ms apply to **C p99**; the K-8 tail cap S_p99 ≤ 1.458 s applies to expanded cells. Fake-clock four-case test
   (on-time accept; publication-late; consumption-late; wrong-frame) must pass before real timing (Z-2).
5. **L4 → strict arrow — Round 8's weakened clause (budget status) vs 05's joint tuple.** Both: the weakened clause stands as the
   *transfer filter*; 05's tuple makes it executable (mass precedence CLOSED_BELOW_BAR → BAR_MET → INCONCLUSIVE; transfer FILTER_MET /
   TRANSFER_UNRESOLVED separately; `strict_admission = ADMITTED` only on BAR_MET ∧ FILTER_MET ∧ guards ∧ slack ≥ 0). Neither status
   overwrites the other; `release_status = RELEASE_HOLD` on every L4 receipt.
6. **Week 2 — alpha split (10) vs "both lines next week" (draft) vs one family per artifact (judge).** One family per artifact, no
   retry, no split. The handoff records FAMILY_UNOPENED or FAMILY_OPENED{members, p-values, outcome}. Week-2's family is registered
   prospectively from artifacts with no prior strict look (R's nominee; planner-over-R; an S deferred *unopened* with the same identity).
   An S that opened and failed is closed; an S that opened and passed is qualified and not retested. A repaired S is a new artifact.
   Fresh banks are mandatory in every case (burn manifest).
7. **R's deliverable — "executable, funded E15" (draft) vs disposition packet (06, 08, 09).** 06/08/09. E9 is BLOCKED and the frozen
   escalation is exhausted, so a **prospective E9 replacement design** (Z-4) is the first R item; nothing in R can execute before the
   funding judge admits it. E15's 192-u envelope is withdrawn as "funded"; 08's 320-u envelope (64 reserve, 2× liability, Y3-qualified
   pause points) is the ratification target — the owner's budget is effectively unlimited, so the fix is to invoice honestly, not to shrink.
   Day 7 for R prints decisions + blockers + receipts; it does not print a nominee and cannot print E16 this week.
8. **Day-7 labels — four scalars (Round 8) vs tuple + reducer (07, 09).** Tuple + reducer: `scientific_verdict ∈ {qualified-skill,
   scientific KILL(scope), INCONCLUSIVE, none}`, `readiness ∈ {READY, NOT_READY(reason)}`, `release_status ∈ {QUALIFIED_AWAITING_OWNER,
   RELEASE_HOLD(SKILL_PASS/PRODUCT_FAIL | serving FAIL | INVALID/INCOMPLETE), NONE}`. Owner front page carries exactly one of 07's two
   signature texts.
9. **O1 GO — does offense enter this week's plan?** No (judge, on the record): O2 is unpaced/offline (≤ 24 u, T34 lane); the served
   harvest operator is a different identity from the T3 certificate controller; nothing paced can test it before week 2. It enters the
   shortlist as lever-1 upside and the week-2 plan as a registered item conditional on O2.
10. **A100/H100.** Unanimous no for the fourteenth time; nothing on either line is GPU-bound this week.

## (d) Running shortlist (carried from Round 8, updated)

| # | Lever | Conf. push / beats B8 at bar | Cost to first signal | Most informative next experiment |
|---|---|---|---|---|
| 1 | **Served exact-sim search on this Mac** (T3 certificate controller, reachable-root EDF burst, W/K-bound Claim B identity, three oracles) — the week's only possible release-track artifact; **upside: O1 GO (harvest branches exist on 35/48 fresh worlds) ⇒ a harvest-aware operator is a week-2 identity** | 25 % / 14 % (down 3/2: Y0 code unlanded, clock seam, W/K guard, L1′ slips; powered for +80 not +40) | Z-0 + Y1 ≤ 4 u → L2′ 4 × 16 u → L1′ 3 Mac-h → L3 9 + L4 32 paced h → strict ≤ 29 h | **Z-0 (chunk-aware readiness) ∥ Y1 ruler ∥ Z-2 (clock seam) → L2′ chunks; L1′ K1 cells in the first qualified window** |
| 1c | RL endpoint as planner continuation (E20) | conditional | 8 u | next week |
| 1d | **Harvest-aware served operator** (new; from O1/O2) | 5 % add-on | O2 ≤ 24 u (registered, T34) | O2 print (D+O vs D vs bare); week-2 identity only |
| 2 | **Representation: geometry residual on the F parent** | 25 % / 14 % (down 3/2: E3 mechanical failure; E14 Thu window at risk; E16 needs E3 architecture + E14 PASS + an E12-qualified parent that cannot exist before E9′) | E3 relaunch (16 u) → E14 3 Mac-h → E16 40 u | E3 relaunch → E14 bare cells |
| 3 | **Long-horizon on-policy RL from B8** (backbone) | 18 % / 8 % (down 4/1: E9 BLOCKED; no executable E2 design) | Z-4 0 u → E4 (running) → E2 20 u → E6/E7 24 u → E12 16 u | **Z-4 prospective E9 replacement** (first R item) |
| 4 | **GPU-resident exact engine** (Colab only) | 28 % (down 2: E1b failed mechanically; fix `88adb3e2` unlaunched) | E1b relaunch 20 u | E1b relaunch when a slot frees |
| 5 | Search-in-the-loop | 15 % / 10 % | after E1b | E16-S (next week) |
| 6 | Restart archive / PLR | 6 % | E7 + E10 | E10 envelope |
| 7 | Opponent-conditioned add-on | 4 % | after E3b | — |
| 8 | World model; league | 2 % / 2 % | — | not funded |
| 1b | Retention sub-arm | conditional | 36 u | on the collateral trigger |

## (e) Still unresolved

- **The owner's W/K serving clause** (pending) and, separately, a **probe-only authorization** for L1′'s expanded cells (new ask).
- **S per batch on Watch6 opponents on this Mac** (L1′) and **R on L4** (Y1) — still unmeasured; L2′ cannot dispatch and L1′ cannot be
  read without them.
- **Whether the chunk-aware readiness code (Z-0) and the clock-seam fix (Z-2) land by Thu 12:00 / Fri 12:00** — both are 03:00-operator
  blockers if they slip.
- **E9′** — whether any prospective design reaches .60 joint funding at +40 under the measured SDs; if none does, Line R's week is E4 +
  E3 + E14 + Y3 + envelope ratification only.
- **E3's print** (relaunch pending) and **E14 on the E3 architecture** — decide whether E16 has any executable treatment next week.
- **L0′ per-doom window table** (sets 1.6 vs 2.4 roots/s) — still unpublished (X8).
- **Phase 0 idle** — an ETA, not a receipt; must be confirmed before any Mac window.
- **O2's outcome** — the only measurement that could raise lever 1's ceiling above T3's scripted-only +92.
- Strict-bank sizing from L4 variance vs the ≤ 29-h ceiling — the plan's honest statement is that a +40-scale effect will most likely
  print INCONCLUSIVE/NOT_READY this week; only a T3-scale effect qualifies.

## (f) Experiments Claude can run now (≤ 20 u each unless stated; dispatch via Sol between rounds; report in LIVE_FACTS.md)

**Z-0 — Land Y0 in code (0 u; Sol-S 0.25 day; today, before Y1).** `readiness.py` validates a *chunk* plan: counts 16/16/8 per chunk,
four chunks disjoint and covering `BANK_L2.json` exactly, per-chunk `0 < 2 × estimate ≤ timeout ≤ 384`, `unit_cap 16`, `reserve 3.2`,
R from Y1's receipt ≥ 2.18/s, parity admitted; `manifests.forecast()` prints `23,120/R + 900 s` per chunk. Kill: none — bookkeeping;
without it L2′ is NOT_DISPATCHABLE under both the old and the new rule.

**Z-1 — Slot priority as slots free (0 u; Claude; standing rule this week).** Record terminal export/release before assigning a freed slot
(08). Order: **Y1 ruler (≤ 4 u, short)** → E3 relaunch (R's gate; 16 u) → E1b relaunch (20 u) → L2′ chunks 1–4 as parity passes (two in
parallel when two are free) → O2 (≤ 24 u) → E4 already running. Y1's R < 2.18/s ⇒ L2′ NOT_DISPATCHABLE; its slot goes to O2.

**Z-2 — X7 clock-seam fix + fake-clock test (0 u; Sol-C 0.5 day; before any real timing).** Requests carry `publish_cutoff_ns`
(= T_r − 2 ticks) checked at visibility and `consume_cutoff_ns` (= T_r + 1 tick) checked at consumption; `_matches` unchanged. Fixture:
on-time accept / publication-late reject / consumption-late reject / wrong-frame reject, under a fake clock; replay reducer reproduces
all four. Kill: any case mis-dispositioned ⇒ X7 `l3_identity_prerequisites` cannot print PASS; L3 blocked.

**Z-3 — L1′ as 04's contract (0 u; 3 exclusive Mac-h; first window after Y1's Mac adapter is accepted and Phase 0 is confirmed idle —
Wed night if Sol-S lands it today, else Thu).** K1 cells (W1/K1, CPU) always; expanded cells and the MPS backend only with a recorded
probe-only authorization and a verified device adapter. Frozen offered-root trace at 5-decision spacing; ≥ 29/20 (12 s) and ≥ 36/24
(15 s) valid consumptions at 2.4/1.6 roots/s; C p99 ≤ 333/542 ms for K1; S p99 ≤ 1.458 s for K8 cells; zero overruns, update p95 ≤
11.688 ms, cadence ≤ 5 %, client p95 ≤ 10 %. Evidence packet at `exp/bl-l1/<window>/<attempt>/` (manifest, cells, sealed events,
mac-receipt, independent audit). Kill: no authorised cell passes ⇒ S's reservation released this week (NOT_READY, reason timing FAIL);
missing/unauthorised cells ⇒ NOT_MEASURED, S deferred.

**Z-4 — Prospective E9 replacement (0 u; Sol-R 0.5 day + funding judge; before any E2).** Using the existing E9 simulator and SDs,
evaluate three prospective designs and admit the first meeting joint funding ≥ .60 at +40 (survival ≥ 0) and family null-pass ≤ .10:
(a) fixed 192/64/64 fresh screen (SE ≈ 16.6; inside E2's cap at the E4-measured episode rate, else INCOMPLETE); (b) 128/48/48 with the
E6 route reported, not Holm-coupled (one .05 E2 route, R/F Holm); (c) two-stage sequential 96/32/32 → +96/32/32 with a registered
futility look per P7 §5. Kill: none meets the bars ⇒ E2 stays BLOCKED; R's week is E4 + E3 + E14 + Y3 + envelope ratification; say so.

**Z-5 — E15 envelope re-ratification (0 u; Sol-R 0.25 day; funding judge; before any 32M).** 08's contract: 320 u incl. 64 reserve;
admit a tranche only if `reconciled debit + outstanding 2× liabilities + liabilities of all remaining mandatory work + reserve ≤ 320`;
chunk estimates ≤ 4 h, timeout = 2× ≤ 8 h, snapshot cadence ≥ 10× measured export; Y3 receipt mandatory; unfundable ⇒ pause at the
Y3-qualified checkpoint, NOT_READY. Kill: missing invoice ⇒ no dispatch.

**Z-6 — Stage-2/3/4 executable receipts (0 u; Codex-I 1 day + Codex-V 0.5 day; Thu 12:00/20:00).** Create-only JSON at
`exp/week1-20261021/admission/{friday,post-l3,pre-strict}.json` (01), `L4/decision.json` + `audit.json` with 05's tuple, the day-7
reducer (09) with the collapsed S × R route table, the typed E13 index (08), and 07's two owner-signature texts. Codex-V rehearsal:
Friday with paced NOT_RUN; oversized bank; missing seal; negative slack; TRANSFER_UNRESOLVED; repaired scheduler; comparator-only
opening; product FAIL after strict PASS. Kill: any case admits strict from incomplete evidence or deploys ⇒ NOT_READY until fixed.

**Z-7 — Week-1 handoff index schema (0 u; Sol 2 h + Codex 4 h; schema today, populated continuously, sealed Tue 23:00).** 10's six
records (identity, evidence, family incl. FAMILY_UNOPENED/OPENED, worlds/burn, resources with liabilities, decisions incl. verbatim
owner answers or PENDING). Kill: unresolvable hashes or liabilities ⇒ dependent week-2 dispatch blocked.

**Z-8 — Owner asks (0 u; Claude; today).** Two separate questions with the measured basis: (1) the serving clause (W ≤ 4 / K ≤ 8 in the
Claim B identity; silence ⇒ W1/K1, which needs C p99 ≤ 333–542 ms on one thread, no measured basis); (2) **probe-only** authorization to
run expanded L1′ cells for measurement (no serving change). Record the verbatim answers in LIVE_FACTS.

**Unchanged:** Y1 (ruler, ≤ 4 u), Y3, Y4, Y5, Y6 (+05's L4 transition tests), Y7, Y8 (superseded by Z-5 for E15), X3 acceptance, X5, X8,
E3/E1b relaunches, E4 (running), O2 (registered, Sol building), E14 bare cells (Thu, 3 Mac-h, after the E3 architecture receipt),
contract amendment text (Round 7 §f). A100/H100: no.

**Dispatch order:** Builders today — Sol-S: Z-0 → Y1 (L4 half + Mac adapter) → Y7; Sol-C: Z-2 → X7 adapter → Y5; Sol-R: Z-4 → Y3 → Z-5;
Codex-I/V: Z-6, Z-7, Y6, Y9. Colab (slots per Z-1). Mac: Phase 0 idle receipt → Z-3 (first qualified window) → X5 → E14 (Thu) →
[Fri 23:59 provisional] → L3 → post-L3 gate → L4 → pre-strict gate → strict (≤ C_strict) → product → serving → Tue 21:00 seal.
Budget this inter-round: Y1 4 + L2′ 64 + E3 16 + E1b 20 + O2 24 + E4 2 ≈ 130 u; everything else 0 u.
