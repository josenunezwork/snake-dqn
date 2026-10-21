# PLAN.md — the big-lever push (week 1: Wed Oct 21 → Tue Oct 27, 2026, Pacific time; week-2 handoff)

Written by the round-10 Fable judge, 2026-10-21 ~03:30 PT, from `round-01..10/JUDGMENT.md`, `LIVE_FACTS.md`, `QUEUE.md` and the
repo at `4f15254f`. Authority order: the owner > registered study rules/amendments (byte-pinned) > this plan > any advisor memo.
Nothing here dispatches, registers, spends or deploys by itself; every action below names who does it and what receipt proves it.
Owner-only decisions: release OK; the serving clause (W/K); the probe-only grant. Everything else is pre-delegated (owner 2026-10-20).

## §0 Power statement, scope, and what a release claim requires

**Power (measured T23 paired SDs 365 / 263 / 212 scripted / frozen / mixed; equal-mix SE = √(Σ s²/n_m)/3).**
L2′ (Colab, unpaced, 32/64/64) pooled SE 25.7; FUND_SYSTEMS needs frozen ≥ 42.6 ∧ mixed ≥ 34.3 ∧ pooled ≥ 40, ≈ 20–30 % at a
uniform true +40, ≈ 80 % at +80, and ≈ 1 % for a scripted-only (0, 0, 240) pooled +80 — by design: scripted-only gains are not the
lever. L4 (Mac, paced, 32/mix) SE 29.3, one-sided 90 % half-width 37.6: "point ≥ 40 ∧ LB90 > 0" passes ≈ 50 % at +40, ≈ 90 % at +80;
an observed +40 at L4 has LB90 ≈ +2.4, so a single paced stage *can* separate +40 from 0 — what it cannot do is pass the point bar
reliably at +40. Strict (illustrative 128/mix, SE 14.7) has ≈ 86 % power for mass > 0 at +40 and the point bar halves that again.
Three stacked point filters ≈ 12.5 % at a uniform +40. **Unconditional qualification this week ≈ 5–8 % at +40 and ≈ 60–65 % at +80
are planning sensitivities — UNVALIDATED until Y6 (§2) runs the scenario set (40,40,40), (80,80,80), (0,0,240), below-bar, partial
nulls.** The week qualifies a T3-scale served-search gain; a +40-scale effect most likely prints INCONCLUSIVE / NOT_READY. No bar is
loosened to change this. "Qualified" means positive mass evidence + observed pooled ≥ 40 + registered survival/product margins; it
does not certify a population gain ≥ 40.

**Scope.** One release track this week: **S** = served exact-sim search on this Mac (T3 certificate controller, reachable-root EDF
burst, W/K-bound identity, Claim B "Assisted hero — [controller ID]"). **R** (long-horizon PPO + geometry) is research-only this
week; it cannot nominate before week 2 (E9 BLOCKED; E4 failed; E16 A/G-only needs E3 architecture + E14 PASS + E12 parent).
Harvest (O1 GO) is a week-2 S identity. Serving and gates stay on this Mac (owner 00:30 PT). A100/H100: none. Colab is **not** the
binding resource (142–202 u planned vs ≈ 1,140 available); the Mac's serial clock and the builders (Codex-V most of all) bind.

**Release claim = all of the following, bound to one controller digest C, for mixed Watch6 only:**
Mac strict PASS (candidate paced, bare comparator unpaced under its registered clock-independence receipt; pooled ≥ +40 ∧ LB90 > 0,
survival band v2, scripted no-death ≥ +.03, boost clause fixed pre-data, every deadline miss charged) ∧ product PASS (32 fresh paired
10-min sessions; death-rate paired 95 % UB < 0; relative mass LB95 ≥ −5 %) ∧ serving PASS (contract §4 timing bars, identity,
hero-only assistance, stale/wrong-life/restart, pins, label, Watch8/12 + Play isolation, rollback drains + shows resulting identity)
∧ independent audit of the sealed packet ∧ family {Claim B} frozen *before* the first strict episode (comparator-only counts) ∧ the
owner's release signature (§7, text A). A generic PASS string, a Colab receipt, an E13 transfer receipt, X7's identity PASS or B8's
STRICT_PASS satisfies none of these. Any change to scheduler, cutoffs, fallback, W/K, backend, recorder overhead or load envelope is
a new digest C′ with no inherited evidence. Monitoring (7 days AND 100k decisions) cannot complete by Tuesday; the packet says so.

## §0.1 State at 03:00 Wed Oct 21 (what the operator may rely on)

- Running on Colab: nothing confirmed. E3 (upload cap) and E1b (CUDA dtype; L4 attempt Refused) failed; **E4 failed 02:54 on the same
  upload cap** (`exp/bl-e2/run-e4.log`); O1 census-02 done. Cap = 3 assignments (measured). Verify with `colab usage` before any
  assignment; a slot is free only after a terminal create-only receipt + verified release.
- Prepared, not launched: E1b relaunch (`exp/bl-e1b/cuda-dtype-fix-20261021-01/`, commit `88adb3e2`); E3 recovery-02
  (`exp/bl-e3/recovery-02/DISPATCH.txt`); Y1 fixtures frozen (`research/bl_served_search/RULER_FIXTURES_08.md`, seeds audited in
  `exp/bl-l2/ruler-allocation-08/`). O2: not built (`t34-o2-20261021/` empty). E6/E7: validation v2 done, undispatched.
- Mac: Phase 0 (`phase0-selection`) ETA Wed 10:00–13:00 — an ETA. No Mac window exists until an idle receipt is written.
- Code gaps (verified): `readiness.py:29-41` / `manifests.py:66-95,212-214` single-plan 16-u; `analyze.py:217,223` legacy labels;
  `controller.py:60,118-138,268-277` one cutoff, no visibility timestamp, publication unguarded; `events.py:144-157` Boolean W/K
  guard; `transport.py:66-71` forces CPU; E9 BLOCKED (`E9_RULE.md:56-61`); E16 `IMPLEMENTED = {A, G}`.
- Owner asks PENDING: (1) serving clause W ≤ 4 / K ≤ 8; (2) probe-only grant for expanded L1′ cells. Silence ⇒ W1/K1, no probe.

## §1 Owner answers — the plan branches on each (record verbatim in `exp/week1-20261021/authority/{serving-envelope,probe}.json`)

| Serving clause | Probe-only | What counts at Fri 23:59 | What L1′ runs | Consequence |
|---|---|---|---|---|
| silence / NO | silence / NO | W1/K1 CPU cell only; bar C p99 ≤ 333 ms (2.4 roots/s) or 542 ms (1.6/s, if X8 publishes the 1.6 bar) | K1 CPU cells only; expanded + MPS = NOT_MEASURED | S lives or dies on one-thread latency (no measured basis today) |
| silence / NO | YES | same as above (probe PASS on expanded cells is diagnostic only) | K1 + expanded W1–4/K8 + MPS (with device adapter) | owner can read measured expanded rows before answering the clause |
| YES (W ≤ 4, K ≤ 8) | any | the frozen nominated cell among measured authorised cells; K8 tail cap S p99 ≤ 1.458 s; W=4 ⇒ ≤ 1.458 s, W=3 ⇒ ≤ 1.25 s, W=2 ⇒ ≤ 0.83 s | all cells | identity binds W, K, root rule, checkpoints 32→96→192→300, lead cap 432, fallback |
| YES after Fri 23:59 | — | nothing retroactively; Friday's record stands | — | cannot reopen a DEFERRED reservation this week |

A probe-only YES never enlarges serving authority; a serving YES never supplies a missing measurement. Conflicting records block
admission. Claude puts both questions to the owner at the first contact Wednesday, with the measured basis (none for one-thread).

## §2 Stage 0 — dispatch readiness (Wed 03:00 → Fri 20:00). Builders and gates.

Rule: one writer + a different named reviewer per item; Codex-I builds, Codex-V verifies; receipts are create-only JSON with
`.sha256`/`.complete` under `exp/week1-20261021/stage0/<item>/<source-sha>/acceptance.json`. A missed gate creates
`stage0/slips/<item>-<gate>.json` {builder, reviewer, gate, missing_evidence, blocked_descendants, remaining_hours, earliest_finish,
scientific_verdict_preserved, readiness=NOT_READY, next_authorized_action}; it blocks dependents only, never extends the week's
reservation, never buys reduced evidence. Y9 (Codex, today) names **hours**, not roles: Sol-S 16–22 h, Sol-C 28–36 h, Sol-R 16 h,
Codex-I 26–30 h, **Codex-V 36–42 h** — review rolls continuously Wed–Fri; a second independent reviewer must be named or items print
NOT_READY. Shared-file rule: Sol-S alone edits `bl_served_search/*`; Sol-C alone edits `bl_paced_identity/{events,controller,
transport}.py`; Codex-I owns the replay reducer and `bl_week1/*`; Sol-R owns `bl_ppo/*`.

| Item | Owner → reviewer | Content (verified gaps it closes) | Code / accept | Receipt |
|---|---|---|---|---|
| **Z-9 upload-path qualification** | Sol (per manifest family) → Codex-V | every manifest's Mac preflight exercises the real supervisor upload path at real byte sizes, or the `9f203c4f` hash-bound runtime reconstruction; prints forecast upload time ≤ 50 % of the cap; applies to E4, E3, E1b, Y1, L2′ chunks, O2, E2 | **Wed 12:00 / 16:00** | `exp/<job>/upload-preflight.json` |
| **Z-0 chunk execution path** | Sol-S → Codex-V | full-bank hash + four explicit 40-world (16/16/8) selections covering 160 exactly; runner executes a selection (`manifests.py:212-214`, `ceiling.py:380-404`); per-chunk `0 < 2×est ≤ timeout ≤ 384`, cap 16, reserve 3.2, R ≥ 2.18/s from Y1; per-assignment parity (`ceiling.py:74-87`) inside the 4-u parity envelope; versioned label map `FUND_MAC_SYSTEMS→FUND_SYSTEMS`, `KILL_SERVED_SEARCH_PUSH→PACKAGE_KILL` (`analyze.py:217,223`); pinned-analyzer merge only after four valid terminal chunks | Thu 12:00 / 20:00 (target Wed) | `exp/week1-20261021/l2/plan.json` |
| **Y1 ruler** | Sol-S → Codex-V | L4 half ≤ 4 u on the nine frozen fixtures (≥ 30 batches/root, 12 one-thread workers, aggregate R); Mac adapter for L1′ (same root bytes, same operator); semantic match across platforms or INVALID; timing platform-labelled, never transferred | code Wed; L4 run as a slot allows | `exp/bl-l2/ruler-*/receipt.json` |
| **Y7 ledger fields + TRANSFER_UNRESOLVED** | Sol-S → Codex-V | productive rescue / D300-only / re-death / newly caused deaths / sealer strata / per-mix denominators on L2′/L3/L4 exports; L4 clause (frozen point > 0 ∧ mixed point > 0 ∧ no per-mix UB90 < 0) | Thu 12:00 / 20:00 | with Z-0 plan |
| **Z-2′ clock seam (corrected)** | Sol-C → Codex-I (replay) → Codex-V | visibility timestamp at the coordinator after IPC decode; `publish_cutoff = T(r−2)` enforced in `publish()`; `consume_cutoff = T(r+1)` at consumption; integer `T(r)=t0+⌊r·10⁹/24⌋`; equality is late; four-case fixture (on-time / publication-late / consumption-late / wrong-frame) through controller **and** `replay.py`; versioned event schema | **before any qualifying L1′**; latest Fri 12:00 / 20:00 | `exp/bl-l1/preflight/<attempt>/clock.json` |
| **X7 production adapter** | Sol-C → Codex-V | root selection, EDF on publication deadlines, checkpoints 32→96→192→300, lead cap 432, K = queued+running+unconsumed, invalidation on executed deviation/respawn/epoch, bare on every miss; scoped authority: `authorized()` takes a scope (probe vs serving) and allowed cells, not a Boolean; MPS device adapter (both init paths, placement attested, sync before publish) optional | Fri 12:00 / 20:00 | X7 receipt `l3_identity_prerequisites` |
| **Y5 / Claim B chain** | Sol-C → Codex-V | versioned schema + validator at `exp/claim-b-release-20261024/<C>/{identity,family,strict/receipt,product/receipt,serving/receipt,rollback/receipt}.json`; pin schema beyond `served_checkpoint.py:172-217`; label with per-tick fallback disclosure and digest; rollback matrix (§5); validator recomputes clause verdicts from bytes; mutation tests (old strict receipt / Colab PASS / altered scheduler ⇒ nonzero) | Fri 12:00 / 20:00 | `exp/claim-b-release-20261024/validator-acceptance.json` |
| **Z-4′ E9 replacement** | Sol-R → funding judge (Fable/Opus stand-in) → Codex-V | 192/64/64 fixed (SE 16.6); survival alternative declared pre-data at +.03 (Δ = 0 ceiling ≈ .57 reported, unreachable by any N); option (c) only as a futility look under (a)'s final test; MC95: null-pass UB ≤ .10, funding LB ≥ .60 | Thu 12:00 / ruling by Thu 20:00 | `exp/week1-20261021/R/e9-prime/{design,oc,approval}.json` |
| **Y3 durability** | Sol-R → Codex-V | exact continuation on a *replacement* matched L4 through ≥ 3 updates + episode boundary + 1,000 actions; optimizer/PopArt/RNG/counters/worlds/opponents/carry/pending rollouts; transitive snapshot set (`recovery.py:47-76`); `target=100M` separate from `pause_at` | before E17 / any 32M; ≤ 8 u | `exp/week1-20261021/y3/<identity>/acceptance.json` |
| **Z-5 E15 invoice** | Sol-R → funding judge | 320 u incl. 64 reserve; `debit + outstanding 2× liabilities + remaining mandatory + reserve ≤ 320`; proposed / ratified / reserved / debited kept distinct | **before the first 32M** | `exp/week1-20261021/e15/<plan-sha>/invoice.json` |
| **Y6 family automaton + OC** | Codex-I → Codex-V | state machine (opens = first strict episode incl. comparator-only; members frozen; p = 1 for unavailable; no retry); OC with §0's scenario set; day-7 power field UNVALIDATED until accepted | Thu 12:00 / 20:00 | `exp/week1-20261021/Y6/operating-characteristics.json` |
| **Z-6′ gate registry** | Codex-I → Codex-V | `research/bl_week1/gates.py` + frozen `contract.json` (per item: argv, cwd, source/input hashes, builder, reviewer, criteria, deadline, deps, label map); `g <gate>` validates evidence, writes create-only receipts, never dispatches/deploys; reducer with the S × R route product (§7); slip schema; typed E13 index; signature texts; Codex-V rehearses nine synthetic failures (missing instrument, literal `FUND_MAC_SYSTEMS`, Friday paced NOT_RUN, post-L3 missing replay, oversized N, negative slack, comparator-only opening, product FAIL after strict PASS, changed packet after review) | Thu 12:00 / 20:00 | `exp/week1-20261021/stage0/z6/acceptance.json` |
| **X8 L0′ window table** | Sol-S → Codex-V | per-doom reachable-window duration / root count ⇒ the roots/s bar (2.4 vs 1.6) | before Fri 23:59 | `research/bl_served_search/L0_PRIME.md` |
| **Z-7 handoff INDEX schema** | Codex-I → Codex-V | six records (§7); schema Fri 20:00; sealed Tue 21:00 | Fri 20:00 | `exp/week1-20261021/handoff/INDEX.json` |
| **Sizing policy** | Sol-S + Codex-V | `admission/sizing-policy.json`: estimands, α/family, power alternatives, allocation, rounding, variance treatment, survival sizing, sequential schedule; missing ⇒ NOT_READY(SIZING_UNSPECIFIED) | **before L3 (Fri 20:00)** | `exp/week1-20261021/admission/sizing-policy.json` |

## §3 Stage 1 — measure (Wed → Fri). Colab dispatch order (cap 3) and Mac windows.

**Colab — ready-job priority (Z-1), not FIFO.** Before assigning a slot: `colab usage`, terminal export receipt, verified release
(`supervisor.py:981-986` retained ≠ verified stop), fresh inventory; write `exp/week1-20261021/dispatch/<seq>.json`. Each job needs its
Z-9 receipt. Priority: **Y1 ruler (≤ 4 u, short) → L2′ per-assignment parity (inside 4 u) → E3 recovery-02 (16 u) → E1b relaunch
(20 u) → E4 relaunch (≤ 2 u) → L2′ chunks 1–4 (16 u each; ≤ 2 slots at once; one look after four) → O2 (≤ 24 u, when built) → Y3
(≤ 8 u) → E2 (20 u, only with Z-4′ approval + E4 receipt) → E6/E7 (24 u) → E12 (16 u).** Blocked entries never block ready
successors; no paid waiting; no automatic retry (a mechanical failure needs a new prepared recovery with its own Z-9 receipt).
L2′ timing: each chunk ≤ 384 timeout-min; last chunk must start by **Thu 22:23** (one slot) or **Fri 10:00** (two slots) for a Fri
23:59 print. Y1 R < 2.18/s ⇒ L2′ NOT_DISPATCHABLE at 64 u this week (funding judge may reprice prospectively; the bank never shrinks);
its slots go to O2/E3/E1b. **Units, Stage 1:** 142 u base (Y1 4, parity 4, E4 2, E3 16, E1b 20, L2′ 64, O2 24, Y3 8); +60 u
conditional (E2 20, E6/E7 24, E12 16) = 202 u. These are sums of caps; the supervisor's 2× timeout liability is inside each cap.

**Mac — serial, exclusive, AC, lid open, idle receipt first.**
1. Phase 0 idle receipt (`exp/week1-20261021/mac/phase0-idle.json`, from tmux state + `docs/STATUS.md`) — ETA Wed 10:00–13:00.
2. **L1′ (Z-3)**, 3 h, only after: idle receipt ∧ Y1-Mac acceptance ∧ Z-2′ acceptance ∧ the adapter subset used by the cell accepted.
   **Latest start Thu 16:00** (else E14 loses Thursday). Cells per §1. Frozen offered-root trace at 5-decision spacing; bars C p99 ≤
   333/542 ms (K1), S p99 ≤ 1.458 s (K8); ≥ 29/20 valid consumptions in 12 s, ≥ 36/24 in 15 s; zero overruns; update p95 ≤ 11.688 ms;
   actions identical to the reference or the cell is INVALID; zero consumptions cannot pass. Packet `exp/bl-l1/<window>/<attempt>/`
   {manifest, authorization, cells, sealed events, mac-receipt, independent audit}. A ruler run before Z-2′ is micro-timing, not PASS.
3. **X5** (2 unpaced h): T7 harness on 48 disjoint dev worlds with B8 s3 (and F parent if any) ⇒ episodes/h; sizes bare controls (B).
4. **E14 bare cells** (3 h, Thu if step 2 started by 16:00; else NOT_READY(calendar) → next priced pre-S gap): B8 / F / F+G on the E3
   architecture (`admission.py:259-267`), encoder at W = 0; p95 > 11.688 ms or overrun ⇒ G unservable, E16-G not admitted.
5. Fixture-level semantic + replay qualification for X7 (the "2 + 8 unpaced hours", `PREFLIGHT.md:94-98`) — **prepaid before Friday
   where it is fixture work**; trace replay of Saturday's observations is a separate priced row (§4A). Record actual intervals in
   `exp/week1-20261021/mac/ledger.json`.
Nothing paced runs on the Mac before Sat 00:00; user play and thermals are logged as occupancy.

## §4 Stage 2 — three dated admissions (commands via `g`; absent registry ⇒ `NOT_READY(INSTRUMENT_MISSING)`, block dependents)

**(A) Fri Oct 23 23:59 — provisional reservation** (`g friday` → `exp/week1-20261021/admission/friday.json`). ADMIT iff all:
four L2′ chunks terminal + valid, pinned merge prints **FUND_SYSTEMS** (via the versioned map; raw labels preserved) ∧ the eligible
L1′ cell (§1) PASS with an independent audit ∧ production X7 `l3_identity_prerequisites = PASS` (`receipt.py:412-417`; paced may be
NOT_RUN) ∧ Z-2′ accepted ∧ Y5 validator + L3/L4 registrations + sizing policy frozen ∧ Y6 reviewed ∧ X8 published ∧ Z-9 receipts
for every dispatched job ∧ **priced serial invoice**: rows for L3 (9), L3 replay/audit (R₃), L4 (32), L4 replay/audit (R₄), strict
candidate (sized later; ceiling only), bare controls (B from X5), strict replay/audit (A), product (10.67 + setup), serving (6),
packet (2), recovery (6); `C_strict = 95 − 9 − R₃ − 32 − R₄ − 10.67 − 6 − 2 − 6 − B − A` must be ≥ the smallest admissible bank's
candidate-paced hours (24 fps × 5,000 × N / 3 mixes); each row names a measured rate or a bound and its receipt. Any missing row
⇒ `reservation = DEFERRED`, `readiness = NOT_READY(unpriced)`. Else: DEFERRED, scientific labels preserved, **no strict bank opens
this week**, Mac runs R's unpaced work and probes, S re-enters week 2 unopened with the same identity (§8).

**(B) Post-L3 gate** (`g post-l3` → `admission/post-l3.json`), before L4: L3 = DELIVERY_READY ∧ complete three-oracle PASS incl.
replay of **every** registered attempt (`receipt.py:160-163`: 108 runs, 16 pairs/mix, 6 sessions) ∧ R₃ actually consumed ≤ its row.
Else `L4 = BLOCKED`, `readiness = NOT_READY`, INVALID/STOP/INCONCLUSIVE preserved, downstream hours released.

**(C) Pre-strict gate** (`g pre-strict` → `admission/pre-strict.json`), after L4 seal + audit: L4 tuple from audited
`L4/decision.json` — valid complete paired repetitions only (else INVALID/INCOMPLETE ⇒ block); `mass_status`: UB90 < 40 ⇒
CLOSED_BELOW_BAR (precedes everything) ; point ≥ 40 ∧ LB90 > 0 ⇒ BAR_MET ; else INCONCLUSIVE; `transfer_status`: frozen point > 0 ∧
mixed point > 0 ∧ no per-mix UB90 < 0 ⇒ FILTER_MET else TRANSFER_UNRESOLVED (never erases BAR_MET); guards; timing; nonfinite ⇒
block. Then apply the frozen sizing policy **once** to L4's world-clustered paced variances (historical SDs as a labelled
sensitivity; a favourable L4 mean cannot lower N) ⇒ `strict/sizing.json` with per-mix N and total hours (candidate paced + bare
controls + strict replay/audit). `strict_admission = ADMITTED` iff BAR_MET ∧ FILTER_MET ∧ guards ∧ timing ∧ total ≤ C_strict ∧
`slack = (Tue 23:00 − now) − remaining serial work − 2 − 6 ≥ 0`. Else BLOCKED (NOT_READY), reservations released, **N is never
shrunk**, family stays UNOPENED. `release_status = RELEASE_HOLD` on every L4 receipt. Family `{Claim B}` is frozen and hashed
(`family-freeze.json`, `FINAL_FAMILY_REGISTRATION.md`) **before** the first strict episode; comparator-only execution opens it.

## §5 Stage 3 — the S paced block (only under Friday ADMIT; Sat Oct 24 00:00 → Tue Oct 27 23:00)

| When (PT) | Step | Cap | Labels / kill |
|---|---|---|---|
| Sat 00:00–09:00 | **L3** first paced smoke (16 fresh paired worlds/mix at 24 fps + 6 paired 10-min sessions; stale-rejection + deadline-fallback fixtures) | 9 h | INVALID (identity) / STOP (timing bar, UB95(G) < 40) / INCONCLUSIVE (no exercised intervention) / DELIVERY_READY |
| Sat 09:00 → +R₃ | L3 trace replay + audit, gate B | priced row | BLOCKED ⇒ release hours |
| → Sun ~17:00 | **L4** paced dev efficacy: 32 fresh worlds/mix, candidate and B8, each twice at absolute deadlines, randomised, every miss charged; ledger (Y7) | 32 h | tuple per §4C; `RELEASE_HOLD` |
| Sun 17:00 → +R₄ | L4 replay + audit, gate C with sizing | priced row | BLOCKED ⇒ no strict this week |
| gate C → strict | **Strict** on a fresh collision-audited bank sized by policy: candidate paced, bare unpaced; pooled ≥ +40 ∧ LB90 > 0; survival band v2; scripted no-death ≥ +.03; boost clause as registered; family composite p = max of clause p-values, Holm .05/m | ≤ C_strict | completed FAIL ⇒ KILL of artifact C; integrity ⇒ INVALID (no verdict); PASS ⇒ SKILL_PASS |
| → +10.67 h + setup | **Product**: 32 fresh paired 10-min Watch6 sessions; death-rate paired 95 % UB < 0; relative mass LB95 ≥ −5 % | ≈ 11 h | FAIL ⇒ RELEASE_HOLD(SKILL_PASS/PRODUCT_FAIL); no product retry |
| → +6 h | **Serving** regressions + rollback matrix (below) | 6 h | FAIL ⇒ RELEASE_HOLD; serving-only repair is a proposal with a scope receipt, not a dispatch |
| **Tue 21:00** | `g seal` → `exp/week1-20261021/handoff/INDEX.json` + `evidence-seal.json` (all receipts, raw inventories, concrete default-change diff, monitoring plan, rollback commands); hashes exclude the seal's own hash and any future signature | — | missing ⇒ NOT_READY(PACKET_INCOMPLETE), RELEASE_HOLD |
| Tue 21:00–23:00 | Codex-V independent review of the immutable seal → `packet/review.json` | 2 h | no evidence production in this window |
| **Tue 23:00** | `g disposition` → `disposition.json`; owner front page with exactly one signature text (§7) | — | all-PASS ⇒ `QUALIFIED_AWAITING_OWNER`, never DEPLOYED |

Every phase rechecks remaining serial hours before starting; a bank is never truncated and a check never omitted to meet the seal.
**Rollback matrix (Y5; exercised before the seal):** assisted mixed Watch6 shows C persistently incl. fallback ticks; Watch8/12, Play,
Train keep B8 scope (Train refused); missing/corrupt controller pin ⇒ assistance refused + verified bare + reason; wrong-life/stale
result, worker crash, reset/reconnect ⇒ old responses cannot execute, fallback accounted; rollback with pending work ⇒ disable hook,
drain/invalidate, select verified B8, late completion cannot reattach (`SNAKE_SERVE_CHECKPOINT=b8-s3` alone is insufficient);
FRP3/champion rollback and custom load ⇒ correct actual identity, no inherited label. Repair of anything action-affecting ⇒ C′: fresh
L3 → L4 → strict → product → serving on fresh banks under a new prospective family; at 03:00 with no judge: write
`repair-disposition.json`, release reservations, leave C′ NOT_READY.

## §6 Stage 3′ — R research track (Wed → Tue; Colab + Mac gaps; no R strict bank this week even if S defers)

Order: Z-9 → **E4 relaunch** (rate receipt; < 1k/s ⇒ E2 shrinks to 4M/arm and says so) → **Z-4′ ruling** → E2 (20 u; pre-nominate F
before any efficacy read) → E6/E7 + E10 (24 u; E10 registered before gameplay) → E12 (16 u; one E19 repair allowed, second failure =
recipe KILL) ∥ E3 recovery-02 → E14 (§3) ∥ Y3 → Z-5. **E16 / E17 / E15 are NOT_READY this week by construction**; the earliest E16
execution is Oct 28 under 06's DRAFT_CONDITIONAL registration (`exp/week1-20261021/R/e16-next-week/registration.md`: A/G only, hash-
bound E9′ approval + pre-nominated E2-F SHA + E8/E12 receipts + retained E3 architecture + matching E14 PASS; 40 u / 20 h incl. 4
reserve; forecast ≤ 8 h; eval ≥ 800 world-steps/s; UB90(G) < 40 = ARTIFACT_REJECTED_AT_BAR, not mechanism KILL).
What each print licenses — and nothing more: E2 GO ⇒ planning only (never pick the better R/F); E6 NEEDS_SCALE ⇒ E10 bridge +
replication *registration*; E7 TRANSFER_ADMISSIBLE ⇒ a separately registered archive treatment (delay-only confirmations are not
rescues); E12 PASS ⇒ critic qualified for that parent only; E17 (week 2) beating persistence ⇒ another short test, never a 100M
probability; `100M_ADMISSIBLE` ⇒ 32M-tranche preparation only. Day-7 R deliverable = disposition packet: Z-4′ ruling, E4 receipt,
E2/E6/E7/E12 prints or blockers, E3 print (GO with `selected_architecture = null` is a blocker for E16), E14 receipt, Y3 receipt,
Z-5 invoice, burn manifest, next-week E16 text. Tuple: `(none | study verdicts, NOT_READY(reasons), NONE)`.

## §7 Day 7 (Tue 23:00) — reducer, signature texts, handoff

**Reducer (Z-6′; total domain).** Per study: validity first (INVALID ⇒ no scientific evidence), then its registered stop/KILL/pass/
INCONCLUSIVE rule; literal registered labels preserved. Per line: `scientific_verdict ∈ {qualified-skill, KILL(scope), INCONCLUSIVE,
none}`, `readiness ∈ {READY, NOT_READY(reason)}`, `release_status ∈ {QUALIFIED_AWAITING_OWNER, RELEASE_HOLD(reason), NONE}`.
S route precedence: K (L2′ PACKAGE_KILL) > I (INCONCLUSIVE) > N (NOT_DISPATCHABLE) > T (eligible timing FAIL) > U (no eligible
measurement) > A (FUND_SYSTEMS + eligible PASS) — all but A deny Friday; A still needs every Friday conjunct. R columns G / H / C / U
(architecture / GO-no-architecture / E3 KILL / INCOMPLETE): G permits E14, never E16; every column retains E9′, E4, parent, E12, Y3 and
funding blockers. R cannot veto qualified S; R cannot rescue unqualified S. Post-admission: scientific STOP ⇒ scoped KILL; mechanical
⇒ NOT_READY; completed strict FAIL ⇒ KILL of C; product/serving FAIL ⇒ RELEASE_HOLD with SKILL_PASS preserved; opened-but-incomplete
family stays OPENED (integrity HOLD), burned worlds retained, excluded from automatic week-2 readmission.

**Signature texts (exactly one on the front page; verbatim; owner answer stored in `owner-authorization.json`).**
A (only on all same-C passes + audit PASS): "I authorize [operator] to deploy Claim B [C], backend/UI [hashes], diff [hash], packet
[seal/audit hashes], mixed Watch6 on this Mac within [W/K/load envelope]. Keep Watch8/12 and Play on B8. Display 'Assisted hero — [ID]'.
I authorize tested rollback [hash] and monitoring response [hash]. This is assisted-controller approval. Monitoring remains incomplete."
B (every other outcome): "I acknowledge packet [hashes] and disposition [exact tuple]. This authorizes no deployment or serving change."
Owner silence ⇒ QUALIFIED_AWAITING_OWNER stands; explicit refusal ⇒ RELEASE_HOLD(OWNER_DECLINED), qualification preserved; a
signature authorizes deployment, `deployment.json` records whether it happened.

**Handoff INDEX (Z-7; sealed Tue 21:00; every entry: absolute path, SHA-256, scope, producer, missing-status).** 1 Identity (C: weights,
planner, source closure, root rule, checkpoints, lead cap, both cutoffs, W/K, device, threads/QoS, recorder, fallback, load envelope,
mechanics, opponents, UI build, deployment diff). 2 Evidence (thresholds, estimates/intervals, registered verdicts, audits, terminal
receipts, three S oracles, typed E13 index with `execution_platform` from the bound manifest — Mac analysis of Colab trajectories is
Colab evidence). 3 Family (FAMILY_UNOPENED requires reconciled launch history, never inferred from missing results; OPENED{members,
p-values, outcome}; per-family .05; campaign union bound min(1, Σα); no campaign-wide 5 % claim). 4 Burn (allocated / launched /
revealed / abandoned / quarantined worlds by namespace, mix, roster). 5 Resources (`resources.json`: proposed / ratified / reserved /
debited; balance; outstanding 2× liabilities; remaining mandatory work; reserve). 6 Decisions (verbatim owner answers for probe,
serving, release — or PENDING; judge rulings separate; each blocker names dependency, builder, next permitted action, cap).

## §8 Week 2 (Oct 28 →) — conditional, registered prospectively, fresh banks

- Family: registered from artifacts with no prior strict look — R nominee only if E16 → E17 → 32M → 100M → E13 complete (not before
  ~Nov 3 at 2 slots); planner-over-R; S if deferred *unopened* (same identity C). An S that opened and failed is closed; one that
  opened and passed is skill-qualified and not retested (product/serving may still hold). A repaired S is C′ with a fresh chain.
- Harvest: admitted only through O2's governing receipt — (D+O − D) LB90 > 0 ∧ +40 / LB90 > 0 vs B8 — into a **new** S identity
  (living-root trigger, rescue/harvest priority, ranking, tie rule) with its own ruler (4 u), parity (4 u), L2′-class evidence
  (≤ 64 u), latency, oracles, L4, strict, product, serving. Tentative 72 u ceiling, not authorized. O1 GO admits nothing.
- E15: 320 u incl. 64 reserve (balanced 4 × 100M = 222 u liability at 2k/s; at 1.5k/s it does not fit — pause at the Y3-qualified
  checkpoint, NOT_READY). E16 40 u; E17 40 u (Y4); E13 18 u (two audits, 768 episodes/platform). A100/H100: none.
- Units are not the constraint (week 1 ≈ 142–202 u; week 2 ≤ ~560 u if everything is admitted; balance ≈ 1,140 u). If a lane is
  slot-limited, the lawful acceleration is two L2′ chunks in parallel, E3 ∥ E1b, and O2 as soon as it is built — never a bigger GPU.

## §9 Operator card — lawful next action at 03:00 with no judge

1. `g <gate>` exists? No ⇒ print `NOT_READY(INSTRUMENT_MISSING)`, block dependents, write the slip file, retain evidence. Yes ⇒ run it;
   never hand-write a receipt; never overwrite one.
2. A job failed mechanically ⇒ record the terminal receipt; **no automatic retry**; prepare a recovery with a new Z-9 receipt; the slot
   is free only after verified release.
3. A gate is missed ⇒ slip file; dependents blocked; nothing else changes; the week's reservation never extends.
4. A number is missing (N, rate, hours, R, S) ⇒ NOT_READY with the missing name; never copy a historical bank size or a Colab timing
   onto the Mac; never shrink N; never move a check into review time.
5. The owner has not answered ⇒ W1/K1, no probe, no release; record PENDING verbatim.
6. L2′ prints PACKAGE_KILL ⇒ S's systems work stops this week (scientific KILL scoped to the unpaced package); R and O2 continue; no
   strict bank opens. L1′ eligible cell FAILs ⇒ S NOT_READY(timing FAIL), reservation released; science preserved.
7. Anything that would let a PASS string, a Colab receipt or an old B8 pin stand in for a same-C Mac receipt ⇒ stop; it is the
   laundering vector every advisor named.

Decision points (PT): **Wed 16:00** Z-9 accepted (else no Colab dispatch) · **Thu 12:00 / 20:00** Stage-0 code / acceptance ·
**Thu 16:00** L1′ latest start (E14 Thursday) · **Thu 22:23 / Fri 10:00** last L2′ chunk start (1 / 2 slots) · **Fri 12:00 / 20:00**
Z-2′, X7 adapter, Y5, sizing policy · **Fri 23:59** Friday admission · **Sat 09:00 + R₃** gate B · **Sun 08:00** integrity + hours
check only (no look) · **Sun 17:00 + R₄** gate C · **Tue 21:00** seal · **Tue 23:00** disposition.
