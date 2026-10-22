# PLAN_v3.md — from the L2′ FUND to a served, strict-gate verdict on this Mac (Thu Oct 22 → Sat Nov 7, 2026, PT)

Written by the round-15 Fable judge, 2026-10-22 ~16:30 PT, from `round-15/JUDGMENT.md`, `LIVE_FACTS.md` (15:50 PT), the L2′ advice memo,
PLAN_v2 and the repo at `48f776b2`. **Replaces PLAN_v2 §2–§6 and §9**; PLAN §6–§8 (R-track, reducer, signature texts, week-2 family rules)
stay in force where not overridden. Authority: owner > registered rules/amendments > this plan > advisor memos. Owner-only: release OK.
Owner rules applied: budgets 2× (every cap below is already doubled from its forecast); never idle (Mac + 3 L4 slots, but never two Mac
jobs at once — a saturated Mac invalidates the evidence it is producing); resume-by-default on mechanical failure (one prepared retry,
verified ledger, charged to the same cap; second failure ⇒ judge); advisors before design (every identity/estimand/environment change
below is ruled here or at a dated decision point before Sol builds); serving on this Mac at W ≤ 4 / K ≤ 8 (SC §4 amendment).

## §0 Verdict in one paragraph

L2′ is sign-certain and size-soft: 28 rescues, zero negative worlds, frozen+mixed +41.9 at 100 % delivery, mixed resting on 7 worlds.
Its served value is zero until a rescue is delivered under the 24-fps clock, and its product value against vetoed FRP3+v8 opponents is
unmeasured. The strict point bar (equal-mix pooled ≥ +40) binds at delivered fraction **r ≥ 0.48** (L2′ variances ⇒ SE ≈ 8 at 128/mix,
so LB90 > 0 is the easier clause). Three cheap measurements (M0, M1, M3) decide by **Sat Oct 24 12:00** whether a served identity (C15)
is built; one development bank (C1) decides by **Mon Oct 26 12:00** whether the product premise survives; the delivery bar (B3, ≥ 60 %
of rescues delivered, every miss charged) decides by **Tue Oct 27 23:00** whether the Mac qualification chain opens. B8 s3 stays served.

## §1 First measurements (frozen specs; receipts are create-only JSON + `.sha256`/`.complete`; Mac work under `exp/mac-lease.json`)

**M0 — Lead census on the 28 L2′ rescues.** Owner: Claude (shard part) + Sol-S (replay, ≈ 3 h) → Codex-V (0.25 d).
- *Part A (0 u, tonight, read-only over `exp/bl-l2/mac-amendment-01/chunk-0*/shard.json`):* per rescue world (`world_ledger.productive_rescue`):
  mix, hero length, death cause, detection frame f = `decision` of the first dense batch whose bare branch has `survives == false`, bare
  frames-to-death D at f (`death_decision` / `frames_survived`), `branches_evaluated` at f and the Q-rank of the surviving alternative,
  whether the batch was all-die (delay-ranked), and the count of delay-only changes before f in that world. Published to
  `exp/bl-c15/lead-census/part-a.json` **before** Part B runs.
- *Part B (≤ 1 unpaced Mac-h, Fri 09:00–10:00):* reconstruct each rescue world's **bare** trajectory (seed + plan from the row), continue
  bare from f, and run exact-stop batches (kernel off — T3 roster has no opponent veto; `search_mode=exact_stop`) every 5th decision until
  bare death; a = last decision with a surviving alternative (D300 LAD, as `l0.py`). Attest counters; chosen actions never applied.
- *Outputs:* per rescue (f, D, a, a−f); the curve F(L) = fraction of **frozen+mixed** rescues (18) with a ≥ f + L + 2 for
  L ∈ {48, 72, 96, 144, 192, 240}; the same for scripted; rank/all-die/delay-precursor tables.
- *Lead rule (frozen):* L* = 5·⌈(24·(p99_doom,W4 + 0.25 s) + 2)/5⌉ frames, p99 from M1's frozen-800 W4 cell; floor 72, cap 432.
- *Close rule (frozen):* **F(L*) < 0.50 ⇒ served exact search on this Mac is CLOSED (KEEP_B8 for serving), nothing built.** F(L*) ≥ 0.50
  ⇒ C15 admitted to build. If > ⅓ of frozen+mixed rescues were all-die/delay-ranked at f, C15 must carry the all-die cost in M1's budget.
  Fewer than 12 frozen+mixed rescues reconstructed ⇒ INCOMPLETE (not CLOSED); fix and rerun once.

**M1 — Native exact-stop workers co-resident with the live Watch6 game.** Owner: Sol-K (0.5 d: `l1.py` constructs
`create_stop_operator()`; kernel-row attestation; cold-process cell) → Codex-V (0.25 d) → Claude runs Fri 19:00–21:00 (≤ 2 leased Mac-h, cap 4).
- *Cells (frozen):* W ∈ {1, 2, 4} × K = W, plus one W4/K8 saturated wave; nine Y1 fixtures (frozen-800 is the doom cell), 30 submissions per
  fixture per cell, interleaved; the real mixed Watch6 game (B8 s3 vs 5 × FRP3 s12 + v8, continuous respawn) ticking at absolute 24 fps
  throughout; one cold-process start per cell reported separately; one declared contention block (single 50–100 % core load); no daemon
  killing; AC + lid open; host telemetry sampled off-tick; thermal state logged.
- *Bars (frozen):* zero tick overruns in every cell; Watch6 update p95 ≤ 11.688 ms; chosen actions identical to the all-branch reference;
  `kernel_rows > 0 ∧ stock_rows == 0` on frozen/mixed branch rows. Report S (admission → coordinator visibility) p50/p95/p99 per cell and
  fixture; the dense caps (0.830 / 1.250 / 1.458 s) reported, not gating.
- *Kill rule:* any overrun, p95 > 11.688 ms, or doom-fixture S p99 > 5 s at W4 ⇒ C15 build BLOCKED pending a judged remedy (one remedy
  attempt; second failure ⇒ KEEP_B8 for serving). Missing counters ⇒ INVALID; cap ⇒ INCOMPLETE. PASS admits construction only.

**M3 — Receipt boundary.** Owner: Sol (1 d) → Codex-V (0.5 d), Fri–Sat, no Mac, 0 u. Typed evidence loader that opens the journals and
evidence files, recomputes hashes/clauses and refuses field-only summaries; Codex-V's five negative controls (missing files, copied B8
strict pins, Colab-as-Mac evidence, changed scheduler, synthetic clock proof) must all be refused. **No L1″ is admitted until M3 PASSes.**

**C1 — Served-roster development bank (product premise).** Owner: Sol-S (1 d, Thu–Fri) → Codex-V (0.5 d); Claude runs Sat 12:00 → Sun
(≤ 8 unpaced Mac-h, cap 16, W2 one thread each, lease, nice 10).
- *Design (frozen):* 24 fresh paired 10-min (14,400-frame) mixed-Watch6 sessions, fresh seeds (namespace disjoint from every registered
  bank), continuous respawn, hero bare B8 s3 vs the **unchanged** dense rule (`DenseTrigger(600,5)`, `_choose`, exact-stop, all legal
  actions, exact reacting FRP3 s12 + v8 opponents with independent branch RNG/carry, **native kernel on and attested**). Development only.
- *Endpoints:* paired hero deaths per 1,000 actual hero decisions (own denominator per arm), full-horizon hero mass relative difference,
  distinct rescue-witness lives, same-life re-deaths within 300/1,000 frames, newly caused deaths, delivered boost share, length quantiles.
- *Rules (frozen, one look):* GO ⇒ death-rate paired one-sided UB90 < 0 ∧ relative-mass LB90 ≥ −5 % ∧ witnesses in ≥ 4 sessions.
  STOP (product premise closed ⇒ KEEP_B8 for serving) ⇒ death-rate LB90 ≥ 0 with ≥ 8 baseline deaths, or relative-mass UB90 < −5 %.
  < 8 baseline deaths ⇒ INSUFFICIENT (no extension inside the run; a 24-session extension may be registered once). Else INCONCLUSIVE ⇒
  the chain continues but the Nov 7 claim is withdrawn until L3's six sessions print.

**B2 — Natural-root library (L4, slot C).** Owner: Sol-S (0.5 d) → Codex-V (0.25 d); canary ≤ 1 u then ≤ 20 u (cap 40). 32 fresh
continuous-respawn Watch6 trajectories (FRP3+v8, kernel on), ≤ 4 hash-selected L ≥ 800 roots each, all legal H300 alternatives
+ H1440 continuations of predicted escapes; death causes; full state snapshots. Consumer: C15's fixture bars (≥ 4 distinct frozen/mixed
doom fixtures) and L1″. < 8 rescue-positive lives ⇒ INSUFFICIENT, never evidence of no value.

## §2 The identity to freeze — C15 (built only on D1 GO)

Bare B8 s3 bytes; exact bare shadow from deployment-available state; roots on the 5-decision grid at L ≥ 600, target r = now + L*
(reachable: r ≤ a is what M0 priced); at each root **exactly L2′'s local rule** — bare first; if bare survives H300 stop; else all legal
alternatives in Q/index order until the first H300 survivor; all-die ⇒ `_choose` delay ranking — resumable H32→96→192→300 (same
simulator/opponent/RNG/carry state, no restart); W4/K4 CPU, one math thread per worker, shadow construction inside the worker budget;
EDF by publication cutoff; publish < T(r−2), consume < T(r+1), equality late; validation of session/life/frame/checkpoint/epoch/root
digest; intervention/respawn/reset fences every outstanding descendant; miss, stale, saturated, invalid ⇒ bare; recorder, load
envelope, QoS/nice, warm-up, cold-start policy and fallback accounting bound into digest C. Visible label "Assisted hero — C" on
every tick including fallbacks; proposed vs executed logged; rollback = disable assistance, drain, pin B8 s3. **Any change ⇒ C′, no
inherited evidence.** Certificate-only / top-two (WP-4 as registered) is C′, admitted only by a registered amendment if M0/M1 show the
all-die cost cannot fit. No learned component enters C this campaign.

Fixture bars before L1″ (unchanged from PLAN_v2 §3, now with B2 roots): first valid H300 result ≤ 5 s **and before its own publication
cutoff** on ≥ 50 % of dense-rescuable frozen/mixed opportunities across ≥ 4 distinct natural fixtures; zero overruns; update p95 ≤
11.688 ms; actions identical to the reference or INVALID. **B3 delivery bar (frozen):** paced replay (W4/K4, real clock, every miss
charged) of the 28 L2′ rescue worlds + every C1 rescue life from a restored state ≥ 500 frames before f: **delivered rescues ≥ 60 %**
(≥ 11 of 18 frozen/mixed) ⇒ L1″ admitted; 40–60 % ⇒ one judged remedy then re-run once; < 40 % ⇒ KEEP_B8 for serving.

## §3 Build order and owners (parallel where resources differ; one writer + one reviewer each)

| # | Item | Owner → reviewer | Builder time | Done by (PT) | Gate it feeds |
|---|---|---|---|---|---|
| 1 | M0 Part A script; Part B replay harness | Claude; Sol-S → Codex-V | 0 + 3 h | Thu 23:00 / Fri 09:00 | D1 |
| 2 | M1 wiring (`create_stop_operator()` in L1, counters, cold cell) | Sol-K → Codex-V | 0.5 d | Fri 17:00 | D1 |
| 3 | C1 bank runner (Watch mirror + dense adapter + kernel attestation) | Sol-S → Codex-V | 1 d | Fri 23:00 | D2 |
| 4 | M3 typed evidence loader + negative controls | Sol → Codex-V | 1 + 0.5 d | Sat 18:00 | L1″ admission |
| 5 | B2 library manifest + canary | Sol-S → Codex-V | 0.5 d | Sat 12:00 | C15 fixtures |
| 6 | **C15 controller: production root provider + continuation seam + journal** (D1 GO only) | Sol-C → Codex-V | 2–3 d | Mon Oct 26 20:00 | fixtures, L1″ |
| 7 | Assisted-hero label/recorder/clip exporter (10's spec) | Sol-U → Codex-V | 0.5 d | Mon Oct 26 | serving qual. |
| 8 | B3 paced replay harness (restore ≥ 500 frames pre-f, real clock, charge misses) | Sol-S → Codex-V | 1 d | Mon Oct 26 | D3 |
| 9 | Strict sizing sheet (paced variances from L4; floor 128/mix; alternative +54; nulls) | Claude → Fable judge | 0.5 d | Sat Oct 31 | D4 |
| 10 | Packet validator / independent review tooling (reuses M3) | Codex-I → Codex-V | 1 d | Nov 2 | D5/D6 |

Shared-file rule: Sol-S alone edits `bl_served_search/*`; Sol-C alone edits `bl_paced_identity/*`; Sol-K edits `bl_mac_latency/l1.py`
and `bl_k1/*`; Sol-R owns `bl_ppo/*`; Sol-U owns `web/backend` label/recorder only. Slips ⇒ `stage0/slips/<item>.json`, block dependents only.

## §4 Qualification chain after D3 (serial, same digest C, this Mac)

L1″ latency (3 h; production provider, natural roots from B2, W4/K4, cold + contention; bars = SC §4 + fixture bars; independent audit)
→ Z-2′ clock-seam acceptance on real transport (inside L1″) → L3 paced smoke (9 h: 16 paired worlds/mix + 6 paired 10-min sessions)
→ all-trace replay → L4 paced development efficacy (32 h: 32 fresh worlds/mix, candidate and bare each twice at absolute deadlines)
→ **D4** → strict sizing frozen → family {Claim B} frozen before the first (comparator-only) strict episode → Mac strict on a fresh
collision-audited bank (candidate paced, bare unpaced under its clock-independence receipt; ≈ 22 h candidate at 128/mix + controls/audit
≈ 28–30 h) → product (32 paired 10-min Watch6 sessions ≈ 11 h; deaths/1k paired 95 % interval excluding zero, target ≥ 20 % fewer;
relative mass LB95 ≥ −5 %) → serving regressions + rollback matrix (6 h) → sealed packet + independent review (2 h) → owner signature
text A. Bars unchanged: pooled ≥ +40 ∧ LB90 > 0, survival band v2, scripted no-death ≥ +0.03, boost [0.5 %, 10 %], zero overruns,
update p95 ≤ 11.688 ms. Likeliest NOT_READY prints, in order: production provider/replay qualification (L1″), unpriced strict invoice
(X5 + L4 variance), B2 fixture coverage, host disruption during a 30-h paced run. ≈ 85 serial Mac-h + ≈ 15 h controls/audit/recovery.

## §5 Day by day — what the Mac and each Colab slot do (every slot has a consumer or is released; nothing paced overlaps anything)

| Day (PT) | Mac (exclusive lease, serial) | Slot A | Slot B | Slot C | Builders |
|---|---|---|---|---|---|
| Thu 22 | 20:00–22:00 **X5** bare strict timing (2 h) | P1 Y3 source-latebody-05 canary → Y3 | DENSITY-2 completion-03 | released (no consumer) | M0-A (Claude); Sol-S M0-B + C1; Sol-K M1 |
| Fri 23 | 09:00 **M0-B** (≤ 1 h); 19:00–21:00 **M1** (≤ 2 h); gap: V0 C2/C3 filler if ready | Y3 → 1M qualification | DENSITY-2 → export verification | B2 canary (≤ 1 u) | Sol M3; Codex-V reviews M0/M1/C1 |
| Sat 24 | **12:00 D1 ruling**; 12:00 → **C1** bank (≤ 8 h) | 1M → **P1 100M tranche** (≤ 120 u) | B2 library shards (≤ 20 u) | B2 shards | Sol-C C15 (if GO); Sol-U label |
| Sun 25 | C1 finishes; one look; judge reads | P1 100M | B2 → export/hash verify | released or P1 seed 2 only if judge funds | Sol-C C15; Sol-S B3 harness |
| Mon 26 | **12:00 D2**; C15 fixture measurement (2 h) on B2 roots; cold cells | P1 100M | released | released | Sol-C finishes C15; Codex-V review |
| Tue 27 | **B3 paced replay** (≤ 3 h); **23:00 D3** | P1 100M → endpoint | V1 labels only if D3 GO and V0 ran (≤ 40 u) | released | packet tooling |
| Wed 28 | **L1″** (3 h) + audit; **L3** (9 h) overnight | P1 eval 128/mix (≤ 10 u) | V1 fit (if admitted) | — | Sol-S/Claude L4 prep |
| Thu 29 – Sat 31 | **L4 paced dev** (32 h) → audit; **Sat 31 D4** | P1 print Mon | released | released | sizing sheet |
| Sun Nov 1 – Tue 3 | **Strict** (≈ 30 h incl. controls); **Tue 3 D5** | P1 strict only if at bar (separate chain) | — | — | packet |
| Wed 4 – Fri 6 | product (11 h) → serving/rollback (6 h) → packet + review | — | — | — | independent review |
| Sat 7 | **D6 owner** | — | — | — | — |

Colab units: P1 ≈ 60–90 (cap 142 + Y3 24); B2 ≤ 40; DENSITY-2 ≤ 5; V1 ≤ 40 conditional; ≈ 130–180 u this fortnight vs the 39 u/day goal
— the goal is not met by this plan and should not be: nothing on the served chain is GPU-bound (A100/H100/RunPod: none). Owner reports
at 09:00 and 21:00 PT: receipt-backed state, C's status, next decision, remaining serial Mac-hours, slot debit, calendar slack.

## §6 Decision points (judge rules; pre-registered rules; no bar moves)

| When (PT) | Decision | Rule |
|---|---|---|
| Thu 22 23:00 | D0 | M0-A table published; X5 running; LIVE_FACTS updated (+ "stop quoting +83") |
| **Sat 24 12:00** | **D1 build C15?** | M0 F(L*) ≥ 0.50 ∧ M1 PASS ⇒ GO (L* fixed by formula). M0 CLOSED ⇒ KEEP_B8 (serving). M1 FAIL ⇒ one judged remedy by Mon 12:00, else KEEP_B8 (serving). INCOMPLETE ⇒ rerun once, D1 slips ≤ 24 h |
| **Mon 26 12:00** | **D2 product premise** | C1 GO / INCONCLUSIVE ⇒ continue. STOP ⇒ KEEP_B8 (serving). INSUFFICIENT ⇒ one registered 24-session extension (Mac, ≤ 8 h) |
| **Tue 27 23:00** | **D3 open the Mac chain?** | C15 fixture bars PASS ∧ B3 ≥ 60 % ∧ M3 PASS ⇒ L1″ admitted. B3 40–60 % ⇒ one remedy + re-run by Thu 29. < 40 % or fixtures FAIL ⇒ KEEP_B8 (serving) |
| Sat 31 | D4 strict family | L4: pooled point ≥ 40 ∧ LB90 > 0 ∧ frozen & mixed points > 0 ∧ no per-mix UB90 < 0 ∧ guards ∧ timing ⇒ size + freeze strict. UB90 < 40 ⇒ candidate closed (KEEP_B8). Else NOT_READY, no extension |
| Tue Nov 3 | D5 strict verdict | PASS ⇒ product/serving. FAIL ⇒ KEEP_B8 (search family, serving). Chunk/host fault ⇒ INCONCLUSIVE, no partial look |
| Sat Nov 7 | D6 owner | Only with a sealed all-PASS packet; otherwise text B. If nothing is qualified by **Tue Nov 10 23:59**, the campaign closes to ≤ 1 research slot unless the owner re-charters |

**KEEP_B8 (re-founded; supersedes PLAN_v2 §6).** Declared for *serving* on the earliest of: (1) M0 CLOSED; (2) M1 FAIL unremedied by
Mon 26 12:00; (3) C1 STOP; (4) B3 < 40 % (or < 60 % after one remedy); (5) L4 UB90 < 40; (6) strict FAIL; (7) **Tue Nov 10 23:59 with no
sealed packet.** Under KEEP_B8: B8 s3 unchanged; C15 and every search-serving lane closed; Colab ≤ 1 slot (P1 chain only if its eval
printed point ≥ 20 ∧ LB90 > 0); the owner is told in one sentence and offered exactly one choice (PLAN §7). Science is preserved: L2′
stands as the offline ceiling of the search family; reopening serving needs a new charter.

## §7 What to STOP now, what to KEEP, and the honest odds

**Stop:** all L4/Colab L2′ follow-ups (native-L4 ruler retries, mode-B forecasts, replication bank); Round 0 re-registration and Round
0-D; O2 (parked → closed this campaign; reopens only on a C1 harvest signal); Claim A descendants; E1b; E14/E16; V1 before D3; WP-4 as
registered (superseded by C15); any learned component in C; A100/H100; RunPod; quoting "+83".
**Keep:** B8 s3 served with no label change; all bars; P1 on slot A under its own registration; DENSITY-2 completion; V0 as Mac-gap
filler only; the round-12 process rules (canary, premise register, attestation, lease, pre-claim ledger check).

**Odds (judgment, not operating characteristics):** P(D1 GO) ≈ .55; P(C1 not STOP | D1 GO) ≈ .7; P(B3 ≥ 60 % | both) ≈ .45; P(L4 at bar |
delivered) ≈ .5; P(strict ∧ product ∧ serving ∧ review clear mechanically by Nov 7) ≈ .5 ⇒ **≈ 4–5 % qualified, owner-approved release
by Nov 7; ≈ 10 % by Nov 14; ≈ 70 % that the served-search lever has a clean verdict (admitted or KEEP_B8) by Nov 7.** The calendar has
zero slack; every slip moves the owner's date one-for-one and is reported the same day.

## §8 Operator card

1. Mac: idle receipt + lease before every job; AC + lid open; log foreign load; never kill daemons; never two Mac jobs; stopped attempts retained.
2. Colab: `colab usage` + terminal receipt + verified release + real-transport canary before any new manifest family; release a slot with no consumer.
3. A missing number ⇒ NOT_READY with its name; never transfer L4 timing to the Mac or vice versa; never multiply a hook speed-up into a serving claim.
4. One look per bank; no partial analysis; no N shrink; no retries of bank chunks; mechanical retry = one prepared recovery inside the cap.
5. A PASS string, a Colab receipt, a synthetic clock proof or a B8 pin never stands in for a same-digest Mac receipt (M3 enforces).
6. Owner is asked exactly two things: nothing until D6 (release), and the one §7 choice if KEEP_B8 fires.
