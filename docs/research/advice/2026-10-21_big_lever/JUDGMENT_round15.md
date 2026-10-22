# Round 15 judgment — from the L2′ FUND to a served, strict-gate verdict on this Mac

Judge: Fable (round-15 judge), 2026-10-22 **~15:40–16:30 PT** (`date` = 15:37 PDT at start). Inputs: `../CHARTER.md`, `../PLAN.md`,
`../PLAN_v2.md`, `../round-11..14/JUDGMENT.md` (in full; rounds 1–10 via PLAN.md), `../LIVE_FACTS.md` (newest 15:50 PT, authoritative),
`../QUEUE.md`, `QUESTION.md`, `personas.txt`, `astra_01..10.md` (all read in full), the separate Fable advice
`docs/research/advice/2026-10-22_L2prime_fable_advice.md`, `docs/research/results/L2prime_mac_analysis_20261022.json`, the 160 L2′ rows
in `exp/bl-l2/mac-amendment-01/chunk-0*/shard.json` (my own recount), `exp/bl-l2/y1-native-mac/ruler-receipt.json`, `exp/bl-k1/timing.json`.
Repo read-only at `48f776b2` (main HEAD; advisors' snapshot `b416719e` differs only by the advice commit). Owner override for this
round: **no round-16 inputs**; products are this file and `../PLAN_v3.md`. Sealed study not opened. No jobs, Colab or tmux.

## 0. Facts verified tonight (newer or sharper than the question / LIVE_FACTS / the advice)

- **The "0-u lead census from existing shards" is only half available.** The L2′ baseline arm ran **no search**: every row has
  `baseline.batches = 0`, `branches_evaluated = 0` (recount of 160 rows). The dense arm's `batch_diagnostics` (per batch: `decision`,
  `bare`, `evaluation_order`, per-action `survives` / `frames_survived` / `death_decision`, `branches_evaluated`) give, for each of the
  **28 productive rescues (13/5/10, recount matches the analysis)**, the detection frame f (first batch whose bare branch dies within
  H300), the bare frames-to-death D ≤ 300 at f, the rank of the surviving alternative in Q-order, and any delay-only changes preceding
  the rescue — all at 0 u. The **last avoidable decision a** (the quantity the lead actually needs) is *not* in the shards: the dense arm
  intervened at f, so the bare trajectory after f was never searched. Getting a needs a replay of the 28 bare trajectories from f with
  exact-stop batches every 5th decision until no alternative survives — ≤ 28 × ~60 batches ≈ **≤ 1 unpaced Mac-hour**, Sol ≈ 3 h on
  the existing `l0.py` / `ceiling` harness. The advice's close rule is kept; its cost line is corrected (PLAN_v3 M0).
- **Zero-inflation recount (per-world `paired_delta.mass_integral`):** frozen 47/64 exactly 0, 0 negative, top-5 = 61 % of the gain,
  drop-top-3 mean +34.5; mixed 57/64 zeros, top-5 = 98 %, drop-top-3 mean **+9.1**; scripted 15/32 zeros, top-5 = 52 %. The advice's
  numbers are exact. Paired SDs 131.6 / 95.7 / 218.2 ⇒ equal-mix pooled SE at 128/mix ≈ **8.0** (04's table), not the T23-based 14.7 in
  PLAN §0 — so for this mechanism the **point bar (≥ +40), not LB90, binds**: delivered fraction r ≥ 0.48 for the point; ≥ 0.57 to pass
  it with ≈ 80 % probability. Scripted counts in SC §2 (equal-mix strict family), not in SC §3 (Watch6 has no scripted) — both must pass.
- **Native ruler exact-stop (W1, kernel, FRP3+v8 opponents, quiet lease):** frozen-800 (the only doom fixture) warm median 1.067 s /
  "p99" 1.269 s with **2 branches**; frozen-1500 0.684 / **0.844 s** (above the W2 cap 0.830); mixed-1200 0.517 / 0.622 s; all-branches
  frozen-800 1.81 / 2.19 s (4 branches). `_distribution()` makes "warm p99" the **maximum of 29 samples** (`native_ruler.py:177-182`) —
  02 right. Nothing in this receipt is W > 1, co-resident with a game, cold-process, or a natural root.
- **The L1 harness still times the wrong path**: `research/bl_mac_latency/l1.py:302` constructs `ExactT3Operator()` whose defaults are
  `native_mode="stock", search_mode="all"` (`ruler_operator.py:115-121`); `create_stop_operator()` (`:239-241`) exists and is unused
  there. 03/08/09/10 right. Any contention measurement must switch the constructor (a ≤ 0.5-day wiring ticket) and attest kernel rows.
- **This Mac is 6 "Super" + 12 "Performance" cores** (`sysctl hw.perflevel0/1`: Super 6, Performance 12), AC attached, lid open, load
  1.5 at 15:37. 02's architecture correction stands; worker placement/QoS is unmeasured.
- **Two validators accept field-only summaries** (07's negative controls, confirmed by reading): `clock_seam_passed()` checks schema,
  verdict string, a caller-supplied digest and four `scheduler_verdict == "PASS"` strings — no journal (`clock_seam.py:16-35`);
  `strict_pin_problems()` checks presence/format of paths and 64-hex strings and `_resolve_pinned()` hashes only the checkpoint bytes
  (`served_checkpoint.py:172-217, 262-283`). Neither opens evidence files. This is a laundering boundary, not a breach.
- **Paced-identity code state**: the controller counts queued + running + visible-unconsumed against K (`controller.py:47-75`; 01's
  K/residence arithmetic is right); `publish()` records late results as visible and only `poll/consume` rejects them
  (`:118-138, 216-220, 271-281`); cutoffs are `T(r−2)` publish / `T(r+1)` consume on a 5-frame grid with a 432 cap
  (`adapter.py:16-18, 194-200, 267-268`); the adapter "does not create roots, call a planner, or qualify a candidate" (`adapter.py:1-5`).
  X7 preflight-03 `semantic.json`/`replay.json` both PASS (harness evidence only). The lease helper exists (`exp/mac-lease.json.live/.lock`).
- **Exact-stop semantics** (`exact_stop.py:45-57`): bare first; alternatives in Q/index order until the first H300 survivor; **all-die ⇒
  every branch evaluated and `_choose` ranks by delay**. L2′'s 154 changes = 74 horizon-surviving + **80 delay-only** (26/7/47). WP-4 as
  registered ("bare + two alternatives, delay-only rejected") is therefore a double ablation of the measured rule — 06 right.
- Lane state: V0 smoke-01 INVALID (K-1 audit PID active), prepared-02 unrun; X5 still NOT_READY; P1 `source-latebody-05` repinned to the
  requalified transport, not dispatched; DENSITY-2 completion-03 prepared; O2 parked. Mac otherwise quiet.
- Advisor arithmetic checked: 04's SDs and SE 14.534 (JSON) ✓, r = 48.2 % ✓; 03/08 strict candidate time 384×5000/24/3600 = 22.2 h ✓,
  08's 85.2 h chain ✓ and 1,100 u / (111–144 u/day) = 7.6–9.9 days ✓; 02's 0.99^29 ≈ 75 % ✓, utilisation 4.8×0.684 = 3.28 worker-s/s ✓;
  06's 74/80 split ✓; 10's "one changed mixed action per ≈ 10 min" ✓ as an illustration. **Factual errors that change a conclusion:
  none among the ten.** Shared blind spot: none opened the baseline arm's `batches = 0`, so all ten (and the advice) priced the lead
  census as shard-only.

## 1. Bottom line (for the owner)

1. **L2′ is a real, sign-certain offline win and a soft, scripted-carried number.** Frozen+mixed average +41.9 at 100 % delivery; mixed
   is seven worlds; the mechanism is 28 rescues, nothing else. Its served value today is **zero** until a rescue is delivered under the
   24-fps clock. Stop quoting +83; quote "frozen/mixed +28..+56 × delivered fraction".
2. **The lever is still alive, and it is cheap to kill or admit.** Three measurements, none over 2 Mac-hours or 0 u, decide by **Sat
   Oct 24 12:00 PT** whether anything gets built: (M0) the lead census on the 28 rescues (shard part tonight at 0 u; last-avoidable-
   decision replay ≤ 1 Mac-h Fri AM); (M1) native exact-stop workers W1–4/K≤8 **co-resident with the live Watch6 game** on this Mac
   (≤ 2 leased Mac-h Fri PM; the first timing with the kernel actually on); (M3) closing the two field-only validators (Sol/Codex, no Mac).
3. **The served identity I freeze (C15, PLAN_v3 §2): sparse future roots + L2′'s complete local rule.** Bare B8 s3; exact bare shadow;
   roots on the 5-decision grid at L ≥ 600 with lead L* fixed pre-data from the census curve and the M1 doom-root p99; at each root the
   **same** exact-stop / `_choose` semantics L2′ measured (all legal alternatives in Q order, delay ranking on all-die), resumable
   H32→96→192→300; W4/K4 CPU, one math thread per worker; publish < T(r−2), consume < T(r+1), miss ⇒ bare. Not WP-4's "top-two,
   certificate-only": that ablates two things at once and inherits nothing from L2′ either. A certificate-only C′ is a registered
   fallback only if the clock forces it.
4. **The product game is a separate, unmeasured premise.** Watch6 = 5 × FRP3 s12 **with v8**, continuous respawn, deaths/1k. No exact-
   search value against that roster exists. One development bank (C1: 24 fresh paired 10-min Watch6 sessions, bare vs unchanged dense
   exact-stop, native kernel on, **on this Mac Sat–Sun, ≈ 6–8 unpaced Mac-h**) measures it while Sol builds C15. It is dispositive in the
   negative direction (confident non-benefit ⇒ KEEP_B8 for serving) and merely admitting in the positive one.
5. **Delivery is the whole risk.** Break-even r ≈ 0.48 for the strict point bar. The delivery bar (paced replay of the 28 rescue worlds
   + C1 rescue lives, W ≤ 4 / K ≤ 8, every miss charged): **≥ 60 % of rescues delivered**, pre-registered now. Below it the served lever
   is KEEP_B8; no bar moves.
6. **Calendar and odds (honest):** C15 earliest frozen Tue Oct 27; L1″ Wed 28; L3 Wed night; L4 paced dev Thu–Sat (32 h) → D4 Sat
   Oct 31; strict Nov 1–3 (≈ 25–30 Mac-h incl. bare controls, sized pre-data, never shrunk); product Nov 4 (11 h); serving/rollback Nov 5;
   packet/independent review Nov 6; owner Nov 7 — **zero slack**. P(owner-approved, strict-qualified served release by Nov 7) ≈ **5 %**;
   by Nov 14 ≈ **10 %**; P(a clean scientific verdict on the served-search lever — admitted or KEEP_B8 — by Nov 7) ≈ **70 %**.
7. **Stop now:** all L4/Colab L2′ follow-ups (native-L4 ruler retries, replication bank), Round 0 re-registration and Round 0-D, O2 (parked →
   closed for this campaign), Claim A descendants, E1b, E14/E16, learned-tail V1 before D3, A100/H100, RunPod. **Keep:** B8 s3 served,
   bars unchanged, P1 on its own slot, DENSITY-2 completion, X5 tonight.
8. **KEEP_B8 is re-founded** (PLAN_v2 §6 clause 2 can no longer fire: K-1 passed and L2′ printed FUND) — six dated triggers in PLAN_v3 §6.

## 2. Per-advisor verdicts

| ID | Persona | Verdict | Why (verified facts in **bold**) |
|---|---|---|---|
| 01 | Real-time search-serving | **keep** | **K counts queued + running + unconsumed (`controller.py:47-75`)** ⇒ delivered throughput ≤ K/residence; "a completed batch is never a rescue"; the adapter supplies no roots; EDF by publication cutoff with deterministic ties. Its instrument-first package *is* C15's provider — built after D1, not before the two 0-u/2-h falsifiers. |
| 02 | Apple-silicon perf | **keep — the round's engineering catches** | **warm p99 = max of 29 (`native_ruler.py:177-182`)**; **frozen-1500 0.844 > W2 cap 0.830**; W4 doom-root headroom 14.9 %; **6 Super + 12 Performance cores (sysctl)**; worker timer starts after imports. Co-residency falsifier adopted as M1 with its cold/contention cells. |
| 03 | Release-qualification | **keep** | **`l1.py:302` builds the stock/all operator**; L1 exports zero valid consumptions by construction; X7 is not the provider; strict candidate 22.2 h; "NOT_READY must name the missing receipt". Chain and NOT_READY ranking adopted (PLAN_v3 §4). |
| 04 | Evaluation statistician | **keep** | **SDs 131.6/95.7/218.2, SE 14.53, r = 48.2 %**; point bar binds, N cannot fix a boundary; score the actual trajectory, never credit hypothetical rescues; family freeze incl. comparator-only episodes. 160/mix is deferred to the sizing step after paced variance exists (floor 128, never shrunk). |
| 05 | Product-transfer skeptic | **partly** | Right that no FRP3+v8 value exists and that terminal-H5000 mass does not convert to respawn deaths/1k; its 16-session screen is C1's shape. Overruled on *platform* (the Mac is idle Sat–Mon and 7–10× faster per worker than L4; L4 needs a canary and has failed 10+ times) and on *order* (M0/M1 are cheaper and can close the lever outright). |
| 06 | Search researcher | **keep — the round's decisive design catch** | **74 horizon-surviving + 80 delay-only changes; all-die evaluates everything (`exact_stop.py:45-57`)**; WP-4 = double ablation; reachability r ≤ a with visibility < T(r−2). C15 adopts its full-T3 local rule; its slot-B ablation study is folded into the lead census (rank of survivor, delay-only precursors) at 0 u. |
| 07 | Integrity reviewer | **keep — the round's integrity catch** | **`clock_seam_passed()` and `strict_pin_problems()` accept field-only summaries (code read)**; `publish()` records late results; `_commit()` journals before the game latches. M3 adopted: typed evidence loader + Codex negative controls, gating L1″ admission. |
| 08 | Operator / calendar | **keep** | **85.2 h chain → ≈ 100 h reservation; 12 h/day Oct 27–Nov 7 = 144 h; 3 slots burn 111–144 u/day**; 09:00/21:00 owner reports; "never-idle ≠ saturate the Mac". Day table adopted (PLAN_v3 §5). |
| 09 | Red team | **keep (standing falsifier)** | Deadlines preferentially drop the expensive doom roots; respawn removes the absorbing-zero counterfactual; its 32-pair paced Watch6 test is the SC §3 product stage and the only thing that changes its view. Adopted as the delivery bar's rationale and the product stage. |
| 10 | Spectator | **keep** | Persistent "Assisted hero — C" label incl. fallback ticks; normal-speed clips with ≥ 30 s after; "surviving H300 is not surviving the late game"; recorder/label built before C freezes (0.5 Sol-d) — adopted into B1. |

Advice memo (`2026-10-22_L2prime_fable_advice.md`): **keep with one correction** — the lead census needs a ≤ 1-Mac-h replay for the
last avoidable decision (baseline arm has no search); its zero-inflation, scripted-carried, +41.9, break-even and stop list are exact
and adopted; its 10 % by Nov 7 is revised to ≈ 5 % (zero calendar slack) / ≈ 10 % by Nov 14.

## 3. Disagreements ruled

1. **First measurement — lead census (advice) vs co-residency falsifier (02/03/04/06/08/09/10) vs provider instrument (01) vs product
   screen (05) vs receipt boundary (07).** All three cheap ones run before Sat 12:00 on disjoint resources: M0 (shards tonight, Mac ≤ 1 h
   Fri AM), M1 (Mac ≤ 2 h Fri PM), M3 (builders, no Mac). 05's screen runs Sat–Sun on the Mac while C15 is built. 01's instrument is C15.
2. **Identity — WP-4 top-two/certificate-only (01, 04, 07, 08, 09, 10) vs full-T3 local rule at sparse roots (06).** 06. The served
   identity inherits no L2′ *efficacy* either way; keeping the measured rule removes one unexplained ablation and lets the census say
   what the clock costs. Certificate-only becomes C′ only by a registered amendment if M1/M0 show the all-die cost does not fit.
3. **Lead = 144 frames fixed vs chosen from data.** Fixed **by formula, pre-data**: L* = 5·⌈(24·(p99_doom,W4 + 0.25 s) + 2)/5⌉ frames
   from M1's doom-fixture receipt, floor 72, cap 432; the census publishes the reachable-fraction curve at L ∈ {48, 72, 96, 144, 192, 240}
   and the GO rule reads it at L*. No lead shopping after the curve is seen (the formula and floors are frozen here).
4. **C1 on L4 (05) vs on the Mac.** Mac (unpaced, leased, Sat 12:00 → Sun). Colab slots are not idle: P1, DENSITY-2 completion, and the
   FRP3+v8 natural-root library (B2) that C15's fixture bars need (Y1 has one doom fixture; "≥ 4 distinct fixtures" cannot be met otherwise).
5. **Strict N: 160/mix (04) vs 128 (PLAN).** Sized once, pre-data, after L4 paced variance exists; floor 128/mix; never shrunk; the
   planning alternative is pooled +54 (r ≈ 0.65), not +40.
6. **Odds 5–10 % (05, 09) vs 10–20 % (02, 04).** 5 % by Nov 7 (no slack), ≈ 10 % by Nov 14. Nothing in L2′ raised delivery odds.

## 4. What carries over lawfully, and what does not

| Evidence | Carries as | Never carries as |
|---|---|---|
| L2′ FUND (160 rows, one look) | mechanism exists on the T3 roster family; per-world variances for sizing; the 28 rescue worlds as a **development** replay set (labelled, never strict) | served efficacy; product (SC §3) value; any paced number; a strict-bank world |
| K-1 (20.13×, 294 identity cases) | component identity of the kernel on audited rows; speed of the hook | W > 1 behaviour; a serving timing receipt (`serving_or_release: NOT_GRANTED`) |
| Native ruler (270/270, W1 fixture timings) | exact-stop decision identity on fixtures; **pricing input** for L* and forecasts | co-resident latency; natural-root cost; cold-process cost; any PASS for C |
| X7 semantic/replay PASS, synthetic clock proof | harness correctness | production clock-seam acceptance (`l3_identity_prerequisites` unmet) |
| B8 s3 STRICT_PASS pins | nothing for Claim B (different identity) | — |

## 5. Running shortlist (carried from round 14, updated)

| # | Lever | Conf. push / beats B8 at bar | Cost to first signal | Most informative next experiment |
|---|---|---|---|---|
| 1 | Exact-sim search over B8 — mechanism (offline) | **FUNDED** (sign certain; size soft) | done | none — stop replication |
| 1a | **Served sparse exact identity C15 on this Mac** | **20 % / 5 % by Nov 7 (10 % by Nov 14)** | M0 0 u + 1 Mac-h; M1 2 Mac-h; build 3 Sol-d | **M0 lead census + M1 co-residency → D1 Sat 12:00** |
| 1c (new) | Product transfer of exact search to FRP3+v8 Watch6 | 40 % premise holds / — | C1 6–8 Mac-h | **C1 Sat–Sun → D2 Mon 12:00** |
| 1b-i | Tail critic after H64 prefix (V0/V1) | 5 % / 2 % | V0 1 Mac-h filler; V1 ≤ 40 u | only after D3 GO, Mac gaps only |
| 3 | P1 100M PPO endpoint | 12 % / 6 % | Y3 ≤ 24 u → 1M → 100M ≤ 120 u | Y3 source-latebody-05 dispatch (slot A) |
| 9 | Late-state densification / restart archive | 5 % / 2 % | DENSITY-2 ≤ 5 u | DENSITY-2 completion-03 (slot B) |
| 1d | Harvest operator (O1 GO) | closed this campaign | — | reopens only if C1 shows harvest value |
| 2, 4, 5, 6–8 | Representation; GPU engine; search-labelled student; world model/league | 0 / 0 / 3 / 1 % | — | closed / not funded |

## 6. Still unresolved

The last-avoidable-decision distribution of the 28 rescues (M0); whether four native workers leave a live 24-fps game overrun-free
(M1); whether rescuable situations exist against vetoed FRP3+v8 opponents and whether a saved life beats the baseline's next life
(C1); the delivered fraction under the real clock (B3); the all-die/delay cost on natural doom roots at W4; cold-process and thermal
behaviour over 30+ h paced runs; the boost share of any search controller (never reported for L2′); X5 (bare strict episodes/h);
the owner's appetite if D3 prints KEEP_B8 (asked Tue Oct 27 23:00 with the receipts).

## 7. Experiments Claude can run now (specs frozen in PLAN_v3 §1; report in LIVE_FACTS.md)

- **M0 lead census** — shard part tonight (0 u, Claude, read-only script over the 8 shards; publish the per-rescue table before the
  replay); LAD replay Fri 09:00–10:00 under the lease (≤ 1 Mac-h; Sol ≈ 3 h on `l0.py`/`ceiling`). Close rule frozen in PLAN_v3 §1.
- **X5 bare strict timing** — tonight 20:00–22:00 under the lease (2 unpaced Mac-h; as registered). Prices strict.
- **M1 native co-residency falsifier** — Fri 19:00–21:00 under the lease (≤ 2 Mac-h, 0 u; Sol 0.5 d wiring `create_stop_operator()`
  into the L1 harness + kernel-row attestation; Codex-V 0.25 d).
- **M3 receipt-boundary closure** — Sol 1 d + Codex-V 0.5 d Fri–Sat; no Mac; gates L1″ admission.
- **C1 served-roster development bank** — Mac Sat 12:00 → Sun (≤ 8 unpaced Mac-h, cap 16); Sol 1 d build Thu–Fri; Codex-V 0.5 d.
- **B2 natural-root library** (FRP3+v8, continuous respawn, hash-selected L ≥ 800 roots, all legal H300 alternatives) — L4 slot C after
  a ≤ 1-u canary, ≤ 20 u (cap 40), Sat–Sun. Consumer: C15 fixture bars (≥ 4 distinct frozen/mixed doom fixtures) and L1″.
- **P1 Y3 source-latebody-05 → 1M → 100M** (slot A, ≤ 24 u Y3 cap) and **DENSITY-2 completion-03** (slot B, ≤ 5 u) — as registered.
- Not funded: L4 L2′ rulers/replication, Round 0 re-reg/0-D, O2, V1 before D3, E14/E16, A100/H100, RunPod.
