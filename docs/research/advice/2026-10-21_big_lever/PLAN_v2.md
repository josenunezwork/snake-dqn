# PLAN_v2.md — re-plan after Line S failed (Wed Oct 21 16:00 → Sat Oct 24 16:00 PT, and the week to Tue Oct 27 23:00 PT)

Written by the round-11 Fable judge, 2026-10-21 ~15:00 PT, from `round-11/JUDGMENT.md`, `LIVE_FACTS.md` (14:30 PT), `QUEUE.md`,
`PLAN.md` and the repo at `18a28ae3`. Authority order unchanged: owner > registered rules/amendments (byte-pinned) > this plan >
advisor memos. **This file replaces PLAN.md §0–§5 (the week-1 release track) and §9's decision points.** PLAN §6 (R-track print
licences), §7 (reducer, signature texts), §8 (week-2 family rules) remain in force where not overridden below. Nothing here
dispatches, registers, spends or deploys by itself; each action names its owner and receipt. Owner-only: release OK (none this
week). Everything else is pre-delegated (owner 2026-10-20) with Fable-judge rulings recorded in `CAMPAIGN_LOG.md`.

## §0 Verdict and scope

- **No served gain over bare B8 s3 can be qualified this week.** Served exact search is 15–25× too slow on this Mac
  (`exp/bl-l1-20261021/RESULT_L1.md:29-44`), measured with the stock Python opponent-veto hook (S2 native path was not
  switched on in the timed code — `round-11/JUDGMENT.md`, facts block). GPU engine STOP, Claim A KILL, E2 screen unfunded, O2
  unaffordable at power. Any candidate still needs ≈ 80 serial Mac-hours of L3/L4/strict/product/serving (`PLAN.md:153-159`).
- **The week buys decisions for week 2:** (M1) the offline frozen/mixed ceiling of exact search (L2′); (M2) whether the
  opponent-veto cost can be removed *exactly* on this Mac (K-1); (P1) one fixed-endpoint 100M PPO artifact with a fresh paired
  evaluation. Plus E3's print and two priced Mac receipts (X5, E14).
- **B8 s3 stays served; no strict family opens; the only signature text that can appear is PLAN §7 text B.**
- Bars do not move: Mac strict (pooled ≥ +40 ∧ LB90 > 0, survival band v2, scripted no-death ≥ +.03, pre-data freezing),
  serving qualification (24 fps Watch6, zero overruns, update p95 ≤ 11.688 ms, W ≤ 4 / K ≤ 8 per SC §4 amendment), owner release OK.

## §1 Sol work packages (parallel; one writer + one named reviewer each; receipts create-only JSON + `.sha256`/`.complete`)

| WP | Owner → reviewer | Content | Due (PT) | Receipt |
|---|---|---|---|---|
| **WP-1 L2′ transport + re-chunk + reprice** | Sol-S → Codex-V | (i) Fixture/root transport that cannot hit the upload cap: runtime reconstruction of the 9 Y1 roots and the 160 L2′ worlds from seeds/bank hashes (preferred), or one compressed bundle ≤ 6 MB; Mac preflight exercises the **real** supervisor transport at real bytes and prints forecast ≤ 50 % of the 300-s cap (Z-9 as written, now for fixture directories). (ii) Re-chunk `BANK_L2.json` into **8 immutable 20-world chunks (8/8/4)** covering 160 exactly; per chunk `2 × estimate ≤ timeout ≤ 12 h`, estimate from the Y1-L4 receipt (or Mac Y1 rate × 2 if Y1-L4 has not landed by Thu 12:00 — judge's ruling, sizing input only). (iii) Readiness/manifests accept the repriced envelope **≤ 200 u incl. 2× liability + 20 % reserve**; label map, analyzer, one look, bank unchanged (`analyze.py` versioned map `FUND_MAC_SYSTEMS→FUND_SYSTEMS`, `KILL_SERVED_SEARCH_PUSH→PACKAGE_KILL`). Pinned-analyzer merge only after all 8 chunks terminal and valid. | code **Thu 10:00**; Y1-L4 dispatch Thu AM; first chunks **Thu 12:00** | `exp/bl-l2/v2/{transport-preflight,plan,chunk-NN/receipt}.json` |
| **WP-2 K-1 exact native opponent veto in branch rollouts** | Sol-K (separate builder) → Codex-V | (a) Enter `research/t34_speed_s2/native_v8.native_scope` for frozen/mixed branch rows in `bank_engine.rollout_all` and in `bl_served_search/ruler_operator` (shadow/root rows keep the stock hook as in T34), attest native vs fallback row counts in the receipt. (b) **Kernel-only mode**: `v8_nb.v8_decide` without the Python counter replay (counters are diagnostics, never decisions — `native_v8.py:12-17`), audited action-identical against the stock hook on the 9 Y1 fixtures × 30 batches + 24 fresh long-body roots (3 mixes × 8), Q bytes/actions/world/RNG/carry compared. (c) Optional, after (b) PASSes: pack the six branches' opponent rows into one kernel call per frame. Mac timing with the Y1 receipt schema (`stage_seconds`), W1, one thread, night window, load logged, no daemon killing. | code Thu 16:00; Mac timing **Thu 20:00–22:00** | `exp/bl-k1/{identity,timing}.json` |
| **WP-3 P1 fixed-endpoint PPO registration** | Sol-R → Codex-V; funding ruling: Fable judge (this plan pre-ratifies conditionally, §3) | Prospective registration `research/bl_ppo/REGISTRATION_P1.md`: F recipe from the B8 trunk, E8 critic with held-out late-body EV ≥ .1 gate, seed/recipe/endpoint = 100M frozen, `target` separate from `pause_at`, no checkpoint selection, E8 tripwires (nonfinite, collapse, decile drift, delivered boost in [0.5 %, 10 %]); **Y3 durability** (exact continuation on a replacement L4 through ≥ 3 updates + episode boundary + 1,000 actions; transitive snapshot set) and a **1M qualification** run that measures the complete-job rate (collection + updates + probes) before the tranche; evaluation manifest: 128/mix fresh paired H5000 vs B8 s3 on Colab, one look, nomination filter = band v2 ∧ boost ∧ scripted no-death point ≥ 0 ∧ pooled point ≥ 40 ∧ LB90 > 0 (rank n/a: one artifact). Invoice: Y3 ≤ 8 u, 1M ≤ 4 u, 100M ≤ 120 u incl. 2× liability + reserve, eval ≤ 10 u. Upload path per WP-1(i) (18 kB manifests; checkpoints fetched + hash-verified on the runtime). | registration + Y3 manifest **Thu 14:00**; 1M qual Thu PM; 100M dispatch **Thu night** when slot A frees | `exp/bl-p1/{registration,y3,qual-1m,tranche-100m,eval}/receipt.json` |
| **WP-4 sparse future-root burst controller (build only; conditional on K-1 ≥ 5× by Thu 22:00)** | Sol-C → Codex-V | Per 01/08: production root provider targeting a reachable future root ≥ 144 frames ahead from the exact bare shadow; at most bare + two Q-ordered alternatives (ties by action index), resumable single-action continuations H32→96→192→300 without restart, dead branches killed, **first alternative surviving H300 published when bare dies**, delay-only rejected; W = 4 / K = 4 CPU; visibility timestamp + publication/consumption cutoffs per Z-2′ (`PLAN.md:81`); every timestamp (admission, shadow-ready, horizons, visibility, consumption) recorded. New identity: no T3/L1′ inheritance. Fixture measurement Sat (§2). | code Fri 20:00 | `exp/bl-wp4/build-receipt.json` |
| **WP-5 reviews, E3 consumption, closeout** | Codex-V (+ Sol as needed) | Independent review of WP-1..4; E3 recovery-04 terminal receipt → print (GO-with-architecture / GO-null / KILL) → E14 admission if GO-with-architecture; X5 and E14 Mac receipts; **Sat 16:00 handoff packet** `exp/week1-20261021/handoff/INDEX_v2.json` (PLAN §7 schema; six records; verbatim owner answers or PENDING). | continuous; packet Sat 15:00 | `exp/week1-20261021/handoff/INDEX_v2.json` |

Shared-file rule unchanged (`PLAN.md:72-73`): Sol-S alone edits `bl_served_search/*`; Sol-K edits only `bank_engine.rollout_all`'s
frozen/mixed path, a new `research/bl_k1/`, and the ruler operator's opt-in scope; Sol-C alone edits `bl_paced_identity/*`;
Sol-R owns `bl_ppo/*`. Any slip ⇒ `stage0/slips/<item>-<gate>.json` (PLAN §2 schema); it blocks dependents only.

## §2 The 72 hours — Colab slots (cap 3, measured) and the free Mac

**Colab rules (unchanged + one addition):** before any assignment: `colab usage`, terminal create-only receipt + verified release
of the previous job, fresh inventory, a **real-transport upload receipt** for the manifest family (fifth upload failure = Y1-L4,
`exp/bl-l2/run-y1b.log`). No paid idle; no automatic retry; a mechanical failure needs a new prepared recovery. L4 only; A100/H100
none (nothing is GPU-bound). Units are not the constraint (§4).

| Slot | Wed 16:00 → Thu 12:00 | Thu 12:00 → Fri 12:00 | Fri 12:00 → Sat 16:00 | After Sat |
|---|---|---|---|---|
| **A** | E3 recovery-04 (running; 16-u envelope) → terminal receipt | P1 Y3 (≤ 8 u) → P1 1M qualification (≤ 4 u) → **P1 100M tranche** (≤ 120 u; est. 20–32 runtime-h) | P1 100M continues | P1 endpoint ~Sat night/Sun → **P1 eval** 128/mix (≤ 10 u) Sun → print Mon |
| **B** | Y1-L4 ruler (≤ 4 u) as soon as WP-1(i) lands (target Thu 09:00) | **L2′ chunks 1, 3, 5, 7** (sequential, ≤ 12 h timeouts) | remaining L2′ chunks / parity | idle → release, or P1 seed 2 only if L2′ terminal and owner spend goal (§4) |
| **C** | idle until WP-1 (release; no placeholder) | **L2′ chunks 2, 4, 6, 8** | remaining L2′ chunks → **pinned merge + one look → print Fri night / Sat AM** | idle → release |

Forecast (Mac W1 rates from `y1-mac-01/mac-ruler.json` per_mix: frozen 0.055, mixed 0.104, scripted 1.50 batches/s; 92,480
batches; L4 cores assumed 1.5–2× slower): ≈ 290 Mac-core-worker-hours ⇒ 36–48 L4 runtime-hours at 12 workers ⇒ on two slots
≈ 18–24 h wall from Thu 12:00 ⇒ chunks terminal **Fri 06:00–12:00**, print Fri evening; a slipped chunk ⇒ Sat. With K-1(a)
adopted into the L2′ runner only if its parity receipt exists before Thu 12:00 (otherwise L2′ runs the stock hook — timing, not
semantics, is at stake). Expected units ≈ 60–100; envelope 200 incl. liability.

**Mac (serial, exclusive, AC, lid open, idle receipt first; log foreign load; never kill daemons; nothing paced):**

| When (PT) | Item | Hours | Receipt / kill |
|---|---|---|---|
| Wed 19:00–21:00 | **X5** bare strict timing: T7 harness, 48 disjoint dev worlds, B8 s3 ⇒ episodes/h | 2 unpaced | `exp/week1-20261021/mac/x5.json` |
| Thu 20:00–22:00 | **K-1 timing** (WP-2 a/b on the 9 fixtures; c if ready) | 1–2 | `exp/bl-k1/timing.json`; < 5× on `opponent_v8` (frozen-1500) or any action mismatch ⇒ exact served search CLOSED for week 2 |
| Fri 19:00–22:00 | **E14 bare cells** on the E3 architecture (B8 / F / F+G, encoder at W = 0), only if E3 printed GO with an architecture | 3 | `exp/week1-20261021/mac/e14.json`; p95 > 11.688 ms or overrun ⇒ G unservable |
| Sat 08:00–10:00 | **WP-4 fixture measurement** (only if built): 3 mixes × L800/1200/1500 × 10 reps, cold start, declared contention | 2 | `exp/bl-wp4/fixture-timing.json`; bars §3 |
| any gap | P1 bare actor Mac timing probe (frozen B8 + trainable encoder forward, `bl_ppo/model.py:204-223`) once a 1M checkpoint exists | 1 | `exp/bl-p1/mac-actor-timing.json` |

## §3 Decision points (dates) and pre-registered rules

| When (PT) | Decision | Rule |
|---|---|---|
| **Thu 12:00** | D1 L2′ dispatchable? | WP-1 transport receipt ∧ chunk plan ∧ (Y1-L4 receipt or Mac×2 forecast ruling) ⇒ dispatch chunks on B and C. Missing ⇒ slip file; L2′ latest first-chunk start for a Sat 16:00 print is **Thu 20:00**; later ⇒ print slips to Sun and the Sat handoff says NOT_READY(L2′). |
| **Thu 14:00** | D1′ P1 funded? | Pre-ratified **conditionally** by this plan: registration bytes frozen ∧ Y3 PASS ∧ 1M qualification shows complete-job forecast ≤ 46 h for 100M incl. probes ∧ critic gate PASS ⇒ tranche dispatch. Any fail ⇒ P1 NOT_READY (no silent shrink of the endpoint; no seed/LR changes). |
| **Thu 22:00** | D2 K-1 | PASS = action-identical ∧ `opponent_v8` ≥ 5× on frozen-1500 (⇒ batch ≤ ~6 s) ⇒ WP-4 builds. FAIL ⇒ WP-4 cancelled; exact served search closed for week 2; approximations (lever 1b) become the only served-search route and need their own fidelity tests (week 2 registration, after L2′). |
| **Fri 23:59** (day-3 decision, `QUEUE.md:23`) | D3 L2′ print | **PACKAGE_KILL** (pooled UB90 < 40) ⇒ **KEEP_B8 for the search family now** (§6): stop WP-4, no approximations, no O2, no search-labelled students. **FUND_SYSTEMS** (frozen & mixed LB90 > 0 ∧ pooled ≥ 40) ⇒ week-2 engineering program registered (§5). **INCONCLUSIVE** ⇒ pooled point < 40 ⇒ treated as KEEP_B8 for serving work (science preserved); point ≥ 40 ⇒ replication bank (another 64/64/32, ≈ 60–100 u) registered before any engineering. Chunk fault ⇒ INCONCLUSIVE, never partial analysis. |
| **Sat 16:00** | Handoff | `INDEX_v2.json` sealed; owner front page: the three prints (or their NOT_READY reasons), B8 unchanged, text B; owner asked ONE question (§7). |
| **Sun–Mon** | P1 eval print | UB90 < 40 ⇒ ARTIFACT_REJECTED_AT_BAR (mechanism not killed; one endpoint). point ≥ 40 ∧ LB90 > 0 ∧ guards ⇒ Mac strict (unpaced, sized by X5 and measured variance; N never shrunk) registered for week 2; monitoring/product/serving bars unchanged. |
| **Tue 23:00** | Week disposition | PLAN §7 reducer; per line `(scientific_verdict, readiness, release_status)`; KEEP_B8 criterion evaluated (§6). |

WP-4 fixture bars (registered before Sat's run, else it is micro-timing only): first valid H300 certificate ≤ 5 s on ≥ 50 % of
dense-rescuable frozen/mixed opportunities **and before each root's own publication deadline**; deliveries in ≥ 4 distinct
frozen/mixed fixtures; zero overruns; update p95 ≤ 11.688 ms; actions identical to the reference or INVALID. PASS buys only
permission to price a fresh paced development bank (L3-class, 9 Mac-h) in week 2 — never efficacy, never release.

## §4 Units and resources

| Item | Cap (u) | Expected (u) | Note |
|---|---|---|---|
| E3 recovery-04 | 16 (existing) | ≤ 16 | running |
| Y1-L4 ruler | 4 | 2–4 | after WP-1(i) |
| L2′ (8 chunks + parity) | 200 incl. 2× liability + 20 % reserve | 60–100 | prospective repricing of the 64-u envelope; bank unchanged |
| P1: Y3 + 1M + 100M + eval | 8 + 4 + 120 + 10 = 142 | 60–90 | one endpoint; second seed only on idle slots after L2′ terminal |
| **Total** | **≈ 362** | **≈ 180–210** | balance ≈ 1,135 u; owner goal ≈ 39 u/day — this week ≈ 2× that, lawfully |

Builders: Sol-S (WP-1) ≈ 10–14 h; Sol-K (WP-2) ≈ 8–12 h; Sol-R (WP-3) ≈ 8–10 h; Sol-C (WP-4, conditional) ≈ 12–16 h; Codex-V
≈ 16–20 review-hours Wed–Sat (half of PLAN's load: no gate registry, no Claim-B chain, no L3/L4 this week). Mac: ≈ 10 h, all
unpaced/diagnostic. A100/H100: none. RunPod: none.

## §5 The week after Saturday (Sun Oct 25 → Tue Oct 27) and what week 2 looks like under each print

- **L2′ FUND ∧ K-1 PASS:** week 2 = served sparse-burst identity program: WP-4 → Z-2′ clock seam acceptance → L1″ (new bars,
  3 Mac-h) → L3 (9 h) → L4 (32 h) → strict/product/serving as PLAN §4–§5 with a fresh priced invoice; realistic earliest
  qualification **Nov 3–7**, probability of a +40-at-bar qualified release in week 2 ≈ 10–15 % (PLAN §0 power arithmetic, unchanged).
- **L2′ FUND ∧ K-1 FAIL:** served exact search is dead on this Mac; week 2 = one approximation fidelity test (10's proxy-opponent
  test or 02's learned tail; ≤ 36 u, 1 Mac-h) registered before any controller; a served candidate is 3–6 weeks away at best;
  the owner decides whether to fund that (§7).
- **L2′ KILL/INCONCLUSIVE-below-bar:** KEEP_B8 for the whole search family (served, harvest, search-labelled students). Week 2 =
  R only (E3 → E14 → E16 if admitted; P1 outcome) at ≤ 1 slot unless P1's eval prints at bar.
- **P1 at bar (any L2′ print):** Mac strict (unpaced ≈ 7–10 h + controls/audit) registered week 2; product + serving ≈ 17 h;
  a bare artifact is the simplest serving path; monitoring 7 days AND 100k decisions before any "beats B8" claim completes.
- **Nothing at bar:** §6.

## §6 "Stop and keep B8" criterion (explicit)

**KEEP_B8 is declared on the earlier of:**
1. **Fri Oct 23 23:59 PT** — L2′ prints **PACKAGE_KILL** (pooled UB90 < 40 on the 64/64/32 fresh unpaced bank), or INCONCLUSIVE
   with pooled point < 40 (scoped to serving/engineering work; science preserved); **or**
2. **Tue Oct 27 23:00 PT** — no artifact has printed, on a fresh registered unpaced bank, **pooled point ≥ 40 ∧ LB90 > 0 vs B8 s3**
   (L2′ FUND_SYSTEMS counts for the search family; P1's 128/mix eval counts for R) **and** K-1 < 5×.

**Under KEEP_B8:** B8 s3 remains the served Watch hero (no change, no label, no rollback needed); no serving, strict, product or
Claim-B machinery is built or kept warm; WP-4 and every search-approximation lane are closed; O2, Round 0 re-registration, E1b
and search-labelled distillation stay closed; Colab drops to ≤ 1 slot of registered research (E3 → E14 → E16; P1 follow-ups
only if its eval printed point ≥ 20 with LB90 > 0, as a mechanism signal); the owner is told in one sentence that the big-lever
campaign produced no served gain and is offered exactly one choice (§7). Reopening a release ambition requires a new charter and
a candidate with a fresh-bank print at bar — not a plan.

**Partial outcomes:** L2′ FUND with K-1 FAIL is *not* KEEP_B8 for science but *is* KEEP_B8 for serving until an approximation passes
its fidelity test; P1 at bar with L2′ KILL closes search and keeps R open on its own strict chain.

## §7 What the owner can trust at Sat 16:00, and the one question

Trust: (i) the served hero is unchanged; (ii) three measured, receipt-backed facts — the offline frozen/mixed value of exact search
(or why it is not yet known), whether the opponent-veto cost is removable exactly on this Mac, and the state of one honest 100M
PPO run — plus E3's print; (iii) no family opened, no bar moved, nothing laundered (no PASS string, Colab receipt or old pin
stands in for a Mac same-identity receipt); (iv) a dated week-2 route for each print (§5) with its honest probability.

The one question for the owner at Sat 16:00 (asked only then, with the prints in hand): *"Given these three results, do you
want week 2 spent on the route they admit (§5), or do you want B8 kept and the campaign closed to ≤ 1 research slot?"* No release
question exists this week.

## §8 Operator card (lawful next action with no judge)

1. Before any Colab assignment: `colab usage` + terminal receipt + verified release + **real-transport upload receipt**; else NOT_DISPATCHABLE.
2. A mechanical failure ⇒ terminal receipt, no retry, new prepared recovery with its own transport receipt; slot free only after verified release.
3. A missing number (R, chunk estimate, complete-job rate, episodes/h) ⇒ NOT_READY with the missing name; never copy a Colab timing onto the Mac; the Mac×2 forecast is a sizing input for L2′ only, never evidence.
4. L2′: never analyze a partial bank; one look under the pinned analyzer after all 8 chunks are terminal and valid.
5. K-1: any action mismatch ⇒ that mode INVALID; report native/fallback row counts; no speed claim without the identity receipt.
6. P1: no endpoint change, seed change, LR change or checkpoint selection; caps expire as INCOMPLETE.
7. Mac: idle receipt first; log load; stopped attempts retained; nothing paced; no daemon killing; AC + lid open.
8. Nothing this week can print QUALIFIED_AWAITING_OWNER; only signature text B exists. Anything that would let a PASS string, a Colab receipt or an old B8 pin stand in for a same-identity Mac receipt ⇒ stop.
