# Round 12 — RETROSPECTIVE: why did the week-1 levers go badly, and is PLAN_v2 still the plan?

Written by the round-12 Fable judge, 2026-10-21 ~18:00 PT. Base `B = /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever`.
Repo `/Users/josenunez/Projects/ml/snake-dqn` at `1a000f3e` (main). Read-only for you; light commands only; cite file:line and
receipt paths; never open `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-study-01/`. No jobs, Colab, tmux.

## Owner's request (verbatim)

"have astras and fable look at why we didnt do well on those and if its still going according to plan"

This round is a **blameless retrospective**, not a lever-hunt. The prior rounds' shortlist and PLAN_v2 are inputs, not things to
re-derive. Your job: explain the outcome record below, attribute each bad outcome to its real cause class, and say whether the
plan the campaign is now executing (`B/PLAN_v2.md`) is still on track and still right for Thu–Tue.

## Read first (in this order)

`B/CHARTER.md` → `B/PLAN.md` (round-10, superseded §0–§5) → `B/round-11/JUDGMENT.md` → `B/PLAN_v2.md` (current) →
`B/LIVE_FACTS.md` (authoritative measured facts, newest last) → `B/QUEUE.md` →
`docs/research/final_run_20261019/CAMPAIGN_LOG.md` lines 78–109 (rows from 2026-10-20 ~18:45 PT through 2026-10-21 ~17:30 PT).
Receipts are listed per outcome below; open them before attributing cause.

## The outcome record since 2026-10-20 evening (every row verified by this judge in the named receipt)

### Failures / stops (chronological)

| # | Outcome | What was measured | Receipt |
|---|---|---|---|
| F1 | **Path B v2 deliverability smoke STOP** (10-20 ~23:45) | Served exact-sim lookahead at 24 fps on the Mac: N=1 delivered 7/75 T3 improvements (all delay-only), 0/34 rescues, 0/5 M3; N=6 0/75 and 4.97–15.92 fps. Operator-02 chain and the paced Path B study cancelled. | `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-pathb-smoke-20261021/RESULT.md` (stopping rule committed before data, `7622d62e`) |
| F2 | **Claim A D0 = KILL** (first run 10-20 ~23:30 lost its receipts to an export/sidecar bug; registered rerun 10-21 ~01:10) | "no corrected objective keeps the anchor guards for >= 512 updates" → corrected-loss distillation of the rescue teacher into B8 closed; R1 labels have no consumer. | `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-claim-a-20261020/d0-l4-02/output/exports/D0_DECISION.json` (`decision: KILL`, `registered_run: true`, report sha `790d532a…`); export fix `da1f88c0`/`ab4fe3a8` |
| F3 | **Round 0 integrity KILL (operator error)** (10-21 ~00:20; ruling KILL_STANDS 00:40) | All 240 pairs sealed COMPLETE, Mac replay PASS, but the replay ledger append was not published before the sole analysis; `require_published_history` raised first; analyze maps HistoryError → KILL; registered verdict precedes clauses. No outcome was ever read. Lane CLOSED (no consumer). | `docs/research/advice/2026-10-21_round0_integrity/JUDGMENT.md`; CAMPAIGN_LOG rows 00:20 / 00:40 |
| F4 | **E1b GPU engine NO_CONTINUE / INCOMPLETE** (first L4 attempt failed on a CUDA dtype — bool `scatter_reduce_` — that the CPU-only preflight could not see; CUDA-fix relaunch 10-21 ~07:00) | 1 of 18 decisions in a 10,820-s wall budget; **0.217 full world-steps/s on L4** vs the frozen ≥ 20,000/s bar → engine thesis STOP for the week; lever 4 (GPU-resident sim / search-in-the-loop at scale) off the table. Mac CPU tiny preflight had printed 139 steps/s ("not predictive of CUDA"). | `B/exp/bl-e1b/cuda-dtype-fix-20261021-01/l4-output/exports/benchmark.json` (`full_world_steps_per_second: 0.21702…`, `NO_CONTINUE`); first-attempt error `B/exp/bl-e1b/l4-output/poll-000010.json` (`scatter_reduce_(… ok.reshape …)`); fix commit `88adb3e2` |
| F5 | **E9 / E9′ screen underpowered → E2 PPO screen NOT funded** (10-21 ~02:45 and ~07:10) | With measured T23 SDs (365/263/212) and the .05 E2 + .05 E6 Holm family, joint funding probability at a true +40 = .27 (96/32/32) / **.31** (128/48/48); best replacement design (fixed 192/64/64) **.362** — all < the required .60. R track research-only. | `research/bl_ppo/E9_RULE.md`, `research/bl_ppo/E9_REPLACEMENT.md:132-133` (SDs); `B/exp/bl-e2/e9-replacement/report.json`; `B/exp/bl-e2/e9-oc-04/report.json` (`INCOMPLETE_NOT_SIMULATED` for S1-variance scenarios) |
| F6 | **O2 (harvest value test) needs ~5,100 worlds** (10-21 ~09:55) | 80 % power for +10 incremental over D needs **5,097** paired worlds (~150 runtime-h); the budget fallback 1,260 worlds has .34 power. Deferred behind L2′. | `research/t34_o2_harvest/REGISTRATION_O2_V2.md:51-52,81`; CAMPAIGN_LOG 09:55 row |
| F7 | **L1′ Mac latency: served exact search 15–25× too slow** (10-21 ~14:30) — **measured with the STOCK Python opponent-veto hook; the S2 native path (merged 10-20 23:58, 1.90× whole-workload) was never enabled in the timed code** (erratum 14:50) | Per-root dense search S p50/p99 on CPU 19.7/22.9 s (W1) → 23.7/27.2 s (W4); MPS 32–41 s; tick overruns in every trial; ceilings 0.83–1.458 s. Mac Y1 6×H300 batch: frozen 10.6–26.3 s, mixed 7.6–11.0 s, scripted ~0.6 s; `opponent_v8` = 78–90 % of every frozen/mixed batch. Formal verdict NOT_READY(INSTRUMENT_MISSING) — the production root provider/consumption path was never built; quiet host never achieved (analyticsd/mediaanalysisd). | `B/exp/bl-l1-20261021/RESULT_L1.md` (tables at :14-24, :29-44), `y1-mac-01/mac-ruler.json` (per_mix frozen 0.0550 / mixed 0.1037 / scripted 1.501 batches per worker-s); erratum: `round-11/JUDGMENT.md` facts block; `research/t34_speed_s2/native_v8.py:116,140` (opt-in `native_scope`), `research/bl_served_search/ruler.py:134,214` (Boolean only) |
| F8 | **Y1 L4 ruler R = 0.166 batches/s → L2′ NOT_DISPATCHABLE** (10-21 ~17:30; two earlier attempts failed mechanically: manifest `other_assignments` list-vs-map, then the upload cap) | 12 workers on L4 (12 vCPU): frozen 0.0074 / mixed 0.0136 / scripted 0.147 batches per worker-s vs Mac 0.0550 / 0.1037 / 1.501 ⇒ **Colab L4 vCPUs 7–10× slower per worker than the Mac**, stock hook. Bar was R ≥ 2.18. WP-1 re-chunk on Mac×2 forecast: 566 min/chunk (needs 1,132 > 720 cap), 342 u > 200 u → still NOT_READY. L2′'s sizing, chunking and bar were all built on Mac rates and the T3-harness assumption. | `B/exp/bl-l2/manifests-y1-upload-recovery-20261021-01/y1-ruler-undispatched-runtime/exports/ruler-receipt.json` (`aggregate_batches_per_second: 0.16638`, `status: NOT_DISPATCHABLE`, `hardware.vcpu_count: 12`); failures `B/exp/bl-l2/run-y1.log` (AccountingError), `run-y1b.log` (TransferError upload cap) |
| F9 | **E3 = KILL** (10-21 ~17:30; recoveries 01–03 all failed mechanically before 04 ran) | Geometry student B: −26.6 % regret but LB90 **−5.41** and **56.25 %** unnecessary switches; history C −18.2 % → representation lever closed; E14/E16-G moot. | `B/exp/bl-e3/recovery-04/colab-output-04/exports/e3/fits/fit_report.json` (`-5.4095…`, `0.5625`); CAMPAIGN_LOG 17:30 row |
| F10 | **Repeated mechanical Colab failures** (7 `INCOMPLETE:` lines across `B/exp/*/run-*.log`, plus D0's lost receipts and a crashed owner concurrency test) | (a) D0 run-01 exported without supervisor sidecars (receipts lost, rerun needed); (b) E3 run-01 upload cap (57.8 MB); (c) E4 upload cap (checkpoint `best_apex_stage1_fs.pth`); (d) Y1-L4 upload cap on `ruler-fixtures/*.sha256` **after Z-9 was declared fixed** — Z-9 covered checkpoints, not fixture directories; (e) E1b CUDA bool `scatter_reduce_` not caught by CPU-only preflight; (f) E3 recovery-02 supervisor `exec` poll timeout 51 s during reconstruction 33/256; (g) E3 recovery-03 Mac↔x86 float drift (identity mismatch at decision 2916 — known since 10-08, `mac-pod-float-ties`) → replay-recorded-actions fix; (h) Y1 run-01 manifest `other_assignments` list-vs-map AccountingError; (i) owner concurrency test #2 script crashed (CPU runtime ReadTimeout then bash-3.2 empty-array error) — 2.5 h of idle Colab on the critical path. Also X5 (bare strict timing, 0 u) blocked twice because the K-1 identity audit held a CPU slot in the "exclusive" Mac window. | `B/exp/bl-e2/run-e4.log`, `B/exp/bl-e3/run-01.log`, `run-02.log` (TimeoutExpired), `run-03.log` (Refused foreign generation); `B/exp/bl-e3/recovery-04/decision-2916-verification.json`; `B/exp/bl-e1b/l4-output/poll-000010.json`; `/Users/josenunez/Projects/ml/snake-dqn-artifacts/colab-limit-test/{test2.sh,result2.log}`; `B/exp/week1-20261021/mac/x5-disposition.json` (`status: NOT_READY`, `timing_measurement: UNAVAILABLE`); CAMPAIGN_LOG 02:55, 07:10, 09:55 rows |

### Successes (so the retrospective is honest about the base rate)

| # | Outcome | Receipt |
|---|---|---|
| S1 | **O1 harvest census = GO**: 35/48 fresh B8 s3 worlds have a surviving single-action H500 branch ≥ +5 from death-drop intake (frozen 9 / mixed 13 / scripted 13; KILL bar < 8/48). Mechanism exists; serving it is moot until a served search exists. | `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-o1-20261021/census-02-extracted/decision.json` |
| S2 | **S2 native v8 MERGED**: D3 1.8994× whole-workload (98.5 → 51.9 s median, contended Mac), 144/144 export files byte-identical. | `research/t34_speed_s2/RESULT_D3_MAC_CONTENDED_01.md:3-12` |
| S3 | **K-1 action identity PASS 294/294** (native veto in frozen/mixed branch rows; 270 fixture batches + 24 fresh roots; independent Codex-V reconciliation). Timing NOT_RUN yet (Thu 20:00–22:00 PT). | `B/exp/bl-k1/identity.json` (`ACTION_IDENTITY_PASS`, `completed: 294`, `timing_qualification: NOT_RUN`), `identity-independent-audit.json`, `timing-prepared.json` (`PREPARED_NOT_TIMED`) |
| S4 | **E4 collection rate**: ~1,440 active hero transitions/s per L4 (6 procs 1,403 vs 12 procs 1,444 — not CPU-scaling), ~850/s allocated → 100M ≈ 19 h active / 32 h allocated per L4. | `B/exp/bl-e2/e8-preflight-03/e4-ruler.json` |
| S5 | **Claim A D0 registered rerun completed** after the export fix (the KILL itself is a clean scientific answer, F2). | as F2 |
| S6 | **Phase 0 finished** 160/160 (Mac) 10-21 12:46; Mac free. | CAMPAIGN_LOG 14:35 row; `B/exp/bl-l1-20261021/phase0-idle.json` |
| S7 | L4↔Mac S0 mismatch explained (stored-Q drift ≤ 7.6e-6, 1,800/1,800 root actions agree); Colab concurrency measured (3 L4 max, 4th refused; T4 max 3 at 2 vCPU). | LIVE_FACTS 23:58 and 01:10 rows |
| S8 | **P1 registered** (F recipe, seed 20261021, 100M fixed endpoint) with Y3 durability prepared; E8 critic landed (101 tests). | `B/exp/bl-p1/registration/receipt.json`, `B/exp/bl-p1/y3/receipt.json` (`PREPARED_NOT_DISPATCHED`, cap 8 u) |

### Process facts you should weigh

- Builder throughput: **338 commits on main since 10-20 17:00** (all by one git identity; Sol/Codex via handoff). Subject-word
  census: 89 `Merge`, 29 `docs:`, 21 `Freeze`, 18 `Record`, 12 `Implement`, 8 `Fix`, 7 `Register`. Most of the week's code is
  registration/receipt/preflight scaffolding; the science-bearing code paths (ruler operator, served-search controller, E1b
  kernels) were each run for real exactly once.
- Colab spend is small: ~1,154 u at 10-20 19:20; week-1 expected ≈ 180–210 u (`PLAN_v2.md §4`). Units were never the
  constraint; **the Mac's serial clock, the 3-slot cap, upload transport and builder/review hours were.**
- Every job that reached its science printed a clean registered answer (D0 KILL, E1b STOP, O1 GO, E3 KILL, Y1 R). None was
  laundered; no bar moved; B8 s3 is still served. The pre-registration machinery worked *as a brake*; whether it worked as a
  *steering wheel* is the question.
- PLAN (round 10, 03:30 PT) assumed: S2 native on in the timed path (`PLAN.md §3` step 2); Y1 R ≥ 2.18/s on L4 (from Mac
  numbers); Z-9 fixed uploads; a served exact-search identity could be qualified by Tue. All four were wrong or unmeasured
  within 11 hours. PLAN_v2 (15:00 PT) assumed Y1-L4 ≈ 1.5–2× slower than Mac cores (measured 7–10×) and that L2′ could be
  re-chunked to ≤ 12-h chunks at ≤ 200 u (measured: 566 min/chunk at Mac×2, 342 u).
- The 10 big-lever rounds (10-20 23:30 → 10-21 03:30) ran **in parallel with** the first mechanical failures; the plan they
  produced was committed before Y1, L1′, E1b or Z-9 had a real measurement.

## Where PLAN_v2 stands at 18:00 PT Wed (what "on plan" would mean)

PLAN_v2 §2/§3 promised: D1 Thu 12:00 L2′ dispatchable (needs WP-1 transport + chunk plan + a rate) — **currently NOT_READY on the
rate alone** (F8: 0.166 vs 2.18; even Mac×2 forecast exceeds the chunk cap and the 200-u envelope); D1′ Thu 14:00 P1 funded
(registration frozen, Y3 PASS, 1M qualification, critic gate) — Y3 prepared, not dispatched; D2 Thu 22:00 K-1 ≥ 5× on
`opponent_v8` — identity PASS, timing not run; X5 — NOT_READY (contended Mac); E3 — KILL (E14/E16-G moot, so WP-5's E14 row
is dead); D3 Fri 23:59 L2′ print — cannot happen on the current numbers. The KEEP_B8 criterion (§6) fires Tue 23:00 if no
fresh-bank print reaches pooled ≥ 40 ∧ LB90 > 0 and K-1 < 5×. L2′ was the gate for the whole search family (1a/1b/1d/5); if it
cannot run on Colab, the family's offline ceiling is unmeasurable this week unless it runs on the Mac (≈ 290 Mac-core-worker-h
at W1 stock hook; ÷ K-1 speed-up if K-1 lands) or the bank is re-registered.

## THE ONE OPEN QUESTION

**Why did these outcomes go badly — and is PLAN_v2 still on plan and still the right plan for Thu–Tue?** Answer in three
separated ledgers, each row naming the receipt it rests on, then one integrated verdict:

(a) **World / science** — which of F1–F9 are *genuine* findings about the levers (the lever is weak, the hardware is what it
is, the effect is small relative to measured variance) that no better planning would have changed? E.g. is exact search over B8
simply too expensive per decision on any CPU host; are PPO/distillation gains over B8 just small (+40 against SDs 365/263/212);
is the GPU engine a dead end or merely unbuilt?

(b) **Planning errors** — wrong assumptions, unmeasured premises, and feasibility that was *priced in* rather than measured:
L1′/L2′ timed on the stock hook while S2 sat merged and opt-in; L2′ sized on Mac rates then run on L4 vCPUs; a 2.18/s bar
derived without a single L4 batch; PLAN committed before Y1/L1′/E1b/Z-9 had data; E2 screens designed before their power was
simulated; O2 built at N96 then found to need 5,097; a "release this week" scope that needed ≈ 80 serial Mac-hours of gates on
a Mac that was busy with Phase 0 until Wed noon. Which of these were *knowable* on 10-20 evening from receipts that already
existed (S0 ruler, S2 D3, T34 pod notes on Mac↔x86 drift and L4 CPU speed, `mac-pod-float-ties`, Colab CLI QUALIFICATION), and
which were genuinely unknowable without the run?

(c) **Process / operations** — preflight gaps (CPU-only preflight for CUDA jobs; Z-9 "fixed" for checkpoints but not fixture
directories; manifests not validated against the supervisor schema before launch), the upload-transport class (5 failures),
exec-poll timeouts, x86 float drift on a lane that needed bit-identity, the sidecar export bug, an operator error that KILLed a
sealed 240-pair study, an ad-hoc concurrency-test script that cost 2.5 critical-path hours, an "exclusive Mac window" that was not
exclusive (X5 vs K-1 audit), and the governance load (338 commits, 89 merges, receipts/sha256/.complete for everything) relative
to the number of real measurements produced. What is the cost of the registration machinery in builder/review hours and
calendar, and where did it prevent an error versus merely document one?

Then rule: **is PLAN_v2 (L2′ + K-1 + P1, KEEP_B8 criterion Fri/Tue) still on plan** given F8 (L2′ not dispatchable on Colab at
any lawful envelope) and F9 (E3 KILL)? **Is it still the right plan** — i.e. do its three measurements still produce a decision
the owner can act on by Sat 16:00 / Tue 23:00, or has F8 removed its central measurement? **What should change now** —
concretely for Thu–Tue (what to run, what to stop, where L2′ or its question goes, whether K-1 timing on Thu night is now the
single most important receipt, whether P1 should still take a slot), and **which process changes would have caught F4, F7, F8
and F10 earlier** (name the check, where in the pipeline it goes, what it costs, and which of this week's failures it would
have intercepted). Separate what you *know* from the receipts from what you *conjecture*.

Do not relitigate the bars (Mac strict, serving qualification, owner release OK, one look, pre-data freezing); they are fixed.
Do not propose new levers; this round judges execution and planning, not the lever space. A proposal to stop and keep B8 is in
scope only as an answer to "is this still the right plan".

## Deliverable (your `astra_ID.md`)

≤ 8-line bottom line → three ledgers (a)/(b)/(c), each row `outcome | cause | receipt | knowable-when` → verdict on PLAN_v2
(on plan? right plan? what changes Thu–Tue) → process changes (check, location, cost, failures intercepted) → 'Disagreements I
expect'. ≤ ~1,500 words. Cite receipt paths and file:line; mark conjecture as conjecture.
