# Are we moving slow? Direction and self-imposed rules — independent advisor (Fable, 2026-10-20 ~22:30 PT)

Owner's question, verbatim: "do u think we're moving slow — would you change direction, remove any restrictions we
imposed on ourselves, etc." Plus: "we know i want to use up my colab credits to help out."

Read-only. Sources: AGENTS.md, STATUS.md "Campaign authority", CAMPAIGN_LOG.md, HANDOFF_2026-10-20.md, SUCCESS_CONTRACT,
DESIGN_V2_SHADOW (+A1/A2), FINAL_PLAN, P7, the speed-vs-constraints and colab-lanes JUDGMENTs, the NO-GO branch plan,
Round 0 / 0-D / Path B / Claim A packages, T3 ceiling/screen/diagnostics, shadow RESULTS, the Claim A design review,
the units ledger, live tmux/log tails (operational lines only; nothing under the sealed Round 0 chunks was opened).

## Bottom line (≤ 10 lines)

1. **Yes, we are slow — but not where the speed JUDGMENT looked.** Since 10-19: 296 commits, of which ~123 are
   review/ratification/handoff/log, 63 merges, 31 docs catalogs/reorg proposals, ~17 results. Zero students trained.
2. **The single most important number in the campaign has never been measured:** whether the Path B v2 worker can
   deliver a rescue *before* the last avoidable decision at 24 fps. M3 said no for the serial path (30–63 s per doom vs
   12–25 s of lead); v2's first-result latency is `NOT_OBSERVED` (VERIFICATION_V2.md:140). We are about to spend ~27 h
   of quiet Mac on a controller that may be a predictable KILL. A 1–2 h development smoke answers it tonight.
3. **Direction: keep the mechanism (exact-sim search over B8), but invert the priorities.** Claim B (served lookahead)
   is the only lever that has ever beaten B8 by a wide margin (T3-dense +92 mass, 54/64 vs 38/64). Claim A
   (distillation) has lost every time it was tried (T3 screen −81…−88 vs CTRL; frozen loss flips 0.4 % of decisive
   rows). Run Claim B as the critical path on the Mac; run Claim A as a Colab-only background lane that never touches
   the Mac calendar until it has a screen-passing student.
4. **Lift one owner rule in SUCCESS_CONTRACT §4: "one extra CPU thread".** The Mac has 18 cores and the Watch server
   uses one. Six worker threads turns M3's 30–63 s searches into ~5–10 s, inside the lead. This is the cheapest path
   from "search works offline" to "search works served". Keep every timing bar (24 fps, p95 ≤ 11.688 ms, zero overruns).
5. **Colab in the next 48 h (~180 u):** lift the sequencing rules that only save units (labels wait for Round 0 GO;
   0-D only on NO-GO; fits wait for the Path B disposition). Collect R1 labels, run Round 0-D, the projection 2×2 and
   the unpaced Path B arm *now*, in parallel, as data — not evidence.
6. **Keep, untouched:** freeze-before-data, sealed outcomes/no-peek, fresh disjoint paired banks, Mac-only strict gate and
   serving qualification, paced evaluation of any served assistant, one look/fixed N, never relabel, owner release OK.
7. **Drop as ceremony:** 48 h dispatch-deadline = KILL; judge ratification for non-evidence code; ≥ 6-persona advice
   rounds; docs reorganisation mid-campaign; the "2 threads while registered jobs run" cap for throughput jobs.
8. Expected effect: Round 0 verdict Wed ~03:00 instead of ~15:00; Path B go/no-go on deliverability by Wed morning;
   first Claim A student with a working loss on Colab by Thu–Fri; Claim B strict gate as early as Sun 10-25 if the
   smoke passes. The release-integrity bar is unchanged by any of this.

---

## 1. Where the time actually went (verify)

| Item | Evidence | Reading |
|---|---|---|
| 296 commits since 10-19; ~123 governance (FIX-FIRST, review, ratify, judge, ruling, amendment, handoff, log, spec, advice), 63 merges, 31 docs catalog/reorg, ~17 results | `git log --since=2026-10-19` classified by message | Agent time is overwhelmingly on process artifacts. The speed JUDGMENT (§1.4) counted only wall-clock of the six retention FIX-FIRST rounds (~2 h); it did not count that the same pattern repeated for Round 0 (3 rounds), the bare-unpaced amendment (3 reviews + panel + judge), the boost-floor amendment (review + judge), 0-D (panel + rescope + re-review), Claim A (3-reviewer panel + fixes + pending judge), the A100 question (10 Astra + 3 Fable + 3 GPU experts), §5 label ordering (judge), the GO-file ruling (judge). |
| Students trained on search labels since 10-19 | CAMPAIGN_LOG "Measured results" | **0.** Claim A pipeline exists (9230b150) but its loss cannot flip decisive rows (design.md finding 1). |
| Phase 0: 88.85 h of 6 Mac workers on the stochastic N8/H128 teacher | PREREGISTRATION.md:110; FINAL_PLAN §2 ("distilling it distils noise") | A superseded teacher occupies the critical-path host for 4 days as a "cross-check". Sunk now (~12 h left); the lesson is: when a design is superseded, stop its jobs unless the cross-check is cheap. |
| Path B v2 first-result latency | `research/t34_pathb/VERIFICATION_V2.md:140` ("NOT_OBSERVED"), README_V2.md:198 ("engineering target; no final timing claim") | The deliverability of the whole Claim B line is unmeasured. M3 (`research/t34_shadow/RESULTS.md`) measured the *serial* path: exact searches exceed the D=300 lead in all five sampled dooms. Amendment 2 replaced it with earliest-first batched expansion and set a ≤ 2 s target, but no one ran it on a doom. |
| Path B chain: timing gate → T3-bank pre-check (~6–7 h) → study (18–19 h after the bare-unpaced amendment) | CAMPAIGN_LOG; amendment spec §5 | ~27 h of quiet Mac before the first number that says whether rescues arrive in time. The pre-check's ≥ 50/64 bar is the first deliverability measurement. |
| Round 0 Mac replay waits for Phase 0 | NO-GO plan §2; CAMPAIGN_LOG "Mac order" | The replay is a deterministic identity check on 10 % of pairs (README "Future Mac replay"). Contention cannot change a deterministic replay. It does not need a quiet host; it needs one thread for 1–2 h. Chunk 002 should be terminal around Wed 01:00–03:00. |
| Claim A loss: free-Q probe flips 0.4 % of 225 decisive rows; 1:1 sampler → ~1 optimizer step/epoch | `snake-dqn-artifacts/fable-panel-claima/design.md` | The frozen distillation objective is non-functional. The repair (branch `t34-claim-a-amendment-20261021`, 2,064 lines WIP) is the right work and is correctly first in Sol's queue. |
| Distillation prior | T3 screen: −81…−88 scripted mass vs CTRL across three arms, 5 seeds each; diagnostics: bank learned (94–99 % agreement), loss is spillover (75–85 % of disagreements are boost-mode) + late-growth deficit | Every training-based lever since B8 has lost (S1 −12/−15, T23 −80.8, M3-C −250, FRP-v4/v5). Claim A has a real design idea (anchor only on non-decisive rows; frozen trunk + zero-init residual) but no demonstrated win. |
| Search-acting prior | T3-dense: 54/64 vs 38/64, +92 pooled mass, 75 changed decisions of 301,589 (reused scripted dev bank); teacher_v2 recovers 39/42 true horizon rescues offline (92.9 %) | The *mechanism* works when compute is unbounded. Round 0 (240 fresh paired worlds, 3 mixes) is the right replication and is nearly done. |

**Conclusion.** The campaign's calendar is set by (a) a serial Mac chain whose first informative step is 27 h away and
whose feasibility was never smoke-tested, and (b) a Claim A line whose objective was broken until yesterday. Neither
is fixed by engine speed (the JUDGMENT was right about that) and neither is fixed by better reviews. They are fixed by
measuring the one unmeasured number and by re-ordering what the Mac and Colab do.

## 2. Would I change direction?

**Keep the mechanism; change the priority and the serving budget.**

- **Claim B is the fastest credible route to a served hero that beats B8 s3**, conditional on one fact: that the worker
  delivers rescues before the LAD under real pacing. Everything we know says the search finds the rescue
  (T3-dense, M2's 39/42) and that the serial worker is 2–5× too slow (M3). The question is purely one of compute
  budget inside the tick deadline.
- **The budget is a self-imposed rule.** SUCCESS_CONTRACT §4: "one extra CPU thread, ≤ 1 outstanding request".
  The owner ratified it on 10-19; it is not a statistical protection and not an integrity rule — it is a product
  constraint for a server that runs on an 18-core Mac and uses one core for the tick (Watch6 update p95 7.8–8.4 ms
  of an 11.688 ms budget at 24 fps). Allowing N = 6 worker threads (with the same "never blocks a tick, deadline miss
  → bare action, ≤ 1 outstanding request" rules) is the single change most likely to convert an offline +92 into a
  served gain. It is pre-data for Path B efficacy (only plumbing smokes exist), so it is a legal prospective
  amendment. If the owner wants a low-footprint product too, make N a registered parameter and measure N = 1 and
  N = 6 in the same timing gate.
- **Claim A stays, as a Colab-only background lane.** It is cheap now (~0.036 u per teacher world; a fit on L4 is
  GPU-bound and ~1 h), the owner wants the units spent, and the L3 2×2 is a genuine design question. But it should
  not hold the Mac calendar or the release decision: no Mac screen or strict slot until a student passes a Colab
  paired screen (96/32/32) with mass UB90 > +40 and scripted survival ≥ B8 − 0.03. Honest prior that Claim A beats B8
  at the contract bar by mid-November: ~20 %. Honest prior for Claim B given in-time delivery at N = 6: ~60 %;
  given N = 1: ~25 %.
- **Do not switch to a new mechanism** (self-play, PPO, league, GPU sim). Each is 2+ weeks to first signal and has a
  worse track record here. The exact-sim search over B8 is the one lever with a measured large effect.

## 3. Rules: what protects a "beats B8" claim and what is ceremony

The release-integrity bar is: a fresh, disjoint, collision-audited paired bank; a frozen candidate identity (source,
checkpoint, controller parameters, pacing); bars and look rules written before data; one look; no relabelling;
Mac-only strict gate and serving qualification on the owner's machine; survival band v2; the owner's OK. Nothing below
touches that.

| Rule | Verdict | Why |
|---|---|---|
| Freeze-before-data (candidate identity, metric, bank, N, selection, looks, stop rules) | **KEEP** | This is the protection against a false release. |
| Sealed outcomes / no peek before all chunks terminal | **KEEP** | Same. |
| Fresh disjoint paired banks; collision audit; strict banks never touched by development | **KEEP** | Same. |
| Mac-only strict gate + serving qualification; paced evaluation of any served assistant | **KEEP** | The claim is about the Mac-served controller under its real deadline. |
| One look / fixed N / Holm; never relabel a registered gate | **KEEP** | Same. |
| Owner's OK for release | **KEEP** | Non-negotiable. |
| **Review loop before ratification (independent reviewer → FIX-FIRST → Sol → re-review → judge)** | **KEEP for evidence-producing registrations (Round 0, 0-D, retention, strict gate, R1 screens); DROP for everything else** | Speed program, docs, tooling, Claim A *development* fits, descriptive arms: one reviewer or none. Today's loops found real defects in registrations — fine — but they were also run on amendments whose only effect was unit cost or ordering. |
| **Judge ratification for every amendment** | **RELAX: judge only when the amendment changes a bar, a bank, a look, or a candidate identity.** Ordering, cost caps, lane funding, scheduling: operator decides, logs it, moves on. | §5 label ordering, the GO-file ruling, the 0-D rescope cap and the boost-floor mechanics each consumed an independent judge for decisions that do not change what a PASS means. (The boost-floor *bar* change correctly went to the owner.) |
| **48 h dispatch deadline → KILL** (Round 0 FIX-FIRST #1) | **DROP** | It turns an ops hiccup into losing 13–33 u of sealed evidence. Integrity needs "no analysis before all chunks terminal, chunk order fixed, no outcome-based retry" — all already present. A deadline protects nothing statistical. |
| **"Mac serial chain": Round 0 replay only after Phase 0** | **DROP** | Deterministic replay; one thread; run it the moment chunk 002 publishes. |
| **≤ 2 threads of side work while registered jobs run** | **RELAX to "≤ 2 threads while a *timing/paced* phase is admitted or running"** | Phase 0 is a throughput job; 10–12 cores sat idle beside it for 4 days. Thermal guard + AC + lid rules stay. |
| **One paced session at a time on a quiet Mac** | **KEEP for operator-02 as ratified; register S4 multi-session for the strict gate** | The JUDGMENT got this right; the S4 memo (a907c649) is the right next registration. |
| **"One extra CPU thread, ≤ 1 outstanding request" for the served assistant** (SUCCESS_CONTRACT §4) | **LIFT the thread count (owner decision); keep the rest** | See §2. The Mac has the cores; nothing in the product uses them. |
| **R1 labels only after a published Round 0 GO** (§5 amendment) | **LIFT** | Labels are data, not evidence. The rule saves ≤ 60 u on a NO-GO; the owner has asked to spend units. Keep the condition that no R1 output is cited in or influences the Round 0 or retention verdict (that is a no-peek rule, and it is cheap to honour). |
| **Round 0-D only on a Round 0 NO-GO** | **LIFT: run it now in parallel** | It answers "rule loss vs premise shrinkage" (NO-GO plan §1) under *either* Round 0 verdict, and its teacher-arm labels are the dense labels (the only ones with a large measured effect). 27–39 u. |
| **Fits/screens/R2 wait for the Path B disposition** | **LIFT for Colab fits and Colab screens; KEEP for Mac screens** | The rule existed to protect Mac time. Colab fits do not touch the Mac. |
| **"Spend to meet a daily rate" excluded** (colab JUDGMENT §6) | **Keep the words, change the behaviour** | Do not pad a study; but do run every parallel data lane that *could* shorten a decision, even if it may be wasted on a NO-GO. At ~0.04 u/world and 1,150 u in the bank, the expected cost of a wasted lane is irrelevant next to a day of calendar. |
| Per-lane unit caps | **KEEP as liability bounds; set them from measured prices with 2× headroom** | Caps are fine; the problem was lanes that could not start. |
| RunPod banned for T34; no A100 for host-bound work | **KEEP** | Colab is pre-approved and L4 is the right host for host-bound teacher work; A100 for fits only on the JUDGMENT's trigger. |
| Advisor rounds: many personas + judge | **CAP at 2 advisors + 1 judge, as the owner's own 10-19 rule says** | The A100 round used 16 model calls to confirm a number in one profile. |
| Docs reorganisation / catalogs / reorg proposals (31 commits tonight) | **STOP until a verdict exists** | Pure agent-hours; the owner asked for a better snake. |
| Colab CLI re-acceptance on any `research/colab_cli` change | **KEEP** | It caught real defects; it protects the only legal transport. |

## 4. The next 48 h (what I would do, in order)

### Mac (18 cores; Phase 0 holds 6 until Wed ~10:00–13:00)

1. **Tonight, 1 thread then 6 threads, ~1–2 h: Path B v2 deliverability smoke (development, excluded).** Replay the
   five M3 doom samples plus the 22 T3 worlds with action changes through `research/t34_pathb/controller_v2.py` /
   `adapter_v2.py` at real 24 fps pacing, and record per doom: lead (frames from doom detection to the LAD), first
   fully evaluated improving action time, whether the result arrived before its frame, late/stale count. Run at
   N = 1 worker thread (the contract) and at N = 6. Report the fraction of T3 rescues that would have been executed.
   Write the stopping rule *before* running: "if N = 1 delivers < 50 % of T3's rescues on these worlds, do not start
   the operator-02 chain; amend §4 to N worker threads and re-time."
   This is the measurement M3 asked for and Amendment 2 deferred to a 6–7 h pre-check behind a timing gate.
2. **As soon as chunk 002 publishes (~Wed 01:00–03:00): Round 0 Mac replay (10 %) + sole analysis + amended
   `decide` + condition-5 gate**, beside Phase 0, one thread. Verdict by Wed morning, not Wed afternoon.
3. **When Phase 0 ends:** if the smoke in (1) says v2 delivers at N = 1, start operator-02 exactly as ratified. If it
   delivers only at N = 6, draft the §4 thread amendment, one reviewer, owner's word on the product constraint, new
   trace receipt, then the chain. If it does not deliver at N = 6 either, do **not** burn 27 h on the paced study;
   put the Mac on the strict-gate-ready work that exists (none yet) and on the S4 multi-session build.
4. **Nothing else on the Mac** except the speed program at nice 10 outside quiet windows (as ruled).

### Colab (3 concurrent L4 assignments; ~1,150 u; ~180 u in 48 h)

| When | Lane | Units | Rule it needs lifted |
|---|---|---:|---|
| now → Wed 03:00 | Round 0 chunk 002 (existing) | ~5 | none |
| now | Claim A L4 timing qualification (existing, running) | ≤ 5 | none |
| Wed 00:00 | **R1 label collection, 800 + 200 held-out fresh worlds, bare B8 s3 + teacher_v2, β = 1**, 2 lanes, ~5 h | ~40–60 | §5 "after a published Round 0 GO" → lift; keep "never cited in Round 0 / retention verdicts" |
| Wed 03:00 (lane freed by chunk 002) | **Round 0-D (dense premise, 240 pairs)** after the rescope FIX-FIRST at 392a97f9 gets its one reviewer | ~30–40 | "only on Round 0 NO-GO" → lift; one reviewer, no panel |
| Wed 08:00 | **L1 projection diagnostics** on the 39 excluded A5 worlds (free-Q 2×2, small fits) from branch `t34-claim-a-amendment-20261021` | ~12 | the "prospective ruling" → operator decision + one reviewer (it is development, not evidence) |
| Wed 12:00 (after R1 labels land) | **L3 projection 2×2 fits on L4** (frozen vs corrected loss × trainable vs frozen trunk), then a Colab paired screen 96/32/32 vs bare B8 s3 | ~60–120 | "fits wait for the Path B disposition" → lift for Colab |
| Thu | **Unpaced Path B arm** at `81c54731` (descriptive upper bound without deadlines) | ≤ 42.5 | "dispatch when the Mac study starts" → dispatch now; its result is development information for the Mac decision in step 3, with the stopping rule written first |
| Thu–Fri | R2 (β = 0.5) on the L3 winner if any arm passes its tripwires | ~80 | none once L3 exists |

That is roughly 180–280 u by Friday, i.e. the owner's spend goal is met by work that produces decisions, not by
padding. A100: only if an L4 fit is measured ≥ 60 min with GPU ≥ 40 % (the existing trigger); the host-bound teacher
stays on L4.

### Governance in the same 48 h

- Claim A loss amendment: operator writes it from the WIP branch, **one** reviewer, land. No judge unless it changes
  the ≥ 70 % held-out tripwire or the probe bank (it should not).
- 0-D rescope: one re-review of 392a97f9, land, dispatch. No panel.
- Stop the docs reorganisation agents; stop multi-persona advice rounds; stop re-hashing registrations for repo
  changes outside their import closure (the JUDGMENT already ruled this).
- Record the lifted rules as one dated operator note in CAMPAIGN_LOG (this file is the rationale). Only the §4 thread
  count needs the owner's word, because it is his product constraint.

## 5. On the speed-vs-constraints JUDGMENT

It is right that engine speed does not set the date and that review loops are agent-hours, not calendar. It is wrong
in one place and silent in two:

- **Wrong:** "the restrictions are not why the next verdict is slow". The restrictions that are slow are not the ones
  it audited (parity, digests, receipts). They are the *ordering* restrictions: replay after Phase 0, labels after
  Round 0 GO, 0-D after NO-GO, fits after Path B, one worker thread for the served assistant. Each of these converts
  parallel work into serial calendar in exchange for units the owner does not want to save.
- **Silent 1:** it accepted the 27 h Path B chain without asking whether the controller can deliver at all. M3 says
  the serial one cannot; v2 is unmeasured. A 1–2 h smoke is cheaper than the chain and than the six FIX-FIRST rounds
  already spent on it.
- **Silent 2:** it noted 16 idle cores for 25 h but did not notice the same 10–12 idle cores for 88 h beside Phase 0,
  nor that the "one extra thread" rule is why the served assistant cannot use them.

## 6. What this does not change

A "beats B8 s3" claim still means: fresh paired banks, frozen candidate (now including N worker threads and pacing),
Round 0-style bars, the Mac strict gate (Tier 1 + Tier 2, survival band v2) on untouched banks, serving qualification
with zero overruns, and the owner's OK. Lifting ordering rules and thread counts changes *when* we learn, not *what
counts*. Every development result above (smoke, unpaced arm, Colab screens, R1 labels) is labelled development
evidence and is never cited in a registered verdict.
