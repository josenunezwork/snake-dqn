# T34 NO-GO branch plan: Phase 0 and/or Round 0 negative (written 2026-10-20 ~18:40 PT, before either verdict)

Status: **operator planning document, pre-data.** Written and pushed before any Round 0 or Phase 0 outcome
value was opened. Phase 0 had 102/160 selection worlds on disk and Round 0 chunk 000 was running. This
document **does not** amend a registration, open a gate, or fund a label round. It does three things: it fixes
which branch each verdict code triggers, it fixes the one new premise study the negative branches need
("Round 0-D", §4), and it states which in-flight work becomes moot under which outcome (§6). Sources were read at main
`738250a8`. Line numbers refer to that commit.

## 0. Answer in brief

- **Phase 0 negative, Round 0 GO.** Nothing changes. Phase 0 is a cross-check only
  (`P7_STATISTICS.md:25-26`; `PREREGISTRATION_ROUND0.md:26-29`). A Phase 0 negative closes only the stochastic
  N8/H128 family, which `FINAL_PLAN.md:12` had already superseded. Run the GO plan as written.
- **Round 0 NO-GO (scientific STOP or INCONCLUSIVE).** Path B retention continues unchanged on the Mac. Claim B
  never depended on Round 0 (`SUCCESS_CONTRACT.md:61-68`). On Colab L4, run **Round 0-D**. This is the same
  240-pair premise test with the same bars, but the teacher is the exact **T3-dense** rule, the only lever that has
  ever beaten B8 by a wide margin. If Round 0-D returns GO, its sealed teacher-arm labels seed a dense R1, and
  Claim A runs on dense labels. Estimates: ~10–16 L4 lane-h (18–32 u, cap 40 u), verdict about Fri 10-23, Claim A
  candidate at the strict gate about Oct 28–Nov 1.
- **Both negative.** Same route, with a lower prior. Round 0-D is the last premise test for the search-teacher
  family. If it also fails, Claim A closes ("this teacher family failed its registered bars", P7 §6), B8 s3 stays
  served, and only the Path B verdict and the 2-deviation probe remain.
- **Fastest gated better snake in every case: Path B (Claim B).** Its verdict lands Fri–Sat and the strict gate
  could follow by about Oct 26–28. No verdict covered here makes it moot. Never abort it, because an operator abort
  counts as KILL (`PREREGISTRATION_RETENTION_V2.md:585`).

## 1. Evidence that drives the branches

| # | Fact | Source |
|---|---|---|
| E1 | T3-dense (every 5th decision from the first L ≥ 600, every advisory-legal action rolled D=300 with B8 greedy continuation, longest-alive chosen including all-die delays) scored **54/64 vs 38/64** scripted survival, mass 700 vs 608, and changed only **75** of 301,589 decisions. It ran on a **reused** census bank of 64 scripted-only worlds, never on fresh or frozen/mixed worlds. | `docs/research/t3_ceiling_test_2026-10-15.md:113-129`; `research/t3_rollout_labels_20261015/ceiling.py:103-140` |
| E2 | Sparse Q-triggered search on the same worlds reached only 41/64. Coverage of the doom window carries the value, not the quality of each search. | `t3_ceiling_test_2026-10-15.md:5,27` |
| E3 | Round 0's teacher is `teacher_v2`: shadow doom → bisection radius 3 → **latest** discovered horizon rescue. All-die delays never execute. | `research/t34_round0/PREREGISTRATION_ROUND0.md:33-43`; `DESIGN_V2_SHADOW.md:82-86` |
| E4 | Offline on T3's trajectories, bisection recovered **39/75** T3 changes (52%) but **39/42** true horizon rescues (92.9%). The 33 misses were all-die delays (+1–4 frames) in worlds that died anyway. | `research/t34_shadow/RESULTS.md:18-19,119`; `DESIGN_V2_SHADOW.md:84-85` |
| E5 | Round 0 GO needs scripted no-death ≥ 0.66 **and** ≥ +0.10 vs bare, pooled mass ≥ +40 with LB90 > 0, boost share in [0.5%, 10%], and survival guards A/B. The point-bar failures are STOP; interval-only failures are INCONCLUSIVE. | `PREREGISTRATION_ROUND0.md:112-131` |
| E6 | Phase 0 (stochastic N8/H128, mass + Q-leaf score, searching at every L ≥ 600 decision ≈ 43% of decisions, fresh 96/32/32 bank) uses its own bars: STOP if UB90 < +40 or scripted gain < +0.05; GO if ≥ 0.66, ≥ +80, LB90 > +40. | `research/phase0_search_20261019/PREREGISTRATION.md:60-71,103-106` |
| E7 | Distilling T3-dense labels already failed once (T3 screen −81…−88 scripted mass vs CTRL). The cause was spillover, mostly boost-mode disagreement, plus a late-growth deficit, not failure to learn the bank. | `docs/research/t3_screen_result_2026-10-15.md:9-14`; `docs/research/t3_diagnostics_2026-10-16.md:183-185` |
| E8 | T3-dense cost is 859k env-frames/world, i.e. 763 Mac worker-s/world at 1,126 env-frames/s in the T3 `bank_engine`. On a 12-process L4 lane that is an ESTIMATED 15–23 worlds/h: about the same per-world cost as the production shadow lane, because that engine is ~10× slower per row-step (digests). No GPU is needed. | `t3_ceiling_test_2026-10-15.md:129`; `RESULTS.md:109`; `advice/2026-10-20_a100/gpu_expert_2.md:71-77`; `CAMPAIGN_LOG.md:57` |
| E9 | T3-dense **cannot be served** under the contract. 48,844 worker-s / 36,996 batches = **1.32 s per batch on one core**, while every 5th decision at 24 fps allows ≈ 208 ms. That is 6.3× over, with only one extra thread and ≤ 1 outstanding request allowed. The servable form is Path B v2 (earliest-first). | `t3_ceiling_test_2026-10-15.md:125-129`; `SUCCESS_CONTRACT.md:50-54`; `DESIGN_V2_SHADOW.md:87-92` |
| E10 | Path B KILL leaves "teacher labels then wait on Round 0". The Path B T3-bank pre-check (≥ 50/64 on T3's own worlds, ~6–7 h) runs before its study. | `PREREGISTRATION_RETENTION_V2.md:348-356`; `DESIGN_V2_SHADOW.md:94-95`; `advice/2026-10-20_a100/SYNTHESIS.md:35-36` |
| E11 | No plan exists for a Round 0 negative. The only text is "the designed off-ramp". | `FINAL_PLAN.md:87,98`; `advice/2026-10-20_a100/REVIEW_C_adversarial.md:16-34` |

**Inference.** E4 says teacher_v2 already captures ~93% of T3-dense's true rescues. So a Round 0 failure has two
candidate causes, and they need different responses:
(i) **premise shrinkage**: T3's +0.25 on 64 reused scripted worlds does not survive fresh worlds and frozen/mixed
rosters (E1 is the only precedent; half of +0.25 ≈ the +0.10 bar, REVIEW_C:23-24);
(ii) **rule loss**: what T3-dense does that teacher_v2 does not. That means earliest-of-window intervention every
5th decision versus the latest rescue, executed delays, and re-searching after every change.
Round 0 alone cannot tell (i) from (ii). Round 0-D tests exactly that difference on fresh worlds at the same cost.
Path B's T3-bank pre-check isolates the rule on T3's own bank for free. **Branch choice depends only on verdict
codes (§3), never on descriptive Round 0 numbers.**

## 2. Calendar facts that bind every branch (PT)

- Round 0 chunks 000/001/002 (100/100/40 pairs, serial, one lane) should all be terminal by about **Wed 10-21 01:00**
  if each is dispatched when the previous one publishes. The 48 h dispatch deadline is about **Thu 10-22 16:45**
  (`HANDOFF_2026-10-20.md:143`).
- Round 0's sole look needs the **Mac** replay of 10% of pairs, both arms (`research/t34_round0/README.md:198-225`).
  Phase 0 holds the Mac until **Wed ~10:00–15:00**.
- **Scheduling ruling (this plan): run the Round 0 Mac replay and analysis immediately after Phase 0 finishes and
  before the Path B timing gate.** Estimated ≤ 1–2 h. Otherwise the Path B quiet-host study holds the Mac for 27–40 h
  and delays the Round 0 verdict, and with it this whole branch, to about Fri. Expected Round 0 verdict: **Wed 10-21
  ~13:00–17:00**. Phase 0 verdict: about the same hour.
- Path B: timing gate → T3-bank pre-check (~6–7 h) → study (≤ 40 h cap, ~27 h expected; ~10–12 h less if the
  bare-arm-unpaced amendment is ratified). Expected verdict **Fri 10-23 – Sat 10-24**.

## 3. Branch table, keyed by verdict codes only

| Phase 0 | Round 0 | Branch | Hour-zero actions (operator) |
|---|---|---|---|
| any | **GO** | **GO plan** (unchanged; §5 overlap ruling per SYNTHESIS item 3) | Log the verdict. Phase 0 STOP/INCONCLUSIVE retires the stochastic family only. Shelve Round 0-D (do not run it). Its engine stays available as an optional dense-label engine. |
| GO | **STOP / INCONCLUSIVE (scientific)** | **B: Round 0-D** (§4), high prior | Log. Run the frozen descriptive decomposition (§3.1, ~15 min Mac, after replay). Dispatch Round 0-D the moment it is ratified. Path B continues. |
| STOP / INCONCLUSIVE / INVALID_INCOMPLETE | **STOP / INCONCLUSIVE (scientific)** | **C: Round 0-D as last premise test**, low prior | As B. Dense R1 is **not** pre-funded beyond the sealed 240 worlds. Also run the 2-deviation probe (§5.3) on the same day. |
| any | **STOP by unconditional controller fault** (non-finite Q, illegal action, > 90% one action) | **Bug, not science** | Fix teacher_v2 and re-register Round 0 on a fresh namespace. Do not switch teachers on a bug. |
| any | **INCONCLUSIVE by integrity** (missing chunk, divergence, cap, deadline) | **Mechanical** | New Round 0 registration on a fresh namespace with the same rule. Dense is not substituted. |
| STOP / INCONCLUSIVE | not yet terminal | wait | No action. Phase 0 alone never changes the route (P7 §2). |

"Scientific" INCONCLUSIVE means the point bars pass and only an interval fails, or a guard bound is inside its
band. It authorizes no extension of Round 0 (`PREREGISTRATION_ROUND0.md:129-131`). Round 0-D is a **new study with
a new controller and a fresh bank**, not an extension. In particular, it is **not** "more worlds for teacher_v2".

### 3.1 Descriptive decomposition (frozen now; reported, never a gate, never changes the branch)
On the complete Round 0 exports after the sole analysis, report:
- per-mix paired mass, alive-frame and no-death deltas;
- scripted no-death gain next to T3's +0.28 for B8 s3 (E1);
- interventions per world, and the share of teacher-arm deaths preceded by an executed rescue vs none found;
- boost share per mix.

The same list is computed for Round 0-D, which makes the (i)/(ii) attribution in §1 explicit.

## 4. Round 0-D: dense-teacher premise test (pre-register now; Colab L4)

**Question.** Does the exact T3-dense acting rule over bare B8 s3 clear the Round 0 premise bar on fresh,
three-mix worlds?

**Controller (frozen).** T3-dense, exactly as in `ceiling.py:103-140` and E1:
- trigger every 5th episode-local decision, counted from the first L ≥ 600 decision; the clock keeps advancing
  through later length dips; no Q threshold; at most one batch per 5 decisions;
- every advisory-legal action, boosts included, rolled D=300 (or to H5000) with B8 greedy continuation, eps 0,
  no veto, real opponents, cloned RNG;
- select the action with maximum alive transitions; ties → B8 argmax → highest Q → lowest index;
- all-die delays **do** execute (this is the T3 rule);
- no advisory-legal action → B8 normal-only fallback.

Engine: `research/t3_rollout_labels_20261015/engine.py:161` `batched_rollout`. It was revalidated against the current
GridBatchSim because historical T3 source hashes differ (`RESULTS.md:93-98`).

**Design (copied from Round 0 so the two are comparable).** 240 paired terminal-hero H5000 worlds (96 scripted /
72 frozen / 72 mixed), slot 0, six snakes, mechanics v1, rosters as in Round 0, both arms in the same process.
- Fresh namespace `t34-round0d-dense-20261021/v1`, collision-audited and disjoint from the Round 0, Phase 0,
  Path B retention, T3 and strict banks; any hit rejects the whole allocation.
- **Bars and ordered decisions copied verbatim from `PREREGISTRATION_ROUND0.md:100-131`**: GO/STOP/INCONCLUSIVE,
  survival guards A/B, boost band, P7 inference, one look, no peeks, faults → STOP.
- OC: reuse `OC_ROUND0.json`; report the T3-like power, not re-sized.

**Two deliberate differences from Round 0, frozen now:**
1. **Replay on a second L4 runtime, not the Mac.** Same stratified 10% (24 worlds, 48 traces), cross-runtime
   A2-style; any unexplained divergence → INCONCLUSIVE. Reason: during Round 0-D the Mac is inside the quiet-host
   Path B study. Mac strict gate and serving remain Mac-only.
2. **Teacher-arm labels are exported, sealed and conditionally admissible.** They use the
   `research/t34_exit_b8/exports.py` per-action schema (`exports.py:132-171`): survive / alive_steps / mean_mass per
   action, forbidden set, and the `delay` tag per Amendment 2 §1 (`DESIGN_V2_SHADOW.md:82-86`).
   - Sealed until the verdict.
   - Admissible to R1 fitting **only on GO**.
   - **Never** admitted to any screen, probe or gate bank.

   This saves 240 of R1's 800 worlds. The premise bank is not reused as evidence downstream, because every
   screen and the strict gate use fresh banks.

**Prep before dispatch** (Sol, worktree `../snake-dqn-round0d`, started tonight as insurance; review loop with one
reviewer and one Fable approval judge):
- generalize `ceiling.py`'s dense controller into the Round 0 harness as a controller switch, inheriting Round 0's
  bank/integrity/ledger/deadline/faults code unchanged;
- add the exports-schema writer;
- **A5-D** (one L4 lane, ≤ 60 min, ≤ 3 u), with two parts:
  - fidelity: 8 T3 dense census worlds (b8_s3, first 8 in bank order) must reproduce the historical per-world
    endpoints and changed actions, or the revalidated M2 queries must explain each difference; anything
    unexplained → hold;
  - throughput: completed dense worlds/h/lane over a 20-min steady window. This sizes the chunks with Round 0's
    formula (`PREREGISTRATION_ROUND0.md:168-179`). **Kill: < 8 worlds/h/lane** (then 240 pairs need > 10 h on 3
    lanes; report and re-plan on Mac capacity after Path B).
- A6 Mac preflight on 2 excluded H500 fixtures with 1 thread, while Phase 0 runs.

**Compute and time (ESTIMATED from E8).**
- Teacher arm: 240 / 15–23 worlds per lane-h ≈ **10.4–16 lane-h**. Bare arm is negligible. With a 30% derate for
  vector61 opponents: up to ~21 lane-h.
- Two lanes (the third is kept for the unpaced Path B arm): **5–10 h wall**, **18–32 u expected, cap 40 u** (+20%
  reserve held, no automatic retry). Replay ~1 h, ≤ 2 u.
- Earliest timeline: Sol Wed → review/ratify Thu morning → A5-D Thu ~09:00–10:00 → dispatch Thu ~11:00 → verdict
  **Thu night–Fri 10-23 morning**.
- Mac use: zero, apart from the A6 preflight.

**What a Round 0-D GO buys (and nothing more).** Dense R1:
- the sealed 240 teacher-arm worlds + 560 fresh dense worlds;
- 560 / (3 × 15–23 per h) ≈ **8–12 h on 3 lanes, 15–25 u**, Fri–Sat;
- decisive rows weighted per FINAL_PLAN §3; delay rows weighted ~0 per Amendment 2;
- dense windows give ~300 decisive-class rows per doom instead of one LAD row (`gpu_expert_2.md:81-84`).

The fit uses the Claim A pipeline (§6) with the FINAL_PLAN §8 anchor and tripwires. The outcome-head arm is added
only if its probe returned GO (`gpu_expert_3.md:126-132`). Then come the Mac screen (after Path B releases the Mac,
about Sat), the strict gate (7–9 h, about Oct 26–28) and serving qualification. **Claim A candidate ≈ Oct 28 – Nov 1.**
E7 is the known risk: the T3 labels were learned and still lost. The registered anchor and tripwires
(τ=0.05, margin 0.1, all-length disagreement ≤ 5%, boost ≤ 1.5×) are the only answer, and R1 is non-terminal
(`FINAL_PLAN.md:89`).

**Round 0-D NO-GO.** The search-teacher family for Claim A closes ("this teacher failed its registered bar"):
no dense rounds, no further premise tests. B8 s3 remains served. Claim B is decided by Path B alone.

## 5. Per-case route, with the alternatives the evidence rules out

### 5.1 Case (a): Phase 0 negative, Round 0 GO or pending
Route: the GO plan. Claim B (Path B) is about Oct 25–28. Claim A is R1 teacher_v2 labels on L4 (800 worlds ≈ 29 u at
57 worlds/lane-h, `CAMPAIGN_LOG.md:57`), about Oct 26–30. A Phase 0 negative is consistent with `FINAL_PLAN.md:12`:
a noisy stochastic teacher distils noise. It retires T33-style N8/H128 search and nothing else. Mac order after
Phase 0: Round 0 replay → Path B timing gate.

### 5.2 Case (b): Round 0 scientific NO-GO, Phase 0 GO or still running
Route: **Round 0-D → dense R1 → Claim A**, with Path B in parallel on the Mac. A Phase 0 GO on its fresh 3-mix bank
would show that broad per-decision exact search helps on fresh worlds. That points to rule loss (ii) over premise
shrinkage (i) and makes Round 0-D's prior high. Ruled out:
- **Serving T3-dense as the planner.** 6.3× over the one-thread tick budget (E9).
- **Serving Phase 0's planner.** 4.1–5.4 s per search, 101–132 tick budgets (`research/phase0_search_20261019/README.md:84-86`).
- **Distilling Phase 0's stochastic labels.** 98/300 argmax agreement between draws (`FINAL_PLAN.md:12`).
- **Re-running teacher_v2 on more worlds.** That would be an extension in substance; INCONCLUSIVE authorizes none.
- **A GPU/device port.** ~3 weeks; it breaks even only above ~10k dense worlds (`gpu_expert_1.md:23-28`;
  `gpu_expert_2.md:136-143`).

### 5.3 Case (c): both negative
Route: **Round 0-D as the single last premise test**, with Path B in parallel. Two fresh-world tests of
"search over B8 helps" will have failed by then. The only positive evidence is E1 on 64 reused worlds, so expect
NO-GO. It is still worth running: it is the one study that tests the precedent itself, and it costs ≤ 40 u
with no Mac time. Same day: launch the 2-deviation rescue probe being built separately
(`gpu_expert_2.md:145-165`; 10 all-die T3 dooms, one L4 lane, ≤ 3 h, ≤ 6 u; ≥ 3/10 rescued → real, ≤ 1/10 →
the 1-deviation ceiling stands). It does not produce a snake this month. It decides whether a successor search
campaign has a mechanism above 54/64. If Round 0-D fails, **report that no credible route to a better bare snake
exists in this campaign window.** B8 s3 stays, and Claim B depends only on the Path B verdict.

## 6. What becomes moot, by outcome

| In-flight work | Moot when | Not moot when / note |
|---|---|---|
| **Claim A pipeline** (opus-claim-a-pipeline: ReplaySource, probe boundary, CUDA fit qualification, student fan-out; `research/t34_exit_b8/label.py:108-111`) | **Wholly moot only if Round 0 NO-GO and Round 0-D NO-GO.** After a Round 0 NO-GO alone, only the **teacher_v2 student fan-out** piece is moot. | Cache/ReplaySource/CUDA fit/probe carry over to dense labels, provided Round 0-D exports use the `exports.py` schema (§4, frozen). Keep building. If Round 0 GO **and** Path B GO, it is still needed but off the critical path (owner decides on label rounds, `SUCCESS_CONTRACT.md:65-66`). |
| **Path B retention** (ratified 7cd34b2e; plus the bare-arm-unpaced amendment, opus-pathb-amendment) | **Never mooted by Phase 0 or Round 0.** It ends only on its own timing KILL, pre-check < 50/64, or study verdict. | Claim B is independent of Round 0 (`SUCCESS_CONTRACT.md:61-68`). In the negative branches it is the **only** fast route to a gated snake. Do not abort it (abort = KILL). Its T3-bank pre-check is the free rule-vs-bank discriminator for §1. The unpaced amendment stays valuable in every case: it saves ~10–12 Mac-h. |
| **Outcome-head probe** (opus-outcome-head; 39 A5 teacher_v2 export worlds) | Its **funding consequence** ("fund R1 with S1c") is moot if both Round 0 and Round 0-D are NO-GO. | Its question (is the H300 per-action outcome recoverable from B8's state?) holds regardless of Round 0. Dense exports carry the same per-action outcome fields. Keep running. In case (c) it decides, with the 2-deviation probe, whether any distillation lever remains for a successor campaign. |
| Phase 0 analysis | never | Cross-check only. Still run its one look (`README.md:196-198`). |
| Round 0 chunks 001/002 | never | Must be dispatched by the 48 h deadline whatever happens elsewhere (missing chunk = KILL). |

## 7. Pre-register now (before any Round 0 / Phase 0 outcome is opened)

1. **This document** on main: the branch table (§3), the frozen descriptive list (§3.1), and the Mac ordering
   ruling (§2).
2. **Round 0-D Sol spec + draft pre-registration**, `research/t34_round0d/PREREGISTRATION_ROUND0D.md`, committed
   before the Round 0 verdict:
   - controller rule and engine;
   - namespace and audit scope;
   - bars copied verbatim from Round 0;
   - L4-replay deviation;
   - sealed-label admissibility rule;
   - A5-D fidelity and throughput kills;
   - caps (40 u, 360-min chunks, 48 h dispatch deadline, no automatic retry).

   It is ratified through the normal review loop. **It is dispatched only on a branch-B or branch-C code.** A Round 0
   GO shelves it unrun.
3. **Dense R1 sizing** (560 fresh worlds + 240 sealed, ≤ 30 u). It is written into the Round 0-D pre-registration
   as conditional on GO, so no second approval round sits on the critical path. Fit/screen/gate rules are unchanged
   from FINAL_PLAN §3/§8 and P7.
4. **2-deviation probe thresholds** exactly as `gpu_expert_2.md:161-165`, frozen before its first doom is expanded.
5. CAMPAIGN_LOG entry: "NO-GO branch plan adopted pre-data". Fable approval is needed only for (2) and (3) under
   the approvals rule.

## 8. Resource summary

| Item | Where | Wall | Units | When |
|---|---|---|---|---|
| Round 0 Mac replay + analysis | Mac (after Phase 0, before Path B timing) | ≤ 1–2 h | 0 | Wed ~13:00 |
| Round 0-D Sol build + review | Sol worktree, Claude review | ~1–1.5 days | 0 | Tue night – Thu AM (insurance) |
| A5-D fidelity + throughput | 1 L4 lane | ≤ 1 h | ≤ 3 | Thu AM |
| Round 0-D 240 pairs | 2 L4 lanes | 5–10 h | 18–32 (cap 40) | Thu – Fri AM |
| Round 0-D replay | 1 L4 (second runtime) | ~1 h | ≤ 2 | Fri AM |
| Dense R1 (560 new worlds, on GO) | 3 L4 lanes | 8–12 h | 15–25 | Fri – Sat |
| R1 fit + probe | 1 L4 | ~1 h | ~2 | Sat |
| Mac screen → strict gate → serving | Mac only | ~1 h + 7–9 h + ~1 day | 0 | after Path B, ~Oct 26–29 |
| 2-deviation probe (case c) | 1 L4 lane | ≤ 3 h | ≤ 6 | same day as the Round 0 verdict |

Negative-branch total ≈ 40–70 u, well inside the portfolio's L1/L2 lanes and the ~39 u/day spend floor. None of it
needs an A100, a G4 or a device port. Units are not the bottleneck: Sol time, the review loop and Mac occupancy are
(`REVIEW_C_adversarial.md:102-107`).
