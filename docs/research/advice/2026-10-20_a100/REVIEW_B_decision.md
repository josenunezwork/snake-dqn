# REVIEW B — the decision (independent Fable reviewer, 2026-10-20, Mac clock 17:40 PT at review time)

Read-only review of the five Astra answers (P1 systems, P2 GPU search, P3 distill, P4 critical path, P5 Colab ops;
each = call 1 + cross-examination) against `FACTS_AND_QUESTION.md`, HANDOFF_2026-10-20, CAMPAIGN_LOG,
SUCCESS_CONTRACT, P7, DESIGN_V2 Amendment 2, the compute-portfolio judgment, and the code/receipts where the
advisors' points are decided. I wrote none of the advice. Probes: `sed`/`grep`/one 1-thread `nice -n 10` JSON parse of
the chunk-000 supervisor log; nothing launched, nothing edited outside this file.

## 0. Live state I verified (anchors every date below)

| Item | Verified now | Consequence |
|---|---|---|
| Round 0 chunk 000 | Supervisor started 16:46:45; log polls show 30 pair-files at 1,936 s (= 15 pairs, matches the handoff) and **48 pair-files at 2,562 s = 24 pairs at 42.7 min** (2 files/pair). Registered est 166 min, timeout 332. | Pace ≈ **3.0 h per 100-pair chunk**, slightly *better* than the handoff's 3.5 h and ~8% over the frozen 166-min forecast. Chunk 000 ends ≈ **19:45–20:15 PT**. |
| Phase 0 | 101 result files (~100/160 worlds); 86→100 between 13:39 and 17:30 ≈ 3.6 worlds/h. | Mac frees **Wed 10-21 ~10:00–13:00 PT**, as the handoff says. |
| tmux | `git-sync`, `phase0-selection`, `t34-round0-c000` only. Sol test-hygiene job running (pid 63605). | Two Colab assignments are idle; nothing else is using them. |
| Code claims the decision rests on | `research/t34_exit_b8/label.py:108–111`: `nproc > 1` with `policy_factory`/`student_checkpoint` → `ValueError` (student fan-out unsupported). `research/t34_exit_b8/README.md:110–112`: manifest builder refuses nondefault checkpoint inputs. `research/t34_exit_b8_fit/README.md:28–31, 46–50`: CUDA path unqualified; no production v2 export schema or `ReplaySource`. `research/t34_pathb/retention_v2.py:1380–1425`: pre-check runs all 64 rows × both arms paced at H5000 / 24 fps. `loss.py:43–60`: KL has no τ² factor; hinge/adv-MSE masked only on `kl_anchor_only` rows. | P4's "the R1→R2 production pipeline is not built" is **true in code**. The "~1.5 h pre-check" in Amendment 2 §3(a) is **wrong** for the ratified runner (≥ 2.9 h for 50 surviving paced cases alone; ~6–7 h expected). P3's loss objection is **real as written**. |
| Contract text | SUCCESS_CONTRACT §5: "Path B deadline-feasibility check runs before R1/R2 label rounds"; DESIGN_V2 Amendment 2 §3 "no label rounds funded before it" but §4 "Teacher label rounds (R1+) wait for (b) **or** Round 0"; STATUS.md ranks SUCCESS_CONTRACT above DESIGN_V2. SUCCESS_CONTRACT §4 "owner OK required for release" vs STATUS.md "release/serving changes → one independent Fable judge". | Two prospective rulings needed (§3 below). |

## 1. Where the five advisors agree (I concur on all five; evidence checked)

1. **The A100 GPU is not a lever for any stage that exists today.** Teacher/Round 0/labels are host-bound: exclusive
   B8/opponent inference is ~7.5–7.8% of worker wall in the A5 receipts and 13.3% in the CPU-only diagnostic (P1 call 2);
   even removing it entirely gives ≤ 1.15×. Phase 0 is already running on the Mac. Paced retention, strict gate and
   serving qualification are Mac-only by contract. (P1, P2 call 2, P3, P4, P5 all say this; the portfolio already
   excluded A100/G4/H100 for host-bound work.)
2. **An A100 runtime's host is not known to be faster than an L4's.** The project's own T16 record
   (`docs/research/t16_colab_smoke02_results_2026-10-17.md:4, 210, 497`) shows A100-40 = 12 vCPU / 6 physical,
   A100-80 = 12 vCPU, L4 = 12 vCPU; two third-party 2026 observations agree (P5). The "unmeasured" in the facts brief
   should read "unmeasured for a *new* assignment". P5's prior for ≥ 1.25× host gain on the unchanged teacher: 10–20%.
3. **The only plausible A100 mechanism is the cached student fit**, and its production duration is unmeasured
   (16-row synthetic smokes only; `gpu_busy_percent` never sampled). P3 call 2's single-thread loader probe
   (14.4 µs/row) plus the historical M3-C 30 vs 80 upd/s suggests A100 ≈ 1.3–1.8× on the loader+optimizer loop; P1
   estimates optimizer arithmetic at minutes, not hours; M2b receipts (23–29k rows/s) and T4c bare screens (160 H5000
   worlds in 6.7 min on the Mac) show the portfolio's 3 u/fit and 30 u/screen allowances are oversized.
   DESIGN_V2:70–72's rule ("time a cached L4 fit; A100-40 only if GPU-bound beyond ~1 h") stands.
4. **Calendar is bound by the Mac sequence and by unbuilt engineering, not by Colab compute.** Claim B: Mac admission
   Wed → timing + ~6–7 h pre-check + ~17–21 h retention ≈ 27–28 h → verdict **Thu 10-22 afternoon/evening** (Fri if
   caps are approached), then an unsized paced strict gate. Claim A: Round 0 verdict (Wed) + the unbuilt pipeline
   (P4: 2–4 Sol days, 3–5 with one GPU/replay failure) + ~20 h of labels/fits/screens + a 7–9 h strict gate.
   A100 touches none of the binding links. P4's revised windows (Claim A Oct 26–29, Claim B Oct 25–28 only with a compact
   powered design) are the honest ones; its call-1 "Oct 24–26" was withdrawn and should not be quoted.
5. **A GPU-resident search port does not pay back on the registered rounds.** P2 call 2's break-even is ≈ 206·D worlds
   (D = critical-path engineering hours); at W = 2,400 worlds (R1+R2+R3) no finite speedup clears a 48 h delay. Only a
   multi-million searched-decision workload (e.g. searching every long-length decision, ~1.7M/round) would.

## 2. Where they disagree, and the resolution on the merits

| # | Disagreement | Positions | Resolution (why) |
|---|---|---|---|
| D1 | **Which single A100 measurement first** | P1: matched 48-world teacher run A100-40 vs L4 (≤ 13.3 u). P2: 96-world L4 vs A100 CPU-best **plus a device-resident prototype** (~27 u; needs code that does not exist). P3: matched L4/A100-40 phase-resolved fit packet on a ~65k-row cache (~2 u, 30–40 min). P4: matched *complete production* cached fit (15–30 u). P5 (after cross-exam): **withdraw** the A100 teacher test; profile the first real L4 fit; A100 fit packet only if the L4 profile shows ≥ 30–40% accelerable time; probe **G4** host capacity instead. | **P5/P3 win.** A teacher placement test has a low prior (same 12 vCPU) and labels are not on the critical path for days. P2's prototype is engineering, not a measurement. P4's "complete production fit" cannot exist until the pipeline is built. The one defensible A100 measurement is the fit packet, and it is only worth buying if the L4 profile (a by-product of the already-funded CUDA fit qualification, portfolio L2 ≤ 10 u) shows a GPU leg large enough to matter. Verdict in §4. |
| D2 | **Does A100 earn any calendar credit now?** | P1 call 1: "not a demonstrated shortcut"; call 2: worth a host comparison, switching rule ≥ 1.5×. P4: zero credit until measured. P5: adopt at ≥ 1.25× end-to-end if it moves the next dependent result. | **Zero credit** (P4) for scheduling; P5's 1.25× is the right *adoption* threshold once a measurement exists, because units are prepaid and the scarce things are wall-clock and the 3-assignment cap (unit break-even 2.7×/3.75× is economics, not admission). Both are compatible: credit nothing, adopt at ≥ 1.25× measured. |
| D3 | **Fit allowance 4 h: real or not?** | P1 call 1 and P4 call 1 used 4 h fits in the chain; P1 call 2 and P3 call 2 withdrew it (allowance, optimizer minutes; M2b 67–141 s/epoch). | 4 h is a cap, not a forecast. Plan the Claim A compute chain at labels ≈ 5 h/round on 3 lanes + fits ≈ tens of minutes + screens ≈ 10–20 min (T4c evidence). This *strengthens* the conclusion that engineering, not compute, binds Claim A. |
| D4 | **Pre-check duration** | Amendment 2 and P1 call 1: ~1.5 h. P1 call 2 and P4: ≥ 2.9 h for survivors alone, ~6–7.4 h. | **6–7 h** (code verified, §0). Claim B verdict moves from "Thu midday" to **Thu evening / Fri**. No relabel: the 8 h cap is registered and covers it. |
| D5 | **Round 0 slack** | P1/P4 call 1: 7–12 h slack. P4 call 2: slack exists only with staffed overnight handoffs. | Measured pace is 3.0 h/chunk, so with Claude staffing the two publication barriers tonight all three chunks are terminal ≈ **01:30–02:30 Wed**; verdict needs the Mac replay (24 pairs, trace replay, no NN forward) which fits in the gap after Phase 0 exits and before quiet admission → verdict **Wed ~11:00–14:00**. If handoffs slip to morning, verdict slides to Wed late afternoon; still before the retention disposition, so Round 0's *verdict* is not on the pre-Friday critical path unless Path B fails early. The 48 h rule is a dispatch deadline (Thu 16:45), not a verdict deadline. |
| D6 | **G4 for host-bound labels** | P5: probe it (48 logical CPUs reported by two third-party sources + GCP `g4-standard-48` spec; cu128 bootstrap already present; ~13.2 u). Portfolio: excluded. Others: silent. | **Worth one bounded probe, but only after Round 0 GO** (it informs R1/R2 lane placement and the unpaced retention arm; on STOP it decides nothing). Needs a timing-only registration and a Fable ruling amending the portfolio exclusion. It is the better host-capacity question than A100; adopt only at ≥ 2× sustained useful throughput at K ≤ 48 with valid outputs (P5's rule). Caveat P5 found in code: `supervisor.py:1351` admits on `os.cpu_count()` not quota, and `benchmark.py:298` caps workers at `max_worlds` → use ≥ 48 timing worlds. |
| D7 | **Label-round ordering vs Path B** | P4: contract §5 "before R1/R2" is binding; Amendment 2 §4 "or Round 0" is a textual conflict; overlap could save 1–2 days. | Conflict confirmed (§0). Practical stake is smaller than 1–2 days because the pipeline is not built: a Round 0 GO Wed + labels Thu would still wait for Sol. Still worth a **prospective ruling now** (free, and it must land before Round 0's `decision.json` is produced). See §3. |
| D8 | **Numeric CPU simulator state contract as "first systems investment"** (P2 call 2) | P2 only. | Decline for this campaign window: Sol days are the scarce resource and the binding Claim A link is the production pipeline (D1/§1.4). Record as a post-campaign item. |

## 3. Contract / registration conflicts needing a prospective ruling (no relabelling)

1. **R1/R2 label collection vs Path B disposition.** SUCCESS_CONTRACT §5 says the Path B check "runs before R1/R2
   label rounds"; Amendment 2 §4 says labels "wait for (b) **or** Round 0"; the retention registration (KILL → "teacher
   labels then wait on Round 0") reads with the contract. Ask one Fable judge, **before Round 0's sole analysis is run**,
   for a prospective ruling on exactly this: *may R1 label collection (Colab, non-Mac) be dispatched on a published
   Round 0 GO while the Mac retention phases are still running, with fits/screens still gated on the Path B
   disposition?* If the judge finds this is an amendment of owner-ratified text rather than an interpretation, it
   goes to the owner; either way record it in CAMPAIGN_LOG before the verdict is opened.
2. **Release approval authority.** SUCCESS_CONTRACT §4 and the owner's 2026-10-20 delegation ("releases still need
   OK") keep release with the owner; STATUS.md's authority block says release/serving changes go to a Fable judge.
   Record that the owner's OK governs releases and fix STATUS.md's sentence (operator edit, not a relabel).
3. **Amendment 2 §3(a) "~1.5 h Mac" for the pre-check** is an estimate inconsistent with the ratified runner (6–7 h).
   Do not touch the registration; add a dated note to the campaign log so nobody schedules Thursday around 1.5 h.
4. **Claim B strict-gate design is unregistered and unsized** (P7 gives only a 16/mix floor; the historical
   `sequential_gate.py` pools survival across mixes by index, which P7 forbids without covariance-aware pooling;
   product-session bars are new). Its skeleton can and should be frozen *now* (P4 §6); its N must come from the
   retention variance, so the numeric N is filled in the day retention finishes — still pre-data for the gate.
5. **Portfolio exclusion of G4/A100 for host-bound work** stays in force until a Fable ruling admits the specific
   G4 timing-only probe (D6). The A100 fit packet (§4) is *inside* the portfolio (L2 "CUDA fit qualification") and
   DESIGN_V2:70–72, so it needs no exclusion ruling, only the usual registration.

## 4. Verdict on A100 runtimes

**Do not use A100 runtimes for anything running or scheduled this week (Round 0, the unpaced retention arm, R1/R2
labels, screens).** Measured mechanism: none (host-bound, same 12-vCPU host class). Calendar credit: zero.

**Measure first — one measurement, conditionally triggered, for the fit stage only:**

- **Trigger (free, from work already funded):** the first production-shaped cached fit on L4 — the CUDA fit
  qualification in portfolio lane L2 (≤ 10 u), run on the first real R1 cache (or, if the ruling in §3.1 allows no
  labels yet, on the Round 0 export replayed through the fit adapter as a *timing-only* input, outcomes unopened). It
  must report the phase decomposition P3/P5 specify: cache build/hash, loader, H2D, fwd/bwd/Adam, held-out scoring,
  32-world probe, export; and sample GPU busy (currently always `None`).
- **Pre-registered rule:** if (whole-fit wall ≥ 60 min) **and** (fwd/bwd/Adam + held-out forward share ≥ 40%), run
  **the measurement**: a matched L4 vs A100-40 phase-resolved fit packet (P3 call 2 design: same cache ≥ 65k rows,
  identical checkpoint/recipe/batch order/fp32, 1,024 identical updates in four 256-update blocks, cold cache
  reopen on both, correctness hashes). **Unit cap 12 u** (2 u/h L4 + 8 u/h A100 ceilings, 30-min target, 60-min
  timeouts, 2 u reserve); two assignments for ≤ 1 h; incomplete = inconclusive, no extension.
  **Adopt A100-40 for fits** iff whole-packet speedup ≥ 1.25, all four blocks favour A100, correctness passes, and the
  phase model predicts ≥ 1 h saved across the fits admitted in the current round. **Reject** (A100 closed for this
  campaign) if speedup ≤ 1.10 or the trigger never fires. Between: keep L4, do not re-test.
- Expected outcome, stated now so it can be scored later: P1's arithmetic (337 TFLOP of S1b work ≈ minutes) and the M2b
  receipts make it **more likely than not that the trigger never fires** (fit < 1 h on L4). That is the cheapest
  possible closure of the A100 question and costs nothing beyond the already-funded qualification.

**A100 for the teacher (P1's 13.3 u matched run):** rejected — low prior, no critical-path consumer this week, and
if host capacity is the question G4 is the right probe (D6). **A100 via a GPU-resident search port (P2):** rejected for
this campaign (§1.5).

## 5. Ordered actions that most shorten calendar time to a strictly-gated better agent (from tonight)

Legend: **Claude** = operates/specs/reviews; **Sol** = implements; **Mac** = owner's Mac; **L4** = Colab L4 lane.
"∥ Phase 0" / "∥ Round 0" = can run alongside the running Mac study / the serial Round 0 lane.

| # | Action | Who / where | When | Parallel with | Why it moves the date |
|---|---|---|---|---|---|
| 1 | **Staff Round 0's two publication barriers tonight**: publish chunk 000 ledger when `EXIT=` lands (~19:45–20:15), dispatch 001 within 15 min, publish, dispatch 002 (40 pairs, ~1.2 h), publish. Do not open outcomes. | Claude, L4 lane (registered, ~13.5 u expected) | Tonight → ~01:30–02:30 Wed | ∥ Phase 0 | Makes the Round 0 verdict available Wed midday instead of Wed evening/Thu; removes the one place overnight idleness costs a day (D5). |
| 2 | **Ask one Fable judge for the §3.1 ruling** (label collection on Round 0 GO vs Path B disposition) and record the §3.2 release-authority note. | Claude → Fable judge; log in CAMPAIGN_LOG | Tonight, before any Round 0 analysis | ∥ both | Freezes the sequencing *before* data; worth up to a day on Claim A if the pipeline is ready first. |
| 3 | **Spec the production Claim A pipeline for Sol** (one spec, one worktree, no `colab_cli` changes so no re-acceptance): (a) student fan-out in `label.py` (per-process student load; drop the `nproc>1` + adapter refusal), (b) manifest builder SHA-binding of a published student checkpoint, (c) production v2 export schema → `ReplaySource` + previous-student Q + registered 32-world probe hook, (d) CUDA fit qualification harness with the §4 phase timers and GPU-busy sampling, (e) Mac/L4 replay identity on ≥ 150 worlds × 3 mixes (FINAL_PLAN §7 A2). Review loop as usual. | Claude spec tonight; Sol 2–4 days; tests on Mac at ≤ 2 threads (compute policy) | Tonight → target ready Fri 10-23 | ∥ Phase 0, ∥ Round 0, ∥ retention | This is the binding Claim A link (P4, verified in code). Nothing on Colab can start R1 without it. |
| 4 | **Freeze the Claim B strict-gate pre-registration skeleton** (P4 §6): one frozen assisted controller vs B8 s3, real 24 fps pacing both arms, fresh disjoint banks per mix + complete Watch6 session pairs, fixed-N default, covariance-aware pooled survival per P7, every contract bar incl. timing; N to be filled from retention variance; caps and audit interface. Exposure formula T = 2(3N+S)·5000/24/3600 h (48/32 → 20.4 h; 96/64 → 40.7 h). | Claude spec; Sol writes the registration + runner deltas vs `sequential_strict_template`; independent review | Tonight/Wed → ratifiable the hour retention finishes | ∥ everything | If retention is GO Thu/Fri, the strict gate can start the same day instead of after 1–2 days of design/review. This is the largest single Claim B saving. |
| 5 | **Phase 0 close-out + Round 0 verdict in the Mac gap**: when Phase 0 exits (~Wed 10–13 PT) run its analyze (cross-check only), then the Round 0 10% replay (24 pairs/48 traces) → publish → sole analysis → publish → record. Then quiet admission → timing gate → pre-check (plan 6–7 h) → retention (~17–21 h). | Claude, Mac (frozen worktree `snake-dqn-round0-frozen-01`, retention from a worktree at 97232d51) | Wed ~10:00 → Thu evening (Fri if caps) | Round 0 replay is before quiet admission; nothing else on the Mac | Keeps the Mac sequence tight; the replay cannot overlap scored timing, so do it first. |
| 6 | **Dispatch the unpaced retention arm on Colab** (merged, 42.5 u cap, descriptive) on the assignment freed by Round 0, as soon as `research/t34_pathb/README_V2.md` admits it (it is Colab-only and cannot affect the Mac verdict). | Claude, L4 | Wed, after chunk 002 is terminal (or on a second assignment if the registration permits earlier) | ∥ retention | Spends units on a registered lane; its attribution feeds portfolio L4 (search successor). |
| 7 | **G4 host-capacity probe** (D6): timing-only registration + Fable ruling; ≥ 48 frozen timing worlds across mixes, K = 6/12/24/48, one matched L4 control, CPU model/quota/throttling recorded; **cap 13.2 u**; adopt at ≥ 2× sustained useful throughput with valid outputs. | Claude registers; Fable judge admits; Claude dispatches on the second spare assignment | **Only after a published Round 0 GO** (Wed afternoon); skip on STOP | ∥ retention, ∥ Sol #3 | Decides R1/R2 lane placement (~2 h/round at 3×) before labels are admitted; also the only cheap test of "more host per assignment slot". |
| 8 | **R1 labels on 3 L4 lanes (or G4 + 2 L4 if #7 passed)** ≈ 5 h, then the L4 cached fit with the §4 phase timers (= the A100 trigger measurement), screens, then R2 (β = 0.5) ≈ 5 h + fit + screen. | Claude operates; Colab | Earliest Fri/Sat 10-23/24, conditional on Round 0 GO + #3 qualified + the §3.1 ruling (else after retention disposition) | ∥ retention / strict gate on the Mac | Claim A compute chain ≈ 20 h once admitted; A100 packet only if the trigger fires (§4). |
| 9 | **Strict gate + serving qualification on the Mac** for whichever claim qualifies first (bare student wins unresolved ties). Owner OK for release. | Claude, Mac | Claim B: Fri/Sat → Oct 26–28 earliest; Claim A: Oct 27–30 earliest | — | Mac-only by contract; no accelerator applies. |

Not on the list, deliberately: any A100 runtime this week; a GPU-resident search port; the numeric CPU simulator
contract (post-campaign); re-sizing or re-ordering any registered phase; extra Phase 0/Round 0 looks.

**Earliest credible dates (conditional on every bar passing):** Claim B strict-gated hero **Oct 26–28** if retention
is GO Thu/Fri and #4 is ratified in time; Claim A bare student **Oct 27–30**. The single change that moves either
date most is #3/#4 — doing the engineering and the strict-gate design *during* the 40–50 h of Mac-bound waiting that
starts now. A100 moves neither date.

## 6. Unit plan for the next ~3 days (owner goal ≈ 39 u/day on decision-producing work)

Round 0 remainder ~9 u (Tue night) · unpaced retention arm ≤ 42.5 u (Wed–Thu) · G4 probe ≤ 13.2 u (Wed/Thu, on GO) ·
CUDA fit qualification ≤ 10 u (Fri, needs #3) · R1 labels ~29 u + conditional A100 fit packet ≤ 12 u (Fri/Sat).
≈ 100–115 u over Tue–Sat, every unit attached to a registered decision; no idle allocations; no A100 unless §4 fires.

## 7. Confidence and what would overturn this review

- High: A100 has no mechanism on the teacher/labels/Mac stages (receipts + T16 host record + code). Would be overturned
  only by a measured A100 host ≥ 1.5× on the frozen K12/W1 teacher — which I am declining to buy because G4 is the
  cheaper host-capacity probe and no consumer exists this week.
- Medium: the fit trigger never fires (fit < 1 h on L4). If it fires, the 12 u packet decides it; adoption saves at
  most ~0.5–1 h per fit wave, i.e. hours, not days.
- Medium: Claim A pipeline 2–4 Sol days. History (retention: protocol→ratified 13 h; Round 0: package→ratified 4.4 h)
  says Sol can be faster; FINAL_PLAN's "7–9 prep days" says slower. Either way it, not compute, binds Claim A.
- The chunk-000 pace (3.0 h) is from 24/100 pairs; the registered 166-min forecast may still be ~8% optimistic. No
  action depends on it beyond staffing the barriers.
