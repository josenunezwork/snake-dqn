# REVIEW C — adversarial: what all ten Astra answers missed (2026-10-20 ~17:50 PT)

Reviewer: independent Fable (wrote none of the advice). Read: `FACTS_AND_QUESTION.md`, `astra_p1..p5` (10 answers).
Spot-checked on main 6f1e751f + artifacts (read-only; probes ≤ 30 s, 1 thread): `SUCCESS_CONTRACT.md`,
`FINAL_PLAN.md` §1–3, 8–10, `DESIGN_V2_SHADOW.md` Amendments 1–2, `research/phase0_search_20261019/PREREGISTRATION.md`,
`research/t34_round0/{README.md,PREREGISTRATION_ROUND0.md,REGISTRATION_ROUND0.json}`, `research/t34_shadow/RESULTS.md`,
`research/t34_pathb/{retention_v2.py,PREREGISTRATION_RETENTION_V2.md,RETENTION_OC_V2.json}`,
`docs/research/t12_b8_serving_qualification_2026-10-17.md`, live logs (Round 0 chunk 000, Phase 0), Mac hardware.

Where the ten agree, they are right: the A100 GPU is not a lever for any current stage (teacher inference ~13% exclusive;
fit optimizer arithmetic is minutes; A100 hosts report the same 12 vCPU as L4). The problems are in what *none* of them
asked. Ranked by how much wall-clock is at stake.

---

## Gap 1 — Every claim rests on one untested premise, and there is no NO-GO plan

**Evidence.** Round 0's acting teacher is `teacher_v2`, β=1, H=300, shadow D=300, *bisection radius 3*
(`PREREGISTRATION_ROUND0.md:35–36`). The only lever that ever beat B8 by a large margin is **T3-dense** (verify every
decision; 54/64 vs 38/64 scripted survival, +92 mass at 64 worlds — `FINAL_PLAN.md` §2). The bisection rule, measured
offline on T3's own trajectories, recovered **39/75 = 52%** of T3's interventions (M2 NO-GO, `t34_shadow/RESULTS.md`).
Round 0's GO needs scripted survival ≥ 0.66 **and** +0.10 vs bare **and** pooled mass ≥ +40 with LB90 > 0
(`PREREGISTRATION_ROUND0.md:112–119`). Half of T3's +0.25 survival gain is ≈ +0.13: the bar is within noise of the
precedent. Path B uses the same mechanism (earliest-first from shadow doom, Amendment 2 §2), so a Round 0 STOP sinks
Claim A **and** Claim B. The plan's only text for that case: "this teacher failed its registered bar"
(SUCCESS_CONTRACT note; `FINAL_PLAN.md:98` calls it "the designed off-ramp" and stops there).
All ten answers scheduled the GO branch only. The prompt's own question — "if Phase 0 or Round 0 says NO-GO, does any of
this matter?" — went unanswered: **no, none of it matters, and the fastest route after a NO-GO is undefined.**

**What to do (now, pre-data, costs nothing on the Mac).** Write the NO-GO branch before Wednesday's verdicts: the
fallback teacher is T3-dense acting online (dense verify at L ≥ 600, the mechanism with the +92 precedent), at the
plan's own 4–10× cost per world. That branch is purely host-core-bound and parallel across worlds, so its binding
hardware question is **usable host cores per runtime slot**, i.e. the G4 (48 vCPU reported) probe P5 proposed —
the only hardware measurement in the whole packet with option value (≈ 13 u). A100 is irrelevant on this branch too.

## Gap 2 — ~11 of the ~27 Mac hours on the Path B critical path are `time.sleep`

**Evidence.** `retention_v2.py:1286–1289`: every tick does `time.sleep(max(0, 1/fps − elapsed))`. Both the pre-check
(lines 1407–1416, `for arm in ("bare","paced")`, `fps=24`) and efficacy (1593–1601, `arm_fps = 24` for every arm in
`MAC_ARMS`) pace the **bare** arm too (`PREREGISTRATION_RETENTION_V2.md:219`: bare "paced at 24 fps"). Bare B8 s3 is
rowwise fp32 CPU with no controller and no clock dependence; pacing it measures nothing. Worlds run serially, one at a
time, on an 18-core M5 Pro (6P+12E, 64 GB) that must otherwise be "quiet". Bare exposure: pre-check 3.19 h (P4 computed
it from `RETENTION_OC_V2.json` and did not draw the conclusion) + efficacy 144 terminal × ~0.8 × 208 s + 32 Watch6 ×
208 s ≈ 8.5 h → **≈ 11–12 h asleep**. Unpaced bare worlds run at ~2.5 s/world amortized (T4c receipts, P3) → minutes.
P1 and P4 both did the 176×2×5000/24 arithmetic and neither noticed that half of it is the comparator sleeping.

**What to do.** The retention study has not started (waits for Phase 0), so "freeze before data" is satisfied: a
one-line pre-data amendment (bare arm `fps=0`, or for the pre-check reuse the bare outcomes already bound in the T3
receipts — `retention_v2.py:346` only requires the field, not a re-run), reviewed by one Fable judge **before Phase 0
ends** (ETA Wed 10:00–13:00). Saves ~10–12 h of the Mac chain: Path B verdict Thursday morning instead of Thursday
night/Friday. Caveat I could not settle read-only: whether the host-guard/contention bookkeeping assumes symmetric
arms; if so, run the bare arm unpaced as a separate phase, not interleaved. Lower-confidence second step: two paced
worlds concurrently (1 controller thread each on 6 P-cores) — affects deadline-miss statistics, leave to the judge.

## Gap 3 — The binding link for Claim A can be removed tonight on idle lanes, with data that already exists

**Evidence.** P4 (call 2) identified the un-built production pipeline as the most likely binding link (2–4 days):
`label.py:108–111` refuses student fan-out; manifest builder refuses nondefault student inputs; fit `ReplaySource`/probe
boundary unsupplied; CUDA fit unqualified. Yet every advisor scheduled fit-pipeline qualification "after R1 labels
exist" (P3's table literally starts there). **324 MB of real v2 teacher exports (39 A5 worlds, `a5-l4-s{0,1}-output/
exports/worlds/`) are on disk now**, and 2 of 3 Colab runtimes are idle tonight.

**What to do.** Sol: ReplaySource + probe integration and the student fan-out fix, against the A5 exports; then one
cache-build + CUDA fit + 32-world probe on an idle L4 tonight (~5–8 u). This takes the binding link off the serial
post-verdict path *and* produces the production fit-timing receipt that four of the five lenses wanted — for free.

## Gap 4 — "Measure first" here is five probes for a question whose answer everyone already gave

**Evidence.** Proposed probes: P1 ≤ 13.3 u, P2 ≤ 27 u, P3 ~2 u, P4 15–36 u, P5 13.2 + 12 u ≈ **≥ 90 u, 2 of 3 runtime
slots, and days of spec/review/acceptance attention** — while every answer's own prior is that A100 ≈ L4 for host-bound
work (P5: 10–20% prior of ≥ 1.25×; P1: "center near 1×"). None of these probes sits on the critical path; none changes
Wednesday's decisions. The review machinery itself is now the scarce resource P4 named: this question consumed 10 Astra
calls + 3 Fable reviews in one evening (third GPU-hardware round in three days: 10-19 GPU panel, 10-20 portfolio,
10-20 A100), and Round 0 / retention needed 3 and 6 FIX-FIRST rounds.

**What to do.** Zero A100 probes. Close the question with a written trigger instead: reopen only if (a) the first
production L4 fit receipt shows > 1 h GPU-bound time (DESIGN_V2 §Amendment 1 rule, already binding), or (b) a NO-GO
fallback multiplies search per world (Gap 1) — in which case the probe is G4 host capacity, not A100.

## Gap 5 — Nobody asked what a big GPU makes cheap that is *not* in the plan; the answer is still "nothing"

**Evidence.** All ten inherited "A100 for the existing workload". The two alternatives the brief omitted:
(a) **a larger/different student.** Serving headroom exists: measured mixed-Watch6 update p95 8.40 ms vs the 11.688 ms
contract bar (`t12_b8_serving_qualification:184`; `SUCCESS_CONTRACT.md` §4), B8 forward ≈ 0.5 ms → a 3–5× student fits
serving; T3's failure mode (99.6% bank agreement, lost by spillover) is at least consistent with a capacity/architecture
question, and `FINAL_PLAN` §3 already carries an S1a tower arm. But fits of this size are minutes on an L4 (M2b 23–29k
rows/s), so this is an **L4** case, not an A100 case. (b) **GPU-resident search** — P2's payback arithmetic is sound:
pays only at multi-million searched decisions, which no registered round needs.
Conclusion worth writing down so the question stops recurring: on this project there is no algorithm, student or
teacher variant for which one A100 beats three L4 lanes on wall-clock; the "no" is robust to the alternatives, not just
to the current workload.

## Gap 6 — Phase 0 is the de-facto NO-GO hedge, and nobody framed it that way

**Evidence.** Phase 0 occupies the Mac ~17 more hours for a **stochastic** N8/H128 teacher the plan superseded
(`FINAL_PLAN` §2: "distils noise"); its GO "authorizes planning only" (`PREREGISTRATION.md:71`); the plan keeps it only
"as a cross-check" (§10). No advisor asked whether to stop it (INVALID_INCOMPLETE) to free the Mac for Path B a day
early. My call: **keep it — but because a Phase 0 GO is the only evidence that more search per decision helps, i.e.
the hedge for a Round 0 STOP (Gap 1)**, not because the plan says so. In exchange, ratify the Gap 2 amendment before it
ends so the handoff loses no Mac hour (admission needs 120 quiet s; the operator must be present ~Wed noon).

## Gap 7 — "Spend all the units" is distorting the hardware question

**Evidence.** 39 u/day ≈ one L4 lane running continuously (1.54–2.0 u/h). Any plausible plan meets the spend target;
yet three of five lenses still sized their recommendations by unit caps, and the question itself arose from "units
aren't scarce, so why not A100?". Units cannot buy the critical path (Mac occupancy, engineering readiness, premise
risk). Record the owner's spend goal as a floor on useful L4 work, never as a reason to pick a pricier runtime.

---

## Answer to the open question (≤ 12 lines)

1. **No.** A Colab A100 shortens nothing: the teacher is host-bound on the same 12-vCPU host as an L4, fits are minutes
   on an L4, and the stages that bound the calendar (Mac occupancy, pipeline readiness, premise risk) are not GPU stages.
2. Do **no** A100 measurement now; write a reopen trigger (production L4 fit GPU-bound > 1 h, or a dense fallback teacher).
3. What *would* shorten time to a better agent, in order of hours saved per hour of effort:
   a. Pre-data amendment: bare arm of the Path B pre-check/efficacy unpaced (`fps=0`) — ~10–12 Mac-hours off the critical
      path, ratify before Phase 0 ends Wednesday.
   b. Tonight, on the two idle L4 lanes: qualify ReplaySource/probe/CUDA fit and student fan-out against the 39 A5 export
      worlds — removes the 2–4-day Claim A link from the post-verdict path and yields the fit-timing receipt for free.
   c. Write the Round 0 / Path B NO-GO branch now (T3-dense acting teacher, host-core-bound → G4 host probe, ~13 u, is the
      one hardware measurement with option value).
   d. Stop running advice rounds on hardware; the review loop is consuming the Sol/operator hours that bind Claim A.
4. Earliest credible better agent on the current path: Claim B Oct 25–26 with (a); Claim A Oct 26–28 with (b). Unchanged by A100.

## Not verified by me
Third-party A100/G4 host specs (12 / 48 vCPU) — repo T16 records agree for A100; whether the retention registration's
host-guard semantics permit an asymmetric (unpaced bare) arm without touching paced-arm rules; the ten answers' internal
file:line citations beyond those listed in the header. Live checks at ~17:41 PT: Round 0 chunk 000 at 26/100 pairs after
53 min (~3.4 h pace, consistent with the handoff); Phase 0 progressing on frozen-mix selection worlds.
