# Astra P4 — critical path to a released better agent (A100 question, 2026-10-20)

Lens: the full chain from today (Phase 0, Round 0, retention study, distillation rounds, Mac strict gate,
serving qualification, release) and which links bound the calendar. Two Astra calls (`CODEX_MODEL=gpt-6-astra`,
effort high, fast mode) in the read-only worktree `snake-dqn-advice-a100-4` at main 6f1e751f.
Raw outputs: `snake-dqn-artifacts/a100-advice/p4-call1/`, `.../p4-call2/`.

## Call 1 — prompt (verbatim)

```text
You are advising the snake-dqn project (repo at your working directory, a read-only worktree of main at 6f1e751f). Read `docs/research/advice/2026-10-20_a100/FACTS_AND_QUESTION.md` first, then the sources it names: `docs/research/final_run_20261019/HANDOFF_2026-10-20.md`, `CAMPAIGN_LOG.md`, `FINAL_PLAN.md` (sections 5, 6, 8, 9), `SUCCESS_CONTRACT.md`, `P7_STATISTICS.md`, `DESIGN_V2_SHADOW.md` (Amendment 2), `docs/research/advice/2026-10-20_compute_portfolio_FABLE_JUDGMENT.md`, `docs/research/advice/2026-10-20_t34_a5_speed_{astra,fable}.md`, `research/t34_round0/` (README, REGISTRATION_ROUND0.json, sizing), `research/t34_pathb/` (PROTOCOL, REGISTRATION_RETENTION_V2_RATIFIED.json), `research/phase0_search_20261019/README.md`, `docs/STATUS.md` ("Campaign authority"). Also the strict-gate history: `docs/research/b8_s3_release_2026-10-17.md`, `docs/research/t12_b8_serving_qualification_2026-10-17.md`, `docs/research/t7_phase_r_result_2026-10-16.md`, and `research/sequential_strict_template/` for what a strict gate actually costs on the Mac.

HARD RULES: read-only. Do not edit, commit, create or delete any file in the repo or the worktree. At most tiny read-only probes on the Mac (1 thread, `nice -n 10`, each < 30 s); the Mac is running a 6-worker Phase 0 study plus a Round 0 Colab supervisor that must not be disturbed. No Colab, RunPod, or network jobs; no pip installs. Put your FULL answer in your final message (there is no other channel), with file:line evidence and numbers, and label every claim as VERIFIED (you read it in a file or measured it) or ESTIMATED (your inference).

GOAL. A better served snake agent than the current Watch hero (bare B8 seed 3, released 2026-10-17), proven by the usual strict gate on the Mac and a serving qualification, as soon as possible in wall-clock time. Colab units are prepaid (~1,157 u, target to spend by ~2026-11-20) and are not the scarce resource; calendar days are.

FACTS (today 2026-10-20, ~18:00 PT; verify in the repo).
- Approved path T34: exact-sim rollout search teacher, gated to long/low-margin states, then DAgger-distill into the same B8 CNN for 3–5 rounds (R1/R2/R3 in FINAL_PLAN §5–6), serve the bare student via the strict gate (Claim A); fallback Claim B = served asynchronous lookahead ("Path B"), decided by a paced retention study on the Mac. Every bare-network lever since B8 (fine-tunes, league T23, S1, M-series) was null or negative.
- Running now: Phase 0 premise test on the Mac (6 CPU workers, ~100/160 worlds, ETA Wed 10-21 ~10:00–13:00 PT). Round 0 (teacher acting vs bare B8, 240 paired worlds, 3 chunks run SERIALLY on one Colab L4 lane; chunk 000 started ~16:55 PT at ~3.5 h pace; ratified with a 48 h dispatch deadline ~10-22 16:45 PT; no analysis before all chunks are terminal; then a 10% Mac replay, publication, sole analysis). Path B paced retention study is ratified, Mac-only, ≤ 40 h cap plus a timing gate and an 8 h-capped pre-check, starts only after Phase 0 finishes and the Mac is quiet.
- Each registration went through an independent review loop (reviewer → FIX-FIRST rulings → Sol implements → re-review → RATIFY → Fable approval judge). Round 0 took 3 rounds in ~1 day; the retention prereg took 6 FIX-FIRST rounds in ~2.5 h. Sol (Codex) implements; Claude specs/reviews/operates; releases need the owner's OK.
- Measured speed: teacher/benchmark work is ~95% single-threaded CPython; the L4 GPU sits at 2–8% util; CUDA batch-1 forward was slower than CPU (4.35 vs 0.93 ms). After the engine fan-out fix: 57.2 / 54.4 completed worlds/h per L4 lane (12 single-thread processes), ~0.036 u per teacher world. An 800-world label round ≈ 14–15 lane-hours (≈ 5 h on 3 lanes) ≈ 29 u. Phase 0 teacher ~4.5 s per searched decision on 1 Mac core before a 3.4–4.2x speedup; 1M searched decisions ≈ 129 Mac-hours. Student fit: one L4, streaming loader, ≤ 30 min × 5 epochs budgeted (FINAL_PLAN §5). Colab: max 3 concurrent runtimes, ≤ 24 h each; A100 host vCPU count/speed unmeasured. Strict gate ≈ 5–9 h on 2 Mac slots, Mac-only; screens ~1 h per 96/32/32 bank on the Mac; serving qualification Mac-only.
- FINAL_PLAN §6 wall-time estimate (written 10-19): Round 0 Oct 29; R1 Oct 30–31; R2 Nov 1–2; R3 Nov 3–4 (conditional); strict gate Nov 5; serving qualification Nov 6; verdict ≈ Nov 6–8. Prep has since run ahead of that schedule (Round 0 dispatched Oct 20).
- Rules: strict gates and serving qualification stay on the Mac; freeze designs before data; never relabel a registered gate; the Mac is one machine shared by Phase 0, the retention study, replays, screens, the strict gate and serving qualification.

OPEN QUESTION (answer through the critical-path lens). Trace the full chain from this moment to a released better agent — Phase 0, Round 0, the retention study, the distillation rounds (labels, fits, screens), the strict gate on the Mac, serving qualification, release — and work out which links actually bound the calendar time. For each link, say what it waits on (compute of some kind, a Mac-only resource, a statistics/sample-size requirement, a review/ratification loop, a human decision, serial dependencies between studies) and how long it takes, with evidence. Then answer: is compute of any kind (Colab L4, a Colab A100, the Mac's CPU) on the critical path at all, and if so where exactly and by how many hours; and what would actually make the better agent arrive sooner in calendar days — what is the earliest credible release date on the current path, what single change moves it most, and what does an A100 runtime do to that date (if anything) given that the host vCPU count and per-core speed of an A100 runtime are unmeasured. Be concrete about hours and dates (Pacific time), and be honest about what you could not verify.
```

## Call 1 — Astra answer (verbatim)

**ESTIMATED — Compute is on the critical path, chiefly as occupied Mac time and sequential teacher-label work. An A100 has no demonstrated release-date benefit at this snapshot. I would target October 24–26 PT for an aggressive, successful current-path release, with October 24 as the earliest planning date—not a promised outcome. The largest unresolved scheduling risk is the still-unsized strict confirmation of an assisted controller, followed by unfinished production integration and review.**

**VERIFIED — Scope:** I inspected clean, detached commit `6f1e751f4ec11c280c211634cd09de924b649218`. This was read-only: no files changed, tests or numerical probes run, network access, allocations, or process interventions. “VERIFIED” below means verified in this repository snapshot, not independently observed live. All future durations and dates are **ESTIMATED**, including forecasts quoted from files.

**VERIFIED — The approved route branches before distillation.** The success contract requires the Path B deadline-feasibility study **before R1/R2 labels**. A passing Path B proceeds toward the Mac strict gate; distillation becomes optional. If Path B loses its benefit under deadlines, Claim B closes and labels proceed only on Round 0 GO. Phase 0 is a cross-check with its own rules, not a scientific prerequisite that can overrule Round 0. [SUCCESS_CONTRACT.md:59–68](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-4/docs/research/final_run_20261019/SUCCESS_CONTRACT.md:59), [P7_STATISTICS.md:24–26](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-4/docs/research/final_run_20261019/P7_STATISTICS.md:24)

**ESTIMATED — The relevant dependency graph is therefore:**

```text
Phase 0 finishes ──→ quiet Mac: timing → T3 pre-check → paced retention
                                                        │
                          ┌─────────────────────────────┴───────────────────┐
                          │ Path B GO                                      │ Path B fails
                          ▼                                                ▼
                 freeze assisted finalist                    Round 0 GO required
                          │                                                │
                          │                                  R1 labels → fits → screen
                          │                                                │
                          │                                  R2 labels → fit → screen
                          │                                                │
                          │                                    R3 only if admitted
                          └──────────────────────┬─────────────────────────┘
                                                 ▼
                              fresh Mac strict + product evidence
                                                 ▼
                                  Mac serving qualification
                                                 ▼
                                         release approval

Round 0: serial Colab chunks → Mac replay → publication → sole analysis
          runs alongside Phase 0; its completion does not itself release anything
```

**VERIFIED — Three corrections materially change the arithmetic.**

- The stochastic **N8/H128 Phase 0 teacher is not the current T34 shadow teacher**. Amendment 2 changes the teacher’s label/search rules and gives Path B a D500 earliest-first controller. The **129 Mac-hour** figure is a six-worker forecast for one million searches with the Phase 0 implementation; it is not the cost of an 800-world T34 label round. [DESIGN_V2_SHADOW.md:74–104](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-4/docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:74), [SPEED_REPORT.md:49–66](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-4/research/phase0_search_20261019/SPEED_REPORT.md:49)
- The concrete funded learning sequence is **R1 and R2, with R3 conditional**. R1 is non-terminal unless a tripwire fires. The inspected plan does not require completing five rounds before considering release. [FINAL_PLAN.md:55–68,89–91](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-4/docs/research/final_run_20261019/FINAL_PLAN.md:55)
- **Thirty minutes × five epochs is a replay-regeneration budget, not a measured complete fit duration.** The newer loader caches regeneration once; CUDA remains unqualified in its verification record. The budget table allocates four lane-hours per fit. [FINAL_PLAN.md:49–61](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-4/docs/research/final_run_20261019/FINAL_PLAN.md:49), [fit README.md:8–31](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-4/research/t34_exit_b8_fit/README.md:8)

| Link | VERIFIED dependency and evidence | ESTIMATED duration and calendar significance |
|---|---|---|
| **Finish Phase 0** | Latest handoff records 100/160 worlds, six Mac workers, ETA October 21, 10:00–13:00 PT. Retention waits for Phase 0 and quiet-host admission. [Handoff:155–157](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-4/docs/research/final_run_20261019/HANDOFF_2026-10-20.md:155) | **16–19 hours remaining from October 20, 18:00.** This is the first binding Mac resource dependency. Faster Colab hardware does not finish the already-running frozen Mac study. |
| **Round 0 Colab execution** | Three serial chunks, **100/100/40 pairs**, with publication between admissions. Registered estimates are **166/166/76 minutes**; timeouts **332/332/152 minutes**. Latest observed first-chunk pace was about 3.5 hours. [Registration:871–934](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-4/research/t34_round0/REGISTRATION_ROUND0.json:871), [runbook:124–137](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-4/research/t34_round0/README.md:124) | Registered total **6.8 hours**, versus roughly **8–9 hours** extrapolating the first chunk’s pace, including some setup but excluding unpredictable operator gaps. With prompt handoffs, cloud collection finishes roughly **October 21, 01:00–03:00**. That leaves about **7–12 hours of slack** before Phase 0 releases the Mac. |
| **Round 0 replay and verdict** | All chunks must be terminal before the sole analysis. A deterministic 10% sample per chunk is replayed on the Mac, both arms: **24 pairs / 48 traces**. Replay reconstructs metrics from exported action traces; it does not rerun teacher search. [Runbook:198–225](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-4/research/t34_round0/README.md:198), [replay.py:114–149](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-4/research/t34_round0/replay.py:114) | **No measured full replay duration verified.** Reserve an explicit Mac interval around the retention phases; do not run it concurrently with scored quiet-host timing. Its duration and publication delay add to the relevant queue. |
| **Path B timing gate** | Ratified, quiet Mac required; **one-hour cap per attempt**, at most three attempts under restricted retry rules. Timing debits the enclosing 40-hour study budget. [Retention preregistration:19–73](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-4/research/t34_pathb/PREREGISTRATION_RETENTION_V2.md:19) | Existing forecast allocates **0.361 hours**, about 22 minutes. A CPU speedup may improve feasibility, but cannot remove the paced observation interval. |
| **T3 pre-check** | **64 cases, bare and paced arms, H5000, 24 fps**, ≥50/64 survival required; separate **8-hour cap**. The current runner executes both arms. [retention_v2.py:1383–1425](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-4/research/t34_pathb/retention_v2.py:1383) | Full-horizon pacing allocation is **64 × 2 × 5000 / 24 / 3600 = 7.41 hours**, before overhead; deaths can shorten it. The amendment’s older **~1.5-hour** estimate is not supported by this execution loop. |
| **Paced retention** | **144 terminal pairs plus 32 Watch6 session pairs**, both Mac arms at 24 fps, fixed complete-bank look. **40 elapsed hours / 60 CPU-hours**, excluding the pre-check. Unpaced Colab results are descriptive and cannot change or block the Mac verdict. [Retention preregistration:120–184](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-4/research/t34_pathb/PREREGISTRATION_RETENTION_V2.md:120) | Full-horizon pacing allocation is **176 × 2 × 5000 / 24 / 3600 = 20.37 hours**; with timing, **20.73 hours**. This is a forecast, not measured completion time. Early terminal deaths shorten it; processing and pauses can lengthen it. [Cost evidence:160–170](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-4/research/t34_pathb/VERIFICATION_RETENTION_FIXES2.md:160) |
| **Each 800-world label round** | A5 recorded **54.4–57.2 worlds/hour/lane**, K12/W1 CPU inference, approximately **0.036 units/world**. R2 depends on the R1 student and selection. [Campaign log:55–62](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-4/docs/research/final_run_20261019/CAMPAIGN_LOG.md:55) | **14.0–14.7 lane-hours**, or **4.7–4.9 elapsed hours on three lanes**, before chunk/export overhead; approximately **29 units**. R1 and R2 cannot simply run concurrently. Later student trajectories may cost differently. |
| **Fits and screens** | R1 has two arms; R2 continues the selected arm. Plan allocates **four hours/fit** and roughly **one Mac hour per 96/32/32 screen**. [FINAL_PLAN.md:42,55–62,89](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-4/docs/research/final_run_20261019/FINAL_PLAN.md:42) | With R1 fits parallel: roughly **4 hours of fit latency plus 1–2 screen hours** for R1, then **4 plus 1** for R2. These are allowances, not verified full-bank timings. Production preparation can exceed the compute. |
| **Strict gate** | Fresh banks, development-variance sizing, calibration, checks, registered looks, independent audit. Fixed controllers use paired worlds; product inference uses paired sessions. [Template README:31–64,105–144](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-4/research/sequential_strict_template/README.md:31), [P7:12–16,54–56](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-4/docs/research/final_run_20261019/P7_STATISTICS.md:12) | **Bare candidate: retain 7–9 hours as a planning allowance. Assisted candidate: duration unresolved until its timing-preserving design and N are frozen.** |
| **Serving and release** | Actual live identity, timing/client behavior, labels, rollback and mode isolation remain required. The v2 adapter explicitly is not an already-qualified production hook. [Path B README:39–44](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-4/research/t34_pathb/README_V2.md:39), [Contract:42–57](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-4/docs/research/final_run_20261019/SUCCESS_CONTRACT.md:42) | Compute may be short; integration, defects, review and approval can consume hours or days. **No measured T34 qualification duration verified.** |

**VERIFIED — The historical strict-gate evidence supports an hours-scale bare gate, but not a universal five-to-nine-hour gate.** The old v8 gate completed 1,170 episodes in **5.7 hours on two Mac slots**. B8’s T7 result instead took **6.828 hours with up to six single-thread workers**, evaluating **5,760 H10000 episodes plus 45 controls**. Those are different designs and worker counts. [Historical timing:67–69](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-4/docs/research/governance_amendment_strict_on_runpod_2026-10-05.md:67), [T7 result:11–19](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-4/docs/research/t7_phase_r_result_2026-10-16.md:11)

**ESTIMATED — A Path B strict test cannot simply run the controller unpaced and inherit that duration.** Deadline misses and fallback behavior are part of the delivered controller. A fast unpaced simulation changes how much search completes before each action. It needs either real pacing or a separately justified, prospectively approved method that preserves the delivered timing behavior.

**VERIFIED — The portfolio contains a concrete warning about this:** a proposed maximum of **900 terminal pairs plus 128 H10000 product-session pairs** would consume approximately **134 paced Mac hours per finalist**. That is an illustrative proposed sample, **not a ratified T34 requirement**. P7 requires choosing sample sizes and fixed versus sequential looks from development variance before strict data. [Evalstats advice:20,27–35](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-4/docs/research/advice/2026-10-20_compute_portfolio_astra_evalstats.md:20)

**ESTIMATED — Consequently, the conditional calendar is:**

| Milestone or branch | Pacific-time forecast and assumptions |
|---|---|
| Mac becomes available | **Wednesday October 21, 10:00–13:00**, using the handoff ETA. |
| Timing + full-horizon pre-check + retention complete | **Thursday October 22, approximately 14:00–17:00**, plus Round 0 replay/publication time and operational overhead. Calculation: **0.361 + 7.407 + 20.370 = 28.14 hours** after Mac admission. |
| If the phases consume their full permitted budgets | They can occupy approximately **48 hours after admission**: eight pre-check hours plus the 40-hour study budget, which already includes timing. That reaches **Friday October 23, 10:00–13:00**. Hitting a cap does not guarantee a usable verdict. |
| **Claim A after a completed unsuccessful retention study, with Round 0 GO** | Two three-lane label rounds ≈**9.3–9.8 hours**, parallel R1 fits plus R2 fit ≈**8 hours**, screens ≈**2–3 hours**: approximately **20–21 hours before overhead**, then **7–9 hours strict**. With preparation already complete and a successful qualification, **Saturday October 24** is an aggressive earliest target; **October 24–26** is the more credible conditional window. |
| Claim A requiring R3 | Add roughly **10 hours of labels/fit/screen compute**, plus handoffs and review: approximately **half to one calendar day** if everything is ready. |
| **Claim B GO** | Skip distillation. A properly powered confirmation taking **one to two days**, followed by ready serving qualification, could support **October 24–26**. This duration is a scenario, not a verified gate size. |
| Claim B needing the illustrative 134-hour confirmation | Starting Thursday afternoon reaches **Wednesday October 28 morning**, before qualification and approval: approximately **October 28–29** if those are ready. |

**ESTIMATED — There is no defensible unconditional release date.** A failed premise, failed transfer, insufficient effect, strict failure or serving failure can prevent release altogether. An early Path B pre-check failure could admit the bare route earlier than the full-retention schedule above. Conversely, missing production work can move every date later.

**VERIFIED — That missing work is substantive.** The fit documentation says the production v2 `ReplaySource`/probe boundary is not supplied by its preparation, and its CUDA path is unqualified. More concretely, the current label code refuses process fan-out when a student checkpoint or rollout-policy adapter is supplied. Thus the measured K12 B8 teacher rate is not yet demonstrated as an executable R2 student-label pipeline. [Fit README:28–50,166–176](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-4/research/t34_exit_b8_fit/README.md:28), [label.py:108–111](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-4/research/t34_exit_b8/label.py:108)

**ESTIMATED — My single highest-priority change is to prepare the complete conditional release pipeline during the existing Phase 0/retention wait.** Have the production label/fit integration, strict evaluator, operating-characteristic machinery, serving hook and review packet ready before the deciding results arrive. Final variance-based sizing and ratification must still wait for the appropriate completed development evidence. This moves engineering and review off the serial post-verdict path without changing the running studies.

**ESTIMATED — Its saving is the amount of preparation that would otherwise start after the verdict: plausibly one to two calendar days, but not measured.** There are roughly **44–47 hours from the question’s timestamp to the central retention forecast** in which compatible preparation can progress. Heavy Mac verification must respect the occupied/quiet-host periods.

**VERIFIED — Review latency is real, but the files do not support treating every registration as a fixed one-day wait.** Retention’s six FIX-FIRST rounds occurred over roughly 2¼ hours, while Round 0’s final three-round review sequence is logged from about 16:05 to 16:55. The larger approximately one-day interval also included implementation and qualification. The repeatable improvement is a complete, internally consistent packet and early review, not omitting independent review. [Campaign log:41–62](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-4/docs/research/final_run_20261019/CAMPAIGN_LOG.md:41)

**ESTIMATED — The largest readily quantified compute scheduling benefit is using three qualified L4 lanes for future label rounds.** Compared with one lane, that saves approximately **9.3–9.8 hours per 800-world round**, or **19–20 hours across R1 and R2**. The plan already permits this after qualification; it should be realized rather than counted as a new discovery. Accelerating Round 0 tonight is less valuable because its expected completion has substantial slack before the Mac-dependent work.

**VERIFIED — The A100 evidence is weaker than the facts brief’s shorthand suggests.** The speed review labels the “~95% single-threaded CPython” decomposition an **estimate**, based partly on Mac probes. The reported 4.35 ms CUDA evaluation and 0.93 ms CPU evaluation were not a matched A100-versus-L4 benchmark. They establish a strong host-bottleneck diagnosis, not a measured A100 speedup or a proof that every A100 host has identical CPUs. [Fable speed review:8–35](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-4/docs/research/advice/2026-10-20_t34_a5_speed_fable.md:8)

| A100 use | ESTIMATED effect on the release date |
|---|---|
| Replace tonight’s L4 Round 0 lane | **No demonstrated date gain.** Even a substantial acceleration likely remains inside the 7–12-hour slack before Mac availability. Changing the running registration is not assumed. |
| Replace one of three future label lanes | If its **useful host throughput** were twice an L4’s, aggregate capacity would rise from three to four L4-equivalents. A 4.9-hour round becomes about **3.7 hours**: roughly **1.2 hours saved per round**, **2.4 hours over R1/R2**. The required host speedup is unmeasured. |
| Run cached student fits | Potentially useful after qualification if genuinely GPU-bound. If the critical four-hour fit stages halved, R1+R2 could save about **four hours**. Replay/cache/CPU time may prevent that gain. |
| Phase 0, paced retention, strict gate, serving qualification | **No direct substitution benefit:** these stages remain on the Mac. Faster remote hardware cannot compress registered real-time pacing or supply Mac serving evidence. |

**ESTIMATED — I would credit the release schedule with zero A100 hours saved until a matched useful-output measurement exists.** That is not a claim that A100 cannot help. It is the correct scheduling treatment of an unmeasured accelerator/host combination. With prepaid units and a calendar objective, any qualified positive wall-time saving can be worthwhile; the higher unit price alone is not a reason to reject it. At present, however, the plausible benefit is **hours on the conditional learning branch**, not the several days controlled by Mac confirmation, production readiness and approval.

**VERIFIED — Release approval wording is inconsistent across the snapshot:** the contract and campaign-log role statement retain owner approval, while newer `STATUS.md` delegates release/serving approvals to an independent Fable judge. Your question explicitly retains the owner’s OK, so this forecast includes that final human decision. No source supplies its turnaround time. [Contract:55–57](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-4/docs/research/final_run_20261019/SUCCESS_CONTRACT.md:55), [STATUS.md:22](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-4/docs/STATUS.md:22)

**ESTIMATED — The decision I would make now is to protect and prepare the Mac release sequence, resolve the assisted strict-gate cost before promising a date, and finish the production R1→R2 path while the existing studies run. October 24–26 is a conditional accelerated target; November 6–8 is an older plan allocation, not a compute-imposed minimum. An unmeasured A100 should not move either date on the schedule.**


## Call 2 — prompt (verbatim, cross-examination)

```text
Follow-up on your previous answer (critical path to a released better snake agent; same repo, same read-only worktree at 6f1e751f). Your full previous answer is at /Users/josenunez/Projects/ml/snake-dqn-artifacts/a100-advice/p4-call1/snake-dqn-advice-a100-4.last-message.md — re-read it first. Same HARD RULES: read-only; do not edit, commit, create or delete any file in the repo or worktree; at most tiny read-only probes on the Mac (1 thread, `nice -n 10`, each < 30 s; the Mac is running the 6-worker Phase 0 study and the Round 0 Colab supervisor and must not be disturbed); no Colab, RunPod or network jobs; full answer in your final message with file:line evidence and numbers; label every claim VERIFIED or ESTIMATED.

I am cross-examining your answer. Treat each point as an open question and argue against yourself where the evidence is thin.

1. Your central conditional date (Claim A release "October 24–26", labels+fits+screens ~20–21 h then a 7–9 h strict gate) sits beside your own VERIFIED finding that `research/t34_exit_b8/label.py:108–111` refuses process fan-out when a student checkpoint or rollout-policy adapter is supplied, that the fit's production ReplaySource/probe boundary is not supplied, and that the CUDA fit path is unqualified. Reconcile these: how much implementation, acceptance (the `research/colab_cli` rule: any code change needs a fresh final-only acceptance), review and ratification work actually stands between today and an executable R1→R2 pipeline, in calendar days, with evidence from the repo (specs, READMEs, QUALIFICATION.md, the history of how long S1/Round 0/retention took from spec to dispatch)? Does that work, not Mac occupancy, bound the Claim A date?

2. You put Path B retention before the label rounds in the dependency graph. SUCCESS_CONTRACT §5 says the deadline-feasibility check "runs before R1/R2 label rounds". Is that a hard ordering constraint that puts ~28–48 Mac-hours on Claim A's critical path too, or a funding/decision rule that permits label rounds (Colab) to proceed in parallel on a Round 0 GO? Read the exact wording and the registrations and say what the binding documents actually permit, and what calendar days are at stake.

3. Your Round 0 "7–12 h slack" assumes prompt chunk handoffs: each chunk requires an operator to publish a ledger (commit, merge, push) before dispatching the next, chunks are serial, and the pace is ~3.5 h per 100-pair chunk starting 16:55 PT. Walk the real clock through the night (operator availability is not guaranteed overnight) and say when the three chunks, the 10% Mac replay, publication and the sole analysis credibly finish, and whether Round 0's verdict (not its data) is on the critical path for anything that starts before Friday.

4. Your pre-check figure (7.41 h full-horizon pacing from `retention_v2.py:1383–1425`) contradicts the amendment's ~1.5 h; your 20.37 h retention figure is also a full-horizon allocation. Which is right, and what is the single cheapest read-only check (or a measurement that someone could run later on a quiet Mac) that settles the real Mac occupancy of the retention phases?

5. You credit zero A100 hours. What exactly would change your mind — what single measurement, how long it takes, what it costs in units, and what threshold result would make an A100 lane worth adding to the label rounds or the fits? Conversely, what result would close the question permanently?

6. The strict gate for an assisted (Claim B) controller is "unresolved until its timing-preserving design and N are frozen". From the P7 rules and the strict template, what is the smallest defensible design and what Mac-hours does it imply, and could its design be frozen now, during the Phase 0 wait, without touching running studies?

Finally: after re-examining, state your revised earliest credible release date for each claim, the one link you are now most confident binds the calendar, the one claim in your previous answer you are least sure of, and the single measurement that would settle the A100 question.
```

## Call 2 — Astra answer (verbatim)

**ESTIMATED — I withdraw October 24–26 as the central release forecast.** It was a best-case compute schedule presented too much like an executable delivery schedule. My revised planning windows are **Claim A: October 26–29 PT**, and **Claim B: October 25–28 PT only if a compact, adequately powered paced strict design is justified**. A larger assisted confirmation moves Claim B into October 29 or later. These are conditional planning dates, not statistical predictions of success.

**VERIFIED — Scope:** I reread the previous answer first and checked the clean, detached worktree at `6f1e751f4ec11c280c211634cd09de924b649218`. No repository files changed; no tests, gameplay, network requests, or remote jobs ran. The only numerical check was a tiny, single-threaded aggregation of already committed historical development data. “VERIFIED” below means verified in this snapshot, not observed live.

**1. VERIFIED — The production pipeline is unfinished in several independent places.**

| VERIFIED blocker | Consequence |
|---|---|
| Student checkpoints and rollout adapters are rejected when label process fan-out is enabled. [label.py:108–111](research/t34_exit_b8/label.py#L108) | The measured twelve-process B8 throughput is not an executable R2 throughput result. |
| The production manifest builder refuses nondefault student inputs until publication and SHA binding exist. [Engine README:99–115](research/t34_exit_b8/README.md#L99) | Fixing process creation alone does not make R2 dispatchable. |
| Production v2 exports, `ReplaySource`, previous-student Q, and the registered probe remain an integration boundary. [Fit README:46–50](research/t34_exit_b8_fit/README.md#L46), [294–328](research/t34_exit_b8_fit/README.md#L294) | A successful synthetic cache smoke is not a production fit pipeline. |
| CUDA prefetch is implemented but unqualified. [Fit README:28–31](research/t34_exit_b8_fit/README.md#L28) | Four hours per fit remains an allowance, not a measured production duration. |

**VERIFIED — The fresh final-only acceptance rule is narrower than “any code change.”** It applies to changes in **`research/colab_cli` executable code**. Workload changes outside that boundary do not automatically invalidate the transport qualification. They still require their own source-bound preflight, replay, CUDA and review evidence. The qualification record explicitly distinguishes executable changes from documentation changes and quotes approximately **1.5 units** for a new final-only acceptance. [QUALIFICATION.md:3–5,23–32](research/colab_cli/QUALIFICATION.md#L23)

**VERIFIED — The remaining acceptance work includes real end-to-end seams:** changed-entrypoint A6; production CUDA fitting; checkpoint/export provenance; and A2 replay qualification on **at least 150 worlds across three mixes**, L4→Mac and L4→another L4 session. The plan also requires committed registration and bank rules. Existing component tests do not collectively prove these seams work. [FINAL_PLAN.md:74–81](docs/research/final_run_20261019/FINAL_PLAN.md#L74)

**ESTIMATED — I would budget the following remaining engineering time, with independent engine and fit work overlapping:**

| Work | Calendar allowance |
|---|---:|
| Production export/replay/probe integration; student fan-out and checkpoint binding | 1–2 days |
| Integrated replay/CUDA qualification, measured pricing and manifest preflight | 0.5–1 day |
| Repairs, independent review, final source binding and ratification | 0.5–1 day |
| **Qualified executable R1→R2 path** | **2–4 days; 3–5 if one substantive GPU/replay failure occurs** |

**VERIFIED — History argues against both zero preparation time and an automatic week-long delay.**

- Retention’s initial v2 protocol commit to ratification took about **13 h 19 min**; its registration draft to ratification took about **4 h 59 min**. It was ratified, not dispatched: Phase 0 still blocked admission.
- Round 0’s first package to ratification took about **4 h 21 min**; its final three-round review sequence occupied approximately **50 minutes**.
- The plan records **ten S1 acceptance runs and four dispatches**, and originally allocated **7–9 preparation days**. That allocation is historical planning, not a measured remaining duration.
- The fit preparation had already existed for roughly a day while the production source/probe and CUDA qualification remained pending.

The commit intervals come from local Git history; the corresponding review and ratification events are recorded in [CAMPAIGN_LOG.md:39–62](docs/research/final_run_20261019/CAMPAIGN_LOG.md#L39). S1’s counts and original preparation allowance appear in [FINAL_PLAN.md:68](docs/research/final_run_20261019/FINAL_PLAN.md#L68).

**ESTIMATED — Engineering readiness now plausibly co-binds or overtakes Mac occupancy for Claim A.** The correct start condition is:

`max(Path B disposition, published Round 0 GO, qualified R1→R2 pipeline)`

My previous calculation effectively assumed the third term was already satisfied. Fast implementation could still finish during the Mac wait; the evidence does not establish that it will.

**2. VERIFIED — The binding documents do not clearly authorize parallel production labels on Round 0 GO alone.**

**VERIFIED — The strongest wording is unambiguous:** SUCCESS_CONTRACT §5 says the deadline-feasibility check “runs before R1/R2 label rounds.” It defines that check as both deadline-constrained efficacy and timing, not merely a short latency smoke. Path B GO makes labels optional research; Path B failure permits labels only on Round 0 GO. [SUCCESS_CONTRACT.md:59–68](docs/research/final_run_20261019/SUCCESS_CONTRACT.md#L59)

**VERIFIED — There is nevertheless a genuine textual conflict.** Amendment 2 says:

- “Deciding sequence (**no label rounds funded before it**)” at line 93.
- “Teacher label rounds (R1+) wait for **(b) or Round 0**” at line 103.

The latter supports the permissive reading you raise. But the concrete retention registration says KILL drops Claim B and teacher labels **then** wait on Round 0. STATUS explicitly ranks SUCCESS_CONTRACT above DESIGN. [DESIGN_V2_SHADOW.md:93–104](docs/research/final_run_20261019/DESIGN_V2_SHADOW.md#L93), [Retention registration:348–356](research/t34_pathb/PREREGISTRATION_RETENTION_V2.md#L348), [STATUS.md:10–14](docs/STATUS.md#L10)

**ESTIMATED — My operational interpretation remains retention disposition before production R1/R2 labels.** This is a funding/sequencing rule, not a physical computing dependency. The word “or” deserves an explicit prospective judge ruling; it is insufficient grounds to claim verified permission to bypass the higher-priority contract.

**ESTIMATED — A prospective overlap amendment could save approximately one to two calendar days**, provided the production pipeline and Round 0 verdict are ready. Under the current reading, successful completion of the full retention sequence keeps labels behind Thursday October 22 afternoon/evening, or Friday October 23 if the phases approach their caps. An early Path B failure can resolve the branch sooner. Cap expiry is not automatically a scientific failure or label-launch permission.

**3. ESTIMATED — Walking the actual Round 0 clock removes the supposed guaranteed overnight slack.**

**VERIFIED — Each chunk has a publication barrier:** commit, merge and push its ledger before the next dispatch. After all chunks are terminal, the sequence remains **Mac replay → publication → sole analysis → publication**. The handoff explicitly gives this operator workflow. [HANDOFF:140–149](docs/research/final_run_20261019/HANDOFF_2026-10-20.md#L140)

**ESTIMATED — Using the observed approximately 3.5-hour pace per 100 pairs and allowing 15 minutes per prompt handoff:**

| Operator availability | Data completion | Published verdict, assuming a suitable Mac interval |
|---|---|---|
| Every night handoff staffed | Chunk 000 ends Tuesday ~20:30; 001 runs ~20:45–00:15; 002 ~00:30–02:00 Wednesday | Wednesday late morning/afternoon, after Mac replay |
| 20:30 handoff staffed; operator returns 08:00–09:00 | Chunk 001 finishes overnight; 002 runs roughly 08:15–09:15 to 09:45–11:00 Wednesday | **Wednesday ~11:00–15:00** |
| No 20:30 handoff; operator returns 08:00–09:00 | Chunk 001 finishes ~11:45–12:45; 002 finishes ~13:30–14:45 Wednesday | **Wednesday ~14:00–16:00** |
| Handoffs slip into Thursday | Potentially still within the dispatch deadline | Thursday evening or Friday |

**VERIFIED — The pace is a forecast from 15/100 pairs at 32 minutes, not a completed-chunk measurement.** Phase 0’s recorded Mac-release ETA was Wednesday 10:00–13:00. [HANDOFF:155–157](docs/research/final_run_20261019/HANDOFF_2026-10-20.md#L155)

**ESTIMATED — The verdict windows reserve roughly 0.5–1 hour for replay, verification, analysis and publication, but that is unmeasured.** Replay covers **24 pairs / 48 traces** without rerunning teacher search. It must receive a separate Mac interval before scored timing or between retention phases; it cannot simply overlap quiet-host timing. [Round 0 README:198–230](research/t34_round0/README.md#L198)

**VERIFIED — The 48-hour requirement is a dispatch deadline, not a verdict deadline.** A final-chunk dispatch around Thursday 16:45 could consume its **152-minute timeout**, reaching approximately 19:17 before replay and analysis. [Registration:912–929](research/t34_round0/REGISTRATION_ROUND0.json#L912), [HANDOFF:143](docs/research/final_run_20261019/HANDOFF_2026-10-20.md#L143)

**ESTIMATED — Therefore Round 0’s verdict probably does not bind ordinary pre-Friday engineering or Path B work, but it can bind labels after an early Path B failure.** With normal daytime handoffs, Wednesday afternoon verdict publication leaves useful slack before full retention finishes. With Thursday operator gaps, that slack disappears. My previous **“7–12 hours slack” applied only to cloud data completion under staffed-night handoffs**, and should not have been used as a general scheduling conclusion.

**4. VERIFIED — The 1.5-hour passing pre-check estimate is wrong for the current runner; 7.41 hours is conservative exposure arithmetic, not expected occupancy.**

**VERIFIED — The runner executes all 64 original T3 cases, both bare and paced arms, at H5000 and 24 fps.** It stops terminal worlds after death and zero-pads the remaining score. Continuous-respawn Watch sessions continue through the horizon. [retention_v2.py:1383–1425](research/t34_pathb/retention_v2.py#L1383), [1286–1294](research/t34_pathb/retention_v2.py#L1286)

**ESTIMATED — Fifty surviving paced cases alone require**

`50 × 5000 / 24 / 3600 = 2.894 hours`

before any bare case, failed paced case or overhead. Thus **1.5 hours cannot describe a passing pre-check**.

**VERIFIED — The cheapest useful check was already available without running gameplay.** Aggregating the committed historical T3 development pool gives **64 bare trajectories, 275,586 alive frames and 38 full-horizon survivors**. The pool schema identifies the bare survival-fraction and no-death columns explicitly. [RETENTION_OC_V2.json:113](research/t34_pathb/RETENTION_OC_V2.json#L113), [retention_oc_v2.py:98–115](research/t34_pathb/retention_oc_v2.py#L98)

**ESTIMATED — Assuming those original bare trajectories replay identically:**

- Bare pacing exposure, including 26 terminal death ticks: **3.190 hours**.
- A passing paired pre-check: approximately **6.084–6.894 hours**, before setup, guards and worker shutdown.
- The original **7.407 hours** assumes every bare and assisted trajectory survives H5000.

**ESTIMATED — Retention’s 20.370 hours likewise assumes full-horizon survival for every terminal trajectory.** Its **32 paired continuous Watch6 sessions alone require 3.704 hours**; terminal deaths shorten the remaining exposure. Setup, shutdown, tick overruns and pauses can increase actual wall time. The registration itself labels the full-horizon assumption. [Retention registration:424–435](research/t34_pathb/PREREGISTRATION_RETENTION_V2.md#L424)

**ESTIMATED — Replacing the all-survive pre-check assumption with the historical bare durations gives approximately 26.82–27.63 hours for timing + passing pre-check + full-exposure retention**, before publication and other overhead. That is modestly below my previous 28.14 hours; it does not remove the Thursday/Friday dependency.

**VERIFIED — The cheapest definitive occupancy measurement is to read the phase timing receipts after the existing registered run.** Timing and pre-check receipts record elapsed/CPU seconds; efficacy’s completion record does too. Crucially, efficacy elapsed already includes the timing debit: total phase elapsed is **pre-check elapsed + efficacy elapsed**, without adding timing twice. Publication gaps are separate. [retention_v2.py:1443–1444](research/t34_pathb/retention_v2.py#L1443), [1539–1544](research/t34_pathb/retention_v2.py#L1539), [1654–1655](research/t34_pathb/retention_v2.py#L1654)

**ESTIMATED — No sub-30-second probe can determine future assisted lifetimes across the complete bank.** A later quiet-Mac cost pilot could run one disjoint terminal pair and one Watch6 pair through the same collector: **13.89 minutes maximum nominal pacing, plus overhead**. It would measure overhead and confirm early-exit behavior, but would still not settle the complete bank’s death-time distribution.

**5. ESTIMATED — The single A100 measurement I would prioritize is a matched, complete production cached fit on L4 and A100-40.**

**VERIFIED — That follows the existing design:** cache observations first, time a complete L4 fit, and consider A100-40 if the fit is GPU-bound beyond approximately one hour. The current code has not yet supplied that evidence. [DESIGN_V2_SHADOW.md:70–72](docs/research/final_run_20261019/DESIGN_V2_SHADOW.md#L70), [Fit README:28–31](research/t34_exit_b8_fit/README.md#L28)

**ESTIMATED — The comparison should hold fixed** the production input hashes, architecture, fp32 recipe, batch size, sample order, epochs and probe. Measure **allocation-to-verified-export time**, with replay/cache, optimizer, probe and export timings separated. Compare accepted artifacts, not GPU utilization or isolated update throughput.

**ESTIMATED — Concrete time and unit budget, using the existing four-hour L4 allowance:**

| Case | Runtime time | Units at L4 2.0 / A100-40 5.4 u/h |
|---|---:|---:|
| L4 4 h; A100 achieves 2× end-to-end speedup | 6 aggregate hours | **18.8 u** |
| L4 4 h; A100 achieves 3× | 5.33 aggregate hours | **15.2 u** |
| Prospective cap of 4 h on each runtime | 8 aggregate hours | **29.6 u**, or **35.52 u including 20% reserve** |

**ESTIMATED — Calendar duration is roughly four hours if two assignments are available and run concurrently, otherwise up to eight, plus any preparation not already included.** These are pricing scenarios, not measured fit times. A timeout is incomplete evidence, not proof of poor A100 performance. The newer repository planning rates are **5.4/7.5 u/h for A100-40/80**, rather than the older 5.3/6.8 figures. [Hardware advice:9–19](docs/research/advice/2026-10-19_t34_gpu_astra_search.md#L9)

**ESTIMATED — My prospective adoption threshold would be ≥1.5× whole-fit speedup, correct artifacts, and at least one hour of net savings on the remaining critical path after additional qualification and handoffs.** That is a calendar threshold, not unit break-even. Unit break-even requires **>2.70× for A100-40** or **>3.75× for A100-80** at those rates.

**ESTIMATED — Labels require a different measurement:** identical completed, replay-valid student-trajectory worlds on both hosts, with the same K12/W1 CPU layout and all mixes. If one A100 replaces one of three L4 lanes and supplies `g` times an L4’s useful throughput, a balanced round changes from `T` to `3T/(2+g)`. At `g=2`, a 4.8-hour round becomes 3.6 hours: **about 2.4 hours saved across two rounds**, before extra handoffs. That could justify a prospectively amended lane; an isolated GPU benchmark could not.

**ESTIMATED — Nothing would close the A100 question permanently across future workloads.** A matched result showing **≤1× useful end-to-end speed and no memory advantage** would close it for the frozen production workload and observed host configuration. Label performance does not settle fitting, and fitting does not settle labels. My earlier zero credit remains correct accounting for **unmeasured savings**, not evidence that A100 cannot help.

**6. VERIFIED — There is no numerically determined “smallest defensible” assisted strict bank in this snapshot.**

**VERIFIED — P7 supplies a floor of 16 paired worlds per mix, not a power justification.** It requires complete paired sessions for product inference and fixed-N versus sequential selection by expected cost at +40 using development variance. The strict template also requires development inputs, operating-characteristic and band-cost reports, source binding and an independent audit. [P7:12–16,54–56](docs/research/final_run_20261019/P7_STATISTICS.md#L12), [Strict README:31–49](research/sequential_strict_template/README.md#L31)

**ESTIMATED — The smallest sensible scope is one frozen assisted controller versus B8 s3, with:**

- Fresh, disjoint terminal banks for all three mixes.
- Fresh complete H5000 Watch6 session pairs.
- Real 24 fps deadlines, including misses and fallback actions.
- One delivered controller at a time on the quiet Mac.
- Fixed-N unless development-only operating-characteristic calculations establish a cheaper sequential design.
- Every contract skill, survival, product and timing bar retained.

**VERIFIED — There is additional implementation work hidden by “use the strict template.”** The historical gate tests superiority in at least two mixes and scripted noninferiority; its pooled survival implementation combines corresponding indices across mixes. T34 requires disjoint-stratum or covariance-aware inference and adds binding product bars. The strict specification and auditor must reconcile those rules prospectively; filling in an episode runner is insufficient. [sequential_gate.py:14–40,73–84](src/evaluation/sequential_gate.py#L14), [P7:22–39](docs/research/final_run_20261019/P7_STATISTICS.md#L22), [Retention inference:300–329](research/t34_pathb/PREREGISTRATION_RETENTION_V2.md#L300)

**ESTIMATED — For serial execution with both arms paced, `N` terminal pairs per mix and `S` H5000 product-session pairs, full-horizon exposure is:**

`T = 2 × (3N + S) × 5000 / 24 / 3600 hours`

| ESTIMATED design example—not a power claim | Nominal pacing |
|---|---:|
| 16/mix, before adding product sessions | **5.56 h** |
| 48/mix + 32 product sessions | **20.37 h** |
| 96/mix + 64 product sessions | **40.74 h** |
| 300/mix + 128 product sessions | **118.98 h** |

**ESTIMATED — Calibration, timing probes, setup, audit and serving qualification are additional.** Terminal deaths reduce exposure; complete product sessions do not. The prior **134-hour** example used H10000 product sessions; H10000 is not a mandatory product horizon here.

**VERIFIED — Retention’s favorable OC projection does not establish adequate strict power at 48/32.** At a true gain exactly +40, the point requirement “estimate ≥40” itself passes only approximately half the time under symmetric noise. Sizing needs error control at the boundary and power scenarios above it, plus the actual paired product variance. [RETENTION_OC_V2.json:18](research/t34_pathb/RETENTION_OC_V2.json#L18)

**VERIFIED — Much of this design can be frozen now.** P7 requires freezing before the strict study’s own data, not after Phase 0. Controller identity, hypotheses, pairing, bank-generation rules, timing semantics, sizing procedure, caps and audit interface can be prepared independently of running studies. A numeric N could also be frozen now if existing development evidence supports a reviewed justification; completed paced retention would provide more relevant variance. [P7:6–7](docs/research/final_run_20261019/P7_STATISTICS.md#L6), [Strict README:43–49](research/sequential_strict_template/README.md#L43)

**ESTIMATED — My revised calendar and confidence are therefore:**

| Claim | Earliest date I would put in a working plan | Conditions and wider window |
|---|---|---|
| **A: bare student** | **October 26 PT** | Plan **October 26–29**: qualified pipeline around October 23–25, published admission decisions, approximately 20–21 compute hours plus handoffs, then newly sized strict/product evidence and serving qualification. October 24–25 remains an exceptional best case, not my central forecast. |
| **B: assisted hero** | **October 25 PT** | Plan **October 25–28 only if** retention passes and a roughly 20–41-hour paced strict design is justified and prepared. A roughly 119–134-hour design instead points to **October 29 or later**, before unresolved serving work. |

**ESTIMATED — The one link I am now most confident can bind Claim A’s calendar is the qualification of the complete student-label → replay/cache → fit/probe → next-student-label pipeline.** Its missing interfaces are verified; its completion time is not. More accelerator capacity cannot bypass that link.

**ESTIMATED — The claim I am least sure of from my previous answer is “Claim B could release October 24–26 with one to two days of strict confirmation.”** I had no frozen sample size or powered design supporting that duration. The inherited 7–9-hour bare-gate allowance also needs replacement by a T34-specific estimate.

**ESTIMATED — The single measurement I would use next to decide whether A100 deserves calendar credit is the matched complete production cached fit described above.** Until it produces a valid whole-job speedup, I would continue assigning A100 **zero demonstrated release-time savings**.
## Prompter notes (fable-prompter-4)

- Astra withdrew its call-1 central date (Oct 24–26) under cross-examination; revised plan dates: Claim A Oct 26–29 PT, Claim B Oct 25–28 PT (only with a compact powered paced strict design), else Oct 29+.
- Binding link for Claim A (Astra's most confident claim): the un-built production student-label → replay/cache → fit/probe → next-label pipeline (label.py:108–111 refuses student fan-out; manifest builder refuses nondefault student inputs; fit ReplaySource/probe boundary and CUDA path unqualified). Estimated 2–4 days (3–5 with one GPU/replay failure). No accelerator bypasses this.
- Claim B: strict design for an assisted controller is unsized; P7 gives only a 16/mix floor. Astra's exposure formula T = 2(3N+S)·5000/24/3600 h (48/mix+32 sessions = 20.4 h paced; 96/64 = 40.7 h). Can be frozen now during the Phase 0 wait.
- Mac occupancy: timing + passing pre-check + retention ≈ 26.8–27.6 h after Mac admission (Wed 10-21 10:00–13:00) → Thu 10-22 afternoon, Fri if caps are approached. Pre-check "~1.5 h" is wrong for the current runner (≥ 2.9 h for 50 surviving paced cases alone).
- Contract ordering: SUCCESS_CONTRACT §5 puts the Path B check before R1/R2 labels; DESIGN_V2 line 103 says "wait for (b) or Round 0" — a real textual conflict worth a prospective judge ruling; an overlap amendment could save ~1–2 calendar days.
- Round 0: the "7–12 h slack" was only valid under staffed overnight handoffs; serial chunks + publication barriers mean the verdict lands Wed ~11:00–16:00 with daytime handoffs, Thu/Fri if handoffs slip.
- A100: zero demonstrated calendar credit; one measurement to decide it = matched complete production cached fit L4 vs A100-40 (~15–30 u, ~4–8 h), adoption threshold ≥ 1.5× whole-fit and ≥ 1 h net saved; label-lane credit needs a separate matched K12/W1 student-world measurement (g=2 saves ~2.4 h over R1+R2).
- Caveat: Astra's calendar arithmetic is self-labelled ESTIMATED; its VERIFIED file:line cites were not independently re-checked by me beyond the facts brief sources.
- Tooling note: `codex-handoff.sh` creates the `.last-message.md` empty at start — wait on `[ -s file ]`, not `ls`.
