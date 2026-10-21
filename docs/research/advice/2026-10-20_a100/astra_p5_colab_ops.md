# Astra P5 — Colab operations and unit economics (2026-10-20)

Advisor: Astra (Codex `CODEX_MODEL=gpt-6-astra`, effort high, fast mode), worktree `snake-dqn-advice-a100-5` at 6f1e751f, read-only (receipts: produced_work=NO, dirty_files=0). Prompter: Fable (fable-prompter-5). Raw outputs: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/a100-advice/p5-call{1,2}/`.

## Call 1 — prompt (verbatim)

```text
You are an independent advisor (Astra) for the snake-dqn project. Read-only engagement: do NOT edit, commit, or create files in this repository. You may run at most tiny read-only probes on this Mac (1 thread, prefixed with `nice -n 10`, each under 30 s) — the Mac is running a 6-worker study that must not be disturbed. Do NOT launch any Colab or RunPod jobs. Web search for public facts is fine and encouraged. Put your FULL answer in your final message (nothing is read from files), with file:line evidence and numbers, and label every claim as VERIFIED (you checked it in the repo or on an official page), THIRD-PARTY (unofficial web source), or ESTIMATED (your reasoning).

START by reading docs/research/advice/2026-10-20_a100/FACTS_AND_QUESTION.md (goal, open question, facts, sources). Then, as needed: docs/research/advice/2026-10-20_compute_portfolio_FABLE_JUDGMENT.md and ..._QUESTION.md, docs/research/final_run_20261019/CAMPAIGN_LOG.md (the "Compute portfolio" section and the 10-20 rows), docs/research/final_run_20261019/HANDOFF_2026-10-20.md, research/colab_cli/ (supervisor, QUALIFICATION.md, ACCEPTANCE.md, accounting.py, admission.py), docs/research/colab_cli_ref/ (official Colab CLI docs: README.md, docs/01_session_management.md, docs/04_automation_and_utility.md), and the A5 benchmark receipts under /Users/josenunez/Projects/ml/snake-dqn-artifacts/ (the CAMPAIGN_LOG row names the path).

GOAL: a better served snake agent than the current Watch hero (bare B8 seed 3), proven by the usual strict gate on the Mac, as soon as possible in wall-clock time. The ~1,157 prepaid Colab units are expiring money the owner wants spent by ~2026-11-20 (~39 u/day) on work that produces decisions.

YOUR LENS: Colab operations and unit economics. Constraints and facts you should weigh (verify them):
- Colab Pro+, official Colab CLI through a Mac supervisor (research/colab_cli), max 3 concurrent runtime assignments, each ≤ 24 h, idle runtimes still bill until stopped (Q1's idle allocation was its dominant cost).
- Rates are third-party, not published by Google: L4 ~1.7 u/h (observed 1.54; we price at 2.0), A100-40GB ~5.4, A100-80GB ~7.5, G4 ~8.7, T4 ~1.2. `colab usage` on 2026-10-18 read 6.84 u/h for one A100 + one L4. The supervisor prices a manifest rate as a ceiling and reconciles against `colab usage` (see accounting.py / admission.py).
- The CLI's `--high-mem` shape applies to CPU/T4/A100 runtimes but not L4 (docs/research/colab_cli_ref/README.md ~line 146). The host vCPU count, physical-core count, and per-core speed of a Colab A100 runtime (standard and high-mem) are UNMEASURED by us; an L4 runtime has 12 vCPU ≈ 6 physical cores; a CPU-only runtime had nproc=2.
- The work is mostly host-bound today: teacher/benchmark worlds are ~95% single-threaded CPython; the L4 GPU sits at 2–8% util; CUDA batch-1 forward was slower than CPU (4.35 ms vs 0.93 ms). After fan-out to 12 single-thread processes per L4 runtime: 57.2 / 54.4 completed worlds/h per lane, ~0.036 u per teacher world, 16–20k rows and ~8.7 MB per world. On the Mac, one core did ~4.5 s per searched decision before a 3.4–4.2× speedup; 1M searched decisions ≈ 129 Mac-hours — the distillation step's label volume is the known bottleneck. CUDA throughput of the search-teacher rollouts has never been measured. Earlier (training-shaped, 2026-10-17) A100-80 facts: self-play 15.9k transitions/s GPU-bound; isolated learner fp32 bs2 82.7 upd/s, bf16 118.
- Running now: Phase 0 premise test on the Mac (ETA Wed 10-21 ~10:00–13:00 PT); Round 0 (240 paired worlds, 3 chunks serial on 1 L4 lane, chunk ~3.5 h; 48 h dispatch deadline). The adopted 30-day portfolio (Fable judge) explicitly did not fund A100/G4/H100 for host-bound work, and sizes L4 lanes for labels, fits and screens.
- Rules: strict gates and serving qualification stay on the Mac; freeze designs before data; never relabel a registered gate; no idle allocations.

OPEN QUESTION (answer it through your lens; do not restrict yourself to anything already proposed): Given that the units must be spent and the work is mostly host-bound, what is the fastest-to-a-better-agent way to turn these ~1,157 units into progress — what role, if any, should A100 runtimes play, for which part of the work, and what is worth measuring before deciding? Please use web search to establish what is publicly known (official vs third-party) about Colab A100 runtime host specs — vCPU count, RAM, standard vs high-mem shape, availability/preemption, and unit rates — and reason explicitly about units per completed teacher world, per labeled decision, and per fit on A100 vs L4 under the host-bound regime, and under any regime you think could be reached (say what would have to be true). Reason about the 3-assignment cap and the ≤ 24 h limit as binding constraints, and about the time value of wall-clock (the owner wants a better agent soon, units are not scarce). State what you would measure first, how long/how many units that measurement costs, and what result would make A100s worth using versus not.

Deliver: numbered findings with evidence, your recommendation through this lens, the single most informative measurement, and a short list of claims you could not verify.
```

## Call 1 — Astra answer (verbatim)

**ESTIMATED — Recommendation:** Keep L4 as the production default, but fund a small, matched A100 placement experiment now. Use A100s when they shorten the path to a qualified candidate—even if they consume more units per result. The strongest likely A100 role is cached student fitting; its value for today’s teacher depends on host performance or demonstrably effective batched CUDA inference. I would also reconsider the blanket exclusion of **G4 for host-bound work**: a public observation of 48 logical CPUs makes it a potentially valuable way to increase throughput within the three-assignment cap.

**VERIFIED — Scope:** I inspected repository snapshot `6f1e751f4ec11c280c211634cd09de924b649218`, local A5 artifacts, and public sources. The worktree remained clean. I made no repository changes and launched no jobs. “VERIFIED” below distinguishes checked source records from new measurements; reported running-study status is the repository’s dated account, not a live runtime poll. Recommendations and proposed thresholds are **ESTIMATED**, not new registrations.

1. **The scarce resource is time on the critical path, especially quiet Mac time—not units alone.**

   **VERIFIED:** The contract permits both a bare student and a visibly labelled assisted hero. If Path B passes its development bars under actual deadlines, it proceeds to the Mac strict gate; label rounds become optional. Thus, finishing all distillation research is not a prerequisite for the earliest successful release. See [SUCCESS_CONTRACT.md:17](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/docs/research/final_run_20261019/SUCCESS_CONTRACT.md:17) and [the sequencing rule at line 59](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/docs/research/final_run_20261019/SUCCESS_CONTRACT.md:59).

   **VERIFIED:** The latest handoff records Round 0 chunk 000 at 15/100 pairs, roughly 3.5-hour pace, and Phase 0 at 100/160 worlds with a Wednesday 10:00–13:00 PT estimate. Round 0’s three chunks run serially with a publication barrier and 48-hour dispatch deadline. These supersede the earlier portfolio’s proposed three-lane Round 0 schedule. See [HANDOFF:139](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/docs/research/final_run_20261019/HANDOFF_2026-10-20.md:139) and [HANDOFF:155](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/docs/research/final_run_20261019/HANDOFF_2026-10-20.md:155).

   **ESTIMATED:** Preserve that registered sequence. Use spare Colab capacity to remove the next dependency—fit qualification, useful labels, or the unpaced retention comparator. Accelerating a fit that finishes before its prerequisite labels arrive saves no release time; completing a qualified fit or comparator before the Mac becomes available can.

   **ESTIMATED:** Spending approximately 39 units/day does not require expensive accelerators. Three L4 lanes at the observed 1.54 u/h consume 39 units in **8.44 hours/day**; at the planning ceiling of 2 u/h, in **6.5 hours/day**. The harder requirement is having enough admitted, useful work ready.

2. **Public evidence does not support assuming that an A100 provides more CPU capacity than an L4.**

   **VERIFIED — official:** Google documents variable hardware availability and resource limits. Pro+ supports continuous execution for **up to 24 hours** with sufficient units; this is not a guaranteed uninterrupted reservation. The official CLI supports `--high-mem` for CPU/T4/A100, while L4 has one shape and ignores the flag. Neither source specifies guaranteed physical-core counts or per-core performance. Sources: [Colab FAQ](https://research.google.com/colaboratory/faq.html), [official CLI shape documentation](https://github.com/googlecolab/google-colab-cli#usage-notes).

   **THIRD-PARTY:** A dated February 26, 2026 measurement reports:

   | Observed runtime | Reported CPU count | Host RAM, GB | GPU RAM, GB | Units/hour |
   |---|---:|---:|---:|---:|
   | L4 | 12 | 53.0 | 22.5 | 1.71 |
   | A100 | 12 | 83.5 | 40.0 | 5.37 |
   | A100 High-Memory | 12 | 167.1 | 80.0 | 7.52 |
   | G4 | 48 | 176.9 | 95.6 | 8.71 |
   | CPU High-Memory | 8 | 51.0 | — | 0.24 |

   These are observations, not guaranteed specifications. The source explicitly distinguishes A100 standard and High-Memory. [Qiita measurement table](https://qiita.com/jin0g/items/8fbfb209ac4e9343c13f).

   **THIRD-PARTY:** An October 14 CLI account independently reports a standard A100-SXM4-40GB allocation with **12 logical CPUs and approximately 83.5 GB host RAM**. March 2026 measurements report T4 1.19, L4 1.71, A100-40 5.40, A100-80 7.52 and G4 8.71 u/h. Older measurements differ substantially, including A100 at 10.59 u/h in 2024. Sources: [Lexidium](https://blog.lexidium.com/cloud/google-colab-cli-a100-gpu), [Chris McCormick](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/), [2024 comparison](https://varlog.info/colab-credit-consumption-comparison-2024/).

   **VERIFIED:** The repository already contains historical descriptions of **12-vCPU A100 hosts**, including A100-40 High-RAM and A100-80. Therefore, “A100 vCPU count entirely unmeasured by us” is too strong; historical records exist, but I did not recover a current standard-versus-high-memory hardware receipt or matched per-core benchmark. See [overnight report:3](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/docs/research/overnight_2026-10-18.md:3) and [A100-80 results:210](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/docs/research/t16_colab_smoke02_results_2026-10-17.md:210).

   **ESTIMATED:** Treat A100 High-Memory primarily as a capacity option until measured otherwise. Extra RAM helps if it removes cache eviction, replay reconstruction, or a worker-count memory limit; it does not establish extra CPU throughput.

3. **The L4 benchmark is real, but 0.036 units/world is a workload-rate estimate—not the complete short-job cost.**

   **VERIFIED:** The A5 artifacts contain:

   | Lane | Completed worlds | Benchmark elapsed | Worlds/hour | Export bytes/world |
   |---|---:|---:|---:|---:|
   | s0 | 20 | 1,259.33 s | 57.17 | 8.88 MB |
   | s1 | 19 | 1,258.00 s | 54.37 | 8.50 MB |

   Both used 12 processes with CPU policy inference. The benchmarks explicitly include unfinished work in the elapsed denominator; these are bounded benchmark samples, not completed 256-world campaigns. Sources: [s0 benchmark:23](/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/manifests-final-3ebf3991/a5-l4-s0-output/exports/benchmark.json:23), [s0 scope:1352](/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/manifests-final-3ebf3991/a5-l4-s0-output/exports/benchmark.json:1352), [s1 benchmark:23](/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/manifests-final-3ebf3991/a5-l4-s1-output/exports/benchmark.json:23).

   **VERIFIED:** The s0 accounting receipt calibrated **1.54 u/h observed versus 2.0 priced**. Full supervisor elapsed times were **2,017.33 and 1,973.17 seconds**; both receipts report PASS and verified stop. See [s0 accounting:7](/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/manifests-final-3ebf3991/a5-l4-s0-output/receipt.json:7), [s0 elapsed:3777](/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/manifests-final-3ebf3991/a5-l4-s0-output/receipt.json:3777), and [s1 elapsed:3547](/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/manifests-final-3ebf3991/a5-l4-s1-output/receipt.json:3547).

   **ESTIMATED:** Applying the 2 u/h ceiling to full supervisor elapsed gives a conservative gross cost proxy of **0.0560 and 0.0577 u/completed world**, versus **0.0350–0.0368** from benchmark throughput alone. Applying 1.54 u/h gives approximately **0.0431–0.0444**. These are elapsed-time calculations, not exact reconciled account debits.

   **ESTIMATED:** Longer chunks can amortize setup, but exports grow with worlds. At 8.7 MB/world, an 800-world round produces approximately **7 GB**. Measure transfer and verification as part of runtime placement; faster computation can expose export as the next bottleneck.

4. **A100 can be worthwhile without reaching unit-cost break-even.**

   **ESTIMATED:** Using **55 completed worlds/hour/L4**, and excluding setup/export for this comparison:

   | Placement assumption | Worlds/hour | u/world | Units per 800 worlds |
   |---|---:|---:|---:|
   | L4, 2 u/h | 55 | 0.0364 | 29.1 |
   | A100-40, 5.4 u/h, equal throughput | 55 | 0.0982 | 78.5 |
   | A100-40, twice L4 throughput | 110 | 0.0491 | 39.3 |
   | A100-80, 7.5 u/h, equal throughput | 55 | 0.1364 | 109.1 |
   | A100-80, twice L4 throughput | 110 | 0.0682 | 54.5 |

   **ESTIMATED:** Unit-cost break-even requires **2.7× L4 throughput** for A100-40 and **3.75×** for A100-80 at the planning prices. Against observed L4 billing of 1.54 u/h, those thresholds rise to **3.51× and 4.87×**.

   **ESTIMATED:** Those are economic-efficiency thresholds, **not mandatory adoption thresholds for this owner**. Suppose one A100-40 delivers twice L4 throughput. Replacing one of three L4 lanes yields **220 versus 165 worlds/hour**. An evenly divisible 800-world job falls from **4.85 to 3.64 hours**, while compute cost rises from **29.1 to 34.2 units**: approximately **5 extra units for 73 minutes saved**. If this moves the next fit or Mac gate earlier, it is attractive.

   **ESTIMATED:** Conversely, an A100 with the same useful throughput consumes more units and occupies the same assignment slot without advancing the result. Expiry does not make that substitution productive.

5. **Price accepted labels separately from root decisions, branch steps and completed worlds.**

   **VERIFIED:** In `teacher_v2`, `decisions` counts root decisions. Labels are conditionally inserted into a separate dictionary; shadow-search counts are recorded separately again. Consequently, the A5 s0 total of **100,000 decisions is not 100,000 searched labels**. See [teacher_v2.py:851](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/research/t34_exit_b8/teacher_v2.py:851), [label output:893](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/research/t34_exit_b8/teacher_v2.py:893), and [search counters:919](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/research/t34_exit_b8/teacher_v2.py:919).

   **ESTIMATED:** Let \(m\) be the mean number of **usable searched labels per completed, validated world**. Then:

   \[
   \text{units/label}=\frac{\text{units/world}}{m}.
   \]

   At the illustrative 55 worlds/hour:

   | Assumption | If \(m=100\): units per million labels | If \(m=1{,}000\): units per million labels |
   |---|---:|---:|
   | L4 | 364 | 36.4 |
   | A100-40, equal world throughput | 982 | 98.2 |
   | A100-40, twice world throughput | 491 | 49.1 |
   | A100-80, twice world throughput | 682 | 68.2 |

   **ESTIMATED:** These are sensitivity calculations, not measured label yields. Report at least **accepted searched labels, decisive/correction labels, anchor-only labels, and rejected labels**. A million easy anchors cannot substitute for a million informative searched decisions.

   **VERIFIED:** The older Phase 0 teacher is a different workload: its final CPU profile is **1.042/1.402/1.657 seconds per search**, with B8 forward taking **0.677/0.664/0.958 seconds**, and CUDA explicitly untested. See [PROFILE.md:43](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/research/phase0_search_20261019/PROFILE.md:43) and [limitations:91](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/research/phase0_search_20261019/PROFILE.md:91).

   **ESTIMATED:** That profile gives CUDA a more plausible opportunity than the earlier Python-dominated shadow benchmark. But it does not establish A100 superiority over L4. The brief’s **129 Mac-hours/million searches** should not be transplanted into T34 v2 pricing; I did not recover its complete concurrency and workload derivation.

6. **A100’s best production opportunity is a qualified cached fit; GPU-based teacher acceleration remains conditional.**

   **VERIFIED:** The earlier A100 records include **15,927 transitions/s** for a self-play row and isolated learner rates of **82.7 updates/s fp32** and **118 bf16**. These are training-shaped measurements, not T34 distillation fits. The portfolio judgment explicitly says fit duration is unmeasured. See [training results:175](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/docs/research/t16_colab_smoke02_results_2026-10-17.md:175), [isolated learner:210](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/docs/research/t16_colab_smoke02_results_2026-10-17.md:210), and [Fable judgment:11](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/docs/research/advice/2026-10-20_compute_portfolio_FABLE_JUDGMENT.md:11).

   **ESTIMATED:** For a fixed fit taking \(T_L\) hours on L4 and achieving end-to-end speedup \(S\):

   \[
   C_L=2T_L,\qquad C_{A40}=5.4T_L/S,\qquad C_{A80}=7.5T_L/S.
   \]

   For example, a **hypothetical 90-minute L4 fit** costs 3 units. At 2× speed, A100-40 takes 45 minutes and costs **4.05 units**; A100-80 costs **5.63 units**. Paying 1.05 extra units to save 45 critical-path minutes would be sensible. If cache construction dominates, the gain may disappear.

   **VERIFIED:** The often-cited **4.35 ms CUDA versus 0.93 ms CPU** comparison combines Colab CUDA measurements with a Mac CPU probe. It is not a controlled same-host device comparison, and the approximately 95% Python diagnosis predates the digest/fan-out changes. See [speed review:3](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/docs/research/advice/2026-10-20_t34_a5_speed_fable.md:3) and [its estimated decomposition:28](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/docs/research/advice/2026-10-20_t34_a5_speed_fable.md:28).

   **ESTIMATED:** If 95% of elapsed work truly remains unaffected host work, infinitely fast GPU execution yields at most **1.053×** overall speedup. A useful CUDA teacher regime therefore requires batching independent inference requests, reducing synchronization and transfer overhead, and keeping the CPU producers supplied—without changing the registered search semantics. Changing precision, horizon, action selection or training batch semantics requires separate qualification.

7. **The supervisor imposes practical limits on the experiment and production schedule.**

   **VERIFIED:**

   - **Three assignments:** the cap includes notebook runtimes, not merely jobs started by this supervisor. [supervisor.py:810](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/research/colab_cli/supervisor.py:810).
   - **High-Memory is not wired through:** allocation currently passes the session and GPU only. Adding a manifest field alone will not request the shape. [supervisor.py:857](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/research/colab_cli/supervisor.py:857).
   - **Transport changes require renewed live acceptance**, estimated in the qualification document at approximately 1.5 units. [QUALIFICATION.md:23](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/research/colab_cli/QUALIFICATION.md:23).
   - **Timeout must be at least twice the estimate**, and its full priced cost must fit the cap, including allocation/bootstrap/export/stop. The inspected validation does not itself enforce a 24-hour maximum. [supervisor.py:187](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/research/colab_cli/supervisor.py:187).
   - **Declared rate is a ceiling:** observed billing is calibrated and reconciled; it does not silently replace the priced spending limit. [accounting.py:184](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/research/colab_cli/accounting.py:184).
   - **The qualified 64 MiB export bound is per file.** Separate approximately 8.7 MB world files fit; aggregate transfer time still matters. [T34 README:128](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/research/t34_exit_b8/README.md:128).

   **ESTIMATED:** Keep productive chunks comfortably below approximately **11 hours expected duration**, so the required doubled timeout and cleanup remain below 24 hours. Prefer smaller independently complete chunks where interruption loss or transfer overhead justifies them. Do not describe final-only artifacts as full-state training resume.

   **VERIFIED:** The historical overnight report documents an A100 retained at approximately 5.3 u/h with only approximately 40 seconds of useful work. The repository also records the October 18 aggregate **6.84 u/h for one A100 plus one L4**. These support measuring allocation cost and stopping promptly, but do not by themselves quantify all Q1 expenditure. [Overnight report:23](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/docs/research/overnight_2026-10-18.md:23), [CLI access record:35](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/docs/research/colab_agent_access_2026-10-18.md:35).

8. **The single most informative measurement is a matched, end-to-end L4 versus standard-A100 teacher placement test.**

   **ESTIMATED — proposed design:** Use the two spare assignments while preserving Round 0’s lane, provided account-wide admission confirms availability. Freeze the same executable, checkpoint hashes, teacher configuration, and **18 disjoint timing worlds—six per mix** before allocation. Complete the same worlds on each runtime; do not compare whichever worlds happen to finish first.

   **ESTIMATED — measurements:**

   - Record CPU model, logical topology, reported physical topology, affinity, CPU quota, RAM, GPU identity/VRAM, allocation delay and calibrated rate.
   - Measure one-worker performance briefly, then use the frozen full-lane process layout.
   - Count validated worlds and usable labels by category.
   - Include setup, compilation, export, Mac verification and release in total turnaround.
   - If the CUDA route is ready and output-qualified, add a small fixed snapshot subtest of CPU versus CUDA search on each host. Otherwise report that route as untested rather than delaying the placement result.

   **ESTIMATED — cost and limits:** Target approximately **45 minutes total elapsed per lane**, with a **90-minute hard timeout**. Price L4 at 2 u/h and A100 at a conservative **8 u/h ceiling**, refusing a higher observed rate. Maximum simultaneous liability is **15 units**, plus a **3-unit contingency reserve**: **18 units total**, about **1.6%** of the remaining balance. At the expected 5.4 u/h A100 rate, a 45-minute pair would consume approximately **5.55 units**. Stop at the first cap; insufficient completed evidence is inconclusive, with no automatic extension.

   **ESTIMATED — prospective decision rules:**

   - **INVALID:** mismatched workload, failed output checks, or unexplained numerical differences.
   - **Adopt for time-sensitive production:** at least **1.25× end-to-end useful-output throughput** over L4, with a conservative forecast showing the next dependent result arrives earlier. Unit-cost break-even is not required.
   - **Prefer broadly on efficiency:** exceed the measured rate ratio—approximately **2.7×** for A100-40 at planning prices.
   - **Do not replace L4:** no meaningful turnaround improvement, or allocation/export overhead erases it.
   - **Inconclusive:** too few matched completions or unstable host performance; retain L4 pending a separately admitted measurement.

   **ESTIMATED:** The 25% threshold is an operational margin against measurement noise and integration effort, not a scientific efficacy gate. It must be frozen before the probe.

9. **Recommended queue: preserve the release path, then buy measured reductions in turnaround.**

   | Priority | Recommendation — **ESTIMATED** | Dependency and stopping rule |
   |---|---|---|
   | 1 | Continue registered Round 0; protect Mac Phase 0 → paced retention → strict/serving sequence. | Existing registrations decide; do not migrate or relabel running work. |
   | 2 | Run the matched A100/L4 measurement above, capped at 18 units including reserve. | Use spare assignments; no automatic retries. |
   | 3 | Qualify the real cached-fit path and compare one fixed fit on L4/A100. | Cache and fixed fit definition ready; measure total fit completion, not update throughput alone; use the existing portfolio’s qualification envelope. |
   | 4 | After a GO permits labels, use the fastest qualified lanes to finish the next label bank; allocate an A100 fit lane only when inputs are ready. | Preserve registered labels, rounds and selection rules. |
   | 5 | Consider a bounded G4 host-throughput probe before investing in A100 High-Memory plumbing. | Independent Fable amendment because host-bound G4 is currently unfunded; hardware and rate must be verified first. |

   **ESTIMATED:** G4 deserves that conditional test because **48 versus 12 logical CPUs**, if translated into even 3× completed-world throughput, would provide approximately **165 worlds/hour in one assignment**. With two L4 lanes, total capacity would be approximately **275 worlds/hour versus 165**—a **40% reduction in divisible-job compute time**. Its hypothetical cost would be **0.0528 u/world at 8.71 u/h**, worse than L4 per unit but potentially excellent per scarce assignment. Neither the host specification nor that speedup is verified for this account.

   **VERIFIED:** The adopted portfolio explicitly excludes A100/G4/H100 for host-bound production. The suggested measurement is therefore a prospective challenge to that hardware exclusion, not an assertion that production use is already funded. [CAMPAIGN_LOG:64](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/docs/research/final_run_20261019/CAMPAIGN_LOG.md:64).

   **ESTIMATED:** I would spend the units by accelerating admitted decisions and advancing the first qualifying candidate, then funding the registered projection-fix and replication work. I would not delay a successful Claim B strict gate merely to finish Claim A research or consume the balance.

10. **Claims I could not verify.**

   **VERIFIED — limits of this review:**

   - No official fixed A100 standard/High-Memory host specifications, physical-core guarantees, unit-price table, allocation probability or preemption rate were found.
   - I did not live-check today’s balance, assignments, expiry schedule or A100 availability.
   - I did not establish current A100/L4 searched-label throughput, valid decisive-label yield, or actual T34 fit duration.
   - I did not verify that “95% CPython” remains accurate after the digest/fan-out changes, or that 12 visible vCPUs correspond to six usable physical cores.
   - I did not recover a complete derivation for 129 Mac-hours per million searches or a complete ledger proving idle allocation was the dominant cost of all Q1 work.


## Call 2 — prompt (verbatim, cross-examination)

```text
You are the same independent advisor (Astra) for the snake-dqn project, lens: Colab operations and unit economics. Same rules as before: read-only — do NOT edit, commit, or create files in this repository; at most tiny read-only probes on this Mac (1 thread, `nice -n 10`, each under 30 s; a 6-worker study is running and must not be disturbed); no Colab/RunPod jobs; web search for public facts is fine. Put your FULL answer in your final message with file:line evidence and numbers; label each claim VERIFIED / THIRD-PARTY / ESTIMATED. Context: docs/research/advice/2026-10-20_a100/FACTS_AND_QUESTION.md and the sources it names.

Your previous answer (fresh session, so here is its substance) recommended: keep L4 as production default; fund a small matched L4-vs-standard-A100 teacher placement test (18 disjoint timing worlds, 6 per mix, ~45 min per lane, ≤ 18 u incl. reserve) using the two spare assignments; adopt A100 for time-sensitive production at ≥ 1.25× end-to-end useful-output throughput even though unit-cost break-even needs ~2.7× (A100-40) / 3.75× (A100-80); A100's strongest likely role is a qualified cached student fit (fit duration unmeasured; cache construction may dominate); third-party Feb/Oct 2026 observations put both L4 and A100 standard/high-mem hosts at 12 logical CPUs (A100 83.5 / 167 GB RAM) and G4 at 48 logical CPUs, 8.71 u/h; you suggested a bounded G4 host-throughput probe might beat A100 high-mem plumbing because 48 vs 12 vCPU could give ~3× worlds/h per assignment; you noted the supervisor does not wire `--high-mem` (research/colab_cli/supervisor.py ~line 857), timeout must be ≥ 2× estimate (~line 187), and the 3-assignment cap counts notebook runtimes (~line 810); labels per world (m) is unmeasured so units/label is a sensitivity table only; the repo's "4.35 ms CUDA vs 0.93 ms CPU" is a cross-host comparison; if 95% of elapsed remains host work, infinite GPU gives at most 1.053×.

Now cross-examine your own advice as a hostile but fair reviewer. In particular:
1. If third-party data says A100 hosts have the SAME 12 logical CPUs as L4, what is your honest prior that the matched A100 placement test shows ≥ 1.25× for host-bound teacher worlds? If that prior is low, is the 18-unit probe still the best first measurement for THIS lens, or does the question collapse into "which runtime has the most usable host cores per assignment slot" — and what does the public record say about per-core speed (CPU model/SKU, SMT, cgroup quota) on L4 vs A100 vs G4 vs CPU high-mem hosts? Check: does the standard A100 runtime actually report 12 vCPU, or does `--high-mem` change the CPU count as well as RAM?
2. Your G4 suggestion: is "48 logical CPUs" verified by more than one source, is it consistent with the ~8.7 u/h price and the 3-assignment cap, and what would it take in this repo (torch cu128 for sm_120, the colab_cli hardware check, manifest rate ceiling) to run a host-throughput probe on G4 safely? Price that probe.
3. "A100's best role is the cached fit": the fits are for a ~1.64M-param CNN on ~16–20k rows/world × hundreds of worlds. Is it plausible that such a fit is GPU-bound enough on an L4 that an A100 shortens the critical path, or will data loading / cache build dominate? What number from the repo (or a ≤ 30 s probe) would tell you?
4. The owner's constraint is that ~1,157 units expire by ~11-20 and the goal is a better agent soonest. Given everything above, what would change your mind in either direction (toward using A100/G4 heavily, or toward never touching them), and what is the single measurement that would settle the A100 question for this project? State its exact cost in units and wall-clock, what it must record, and the pre-registered decision rule.
5. List the three weakest or least-verified claims in your previous answer and say how each should be corrected or bounded.

Open-ended; do not restrict yourself to options already discussed.
```

## Call 2 — Astra answer (verbatim)

**ESTIMATED — I would withdraw the 18-unit A100 teacher-placement test as the first measurement.** My subjective prior that standard A100 delivers a reproducible **≥1.25× end-to-end gain on the unchanged, host-bound teacher is about 10–20%**. That is a judgment, not a calibrated probability. The stronger immediate questions are whether G4 provides materially more **usable host throughput per assignment**, and whether a real L4 student fit spends enough time executing GPU updates to justify an A100 comparison.

**VERIFIED — Scope:** I reviewed clean, detached snapshot `6f1e751f4ec11c280c211634cd09de924b649218`. No repository files were changed, no benchmarks or tests were run, and no Colab/RunPod jobs were launched. “VERIFIED” below means verified in inspected code/documents or official documentation; recorded campaign results were not independently rerun. “THIRD-PARTY” means an external observation; “ESTIMATED” includes arithmetic, inference, and prospective proposals.

**1. The A100 teacher prior should be low; CPU count alone still does not settle placement.**

**THIRD-PARTY — Standard A100 really is reported with 12 logical CPUs.** The February 2026 measurements distinguish the following configurations:

| Runtime | Reported logical CPUs | Host RAM | GPU RAM | Reported units/hour |
|---|---:|---:|---:|---:|
| CPU standard | 2 | 12.7 GB | — | 0.07 |
| CPU high-memory | 8 | 51.0 GB | — | 0.24 |
| L4 | 12 | 53.0 GB | 22.5 GB | 1.71 |
| A100 standard | 12 | 83.5 GB | 40 GB | 5.37 |
| A100 high-memory | 12 | 167.1 GB | 80 GB | 7.52 |
| G4 | 48 | 176.9 GB | 95.6 GB | 8.71 |

Thus, **high-memory changes CPU count for CPU-only runtimes in this observation, but not for A100**. An October 14 CLI report independently describes a standard A100-40GB with 12 logical CPUs and 83.5 GB RAM. These are observations, not allocation guarantees. [February measurements](https://qiita.com/jin0g/items/8fbfb209ac4e9343c13f), [October standard-A100 observation](https://blog.lexidium.com/cloud/google-colab-cli-a100-gpu).

**VERIFIED — The repository already supports the same broad conclusion.** Its older measurements record A100-40GB with **12 vCPU/6 physical cores**, A100-80GB with **12 vCPU**, and L4 with **12 vCPU**. Therefore, “A100 host CPU count is unmeasured” should mean *unmeasured for the contemplated new assignment*, rather than absent from the project’s history. [T16 results:3](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/docs/research/t16_colab_smoke02_results_2026-10-17.md:3), [T16 results:210](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/docs/research/t16_colab_smoke02_results_2026-10-17.md:210), [T16 results:497](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/docs/research/t16_colab_smoke02_results_2026-10-17.md:497).

The public record on **per-core performance** is substantially weaker:

- **THIRD-PARTY:** A published experiment describes L4 and A100 using the same generic **Intel Xeon @2.20 GHz, six-core** host configuration. That establishes neither an exact SKU nor matched per-core throughput. [Experiment hardware description](https://kth.diva-portal.org/smash/get/diva2%3A1985582/FULLTEXT01.pdf).
- **THIRD-PARTY:** CPU high-memory observations include **Xeon @2.20 GHz, four physical/eight logical cores**, while another study reports **AMD EPYC 7B12**. There is no defensible single CPU SKU to assign to “CPU high-memory.” [Xeon experiment](https://www.mdpi.com/2224-2708/15/5/72), [EPYC experiment](https://www.sciencedirect.com/science/article/pii/S0306261926005842).
- **VERIFIED, with a strict scope boundary:** Google Cloud documents G2/L4 and A2/A100 machine families on Cascade Lake, and G4 on AMD EPYC Turin. Its `g4-standard-48` has 48 hardware threads, 180 GB RAM and one 96 GB GPU. **ESTIMATED:** The resemblance to reported Colab G4 resources suggests a newer host platform, but does not prove Colab’s SKU, topology, quota or per-core speed. [Google Cloud machine specifications](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines).
- **ESTIMATED / unresolved:** I found no adequate matched public measurement of this teacher’s per-core performance or effective cgroup quota across those runtime types.

**ESTIMATED — The optimization target is therefore useful work per assignment-hour**, approximately:

\[
\text{usable CPU capacity}\times\text{work per CPU-second}\times\text{parallel efficiency},
\]

with setup, memory pressure, export and allocation delay included. “Most logical CPUs” is only one input.

**VERIFIED — The 95% premise also needs narrowing.** It comes from a pre-fix analysis explicitly labelled an estimate; its CPU timings were measured on a loaded Mac. The later engine changed digest handling and process layout. Consequently, **1.053× is the correct Amdahl ceiling if 95% remains unaffected host work**, but 95% is not a freshly measured post-fix fraction. [Speed review:8](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/docs/research/advice/2026-10-20_t34_a5_speed_fable.md:8), [Speed review:28](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/docs/research/advice/2026-10-20_t34_a5_speed_fable.md:28), [Campaign log:55](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/docs/research/final_run_20261019/CAMPAIGN_LOG.md:55).

**ESTIMATED — My low prior applies to unchanged teacher code on comparable 12-vCPU allocations.** Different batching, a different CPU quota, or a faster host could change the answer. Those would be mechanisms to measure, not evidence that the A100 GPU itself accelerated the teacher.

**2. G4 is a credible host-capacity hypothesis, but “48 CPUs means 3×” remains unproved.**

**THIRD-PARTY — Yes, more than one source reports 48 host CPUs.** Besides the February measurements, an October 5 OCR experiment reports G4 with **48 host CPUs**, versus 12 on L4 and A100. However, the OCR article explicitly imports its GPU prices from another source; it is not an independent October billing measurement. Chris McCormick’s own March observations separately report **8.71 u/h**. [OCR experiment, hardware section](https://withinnoise.dev/blog/posts/ocr-benchmark/), [McCormick’s measured rates](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/).

**VERIFIED — The three-assignment limit is enforced account-wide by this repository, including notebook assignments.** It does not promise that Google will supply three G4s, or even one. Google states that hardware availability and usage limits vary. There is no inconsistency between one expensive 48-thread assignment and a three-assignment broker limit. [Supervisor:817](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/research/colab_cli/supervisor.py:817), [Colab resource limits](https://research.google.com/colaboratory/faq.html).

**ESTIMATED — At 3× L4 throughput, G4 would buy assignment capacity, not unit savings.** Using the public rates, unit-cost break-even is:

| Placement | Required speedup over L4 at 1.71 u/h |
|---|---:|
| A100-40 at 5.37 u/h | 3.14× |
| A100-80 at 7.52 u/h | 4.40× |
| G4 at 8.71 u/h | 5.09× |

My previous **2.7×/3.75×** figures used a **2 u/h L4 planning ceiling**. They are valid on that basis, but should not have been mixed with a 1.7 u/h observed-rate comparison.

**ESTIMATED — Slot compression could nevertheless matter.** At an illustrative 55 worlds/h per L4, three L4s produce 165 worlds/h. If one G4 genuinely produces 3× that per-lane rate, one G4 plus two L4s produces 275 worlds/h. An 800-world backlog then takes approximately **2.91 rather than 4.85 hours**, before overhead. That benefit exists only when the backlog is admitted, parallelizable, and on the critical path.

**VERIFIED — G4 support is already largely implemented:**

- `gpu="G4"` and `allowed_gpus=["G4"]` are accepted. [Supervisor:175](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/research/colab_cli/supervisor.py:175).
- Bootstrap selects **torch 2.9.1/cu128** for compute capability ≥10, covering the proposed sm_120 path. [Bootstrap:23](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/research/colab_cli/bootstrap.sh:23).
- The hardware classifier broadly maps otherwise-unrecognized capability ≥10 devices to G4. Therefore the probe should additionally verify the received **GPU name, capability and memory**, rather than treating that classification as exact SKU proof. [Supervisor:1363](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/research/colab_cli/supervisor.py:1363).
- Host admission checks `os.cpu_count()`, not usable CPU quota. [Supervisor:1351](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/research/colab_cli/supervisor.py:1351).
- Round 0 remains explicitly L4/K12. A G4 experiment needs a separate timing-only registration and manifest. [Round 0 draft:125](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/research/t34_round0/draft.py:125).

**ESTIMATED — My proposed G4 discovery probe:** one G4 plus one matched L4; CPU inference and one thread per worker; frozen timing fixtures covering all mixes and representative long-body states. Record CPU model/family/stepping, SMT topology, affinity, effective cpuset, the tightest quota along the process’s cgroup ancestry, throttling, CPU steal, RSS and useful throughput. Measure one-worker performance and scaling at **K=6/12/24/48**, capped by actual usable capacity.

**VERIFIED — Eighteen worlds cannot test 48-worker capacity:** the benchmark sets `effective_nproc=min(nproc,max_worlds)`. Use at least 48 resident timing worlds, preferably a larger balanced pool. [Benchmark:298](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/research/t34_exit_b8/benchmark.py:298).

**ESTIMATED — Price:** a **30-minute whole-session target per lane**, with **60-minute manifest timeouts**, at ceilings **L4 2/G4 9 u/h**, gives:

- Nominal priced consumption: **5.5 u**.
- Combined operational caps: **11 u**.
- Shared 20% reserve: **2.2 u**.
- Prospective envelope: **13.2 u**, with up to two assignment-hours.

This is a discovery envelope, not a guarantee that complete-world qualification fits inside it. Fixed-duration scaling windows may establish host capacity; they must not be reported as completed-world throughput. I would require **≥2× matched sustained useful throughput**, valid outputs and a material projected backlog reduction before proposing production G4 placement. Completion-rate validation would remain necessary.

**3. “A100’s best role is cached fitting” should become “cached fitting is its most plausible untested role.”**

**VERIFIED — The biggest factual correction is the row count.** The reported **16–20k rows/world are executed teacher simulation row-steps**, not demonstrated fitting examples. The benchmark names that quantity explicitly. The fitter selects searched rows plus an equal number of anchors, and Round 0 expressly supplies no fitting labels. Multiplying 16–20k by hundreds of worlds does not establish the training-set size. [Benchmark:381](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/research/t34_exit_b8/benchmark.py:381), [Fit sampler:80](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/research/t34_exit_b8_fit/data.py:80), [Round 0 README:17](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/research/t34_round0/README.md:17).

**VERIFIED — The fitting path has credible host bottlenecks:**

- Cache construction precedes optimizer updates and iterates replayed worlds.
- Cached loading still performs mmap access, metadata parsing, row normalization, sampling and batch assembly.
- CUDA uses one-batch pinned prefetch, without loader worker processes.
- Epoch timing includes held-out scoring and gameplay probes.
- `gpu_busy_percent` is currently always null; it is not sampled.

[Cache construction:188](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/research/t34_exit_b8_fit/cache.py:188), [Cached loading:370](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/research/t34_exit_b8_fit/cache.py:370), [Prefetch:242](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/research/t34_exit_b8_fit/data.py:242), [Epoch/probe timing:251](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/research/t34_exit_b8_fit/train.py:251), [Timing output:305](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/research/t34_exit_b8_fit/train.py:305).

**VERIFIED — Production fit duration is unmeasured.** The saved cache smokes used **16 synthetic rows and eight optimizer updates per arm**; CUDA prefetch was not qualified. The README also identifies the production v2 replay-source/probe boundary as unfinished. [Fit README:46](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/research/t34_exit_b8_fit/README.md:46), [Fit README:90](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/research/t34_exit_b8_fit/README.md:90).

**ESTIMATED — Parameter count does not rule out GPU boundedness.** A 1.64M-parameter CNN at batch 1,024 can perform substantial convolution work. Conversely, millions of examples can create substantial loading work. The measured synthetic cache densities imply approximately **12.7 GB per million actual S1b rows**, or **66.2 GB for S1a**, before production metadata differences. These are storage estimates, not a measured dataset inventory. [Measured bytes/row:104](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/research/t34_exit_b8_fit/README.md:104).

**ESTIMATED — The decisive first number is the accelerable fraction \(p\) of a complete real L4 fit**, including cold cache preparation and probes. If A100 accelerates that portion by \(s\):

\[
S_{\mathrm{whole\ fit}}=\frac{1}{(1-p)+p/s}.
\]

For **1.25× overall speedup**, \(p\) must be at least **40% with a 2× GPU leg**, or **30% with a 3× GPU leg**. If nonaccelerated work occupies 90%, even an infinitely fast GPU gives only **1.11×**.

**VERIFIED — This is already the binding design’s sequence:** cache observations, time one full L4 fit, then consider A100-40 if the fit remains GPU-bound beyond roughly one hour. [Design amendment:70](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:70). **ESTIMATED:** A ≤30-second Mac probe cannot answer that CUDA placement question; another synthetic Mac timing would add little.

**4. What I would measure next, and what would change my mind.**

**ESTIMATED — The expiring balance increases the value of saving decision time, but does not make higher consumption valuable by itself.** At the planning rate of 2 u/h, one continuously useful L4 would consume **1,157 units in about 24.1 days**. The scarce resource may be admitted experiments, engineering readiness or Mac qualification time—not the ability to spend units. The snapshot records **1,157.6 units**, rather than a live balance checked in this review. [Campaign log:78](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/docs/research/final_run_20261019/CAMPAIGN_LOG.md:78).

My revised dependency order is:

| Priority | Proposed action — ESTIMATED | Decision |
|---|---|---|
| 1 | Inventory actual fit rows and instrument the first qualified L4 fit | Determine whether A100 has meaningful work to accelerate |
| 2 | Separately judged G4 host probe, **13.2 u envelope** | Test usable host capacity per assignment |
| 3 | A100 paired fit diagnostic only after the L4 profile supports it | Test fixed-recipe fit acceleration |
| Conditional | CPU high-memory support and measurement | Cheap host work when assignment slots and latency permit |

**VERIFIED — CPU high-memory needs more preparation than G4:** allocation currently has no `--high-mem` wiring. Transport changes require fresh acceptance, documented at approximately **1.5 u**, plus implementation/review effort. [Allocation:857](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/research/colab_cli/supervisor.py:857), [Qualification:23](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/research/colab_cli/QUALIFICATION.md:23). **ESTIMATED:** Its reported eight CPUs/0.24 u/h make it worth retaining as an option, but a slower cheap lane can be the wrong use of a scarce slot.

**ESTIMATED — Toward heavy A100/G4 use:** I would change my mind for a verified ≥2× G4 host-throughput gain on the actual work, or a measured A100 fit gain that removes hours from admitted repeated fits. Memory capacity that prevents a demonstrated L4 failure is another valid reason. The gain must survive setup, data preparation, validation and export.

**ESTIMATED — Toward avoiding them for this campaign:** unchanged usable host capacity; severe G4 throttling; A100 GPU acceleration hidden behind cache/loading/probes; allocation delay larger than the saved runtime; or a Mac-only gate determining the next decision regardless of remote speed. “Never touch A100” would still be too broad—these are workload-specific conclusions.

**ESTIMATED — The single A100 measurement I would fund is a paired, phase-resolved replay of the real fitting pipeline.** The primary quantity is **time saved for the same qualified fitting work**, not GPU utilization alone.

A concrete bounded diagnostic would freeze:

- The same real, production-sized source/cache inventory, checkpoint, code, batch order, precision and optimizer recipe.
- Cold cache preparation/verification on both hosts, followed by **1,024 identical optimizer updates at batch 1,024**—**1,048,576 row presentations**, not necessarily unique examples.
- Four consecutive 256-update reporting blocks; identical correctness checks and a timing artifact export. These are timing blocks, not independent host replicates.
- One L4 and one standard-A100 request; record the actual A100 variant. No high-memory substitution claim and no synthetic-data replacement if the real source is unavailable.

Record separately: allocation/setup, replay/cache writing/hashing, selected-row counts, loader wait, H2D, GPU forward/backward/Adam time, scoring/probes, export/release, RAM/VRAM, CPU quota/throttling, actual rate and attributable debit. Also retain hashes and numerical qualification results.

**ESTIMATED — Exact prospective envelope:** using conservative ceilings **2 u/h L4 and 8 u/h A100**, a 30-minute whole-session target with 60-minute timeouts gives **5 u nominal**, **10 u combined operational caps**, and **2 u shared reserve: 12 u total reserved**. At the external 1.71/5.40 rates, two 30-minute sessions would consume **3.555 u before any overrun**.

**VERIFIED — An exact future debit or completion time cannot honestly be promised.** Timeout must be at least twice the estimate and fully priced. Reserve is held liquidity, not an automatic extra spending allowance; release cleanup can continue beyond the job deadline. The 30-minute target therefore needs confirmation from the L4 preparation timing before admission. [Manifest validation:187](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/research/colab_cli/supervisor.py:187), [Reserve accounting:824](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/research/colab_cli/supervisor.py:824), [Cleanup:621](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/research/colab_cli/supervisor.py:621).

**ESTIMATED — Proposed pre-data decision rule:**

1. **INVALID/INCOMPLETE:** mismatched inputs/recipe, failed numerical qualification, missing phases, timeout, or unverified export/release. No throughput claim from a partial packet; no automatic extension.
2. **NO ADOPTION:** measured complete-packet speedup ≤1.10, or the measured phase model predicts less than one hour saved across the currently admitted fit backlog.
3. **ADOPT FOR THAT FIT CONFIGURATION:** complete-packet speedup ≥1.25, all four update blocks favor A100, correctness passes, and the measured phase model predicts ≥1 hour saved across admitted fits within their caps.
4. **INCONCLUSIVE:** everything between those outcomes; retain L4 without extending the test.

**ESTIMATED — This settles the practical fit-placement decision provisionally, not every A100 question.** A small packet cannot prove fleet-wide performance or a full production round’s runtime. The first full fit must confirm the projected saving. The host-bound teacher remains a separate placement claim.

**VERIFIED — These proposals would change the adopted portfolio, which explicitly excludes A100/G4/H100 for host-bound work.** They need the independent Fable ruling specified by the current instructions before execution. Nothing here amends Round 0 or authorizes a launch. [Portfolio exclusions:76](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-5/docs/research/final_run_20261019/CAMPAIGN_LOG.md:76).

**5. The three weakest claims in my previous answer, corrected.**

| Previous claim | Correction |
|---|---|
| **An 18-unit A100 teacher comparison is the best first measurement.** | **ESTIMATED:** Low prior for a ≥25% gain on comparable 12-vCPU hosts. Profile the actual fit and test G4 host capacity first. An A100 comparison becomes valuable when a measured mechanism could shorten the critical path. |
| **G4’s 48 logical CPUs could yield roughly 3× worlds/hour.** | **THIRD-PARTY + ESTIMATED:** The CPU count has multiple reports; usable quota, per-core speed, scaling and completed-world throughput remain unverified. An 18-world bank cannot saturate 48 workers. Treat 3× as a scenario, not a forecast. |
| **A100’s strongest role is cached fitting.** | **ESTIMATED:** It is the most plausible untested role. Actual fitting-row volume and production duration are unknown; 16–20k simulation steps/world are not fitting rows. Cache construction, Python batch assembly and probes may dominate. |



## Prompter notes (Fable, ≤ 10 lines)

1. Astra withdrew its own call-1 first measurement under cross-examination: prior for A100 ≥ 1.25× on the host-bound teacher is ~10–20% because A100 standard and high-mem hosts report the same 12 logical CPUs as L4 (third-party, two sources; repo T16 records agree).
2. Net recommendation: (a) inventory real fit rows and instrument the first qualified L4 fit (16–20k rows/world are sim row-steps, not fit rows); (b) a separately judged G4 host-capacity probe (~13.2 u envelope, ≥ 48 timing worlds, K=6/12/24/48) since G4 reports 48 logical CPUs; (c) an A100 paired phase-resolved fit diagnostic (~12 u reserved) only if the L4 fit profile shows ≥ 30–40% GPU-accelerable time.
3. Break-even corrected to observed-rate basis: A100-40 3.14×, A100-80 4.40×, G4 5.09× over L4 at 1.71 u/h — any A100/G4 use buys wall-clock per assignment slot, never units.
4. Useful verified repo facts: `--high-mem` not wired in supervisor.py ~857; 3-assignment cap counts notebook runtimes (~810/817); G4 cu128 bootstrap already present (bootstrap.sh:23); benchmark `effective_nproc=min(nproc,max_worlds)` (benchmark.py:298); fit `gpu_busy_percent` never sampled.
5. Caveats: all public A100/G4 host specs are third-party observations (no official table); the "95% CPython" fraction is pre-fan-out and unmeasured post-fix; G4 "3×" is a scenario, not a forecast. Both answers respect the portfolio's A100/G4 exclusion and require a Fable ruling before any launch.
