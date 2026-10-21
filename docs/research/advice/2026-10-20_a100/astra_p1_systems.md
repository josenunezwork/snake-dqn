# Astra P1 — systems-performance lens (2026-10-20, ~17:30–18:30 PT)

Advisor: Astra (`CODEX_MODEL=gpt-6-astra`, effort high, fast mode) in worktree `../snake-dqn-advice-a100-1` (detached at
6f1e751f). Prompter: Fable advisor-prompter, lens = where end-to-end wall-clock goes (teacher labels, Round 0,
distillation, gates) and whether an A100 runtime's host or GPU changes it. Question: `FACTS_AND_QUESTION.md` (same folder).
Raw outputs: `snake-dqn-artifacts/a100-advice/p1-call{1,2}/` (prompt.md, run.log, *.last-message.md). Read-only; no jobs.

## Call 1 — prompt (verbatim)

```text
You are an independent systems-performance advisor for the snake-dqn project (this worktree is a read-only copy of main at 6f1e751f). Advice only.

RULES (strict): read-only. Do NOT edit, commit, or create files anywhere in this repository or worktree. You may run at most tiny read-only probes on this Mac: 1 thread, prefixed with `nice -n 10`, each under 30 seconds — the Mac is currently running a 6-worker study (tmux `phase0-selection`) plus a Colab supervisor and must not be disturbed. No Colab, RunPod, tmux, or network jobs; no installs. Put your FULL answer in your final message (nothing elsewhere), with file:line evidence and numbers, and label every claim as VERIFIED (you read it in the repo/artifacts or measured it) or ESTIMATED.

START by reading `docs/research/advice/2026-10-20_a100/FACTS_AND_QUESTION.md` (goal, facts, sources). Then read as much of the named sources as you need: `docs/research/final_run_20261019/HANDOFF_2026-10-20.md`, `docs/research/final_run_20261019/CAMPAIGN_LOG.md`, `docs/research/advice/2026-10-20_t34_a5_speed_fable.md`, `docs/research/advice/2026-10-20_t34_a5_speed_astra.md`, `docs/research/advice/2026-10-20_compute_portfolio_FABLE_JUDGMENT.md`, `docs/research/specs/2026-10-19_phase0_*`, `docs/research/specs/2026-10-19_t34_v2_fitcache.md`, `docs/research/specs/2026-10-20_t34_engine_fanout.md`, `research/t34_round0/` (README.md, sizing.py, runner.py), `research/t34_exit_b8/`, `research/t34_exit_b8_fit/`, `research/phase0_search_20261019/` (PROFILE.md if present), `research/colab_cli/`, `src/model/ego2s_network.py`, the GridBatchSim sources under `src/simd_env/`. Artifacts (receipts, polls, exports) live under `/Users/josenunez/Projects/ml/snake-dqn-artifacts/` (e.g. `t34-engine-fanout-20261020/manifests-final-3ebf3991/a5-l4-s*-output/exports/benchmark.json`, `t34-a5-speed-review/`, `phase0-search-20261019/`).

GOAL: a better served snake agent than the current Watch hero (bare B8 seed 3), proven by the usual strict gate on the Mac, as soon as possible in wall-clock time. Colab units are prepaid (~1,157 left, owner wants them spent by ~2026-11-20 on decision-producing work), so units are not the scarce resource; wall-clock, quiet-Mac hours, implementer days and the 3-concurrent-runtime cap are.

OPEN QUESTION (answer it through the SYSTEMS-PERFORMANCE lens only): Where does end-to-end wall-clock time actually go on the approved T34 path — Phase 0 premise test (running on the Mac), Round 0 (teacher acting vs bare B8, 240 paired worlds, L4, ~3.5 h per 100-pair chunk, 3 chunks serial on 1 lane), the label volume for DAgger distillation (~1M searched decisions per round, estimated ~129 Mac-hours at the pre-speedup teacher rate; CUDA throughput of the search-teacher rollouts never measured), the student fits, the screens, and the Mac-only strict gate — and would a Colab A100 runtime (40 GB ~5.4 u/h or 80 GB ~7.5 u/h vs L4 ~1.7 u/h; host vCPU count, per-core speed and RAM of an A100 runtime are UNMEASURED; L4 runtime = 12 vCPU ≈ 6 physical, Colab CPU-only runtime = 2 vCPU) change any of those stages' wall-clock, and through which mechanism (host cores, per-core speed, RAM, GPU batching of rollouts, something else)? Build the critical path from the measured numbers (per-stage hours, what is serial vs parallel, what is Mac-bound vs Colab-bound) and say what single thing most shortens time-to-a-better-agent. Known measured facts: teacher/benchmark work is ~95% single-threaded CPython (state digest, deepcopy, v8 opponent hooks, raster glue), L4 GPU 2–8% util, CUDA batch-1 forward 4.35 ms vs 0.93 ms CPU because of sync; after the fan-out fix (12 single-thread processes per L4 runtime, W1, O(bytes) digest) a lane does 57.2 / 54.4 completed teacher worlds/h at ~0.036 u/world, ~16–20k rows and ~8.7 MB per world; Phase 0 teacher was ~4.5 s per searched decision on one Mac core before a 3.4–4.2x speedup (B8 inference dominant there).

Be concrete: give a per-stage wall-clock table with your confidence, name the binding resource of each stage with evidence, and state what you would need to measure (and how cheaply) to replace each ESTIMATED entry with a VERIFIED one. Do not propose a list of options for us to pick from; give your best single judgment and its reasoning, plus what would change your mind.
```

## Call 1 — Astra answer (verbatim)

**ESTIMATED — My judgment: protect the quiet-Mac Path B → strict-gate route as the first opportunity to deliver a better served hero, while Colab handles the development work in parallel. That is the single largest potential reduction in time-to-release: a successful assisted controller can bypass the label–fit–screen rounds entirely. An A100 is not currently a demonstrated shortcut. Its most plausible immediate benefit would be a faster host per occupied runtime slot; its GPU becomes relevant only for measured batched search or cached student fits.**

**VERIFIED — Scope:** I inspected the clean, detached worktree at `6f1e751f4ec11c280c211634cd09de924b649218`, the named sources, and saved A5 artifacts. I made no files or edits and launched no workloads, network operations, or runtime commands. Below, **VERIFIED** means source/artifact evidence inspected, not independently rerun; **ESTIMATED** includes projections, interpretations, and recommendations.

**VERIFIED — Two corrections materially change the critical path.**

1. **129 Mac-hours already includes the Phase 0 teacher speedup.** The report projects **360.02 single-core hours / 128.98 six-worker wall-hours** for one million searched decisions, using the optimized teacher and a measured **2.791×** six-task scaling factor. It excludes collection, storage, and fitting. Dividing 129 hours by another 3.4–4.2 would double-count the optimization. [SPEED_REPORT.md:47–66](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/research/phase0_search_20261019/SPEED_REPORT.md:47)

2. **The current shadow/LAD teacher is a different workload from one million independent N8/H128 searches.** The amended design obtains confirmation labels from shared shadow trajectories and searches around impending deaths; confirmations are anchors, not independent decisive examples. The planned label rounds contain **800 worlds**, with R1 → R2 and conditional R3, rather than an unconditional five-round chain. [DESIGN_V2_SHADOW.md:24–40](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:24), [amended label rule:82–86](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:82), [FINAL_PLAN.md:59–62](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/final_run_20261019/FINAL_PLAN.md:59)

**ESTIMATED — The defensible per-stage schedule is therefore the following.** Times are execution estimates, not promises of scientific success. Confidence describes timing confidence.

| Stage | Wall-clock estimate and confidence | Binding resource and parallelism | What would replace the estimate with measured evidence |
|---|---|---|---|
| **Phase 0, already running** | **ESTIMATED:** original optimized forecast **32.69 h**, or **39.23 h with reserve**. The Oct 20 17:30 snapshot gives **16.5–19.5 h remaining**, to Oct 21 10:00–13:00. Medium confidence. | **VERIFIED:** six Mac workers; batched B8 inference dominates the optimized search fixtures. Runs concurrently with Colab Round 0. [Speed report:59–66](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/research/phase0_search_20261019/SPEED_REPORT.md:59), [handoff:155–157](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/final_run_20261019/HANDOFF_2026-10-20.md:155) | Read the existing terminal timing/progress receipt when it finishes; no new probe is needed. The quoted ETA is the saved snapshot, not a live status claim. |
| **Round 0: 240 paired worlds** | **ESTIMATED:** frozen forecast **6.76 h**, including three 15-minute allowances. Observed early pace implies about **8.4 h plus remaining chunk/setup/publication overhead**. Medium-low confidence. | **VERIFIED:** **100/100/40** pairs, three serial chunks on one L4, **12 single-thread CPU processes** within each chunk. Each pair includes bare and teacher execution. [Registration:871–929](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/research/t34_round0/REGISTRATION_ROUND0.json:871), [campaign log:62](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/final_run_20261019/CAMPAIGN_LOG.md:62) | Use chunk 000’s completed receipt to separate teacher, bare, metric replay, export, and startup time; retain all three terminal receipts for realized total duration. |
| **Round 0 Mac replay and analysis** | **ESTIMATED:** duration unmeasured; a serial tail after collection, with possible Mac admission delay. | **VERIFIED:** replay selects **10% within each chunk**, both arms—**24 pairs / 48 traces** here. Publication barriers precede subsequent phases. This is trace replay, not rerunning every search. [README:156–175](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/research/t34_round0/README.md:156), [README:221–225](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/research/t34_round0/README.md:221) | Time the required replay when admitted. A few excluded trace replays would narrow the forecast; only the full receipt verifies realized hours. |
| **Path B timing, pre-check, paced retention** | **ESTIMATED:** approximately **20.7 h** for Mac bare+paced efficacy after moving unpaced work to Colab; pre-check originally **~1.5 h**; timing-gate duration additional and unmeasured. Low confidence. **VERIFIED caps:** efficacy **40 h**, pre-check separately **8 h**. | **VERIFIED:** waits for Phase 0 and quiet Mac admission. Paced operation and latency tails are intrinsic to the question; A100 cannot replace them. Unpaced descriptive work may run on Colab concurrently. [Campaign log:45](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/final_run_20261019/CAMPAIGN_LOG.md:45), [design:93–109](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:93) | Existing timing/pre-check/efficacy receipts, once run. Current cost forecasts are prefix extrapolations, not complete-world timings. [Astra review:70–75](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/advice/2026-10-20_t34_a5_speed_astra.md:70) |
| **Current T34 label collection, each 800-world round** | **ESTIMATED:** **14.0–14.7 lane-hours** of collection: **7.0–7.4 h on two lanes**, **4.7–4.9 h on three**, before setup/export, imbalance, and replay/cache work. Medium-low confidence. | **VERIFIED:** A5 measured **54.4–57.2 completed teacher worlds/h/lane**, CPU inference, K12/W1. **ESTIMATED binder:** host execution and available runtime slots. R2 depends on the selected R1 student. [Campaign log:57](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/final_run_20261019/CAMPAIGN_LOG.md:57), [FINAL_PLAN:89](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/final_run_20261019/FINAL_PLAN.md:89) | First actual production label chunk: completed worlds/h **and decisive labels/world**, by mix and length, plus complete export time. Re-measure after the student changes; A5’s B8 distribution does not establish R2 throughput. |
| **Observation cache + student fits** | **ESTIMATED:** old allowance **4 h per fit**; R1 has two fits, R2 one. Two R1 fits could take ~4 h concurrently or ~8 h serially, subject to slot availability. Low confidence. | **VERIFIED:** cache regeneration occurs once, then mmap-backed epochs; CUDA prefetch is implemented but unqualified. **ESTIMATED binder:** cache replay/build, then GPU training or host feeding—the production balance is unknown. [FINAL_PLAN:59–61](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/final_run_20261019/FINAL_PLAN.md:59), [fit README:8–31](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/research/t34_exit_b8_fit/README.md:8) | One actual cache build and cache-enabled L4 fit, recording regeneration seconds, epoch time, probe time, GPU busy, RSS, VRAM, and export. The existing 16-row smokes cannot price this stage. |
| **Development screens** | **ESTIMATED:** old Mac allowance **~1 h per 160-pair bank**. No matched production L4/A100 screen timing established here. Low confidence. | **VERIFIED:** adopted portfolio places bare-student development screens on L4; these are not Mac-only strict evidence. **ESTIMATED binder:** simulation/observations/opponents versus batched inference, requiring measurement. [FINAL_PLAN:42](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/final_run_20261019/FINAL_PLAN.md:42), [portfolio:36–46](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/advice/2026-10-20_compute_portfolio_FABLE_JUDGMENT.md:36) | A short excluded full-horizon paired fixture using the actual screen runner, then its full bank receipt. Teacher-world rates cannot price bare screens. |
| **Strict gate → serving qualification** | **ESTIMATED:** **7–9 h** is the plan’s strict-gate allowance; serving qualification is unpriced. Low confidence, especially for an assisted controller. | **VERIFIED:** Mac-only, downstream of a frozen eligible candidate. The 7–9 h allowance is not a measured Claim B gate duration. [FINAL_PLAN:42,91](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/final_run_20261019/FINAL_PLAN.md:42), [handoff:10–12](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/final_run_20261019/HANDOFF_2026-10-20.md:10) | Price the actual candidate on disjoint full-horizon fixtures under the admitted Mac layout; use gate and serving receipts for realized duration. |

**ESTIMATED — Round 0 is approximately an 8–9-hour collection job, not automatically 10.5 hours.** The last chunk contains 40 pairs, so the early pace gives `3.5 + 3.5 + 1.4 = 8.4 h`, before residual overhead. The frozen model instead allocates approximately **4.41 h teacher execution + 1.53 h bare execution + 0.065 h metric processing + 0.75 h setup allowances = 6.76 h**. The observed pace is roughly 27% slower than its 100-pair forecast. That discrepancy deserves measurement; it should not be concealed by using the teacher-only A5 rate. [Registration:875–929](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/research/t34_round0/REGISTRATION_ROUND0.json:875), [handoff:156](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/final_run_20261019/HANDOFF_2026-10-20.md:156)

**VERIFIED — A5 throughput is real, but it is not complete production throughput.** The aggregate exports record **57.1734** and **54.3721 worlds/h**; both terminal receipts report `PASS` and verified stop. The collector includes unfinished-root work in its rate. Full supervisor elapsed times were **2,017 s** and **1,973 s**, versus approximately **1,259 s** of collector elapsed time. Startup, transfer, and cleanup therefore matter, particularly for small chunks. [Lane 0 benchmark:1354](/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/manifests-final-3ebf3991/a5-l4-s0-output/exports/benchmark.json:1354), [lane 1 benchmark:1360](/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/manifests-final-3ebf3991/a5-l4-s1-output/exports/benchmark.json:1360), [lane 0 receipt:3777](/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/manifests-final-3ebf3991/a5-l4-s0-output/receipt.json:3777), [lane 1 receipt:3547](/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/manifests-final-3ebf3991/a5-l4-s1-output/receipt.json:3547)

**ESTIMATED — At ~8.7 MB/world, an 800-world round exports roughly 7 GB.** Thus “29 units per round” is collection arithmetic, not a complete label-to-ready-cache price. Transfer, validation, chunk boundaries, and regeneration can add hours even while GPU utilization remains low. The present Round 0 export ceiling is **2 GiB**, so this cannot be treated as one unrestricted export. [Campaign log:57](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/final_run_20261019/CAMPAIGN_LOG.md:57), [make_manifest.py:32](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/research/t34_round0/make_manifest.py:32)

**VERIFIED — The dependency graph contains a bypass, not just one long training chain.** Phase 0 and Round 0 are already parallel. After Phase 0, Path B consumes quiet Mac time. The success contract explicitly sends a passing paced controller to the strict gate as **Claim B**, making distillation optional; otherwise Claim A needs a Round 0 GO and the sequential student rounds. [SUCCESS_CONTRACT.md:59–68](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/final_run_20261019/SUCCESS_CONTRACT.md:59)

**ESTIMATED — Using the saved 17:30 snapshot, the Path B development decision is roughly 39–42 hours away:** 16.5–19.5 remaining Phase 0 hours + nominal 1.5-hour pre-check + 20.7-hour efficacy estimate, **plus timing-gate and admission overhead**. This is low confidence and excludes strict/serving. Accelerating Round 0 by several hours probably does not advance that branch because it should finish before the Mac is available.

**VERIFIED — Label-start sequencing has inconsistent wording.** The success contract says Path B’s feasibility check precedes R1/R2; the amended design says label rounds wait for the paced test “or Round 0.” The adopted campaign table places Claim A after L1. I would not count aggressive overlap as established scheduling authority. [Contract:61](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/final_run_20261019/SUCCESS_CONTRACT.md:61), [design:103](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:103), [campaign log:70](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/final_run_20261019/CAMPAIGN_LOG.md:70)

**ESTIMATED — Once Claim A is admitted and implementation is ready, two rounds have a compute-only floor of roughly 26–29 hours:** two × [4.7–4.9 h collection on three lanes + 4 h fit critical path + nominal 1 h screen], followed by a 7–9 h strict gate. Cache construction, exports, fit qualification, operator barriers, serving qualification, and any R3 are additional. This is a lower-bound scheduling scenario, not a release ETA. Using only two collection lanes adds approximately **4.7–4.9 h across two rounds**.

**VERIFIED — The “95% Python” explanation does not apply uniformly to these stages.** Fable’s **~95%** figure appears under an explicitly estimated decomposition of the old W256 Colab execution. Its measured Mac decomposition was digest **64%**, evaluation **16%**, opponent hooks **13%**, simulator **5%**; the later digest fix changed that balance. The campaign records a **6.16 → 3.59 ms/row-step** improvement. Treating 95% as the current post-fix measured fraction would overstate the evidence. [Fable review:8–35](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/advice/2026-10-20_t34_a5_speed_fable.md:8), [campaign log:55](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/final_run_20261019/CAMPAIGN_LOG.md:55)

**VERIFIED — Phase 0 presents a stronger GPU opportunity than the current shadow teacher.** Its optimized CPU profile spends **0.677/1.042 s**, **0.664/1.402 s**, and **0.958/1.657 s** in B8 inference for scripted/frozen/mixed searches: approximately **65%, 47%, and 58%**. It already batches continuations, with roughly one B8 forward per rollout step. CUDA remains untested. [PROFILE.md:43–58](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/research/phase0_search_20261019/PROFILE.md:43), [limitations:94](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/research/phase0_search_20261019/PROFILE.md:94)

**ESTIMATED — Even infinitely fast inference would limit those fixtures to approximately 2.86×, 1.90×, and 2.37× overall speedups on the same host.** An A100 cannot eliminate simulation, observations, opponent carry, sequential rollout depth, or CPU/GPU synchronization. A slower cloud host can consume much of the inference gain. The recorded **116.9–229.8 L4 hours per million searches** are conditional models using Mac MPS service times, not CUDA measurements. [SPEED_REPORT.md:60,75–83](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/research/phase0_search_20261019/SPEED_REPORT.md:75)

**ESTIMATED — For the current approved shadow teacher, A100’s useful mechanisms rank as follows:**

- **Host throughput per slot is the first plausible benefit.** More effective cores or faster per-core execution could increase completed worlds/h without changing search semantics. Neither is established for an A100 assignment. Twelve advertised vCPUs are not twelve independent physical cores.
- **GPU batching needs an actual end-to-end demonstration.** The current qualified layout uses CPU inference. The earlier CUDA batch-1 evaluation took **4.35 ms versus 0.93 ms CPU**, and the adapter explicitly returns results to CPU each forward. A faster GPU does not remove that synchronization. [Fable review:5,13,31](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/advice/2026-10-20_t34_a5_speed_fable.md:5), [policy.py:93–100](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/research/t34_exit_b8/policy.py:93)
- **RAM helps only if it removes measured memory pressure.** The cache is on SSD/mmap, with current/next batches prefetched. The measured synthetic payload sizes imply about **12.7 GB/M rows for S1b** and **66.2 GB/M rows for S1a**; these are not VRAM requirements. There is no present evidence that an 80 GB GPU is needed. [Fit README:8–31,104–110](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/research/t34_exit_b8_fit/README.md:8)

**ESTIMATED — I would spend a small amount of prepaid compute to resolve A100 host uncertainty, but would not migrate production on the strength of its GPU name.** The cheapest useful future comparison is one bounded A100 assignment running the **same frozen CPU/W1 teacher fixtures**: a warmed one-process measurement, then a matched multiprocess measurement, recording effective CPU quota, CPU model, RSS, exclusive stage timings, completed worlds, and usable labels. Use excluded fixtures and the qualified supervisor. A **30-minute total allocation** costs approximately **2.7 units for 40 GB or 3.75 units for 80 GB**, at the question’s unverified planning rates; reserve-inclusive envelopes would be **3.24/4.5 units**. This is a proposed measurement, not an executed or ratified job.

**ESTIMATED — The switching criterion should be time saved per scarce slot, not unit break-even.** If a real A100 host delivers **2×** the useful teacher throughput, replacing one of three equal L4 lanes raises aggregate throughput from three lane-equivalents to four: about **25% less collection time**, or roughly **1.2 hours per 800-world round**. It does not halve the whole campaign. A slower allocation/startup process or an additional implementer day could erase that gain.

**ESTIMATED — Cached fits are the second legitimate A100 test.** First obtain one production L4 cache-build/fit receipt. If optimizer execution is demonstrably GPU-bound and takes several hours, compare the identical fixed-precision fit on A100. Halving a four-hour fit saves two hours; it does not accelerate cache replay, the next round’s dependent labels, or the Mac release gates. The repository itself recommends trying A100-40 only after a full cached L4 fit demonstrates substantial GPU-bound time. [DESIGN_V2_SHADOW.md:70–72](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:70)

**ESTIMATED — What would change my judgment:** a matched A100 receipt showing substantially higher **complete, validated output per occupied slot**, or a production fit receipt showing several hours of GPU-bound optimizer work. A Path B failure would also move the binding release path to Claim A’s serial label–fit–screen chain. Until then, the best systems decision is to keep the Mac’s paced decision and release work uninterrupted, complete the existing Round 0 sequence, and make the already-proven Colab process fan-out carry any admitted label rounds. That advances the earliest release opportunity without spending implementer days on an unmeasured GPU rewrite.



## Call 2 — cross-examination prompt (verbatim)

```text
Follow-up, same role and same RULES as before (read-only; no edits/commits/new files in the repo; at most tiny read-only Mac probes, 1 thread, `nice -n 10`, < 30 s each — the 6-worker Phase 0 study and the Colab supervisor must not be disturbed; no Colab/RunPod/network jobs; full answer in your final message with file:line evidence and numbers, every claim labelled VERIFIED or ESTIMATED). Your previous answer is saved at /Users/josenunez/Projects/ml/snake-dqn-artifacts/a100-advice/p1-call1/snake-dqn-advice-a100-1.last-message.md — re-read it first. Same goal (a better served hero via the Mac strict gate, soonest wall-clock; units not scarce; wall-clock, quiet-Mac hours, implementer days and the 3-runtime cap are).

I am now cross-examining your answer. Treat your own previous claims as a hostile reviewer would; where you were right, say why with evidence, and where you were loose, correct yourself.

1. Your headline judgment rests on Path B being "the first opportunity" and on a ~39–42 h Path B decision horizon with LOW confidence (20.7 h efficacy is a prefix extrapolation; timing gate and admission unmeasured). What is the realistic range for that horizon and what in the repo bounds it? If Path B's paced controller fails or runs long, how does the critical path change, and does the A100 answer change with it?

2. You asserted that replacing one of three L4 lanes with an A100 whose host is 2x faster saves only ~25% of collection. That assumes the A100 host is only ~2x and that lanes are interchangeable. The A100 runtime's host (cores, per-core speed, RAM, CPU quota) is unmeasured; what is the plausible range given what the repo and receipts show about Colab hosts (L4 12 vCPU, CPU-only 2 vCPU, any A100/H100 receipts under snake-dqn-artifacts or docs/research/advice/2026-10-17_colab_*.md, 2026-10-18_overnight_a100.md), and at what host multiple does the A100 become the better use of a scarce runtime slot for label collection? Show the arithmetic against the whole Claim A chain, not just one round.

3. The current qualified teacher layout is 12 single-thread CPU processes with batch-1 CPU inference (0.93 ms) and a per-row digest. Within that layout, what is the measured or best-evidenced per-row cost breakdown after the O(bytes) digest fix (3.59 ms/row-step), and what fraction is B8/opponent inference that a shared batched GPU inference server across the 12 processes could remove? Say whether that is a GPU question at all, or a host question, and what the Round 0 pace being ~27% slower than its frozen forecast tells you about where time goes.

4. You flagged ~7 GB of exports per 800-world round and a 2 GiB export ceiling as a possible hidden serial cost. Is that verified as a wall-clock term anywhere (transfer rates, supervisor export timings in receipts), or is it speculation? Quantify it or retract it.

5. For student fits: the repo says try A100-40 only after a cached L4 fit shows GPU-bound time. From the fit harness code and the fitcache spec, what is your best estimate of the GPU-bound fraction of a production fit (rows per round, epochs, batch size, model size ~1.64M params), and is a 4 h fit allowance even plausible on an L4?

Finally: name the single measurement that would most change your overall judgment on A100 vs L4 for time-to-a-better-agent, specify it precisely (fixture, layout, duration, metrics, cost in units), and state what result would flip you each way. Still no option lists — one judgment, with reasoning and the evidence that would overturn it.
```

## Call 2 — Astra answer (verbatim)

**ESTIMATED — My revised judgment: preserve Path B’s quiet-Mac execution, but treat an A100 host comparison as worthwhile preparation for Claim A. A measured 1.5–2× improvement in useful collection throughput could justify an A100 runtime slot when units are not scarce. My earlier “only 25%” wording undervalued that benefit. However, neither the current inference profile nor the student-fit arithmetic justifies an A100 GPU migration yet.**

**VERIFIED — Scope and terminology:** I reread the saved answer first and inspected the same clean, detached worktree at `6f1e751f4ec11c280c211634cd09de924b649218`. No repository files were changed, no workloads or network operations were launched, and neither running study was disturbed. Here, **VERIFIED** means inspected source or saved receipt, including arithmetic over its fields; **ESTIMATED** means a forecast, interpretation, or proposed decision rule.

**VERIFIED — 1. The Path B schedule needs two substantive corrections.**

First, **20.7 hours is not the 64-frame prefix extrapolation**. The registered Mac work is 144 terminal pairs plus 32 Watch6 pairs, both arms paced at 24 fps:

\[
176\times2\times5000/24/3600=20.370\text{ hours}.
\]

Four timing probes add approximately **0.361 hours of pacing**, giving **20.731 hours before setup and other overhead**, if every terminal hero reaches H5000. Terminal deaths shorten execution while preserving zero-scored dead frames. The uncertain prefix extrapolation produced the **17.722-hour unpaced estimate**, subsequently moved to Colab. I conflated these sources in my earlier answer. [Registration:431–435](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/research/t34_pathb/PREREGISTRATION_RETENTION_V2.md:431), [cost-smoke accounting:160–170](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/research/t34_pathb/VERIFICATION_RETENTION_FIXES2.md:160)

Second, **the old 1.5-hour pre-check estimate is incompatible with the implemented serial, paced workload**. The runner executes both bare and assisted arms on all 64 source trajectories at 24 fps. Full survival would consume:

\[
64\times2\times5000/24/3600=7.407\text{ hours}.
\]

Even the required **50 assisted survivors alone consume 2.894 hours**, before the bare arm, other trajectories, and setup. This is a particularly clear error in my previous schedule. [Pre-check loop:1394–1425](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/research/t34_pathb/retention_v2.py:1394)

**ESTIMATED — A more realistic planning range is approximately 40–50 hours from the same saved Oct 20 17:30 snapshot to a completed Path B development decision, assuming prompt admission and no operational interruption.** That comprises:

| ESTIMATED component | Planning range | Basis |
|---|---:|---|
| Remaining Phase 0 | 16.5–19.5 h | Saved ETA, not a newly verified completion forecast |
| Pre-check | 6–8 h | Serial paired pacing; historical T3 alive fractions suggest about 6.7 h of pacing |
| Retention efficacy | 17–21 h | Terminal alive-frame fraction approximately 0.8–1.0, plus complete Watch sessions |
| Timing and modest overhead | 0.5–1.5 h | Approximately 0.361 h pacing; remaining overhead unmeasured |

**VERIFIED — The repository supplies resource caps, not a finite calendar guarantee.** Timing has a **one-hour cap and counts inside the 40-hour study cap**; the pre-check has a separate **eight-hour cap**. Thus the admitted successful-attempt chain has **48 hours of phase allowance**, not 49. There can be up to three timing attempts, with only specified retryable outcomes permitting another attempt; admission requires 120 quiet seconds. Waiting between attempts or for admission is not bounded by an overall calendar deadline. [Caps and retry rules:27–73](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/research/t34_pathb/PREREGISTRATION_RETENTION_V2.md:27)

**ESTIMATED — Using that saved Phase 0 ETA, the nominal cap envelope reaches approximately 64.5–67.5 hours from the snapshot, potentially another two hours for preceding timing attempts, plus admission gaps.** That endpoint can be **INCONCLUSIVE**, not a successful candidate. Conversely, a timing or pre-check failure can decide the branch much earlier. The previous **39–42 hours** was too narrow and used an understated pre-check.

**VERIFIED — “First opportunity” is defensible as the prescribed sequence, not as a prediction of the first successful release.** The success contract puts Path B feasibility before R1/R2. Passing sends Claim B toward strict evaluation; losing efficacy under deadlines rejects Claim B, with Claim A proceeding only after Round 0 GO. The descriptive unpaced Colab arm cannot hold up or overturn the Mac verdict. [Success contract:59–68](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/final_run_20261019/SUCCESS_CONTRACT.md:59), [unpaced decision boundary:148–179](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/research/t34_pathb/PREREGISTRATION_RETENTION_V2.md:148)

**ESTIMATED — If Path B fails, collection, cache preparation, fits, and screens become the candidate-producing critical path. A100 host throughput then matters directly. If Path B merely runs long, accelerating those Colab stages cannot recover the preceding Mac admission delay.** The old **7–9-hour strict allowance remains unverified for an assisted controller**, and serving qualification remains unpriced; neither belongs inside a confident release ETA.

**VERIFIED — 2. There is no measured 2× A100 host advantage, and the historical evidence argues against assuming more cores automatically.**

| VERIFIED evidence | What it establishes |
|---|---|
| Current A5 L4 receipts | **12 visible CPUs**, **56,863,334,400 bytes RAM**, twelve single-thread CPU workers |
| Current CPU-only diagnostic | **2 visible CPUs**, affinity/cpuset `0–1`, `cpu.max="max 100000"`, **13,605,830,656 bytes RAM** |
| Historical A100-40 report | **12 vCPU / six physical cores** |
| Historical A100-80 report | **12 vCPU** |
| Oct 18 overnight advice | Approximately 12 vCPU and High-RAM were explicitly **owner-supplied**, not live inspection |

Sources: [L4 hardware:1314–1331](/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/manifests-final-3ebf3991/a5-l4-s0-output/exports/benchmark.json:1314), [CPU-only hardware:261–270](/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/manifests-final-3ebf3991/cpu-profile-output/exports/profile-cpu.json:261), [A100-40 report:3–4](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/t16_colab_smoke02_results_2026-10-17.md:3), [A100-80 report:210](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/t16_colab_smoke02_results_2026-10-17.md:210), [overnight advice:8–11](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/advice/2026-10-18_overnight_a100.md:8).

**VERIFIED — I found no matched T34 A100/H100 receipt establishing CPU model, effective quota, per-core throughput, or memory bandwidth.** The older advice’s “same vCPU class” assertion is not a hardware guarantee. Its statement that L4 has fewer vCPUs is contradicted by the later measured 12-vCPU L4 configuration. [Older advice:74–86](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/advice/2026-10-17_colab_review.md:74), [later L4 report:497](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/t16_colab_smoke02_results_2026-10-17.md:497)

**ESTIMATED — I would center a planning assumption near 1× and stress-test 0.5–2× useful throughput per runtime. That is an uncertainty scenario, not a confidence interval or an upper hardware bound.** A result above 2× would be valuable evidence; the available record neither establishes nor rules it out. More GPU VRAM alone does not demonstrate more useful host capacity.

**ESTIMATED — The correct whole-chain arithmetic is as follows.** Let an L4 collect \(r=54.4–57.2\) worlds/hour and let the A100 deliver \(mr\). With work apportioned to throughput, two sequential 800-world rounds require:

\[
C_{\mathrm{L4}}=\frac{1600}{3r}=9.32–9.80\text{ h},
\qquad
C_{\mathrm{mixed}}=\frac{1600}{(2+m)r}.
\]

Let \(U\) contain the remaining critical-path work: production integration, cache creation, R1’s slower fit, R2’s fit, screens, non-overlapped transfers, Mac admission, strict gate, and serving qualification. Then the complete chain is \(C+U\), with no claim that \(U\) is measured.

| ESTIMATED A100 multiple \(m\) | Collection across R1 + R2 | Whole-chain time saved, holding \(U\) fixed |
|---:|---:|---:|
| 0.5× | 11.19–11.76 h | **−1.86 to −1.96 h** |
| 1× | 9.32–9.80 h | 0 |
| 1.5× | 7.99–8.40 h | **1.33–1.40 h** |
| 2× | 6.99–7.35 h | **2.33–2.45 h** |
| 3× | 5.59–5.88 h | **3.73–3.92 h** |
| 4× | 4.66–4.90 h | **4.66–4.90 h** |

**ESTIMATED — Any \(m>1\) improves the slot’s contribution if switching has no additional critical-path cost.** With incremental switching delay \(H\), the break-even condition is:

\[
C_{\mathrm{L4}}\frac{m-1}{m+2}>H.
\]

Using \(C_{\mathrm{L4}}\approx9.6\) hours, **one additional hour requires \(m>1.35\)**; **two hours require \(m>1.79\)**. Fixed downstream work dilutes the percentage improvement; it does not make the improvement disappear. Using my old 17–19-hour downstream allowance solely as a sensitivity, a 2× host saves about **8–9% of that whole chain**.

**ESTIMATED — The allocation assumption matters as much as the formula.** If all three lanes retain equal fixed shards, accelerating one leaves the two L4 lanes as stragglers and may save essentially no collection wall time. At \(m=2\), an approximately balanced assignment is **400 worlds on A100 and 200 on each L4**, with mix/workload balance and bounded chunks preserved. R2 must be repriced after the student changes. The previous 25% arithmetic was correct only under this throughput-weighted scheduling assumption.

**ESTIMATED — I also withdraw “26–29 hours” as a compute floor.** It used four-hour fit allowances as though they were irreducible execution times. They are not. It remains an illustrative old-budget scenario, with important implementation and release work excluded.

**VERIFIED — 3. The 0.93 ms figure is not pure inference, and the post-fix evidence is more informative than my earlier answer suggested.**

The original Mac table assigns **0.93 ms to `teacher.evaluate`, including raster construction**, with only about **0.46 ms identified as batch-one CPU forward**. The matched post-fix **3.594 ms/row-step** measurement is a loaded-Mac, three-world, H120 fixture—not the twelve-process L4 production layout. [Original decomposition:8–17](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/advice/2026-10-20_t34_a5_speed_fable.md:8), [post-fix fixture and timings:193–227](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/research/t34_exit_b8/VERIFICATION_ENGINE_FANOUT.json:193)

**VERIFIED — The best saved exclusive decomposition after the fix comes from the separate CPU-only Colab diagnostic:** 1,689 row-steps in 60.080 seconds, covering short/long bodies and all three mixes.

| VERIFIED diagnostic component | ms/row-step | Fraction of instrumented wall |
|---|---:|---:|
| Digest | 12.60 | 35.4% |
| B8/opponent module inference | 4.75 | 13.3% |
| Raster construction | 2.49 | 7.0% |
| Simulator | 4.36 | 12.3% |
| v8 hooks | 7.07 | 19.9% |
| Remaining work/instrumentation | 4.31 | 12.1% |
| **Total** | **35.57** | **100%** |

**VERIFIED — These are exclusive totals; the inference field’s larger inclusive total must not be added again.** The instrumentation wraps all Torch module calls, including nested calls, and subtracts named children. This is an instrumented CPU-only diagnostic, not a decomposition of the 3.594 ms Mac result or the production L4 lane. [Diagnostic:49–83](/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/manifests-final-3ebf3991/cpu-profile-output/exports/profile-cpu.json:49), [instrumentation:117–188](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/research/t34_exit_b8/profile_cpu.py:117)

**VERIFIED — Production A5 provides a complementary measurement:** summing worker receipts, B8 forward/transfer time is **1,133/15,021 worker-wall seconds = 7.54%** on lane 0 and **1,172/15,008 = 7.81%** on lane 1. These timers do not fully instrument native CPU opponent forwards. On the median post-fix Mac fixture, B8 forward/transfer is **0.716/5.812 seconds = 12.3%**. [A5 worker timer fields:81–84](/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/manifests-final-3ebf3991/a5-l4-s0-output/exports/benchmark.json:81), [Mac fixture:60–97](/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/after-v3-serial-r7.json:60)

**ESTIMATED — The best-evidenced removable inference fraction is therefore around the low teens, with the exact K12 combined B8/opponent fraction still unmeasured.** Eliminating a 13.3% fraction completely gives only:

\[
1/(1-0.133)=1.15\times.
\]

A shared GPU server cannot eliminate raster construction, digesting, simulation, v8 decision logic, or all IPC/transfer costs. It could be useful engineering, but that is first a **batching and synchronization question**, not evidence that A100 beats L4. Batch-N forwards also require qualification because the adapter explicitly warns that their float32 results need not match batch-one bits. [Policy batching boundary:126–133](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/research/t34_exit_b8/policy.py:126)

**VERIFIED — Round 0’s approximately 27% longer early elapsed forecast does not identify the missing component.** Its frozen 100-pair budget is **110.35 minutes teacher + 38.30 bare + 1.62 metrics + 15 setup = 165.27 minutes**; the saved early pace was approximately 210 minutes. That is about **45 additional minutes**, or approximately **21% lower throughput**, not a profile. [Frozen sizing:875–891](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/research/t34_round0/REGISTRATION_ROUND0.json:875), [saved progress:155–157](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/final_run_20261019/HANDOFF_2026-10-20.md:155)

**ESTIMATED — It tells us that teacher-only A5 throughput is insufficient to forecast paired production completion.** Mix-dependent search, incomplete-worker tails, bare execution, startup, contention, and exports remain competing explanations. It does not establish that the GPU is the missing 27%.

**VERIFIED — 4. Transfer cost exists in receipts; my attribution of it to a hidden serial “hours” term was too loose.**

Summing the supervisor’s recorded payload transfers:

| VERIFIED A5 lane | Downloaded payload | Sum of download durations | Effective payload rate |
|---|---:|---:|---:|
| L4 lane 0 | 177.70 MB | 312.75 s | 0.568 MB/s |
| L4 lane 1 | 161.61 MB | 150.05 s | 1.077 MB/s |

**VERIFIED — Typical world transfers took approximately 8.8 seconds; lane 0 includes one 133.6-second outlier.** These are actual transfer records, not a throughput inferred from the overall job duration. [Lane 0 transfer records:4010 onward](/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/manifests-final-3ebf3991/a5-l4-s0-output/receipt.json:4010), [lane 1 examples:3961 onward](/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/manifests-final-3ebf3991/a5-l4-s1-output/receipt.json:3961)

**ESTIMATED — Approximately 7 GB per 800 worlds remains a reasonable byte-volume projection.** At those measured payload rates it represents approximately **1.8–3.5 aggregate transfer-hours**, or approximately **0.6–1.2 hours per lane** if divided evenly over three independent transfer streams. Those are transfer-work estimates, not necessarily additional elapsed time.

**VERIFIED — The supervisor downloads immutable completed worlds during polling, before the runtime becomes terminal. Also, the 2 GiB limit is per exported file, not a total-round or total-runtime ceiling.** Individual A5 world files are approximately 8.5–8.9 MB. My earlier use of “7 GB versus 2 GiB” implied a stronger serialization constraint than the code establishes. [Streaming exports:1438–1455](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/research/colab_cli/supervisor.py:1438), [per-export limit:109–119](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/colab-cli.md:109)

**ESTIMATED — I retract the claim that this demonstrates hours added serially to the critical path.** At roughly 55 worlds/hour, a lane produces only about **0.13 MB/s**, below the measured transfer service rate. Substantial overlap is plausible. The non-overlapped tail, sidecar/publication overhead, and subsequent fit-input movement still need measurement.

**VERIFIED — 5. Four hours is an old all-in fit allowance, not evidence of four GPU-hours.**

The current fitter defaults to **five epochs, batch 1,024**, and selects all searched rows plus an equal number of anchors. It builds an observation cache once; each S1b optimizer step performs one student forward, backward, and Adam update, with previous Q values supplied in the data. Every epoch also performs held-out scoring and a **32-world H5000 gameplay probe**. [Fit recipe:119–144](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/research/t34_exit_b8_fit/train.py:119), [optimizer:198–234](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/research/t34_exit_b8_fit/train.py:198), [cache and sampling:8–31](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/research/t34_exit_b8_fit/README.md:8), [probe contract:167–175](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/research/t34_exit_b8_fit/README.md:167)

**VERIFIED — Production selected-row count is not established.** The A5 **16–20k row-steps/world are simulator work**, not training examples. The v2 teacher emits labels at actual root decisions; shadow confirmations are anchors. The fit README still identifies a missing production v2 source/probe integration boundary. Thus even the fit’s production input inventory is an implementation dependency. [Fit-side boundary:35–50](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/research/t34_exit_b8_fit/README.md:35)

**ESTIMATED — A near-one-million-selected-row scenario is useful for sizing, but must not be presented as the measured v2 round.** The old 600-searched-decisions/world assumption would yield approximately **960,000 selected rows** for 800 worlds. The current H5000 round has a loose ceiling of **four million root observations**, with selected rows generally fewer.

**ESTIMATED — For S1b, using the historical approximately 22.5 MFLOP/forward-row analysis and backward ≈ twice forward:**

\[
1{,}000{,}000\times5\times67.5\text{ MFLOP}
=337.5\text{ TFLOP}.
\]

At an explicitly hypothetical **0.5–5 TFLOP/s effective L4 rate**, that is **1.1–11.3 minutes of neural arithmetic**, before loss/optimizer/launch overhead—not four hours. The FLOP estimate is documented; that effective-rate range is a sensitivity assumption, not a measured fitter benchmark. [Historical model arithmetic:86–97](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/advice/2026-10-17_colab_slower.md:86)

| ESTIMATED all-in duration, 1M selected rows | Arithmetic-time fraction under that sensitivity |
|---|---:|
| 30 min | 4–38% |
| 1 h | 2–19% |
| 4 h | 0.5–4.7% |

**ESTIMATED — My working expectation is minutes to tens of minutes of S1b optimizer execution, with a low-tens-percent-or-smaller GPU-sensitive share of a host/probe-heavy production fit.** The exact fraction is not identifiable yet. Four million selected rows multiply the arithmetic estimate by four; aggregated rounds and S1a’s additional tower/auxiliary path can increase it further.

**ESTIMATED — Four pure optimizer-hours at one million selected rows would mean approximately 2.95 seconds per 1,024-row update, or only 347 trained row-visits/second.** That would warrant investigating feeding, synchronization, and implementation before concluding L4 arithmetic capacity is inadequate.

**VERIFIED — Several host costs survive caching.** Every epoch reconstructs Python row dictionaries, parses metadata, normalizes and validates rows, and stacks selected batches. Five full 32-world probes can involve **800,000 candidate gameplay frames**. The old plan budgeted regeneration at **30 minutes × five epochs**, already accounting for **2.5 of its four hours**; caching removes repeated regeneration, not every host cost. The current receipt explicitly reports `gpu_busy_percent=None`. [Cache reader:370–385](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/research/t34_exit_b8_fit/cache.py:370), [loader:188–201](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/research/t34_exit_b8_fit/data.py:188), [old allowance:49–61](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/final_run_20261019/FINAL_PLAN.md:49), [timing limitations:305–316](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/research/t34_exit_b8_fit/train.py:305)

**ESTIMATED — Four hours remains plausible as a conservative all-in allowance; it is not a defensible optimizer forecast or floor.** The repository’s instruction—time a full cached L4 fit, then consider A100-40 only if GPU-bound time exceeds approximately one hour—is therefore still sound. [Hardware decision rule:70–72](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:70)

**ESTIMATED — The single measurement most likely to change my hardware judgment is a matched, complete-world A100-40 versus L4 teacher run, including validated export completion.**

**ESTIMATED — Proposed fixture and layout:** the same frozen fast-digest teacher source and B8 checkpoint, **48 excluded H5000 worlds—16 per mix**, identical world order and process shards, **12 processes × one numerical thread × W1, CPU inference on both hosts**. Finish the fixed inventory rather than selecting only worlds that finish inside a short window. Allow **90 minutes per whole session**, including setup, export, and verified stop; cap expiry means an incomplete comparison, with no automatic extension.

**ESTIMATED — Required measurements:** CPU model, visible CPUs, affinity/quota/throttling, RAM and peak RSS; completed worlds and retained usable labels by mix; row-steps; exclusive stage timings; worker imbalance; collector elapsed; transfer/publication time; and **allocation-to-validated-output elapsed**. Compare both useful collection rate and the projected complete 800-world chunk schedule. This directly tests the unmeasured host advantage without first spending implementer days on a GPU server.

**ESTIMATED — Cost envelope:** at planning rates **2.0 u/h L4 + 5.4 u/h A100-40**, two 90-minute sessions cost at most **11.1 units before reserve**, or **13.32 units with 20% reserve**. Those are historical planning rates, not a new live quote. The comparison would occupy two of the three permitted runtime slots and is a proposal only; nothing was launched. [Recorded planning rates:54–55](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-1/docs/claude-skills/colab/SKILL.md:54)

**ESTIMATED — My prospective switching rule is to choose A100 for admitted Claim A collection if that comparison demonstrates at least 1.5× useful throughput, preserves output validity and label coverage, and leaves no more than one hour of additional critical-path integration/allocation delay.** That buys approximately **1.3–1.4 hours across R1/R2** at 1.5×, or **2.3–2.5 hours at 2×**, provided shards are weighted appropriately. A result near 1×, or a transfer/straggler tail that consumes the gain, keeps L4 as the better slot. That measurement would change the collection decision; it would not accelerate Mac pacing or establish a reason to move student fits to A100.



## Prompter notes (Fable)

- Spot-checked call 1's load-bearing citations on main: SPEED_REPORT.md:47–66 (128.98 six-worker h for 1M decisions is already post-speedup — the facts brief's "129 Mac-hours ... before a 3.4–4.2x speedup" wording is misleading); PROFILE.md:43–58 (B8 forward 0.68/0.66/0.96 s of 1.04/1.40/1.66 s); CAMPAIGN_LOG:55 (digest 6.16→3.59 ms); REGISTRATION_ROUND0 chunks 100/100/40; make_manifest.py:32 (2 GiB is MAX_EXPORT_BYTES per export); DESIGN_V2_SHADOW.md:70–72 (A100-40 only if a cached L4 fit is GPU-bound > ~1 h); SUCCESS_CONTRACT §5 (Path B before R1/R2). All match.
- Call 2 corrected three call-1 claims on its own: Path B pre-check is ~6–8 h serial paced (not 1.5 h) → Path B decision ~40–50 h from the 17:30 snapshot, cap envelope ~65–68 h; "7 GB vs 2 GiB serial export cost" retracted (2 GiB is per file, transfers overlap polling, ~0.13 MB/s produced vs 0.57–1.08 MB/s measured transfer); "26–29 h compute floor" withdrawn (4 h fits are allowances, not measured).
- Not independently re-verified by me: the CPU-only diagnostic breakdown (profile-cpu.json:49–83), A5 worker B8-forward shares (7.5–7.8%), A5 receipt transfer records, retention_v2.py:1394 pre-check loop, train.py fit defaults. Paths cited are concrete; orchestrator can grep them.
- Astra's net answer: A100 GPU is not a lever for any current stage (inference ≈ 13% exclusive on the teacher; fit optimizer arithmetic is minutes); the only live A100 mechanism is an unmeasured host advantage (historical A100-40 receipt showed 12 vCPU / 6 physical, i.e. same as L4). Switching rule: ≥ 1.5x useful complete-world throughput with ≤ 1 h integration delay; gain is only ~1.3–2.5 h across R1+R2.
- Open points: Round 0 chunk 000 running ~21% below its frozen throughput forecast — unexplained, needs the chunk receipt; production fit GPU-bound share unknown until one cached L4 fit receipt exists; Astra's proposed matched A100-40 vs L4 probe (48 excluded worlds, 12 procs CPU inference, 2x90 min, ≤ 13.3 u) is a proposal only, not dispatched.
