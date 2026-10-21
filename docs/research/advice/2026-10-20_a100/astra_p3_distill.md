# A100 question — Astra lens P3: learning / distillation

2026-10-20 · Astra (`CODEX_MODEL=gpt-6-astra`, fast mode, `-e high`) via `codex-handoff.sh`, worktree
`../snake-dqn-advice-a100-3` at main `6f1e751f`, read-only. Prompter: Fable (fable-prompter-3). Two calls: the open
question through the distillation lens, then a cross-examination of call 1. Raw outputs:
`snake-dqn-artifacts/a100-advice/p3-call1/` and `p3-call2/` (`prompt.md`, `run.log`, `*.last-message.md`).

## Prompter notes (Fable)

- Call 1 (~17:30–17:50 PT) said: keep L4; fit throughput unmeasured, probably host-dominated; the KL(prev‖student) loss
  can refuse rescues; spend ~100 u on CUDA fit qualification + 1 label round + initial L3 fits + 1 finalist screen.
- Call 2 made Astra retract the host-dominance claim: a 0.65–0.93 s single-thread probe of the real loader measured
  ~14.4 µs/row vs ~32.6 µs/row implied optimizer time at 30 upd/s → optimizer ≈ 69% of loader+train on L4; A100 at
  ~80 upd/s would give ~1.3–1.8× on that loop. It also found M2b receipts (23–29k rows/s per epoch) and T4c bare-screen
  receipts (160 H5000 worlds in 6.7 min) showing the portfolio's 3 u/fit and 30 u/screen allowances are oversized.
- The loss objection strengthened: at realistic prior gaps 0.06/0.10/0.45 the stationary student gap stays +0.047/
  +0.080/+0.147 (wrong ordering) — derived from loss.py as written, incl. the missing τ² factor. Not shown to be T3's cause.
- Net: A100 is plausibly a ~1.3–1.8× lever on the fit only; the distillation loop is label-bound (R2 ≈ 5–6 h on 3 L4
  lanes) once projection works, and the first blocker is the loss/projection question, not hardware.
- Astra's single settling measurement: a matched L4 vs A100-40 timing packet on one ~65k-row production cache
  (~2 u, 30–40 min elapsed) decomposing cache reopen / host load / transfer / fwd-bwd-Adam / held-out / probe.
- Caveat: both calls lean on historical numbers (M3-C upd/s, M2b rows/s likely on Mac MPS, T4c Mac screens); none is an
  L4 or A100 production-fit receipt. Astra's "12 vCPU on A100" comes from the T16 record, not the current assignment.

## Call 1 — prompt (verbatim)

```text
You are an advisor on the snake-dqn project (this worktree is a read-only checkout of main at 6f1e751f). Lens for this question: LEARNING / DISTILLATION — turning search-teacher labels into a better bare B8 student.

RULES (owner rules, strict):
- Read-only. Do not edit, commit, or create files in the repo. Do not write docs.
- At most tiny read-only probes on the Mac: 1 thread, `nice -n 10`, each < 30 s wall. The Mac is running a 6-worker study (Phase 0) that must not be disturbed. No training, no fits, no benchmarks longer than 30 s.
- No Colab, no RunPod, no network jobs.
- Put the FULL answer in your final message. Give file:line evidence and numbers. For every quantitative claim, state whether it is VERIFIED (you read it in the repo / measured it) or ESTIMATED (your reasoning), and show the arithmetic for estimates.

READ FIRST: docs/research/advice/2026-10-20_a100/FACTS_AND_QUESTION.md (goal, open question, facts, sources). Then as needed: docs/research/final_run_20261019/FINAL_PLAN.md (sections 3 "Student fit", 4, 9), docs/research/final_run_20261019/HANDOFF_2026-10-20.md, CAMPAIGN_LOG.md (same folder), docs/research/advice/2026-10-20_compute_portfolio_FABLE_JUDGMENT.md (lanes L2 and L3 are the distillation lanes), docs/research/advice/2026-10-20_compute_portfolio_astra_distill.md (a prior distillation-lens proposal; its loss counterexample is relevant), research/t34_exit_b8_fit/ (the fit package: loss.py, train.py, cache.py, README.md), research/redesign_m2_20261008/train_student.py (the earlier student fitter), src/model/ego2s_network.py (B8 = ego2s-bv8 dueling CNN, ~1.64M params), docs/research/specs/2026-10-19_t34_v2_fitcache.md.

GOAL: a better served snake agent than the current Watch hero (bare B8 seed 3), proven by the strict gate on the Mac, as soon as possible in wall-clock time. Colab units are prepaid (~1,157 u, owner wants them spent by ~2026-11-20 on decision-producing work).

FACTS (verify in the repo):
- The approved path (T34) is: exact-sim rollout search teacher labels the student's own trajectories (gated to long/low-margin states and death windows), then DAgger-distill into the same B8 CNN for 3-5 rounds (R1 beta=1, R2 beta=0.5, R3 beta=0), serve the bare student via the strict gate.
- Student fit recipe (FINAL_PLAN section 3): rows = all searched decisions + 1:1 unsearched on-policy rows; aggregated DAgger bank (1.0 current / 0.5 older); death-window rows weighted per world; loss = KL(softmax(Q_prev/tau) || softmax(Q/tau)) at tau=0.05 + forbidden-action hinge (margin 0.1) + small advantage-MSE; Adam 2.5e-5, batch 1,024, 4-6 epochs; epoch chosen by held-out decisive agreement + a 32-world survival probe; no TD term. One L4 per fit. The v2 fit cache regenerates observations once into a memory-mapped uint8 cache, then epochs stream from it; the CUDA path is implemented but NOT qualified.
- Label volume: a round is ~800 worlds, ~600 searched rows/world (~0.4M active sim row-steps/world), i.e. ~0.5M searched rows/round plus 1:1 anchors; the facts brief cites "~1M searched decisions/round" and "1M searched decisions ~ 129 Mac-hours" from the Phase 0 teacher (4.5 s/decision/core before a 3.4-4.2x speedup, B8 inference dominant). Each teacher world exports ~16-20k rows, ~8.7 MB.
- Measured where time goes in TEACHER work: ~95% single-threaded CPython on the host; L4 GPU 2-8% utilised; CUDA batch-1 forward slower than CPU (4.35 ms vs 0.93 ms). After the fan-out fix: 57.2 / 54.4 completed worlds/h per L4 lane (~0.036 u/world). Host of an A100 runtime: vCPU count and per-core speed UNMEASURED (L4 runtime = 12 vCPU ~ 6 physical).
- FIT work on GPU has never been timed on Colab; the portfolio judgment budgets ~3 u per cached fit on L4 and says "fits and screens are the only GPU-bound Colab work". Screens: bare-student 96/32/32 paired worlds (~1.6M rows, ~30 u each on L4).
- History: T3 students learned the bank (99.6% agreement) and retained labelled behaviour (89-92%) yet lost via behavioural spillover (boost-mode disagreement at all lengths, late-growth deficit). Every bare-network lever since B8 (TD fine-tunes, league T23, S1, M-series) was null or negative. Remaining deaths = self-enclosure at long length, decided 50-80 steps before death.
- Colab rates (third-party): L4 ~1.7 u/h, A100-40GB ~5.4 u/h, A100-80GB ~7.5 u/h. Max 3 concurrent runtimes, <= 24 h each. Strict gates and serving qualification are Mac-only. Freeze designs before data.

OPEN QUESTION (answer through the learning/distillation lens): In the loop "collect labels -> fit student -> screen -> next DAgger round", where would an A100 runtime (instead of or alongside L4) actually shorten the wall-clock time to a better bare B8 agent, and where is the real bottleneck instead? Be concrete about the fit step: with ~1.64M parameters, batch 1,024, ~1-2M rows x 4-6 epochs, what does a fit cost on L4 vs A100 and what dominates it (cache regeneration on the host, dataloading, the optimizer step)? What about the loss design and label volume: is the time to a better agent limited by how many searched decisions we can produce, by how well the fitter projects them into B8 without spillover, by the screening/evaluation that decides each round, or by something else — and what evidence in the repo supports that? If more/faster fits would help (sweeps over tau/margin/weights, more seeds, the L3 2x2 projection-fix arms), say what the fit throughput actually is and whether it is GPU-bound at all. Say what you would do with the next ~100 units to reduce wall-clock time to a strict-gate candidate, and what you would NOT spend on. Do not assume anything about A100 host speed that you cannot verify; say how you would measure it cheaply if it matters.
```

## Call 1 — Astra answer (verbatim)

**I would keep L4 as the default and spend the next increment on making the distillation experiment decisive: production replay integration, CUDA fit qualification, and the bounded projection-fix comparison. An A100 could shorten a genuinely GPU-bound cached fit, but this checkout contains no measured L4 or A100 production-fit throughput. The strongest unresolved learning risk is whether the fitter can absorb useful rescues without damaging ordinary play. Faster optimization alone does not resolve that.**

I inspected the clean, detached checkout at `6f1e751f`. No files were created or changed, and no fits, training, benchmarks, or network jobs were run. Below, **VERIFIED** means inspected code or a result recorded in this repository; **ESTIMATED** means arithmetic, a conditional forecast, or a proposed budget. Recorded hardware rates are planning inputs, not current account measurements.

**The label-volume premise needs updating before comparing hardware.**

The question combines the older dense teacher with the newer shadow teacher:

| Quantity | Evidence and interpretation |
|---|---|
| Old round size | **VERIFIED as a planning estimate:** approximately 800 worlds and 600 searched rows/world. **ESTIMATED arithmetic:** `800 × 600 = 480,000` searched rows; equal anchors produce `960,000` total training rows. That supports approximately one million **total rows**, not one million searched decisions. |
| Current teacher semantics | **VERIFIED:** the newer design supersedes the old trigger/death-window mechanics. Genuine horizon rescues are decisive; all-die delay choices receive approximately zero weight; shadow confirmations remain anchors. Dense confirmation rows are not independent rescue examples. |
| Current label-production speed | **VERIFIED recorded A5 result:** `57.2 / 54.4` completed worlds/hour per L4 lane, approximately `0.036 u/world`, after the process fan-out and digest fixes. |
| Current round projection | **ESTIMATED, assuming that A5 throughput transfers:** `800 / (57.2–54.4) = 14.0–14.7` lane-hours. Ideal three-lane execution is `4.66–4.90` hours, before startup, export and imbalance. At the conservative portfolio rate of `2 u/hour`, that is `28.0–29.4 u`; the recorded rounded estimate gives `800 × 0.036 = 28.8 u`. |

Sources: [FINAL_PLAN.md:24](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/final_run_20261019/FINAL_PLAN.md:24), [DESIGN_V2_SHADOW.md:82](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:82), [CAMPAIGN_LOG.md:57](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/final_run_20261019/CAMPAIGN_LOG.md:57).

Consequently, I would not use the brief’s **VERIFIED recorded estimate** of “129 Mac-hours per million searched decisions” to forecast the current label round. It concerns a different teacher workload. Nor would I equate exported simulation rows with selected fit rows or useful corrective labels. The production adapter must report those inventories separately.

There is also a qualification to “student’s own trajectories”: **VERIFIED**, the plan explicitly acknowledges that the first round is off-policy because the teacher acts; the subsequent mixture is where student visitation is tested. Its schedule is `β = 1, 0.5, 0`, and the first-round screen is deliberately non-terminal absent tripwires. [FINAL_PLAN.md:22](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/final_run_20261019/FINAL_PLAN.md:22), [FINAL_PLAN.md:89](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/final_run_20261019/FINAL_PLAN.md:89).

**The fit’s immediate bottleneck is readiness; its eventual runtime bottleneck is still unmeasured.**

**VERIFIED:** this snapshot has the cache and fitter, but the documented production boundary still lacks the replay-source/probe integration. CUDA prefetch is implemented but unqualified. That engineering work lies directly on the path to a candidate, regardless of accelerator. [README.md:28](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/research/t34_exit_b8_fit/README.md:28), [README.md:46](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/research/t34_exit_b8_fit/README.md:46).

The implemented recipe matches the stated supervised approach: **VERIFIED**, batch `1,024`, `4–6` epochs, pretrained Adam learning rate `2.5e-5`, KL temperature `0.05`, forbidden-action margin `0.1`, advantage-MSE coefficient `0.01`, and no TD term. [train.py:44](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/research/t34_exit_b8_fit/train.py:44), [train.py:119](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/research/t34_exit_b8_fit/train.py:119), [loss.py:12](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/research/t34_exit_b8_fit/loss.py:12).

But “cached fit” still includes substantial work outside the GPU:

- **VERIFIED:** cache construction regenerates and stores every source row, including held-out rows, before the training sampler chooses searched rows and matching anchors. The cache therefore need not contain only the eventual training subset.
- **VERIFIED:** every epoch reconstructs Python row dictionaries, parses metadata, normalizes and validates rows, selects samples, and copies selected arrays with `np.stack`.
- **VERIFIED:** CUDA prefetch stages a following batch on a copy stream, but there is no independent loader process or producer thread. Each update also converts its loss to a Python float, requiring a CUDA result to become CPU-visible.
- **VERIFIED:** reopening a cache hashes its payloads. That verification is not fully represented by the stored regeneration time or epoch times.
- **VERIFIED:** every epoch scores held-out decisive labels and invokes the gameplay probe.

Sources: [cache.py:240](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/research/t34_exit_b8_fit/cache.py:240), [cache.py:345](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/research/t34_exit_b8_fit/cache.py:345), [cache.py:370](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/research/t34_exit_b8_fit/cache.py:370), [data.py:178](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/research/t34_exit_b8_fit/data.py:178), [data.py:242](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/research/t34_exit_b8_fit/data.py:242), [train.py:230](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/research/t34_exit_b8_fit/train.py:230).

Thus the relevant measurement is:

\[
T_{\rm candidate} =
T_{\rm setup/cache/validation}
+\sum_{\rm epochs}(T_{\rm loading+training}+T_{\rm heldout}+T_{\rm gameplay\ probe})
+T_{\rm export}.
\]

The cache removes repeated simulation regeneration. It does **not** establish that the remaining workload is GPU-bound.

**A concrete L4-versus-A100 calculation is possible only as a sensitivity analysis—not an actual fit quote.**

For the requested workload, **ESTIMATED update counts** are:

- `ceil(1,000,000 / 1,024) × 4 = 3,908 updates`.
- `ceil(2,000,000 / 1,024) × 6 = 11,724 updates`.

There is relevant historical hardware evidence: **VERIFIED**, the older M3-C workload recorded approximately `29.3–29.8 updates/s` on L4 and `78–82.7 updates/s` for an isolated A100 learner. These are **not this fitter’s throughput**: the older workload used different replay gathering, additional next-state forwards, and different GPU/host scheduling. [t16_colab_smoke02_results_2026-10-17.md:187](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/t16_colab_smoke02_results_2026-10-17.md:187), [same file:212](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/t16_colab_smoke02_results_2026-10-17.md:212), [same file:497](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/t16_colab_smoke02_results_2026-10-17.md:497).

Using rounded historical rates purely as hypothetical optimizer throughput gives:

| Device scenario — all results **ESTIMATED** | Optimizer-only time | Optimizer-only units |
|---|---:|---:|
| L4 at an assumed `30 updates/s` | `3,908–11,724 ÷ 30 = 2.17–6.51 min` | At `1.7 u/h`: `minutes × 1.7 / 60 = 0.062–0.185 u` |
| A100-40 at an assumed `80 updates/s` | `3,908–11,724 ÷ 80 = 0.81–2.44 min` | At `5.4 u/h`: `0.073–0.220 u` |
| A100-80 at the **same assumed throughput** | `0.81–2.44 min` | At `7.5 u/h`: `0.102–0.305 u` |

The rates are **VERIFIED as third-party planning inputs recorded in the brief**, not independently measured current charges. There is no evidence here that the larger A100 would improve this fit’s speed. [FACTS_AND_QUESTION.md:30](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/advice/2026-10-20_a100/FACTS_AND_QUESTION.md:30).

This calculation says that the gradient-update workload is plausibly modest. It does **not** establish that an entire fit takes minutes. The documented approximately `3 u/fit` is **VERIFIED as an allowance**, with duration explicitly unmeasured. **ESTIMATED**, at the portfolio’s `2 u/h`, that allowance represents `3 / 2 = 1.5` allocated hours—not a measured runtime. [FABLE_JUDGMENT.md:11](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/advice/2026-10-20_compute_portfolio_FABLE_JUDGMENT.md:11), [same file:36](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/advice/2026-10-20_compute_portfolio_FABLE_JUDGMENT.md:36).

Two other costs could easily outweigh those optimizer minutes:

- **Cache regeneration and host processing.** **ESTIMATED from verified tensor shapes**, bare observation/target arrays consume  
  `7×31×31 + 4×37×37 + 60×4 + 6×4 + 6 = 12,473 bytes/source row`, before metadata. That is approximately `12.47–24.95 GB` for `1–2M` source rows. The actual source inventory may be larger than the selected training inventory. A GPU upgrade does not inherently accelerate replay, Python processing, SSD reads, or hashing. [README.md:134](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/research/t34_exit_b8_fit/README.md:134).
- **Epoch-selection gameplay.** **VERIFIED**, the production hook requires a `32`-world probe each epoch. **ESTIMATED**, `4–6 × 32 = 128–192` gameplay worlds per fit, before the separate round screen. These fresh trajectories cannot be replaced with cached observations, because student actions change the future. [train.py:143](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/research/t34_exit_b8_fit/train.py:143), [train.py:247](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/research/t34_exit_b8_fit/train.py:247).

I would therefore reject the portfolio’s blanket assertion that fits and screens are GPU-bound as a measurement claim. It is not supported by the production timing evidence in this checkout.

**The loss design is a more specific obstacle than insufficient optimization throughput.**

The current objective explicitly preserves the previous student even on corrective rescue rows. **VERIFIED**, KL and advantage-MSE both point toward previous action preferences; only the forbidden-action hinge pushes toward the rescue. Multiplying a row’s weight scales both forces together. [loss.py:43](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/research/t34_exit_b8_fit/loss.py:43).

The prior proposal’s counterexample was independently reproduced in the judgment. **VERIFIED recorded synthetic results:** previous Q was `[0.225, −0.225, −10, −10, −10, −10]`, rescue action was `1`, and forbidden action was `0`.

| Student behavior | Recorded loss — **VERIFIED** |
|---|---:|
| Keep original Q | `0.550` |
| Tie old and rescue actions | `0.792` |
| Still favor old action by `0.147` | `0.298` |
| Favor rescue by `0.1` | `2.126` |

The optimizer can reduce loss substantially while retaining the known mistake. More epochs or faster updates can optimize that conflict more efficiently. This is a counterexample to the objective, **not evidence that a production T34 fit has already failed**. [astra_distill.md:11](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/advice/2026-10-20_compute_portfolio_astra_distill.md:11), [FABLE_JUDGMENT.md:9](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/advice/2026-10-20_compute_portfolio_FABLE_JUDGMENT.md:9).

The history makes this worth resolving:

- **VERIFIED:** T3 LABELS attained `99.66%` training-bank agreement, yet its scripted mass deficit was `−75.145`, including `−55.327` on frames when both policies were alive. Bank imitation accuracy was therefore insufficient. The document explicitly treats this as accounting and descriptive evidence, not causal proof of spillover. [t3_diagnostics_2026-10-16.md:26](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/t3_diagnostics_2026-10-16.md:26), [same file:67](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/t3_diagnostics_2026-10-16.md:67), [same file:181](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/t3_diagnostics_2026-10-16.md:181).
- **VERIFIED:** T4c reduced growth drift but achieved only `0.478254` decisive agreement and still received KILL, with scripted survival `537/960` versus `550/960`. Strong preservation alone was also insufficient. [t4c_screen_result_2026-10-16.md:25](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/t4c_screen_result_2026-10-16.md:25), [same file:94](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/t4c_screen_result_2026-10-16.md:94).

My interpretation is that **correction versus retention is the first learning question to isolate**. It does not follow that more data is useless. Diverse student-visited failure states may help; redundant confirmations and neighboring frames from the same escape opportunity may contribute little new information.

The adopted L3 experiment addresses that question directly: **VERIFIED**, its `2×2` compares the frozen loss against anchor-only KL plus rescue correction, crossed with trainable B8 versus frozen trunk plus a zero-initialized residual. Shared labels and cache make this more informative than an unrestricted temperature/margin sweep. [FABLE_JUDGMENT.md:37](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/advice/2026-10-20_compute_portfolio_FABLE_JUDGMENT.md:37).

I would judge that experiment through three linked observations:

1. Does the student learn corrections on **whole-world held-out** rescue states?
2. Do those rescues remain successful when subsequent actions come from the student?
3. Does ordinary play retain its growth, boost behavior and survival?

The third question needs gameplay, not only label accuracy. **VERIFIED existing tripwires** include decisive agreement below `70%`, disagreement above `5%` in any length decile, and boost share above `1.5×` the previous student. [P7_STATISTICS.md:43](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/final_run_20261019/P7_STATISTICS.md:43). If a residual changes the serving architecture, its qualification must be explicit; a foldable linear correction would better preserve the stated same-B8 objective.

**A100 placement should follow the measured critical path.**

For collection, cache regeneration, and gameplay evaluation, I would make **no assumption about an A100 runtime’s host speed or core count**. The teacher’s quoted approximately `95%` CPython share is itself presented as an estimate in the underlying review, and the later engine changed that workload. The recorded batch-one GPU/CPU comparison is also not a matched A100-host benchmark. [t34_a5_speed_fable.md:28](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/advice/2026-10-20_t34_a5_speed_fable.md:28).

For cached fitting, the existing design gives a sensible trigger: **VERIFIED**, cache first, time a full L4 fit, and consider A100-40 only if it remains GPU-bound beyond approximately an hour. [DESIGN_V2_SHADOW.md:70](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:70).

The future qualification should separately measure cache build/validation, host batch preparation, transfers, forward/backward/Adam, held-out scoring, probes, and total wall time. If an A100 comparison becomes justified, use identical cached inputs and the unchanged numerical recipe. Record CPU model/topology and time a fixed replay/packing slice on both hosts; core count alone is insufficient.

**ESTIMATED illustrative Amdahl calculation:** even a `3×` faster GPU improves total fit speed by only  
`1 / (0.9 + 0.1/3) = 1.07×` if GPU work occupies `10%` of the fit. At an `80%` GPU share it becomes  
`1 / (0.2 + 0.8/3) = 2.14×`.

Cost neutrality is secondary to candidate arrival, but **ESTIMATED** at the supplied rates it requires speedups of `5.4/1.7 = 3.18×` for A100-40 or `7.5/1.7 = 4.41×` for A100-80. A smaller speedup can still be worthwhile with prepaid units if it saves material critical-path time. Saving a few optimizer minutes while collection and probes take hours is a weak reason to occupy an assignment.

**For the next approximately 100 units, I would propose this conditional tranche.** These are **ESTIMATED caps/allowances**, to freeze before data; they preserve the existing premise and retention dependencies.

| Work | Units | Decision enabled |
|---|---:|---|
| Production CUDA qualification and fit-time decomposition | `8` | Establish actual full-fit throughput and identify the limiting stage |
| One admitted label round | `30` | Based on approximately `28–29.4 u` from the current A5 projection; reprice for actual student trajectories |
| Initial L3 projection fits on the same cache | `12 = 4 × 3` | Compare objective and trunk choices; the per-fit amount remains provisional |
| One prospectively selected finalist’s fresh development screen | `30` | Decide whether the selected correction transfers to gameplay |
| Withheld reserve | `20` | Setup/export uncertainty and registered contingencies |
| **Total** | **100** | A projection decision and potentially a candidate for the next required stage |

This is **not the complete L3 study**. **ESTIMATED**, four screens plus the initial fits and additional winner fits already cost  
`4 × 30 + (4 + 2) × 3 = 138 u`, before collecting labels. A single-finalist screen would require a prospective selection rule; it cannot be substituted silently for already-required arm screens or presented as a comparative gameplay result for every arm. If those obligations remain unchanged, treat this as the first tranche toward the full study.

Nor does this budget guarantee a strict-gate candidate. Preserve the required next DAgger round when the first round is non-terminal. A surviving checkpoint still needs fresh Mac strict evidence and serving qualification, including the binding Watch product bars. **VERIFIED**, the active contract requires pooled mass gain at least `+40` with positive LB90 and scripted no-death gain at least `+0.03`, alongside its other guards. [SUCCESS_CONTRACT.md:27](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/final_run_20261019/SUCCESS_CONTRACT.md:27).

I would not spend this tranche on A100 teacher farming based on assumed host superiority, broad coefficient grids, extra fit seeds before correction-plus-retention works, or larger label banks justified only by row count. The useful accelerator upgrade is the one that demonstrably brings a gameplay-qualified student forward. At this snapshot, completing the fitter’s production path and resolving its correction–retention conflict have the stronger claim on that role.



## Call 2 — prompt (verbatim)

```text
Follow-up on your previous answer (same worktree, same rules). Same RULES: read-only, no edits/commits/new files in the repo; at most tiny read-only probes on the Mac (1 thread, `nice -n 10`, < 30 s each; a 6-worker Phase 0 study is running and must not be disturbed); no Colab/RunPod/network jobs; full answer in your final message with file:line evidence and numbers; mark every claim VERIFIED vs ESTIMATED.

Your previous answer is saved at /Users/josenunez/Projects/ml/snake-dqn-artifacts/a100-advice/p3-call1/snake-dqn-advice-a100-3.last-message.md — read it first. Its conclusion: keep L4; the fit is not shown to be GPU-bound; the first learning question is correction-vs-retention in the loss (the synthetic counterexample); spend ~100 u on CUDA fit qualification + one label round + initial L3 fits + one finalist screen.

I want you to cross-examine your own answer. Open questions, in your lens (learning/distillation):

1. Your weakest claims. You rejected "fits are GPU-bound" as unmeasured, but the only hardware number you cited (M3-C: ~30 updates/s on L4 vs ~80 on an A100 learner) is a 2.7x GPU-side difference on a workload with the same 1.64M-param network. What in research/t34_exit_b8_fit/train.py, data.py and cache.py lets you bound the per-epoch host cost (row-dict reconstruction, normalize/validate, np.stack, hash-on-reopen) relative to the optimizer time, with actual numbers? Is there any recorded timing of this fitter or its M2 predecessor (research/redesign_m2_20261008/train_student.py, PREREGISTRATION_M2B.md, the M2b receipts, any *.json under the artifacts tree you can read in < 30 s) on the Mac that pins rows/s? If you can bound it with a < 30 s single-thread probe of the cache/data path on a tiny synthetic input, do so and report the numbers; if not, say precisely what cannot be known without the qualification run.

2. The loss counterexample. The synthetic probe used Q_prev = [0.225, -0.225, -10, ...], i.e. a 0.45 gap between the old action and the rescue. The campaign records B8's typical Q gaps as 0.06-0.45 and the death-window/decisive rows get 20x weight within a per-world weight of 10. Does the conflict survive at realistic gaps and weights, or is it an artefact of the probe's numbers? What is the actual gradient balance between the KL(prev||student) at tau=0.05 and the hinge at margin 0.1 on a decisive row where the rescue action trails by 0.1? Show the arithmetic from loss.py as written. Could the conflict be the reason T3 students "learned the bank" but still died, or is there a simpler explanation in the T3/T4c records?

3. The 32-world probe per epoch and the screens. You said gameplay (probe + screens) may dominate a fit. Where does that gameplay run (host CPU, how many workers, what rate), and is there any measured per-world time for a bare-student world on an L4 runtime versus the Mac? Does an A100 runtime change anything there, or is it purely host-bound again?

4. What would change your mind. State concretely the measurement results that would make you say "an A100 shortens the time to a strict-gate candidate": numbers for (a) fit wall-time decomposition on L4, (b) A100 host vCPU/per-core speed, (c) anything else. And the single cheapest measurement (units, wall-clock, what it reports) that would settle the A100 question for the distillation loop specifically.

5. Label volume vs projection. If the L3 2x2 shows the loss fix works (>= 70% held-out decisive agreement, <= 5% all-length disagreement), does the time to a candidate then become label-production-bound (and thus host-bound) or fit/screen-bound? Give your best estimate of the critical path in wall-clock days from "R1 labels exist" to "a bare student enters the Mac strict gate", with the per-stage numbers and which stages an A100 could touch at all.

Be blunt about anything in your first answer you would now retract or weaken.
```

## Call 2 — Astra answer (verbatim)

**ESTIMATED — I would weaken my first answer substantially on throughput, while retaining the loss objection.** A100 acceleration of the cached fit is plausible; I did not have enough evidence to imply that host preparation would overwhelm it. I also missed historical fit and bare-student screen timings that make the **3 u/fit and 30 u/screen allowances poor central estimates**. The loss conflict survives realistic Q gaps and the actual weighting, but it is **not an established explanation of T3’s failure**.

**VERIFIED — scope:** I read the saved answer first and inspected the same clean, detached checkout, `6f1e751f`. I made no repository changes and ran no remote jobs. Two local probes used `nice -n 10`, one-thread limits, disabled bytecode writes, and a 25-second alarm; they completed in approximately **0.93 s and 0.65 s**. Their output stayed in this conversation. Below, **VERIFIED** covers inspected records, code, and directly checked arithmetic; **ESTIMATED** covers extrapolation, interpretation, and proposed measurements.

**1. The host-cost argument was weaker than I presented it.**

**VERIFIED — the code identifies costs, but does not establish their production proportions.** Cache reopening hashes every payload once; each epoch then opens mmap arrays, parses metadata, reconstructs row dictionaries, normalizes and validates all source rows, selects training rows, and stacks selected arrays. Hashing is **per cache opening, not per epoch**. The loader already records its active processing time separately from time suspended while training consumes a batch. Sources: [cache.py:345](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/research/t34_exit_b8_fit/cache.py:345), [cache.py:370](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/research/t34_exit_b8_fit/cache.py:370), [data.py:168](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/research/t34_exit_b8_fit/data.py:168).

**VERIFIED — my tiny probe exercised the actual reconstruction and loader code with 2,048 synthetic rows**, half ranked rescue rows and half shadow confirmations, batch 1,024, bare-B8 observation shapes. It substituted RAM arrays and `StringIO` metadata for filesystem-backed mmap files. Four stage measurements produced:

| Stage | Measured median µs/source row |
|---|---:|
| JSON parsing and row-dictionary reconstruction | 2.63 |
| `normalize_row` plus `validate_row` | 3.94 |
| Bank sampling and sample weights | 1.91 |
| `_stack` | 3.92 |
| Separate complete-loader measurement | **14.43** |

**VERIFIED:** the complete loader consumed 2,048 rows in **0.02956 s: 69,292 rows/s**. Hashing approximately 25 MB of RAM-backed raster bytes in 1 MiB chunks achieved **2.92 GB/s**.

**ESTIMATED — interpretation:** at that tiny-fixture rate, one million source rows, all selected, cost roughly **14.4 seconds/epoch** in loader processing. Hashing the approximately **12.47 GB/M rows** of bare observation/target arrays would take **4.3 seconds** at that RAM hashing rate, before disk and metadata costs. Neither is a production upper bound: the probe excludes mmap creation/slicing overhead, page faults, storage reads, production metadata complexity, and cache regeneration. It also does not establish the source-row/selected-row ratio.

That changes the comparison materially:

- **ESTIMATED:** `30 updates/s × 1,024 rows/update` corresponds to **32.55 optimizer µs/row**.
- **ESTIMATED:** combining that hypothetical optimizer rate with the measured Mac fixture’s **14.43 loader µs/row** makes optimization approximately **69% of their combined time**.
- **ESTIMATED:** accelerating just that optimizer from 30 to 80 updates/s gives **1.76×**, not merely a negligible improvement, for this simplified loading-plus-training loop.
- **ESTIMATED sensitivity:** if the L4 host takes 2.4–3.7 times as long as this Mac fixture, the same calculation falls to approximately **1.31–1.43×**. That host multiplier is a historical planning assumption, not a measurement of this data path.

**VERIFIED — the historical 2.7× signal deserves weight, but is not an isolated GPU comparison.** The L4 figures were full N=3 runs at **29.3–29.8 updates/s, 87–90% GPU utilization**; the A100 figures included isolated learners at **78–82.7 updates/s**. Same network family, different surrounding work and scheduling. [L4 record:497](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/t16_colab_smoke02_results_2026-10-17.md:497), [A100 record:187](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/t16_colab_smoke02_results_2026-10-17.md:187).

**VERIFIED — I missed substantial M2b fit timing evidence.**

| Historical M2b fit | Recorded throughput across epochs | Approximate epoch-1 duration at batch 1,024 |
|---|---:|---:|
| Round 10 | 24,861–29,327 rows/s | **67 s — ESTIMATED** |
| Round 11 | 23,286–25,034 rows/s | **112 s — ESTIMATED** |
| Round 12 | 26,012–26,228 rows/s | **141 s — VERIFIED arithmetic using registered batch** |

The first-epoch receipts contain respectively **1,776, 2,660, and 3,589 updates**. The final M2b recipe registers batch 1,024 and six epochs. Sources: [r10-metrics.json:18](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/research/redesign_m2_20261008/results_m2b/fits/r10-metrics.json:18), [r11-metrics.json:18](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/research/redesign_m2_20261008/results_m2b/fits/r11-metrics.json:18), [r12-metrics.json:19](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/research/redesign_m2_20261008/results_m2b/fits/r12-metrics.json:19), [PREREGISTRATION_M2B.md:23](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/research/redesign_m2_20261008/PREREGISTRATION_M2B.md:23).

**VERIFIED:** those rates include epoch streaming, optimization, and held-out scoring; they exclude initial loading/counting and subsequent checkpoint saving. They do not include a gameplay probe. [train_student.py:259](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/research/redesign_m2_20261008/train_student.py:259).

**ESTIMATED — hardware attribution:** Mac/MPS is strongly suggested by the trainer’s intended use and default `--device mps`, but these metrics do not bind the actual device or command. I cannot present them as hardware-attested Mac receipts. [train_student.py:201](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/research/redesign_m2_20261008/train_student.py:201).

**VERIFIED:** the current fitter also has timing receipts, but the inspected bare-cache example has only **16 synthetic rows**, **0.0432 s cache construction**, and epochs of **0.0762/0.0399 s**. It explicitly says `MAC_SYNTHETIC_SMOKE_ONLY`. These establish instrumentation and wiring, not production throughput. [timing.json:16](/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-v2-fitcache-20261020/verification.jwOsbR/final-s1b/timing.json:16).

**ESTIMATED — revised conclusion:** “Production T34 fit throughput is unmeasured” remains correct. “Host work probably makes A100 irrelevant” was too strong. What remains unknown is actual cache regeneration/reopen time, source-to-selected row ratio, L4 loader throughput, synchronized optimizer time, held-out cost, and gameplay-probe cost.

**2. The loss conflict survives realistic gaps and weights.**

**VERIFIED — exact objective:** for a row of weight \(w\) in a batch of \(B\),

\[
L_i=\frac{w}{B}\left[
D_{\mathrm{KL}}(p_{\rm prev}\Vert p_s)
+\sum_{f\in F}[Q_f-Q_r+0.1]_+
+0.01\,\mathrm{MSE}(A_s,A_{\rm prev})
\right],
\quad p=\mathrm{softmax}(Q/0.05).
\]

There is **no \(\tau^2\) multiplier** on KL. Therefore

\[
\frac{\partial KL}{\partial Q_j}
=\frac{p_{s,j}-p_{{\rm prev},j}}{0.05}.
\]

Both statements follow directly from [loss.py:43](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/research/t34_exit_b8_fit/loss.py:43).

**VERIFIED — arithmetic for a 0.1 previous gap:** take old action \(o\), rescue \(r\), the other four actions far below them, and \(Q_{\rm prev,o}-Q_{\rm prev,r}=0.1\). Then

\[
p_{\rm prev,o}=\sigma(2)=0.880797,\qquad
p_{\rm prev,r}=0.119203.
\]

For one active forbidden action, the following gradients are before multiplying by \(w/B\):

| Student state | KL gradient on old/rescue | Hinge gradient | Weighted-MSE gradient | Total gradient |
|---|---:|---:|---:|---:|
| Still trails by 0.1; identical to previous Q | \(0,\ 0\) | \(+1,\ -1\) | \(0,\ 0\) | **\(+1,\ -1\)** |
| Old and rescue tied | \(-7.61594,\ +7.61594\) | \(+1,\ -1\) | \(-0.000167,\ +0.000167\) | **\(-6.61611,\ +6.61611\)** |

**VERIFIED:** initially the hinge does improve the rescue. At the tie, gradient descent instead raises the old action and lowers the rescue. Thus “KL dominates immediately at initialization” would be wrong; **KL stops the correction before the action flips**.

For the two-action slice, with student gap \(d=Q_o-Q_r\) and previous gap \(d_0\),

\[
\frac{\partial L}{\partial d}\frac{B}{w}
=
\frac{\sigma(d/0.05)-\sigma(d_0/0.05)}{0.05}
+1+\frac{0.01}{6}(d-d_0).
\]

**VERIFIED — solving this expression, including advantage MSE, gives:**

| Previous old-over-rescue gap | Stationary student gap |
|---|---:|
| 0.06 | **+0.04686** |
| 0.10 | **+0.07956** |
| 0.45 | **+0.14712** |

**VERIFIED:** all three retain the wrong ordering. The 0.45 example was not outside the historical rescue regime: T1’s late-death LAD gap median was **0.44695**, with quartiles **0.21879 and 1.27960**. [T1 study:44](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/t1_commitment_study_2026-10-15.md:44).

**VERIFIED — weighting does not repair this ratio.** For a current-round death window containing \(D\) decisive and \(N\) ordinary nonzero-weight rows,

\[
w_{\rm decisive}=\frac{10\times20}{20D+N}.
\]

Examples: one decisive row alone gets weight **10**; one decisive plus 19 ordinary rows gets **5.128**; ten decisive rows get **1 each**. All loss terms receive that same weight. The implementation normalizes **each death window**, not necessarily an entire world containing multiple windows. [data.py:52](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/research/t34_exit_b8_fit/data.py:52).

**ESTIMATED — scope of the objection:** this is a realistic counterexample, not proof that every decisive row fails. Multiple active forbidden actions, all six action probabilities, and shared-network gradients can change the outcome. A gap alone does not fully specify the six-action gradient. The decisive point is that the current objective can prefer retaining a known fatal action even at a 0.06–0.1 gap.

**VERIFIED — T3 is not evidence that this particular conflict caused earlier failure.** T3 used TD plus a surviving-set margin anchor, not this T34 KL objective. Its LABELS arm learned **99.64% decisive agreement**, yet lost **75.15 mass**, of which **55.33** came from lower mass while both policies were alive. It actually had **fewer deaths: 199 versus 217**. Its decisions disagreed with CTRL **67.27%** of the time, with boost-mode changes accounting for **75.30%** of disagreements. [T3 diagnostics:26](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/t3_diagnostics_2026-10-16.md:26), [interpretation:181](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/t3_diagnostics_2026-10-16.md:181), [T3 anchor implementation:208](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/research/t3_rollout_labels_20261015/bank.py:208).

**ESTIMATED:** the simpler explanation supported by those records is learned corrections accompanied by broad behavioral drift and lost growth, plus imperfect transfer from B8-continuation labels to student trajectories. T4c illustrates the other side: growth drift shrank to **−7.38 shared-survivor length**, but decisive agreement was only **47.83%**, and scripted survival was **537/960 versus 550/960**. That supports studying correction versus retention; it does not identify the present KL conflict as the historical cause. [T4c result:25](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/t4c_screen_result_2026-10-16.md:25), [T4c result:94](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/t4c_screen_result_2026-10-16.md:94).

**3. Gameplay might dominate a short fit, but its placement and cost were overstated.**

**VERIFIED:** the current fitter calls an external `probe_hook(model, epoch)`. Validation requires **32 unique complete H5000 trajectories**, with previous/current actions and lengths. It does not select a device, worker count, or evaluator implementation. Production replay/probe integration remains an external boundary. [train.py:261](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/research/t34_exit_b8_fit/train.py:261), [diagnostics.py:69](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/research/t34_exit_b8_fit/diagnostics.py:69), [FITCACHE_VERIFICATION.json:33](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/research/t34_exit_b8_fit/FITCACHE_VERIFICATION.json:33).

**VERIFIED — historical bare-student Mac gameplay is much cheaper than the portfolio allowance:**

| T4c bare student, H5000 | Recorded evaluator wall time | Amortized seconds/world |
|---|---:|---:|
| Scripted, 96 worlds | 196.109 s | 2.043 |
| Frozen, 32 worlds | 110.937 s | 3.467 |
| Mixed, 32 worlds | 95.925 s | 2.998 |
| **Total, 160 worlds** | **402.970 s = 6.72 min** | **2.519** |

**VERIFIED:** these used CPU rowwise inference, numba vector61 opponents, fast anchors, and at most two single-thread worker processes. Worlds advanced through the batched evaluator; the table is amortized bank throughput, not serial single-world latency. T3’s corresponding student bank took **7.26 minutes**. [T4c probes.jsonl:19](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/research/t3_screen_20261015/t4c_results/probes.jsonl:19), [execution conditions:111](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/t4c_screen_result_2026-10-16.md:111).

**ESTIMATED:** proportional scaling gives approximately **1.34 minutes for 32 worlds**, or **5.4–8.1 minutes across four to six epochs**, before additional previous-policy diagnostics and orchestration. A fresh separately simulated comparator approximately doubles screen numerical work. Smaller banks need not preserve the same amortization.

**VERIFIED — evidence limitation:** I found no matched bare-student L4-versus-Mac H5000 measurement. Therefore neither “GPU-bound screens” nor “purely host-bound production probes” is established.

**ESTIMATED:** A100 could accelerate GPU inference if the eventual probe batches it efficiently. It cannot directly accelerate CPU simulation, observation construction, opponent hooks, or Python bookkeeping. More or faster host cores could help independently, but that requires a measured host advantage. My earlier answer blurred those possibilities.

**4. What would change my mind about A100.**

**ESTIMATED — proposed decision rule:** choose A100 when a matched measurement predicts **at least one hour earlier candidate arrival across the remaining dependent stages**, after setup, transfer, allocation, and host-speed differences. This is a proposed decision threshold, not an existing authorization.

A concrete result that would persuade me:

| Measurement | Persuasive result — ESTIMATED threshold/example |
|---|---|
| L4 complete fit decomposition | **60 min total:** 10 cache/reopen, 5 host loading, 40 accelerator work including held-out forwards, 5 gameplay/export |
| Matched A100 optimizer/forward speed | **≥2.5×** at identical batch, fp32 recipe, labels, and update count |
| A100 host | **At least 12 usable vCPUs**, and measured single-worker cache/gameplay throughput no worse than **90% of L4** |
| Remaining dependency chain | At least **three serial fit waves**, so approximately **24 min saved/wave → 72 min** before extra allocation costs |
| Alternative host-driven justification | Matched gameplay/replay throughput **≥1.5× L4** at an appropriate worker count, regardless of GPU utilization |

**VERIFIED:** historical A100 runs already recorded **12 vCPUs**; the statement that A100 topology is wholly unmeasured needs narrowing to the current assignment and relevant workload. Twelve vCPUs do not demonstrate an advantage over the recorded 12-vCPU L4. [A100 record:210](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/t16_colab_smoke02_results_2026-10-17.md:210).

**ESTIMATED — cheapest honest measurement:** one matched **L4/A100-40 distillation timing packet**, using the same small representative production cache:

- Approximately **65,536 rows**, including rescue and anchor metadata, with fixed batch 1,024.
- Warm up, then time repeated fixed update sequences both with the actual loader and with pre-staged batches.
- Separately time cache reopening/hash verification, host preparation, transfer, synchronized forward/backward/Adam, and held-out scoring.
- Run a small fixed bare H5000 bank through the **actual intended probe path**, recording topology and worker count.

**ESTIMATED cost:** budget **15–20 allocated minutes per device**, including setup where feasible: **1.78–2.37 units total** at the supplied 1.7/5.4 u/h rates. Sequential elapsed time is **30–40 minutes**. This is a proposed packet, not a guaranteed runtime; setup or absent integration can invalidate that budget.

**ESTIMATED:** one L4-only profile can rule out a useful A100 upgrade when accelerator work is tiny. It cannot establish actual A100 speed. A matched two-device packet is the cheapest defensible way to settle the comparison rather than substitute another architectural guess.

**VERIFIED:** the existing design currently says to consider A100 only after a cached L4 fit remains GPU-bound beyond approximately one hour. **ESTIMATED:** I would weaken my endorsement of that scientific cutoff: several shorter dependent fits can cumulatively justify A100. Changing that placement rule would be a prospective proposal. [DESIGN_V2_SHADOW.md:70](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/final_run_20261019/DESIGN_V2_SHADOW.md:70).

**5. After a successful L3 projection fix, label production probably becomes the largest next stage—but that is conditional.**

**VERIFIED:** passing 70% decisive agreement and 5% drift clears learning tripwires; it does not establish gameplay improvement. The 5% condition applies to **every length decile**, not just overall disagreement. R1 remains non-terminal; R2 tests student visitation at \(\beta=0.5\). L3 specifies four initial fits plus two additional fits on the winner. [P7_STATISTICS.md:43](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/final_run_20261019/P7_STATISTICS.md:43), [FINAL_PLAN.md:89](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/final_run_20261019/FINAL_PLAN.md:89), [portfolio judgment:37](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/advice/2026-10-20_compute_portfolio_FABLE_JUDGMENT.md:37).

**ESTIMATED — my revised central forecast is 1–3 wall-clock days from “R1 labels exist” to a bare checkpoint entering the Mac strict gate**, assuming production integration and CUDA qualification are already complete, three assignments are available, the historical fast bare evaluator transfers, and every scientific stage passes.

| Dependent stage | Estimated elapsed time | What A100 could affect |
|---|---:|---|
| R1 cache preparation/validation | 0.5–1.5 h | Host/I/O; no inherent GPU gain |
| Four L3 fits, two waves, including epoch probes; initial screens | 1.5–5 h | Optimization and GPU inference |
| Two winner-replicate fits and their screens | 1–2.5 h | Same; screening both is a conservative forecasting assumption |
| R2: 800 new labeled worlds, three lanes | **5–6 h** | Only demonstrated host/rollout-inference gains |
| R2 cache plus selected fit, including probes | 1–3 h | Optimization and inference portions |
| R2 deciding development screen | 0.5–1 h | Depends on actual evaluator placement |
| Freeze, verify exports, prepare strict handoff | 0.5–2 h | Little accelerator benefit |
| **Numerical/operational work before queue/review slack** | **Approximately 10–21 h** | Only part is GPU-sensitive |

**VERIFIED input / ESTIMATED transfer:** the label estimate comes from A5’s **54.4–57.2 completed worlds/hour/lane**: \(800/(3r)=4.66–4.90\) hours, rounded upward for overhead. Student trajectories may change that rate. One lane instead of three makes it approximately **14–15 hours before overhead**. [CAMPAIGN_LOG.md:57](/Users/josenunez/Projects/ml/snake-dqn-advice-a100-3/docs/research/final_run_20261019/CAMPAIGN_LOG.md:57).

**ESTIMATED — bottleneck distinction:** after L3 has already succeeded, R2’s **5–6 hours of labels** likely exceeds its subsequent **1.5–4 hours of cache/fit/screen work**. From the earlier starting point of “R1 labels exist,” the L3 comparison and its repeated probes can collectively rival label production. Thus “label-bound after projection succeeds” is plausible; “label-bound throughout” is not established.

**ESTIMATED — slow case:** retaining the portfolio’s **3 u/fit and 30 u/screen** allowances gives approximately **71 hours** of dependent scheduling under a six-fit/six-screen L3 interpretation, then roughly **4–6 days** with overhead. Historical bare-screen records make that a conservative scenario rather than my central forecast. An additional admitted R3 adds another label/fit/screen cycle. Missing production integration, a failed scientific bar, or an occupied Mac can extend either estimate without a defensible bound.

**ESTIMATED — my revised recommendation:** keep L4 as the provisional placement, but replace the loose hardware argument and oversized screen allowance with the matched timing packet. The strongest retained objection is specific: the frozen loss can refuse realistic rescues. The strongest retraction is also specific: **I had not earned the implication that host overhead makes A100 acceleration unimportant, and I had not adequately researched existing fit and bare-screen timings.**