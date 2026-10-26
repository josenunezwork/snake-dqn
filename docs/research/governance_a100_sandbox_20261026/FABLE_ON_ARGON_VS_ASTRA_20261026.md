# Fable on Argon vs Astra: governance, first A100 jobs, biggest time-saver

2026-10-26 · Fable, independent senior judge · read-only (no edits, dispatch, Colab, tmux; no label/outcome payloads, D4096 or `t34-round0-study-01/` opened). Advice to the owner, not a registration or dispatch authority. Source HEAD 2b3cb441.

Inputs: Argon `exp/argon-20261026/project-review-a100-02/answer.md`; Astra `exp/astra-20261024/ASTRA_X_ARGON_A100_PROJECT_20261026.md`; the Colab-transport pair (Argon -01/-02-xreview, Astra transport review); `docs/notes/governance-and-owner-rules.md`, `docs/STATUS.md`, CAMPAIGN_LOG tail, F5 registration ledger, `src/simd_env/grid_sim.py`, `research/gpu_sim_spike/{device_sim,device_policy}.py`.

## 0. Bottom line

Adopt **Astra's standing-sandbox protocol written with Argon's intent**: one ratified standing rule, after which A100 development jobs need an automated preflight and a 10-line manifest, **never** a per-run judge review; the judge appears only when one frozen candidate crosses into the Mac gate. The two advisors disagree far less on substance than on tone: Astra's rank-1 job *is* the prerequisite for Argon's Job 3. Run that first. Do not run Argon's Job 2 (label factory) as written; run Argon's Job 1 (head/loss sweep) second, under the sandbox rule, as successor development that never touches F5.

## 1. Spot-checks of the disputed claims

| Claim | My reading of the code / rulings | Who is right |
|---|---|---|
| FROZEN14 is a "1-line fix: refill before `step` returns" | `grid_sim.py:719-758` (non-train): the eat loop runs per snake slot over the live `food_set`, and `_maintain_food_env(e)` (`:403-409`, `:755-756`) refills **inside that loop after each eater**, so a later-slot snake in the same tick can eat refilled food and RNG is drawn per eater. `device_sim.py:913-943` precomputes all hits, loops slots, compacts, and defers refill to `prepare()` (`:707-716`). A terminal refill call does not reproduce intra-tick visibility or RNG ordering. | **Astra**, for CPU-exact parity. **Argon** is right that this is irrelevant if the GPU world is declared a separate *development distribution* (see §2). |
| DevicePolicy can host on-policy training | `device_policy.py:26-143`: frozen deep-copied B8 in `.eval()`, `@torch.inference_mode()`, greedy masked argmax; no sampling, log-probs, value head, truncation/terminal bookkeeping or resume. PPO needs a new actor path. This is bounded engineering (days), not research, but Argon's "launch within 48h" is not credible. | **Astra** on readiness; **Argon** on direction. |
| Argon Job 1 (sweep on the 960 selected lives) compromises F5's one-look | F5 is frozen (W vs matched control, `Linear(256,6)`, seed 3403, 2,000 updates, exploratory screening, S2→S3→S4). A separate sweep that never reads F5 outputs, F5 evaluation worlds, or the R30 protected validation parents does not unblind F5. Astra concedes this ("does not by itself mean a separate fit has read F5's future outcomes"). The real constraints are: no feedback into F5, a fresh salt-derived train/val split, and the one-candidate crossing rule. The "once-only selected-corpus admission" is a paper scope, fixable by a short prospective ruling. | **Argon** on statistics; **Astra** on the three guardrails. Astra's "defer all numeric sweeps until F5 terminal" is too conservative. |
| Hash-salted seeds replace historical collision audits | uint64 truncation collision at 1e8 vs 684 seeds ≈ 4e-9: effectively zero. The genuine residual risks are seed normalization (width/mod), aliases and shared ancestry, which a canonical registry catches cheaply. The F5 amendment already accepts known-risk for exploratory use. | **Argon** for development and screens; **Astra** for confirmation banks (registry check stays, retroactive 1.16M-seed archaeology goes). |
| Harness crash before unblinding should not consume a scientific attempt | `JUDGMENT_RESUME_POLICY.md` already separates operational and scientific counters. A stage-02 test-harness OOM is not outcome-correlated in any plausible way. | **Argon**; Astra's "blindness alone does not make crashes outcome-independent" is true in principle and should be a judge checkbox, not a veto. |
| "6,000 world-steps/s = 36,000 hero transitions/s"; "<30-min sweep"; "80-95% confidence" | Seat-steps ≠ hero rows (one learner hero); timings are arithmetic, not forecasts; confidences are conjecture. | **Astra.** Strip these numbers before anyone plans on them. |
| GPU-selected roots + Mac confirmation = 100% CPU provenance | Round-28 judgment: a divergent root's future does not establish natural CPU ancestry; the exact-engine route failed its W12 ≥4× bound. | **Astra.** Job 2 as written is dead; a *declared* GPU development distribution is the live alternative. |

## 2. Governance: adopt this rule text

**Standing A100 Development Sandbox (one-time ratification; no per-job judge review).**

1. *Scope.* A job is a sandbox job if it (a) runs through the qualified supervisor and `tools/ops/dispatch.sh`, (b) reads only inputs on the **Sandbox Input Registry** (initially: closed F0 training rows D256/D1024 excluding F4 fold-0, the 29 census parents, R30 validation parents, and every strict/probe/serving/F5-evaluation bank; and worlds seeded from `uint64(sha256("a100-dev/" + study + "/" + idx))`), (c) writes only checkpoints, engineering timings, or development labels tagged `simulator-D`, and (d) has a finite endpoint (steps or hours) and a 2× runaway timeout.
2. *Process.* Sandbox jobs need an automated preflight (tests + manifest schema + registry check) and a committed 10-line manifest (inputs, salt, endpoint, timeout, excluded studies). **No pre-registration, judge review, timing canary or ratification per job.** They may be chained from a committed ready queue.
3. *Authority.* Sandbox outputs are Tier-1 development evidence at most: never gate evidence, never a label source for a registered study without that study's own premise ruling, never a change to F5 or any running registered study.
4. *Crossing.* At most **one** frozen checkpoint SHA per development line crosses into Mac evaluation (a pre-declared Holm family of ≤2 is allowed if declared before the sweep). Crossing is where the judge appears: freeze weights, commit the registration, fresh salt-derived paired bank, publish-before-read, one look. Mac strict gate, serving qualification and owner release are unchanged.
5. *Isolation.* Sandbox jobs never share an account assignment with a running registered job's manifest and never edit the qualified transport. While F5's timing qualification runs, allocate only if the active F5 contract permits coexistence (3-assignment cap), else wait.
6. *Seeds.* All new studies use domain-separated hash seeds plus an incremental canonical exclusion registry. Retroactive historical collision audits are required only for confirmation banks and are a registry intersection, not an archaeology project.

That is Argon's two zones with Astra's four guardrails (input registry, excluded studies, finite endpoint, one-candidate crossing). It drops Astra's "defer sweeps until F5 terminal" and "no auto-chaining"; both cost calendar time without protecting a look.

## 3. First A100 jobs

**Job A (now, ≈1-2 A100-hours): resident collector + learner feasibility assay** (Astra rank 1 = Argon Job 3's prerequisite). Add a stochastic actor path to DevicePolicy (sampling, log-probs, value, masks, terminal/truncation, resume), E128 shape first, synthetic learner tensors, no F5 inputs. Report useful hero rows/s for collection + learner + resets. Decision: if the integrated cycle beats the qualified L4 P1 path by a measured margin, charter resident PPO (Argon Job 3) with a fixed endpoint and one-look; else stop the port. This is the only job where the A100 is genuinely load-bearing.

**Job B (now, in parallel, under the sandbox rule): Argon Job 1, the frozen-trunk head/loss sweep** on the sandbox-admitted training rows with a fresh salt split (never the R30 validation parents, never F5 evaluation worlds, never F5 outputs). Label it "successor development", not "F5 prep". It is cheap and keeps the Mac free; its value is a ready successor if F5's S2 fails. Fix Argon's architecture (it is `Linear(256,6)`, not 512→3) and drop the "eliminates leakage by construction" gate claim; test it instead.

**Defer:** Argon Job 2 (label factory) until Job A reports and until a named consumer exists; if revived, declare it a `simulator-D` development distribution and add a transfer premise test (student trained on sim-D labels, screened on CPU worlds) rather than claiming CPU provenance. Argon Job 4 (RNG bridge) only if a consumer needs production-generator parity; buffered-MT tape is the path, not a counter-RNG port.

## 4. Biggest process change

**Qualify operations once per transport/supervisor commit, not once per study, and stop sealing operational packets for development work.** The F5 ledger shows roughly ten requalification, canary, refusal and successor packets to get a four-hour timing run started; none of that protects a scientific look. Rule: transport qualification binds to a commit and is reused by every job on that commit; per-job operations are an automated preflight; judge-sealed packets exist only at the Zone-2 crossing. This single change removes most of the calendar cost Argon calls "ceremony" and weakens nothing Astra defends. Second: adopt the attempt-counter clarification (operational failures before any outcome computation never consume a scientific attempt; judge checkbox for outcome-correlated failure modes).

## 5. Where each is right

- Argon right / Astra too conservative: per-run review for development compute; deferring sweeps until F5 terminal; treating hash-truncation collisions as a live risk; "no automatic chaining" against the owner's never-idle rule; counting harness crashes as scientific attempts.
- Astra right / Argon overreaching: FROZEN14 is not one line; DevicePolicy cannot train today; GPU roots cannot acquire CPU provenance; rates, hours and confidences are unmeasured; "F5 awaiting adoption" is stale (W already selected and frozen); transport pin c1484312 is stale (6e8c1982 is current).
- Transport (Argon -02 xreview): Argon conceded every WRONG mark; the surviving idea (single-stream bulk input, prebuilt wheelhouse) is a future package, not a change to the running F5 operation.

Nothing here touches the running F5 L4 timing qualification. Release decisions stay with the owner.
