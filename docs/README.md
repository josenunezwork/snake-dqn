# Documentation map

Start with [STATUS.md](STATUS.md) for the released agent, current research
direction, and operational decisions as of 2026-10-08. Then use the
[research index](research/INDEX.md) to find a dated decision or result, and
the [glossary](research/GLOSSARY.md) for the evaluation vocabulary. Dated
records preserve their original outcome; they are evidence, not permission to
resume an old plan.

## Current reference and decision records

| Document or collection | Status | Use it for |
| --- | --- | --- |
| [STATUS.md](STATUS.md) | **Current** | Released `frp3-s12 + v8`, standing decisions, current results, questions, and compute-policy pointers. |
| [research/INDEX.md](research/INDEX.md) | **Current map** | Chronological locator for research and decision documents on `main` and study branches. |
| [research/GLOSSARY.md](research/GLOSSARY.md) | **Current reference** | Terms used by the release, gate, and redesign records. |
| [research/](research/) | **Mixed: current decisions and historical evidence** | Release receipts, governance amendments, compute/performance notes, diagnostics, and prior study decisions. Use the index to determine a record's status. |
| [ml_algorithm.md](ml_algorithm.md) | **Current reference, limited scope** | The released Apex/vector policy's algorithm and serving-facing concepts. It does not prescribe the active redesign program. |
| [simd_env_spec.md](simd_env_spec.md) | **Current technical contract** | `BatchSim` dynamics and parity/observation contract; consult source and tests for behavior changes after the dated specification. |
| [ml_redesign_blueprint_2026-07.md](ml_redesign_blueprint_2026-07.md) | **Historical foundation** | The July rationale and original redesign blueprint. Its audit and implementation milestones are not a current run plan; use `STATUS.md` for the adopted October direction. |

## Released-agent and evaluation record

| Document or collection | Status | Use it for |
| --- | --- | --- |
| [research/frp3_serving_result_and_release_2026-10-06.md](research/frp3_serving_result_and_release_2026-10-06.md) | **Current release receipt** | Why `frp3-s12 + v8` became the served default and its serving qualification. |
| [research/frp3_strict_result_2026-10-06.md](research/frp3_strict_result_2026-10-06.md) | **Current release evidence** | The strict result, including the survival-band disclosure. |
| [research/governance_amendment_survival_band_v2_2026-10-06.md](research/governance_amendment_survival_band_v2_2026-10-06.md) | **Current for opted-in strict studies** | Ratified survival-band-v2 rule and its scope. |
| [research/governance_amendment_sequential_phase_r_2026-10-07.md](research/governance_amendment_sequential_phase_r_2026-10-07.md) | **Current for Phase R** | Ratified sequential Phase R rule; do not apply it retroactively. |
| [research/governance_amendment_strict_on_runpod_2026-10-05.md](research/governance_amendment_strict_on_runpod_2026-10-05.md) | **Not ratified; superseded 2026-10-08** | Historical proposal for strict gates on RunPod. Never used. Hash-bound, so its text is unchanged; its [status note](research/governance_amendment_strict_on_runpod_2026-10-05_status_2026-10-08.md) records the outcome and what a new amendment would need. Strict gates and serving qualification stay on the Mac. |
| [research/governance_amendment_long_horizon_screen_2026-10-04_erratum_2026-10-08.md](research/governance_amendment_long_horizon_screen_2026-10-04_erratum_2026-10-08.md) | **Erratum** | The branch-only LH-1 amendment (`lh1-screen`) opens with `DRAFT`; it was ratified 2026-10-05 (`ecddfab`). The file is hash-bound and left unedited. |
| [research/governance_tiers_2026-09-26.md](research/governance_tiers_2026-09-26.md) and [governance_amendment_sequential_gates_2026-10-02.md](research/governance_amendment_sequential_gates_2026-10-02.md) | **Historical policy base, amended** | Original tier and sequential-gate rules. Read later amendments before using either as an operating rule. |

## Research archive

| Collection | Status | Contents and starting point |
| --- | --- | --- |
| [experiments/](experiments/) | **Historical study archive** | Per-study protocols, results, ledgers, and tables. Start with [the September portfolio](experiments/rl_research_portfolio_2026-09-12/README.md), then use each study `README.md`; tables are supporting data for the adjoining report. |
| `experiments/controlled_encounter_*` | **Historical study family** | September controlled-encounter diagnostics and learning screens. Each directory names the intervention and date. |
| `experiments/solo_*` and `experiments/solo_food_*` | **Historical study families** | Solo food, safety, native-policy, and PQN probes. Names encode the question; read the family portfolio before interpreting a single result. |
| `experiments/apex_*` | **Historical study family** | Apex continuation, retention, scaling, and veto screens. The v5-v8 subdirectories are evidence leading to the released veto; the release receipt above is authoritative for what is served. |
| `experiments/task_aligned_*` | **Historical bridge/strict family** | Qualification, recovery, and strict-pilot records for the earlier raster path. |
| [experiments/four_hour_2026-09-13/README.md](experiments/four_hour_2026-09-13/README.md) | **Historical campaign** | Campaign overview; its ledger, handoff, next-waves note, and proposal are adjacent and preserve the campaign state at that date. |
| [experiments/pqn_followup_2026-09-12.md](experiments/pqn_followup_2026-09-12.md), [pqn_sampler_results_2026-09-06.md](pqn_sampler_results_2026-09-06.md), and [pqn_decision_phase_2026-09-12.md](pqn_decision_phase_2026-09-12.md) | **Historical / scoped PQN records** | Respectively: follow-up execution, sampler screen, and decision-phase contract. They do not establish a promoted raster policy. |
| [research_review_2026-09-24/README.md](research_review_2026-09-24/README.md) and [research_findings_2026-09-24.md](research_findings_2026-09-24.md) | **Historical synthesis** | A review and an index of the September evidence; use them to trace reasoning, then check `STATUS.md` before acting. |
| [plans/](plans/) | **Historical plans and packages** | The September repair plan and associated packages. Plans record proposed work and their contemporary boundaries; they are not active instructions. |

## Earlier engineering and provenance notes

| Document | Status | Use it for |
| --- | --- | --- |
| [codebase_quality_2026-09.md](codebase_quality_2026-09.md) | **Historical handoff** | The scoped September quality pass and its verification boundaries. |
| [rl_review_2026-09-06.md](rl_review_2026-09-06.md) and [state_and_system_review_2026-09-06.md](state_and_system_review_2026-09-06.md) | **Historical audits** | The September architecture, contract, and evaluation findings that informed later work. |
| [rl_assessment_2026-09-06.md](rl_assessment_2026-09-06.md) and [project_history_and_findings.md](project_history_and_findings.md) | **Historical evidence/provenance** | Corrected history, audit evidence, and limits of older measurements. Do not use their former “current status” sections as present state. |
| [h100_training_recipe.md](h100_training_recipe.md) | **Superseded as an active recipe** | Reproducibility and provenance for the old Apex/vector training line. The frozen vector line is no longer the main research plan. |

## Compute and RunPod pointers

The source-controlled pointers are [compute policy](research/compute_policy_2026-10-02.md),
[performance plan](research/perf_plan_2026-10-07.md), [x86 consistency check](research/runpod_x86_consistency_2026-10-03.md)
(historical: its all-identical sample is superseded by the 2026-10-07 float32-tie diagnosis),
and the RunPod governance status note above. They are supplemented by the operator notes in
`~/.claude/skills/runpod/`; those external notes are not versioned with this repository.
For the current choice of work and any spend boundary, use `STATUS.md` first.

## Reconciled discrepancies and source precedence

The newest decision record and `STATUS.md` describe current state. A branch launch note,
an artifact namespace, a campaign's initial registration, or an older external memory may
preserve an earlier state without changing the recorded outcome.

| Discrepancy | Current interpretation |
| --- | --- |
| Older release and redesign descriptions | Champion+v8 was replaced by **frp3-s12+v8 on 2026-10-06**. The July raster/PQN blueprint is historical; the October main line is **GridBatchSim + ego2s-b + distilled DQN**, with PQN deferred. See [STATUS.md](STATUS.md). |
| Old 5x RunPod rule and “always serverless” preference | The threshold was amended to **at least 4x wall-clock per step**. Waivers below 4x are granted by the user case by case and listed in [STATUS.md](STATUS.md#operating-rules-current). When RunPod is used, cheap pods are checked first; serverless is only a fallback because its real cost has run at about 6–10x the catalog rate. The Mac remains available on AC. See the external `~/.claude/projects/-Users-josenunez-Projects-ml-snake-dqn/memory/runpod-5x-rule.md` and `runpod-cheap-pods-first.md`. |
| Strict gates on RunPod | The [strict-on-RunPod amendment](research/governance_amendment_strict_on_runpod_2026-10-05.md) is **NOT RATIFIED and was superseded on 2026-10-08**. It was never used: the FRP-v3 strict gate ran on the Mac. Mac↔x86 bitwise identity fails at random on exact float32 ties. The amendment is hash-bound by template v3 `prepare`, so its "Pending" footer stays unedited; the [status note](research/governance_amendment_strict_on_runpod_2026-10-05_status_2026-10-08.md) governs. Future strict-on-pods use needs a new amendment with a tie-explaining identity check and a per-world single CPU model. Strict gates and serving qualification stay on the Mac. |
| FRP-v5-S versus FRP-v5-S2 | They are different studies. **v5-S** did not pass fidelity and did not train or enter Phase R; fresh-world **v5-S2** recorded `GO_S_RECIPE`, not a gate or promotion. |
| S2 date versus its `20261009` artifact namespace | The namespace is not the result date. The recorded S2 merge/result date is **2026-10-08**. The suffix is a namespace label; see the note in the [research index](research/INDEX.md). |
| LH-1 appears as `DRAFT` in an opening document | The amendment was **ratified 2026-10-05 (`ecddfab`)**; LH-1 recorded `CLEAR` for FRP-v3. The document is hash-bound by the FRP-v3 LH-1 screen intent, so its header stays unedited; see the [erratum](research/governance_amendment_long_horizon_screen_2026-10-04_erratum_2026-10-08.md). Do not relabel the result from an opening-state label. |
| Campaign `training_bet_*` entries still say `REGISTERED_NAMESPACES` | Those `status` fields are initial campaign metadata and are never rewritten. On 2026-10-08 a `final_outcome` field (status, date, summary and pointers) was **added** to `training_bet_frp_v4`/`_v5h`/`_v5s`/`_v5s2`, `governance_strict_on_runpod`, `sequential_strict_gate` and `sequential_strict_runner`. Read `final_outcome` for the result. |
| “CNN variants lost” in older prose | Retracted by git archaeology: the **June CNN code** was never trained and produced no checkpoints. Only the older GRU/DRQN result exists; this does not describe later raster/PQN or October CNN-student work. See [the blueprint correction](ml_redesign_blueprint_2026-07.md#corrected-historical-record). As of 2026-10-08 every repository occurrence on main is already a retraction or carries a correction (README, AGENTS.md, `ml_algorithm.md`, `h100_training_recipe.md`, `project_history_and_findings.md`). |

The external current-state source is
`~/.claude/projects/-Users-josenunez-Projects-ml-snake-dqn/memory/` (especially
`standing-decisions-2026-10-08.md`, `redesign-scope.md`, and the dated result notes).
Its entry point is `MEMORY.md`. The read-only artifact evidence is
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/HANDOFF_2026-10-05.md` and
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/campaign.json`;
the JSON is campaign metadata. Its existing fields must not be rewritten; additions such as `final_outcome` are allowed. The companion RunPod material is
`~/.claude/skills/runpod/{SKILL,NOTES,OPTIMIZATION,HARDWARE,CPU_PROVIDER_COMPARISON}.md`.
These external locations are not versioned in this repository; use the linked repository
records for reviewable, commit-pinned history.

## Status labels

- **Current** describes the maintained release, contract, or decision record in its stated
  scope. It does not turn an old study package into a new run authorization.
- **Historical** preserves a dated result, audit, plan, or handoff. Its numbers and verdicts
  remain attributable to the recorded protocol.
- **Superseded** means a later record replaced the document as an active plan or status source;
  its provenance remains useful.
- **Conditional** means a record has stated prerequisites or ratification conditions. Read the
  record before applying it.
