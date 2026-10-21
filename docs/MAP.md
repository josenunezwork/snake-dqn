# docs/ tree map

One line for every top-level file and folder in `docs/` (snapshot 2026-10-20). For per-document
status tables see [README.md](README.md); this file is only the whole-tree layout. Descriptions come
from each file's own header; "unclear" means the header did not say.

## Start here (new reader)

1. [STATUS.md](STATUS.md): released agent, standing decisions, campaign authority.
2. [research/final_run_20261019/HANDOFF_2026-10-20.md](research/final_run_20261019/HANDOFF_2026-10-20.md),
   then [CAMPAIGN_LOG.md](research/final_run_20261019/CAMPAIGN_LOG.md): the active T34 campaign.
3. [research/final_run_20261019/SUCCESS_CONTRACT.md](research/final_run_20261019/SUCCESS_CONTRACT.md) and
   [P7_STATISTICS.md](research/final_run_20261019/P7_STATISTICS.md): the ratified pass rules.
4. [GLOSSARY.md](GLOSSARY.md) (or [research/GLOSSARY.md](research/GLOSSARY.md)) and [TIMELINE.md](TIMELINE.md).
5. [architecture/](architecture/) and [ml_algorithm.md](ml_algorithm.md): how the system works.
6. [research/INDEX.md](research/INDEX.md): find any dated result or decision.

Files being added in parallel on 2026-10-20 (links may 404 until they land): [GLOSSARY.md](GLOSSARY.md),
[TIMELINE.md](TIMELINE.md), [architecture/](architecture/), [experiments/README.md](experiments/README.md),
and a rewrite of [research/INDEX.md](research/INDEX.md).

## Current campaign & governance

| Path | What it is |
| --- | --- |
| [STATUS.md](STATUS.md) | Current state (release updated 2026-10-17) and owner-ratified campaign authority. |
| [research/final_run_20261019/](research/final_run_20261019/) | T34 "ExIt-B8" campaign: handoff, campaign log, final/merged plans, design v2, success contract, P7 statistics, speed advice. |
| `research/governance_*.md` (in [research/](research/)) | Governance tiers (2026-09-26) and dated amendments; [README.md](README.md) lists which are current. |
| [research/ops/](research/ops/) | 2026-10-20 operational snapshots: artifacts inventory, worktree inventory, Colab units ledger. |

## Research results & indexes

| Path | What it is |
| --- | --- |
| [research/](research/) | ~135 dated records (T1..T16 studies, M3..M6 results, releases, death censuses, perf notes, JSON evidence). |
| [research/INDEX.md](research/INDEX.md) | Chronological locator for research and decision documents. |
| [research/GLOSSARY.md](research/GLOSSARY.md) | Evaluation vocabulary used in release, gate and redesign records. |
| [research_findings_2026-09-24.md](research_findings_2026-09-24.md) | September evidence index (superseded as a status source 2026-10-08). |
| [research_review_2026-09-24/](research_review_2026-09-24/) | Astra four-agent review of the learning program (evaluation, learning, operations, strategy). |
| [project_history_and_findings.md](project_history_and_findings.md) | Dated evidence/provenance record (superseded as status source 2026-10-08). |
| [rl_review_2026-09-06.md](rl_review_2026-09-06.md) | RL review, 2026-09-06. |
| [rl_assessment_2026-09-06.md](rl_assessment_2026-09-06.md) | RL assessment: keep Apex incumbent, raster/PQN plausible. |
| [state_and_system_review_2026-09-06.md](state_and_system_review_2026-09-06.md) | State/system audit: keep vector61 Apex frozen. |
| [pqn_sampler_results_2026-09-06.md](pqn_sampler_results_2026-09-06.md) | PQN complete-epoch sampler screen: no established improvement. |
| [pqn_decision_phase_2026-09-12.md](pqn_decision_phase_2026-09-12.md) | Opt-in Watch-aligned decision phase for raster PQN. |
| [codebase_quality_2026-09.md](codebase_quality_2026-09.md) | September codebase quality handoff. |

## Specs & advice

| Path | What it is |
| --- | --- |
| [research/specs/](research/specs/) | ~96 dated implementation specs (T34 teacher, ExIt engine, Colab fixes, etc.). |
| [research/advice/](research/advice/) | Dated advisor memos (Astra/Fable panels, 2026-10-15 onward). |
| [research/panel_2026-10-15/](research/panel_2026-10-15/) | Two-round mechanics/RL panel and its decision memo. |
| [simd_env_spec.md](simd_env_spec.md) | Batched simulator (`BatchSim`) dynamics and parity contract. |

## Experiments (historical)

| Path | What it is |
| --- | --- |
| [experiments/](experiments/) | ~204 study folders dated 2026-09-06..2026-10-03 (`solo_*`, `controlled_*`, `apex_*`, `task_aligned_*`, `pqn_*`, ...). Catalog: [experiments/README.md](experiments/README.md). |

## Architecture & ops

| Path | What it is |
| --- | --- |
| [architecture/](architecture/) | Architecture write-ups (being added 2026-10-20). |
| [architecture.svg](architecture.svg) | System architecture diagram. |
| [ml_algorithm.md](ml_algorithm.md) | Apex DQN algorithm design (historical vector policy; limited current scope). |
| [colab-cli.md](colab-cli.md) | Operator guide for the Mac-side Colab CLI supervisor. |
| [colab-cli-pricing-notes-2026-10-18.md](colab-cli-pricing-notes-2026-10-18.md) | First live Colab CPU rate-calibration evidence. |
| [research/colab_cli_ref/](research/colab_cli_ref/) | Vendored reference copy of the Colab CLI's own docs and skills. |
| [research/colab_cli_receipts/](research/colab_cli_receipts/) | Colab CLI acceptance receipt (`dab00823`). |
| [runpod_selfdelete_key.md](runpod_selfdelete_key.md) | RunPod runtime self-delete credential notes (updated 2026-10-17). |
| [testing_openmp_macos.md](testing_openmp_macos.md) | macOS pytest OpenMP routing. |
| [h100_training_recipe.md](h100_training_recipe.md) | H100 recipe for the Apex vector line (superseded 2026-10-08; provenance). |
| [claude-skills/](claude-skills/) | Repo copies of the `colab` and `colab-jobs` Claude skills. |
| [codex-skills/](codex-skills/) | Backup mirror of `~/.codex/skills/snake-dqn-*` (18 Codex skills). |

## Plans

| Path | What it is |
| --- | --- |
| [plans/rl_repair_2026-09-06/](plans/rl_repair_2026-09-06/) | 24-package RL repair plan (README says "planned, not started"), runbook, dispatch, validator. |
| [plans/rl_repair_2026-09-08-followup.md](plans/rl_repair_2026-09-08-followup.md) | X0 follow-up waves; not executed. |
| [ml_redesign_blueprint_2026-07.md](ml_redesign_blueprint_2026-07.md) | July redesign blueprint (superseded as active plan 2026-10-08). |

## References

| Path | What it is |
| --- | --- |
| [README.md](README.md) | Documentation map with per-document status (current vs historical). |
| [MAP.md](MAP.md) | This file: whole-tree layout. |
