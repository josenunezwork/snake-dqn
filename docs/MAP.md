# docs/ tree map

One line for every top-level file and folder in `docs/` (tables last fully refreshed 2026-11-01; start-here list 2026-11-11). Reading order for a new reader is on [README.md](README.md).

## Start here (new reader)

1. [STATUS.md](STATUS.md): one-page current state and binding rules. [FINDINGS.md](FINDINGS.md): what was learned.
2. [INDEX.md](INDEX.md): what the project is, where source lives, run commands.
3. [../research/INDEX.md](../research/INDEX.md): live research code first, then every closed study package.
4. [PRODUCT_CODE_GUIDE.md](PRODUCT_CODE_GUIDE.md), [architecture/WEB_MAP.md](architecture/WEB_MAP.md), [architecture/TESTS_MAP.md](architecture/TESTS_MAP.md): how the served product works.
5. [research/history/INDEX.md](research/history/INDEX.md), [STATUS_ARCHIVE.md](STATUS_ARCHIVE.md), [TIMELINE.md](TIMELINE.md): closed lines and history (T34 campaign log and pass rules: [research/final_run_20261019/](research/final_run_20261019/)).
6. [notes/README.md](notes/README.md), [research/GLOSSARY.md](research/GLOSSARY.md), [notes/GLOSSARY.md](notes/GLOSSARY.md): topic notes and vocabulary.

## Current campaign & governance

| Path | What it is |
| --- | --- |
| [INDEX.md](INDEX.md) | Top-level project map: what the repo is, served Watch/Play today, active T34 studies, experiment locations, and run commands. |
| [STATUS.md](STATUS.md) | One-page current state (2026-11-11; older detail, release evidence and rollback targets in [STATUS_ARCHIVE.md](STATUS_ARCHIVE.md)) and standing owner rules. |
| [research/ROADMAP.md](research/ROADMAP.md) | Round-27 synthesized current-round and later-round research roadmap (advice only, not ratified decisions). |
| [research/final_run_20261019/](research/final_run_20261019/) | T34 "ExIt-B8" campaign: handoff, campaign log, final/merged plans, design v2, success contract, P7 statistics, speed advice. |
| `research/governance_*.md` & [research/governance_a100_sandbox_20261026/](research/governance_a100_sandbox_20261026/) | Governance tiers (2026-09-26), dated amendments, and A100 sandbox governance; [README.md](README.md) lists which are current. |
| [research/ops/](research/ops/) | 2026-10-20 operational snapshots: artifacts inventory, worktree inventory, Colab units ledger. |

## Research results & indexes

| Path | What it is |
| --- | --- |
| [research/](research/) | Dated records (T1..T15 studies, M3..M6 results, releases, death censuses, Round 30 & Watch6 records, JSON evidence). |
| [research/INDEX.md](research/INDEX.md) | Chronological locator for kept research and decision documents in `docs/research/`. |
| [research/history/INDEX.md](research/history/INDEX.md) | **54 consolidated lineage records** summarizing all completed/closed research series (replacing 761 historical `.md` files). |
| [research/results/](research/results/) | Sidecars and decision summaries (e.g. `P1_Y3_offline_critic_DECISION_20261022.json`, `L2prime_mac_analysis_20261022.json`). |
| [research/round30_postf4_20261025/](research/round30_postf4_20261025/) | Post-F4 Round 30 intent, handoff, and ruling records. |
| [notes/](notes/) | Fact-checked, line-cited topic notes and [notes/GLOSSARY.md](notes/GLOSSARY.md) (snapshot 2026-10-25). |
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
| [research/specs/INDEX.md](research/specs/INDEX.md) | Preserved implementation specs and amendments (with lineage mappings for consolidated specs). |
| [research/advice/INDEX.md](research/advice/INDEX.md) | Preserved advisor syntheses and judge rulings (with lineage mappings for consolidated memos). |
| [research/panel_2026-10-15/](research/panel_2026-10-15/) | Two-round mechanics/RL panel decision memo (`PANEL_DECISION_MEMO.md`). |
| [simd_env_spec.md](simd_env_spec.md) | Batched simulator (`BatchSim`) dynamics and parity contract. |

## Experiments (historical)

| Path | What it is |
| --- | --- |
| [experiments/](experiments/) | Kept Sept 2026 study receipts and series indexes (`SOLO_FOOD_SERIES.md`, `OTHER_SERIES.md`, `TASK_ALIGNED_AND_APEX_SERIES.md`). Catalog: [experiments/README.md](experiments/README.md). |

## Architecture & ops

| Path | What it is |
| --- | --- |
| [architecture/](architecture/) | Architecture maps: [WEB_MAP.md](architecture/WEB_MAP.md) (`web/`) and [TESTS_MAP.md](architecture/TESTS_MAP.md) (`tests/`). |
| [architecture.svg](architecture.svg) | System architecture diagram. |
| [ml_algorithm.md](ml_algorithm.md) | Apex DQN algorithm design (historical vector policy; limited current scope). |
| [colab-cli.md](colab-cli.md) | Operator guide for the Mac-side Colab CLI supervisor (`research/colab_cli/`), consolidated with [`research/history/colab-ops.md`](research/history/colab-and-process.md#colab-ops). |
| [colab-cli-pricing-notes-2026-10-18.md](colab-cli-pricing-notes-2026-10-18.md) | First live Colab CPU rate-calibration evidence. |
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
| [README.md](README.md) | Front door: reading order, reconciled discrepancies, status labels. |
| [STATUS_ARCHIVE.md](STATUS_ARCHIVE.md) | The status page as it stood on 2026-11-10: release evidence, rollback targets, T34, October history. |
| [PRODUCT_CODE_GUIDE.md](PRODUCT_CODE_GUIDE.md) | Product code overview, architecture contracts, commands and key files (moved out of `AGENTS.md`). |
| [FINDINGS.md](FINDINGS.md) | Plain-language map of what the project has learned. |
| [MAP.md](MAP.md) | This file: whole-tree layout. |
