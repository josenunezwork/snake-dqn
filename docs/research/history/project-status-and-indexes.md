# Project Status, Top-Level Indexes, and Compendium Provenance (`project-status-and-indexes`)

## Question and Goal

How are current project truth, campaign status, served checkpoints, chronological timelines, and documentation navigation organized across the repository's top-level index files (`AGENTS.md`, `README.md`, `docs/STATUS.md`, `docs/INDEX.md`, `docs/MAP.md`, `docs/TIMELINE.md`, `docs/notes/README.md`), and how was the October 25, 2026 `docs/notes/` compendium constructed from the repository's primary evidence?

## Timeline of Versions and Steps

| Date | Milestone / Document Group | What Was Recorded / Built | Status |
|---|---|---|---|
| 2026-06 – 2026-09 | [`docs/project_history_and_findings.md`](../../project_history_and_findings.md), [`docs/research_findings_2026-09-24.md`](../../research_findings_2026-09-24.md), [`docs/experiments/README.md`](../../experiments/README.md), [`docs/experiments/OTHER_SERIES.md`](../../experiments/OTHER_SERIES.md) | Consolidated early project history (`vector58` → `champion_a5` `vector61`), September 2026 `BatchSim` / `raster31v2` / PQN & solo-imitation findings, and the `docs/experiments/` index. | **HISTORICAL COMPENDIUM** |
| 2026-10-06 – 2026-10-17 | [`docs/STATUS.md`](../../STATUS.md), [`docs/INDEX.md`](../../INDEX.md) | Recorded the promotions of `frp3-s12 + v8` (2026-10-06) and bare `b8-s3` (`Ego2sNet`, `1,641,671` params, SHA-256 `3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`, promoted 2026-10-17 after T7 Phase R `+45.22` mass vs `frp3-s12+v8` and T12/T15 serving qualification). | **ACTIVE TRUTH** |
| 2026-10-25 | `docs/notes/` compendium pass (`notes-cleanup-20261025`, [`docs/notes/README.md`](../../notes/README.md), [`docs/MAP.md`](../../MAP.md), [`docs/TIMELINE.md`](../../TIMELINE.md)) | Two-stage documentation synthesis: 10 Opus fact-sheet agents extracted verified citations into `docs/notes/_facts/`, followed by 14 Astra writer agents producing `docs/notes/*.md`, `docs/MAP.md`, and `docs/TIMELINE.md`. | **ACTIVE COMPENDIUM** (transient `_scratch/` & `STYLE.md` briefs removed) |
| 2026-10-30 – 2026-11-01 | [`AGENTS.md`](../../../AGENTS.md), [`docs/STATUS.md`](../../STATUS.md), [`docs/INDEX.md`](../../INDEX.md) | Ratified the Fast-Process Rule (2026-10-30) and promoted the 132-session confirmed `exact-stop-l2-prime` Watch replay bank (`watch-replay-bank-live-132`, `+73.17` paired mass vs bare `b8-s3`, `p = 0.00496`) on 2026-11-01 while keeping live bare `b8-s3` for Play and Watch fallback. | **CURRENT SERVED STATE** |

## Final Verdict and Key Numbers

Current project truth is maintained in [`docs/STATUS.md`](../../STATUS.md), [`docs/INDEX.md`](../../INDEX.md), and [`AGENTS.md`](../../../AGENTS.md), with topic-by-topic synthesis in [`docs/notes/overview-and-serving.md`](../../notes/overview-and-serving.md):

- **Current served Watch configuration (since 2026-11-01):** Serves the 132-session confirmed `exact-stop-l2-prime` (`K4` + `L2'` exact-stop assisted search over `b8-s3`) replay bank `watch-replay-bank-live-132` (`SNAKE_WATCH_REPLAY_BANK=/Users/josenunez/Projects/ml/snake-dqn-artifacts/watch6/packs/watch-replay-bank-live-132`, `SNAKE_WATCH_REPLAY_CONTROLLER=exact-stop-l2-prime`, `132` full 120-s / 2,880-frame 6-snake sessions, `0` hero deaths, median length `837.5`, max length `2,672`, confirmed on `N = 320` fresh paired H5000 worlds at **`+73.17` mass vs bare `b8-s3`**, one-sided `p = 0.00496`), with live bare `b8-s3` fallback ([`docs/STATUS.md`](../../STATUS.md), [`docs/INDEX.md`](../../INDEX.md)).
- **Current served Play configuration (since 2026-10-17):** Serves live bare **`b8-s3`** (`saved_snakes/b8_s3_20261017.pth`, `Ego2sNet`, `obs_spec: ego2s-bv8`, `1,641,671` parameters, SHA-256 `3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`) with `SNAKE_SERVE_VETO_VARIANT=none` ([`docs/STATUS.md`](../../STATUS.md), [`docs/notes/overview-and-serving.md`](../../notes/overview-and-serving.md)).
- **Historical served checkpoints and rollback pins:**
  - `frp3-s12 + v8` (`saved_snakes/frp3_s12.pth`, SHA-256 `eec144bf2ee8a6ca7effcd4ffd81eb4bd89dd687df28dd2a29ec819aa19db866`, `ApexNetwork` 61-D, released 2026-10-06, rollback via `SNAKE_SERVE_CHECKPOINT=frp3_s12` + `SNAKE_SERVE_VETO_VARIANT=v8`) ([`docs/notes/overview-and-serving.md`](../../notes/overview-and-serving.md)).
  - `champion_a5 + v8` (`saved_snakes/champion_a5_freespace_20260621.pth`, SHA-256 `43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93`, `ApexNetwork` 61-D, rollback via `SNAKE_SERVE_CHECKPOINT=champion`) ([`docs/notes/overview-and-serving.md`](../../notes/overview-and-serving.md)).

### Provenance of the 2026-10-25 `docs/notes/` Compendium Pass

The four deleted files in this lineage (`docs/notes/STYLE.md`, `docs/notes/_facts/overview-and-serving.md`, `docs/notes/_scratch/ASTRA_BRIEF.md`, `docs/notes/_scratch/COMMON_BRIEF.md`) were the transient worker instructions and intermediate fact sheet from the October 25, 2026 documentation synthesis (`notes-cleanup-20261025`). Every fact and table in `docs/notes/_facts/overview-and-serving.md` was incorporated into [`docs/notes/overview-and-serving.md`](../../notes/overview-and-serving.md), while `COMMON_BRIEF.md`, `ASTRA_BRIEF.md`, and `STYLE.md` specified the formatting rules for the completed [`docs/notes/`](../../notes/README.md) chapters.

## What Was Learned

- **Single source of truth for serving and campaign state:** Centralizing active campaign rules in [`AGENTS.md`](../../../AGENTS.md) and current serving/promotion state in [`docs/STATUS.md`](../../STATUS.md) + [`docs/INDEX.md`](../../INDEX.md) prevents stale dated notes from being mistaken for active instructions.
- **Fact-sheet + writer pipeline scales documentation without losing primary citations:** Separating raw fact extraction (`_facts/`) from narrative synthesis (`docs/notes/*.md`) preserved exact SHA-256 pins, commit hashes, and artifact paths across 11 topic chapters ([`docs/notes/README.md`](../../notes/README.md)).

## Current Status

**Active index and status lineage.** The 21 `KEEP` files below serve as the live entry points and navigation indexes for the repository.

## Keep Files in This Lineage

- [`AGENTS.md`](../../../AGENTS.md) & [`CLAUDE.md`](../../../CLAUDE.md) — Standing agent operating rules, campaign authority, Colab/Mac policies, and essential commands.
- [`README.md`](../../../README.md) — Top-level repository README and quickstart guide.
- [`docs/STATUS.md`](../../STATUS.md) — Current-truth status document (released agents, active campaign, and recent results).
- [`docs/INDEX.md`](../../INDEX.md) — Master index of documentation, served checkpoints, and chronological experiment table.
- [`docs/MAP.md`](../../MAP.md) — Repository directory map and guide to finding code, configs, tests, and research packages.
- [`docs/TIMELINE.md`](../../TIMELINE.md) — End-to-end chronological timeline of the project from June to November 2026.
- [`docs/README.md`](../../README.md) — Documentation directory landing page.
- [`docs/project_history_and_findings.md`](../../project_history_and_findings.md) — June–September 2026 project history and findings.
- [`docs/research_findings_2026-09-24.md`](../../research_findings_2026-09-24.md) — September 24, 2026 research findings summary.
- [`docs/experiments/README.md`](../../experiments/README.md) & [`docs/experiments/OTHER_SERIES.md`](../../experiments/OTHER_SERIES.md) — Index and catalog of the September 2026 `docs/experiments/` studies.
- [`docs/notes/README.md`](../../notes/README.md) — Index of the 11-chapter `docs/notes/` research compendium.
- [`docs/notes/GLOSSARY.md`](../../notes/GLOSSARY.md) — Compendium glossary of architectures, metrics, controllers, and governance terms.
- [`docs/notes/overview-and-serving.md`](../../notes/overview-and-serving.md) — Overview of the three model stacks, served checkpoints, and Watch/Play architecture.
- [`docs/notes/lessons-and-incidents.md`](../../notes/lessons-and-incidents.md) — Cross-campaign catalog of engineering, statistical, and operational lessons.
- [`docs/research/advice/INDEX.md`](../advice/INDEX.md) — Index of advisory reviews and judge rulings under `docs/research/advice/`.
- [`docs/research/specs/INDEX.md`](../specs/INDEX.md) — Index of design and implementation specifications under `docs/research/specs/`.
- [`research/INDEX.md`](../../../research/INDEX.md) — Index of executable research packages under `research/`.
- [`tests/README.md`](../../../tests/README.md) — Test suite overview and execution guide.
- [`web/README.md`](../../../web/README.md) — Web application (`web/backend/` + `web/frontend/`) overview and setup guide.
