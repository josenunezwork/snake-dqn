# Project Status, Top-Level Indexes, and Compendium Provenance (`project-status-and-indexes`)

## Question and Goal

How are current project truth, campaign status, served checkpoints, chronological timelines, and documentation navigation organized across the repository's top-level index files (`AGENTS.md`, `README.md`, `docs/STATUS.md`, `docs/INDEX.md`, `docs/MAP.md`, `docs/TIMELINE.md`, `docs/notes/README.md`), and how was the October 25, 2026 `docs/notes/` compendium constructed from the repository's primary evidence?

## Timeline of Versions and Steps

| Date | Milestone / Document Group | What Was Recorded / Built | Status |
|---|---|---|---|
| 2026-06 – 2026-09 | [`docs/project_history_and_findings.md`](../../project_history_and_findings.md), [`docs/research_findings_2026-09-24.md`](../../research_findings_2026-09-24.md), [`docs/experiments/README.md`](../../experiments/README.md), [`docs/experiments/OTHER_SERIES.md`](../../experiments/OTHER_SERIES.md) | Consolidated early project history (Apex `vector61` incumbent; Raster/PQN `raster31v2` / `raster31v3` an unpromoted candidate path; both documents bannered superseded as current-status sources on 2026-10-08), the research-findings index through 24 September 2026 (Apex scaling/horizon studies, PQN and solo-native experiments), and the `docs/experiments/` index. | **HISTORICAL COMPENDIUM** |
| 2026-10-06 – 2026-10-17 | [`docs/STATUS.md`](../../STATUS.md), [`docs/INDEX.md`](../../INDEX.md) | Recorded the promotions of `frp3-s12 + v8` (2026-10-06) and bare `b8-s3` (`Ego2sNet`, ~1.64M params, SHA-256 `3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`, released 2026-10-17 without a veto after T7 fresh-bank Phase R `STRICT_PASS` + Tier 2 PASS, all eight T12 serving checks, and T15 mixed-roster qualification). | **ACTIVE TRUTH** |
| 2026-10-25 | `docs/notes/` compendium pass (`notes-cleanup-20261025`, [`docs/notes/README.md`](../../notes/README.md), [`docs/MAP.md`](../../MAP.md), [`docs/TIMELINE.md`](../../TIMELINE.md)) | Documentation synthesis on branch `notes-cleanup-20261025`: 10 Opus fact-sheet agents (one per topic) wrote cited fact sheets into `docs/notes/_facts/`, reviewed by a Fable coordinator, then 14 Astra writers produced one note each, followed by a coordinator consistency pass ([`docs/notes/README.md`](../../notes/README.md) Provenance). | **ACTIVE COMPENDIUM** (`_scratch/` briefs, `STYLE.md` and `_facts/overview-and-serving.md` removed in consolidation; see git c4c4ab41) |
| 2026-10-30 – 2026-11-01 | [`AGENTS.md`](../../../AGENTS.md), [`docs/STATUS.md`](../../STATUS.md), [`docs/INDEX.md`](../../INDEX.md) | Owner's standing Fast-process rule (2026-10-30); Watch6 V3 efficacy `GO_EFFICACY_ONLY` (2026-11-01 14:40 PT) and release of the 132-session confirmed `exact-stop-l2-prime` Watch replay bank (`watch-replay-bank-live-132`, 2026-11-01 ~16:00 PT, `f5912f44`) while keeping live bare `b8-s3` for Play and as Watch fallback. | **CURRENT SERVED STATE** |

## Final Verdict and Key Numbers

Current project truth is maintained in [`docs/STATUS.md`](../../STATUS.md), [`docs/INDEX.md`](../../INDEX.md), and [`AGENTS.md`](../../../AGENTS.md), with topic-by-topic synthesis in [`docs/notes/overview-and-serving.md`](../../notes/overview-and-serving.md):

- **Current served Watch configuration (since 2026-11-01 ~16:00 PT):** Serves the 132-session confirmed `exact-stop-l2-prime` replay bank `snake-dqn-artifacts/watch-replay-bank-live-132` (banks `01–08`, `10–12`; `b8-s3` hero + `exact-stop-l2-prime` at `L >= 600`, interval 5, `H = 300`, vs `5× frp3-s12 + v8`; 6,000-frame sessions labelled "Recorded assisted hero"), launched via `.claude/launch.json` (`snake-web`) with `SNAKE_WATCH_SOURCE=replay`, `SNAKE_WATCH_REPLAY_DIR=.../watch-replay-bank-live-132`, and `SNAKE_WATCH_REPLAY_CONTROLLER=exact-stop-l2-prime`, with live bare `b8-s3` as automatic fallback. Efficacy basis: Watch6 V3 `GO_EFFICACY_ONLY` over `96` fresh paired `H = 14,400` sessions — deaths/1k frames `−0.125` (paired-t CI95 `−0.153 .. −0.097`, bar `CI95_upper < 0`) and pooled mass `+63.7%` vs bare (Fieller CI95 `+49.8% .. +78.5%`, bar `lower >= −5%`) ([`docs/STATUS.md`](../../STATUS.md), [`docs/INDEX.md`](../../INDEX.md)).
- **Current served Play configuration (since 2026-10-17):** Serves live bare **`b8-s3`** (gitignored copy `saved_snakes/b8_s3_t7_20261016.pth`, `Ego2sNet`, `obs_spec: ego2s-bv8`, ~1.64M parameters, SHA-256 `3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`) without a veto; `1` human vs `N` (default `5`, adjustable `1–11`) live bare `b8-s3` AI opponents ([`docs/STATUS.md`](../../STATUS.md), [`docs/notes/overview-and-serving.md`](../../notes/overview-and-serving.md)).
- **Historical served checkpoints and rollback pins:**
  - `frp3-s12 + v8` (FRP-v3 M3 seed 12 @60000 `vector61` checkpoint, gitignored copy `saved_snakes/frp3_m3_s12_u60000_20261005.pth`, SHA-256 `eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723`, released 2026-10-06, now the mixed-Watch opponent; rollback via `SNAKE_WATCH_SOURCE=live SNAKE_SERVE_CHECKPOINT=frp3-s12`) ([`docs/notes/overview-and-serving.md`](../../notes/overview-and-serving.md), [`docs/INDEX.md`](../../INDEX.md); SHA and filename from `docs/notes/_facts/overview-and-serving.md` (removed in consolidation; see git c4c4ab41)).
  - `champion_a5 + v8` (`champion_a5_freespace_20260621.pth`, SHA-256 `43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93` per [`docs/research/b8_s3_release_2026-10-17.md`](../b8_s3_release_2026-10-17.md), `ApexNetwork` 61 → 512 → 256 dueling, last-resort fallback; rollback via `SNAKE_WATCH_SOURCE=live SNAKE_SERVE_CHECKPOINT=champion`) ([`docs/notes/overview-and-serving.md`](../../notes/overview-and-serving.md), [`docs/INDEX.md`](../../INDEX.md)).

### Provenance of the 2026-10-25 `docs/notes/` Compendium Pass

The four deleted files in this lineage (`docs/notes/STYLE.md`, `docs/notes/_facts/overview-and-serving.md`, `docs/notes/_scratch/ASTRA_BRIEF.md`, `docs/notes/_scratch/COMMON_BRIEF.md`; all removed in consolidation; see git c4c4ab41) were the worker instructions and an intermediate fact sheet from the 2026-10-25 documentation synthesis (`notes-cleanup-20261025`). `COMMON_BRIEF.md` briefed the fact-sheet agents, `ASTRA_BRIEF.md` briefed the Astra note writers (one topic each, fact sheet as the only source of claims), and `STYLE.md` was the style guide every `docs/notes/` file followed. `docs/notes/_facts/overview-and-serving.md` was the raw, cited input from which [`docs/notes/overview-and-serving.md`](../../notes/overview-and-serving.md) was written.

## What Was Learned

- **Single source of truth for serving and campaign state:** Centralizing active campaign rules in [`AGENTS.md`](../../../AGENTS.md) and current serving/promotion state in [`docs/STATUS.md`](../../STATUS.md) + [`docs/INDEX.md`](../../INDEX.md) prevents stale dated notes from being mistaken for active instructions.
- **Fact-sheet + writer pipeline scales documentation without losing primary citations:** Separating raw fact extraction (`_facts/`) from narrative synthesis (`docs/notes/*.md`) preserved exact SHA-256 pins, commit hashes, and artifact paths across 13 topic notes plus a glossary ([`docs/notes/README.md`](../../notes/README.md)).

## Current Status

**Active index and status lineage.** The 21 `KEEP` files below serve as the live entry points and navigation indexes for the repository.

## Keep Files in This Lineage

- [`AGENTS.md`](../../../AGENTS.md) & [`CLAUDE.md`](../../../CLAUDE.md) — Standing agent operating rules, campaign authority, Colab/Mac policies, and essential commands.
- [`README.md`](../../../README.md) — Top-level repository README and quickstart guide.
- [`docs/STATUS.md`](../../STATUS.md) — Current-truth status document (released agents, active campaign, and recent results).
- [`docs/INDEX.md`](../../INDEX.md) — Master index of documentation, served checkpoints, and chronological experiment table.
- [`docs/MAP.md`](../../MAP.md) — Repository directory map and guide to finding code, configs, tests, and research packages.
- [`docs/TIMELINE.md`](../../TIMELINE.md) — End-to-end chronological timeline of the project from 2026-06-16 (first commit) to 2026-11-01.
- [`docs/README.md`](../../README.md) — Documentation directory landing page.
- [`docs/project_history_and_findings.md`](../../project_history_and_findings.md) — June–September 2026 project history and findings.
- [`docs/research_findings_2026-09-24.md`](../../research_findings_2026-09-24.md) — September 24, 2026 research findings summary.
- [`docs/experiments/README.md`](../../experiments/README.md) & [`docs/experiments/OTHER_SERIES.md`](../../experiments/OTHER_SERIES.md) — Index and catalog of the September 2026 `docs/experiments/` studies.
- [`docs/notes/README.md`](../../notes/README.md) — Index of the `docs/notes/` research compendium (13 notes plus glossary).
- [`docs/notes/GLOSSARY.md`](../../notes/GLOSSARY.md) — Compendium glossary of the models, studies, labels, metrics, and roles across the fact sheets.
- [`docs/notes/overview-and-serving.md`](../../notes/overview-and-serving.md) — Overview of the three model stacks, served checkpoints, and Watch/Play architecture.
- [`docs/notes/lessons-and-incidents.md`](../../notes/lessons-and-incidents.md) — Incidents, errata and the resulting recorded process rules.
- [`docs/research/advice/INDEX.md`](../advice/INDEX.md) — Index of advisory reviews and judge rulings under `docs/research/advice/`.
- [`docs/research/specs/INDEX.md`](../specs/INDEX.md) — Index of design and implementation specifications under `docs/research/specs/`.
- [`research/INDEX.md`](../../../research/INDEX.md) — Index of executable research packages under `research/`.
- [`tests/README.md`](../../../tests/README.md) — Test suite overview and execution guide.
- [`web/README.md`](../../../web/README.md) — Web application (`web/backend/` + `web/frontend/`) overview and setup guide.
