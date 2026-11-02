# Codex and Claude Operational & Research Skills (`agent-skills`)

## Question / Goal

Maintain a version-controlled repository mirror (`docs/codex-skills/`) of the procedural `snake-dqn-*` skills (`~/.codex/skills/snake-dqn-*`) that guide coding and research agents (Codex/Sol/Astra and Claude/Argon) across codebase architecture, game mechanics, bit-exact simulator parity, Mac resource guardrails, evaluation gates, pre-registration, compute economics, and web-serving qualification.

## Timeline of Versions / Steps

| Date | Step / Skill | Scope & What Was Run / Documented | Verdict / Status |
| :--- | :--- | :--- | :--- |
| 2026-10-18 | `docs/codex-skills/README.md` | Backup index and `cp -R` sync command for the 18 `snake-dqn-*` Codex skill directories (`docs/codex-skills/README.md`) | Active index (`KEEP`) |
| 2026-10-18 | Active codebase & mechanics skills (`codebase`, `game-mechanics`, `parity-determinism`, `git-workflow`) | Codifies the three model stacks (`ego2s-bv8` B8, `vector61` Apex, `raster31v2` PQN), 11-step frame update order, `v2` 1.15x head-on rule, cell-exact `GameState`/`BatchSim`/`GridBatchSim` parity, and git/SHA-256 immutability rules | Active (`KEEP`) |
| 2026-10-18 | Active Mac & serving skills (`mac-ops`, `serving-release`) | Codifies 64 GiB Mac RAM guards (`9.6 GiB` free reserve, `16 GiB` RSS cap, `24 GiB` MPS cap), single-heavy-process rule, `caffeinate -dimsu` usage, and 8-check 24 fps web-serving qualification | Active (`KEEP`) |
| 2026-10-18 | Research & advisor skills (`advisor-brief`, `evaluation-gates`, `experiment-runbook`, `performance`, `preregistration`, `research-ledger`) | Codifies evidence-first advisor briefs, Lan-DeMets OBF sequential boundaries, Survival Band v2 (`-0.015` pooled NI, `-0.050` per-mix floor), 64-bit world-seed namespaces, and `docs/research/` vs `snake-dqn-artifacts/` layout | Retained reference (`KEEP`; exploration ceremony superseded by 2026-10-30 Fast-process rule in `AGENTS.md`) |
| 2026-10-18 | Historical / frozen-line skills (`apex-training`, `m3c-trainer`, `compute-economics`, `queue-jobs`, `runpod`) | Documents frozen `vector61` FRP line (67-D `FRP-v5-S2` default if reopened), `train_m3c.py` B8 training (`+49.0` / `+85.1` mass vs `frp3-s12+v8`), Oct 18 Q0/Q1 `420 u` Colab budget envelope, legacy Drive queue (`research/colab_queue/`), and RunPod 4x rule | Retained historical mirror (`KEEP`; superseded for new T34 work by `AGENTS.md` and `docs/colab-cli.md`) |

## Final Verdict and Key Numbers

- **Skill inventory:** All **18** files in `docs/codex-skills/` (`169,999` bytes total; excluding `docs/codex-skills/snake-dqn-colab-jobs/SKILL.md` tracked under `colab-ops`) are retained as the repository backup mirror of `~/.codex/skills/snake-dqn-*` (`docs/codex-skills/README.md`).
- **Mac operational guardrails:** On the 64 GiB M5 Pro Mac, local jobs must preserve a **15% (`9.6 GiB`)** free RAM reserve, cap process RSS at **`16 GiB`** and MPS allocations at **`24 GiB`**, and cap native math threads (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`) (`docs/codex-skills/snake-dqn-mac-ops/SKILL.md`, `docs/codex-skills/snake-dqn-experiment-runbook/SKILL.md`).
- **Statistical gate defaults:** Sequential strict gates use Lan-DeMets O'Brien-Fleming alpha-spending boundaries at information fractions $t = (0.50, 0.75, 1.00)$ with Holm-Bonferroni correction across the 3 opponent mixes (`scripted`, `frozen`, `mixed`) and Survival Band v2 (`-0.015` pooled non-inferiority margin, `-0.050` per-mix floor) (`docs/codex-skills/snake-dqn-evaluation-gates/SKILL.md`).
- **Throughput baselines:** Numba `GridBatchSim` reaches **`~21,000–28,000`** hero steps/s per process with scripted opponents, and `torch.compile` + `channels_last` + `bf16` provides a **`1.71x`** learner speedup on Colab L4 (`docs/codex-skills/snake-dqn-performance/SKILL.md`).
- **Superseded campaign parameters:** The October 18 Q0/Q1 Colab unit budget (**`420 u`** total / **`90 u`** Q0 / **`60 u`** reserve in `docs/codex-skills/snake-dqn-compute-economics/SKILL.md`) and legacy notebook queue (`research/colab_queue/` in `docs/codex-skills/snake-dqn-queue-jobs/SKILL.md`) were superseded by the qualified CLI supervisor (`research/colab_cli/QUALIFICATION.md`) and the owner's 2026-10-24 standing no-budget rule in `AGENTS.md`.

## What Was Learned

1. Procedural skills that encode durable codebase invariants—11-step `GameState` mechanics, `GridBatchSim` parity, Mac memory/thread caps, SHA-256 immutability of registered artifacts, and 24 fps web-serving checks—prevent recurring operational regressions across agent sessions.
2. Skills that hard-code campaign-specific unit budgets (`420 u`), retired queues (`research/colab_queue/`), or pre-Fast-Process exploration ceremony age quickly; authoritative campaign rules belong in `AGENTS.md` and `docs/STATUS.md`.

## Current Status

**Active (kept in place).** All 18 files in this lineage are kept under `docs/codex-skills/` as the versioned skill mirror.

## Kept Files in This Lineage

- [`docs/codex-skills/README.md`](../../codex-skills/README.md)
- [`docs/codex-skills/snake-dqn-advisor-brief/SKILL.md`](../../codex-skills/snake-dqn-advisor-brief/SKILL.md)
- [`docs/codex-skills/snake-dqn-apex-training/SKILL.md`](../../codex-skills/snake-dqn-apex-training/SKILL.md)
- [`docs/codex-skills/snake-dqn-codebase/SKILL.md`](../../codex-skills/snake-dqn-codebase/SKILL.md)
- [`docs/codex-skills/snake-dqn-compute-economics/SKILL.md`](../../codex-skills/snake-dqn-compute-economics/SKILL.md)
- [`docs/codex-skills/snake-dqn-evaluation-gates/SKILL.md`](../../codex-skills/snake-dqn-evaluation-gates/SKILL.md)
- [`docs/codex-skills/snake-dqn-experiment-runbook/SKILL.md`](../../codex-skills/snake-dqn-experiment-runbook/SKILL.md)
- [`docs/codex-skills/snake-dqn-game-mechanics/SKILL.md`](../../codex-skills/snake-dqn-game-mechanics/SKILL.md)
- [`docs/codex-skills/snake-dqn-git-workflow/SKILL.md`](../../codex-skills/snake-dqn-git-workflow/SKILL.md)
- [`docs/codex-skills/snake-dqn-m3c-trainer/SKILL.md`](../../codex-skills/snake-dqn-m3c-trainer/SKILL.md)
- [`docs/codex-skills/snake-dqn-mac-ops/SKILL.md`](../../codex-skills/snake-dqn-mac-ops/SKILL.md)
- [`docs/codex-skills/snake-dqn-parity-determinism/SKILL.md`](../../codex-skills/snake-dqn-parity-determinism/SKILL.md)
- [`docs/codex-skills/snake-dqn-performance/SKILL.md`](../../codex-skills/snake-dqn-performance/SKILL.md)
- [`docs/codex-skills/snake-dqn-preregistration/SKILL.md`](../../codex-skills/snake-dqn-preregistration/SKILL.md)
- [`docs/codex-skills/snake-dqn-queue-jobs/SKILL.md`](../../codex-skills/snake-dqn-queue-jobs/SKILL.md)
- [`docs/codex-skills/snake-dqn-research-ledger/SKILL.md`](../../codex-skills/snake-dqn-research-ledger/SKILL.md)
- [`docs/codex-skills/snake-dqn-runpod/SKILL.md`](../../codex-skills/snake-dqn-runpod/SKILL.md)
- [`docs/codex-skills/snake-dqn-serving-release/SKILL.md`](../../codex-skills/snake-dqn-serving-release/SKILL.md)
