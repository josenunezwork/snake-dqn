# Codex and Claude Operational & Research Skills (`agent-skills`)

## Question / Goal

Maintain a version-controlled repository mirror (`docs/codex-skills/`) of the procedural `snake-dqn-*` skills (`~/.codex/skills/snake-dqn-*`, the source of truth) used by the Codex agents (Sol = implementer; Astra = advisor personas), covering codebase contracts, game mechanics, simulator parity, Mac operations, evaluation gates, pre-registration, compute economics, and web-serving release (`docs/codex-skills/README.md`).

## Timeline of Versions / Steps

| Date | Step / Skill | Scope & What Was Run / Documented | Verdict / Status |
| :--- | :--- | :--- | :--- |
| 2026-10-18 | `docs/codex-skills/README.md` | Backup index, written 2026-10-18 by Fable agents from the repo, docs and incident history and reviewed by Claude; refresh command copies each `~/.codex/skills/snake-dqn-*/SKILL.md` into `docs/codex-skills/<name>/` | Active index (`KEEP`) |
| 2026-10-18 | Codebase & mechanics skills (`codebase`, `game-mechanics`, `parity-determinism`, `git-workflow`) | Codifies the Apex 58-D state vector and six relative actions (left/straight/right × normal/boost), live game vs `BatchSim` vs `GridBatchSim` parity tests, Mac-vs-x86 BLAS float32 Q-tie flips (~1 in 50 H10000 episodes), and hash-bound (sha256) docs/receipts that must never be edited | Active (`KEEP`) |
| 2026-10-18 | Mac & serving skills (`mac-ops`, `serving-release`) | Codifies the M5 Pro (18 cores, 64 GB) device policy (CPU default, MPS opt-in; B8 learner on MPS), per-process thread caps, CPU slot locks (Tier-1/dev up to 3 slots × 2 threads; strict gates exactly 2 slots) with thermal guard, `caffeinate -dims` + never-on-battery rule, and the pinned served-checkpoint registry (`b8-s3`, `frp3-s12`, `champion`) with `ego2s` speed ceilings (≤ 6 driven AI → 24 fps; 7–12 → 12 fps) | Active (`KEEP`) |
| 2026-10-18 | Research & advisor skills (`advisor-brief`, `evaluation-gates`, `experiment-runbook`, `performance`, `preregistration`, `research-ledger`) | Codifies Astra advisor briefs, Tier-1 screens vs Tier-2 strict gates on fresh namespaces, sequential Phase R looks (OBF for GO, Pocock for KILL), survival band v2 `pooled_ni_continue`, fixed-order pre-registered decision rules, measured throughput, and the dated research ledger | Retained reference (`KEEP`; ceremony rules overridden by the 2026-10-30 Fast-process rule in `AGENTS.md`) |
| 2026-10-18 | Historical / frozen-line skills (`apex-training`, `m3c-trainer`, `compute-economics`, `queue-jobs`, `runpod`) | Documents the frozen `vector61` line (S2's 67-D sight inputs default if reopened), the `train_m3c.py` B8 trainer, the Q0/Q1 Colab budget envelope, the legacy Drive queue (`research/colab_queue/`), and the RunPod per-step ≥ 4x rule | Retained historical mirror (`KEEP`; Q0/Q1 envelope and RunPod-for-T34 superseded per `AGENTS.md`) |

## Final Verdict and Key Numbers

- **Skill inventory:** **18** files in `docs/codex-skills/` (`169,999` bytes total; README plus 17 `SKILL.md`, excluding `docs/codex-skills/snake-dqn-colab-jobs/SKILL.md`, which is tracked under the `colab-ops` lineage) are retained as the repository backup mirror of `~/.codex/skills/snake-dqn-*` (`docs/codex-skills/README.md`).
- **Mac operational guardrails:** On the Apple M5 Pro (18 CPU cores, 20-core GPU, 64 GB), every launcher exports `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`; Tier-1/dev work uses up to 3 CPU slots × 2 threads with the thermal guard, strict gates exactly 2 slots, always on AC with the lid open (`docs/codex-skills/snake-dqn-mac-ops/SKILL.md`).
- **Statistical gate defaults:** Strict gates use sequential OBF looks with Bonferroni per mix, scripted NI, and survival band v2 `pooled_ni_continue` (pooled NI `0.05`, per-mix floor `0.075`, absolute `0.30`); sequential Phase R looks at 1/3, 2/3 and 1 of planned worlds with OBF spending for GO and Pocock for KILL; Tier 2 product bar = pooled scripted episode survival vs `champion_a5+v8` `>= -0.05` (`docs/codex-skills/snake-dqn-evaluation-gates/SKILL.md`).
- **Throughput baselines:** Mac (M5 Pro, MPS learner) B8 recipe ~5.2–5.5k tr/s at ~21 upd/s, learner-bound; Colab A100 isolated learner fp32 78–83 upd/s (full run ~62 upd/s = 3.0x Mac); RTX 4090 pod learner 14.9 → 8.4 ms with `--learner-fast --compile-learner reduce-overhead` (1.74x) (`docs/codex-skills/snake-dqn-performance/SKILL.md`).
- **Superseded campaign parameters:** The Q0+Q1 Colab envelope (≤ 150 units +25 by Amendment A1, Q0 cap 40 units, Q1 training ≤ 60 units; total initial commitment ≤ 250 of ~1,250, in `docs/codex-skills/snake-dqn-compute-economics/SKILL.md`) is listed as superseded in `AGENTS.md`; new Colab work goes through the qualified CLI supervisor (`research/colab_cli`, [`research/colab_cli/QUALIFICATION.md`](../../../research/colab_cli/QUALIFICATION.md)) rather than the legacy Drive queue (`research/colab_queue/` in `docs/codex-skills/snake-dqn-queue-jobs/SKILL.md`).

## What Was Learned

1. Procedural skills that encode durable invariants—state/action contracts, simulator parity tests, Mac thread caps and slot locks, hash-bound receipts, and serving-registry fallbacks—give agent sessions a shared operating baseline.
2. Skills that hard-code campaign-specific budgets (the Q0/Q1 envelope), retired queues (`research/colab_queue/`), or pre-Fast-process ceremony age quickly; authoritative campaign rules belong in `AGENTS.md` and `docs/STATUS.md`.

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
