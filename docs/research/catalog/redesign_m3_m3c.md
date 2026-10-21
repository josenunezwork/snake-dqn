# Catalog: redesign M3 and M3-C packages (2026-10-08)

Packages: [`research/redesign_m3_20261008/`](../../../research/redesign_m3_20261008/) and
[`research/redesign_m3c_20261008/`](../../../research/redesign_m3c_20261008/).
Both are read-only, ratified study code; this catalog is written elsewhere so their bytes stay
unchanged. Every statement below comes from files in the repo (2026-10-20 snapshot).

## Purpose

| Package | Question (from its pre-registration) | Status |
|---|---|---|
| M3 | RL fine-tune (synchronous Double DQN) of the M2b `ego2s-b` student (`student_m2b_final.pth`, sha `36a92948…c48f`), anchored to its teacher; does it, wrapped in v8, beat **I = frp3-s12 + v8** on Tier-1 H5000? | Closed. M3-B slope gate PASS, Phase R `NO_GO_EARLY` at look 1. |
| M3-C | Same student, `train_m3c.py` (M3 algorithm + opt-in changes: multi-actor, short anchor, mass reward, eval-matched worlds). Primary: beat I **without any veto**; secondary: with v8. Arms A / B / L (D dropped by amendment A1, E not run) x 5 seeds x 20M. | Closed. No selection entered Phase R; goal unmet. `docs/STATUS.md` marks "M3-C next" as superseded. |

Sub-stages named in the files: **M3-A** (5M Mac smoke), **M3-B** (5 seeds x 20M on a RunPod
4090, learning-slope gate), **M3 Phase R** (Tier-1 sequential gate), **M3-C stage 0** (boost
screen B0 + reference re-probes), **M3-C pod run** (attempt 1, A1, recovery).

## Structure

### `research/redesign_m3_20261008/`

| File | Role |
|---|---|
| `PREREGISTRATION_M3.md` | Binding pre-registration (supersedes `_DRAFT.md`, which keeps rationale and review rounds). |
| `train_m3.py` | Entrypoint: synchronous vectorized Double DQN fine-tune of the ego2s-b student. |
| `student_record_parity.py` | Prerequisite P2: SIMD grid engine == live engine records with the student acting. |
| `p2_h10000.py` | P2 at H10000. |
| `m3a_report.py` | M3-A smoke summary. |
| `m3b_slope.py` | M3-B learning-slope gate (prereg §6). |
| `m3b_slot_worker.py`, `slotrun.py` | Extra probe workers / run one command under shared CPU slot locks (max 3 slots). |
| `gpu_run.py`, `gpu_agent.py` | RunPod 4090 runner (Mac side) and stdlib pod-side agent for M3-B. |
| `rule_spec.py`, `rule_spec_m3.json` | Tier-1 sequential Phase R rule spec (pinned by `tests/test_m3_rule_spec.py`). |
| `phase_r.py`, `phase_r_verify.py` | Phase R harness and shard/record verifier. |
| `results/` | 79 files (see Results). |

### `research/redesign_m3c_20261008/` (a Python package, has `__init__.py`)

| Group | Files |
|---|---|
| Pre-registration / records | `PREREGISTRATION_M3C.md` (FINAL, ratified §11; amendment A1 §12), `PREREGISTRATION_M3C_DRAFT.md`, `STATUS_A1_TAKEOVER_2026-10-08.md` (A1 pod stalled, ~$2.42 est.), `CHECKPOINTS.md` (opt-in full-state checkpoint how-to) |
| Trainer core | `train_m3c.py` (entrypoint; default flags reproduce `train_m3.py`), `learner_m3c.py`, `obs_pack.py`, `demo_store.py`, `league_pool.py`, `league_random_safe.py`, `compile_m3c.py`, `task_collection.py` (Q1 contract), `q0b_timing.py`, `profile_m3c.py` |
| Full-state checkpointing (T21) | `sync_checkpoint.py`, `async_checkpoint.py`, `actor_snapshot.py`, `demo_snapshot.py`, `execution_snapshot.py` |
| Gates / selection | `rule_spec_m3c.py` (pinned by `tests/test_m3c_rule_spec.py`), `phase_r_m3c.py` (header says DRAFT), `m3c_select.py`, `p2_noveto.py`, `stage0_report.py`, `noveto_summary.py` |
| Pod runner | `gpu_run_v2.py`, `gpu_agent_v2.py`, `plans/m3c_pod_plan{,_a1,_draft}.json` |
| Dev diagnostics / benches | `diag_q.py`, `boost_bias.py`, `probe_watch.py`, `bench_m3c.py`, `bench_gather.py`, `bench_v8_nb.py` |
| `results/` | 37 files: `phase_r_smoke_nv.json` + `diag/` |

## Entrypoints (all have `__main__`)

M3: `train_m3.py`, `student_record_parity.py`, `p2_h10000.py`, `m3a_report.py`, `m3b_slope.py`,
`m3b_slot_worker.py`, `slotrun.py`, `gpu_run.py`, `gpu_agent.py`, `phase_r.py`, `phase_r_verify.py`.

M3-C: `train_m3c.py`, `gpu_run_v2.py`, `gpu_agent_v2.py`, `m3c_select.py`, `phase_r_m3c.py`,
`p2_noveto.py`, `stage0_report.py`, `noveto_summary.py`, `probe_watch.py`, `diag_q.py`,
`boost_bias.py`, `bench_m3c.py`, `bench_gather.py`, `bench_v8_nb.py`.
Exact command lines: prereg §2 (M3) and §2/§3 (M3-C); checkpoint flags in `CHECKPOINTS.md`.

## Results and where recorded

| Result | In-package evidence | Ledger |
|---|---|---|
| P2 parity: student + v8 9/9 worlds identical (no-veto 6/9 differ) | `results/p2_v8.json`, `results/p2_none.json`, `results/m3b_p2/` | `PREREGISTRATION_M3.md` §3 |
| M3-A smoke, no HALT/FLAG | `results/m3a/` | [INDEX.md](../INDEX.md) row 2026-10-08 |
| M3-B slope gate PASS: mean slope +1.21/M, t90 lower +0.46 → "PROCEED to Phase R" | `results/m3b/slope_gate.json` (+ 61 probe jsonl) | INDEX.md; `docs/STATUS.md` (4.25x Mac, $3.92) |
| M3 Phase R `NO_GO_EARLY` look 1: +11.4 (SE 13.7), UB +40.5 < G=50; verify PASS (825 records) | `results/phase_r/look-1.json`, `verify.json`; smokes in `results/phase_r_smoke{,2}/` | INDEX.md; `docs/STATUS.md` |
| M3-C stage 0: boost screen B0 (arm E not run) | `results/diag/stage0_*.json`, `boost*` | `PREREGISTRATION_M3C.md` §4 |
| M3-C Phase R no-veto smoke: look 0 `KILL_V4` (dev only, expected) | `results/phase_r_smoke_nv.json` | `PREREGISTRATION_M3C.md` §8 |
| M3-B no-veto dev probes vs 0M | `results/diag/noveto_summary.json` | unclear (no ledger row found) |
| M3-C final: no-veto A 177.4 / B 172.9 / L 184.6 vs incumbent+v8 434.9; with v8 ~435-448; no selection entered Phase R; ~$7.9 total | **not in package** — `docs/research/m3c_m3d_evidence/` (e.g. `retry-recovery-closeout.json`) | [m3c_m3d_learnings_2026-10-12.md](../m3c_m3d_learnings_2026-10-12.md); INDEX.md row 2026-10-12 |

Design docs: [m3c_design_2026-10-08.md](../m3c_design_2026-10-08.md),
[redesign_scope_2026-10-07.md](../redesign_scope_2026-10-07.md),
[redesign_campaign_retrospective_2026-10-17.md](../redesign_campaign_retrospective_2026-10-17.md).
Large artifacts (checkpoints, pod staging dirs, 4.06 GB demo bundle copies) live outside the repo
under `snake-dqn-artifacts/redesign-m3*`; see
[ARTIFACTS_INVENTORY_2026-10-20.md](../ops/ARTIFACTS_INVENTORY_2026-10-20.md).

## Reusable vs dead

**Live (imported by later work — do not move without updating importers and study hashes):**

- `redesign_m3c_20261008.train_m3c` — the shared trainer: hashed/launched by
  `research/s1_mature_20261018/` (`train.py`, `admission.py`, `audit.py`, `price.py`,
  `final_artifacts.py`, `checkpoint_sizing.py`, `manifests.py`, `run.py`), and imported by many
  tests (`test_m3c_trainer.py`, `test_perf_*`, `test_q1_*`, `test_t21_*`, `test_s1_*`, ...).
- `obs_pack` (most-imported module: `train_layout`, `pack_ref`, `PackLayout`), `learner_m3c`,
  `demo_store`, `league_pool`, `league_random_safe`, `async_checkpoint`, `actor_snapshot`,
  `execution_snapshot`, `compile_m3c`, `q0b_timing`, `profile_m3c`, `task_collection`.
- `gpu_run_v2`, `probe_watch`, `phase_r_m3c`, `m3c_select` — still tested
  (`test_m3c_selfdelete_runner.py`, `test_m3c_select.py`, ...); pod runner role in T34 is unclear
  (T34 uses no RunPod per memory/STATUS).
- `redesign_m3_20261008.student_record_parity` — reused by `research/t12_b8_serving_20261017/parity.py`.
- `redesign_m3_20261008.rule_spec` / `phase_r` — exercised by `tests/test_m3_*` and
  `tests/test_m4_*` (M4 reused the M3 sequential Phase R form).
- `docs/codex-skills/snake-dqn-m3c-trainer/SKILL.md` documents the trainer.
- `src/simd_env/{ego_live_bv8,vector61_merged}.py` reference these packages only in docstrings.

**Dead (historical evidence only; question closed):** `train_m3.py` (superseded by
`train_m3c.py` defaults), `gpu_run.py`/`gpu_agent.py` (v1 runner), `m3a_report.py`,
`m3b_slope.py`, `m3b_slot_worker.py`, `slotrun.py`, `p2_h10000.py`, the `_DRAFT.md`
pre-registrations, `plans/m3c_pod_plan_draft.json`, `stage0_report.py`, `noveto_summary.py`,
`diag_q.py`, `boost_bias.py`, `bench_*.py`. Whether any test still imports each of these
individually was not exhaustively checked (unclear); keep all as ratified evidence.
