# Catalog: redesign packages (other than M2 / M3 / M3-C)

Written 2026-10-20 from the package files and the result docs they point to. Read-only catalog:
it changes no package. Out of scope here (cataloged elsewhere): `redesign_m2_20261008`,
`redesign_m3_20261008`, `redesign_m3c_20261008`. Where a fact could not be found in the files,
it says "unclear".

| Package | Kind | Verdict | Status |
|---|---|---|---|
| [`redesign_scope_20261007`](../../../research/redesign_scope_20261007/) | M1 benchmarks + identity harnesses | M1 built; G1 met on scripted mix, short on vector61 mixes | Reusable (`grid_h5000_identity.py` widely imported) |
| [`redesign_m4a_20261012`](../../../research/redesign_m4a_20261012/) | Pod plans only | Training inputs for M4 (arms C/D/S/B8) | Historical record |
| [`redesign_m4b_20261012`](../../../research/redesign_m4b_20261012/) | `widen_bv8.py` warm start | Produced the B8 init (ego2s-b → ego2s-bv8) | Reusable utility (tested) |
| [`redesign_m4_20261013`](../../../research/redesign_m4_20261013/) | Prereg + selection + Phase R harness | **PARTIAL** (B8 +128.7 mass; guard scripted survival failed) | Historical; rule spec pinned by tests |
| [`redesign_m6_20261014`](../../../research/redesign_m6_20261014/) | Pod plan only | **NULL / not adopted** (run on Mac) | Dead (pod never used successfully) |
| [`redesign_t4_20261015`](../../../research/redesign_t4_20261015/) | Placeholder pod plan (dry-run) | Dry-run PASS, no launch | Dead / unused |
| [`redesign_t7_phase_r_20261016`](../../../research/redesign_t7_phase_r_20261016/) | Prereg + harness + full results | **STRICT_PASS**, seed 3 | Result of record; harness imported by later studies |

---

## `redesign_scope_20261007` — Milestone 1 of the redesign

**Purpose.** Code-only scoping of the redesign (vectorized simulator `GridBatchSim` + ego-raster
"ego2s" spatial observation), approved by the owner 2026-10-07. The package holds the M1
benchmarks and development-only identity checks; the engine itself lives in `src/simd_env/`
(`grid_sim.py`, `grid_sim_nb.py`, `ego_raster.py`, `ego_raster_nb.py`, `ego_live_adapter.py`).

**Entrypoints.**
- `bench_throughput.py` — live physics vs `BatchSim` vs `GridBatchSim` (+ featurizers), one core; `summarize.py` tabulates its JSON.
- `bench_g1.py` — Gate G1 (≥ 20k hero agent-steps/s per process incl. opponent rows); `summarize_g1.py` tabulates `results/bench_g1_20261007.jsonl`.
- `bench_featurizer_split.py` — ego2s featurizer NumPy vs numba cost.
- `bench_network.py` — learner-side cost of the draft ego2s dueling CNN (sizing only).
- `grid_h5000_identity.py` — M1(i): frp3-s12 H5000 episodes on GridBatchSim vs BatchSim, record + per-frame digest identity.
- `ego_live_identity.py` — M1(iii): ego2s from a live `GameState` == ego2s from GridBatchSim.

**Result and where recorded.** [`docs/research/redesign_scope_2026-10-07.md`](../redesign_scope_2026-10-07.md)
§14 (M1 results; later §15–18 cover M2/M2b/pre-M3). Identity: `results/grid_identity_h5000/summary.json`
is labeled "development-only identity check (not a screen, not gate evidence)" and shows 8/8 identical
worlds per (veto arm × mix) cell. G1 (§14.4): the scripted mix reaches ~21k–28k hero agent-steps/s
with M1 (vs ~700 for the prototype); the vector61 gate mixes fall 5–15× short of 20k, which pinned
the M3 training opponent mix to scripted anchors. Independent review (§14.5): GO-with-fixes, no
correctness bug. `results/prelim_pre_4664f24/` holds pre-fix preliminary identity runs.

**Reusable vs dead.** `grid_h5000_identity.py` is live infrastructure: ~95 imports
(`grid_h5000_identity as gid`, `env_digest`, `strip_timing`, `_context`) from tests and later
packages (e.g. `t33_batched_teacher`, `t34_pathb`, `t34_round0`, `q0q1_20261017`,
`t12_b8_serving_20261017`, `s1_mature_20261018`, M3/M3-C). The bench scripts and `summarize*.py`
are one-off measurement tools (historical; rerunnable). `ego_live_identity.py` is referenced a few
times; unclear whether still exercised.

## `redesign_m4a_20261012` — M4-A pod plans

**Purpose.** Pod plans (`m3c-gpu-plan/v1` schema) for the M4-A training step, which fixed the
M3-C training setup so the student learns not to need the veto. Design:
[`docs/research/m4a_design_2026-10-12.md`](../m4a_design_2026-10-12.md) (setup fixes: mass reward
κ 1, kill bonus 0, γ → 0.998, 50/25/25 scripted/frozen/mixed worlds, v8-margin anchor; init = M2b student).

**Entrypoints.** No code. `plans/m4a_pod_plan.json` (arms C, D = C + on-policy v8 labels,
S = D + training-time v8 shield; 20M transitions) and `plans/m4a2_pod_plan.json` (D/S rerun with
the numba v8 because pod-1 D/S ran at 400–650/s, plus the new **B8** arm = `ego2s-bv8` obs with
v8 features as inputs). Training script used: `research/redesign_m3c_20261008/train_m3c.py`
(per PREREGISTRATION_M4.md §2).

**Result and where recorded.** Training outcome feeds M4 (below). Per PREREGISTRATION_M4.md §3,
only arm C had five seeds at selection time; D/S had none (pod capacity). B8 finals used by M4/T7
are at `snake-dqn-artifacts/redesign-m4b-mac-20261013/B8_s<S>/final.pth` (trained on the Mac,
per PREREGISTRATION_T7.md and the M4 result doc).

**Reusable vs dead.** Historical record of what was launched; not imported by code except as
a plan reference. The plan schema is reused by M6/T4 plans.

## `redesign_m4b_20261012` — B8 warm start

**Purpose.** `widen_bv8.py` widens an `ego2s-b` student into an `ego2s-bv8` one: copies every
weight and adds 30 zero-initialized input columns for the v8 feature scalars, so Q is unchanged
on the b part (pinned in `tests/test_ego_raster_bv8.py`).

**Entrypoints.** `./venv/bin/python research/redesign_m4b_20261012/widen_bv8.py SRC_B.pth OUT_BV8.pth`;
`widen_net` imported by tests and other code (5 references).

**Result.** Not a study; it is the init path of arm B8 (the m4a2 plan notes "init widened in memory").

**Reusable vs dead.** Reusable utility; test-pinned.

## `redesign_m4_20261013` — M4 Tier-1 Phase R (first B8 challenge)

**Purpose.** Pre-registered Tier-1 sequential Phase R: does an M4-A student **without a veto**
beat I = frp3-s12 + v8 at H5000 (primary), and with v8 (secondary)?

**Entrypoints.** `PREREGISTRATION_M4.md` (RATIFIED 2026-10-13; §14 Amendment B lets arm B8
enter); `m4_select.py` (arm selection / entry gates; `tests/test_m4_select.py`);
`rule_spec_m4.py` (rule specs, pinned by `tests/test_m4_rule_spec.py`); `phase_r_m4.py`
(thin wrapper around the pinned M3-C harness, sha-checked private copy; `tests/test_m4_phase_r.py`);
`oc_m4.py` (operating characteristics for a 3-seed rule → `results/oc_m4.json`).

**Result and where recorded.** **PARTIAL** —
[`docs/research/m4_phase_r_result_2026-10-13.md`](../m4_phase_r_result_2026-10-13.md).
B8 selected over C (dev probes 545.0 vs incumbent 434.9; C 517.5 failed the scripted survival
guard). Primary mass +128.7 (LB +101.5), 5/5 seeds positive, all survival clauses vs I pass;
**failed only c6 guard scripted survival vs champion_a5 + v8 (−0.156 vs ≥ −0.05)**. Raw results
live outside the repo at `snake-dqn-artifacts/redesign-m4-phase-r-nv-20261013/`; the package's
own `results/` holds only `oc_m4.json` and the dev smoke `phase_r_smoke_nv.json` (note: "dev only
... the KILL_V4 says nothing about M4 PASS"). `research/INDEX.md` currently says "verdict not
recorded here" for this package.

**Reusable vs dead.** Historical. `rule_spec_m4` remains imported (6 references) and the M4
equal-mix seed effects / HK pool were reused by T7's efficacy clause.

## `redesign_m6_20261014` — M6 reward levers (pod plan only)

**Purpose.** Pod plan for paired 8M fine-tunes from the five B8 finals with arms CTRL /
TIGHT (`--tight-cost 2 --tight-min-len 600`) / DEATHW (`--death-mass-mult 2`), aimed at B8's
scripted-opponent survival gap (motivated by the B8 death census: self-enclosure at length ≥ 800).

**Entrypoints.** `plans/m6_pod_plan.json` only. Training flags lived on branch `m6`
(`train_m3c.py` M6 flags).

**Result and where recorded.** **NULL / not adopted** —
[`docs/research/m6_result_2026-10-14.md`](../m6_result_2026-10-14.md). TIGHT − CTRL −24 mass,
scripted survival −0.044; DEATHW − CTRL +10 mass, +0.031 (both within noise). Pod attempts failed
operationally (~$1.9; CPU minimum, torch API mismatch, CUDA-graph OOM, 3090 fallback) and the
batch ran on the Mac; this drove runner hardening. Raw: `snake-dqn-artifacts/redesign-m6-mac-20261014/`.

**Reusable vs dead.** Dead: the plan was never used for the delivered run. Lesson recorded in the
result doc: screens need ≥ 96 scripted worlds (32 worlds detect only ~0.1 survival effects).

## `redesign_t4_20261015` — T4 placeholder pod plan

**Purpose.** Dry-run placeholder pod plan (`"study": "T4 PLACEHOLDER: prefer-nonboost bank v2;
actual arm chosen only after T3 reports"`), five 8M fine-tunes pairing seeds 10–14 with B8_s0–4.

**Entrypoints.** `plans/t4_pod_plan.json` only.

**Result and where recorded.** [`docs/research/t4_pod_plan_2026-10-15.md`](../t4_pod_plan_2026-10-15.md):
"Dry-run PASS; 4× speed-up NOT DEMONSTRATED. No launch is authorized"; $0 spent. The actual T4
arms (T4a TWO_SIDED, T4c TWO_SIDED_T, T4e/T4f RETURN / RETURN_C) ran as Mac screens and were all
KILL / halted — see the T4 rows of
[`redesign_campaign_retrospective_2026-10-17.md`](../redesign_campaign_retrospective_2026-10-17.md)
and `docs/research/t4*_2026-10-16.*`.

**Reusable vs dead.** Dead / unused (INDEX.md: UNUSED).

## `redesign_t7_phase_r_20261016` — T7 B8 fresh-bank Phase R (result of record)

**Purpose.** One-look, fresh-bank Phase R for B8 (no veto) vs served frp3-s12 + v8, with the
Tier 1 guard amendment and a Tier 2 scripted-survival check vs champion_a5 + v8. Ratified
2026-10-16 from `docs/research/specs/t7_phase_r_b8_spec_2026-10-16.md`.

**Entrypoints.** `PREREGISTRATION_T7.md` (pins the five B8 final SHA256s); `rule_spec_t7.py`;
`seed_audit.py` (bank generation / seed audit → `SEED_AUDIT_T7.json`); `phase_r_t7.py` (runner);
`results/launch.sh`, `results/launch-resume-1.sh`, `results/coordinator_resume.py` (operational
recovery adapter). Seed-allocation errata and rejected allocations are kept as
`SEED_ALLOCATION_*` files. `VALIDATION_T7.json` = pre-run validation.

**Result and where recorded.** **STRICT_PASS, candidate seed 3** —
[`docs/research/t7_phase_r_result_2026-10-16.md`](../t7_phase_r_result_2026-10-16.md);
machine record `results/decision.json` (sole full-N decision; 5,805 raw records inventoried),
`results/audit.json`, `results/independent_result_review.json`, `results/README.md`. H5000 mass
+138.7 (LB90 +129.5), 5/5 seeds, all survival clauses pass; Tier 2 scripted survival −0.0156
(≥ −0.05) PASS. Run on commit `7e17343`, Mac CPU, 6.8 h. Release followed via T12:
[`b8_s3_release_2026-10-17.md`](../b8_s3_release_2026-10-17.md).

**Reusable vs dead.** Result of record (do not modify). `seed_audit`, `rule_spec_t7` and
`phase_r_t7` are imported by later code and tests (~19 references, e.g. `tests/test_t7_owned_signals.py`,
`t34_*`, `phase0_search_20261019`); its `SEED_AUDIT_T7.json` is consumed by later seed audits.

---

## How the redesign line led to B8

1. **Scope / M1 (2026-10-07).** The 61-D vector line was near its ceiling; the owner approved a
   redesign: `GridBatchSim` (byte-identical to `BatchSim` at H5000) + ego2s spatial observation.
   G1 throughput forced scripted-anchor training opponents.
2. **M2 / M2b / M3 / M3-C (2026-10-08, other catalog).** frp3-s12(+v8) distilled into an ego2s
   student (M2 NI FAIL, M2b `ego2s-b` NI PASS), then RL. M3-C's no-veto students survived only
   1–11 % of games (per `m4a_design_2026-10-12.md`).
3. **M4-A (2026-10-12).** Setup fixes (mass reward, no kill bonus, γ 0.998, mixed worlds, v8-margin
   anchor). Arm C improved mass but failed scripted survival. The m4a2 plan added **B8**: the v8
   veto's 30 per-action safety features given to the network as **inputs** (`ego2s-bv8`, init by
   `widen_bv8.py`) instead of a decision-time veto.
4. **M4 Phase R (2026-10-13).** B8 no-veto +128.7 mass over frp3-s12 + v8, 5/5 seeds — PARTIAL only
   because of the guard's scripted survival vs champion_a5.
5. **Late-death levers (M6, T2–T6, T11; 2026-10-14..17).** All NULL / KILL (retrospective).
   The guard itself was reconsidered (`specs/guard_clause_amendment_2026-10-15.md`).
6. **T7 (2026-10-16).** Fresh-bank one-look Phase R under the amended guard: STRICT_PASS, seed 3.
7. **T12 / release (2026-10-17).** Serving qualification passed; B8 seed 3 became the served
   Watch hero with no veto.
