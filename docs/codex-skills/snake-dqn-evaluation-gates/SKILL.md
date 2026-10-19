---
name: snake-dqn-evaluation-gates
description: |
  Use whenever you evaluate, compare, screen or gate a snake-dqn candidate checkpoint in
  /Users/josenunez/Projects/ml/snake-dqn: writing or running an evaluator (tournament_eval.py, a Phase R
  producer, the T23/Q1 evaluators under research/), allocating or auditing world banks, choosing metrics
  (mass integral vs survival), reading or writing a result/decision for a candidate, planning a strict gate
  or serving qualification, or interpreting Tier 0/1/2 labels (STRICT_PASS, GO, KILL, PARTIAL ...). It
  explains how candidates are judged here and the pitfalls that already produced wrong conclusions.
---

# Snake DQN: how candidates are evaluated and gated

For a new Colab development evaluation, ship Mac-made receipts as hash-checked `research/colab_cli` supervisor inputs
and retrieve immutable completed shards through that supervisor; no runtime Drive mount. The Drive-folder warning in
this skill is a legacy Q1 queue failure mode only.

Sources of truth: `AGENTS.md` ("Promotion Gate"), `docs/STATUS.md` (current rules; wins over older docs),
`docs/research/GLOSSARY.md`, `docs/research/governance_tiers_2026-09-26.md`. When a dated doc disagrees
with STATUS.md, STATUS.md wins. Never edit a hash-bound governance/research doc; add a sibling erratum.

## 1. The measurement unit: a terminal-hero H5000 paired world

- **World** = one seed + a fixed ordered opponent roster + a deployment profile
  (`promotion-v2-watch-rect`, 5000 frames, digest `d396d3ed...`; H10000 digest `c882e07f...`).
  Only the slot-0 **hero** changes between arms; opponents, seats, wrapper, mechanics are identical.
- **Terminal hero**: the hero's death ends its scoring (no respawn credit); opponents keep evaluation
  respawn semantics. A nested H10000 episode records its exact **H5000 prefix**; direct-H5000
  prefix-identity controls (T7: 45 of them) must match byte-for-byte and are excluded from estimates.
- **Three opponent mixes**, never clones of one checkpoint (`src/scripts/tournament_eval.py` docstring):
  `scripted` (greedy_food anchors, ungameable), `frozen` (cycles the frozen pool), `mixed`
  (frozen + random_safe). Rosters come from the strict-design materializer in
  `research/apex_safety_20260926/dev_screen.py` (`screen_seeds`, `_design_rows`).
- **Paired**: candidate and reference play the same worlds; statistics are on per-world deltas, pooled
  per training seed, then across seeds.

## 2. Metrics and which one decides

| Metric | Definition | Role |
|---|---|---|
| **Mass integral** (primary) | mean per-frame logical mass over the TOTAL horizon, dead frames = 0 (`src/scripts/eval_stats.py: mass_integral`) | efficacy, the only promotion endpoint |
| Alive-frame fraction (`survival_fraction`) | alive frames / horizon | survival guard, pooled per mix (>= -0.03 vs incumbent) and per-seed scripted floor (>= -0.15) |
| No-death episode survival | `prefix_h5000.deaths == 0` (a hero that dies after H5000 still counts) | absolute floor (>= 0.50 per mix), Tier 2 product bar (>= -0.05 vs champion_a5+v8) |
| `mean_mass_alive` | legacy, mass over alive frames only | diagnostic only; the old gate ranked by it and could promote regressions |
| MI10 | mass integral at H10000 | secondary clause (c5_mi10 >= 20) |
| deaths/1k decisions, length-matched hazard, boost rate | descriptive | never decide a gate; boost rate is a training tripwire |

Mass and survival are different quantities with different thresholds; a scripted *mass* point guard is
not the scripted *alive-frame* floor (T23 `EVALUATION.md` makes this explicit). Say which one you mean.

## 3. Statistics (do not improvise)

- Seeds (training seeds), not worlds/frames/seats, are the independent units for the pooled effect.
  Each seed's effect = equal-weight mean of its three mix means (scripted gets no extra weight from its
  larger N); variance `sum(s_mix^2/n_mix)/9`.
- Phase R / M4 / T7: Hartung-Knapp pooled effect with `hksj_floor`, df=4; **LB90/UB90 are one-sided 90%
  bounds** (ends of a two-sided 80% interval), not SEs. Helpers: `src/scripts/eval_stats.py`
  (`paired_stats`, `paired_delta_test`, `student_t_isf`, `holm_three_mix_superiority`,
  `scripted_noninferiority`, `strict_promotion_decision`).
- T23/Q1 development screens: one-sided 90% Student-t across the five seed effects (df 4) on G =
  candidate - B8; hierarchical world-within-seed bootstrap is a *sensitivity*, not more seeds.
- Candidate-seed selection rule (fixed before data): among seeds passing their own scripted floor, pick the
  largest lower endpoint of the two-sided 80% Welch-Satterthwaite interval; ties -> lowest seed.
- Feasibility clause c7: `N_max = ceil(1.02*n_fixed) <= 600` with MDE = max(HK LB90, 0.6*HK mean),
  per-mix alpha 0.05/3, power 0.90.

## 4. Tiers and labels

| Tier | What it is | Can change the served agent? | Examples |
|---|---|---|---|
| 0 | saved-data diagnostic, no new episodes | no | death census, T1 commitment study |
| 1 | development screen on Tier-1 namespaces | no | Phase R (`GO`/`KILL`/`PARTIAL`/`NO_GO_EARLY`), T2-T6, T23 (`GO`/`KILL`/`EXTEND`/`INCONCLUSIVE`), Q1 |
| 2 | strict gate on fresh Tier-2 namespaces, independent audit | yes, after serving qualification | v8 strict, FRP-v3 strict (`STRICT_PASS`), T7 fresh-bank Phase R under the Tier 1 guard |

Labels are study-local (`GO_S_RECIPE`, `KILL_H`, `PARTIAL`, `GO_L`, `GO_R`): never convert them to a generic
PASS/FAIL or re-judge an old study under newer rules. A Tier-1 number is never Tier-2 evidence; it can only
inform a Tier-2 design (variance for power). A `GO` buys the next step, never promotion.

### The current release gate (guard-clause amendment, 2026-10-15; applied in T7)
`docs/research/specs/guard_clause_amendment_2026-10-15.md`, `research/redesign_t7_phase_r_20261016/PREREGISTRATION_T7.md`:
- **Tier 1 (binding)**: efficacy HK LB90 > 0 vs served agent (frp3-s12+v8), mass point >= 20, scripted mass
  >= 0, pooled alive-frame survival >= -0.03 on each mix, per-seed scripted alive-frame floor >= -0.15,
  absolute episode survival >= 0.50 per mix, >= 4/5 seeds positive, MI10 >= 20, prefix controls identical,
  seed-0 guard mass vs champion > 0, feasibility N_max <= 600. All on fresh banks, one look at full N.
- **Tier 2 (reported, product bar)**: pooled scripted episode survival vs champion_a5+v8 >= -0.05.
  Campaign "done" = Tier 1 AND Tier 2. T7 passed by 0.034 at SE 0.023 (narrow; disclosed).
- Clause results live in `research/redesign_t7_phase_r_20261016/results/decision.json` and
  `docs/research/t7_phase_r_result_2026-10-16.md` (full clause table with SEs).

### Older strict gate machinery (vector61 line, still the reference for champion changes)
`research/sequential_strict_template/` (`sequential_runner.py prepare/run/status`, `sequential_audit.py`
stdlib-only, intents hash-bound). Sequential OBF looks, Bonferroni per mix, scripted NI, survival band v2
`pooled_ni_continue` (pooled NI 0.05, per-mix floor 0.075, absolute 0.30; FRP-v3 passed only under it,
`docs/research/governance_amendment_survival_band_v2_2026-10-06.md`). `src/evaluation/strict_promotion.py`
is the byte-bound artifact consumer (`tournament_eval.py --strict-promotion-request ...`); it never computes
worlds itself.

### Sequential Phase R
Default for new Tier-1 studies (`docs/research/governance_amendment_sequential_phase_r_2026-10-07.md`,
`src/evaluation/sequential_phase_r.py`): looks at 1/3, 2/3, 1 of planned worlds, OBF spending for GO,
Pocock for KILL, optional NO_GO futility at G=50; a fixed-N registration is never stopped early (stopping
invalidates it). T7 opted for one look at full N instead; both are legitimate when registered.

## 5. Tools

- `src/scripts/tournament_eval.py` (the shared gate; `--engine live` for vector61, `--engine simd` for
  raster; `--gate` is a *legacy diagnostic* exit code and "never grants strict promotion authority";
  `--evaluation-profile promotion-v2-watch-rect` requires `--frames 5000`; `--pilot` sizes seeds).
  Promotion rule in `promotion_decision()`: paired delta > 0 at 95% CI on >= 2 mixes and no regression vs
  the scripted anchor. With `--engine simd`, baseline/opponents must be raster or `scripted:` stand-ins.
- Phase R producers: `research/redesign_t7_phase_r_20261016/phase_r_t7.py` + `rule_spec_t7.py` +
  `seed_audit.py`; M4's in `research/redesign_m4_20261013/`. They bind commit, checkpoint bytes, banks,
  rosters, unit assignments into a create-only `plan.json`/`start.json`; resume only hash-validated records.
- T23 evaluator (`research/t23_r0_screen_20261017/`): `bank.py` (allocates worlds only; namespace
  `t23-r0-h5000-terminal-hero/v2`, 64/32/32 per seed, one reserved extension bank, 4 parity worlds),
  `evaluate.py --shard i/n --workers k [--include-descriptive] [--phase extension --prior-verdict ...]`,
  `analyze.py` (INVALID/INCOMPLETE precedence, then KILL, GO, EXTEND once, INCONCLUSIVE). Runtime parity
  against `parity/mac.json.gz` runs before any selection world.
- Q0/Q1 (`research/q0q1_20261017/`): `audit_q0_bank.py` (Q0 dev bank, disjoint from Q1),
  `q1_bank.py` (namespace `q1-mature-n1n3-h5000-terminal-hero/20261017/v1`), `q1_parity.py`
  (`--reference` on the Mac, `--verify-reference` on each runtime), `q1_evaluate.py`, `q1_analyze.py`
  (accepts only a `q1-analysis-input/v1` object with bank + collision audit + immutable shard receipts),
  `inventory_receipt.py` (ships the Mac-only collision inventory to Colab as a hash-bound receipt),
  `audit_t23_mass.py --workers N` (deterministic merge order; `--workers 1` is the serial reference).
- Eval runs on the Mac use one math thread per worker (`OMP/MKL/OPENBLAS/VECLIB/NUMBA_NUM_THREADS=1`,
  `SNAKE_DQN_DEVICE=cpu`), at most the slot budget in `docs/research/compute_policy_2026-10-02.md`.

## 6. World banks and collision audits

- Every new study allocates **fresh** worlds under a **new namespace/round key** via
  `dev_screen.screen_seeds(count, namespace, key)`. All identities must be distinct within the study.
- Before any gameplay, a **seed audit** compares the allocation against every prior registration and the
  historical inventories (dev rounds, census, M3/M4 Phase R, FRP training exclusions, T3 labels, T4
  collections, strict banks; T7 scanned 1,182,309 values; T15 1,184,677). A single collision rejects
  the **whole allocation**; take the next consecutive round, never swap one world (T7 rejected rounds 74
  and 75: `SEED_ALLOCATION_ERRATUM_T7.md`, `SEED_ALLOCATION_ERRATUM_76_T7.md`). Record coverage limits
  (opaque binaries, unrecorded external runs).
- Historical FRP exclusion inventories are Mac-resident: a Colab job must consume a Mac-made receipt
  (`inventory_receipt.py`, env `Q0Q1_BANK_INVENTORY_RECEIPT[_SHA256]`), not a partial re-audit.
- Development worlds and strict/gate worlds are disjoint; a world used for labels/training is burned for
  gating.

## 7. Serving qualification (after a pass, before release)

Mac only, one thread, AC/open lid. Pattern: `research/frp3_checkpoint_serving_20261005/protocol.md`
(S1-S6), extended for B8 in `docs/research/t12_b8_serving_qualification_2026-10-17.md` (checks 1-8:
live observation identity vs SIMD, veto absent and says so, registry/pin/rollback matrix, latency and
UI-speed ceilings, 25 Watch H5000 episodes as monitoring baseline, 25 Play episodes, UI/confidence,
Train refusal). Mixed roster: T15 (six checks, `research/t15_mixed_roster_20261017/protocol.md`).
Serving qualification measures correctness, not skill; it makes no mass claim. Details in the
`snake-dqn-serving-release` skill.

## 8. Platform rules

- Strict gates and serving qualification run on the **Mac**. The strict-on-RunPod amendment is NOT
  ratified (`docs/research/governance_amendment_strict_on_runpod_2026-10-05_status_2026-10-08.md`).
- Colab/pod rows are development evidence. Any cross-platform use needs a per-runtime parity check
  against a Mac reference that **explains** divergences (action-hash stream + top-2 Q margin; a mismatch
  is acceptable only at a <= ~4-ulp tie on the Mac side). The 4-world probe is a smoke, not identity.
- Per-world single platform: candidate, reference and controls of one world run on one worker/CPU model.
- RunPod only when >= 4x faster per step than the Mac; cheap pods first (STATUS.md).

## 9. Pitfalls that already produced wrong conclusions (with sources)

1. **Self-play eval inflates skill** (Claude memory `slither-eval-pitfalls.md`, 2026-06-15): all snakes
   sharing one policy -> Red Queen gains; `avg_food` = flow not mass; the S1->S3 "2.4x" was an illusion.
   Always evaluate hero vs frozen/scripted opponents with paired seeds, >= 2 opponent types.
2. **Censored survival**: respawn + frame cap saturate length; `mean_mass_alive` let dying rich outrank
   surviving (`AGENTS.md` Promotion Gate; blueprint section 5.1). Use mass integral with dead frames = 0
   and terminal hero.
3. **Self-play rosters are ungated**: Watch all-B8 and Play rosters were never part of T7's mixes; their
   death rates are monitoring baselines, not skill claims (`docs/research/b8_s3_release_2026-10-17.md`).
4. **Mac vs x86 float ties** (memory `mac-pod-float-ties.md`; status note above): exact float32 Q ties
   flip argmax between Accelerate and MKL (~1/50 H10000 episodes). FRP-v5-H's 54/54 was luck. Bitwise
   identity across platforms is not a valid gate without the tie-explaining check.
5. **Duplicate Drive folders**: two Colab runtimes doing `mkdir -p` on the same shared dir produced
   `evaluation` and `evaluation (1)` (`research/t23_r0_screen_20261017/RESULTS_T23.md`). Pre-create shared
   dirs on the Mac (`research/colab_queue/q.py enqueue`), never from workers.
6. **One-core evaluation on Colab**: the Q0 mass audit ran 480 H5000 episodes serially and hit its 4 h
   timeout; use a process pool with deterministic merge order (`Q0_PARALLEL_R5_STATUS_20261018.md`).
7. **Evaluation noise at small N**: 32 scripted worlds per run vary about +-0.1 in scripted survival; use
   >= 96 scripted worlds per seed for survival questions (`docs/research/m6_result_2026-10-14.md`).
8. **Guard comparator drift**: M4 failed only the scripted-survival guard vs champion_a5+v8 (-0.156 on one
   bank with 19/32 deaths); the fix was one prospective amendment (B+C), fresh banks, never re-scoring the
   look-2 or census data (`guard_clause_amendment_2026-10-15.md`). Do not re-score old banks as evidence.
9. **Best-checkpoint swaps / seed replacement** after data are forbidden (T23, Q1 rules); a tripwired seed
   is a KILL for that arm with no replacement.
10. **Veto confusion**: B8 consumes v8-derived *input features* and was trained with v8 labels but is served
    with **no veto**; "candidate+v8" and bare candidate are different arms. The redesign goal was beat
    frp3-s12+v8 *without* a veto. Do not describe the inspector's "v8 veto features" group as an active veto.
11. **CNN "lost" claim is false** (`AGENTS.md` Architecture history): the CNN was never trained; do not cite it.
12. **Fixed-N stopped early is invalid** (rule of 2026-10-02); interim peeks at outcome metrics in a
    one-look study are protocol violations even if "just monitoring".

## Before you say done
- Name the exact comparator (frp3-s12+v8 served agent? bare B8 `3e649783...`? champion_a5+v8?) and the
  metric for every number you report; give estimate, SE/LB90, threshold, PASS/FAIL per clause.
- State tier and label exactly as registered; say explicitly that a screen result confers no promotion.
- Confirm the bank namespace is new, the seed audit passed and is committed, and worlds are paired.
- Confirm the platform (Mac for strict/serving) and thread settings, and that no outcome was looked at
  before the registered look.
