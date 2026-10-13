# M4 pre-registration (Tier-1 Phase R of the M4-A students)

**Status: RATIFIED (section 12).** `phase_r_m4.py` refuses a real (non-smoke) plan without the
ratification line. Development / Tier-1 only: nothing here promotes; a GO buys LH-1,
the strict sequential gate and Mac serving qualification.

Design and training: `docs/research/m4a_design_2026-10-12.md`; pod plans
`research/redesign_m4a_20261012/plans/m4a_pod_plan.json` (C; D / S superseded) and
`m4a2_pod_plan.json` (D, S with the numba v8 backend; B8). The Phase R machinery is M3-C's
(`research/redesign_m3c_20261008/PREREGISTRATION_M3C.md`, ratified 2026-10-08), forked with
the changes listed in each section.

## 1. Question

Does an M4-A student (ego2s-b or ego2s-bv8, RL from the M2b student with the M4-A setup fixes)
**without any veto** beat **I = frp3-s12 + v8** on Tier-1 H5000 (primary; the standing goal of
2026-10-08), and, wrapped in v8, beat I (secondary)? Which M4-A change moves the no-veto score
(descriptive contrasts)?

## 2. Training (fixed before this file; not part of the decision)

`research/redesign_m3c_20261008/train_m3c.py` with the per-run flags of the pod plans,
20M transitions, training worlds `redesign-m3-train/v1` key `seed<s>`. Arm C has **five
seeds**: 0, 1, 2 on the M4-A pod (commit `f518763`, CUDA learner) and 3, 4 on the Mac (same
commit and flags, MPS learner; `snake-dqn-artifacts/m4a-mac-c34.sh`, outputs
`redesign-m4a-mac-c34-20261013/C_s<seed>/`). The seed-3/4 platform difference (learner device,
actor timing) is a disclosed deviation; training is not bit-reproducible on either platform, so
every seed is an independent replicate of the same recipe. `final.pth` = 20M; a HALTed run
writes `aborted.pth` and is ineligible.

## 3. Arms

| Arm | Beyond the common M4-A setup (mass reward κ 1, kill 0, γ→0.998, 25 % mixed worlds, v8-margin long anchor) | Obs | Seeds trained by selection time |
|---|---|---|---|
| C | — | ego2s-b | 0–4 |
| D | on-policy v8 labels | ego2s-b | none (pod capacity; M4-B) |
| S | D + training-time v8 shield | ego2s-b | none |
| B8 | v8 features as inputs | ego2s-bv8 | 0 (Mac screen, descriptive) |

Only arms with all five seeds can enter a Phase R; at selection time that is **C only**. D, S
and B8 are descriptive M4-B screens and cannot enter this study.

## 4. Dev probes and selection (Mac)

Probes: `probe_watch.py --schedule final --worlds 32 --vetoes none` (round 52, Phase R hero
modes: `--ego2s-forward rowwise --ego2s-mask advisory`), files
`<probes>/<arm>_s<seed>__final__none.jsonl` in `redesign-m4a-probes-20261012` (C) and
`redesign-m4a2-probes-20261012` (D, S, B8). Reference I: M3-C's
`redesign-m3c-probes-20261008-recovery/incumbent__dev__v8.jsonl` (round 52, 32 worlds per mix,
stratified 434.9); the selection checks that its per-mix world set **equals** every final
probe's (else nothing is selected; a matching `--incumbent` must be passed). The secondary
needs v8 final probes (`--vetoes v8`), which the running watchers do not make: without them the
secondary is "not entered" (a later v8 watcher run on the same 32 worlds may supply them before
selection).

Selection: `m4_select.py` (M3-C's `m3c_select.py` rules, 5 seeds; tests
`tests/test_m4_select.py`):

* a seed is eligible iff `summary.json` has `halt = null` and ≥ 20M transitions, `final.pth`
  exists, and its no-veto final probe is complete, made with that `final.pth` (sha), round 52,
  in the Phase R hero modes, and found in exactly one probe / run directory;
* arms are compared on the seeds eligible in every arm (≥ 2, else nothing is selected);
  score(arm, veto) = mean over those seeds of the mix-stratified mean mass;
* **primary arm** = argmax no-veto score (ties: C, D, S, B8); **enters the primary Phase R iff
  all 5 of its seeds are eligible AND score_nv ≥ I − 50 AND, in every mix, the world-paired
  survival difference has mean ≥ −0.03 − 1.28 SE** (as M3-C);
* **secondary arm** = argmax v8 score (only when every arm has complete v8 final probes);
  **enters iff all 5 seeds eligible AND score_v8 − I ≥ +30 AND the same survival check**;
* reported (descriptive): D − C (on-policy labels), S − D (shield), B8 − D (v8 features as
  inputs), seed-paired, df = n − 1 intervals.

Selection uses the probe worlds (round 52), disjoint from the Phase R banks; the Phase R
estimate is therefore not biased by the selection, but the selected arm is the max of four.

## 5. Platform and compute

**Mac only** (strict-on-RunPod not ratified; Phase R plays the Mac like M3 / M3-C), **2 shared
CPU slots** (strict-gate compute policy) via `slotrun.py` (default `--procs 2` in `phase_r_m4.py`), AC + lid guard.

Estimate from measured Mac unit walls (M3 Phase R 2026-10-08: candidate + v8 25.5 s / world at
H10000, incumbent 22.7–23.6, champion 19.6–21.3, M2b + v8 27.0; looks of 2.75 unit-h ran in
57 min on 3 slots; M3-C nv smoke: no-veto student 14.6, M2b no veto 12.5) and the M4 smoke
(section 10). Per look, 5 seeds: candidate 165 worlds, incumbent 165, champion 33, control 33
(+ 21 H5000 prefix controls in look 0):

| Study | per look (unit-h) | to the final look (unit-h) | wall, 2 slots |
|---|---|---|---|
| nv (candidate 15–25 s / world) | 2.1–2.6 | 6.4–7.8 | 3.2–3.9 h |
| v8 | 2.9 | 8.6 | 4.3 h |

Larger bodies (longer survival) raise the candidate's per-world cost; slots shared with the
probe watchers lengthen wall time. An early stop at look 0 / 1 costs ⅓ / ⅔.

## 6. Prerequisites for each Phase R

* **ego2s-b candidate (C, D, S):** as M3-C §7 — P2-NV (`p2_noveto.py --arm none --worlds 3`,
  9 worlds H5000, and `p2_noveto.py h10000`, 3 worlds) for the primary; M3's P2-v8 +
  `p2_h10000.py` for the secondary; every candidate seed, all identical; evidence files
  `<artifacts>/redesign-m4-p2-candidates/{p2_noveto,p2_noveto_h10000,p2_v8,p2_h10000}_seed<s>.json`.
  Checked 2026-10-12: `p2_noveto.py --worlds 1` on C_s0 `ckpt_5000k.pth` (sha `948cffb8…`):
  3 / 3 identical (one world per mix).
* **ego2s-bv8 candidate (B8): P2 cannot run.** `ego2s-bv8` is SIMD-only by design
  (`src/model/obs_spec.py` `EGO2S_SIMD_ONLY_SPECS`; the live loader refuses the spec; there is
  no live bv8 serving path). Its features are computed from the grid simulator and its SIMD
  Phase R hero modes work (checked 2026-10-12: a Q-identical widened bv8 copy of C_s0@5M
  reproduces the ego2s-b SIMD record exactly with the advisory mask and with v8).
  **Decided (a):** B8 may not enter a Phase R.
* Candidate sha256s and the candidates' obs spec (read from the checkpoints; all 5 seeds must
  agree) into each `plan.json` before the first shard.

## 7. Tier-1: sequential Phase R (method `sequential-phase-r-obf-hk-v1`)

Rule specs `rule_spec_m4.py` (FRP-v4 rule in sequential form, NO_GO G = 50 opted in, cap 600,
N = 32 per seed × mix, looks 11 / 22 / 32, seeds 0–4; pins in `tests/test_m4_rule_spec.py`):

| Study | Hero | Baseline | Bank namespace |
|---|---|---|---|
| `redesign-m4-phase-r-nv` (primary) | selected arm, no veto, `ego2s_mask="advisory"`, `ego2s_forward="rowwise"` | frp3-s12 + v8 | `redesign-m4-phase-r-nv/v1` |
| `redesign-m4-phase-r-v8` (secondary) | selected arm + v8 | frp3-s12 + v8 | `redesign-m4-phase-r-v8/v1` |

**No departure from `frp_v4_like`** (five seeds, clause (4) = 4 of 5; tested).

Harness `phase_r_m4.py --study nv|v8`: a private copy of the ratified M3-C harness
(`phase_r_m3c.py`, sha `3a7658fd…3e0f`, not edited) with the study globals rebound (rule
specs, record schema `redesign-m4-phase-r-record/v1`, smoke namespace
`redesign-m4-phase-r-smoke/v1`, this file, P2 directory) and its own `plan` (obs-spec binding,
bv8 rule above). The candidate and the M2b control arm play without a veto in the nv study;
the incumbent and the champion always with v8. A real plan refuses unless this file has the
ratification line and the `--candidates` are the 5 `final.pth` of the arm the `--selection`
JSON (`m4_select.py`) entered for that study. Guard: hero vs champion_a5 + v8 on seed 0's
bank; prefix controls as M3; create-only plan / receipts, look gate, `python -I` audit must
PASS; stops final. The primary runs first when both enter; each study has its own α as
ratified for the method.

Pins: primary rule `f614b80639e1b9b29f32a08ad38426e8b87644d5720cfe54a89d943a66f5e6b4`, plan
`72097708d03e81b4d4b56bc2ea013df4736b9b9f6c0c727ef2a1bb938f6f42d2`; secondary rule
`e53691a6907e37294f9b720e7959d3e13650539f1501dd0a78d36bf92f865c46`, plan
`85d20b3a67de87ad24fbeca9cd4c2f3fe9888e14c5bb2e6fab30c6a53d22b5ea`.

## 8. Operating characteristics

Five seeds and the unchanged `frp_v4_like` rule: the method's ratified 5-seed operating
characteristics apply (`research/sequential_phase_r/oc_results_20261007.json`; GO_R at +65:
0.60–0.71; efficacy type-I error far below 0.10). The 3-seed re-run (`oc_m4.py`,
`results/oc_m4.json`) is kept for reference only.

## 9. What each outcome means

* Primary GO_R → an M4 student beats frp3-s12 + v8 without a veto at Tier 1: LH-1, the strict
  gate and serving qualification of the student alone (B8 only after live bv8 serving + P2,
  section 6).
* GO_R_UNGATEABLE → efficacy shown but clause (7) (feasibility) fails: no strict gate yet;
  more seeds / worlds decided by the owner.
* Primary NOT ENTERED / KILL_V4 / NO_GO_EARLY / PARTIAL → the goal is not met at Tier 1; the
  arm contrasts and probes inform M4-B / M5.
* Secondary GO_R → a with-veto improvement over I (a deployable fallback), not the goal.
* After an early stop: point estimates biased upward, final-only labels unresolved
  (method disclosure).

## 10. Harness smoke (DRAFT evidence, not a decision)

2026-10-12/13, commit `6afd249`, `results/phase_r_smoke_nv.json`: `phase_r_m4.py plan --study
nv --smoke` (bank `redesign-m4-phase-r-smoke/v1`, N = 4) with M4-A C_s0 `ckpt_5000k.pth`
(ego2s-b) standing in for all 3 seeds, then look 0 on 2 slots: 45 units, 69 records (schema
`redesign-m4-phase-r-record/v1`; candidate and M2b control without a veto and with the advisory
mask; incumbent and champion with v8), prefix controls 21 / 21 identical, decision KILL_V4
(dev only: 2 worlds per cell and a 5M stand-in), `python -I` audit **PASS**. Wall 11 min for
0.34 unit-h. Measured s / world (H10000): candidate 16.3, incumbent 25.2, champion 23.3, M2b
12.9; H5000 controls 10.1–13.8. With these, the nv study is ≈ 1.5 unit-h per look, 4.4 to the
final look: ≈ 2.4 h on 2 slots, ≈ 1.5 h on 3 (final-checkpoint students that live longer cost
more; section 5's range).

## 11. Pinned code

The ratification commit pins every file of `research/redesign_m4_20261013/` (`rule_spec_m4.py`,
`m4_select.py`, `phase_r_m4.py`, `oc_m4.py`), the base harness
`research/redesign_m3c_20261008/phase_r_m3c.py` (sha checked at import), `m3c_select.py`,
`p2_noveto.py`, `research/redesign_m3_20261008/{slotrun.py,student_record_parity.py,p2_h10000.py}`,
`src/evaluation/sequential_phase_r.py`, `research/sequential_phase_r/`,
`src/simd_env/{eval_engine.py,ego2s_policy.py,ego_raster.py,ego_raster_bv8.py,v8_nb.py}` and
`research/redesign_m2_20261008/dev_probe.py`. Any later change before the step it governs is a
recorded deviation.

## 12. Ratification

Decisions (owner's standing decision authority for the improvement loop, 2026-10-12, exercised
by the agent): (1) five seeds — arm C seeds 3, 4 trained on the Mac before selection, so the
method's native 5-seed rule applies; (2) clause (4) unchanged (4 of 5); (3) B8 / bv8 option (a):
no Phase R for B8; (4) 2 slots.

**RATIFIED 2026-10-13** under the standing decision authority above. Selection runs only after
all five C finals are probed; nothing in this file changes after the selection is computed.

## 13. Deviations (recorded after ratification)

* **2026-10-13, before any Phase R shard:** the first `m4_select.py` run used a leftover
  `SEEDS = (0, 1, 2)` from the 3-seed draft (contradicting section 2's five seeds and
  `rule_spec_m4.TRAINING_SEEDS`). Fixed to `rule_spec_m4.TRAINING_SEEDS`; the selection was
  re-run on all five seeds. No rule, threshold or probe changed. The 3-seed output is kept as
  `selection-3seed-bug.json` for the record.
