# M4 pre-registration (Tier-1 Phase R of the M4-A students)

**Status: DRAFT — NOT RATIFIED.** The human-side owner ratifies by adding a line that starts
with the bold word RATIFIED and a date (section 12); `phase_r_m4.py` refuses a real (non-smoke)
plan until that line exists. Development / Tier-1 only: nothing here promotes; a GO buys LH-1,
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

`research/redesign_m3c_20261008/train_m3c.py` with the per-run flags of the two pod plans,
20M transitions, seeds 0, 1, 2 per arm, training worlds `redesign-m3-train/v1` key
`seed<s>`. `final.pth` = 20M; a HALTed run writes `aborted.pth` and is ineligible.
Checkpoints are pulled sha-checked to `<runner>/attempt<NN>/out/<arm>_s<seed>/`.

## 3. Arms

| Arm | Runner | Beyond the common M4-A setup (mass reward κ 1, kill 0, γ→0.998, 25 % mixed worlds, v8-margin long anchor) | Obs |
|---|---|---|---|
| C | pod 1 (`redesign-m4a-gpu-20261012`) | — | ego2s-b |
| D | pod 2 (`redesign-m4a2-gpu-20261012`) | on-policy v8 labels, numba v8 | ego2s-b |
| S | pod 2 | D + training-time v8 shield (p 1 → 0, 8M → 16M) | ego2s-b |
| B8 | pod 2 | D's labels + v8 features as inputs (`--obs-version bv8 --label-frac 1.0`) | **ego2s-bv8** |

Pod 1's D / S runs (Python v8, too slow to finish) are not candidates; a D / S run present in
both runners' output directories is "ambiguous" and ineligible (section 4).

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

Selection: `m4_select.py` (M3-C's `m3c_select.py` rules with 3 seeds; tests
`tests/test_m4_select.py`):

* a seed is eligible iff `summary.json` has `halt = null` and ≥ 20M transitions, `final.pth`
  exists, and its no-veto final probe is complete, made with that `final.pth` (sha), round 52,
  in the Phase R hero modes, and found in exactly one probe / run directory;
* arms are compared on the seeds eligible in every arm (≥ 2, else nothing is selected);
  score(arm, veto) = mean over those seeds of the mix-stratified mean mass;
* **primary arm** = argmax no-veto score (ties: C, D, S, B8); **enters the primary Phase R iff
  all 3 of its seeds are eligible AND score_nv ≥ I − 50 AND, in every mix, the world-paired
  survival difference has mean ≥ −0.03 − 1.28 SE** (as M3-C);
* **secondary arm** = argmax v8 score (only when every arm has complete v8 final probes);
  **enters iff all 3 seeds eligible AND score_v8 − I ≥ +30 AND the same survival check**;
* reported (descriptive): D − C (on-policy labels), S − D (shield), B8 − D (v8 features as
  inputs), seed-paired, df = n − 1 intervals.

Selection uses the probe worlds (round 52), disjoint from the Phase R banks; the Phase R
estimate is therefore not biased by the selection, but the selected arm is the max of four.

## 5. Platform and compute

**Mac only** (strict-on-RunPod not ratified; Phase R plays the Mac like M3 / M3-C), ≤ 3 shared
CPU slots via `slotrun.py` (default `--procs 2` in `phase_r_m4.py`), AC + lid guard.

Estimate from measured Mac unit walls (M3 Phase R 2026-10-08: candidate + v8 25.5 s / world at
H10000, incumbent 22.7–23.6, champion 19.6–21.3, M2b + v8 27.0; looks of 2.75 unit-h ran in
57 min on 3 slots; M3-C nv smoke: no-veto student 14.6, M2b no veto 12.5) and the M4 smoke
(section 10). Per look, 3 seeds: candidate 99 worlds, incumbent 99, champion 33, control 33
(+ 21 H5000 prefix controls in look 0):

| Study | per look (unit-h) | to the final look (unit-h) | wall, 3 slots | wall, 2 slots |
|---|---|---|---|---|
| nv (candidate 15–25 s / world) | 1.3–1.7 | 4.0–4.8 | 1.4–1.7 h | 2.0–2.5 h |
| v8 | 1.8 | 5.3 | 1.8 h | 2.7 h |

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
  **OWNER DECISION (open):** (a) B8 may not enter a Phase R (selection then re-runs over C, D, S
  if B8 is selected), or (b) B8 runs a **SIMD-only** Phase R (`phase_r_m4.py plan
  --bv8-simd-only`, recorded in `plan.json` as "P2 not applicable"); a GO_R then means a Tier-1
  SIMD result only, and live bv8 serving + P2 + LH-1 must come before any strict gate.
  Default if not decided at ratification: (a).
* Candidate sha256s and the candidates' obs spec (read from the checkpoints; all 3 seeds must
  agree) into each `plan.json` before the first shard.

## 7. Tier-1: sequential Phase R (method `sequential-phase-r-obf-hk-v1`)

Rule specs `rule_spec_m4.py` (FRP-v4 rule in sequential form, NO_GO G = 50 opted in, cap 600,
N = 32 per seed × mix, looks 11 / 22 / 32, seeds 0, 1, 2; pins in `tests/test_m4_rule_spec.py`):

| Study | Hero | Baseline | Bank namespace |
|---|---|---|---|
| `redesign-m4-phase-r-nv` (primary) | selected arm, no veto, `ego2s_mask="advisory"`, `ego2s_forward="rowwise"` | frp3-s12 + v8 | `redesign-m4-phase-r-nv/v1` |
| `redesign-m4-phase-r-v8` (secondary) | selected arm + v8 | frp3-s12 + v8 | `redesign-m4-phase-r-v8/v1` |

**Departure from `frp_v4_like`:** clause (4) positive seeds = **3 of 3** (`frp_v4_like` gives
6 for any seed count other than 5, unreachable with 3 seeds; 4/5 = 80 % and 6/8 = 75 % in the
earlier families, 2/3 = 67 % would be looser). Everything else is `frp_v4_like` (tested).

Harness `phase_r_m4.py --study nv|v8`: a private copy of the ratified M3-C harness
(`phase_r_m3c.py`, sha `3a7658fd…3e0f`, not edited) with the study globals rebound (rule
specs, record schema `redesign-m4-phase-r-record/v1`, smoke namespace
`redesign-m4-phase-r-smoke/v1`, this file, P2 directory) and its own `plan` (obs-spec binding,
bv8 rule above). The candidate and the M2b control arm play without a veto in the nv study;
the incumbent and the champion always with v8. A real plan refuses unless this file has the
ratification line and the `--candidates` are the 3 `final.pth` of the arm the `--selection`
JSON (`m4_select.py`) entered for that study. Guard: hero vs champion_a5 + v8 on seed 0's
bank; prefix controls as M3; create-only plan / receipts, look gate, `python -I` audit must
PASS; stops final. The primary runs first when both enter; each study has its own α as
ratified for the method.

Pins (DRAFT, re-pinned or confirmed at ratification): primary rule
`67dfe274de90858e4e0cd8d2e3b9bd6e81873bae919e7c838a513a6aeaedf08c`, plan
`c8f3595f029d11c91ccbddf69087ca2cc32bb41544a02610df7a411f4da7c36a`; secondary rule
`e842a5237f756d17780aaa899e41c4dde80e4a0314df5728b03d6816cae86fe8`, plan
`c4d8095d3e358b8703c32b713abd9e3899318af54938e1bcc58ef058ee907a39`.

## 8. Operating characteristics (governance condition 7: 3 seeds)

`oc_m4.py --reps 10000` (the method's simulator and variance pool on this rule; results
`results/oc_m4.json`), pre-registered design (sequential + NO_GO 50), vs the 5-seed v4 family
(`research/sequential_phase_r/oc_results_20261007.json`, same design):

| effect (H5000 mass vs I) | τ² | GO_R (3 seeds) | GO_R (5 seeds) | KILL_V4 | NO_GO | E[fraction] |
|---|---|---|---|---|---|---|
| −20 | 0 | 0.000 | 0.000 | 0.155 | 0.694 | 0.60 |
| 0 | 0 | 0.000 (UNGATEABLE 0.012) | 0.000 (0.011) | 0.028 | 0.473 | 0.79 |
| +20 | 0 | 0.004 | 0.001 | 0.002 | 0.126 | 0.95 |
| +40 | 0 | 0.067 | 0.070 | 0.000 | 0.014 | 0.98 |
| +65 | 0 | 0.513 | 0.713 | 0.000 | 0.000 | 0.95 |
| +65 | 500 | 0.462 | 0.597 | 0.000 | 0.001 | 0.96 |
| +80 | 0 | 0.761 | — | 0.000 | 0.000 | 0.93 |
| +100 | 0 | 0.872 | — | 0.000 | 0.000 | 0.90 |

Reading: efficacy type-I error stays far below 0.10 (GO_R + UNGATEABLE ≤ 0.026 at 0); the cost
of 3 seeds is **power** (GO_R at +65: 0.46–0.51 vs 0.60–0.71) and a weaker KILL (the HK bound
has 2 df: a null candidate mostly ends NO_GO rather than KILL_V4); a +80 / +100 effect is
detected with 0.68–0.87. Harm scenarios (survival −0.05..−0.10 in one mix) block GO as in the
5-seed family. **Accepted as a DRAFT trade-off** (three seeds were the trained design); the
owner may instead require more seeds before ratifying.

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

SMOKE_PLACEHOLDER

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

Open owner decisions: (1) accept 3 seeds with the section 8 power (or add seeds);
(2) clause (4) = 3 of 3; (3) B8 / bv8 option (a) or (b), section 6; (4) slots (2 or 3).

(Not ratified. The owner adds the ratification line here.)
