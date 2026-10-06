# FRP-v3 seed-12 M3@60000 + v8 vs champion + v8: Tier-2 group-sequential strict challenge (champion change; pre-registration)

Method `strict-sequential-obf-bonferroni-v1`, runner `sequential-strict-template/v2-pooled`
(survival band v2) executed as `sequential-strict-template/v3` (RunPod serverless, Mac fallback).
Governance: Tier 2 of
[`governance_tiers_2026-09-26.md`](../../docs/research/governance_tiers_2026-09-26.md) as amended by
[`governance_amendment_sequential_gates_2026-10-02.md`](../../docs/research/governance_amendment_sequential_gates_2026-10-02.md),
[`governance_amendment_paired_bands_2026-10-03.md`](../../docs/research/governance_amendment_paired_bands_2026-10-03.md)
(ratified 2026-10-03),
[`governance_amendment_strict_on_runpod_2026-10-05.md`](../../docs/research/governance_amendment_strict_on_runpod_2026-10-05.md)
and
[`governance_amendment_survival_band_v2_2026-10-06.md`](../../docs/research/governance_amendment_survival_band_v2_2026-10-06.md)
(**both ratifications pending; the owner ratifies both before `prepare`, which refuses
otherwise**; survival band v2 must be ratified with **option 1**, the values below).
Written 2026-10-05 (UTC) after FRP-v3 Phase R closed `GO_R` and **before any calibration or
final episode of this study and before any intent**. **Revised 2026-10-06 (UTC), still before
any intent, calibration or final episode** (ledger: namespace unused): the survival band was
replaced by survival band v2 at the owner's decision, because the frozen template-v2 band
(`paired_ni_at_stop`, per-mix, judged once at the qualifying look) failed 53-72% of
no-regression runs at the qualifying look for a between-checkpoint change. Survival band v2 was
chosen and calibrated on independent, pre-FRP-v3 data only (the amendment's "Data
provenance"); this study's Phase R pool enters only its validity check and the descriptive
planning numbers below. Worlds, namespaces, looks, N_max, MDE, efficacy, NI, futility, sizing,
arms and the remote configuration are unchanged; the superseded band artifacts are kept in
`superseded_v2_band/` (and the first version of this protocol at commit `4eb75af`).

Execution platform: runpod-serverless

Files: `spec.py` (the `StudySpec`), `preflight.py` (world-seed disjointness), `preregistration.py`
(generates the inputs below from Phase R), `phase_r_deltas.json` (skew-check input),
`paired_pool.json` (paired-band check pool), `frozen_plan.json` (the plan the intent freezes),
`operating_characteristics.json` (OC report), `look1_band_cost.json` (survival band v2 cost per
look), `paired_band_check.json` (survival band v2's per-study check,
`research/survival_band_v2_20261006/simulate.py --part study`), `remote_config.json` (template
v3 remote configuration), `superseded_v2_band/` (the template-v2 band's checks, never used),
tests `tests/test_frp3_strict.py`.

## Question and decision it informs

Does replacing the served Watch hero's checkpoint, `champion_a5_freespace_20260621.pth`, with the
FRP-v3 fine-tune `apex_mark_u60000.pth` (arm M3, training seed 12, 60k updates), **with the
released v8 veto (lambda 8) unchanged on both**, improve the H5000 mass integral strictly
(OBF-sequential Bonferroni superiority in >= 2 of 3 mixes, scripted non-inferiority, survival
band v2) on fresh worlds? This is a **champion change**: under the governance tiers only
a Tier-2 STRICT_PASS followed by a Mac serving qualification can change the served agent.
`STRICT_PASS` allows the separate release action below to be considered; `STRICT_FAIL` keeps the
champion and does not show that the candidate is worse.

## Arms and identities

| Role | Hero checkpoint | Wrapper (hero only; opponents unwrapped) |
|---|---|---|
| Incumbent (released) | `champion_a5_freespace_20260621.pth` `43d4e2c5…d747ac93` | `free-space-veto/v8-space-and-head(lambda=8.0)` |
| Candidate | FRP-v3 `arm-M3/seed-12/checkpoints/apex_mark_u60000.pth` `eec144bf…5dd3723` | the same v8, same bytes |

Veto modules (both arms, pinned): `safety_veto_v8.py` `faf3695f…e4ac05`, `_v7.py` `56ff7009…7e2980`,
`_v6.py` `a6117cb9…1757d5`, `_v5.py` `d86d084e…0ec86c`, `_v4.py` `3f0881af…e44578`, `_v3.py`
`ed3a6d86…b5be1`, `safety_veto.py` `1b62d15c…c428`.

`spec.arm_identities()` computes every sha256 from the files and the descriptor from the class,
and fails closed unless:

- **incumbent = the v8 STRICT_PASS receipt's candidate** (`apex-veto-v8-strict-20261003/run-v1/
  output/receipt.json`, sha256 `29b1f7f6…2b7507`, the receipt `web/backend/safety_veto_serving.py`
  pins for the served default since `a0cd04b`): method, descriptor, checkpoint and the seven
  veto source sha256s;
- **candidate** checkpoint sha256 = `eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723`,
  and the FRP-v3 Phase R summary (`frp-v3-20261005/phaseR/rp-v1/merged/summary.json`, sha256
  `0fcb80f6…3ab206`) is a real `GO_R` whose pre-declared candidate is exactly this checkpoint
  (seed 12, highest one-sided 90% lower bound);
- **LH-1 precondition**: the LH-1 screen receipt of this candidate,
  `snake-dqn-artifacts/lh1-screen-frp3-s12/run-v1/receipt.json` (written by
  `research/longh_screen/lh1.py analyze`, branch `lh1-screen`, job `lh1-frp3-s12`, job file
  sha256 `6a52cee4…0be9ff` at `c9adbc3`, job commit `bc48efa6`), **exists and reads `CLEAR`**:
  `kind = screen`, the pinned job id / job sha256 / repo commit, no missing episodes, no problems,
  no decision-code drift, prefix controls verified, one contrast C1 vs I with outcome `CLEAR`.
  `REASON_REQUIRED`, `BLOCK`, `INCOMPLETE`, `INVALID`, a partial (`receipt.partial.json`) or a
  missing receipt refuses, and so does any twin beside it (`receipt.<ts>.json` from a
  re-analysis, or `receipt.partial.json`). The receipt bytes are **pinned**: sha256
  `f9879836e1c478906f346dbd8dc48b73f0faa22932364cf011d0a9152ae0044a` (written 2026-10-05, outcome
  `CLEAR`, analysis commit `c9adbc3`), and the sha256 is frozen in the candidate identity
  (`screen_binding`), so `prepare` refuses any other receipt and any later change to its bytes
  stops the run before the next segment. (Descriptive, LH-1 C1 per mix at H5000 over 48 worlds:
  frozen +76.9, scripted +72.5, **mixed -0.4**, pooled +49.7 [90% CI 21.4, 78.0]; Phase R had
  the candidate's mixed mix at +130.5. Not used for sizing.)

Both arms differ ONLY in the hero checkpoint. Each record carries `hero_checkpoint_sha256`:
the sha256 the arm's frozen identity names (refused unless it is the arm's pinned checkpoint),
whose file the episode loads through the lookup after `verify_checkpoints` re-hashed it against
that sha256; `validate_record` checks the field against the arm. It is a binding of what was
loaded, not a hash computed inside `rollout`. Install: `dev_screen.hero_veto_installer(
install_space_and_head_veto(hero, 8.0))` around `tournament_eval.rollout(hero_safety_veto=True)`
(the built-in vector61 guard runs first), exactly as the v8 strict package. Every episode
re-hashes all five checkpoints (four pool checkpoints, champion included, and the candidate)
before `rollout` loads them, validates the record with
`strict_promotion.validate_strict_world_record(..., candidate_wrapper=<v8 descriptor>)` (profile
`promotion-v2-watch-rect`, digest `d396d3ed…0e8b`), and attaches the v8 `veto_diagnostics`
(reported, not gated). The identities are frozen in the intent and recomputed before every
segment (drift stops the run).

## Profile, mixes, rosters, worlds

- Profile `promotion-v2-watch-rect`, H5000, live engine (`tournament_eval.rollout`); config
  `research/apex_safety_20260926/deployment.yaml` (sha256 `4146baa3…715aa5`, checked by every
  worker).
- Mixes frozen, scripted, mixed; rosters from `dev_screen._design_rows` (strict balanced, slot
  members rotating by world index; the opponent pool is unchanged: champion and the three
  `best_apex*` checkpoints, plus the scripted anchors). A bank prefix has the prefix rows.
- Seeds `uint32_be(sha256("<domain>|worlds|<i>")[:4])`:

| Namespace | Count | Use |
|---|---|---|
| `apex-frp3-strict-dev-v1` | 16 | incumbent-only calibration, 3 mixes |
| `apex-frp3-strict-final-v1` | ordered bank of N_max = 275 | final paired worlds, world-major round robin |

`excluded_seeds()` (= `preflight.excluded_seeds()`, checked fail closed by `prepare`; names
recorded in the intent) is the union of: the v8 strict package's exclusions (first 1000 seeds of
every earlier domain/purpose, challenger namespaces, strict pilot observed seeds, 0..999, saved
v7/v8 seeds); the first 1000 seeds of every domain in `NEWER_DOMAINS` (v8 strict banks, v8
serving/census, v9, growth ceiling, reference/long-horizon banks, LH-1 calibration **and the
LH-1 screen of this candidate**, every FRP-v2 and FRP-v3 domain) and of every `*DOMAIN*`/`*_ID`
literal on **every local branch**; **FRP-v3 and FRP-v2 analytically** (every Phase R / phase /
extension evaluation bank and fidelity bank with its real purposes, the first 20,000 indices of
every training and hazard stream — the candidate's own training worlds — and both studies'
`training_exclusions.json`, sha-checked against their namespace reports); and every integer
under a `*seed*` JSON key in every artifact root (regex fallback for large or unparseable files;
this study's own root **included**, `*.jsonl` files line by line). The gate's own namespaces are
never dropped: if any branch declares one of them as a domain literal, the preflight raises.
`preflight.py` checks the calibration bank and the final bank up to 300 worlds and writes the
report; at commit time (full scan) the banks were disjoint from 2.41 M excluded seeds in 584
sets.

## Calibration (incumbent only, before any final world)

16 worlds x 3 mixes (48 champion+v8 episodes). `delta_NI = 0.03 x calibration incumbent scripted
mean mass_integral` (the v7/v8 rule; development value 9.946 = 0.03 x the Phase R incumbent
scripted mean over 160 worlds). Paired bands have no reference; calibration survival means are
descriptive only. Written create-only to `calibration.json`.

## Plan (frozen in the intent; `frozen_plan.json`)

- N_max = **275** worlds per mix; looks at fractions 0.25/0.5/0.75/1.0 -> sizes **69, 138, 207,
  275**.
- Efficacy per mix: one-sided H0 delta <= 0, Lan-DeMets OBF at 0.05/3, boundaries (z) 4.6380, 3.1843, 2.5392, 2.1739 (nominal p 1.76e-6, 7.26e-4, 5.56e-3, 1.486e-2).
  Required successes 2 of 3 (a crossed mix stays crossed).
- Scripted NI: own OBF at 0.05, boundaries (z) 3.7422, 2.5346, 2.0117, 1.7210 (nominal p 9.12e-5, 5.63e-3, 2.212e-2, 4.262e-2); NI when `mean - t * se > -delta_NI`.
- **Bands: survival band v2, `pooled_ni_continue`** (amendment 2026-10-06, option 1; ratification
  pending, `prepare` refuses until it is ratified with option 1). With `d = candidate -
  incumbent` survival_fraction on the same final world and the pooled per-world delta `p` = the
  mean of `d` over the three mixes, the bands pass at look k iff `mean(p) - t_{n_k-1}(p_k) *
  sd(p) / sqrt(n_k) > -0.05` (pooled NI), every mix's `mean(d_m) - t_{n_k-1}(p_k) * sd(d_m) /
  sqrt(n_k) > -0.10` (per-mix catastrophic NI) and every mix's candidate mean >= 0.30 (floor),
  with `p_k` the `rci_obf` levels at alpha 0.05 (9.12e-5, 5.63e-3, 2.21e-2, 4.26e-2). Judged at
  the qualifying look **and every later look**: STOP_PASS at the first look where they pass; a
  qualified run whose bands fail gets `CONTINUE_BANDS` and plays the next look (efficacy and NI
  stay established, futility no longer stops it); FINAL_PASS / FINAL_FAIL at look 4. Stated
  explicitly: `--paired-band 0.05,0.05,rci_obf,0.30,0.10`. Guarantee (repeated confidence
  bounds): P(PASS and a true pooled survival regression >= 0.05) <= 0.05 and P(PASS and a true
  one-mix regression >= 0.10) <= 0.05 at any stopping look; a 0.05 loss confined to one mix is
  not protected (not attainable at N_max <= 300; see the amendment).
- Futility: non-binding CP < 0.10 under the MDE, `futility_policy = "followed"`,
  `futility_action = "stop"`.
- MDE **65** mass-integral units per mix (sizing below). Interleaving: unit `u = world x 3 +
  mix`, worker `u % 2` runs both arms (incumbent first) on the Mac; on RunPod a unit (both arms
  of one world) is one job on one worker. Template default caps (calibration 1800 s, skew check
  900 s, final 45000 s, audit 900 s, handoff 120 s), no retry, no resume, global slot lock root
  and global ledger, `dry_run = false`.

## Sizing (rule stated before any strict data; development variance only)

Rule (`preregistration.py`, `SIZING_RULE`, the owner's instruction): per-mix SD = sample SD of
**the candidate's own FRP-v3 Phase R paired H5000 deltas** (training seed 12, 32 worlds per mix,
exact H5000 prefixes of H10000 episodes whose prefix identity Phase R verified 18/18);
`n_fixed(MDE)` = smallest n with `n >= ((t_{1-0.05/3,n-1} + t_{0.90,n-1}) x sd_max / MDE)^2`
(90% per-mix power at the MDE in every mix); `N_max = ceil(1.02 x n_fixed)` (sequential
inflation for four OBF looks); MDE = the smallest multiple of 5 with N_max <= 300. The Phase R
mean deltas are not used. SDs: frozen 134.6, **scripted 311.2 (binding)**, mixed 196.4. Result:
MDE 60 -> N_max 322 (infeasible), **MDE 65 -> n_fixed 269, N_max 275**. Sensitivity (not
binding): with all five Phase R seeds' SDs (194/278/181) the rule gives MDE 60 / N_max 259; with
the v8-v7 reference-bank SDs (124/141/97, a much more tied contrast) MDE 30 / N_max 266.
The normal Monte Carlo confirms the efficacy sizing: per-mix sequential crossing at the MDE in
every mix 1.000 / 0.903 / 1.000 (frozen / scripted / mixed; scripted is the binding mix at 90%).

## Operating characteristics (`operating_characteristics.json`)

Normal deltas with the candidate's own per-mix SDs (simulate.py's `Replica`, reused unchanged;
efficacy + NI only, bands not simulated), 200k replicates per null row, 50k per power row,
development delta_NI 9.946:

| Scenario (frozen, scripted, mixed) | P(efficacy+NI pass) | familywise false rejection: futility followed / ignored | E[worlds/mix] |
|---|---|---|---|
| global null (0,0,0) | 0.0006 | 0.0397 / **0.0495** | 234 |
| global null, Phase R between-mix correlation | 0.0006 | 0.0396 / 0.0491 | 235 |
| one effect, two nulls (260,0,0) | 0.019 | 0.0333 / 0.0333 | 243 |
| scripted at NI margin (260,-9.95,260) | 0.050 | - | 273 |
| MDE in every mix (65,65,65) | 0.989 | - | 178 |
| two mixes at the MDE, scripted 0 | 0.131 | - | 270 |
| candidate's own Phase R means (98.1, 68.5, 130.5) | 0.994 | - | 164 |
| all-seed Phase R means (64.4, 65.5, 67.8) | 0.991 | - | 175 |
| half the all-seed means | 0.618 | - | 256 |

Boundary precision (2M replicates): efficacy any-look at zero 0.01678 (target 0.01667), NI at
its null 0.0500 (target 0.05). Futility is non-binding; the guaranteed familywise error is the
futility-ignored column.

**Joint bootstrap of the Phase R worlds through the whole gate, survival band v2 included**
(world indices resampled jointly across mixes, both arms' survival from the same worlds, a
16-world calibration delta_NI per replicate, survival band v2 judged at the qualifying look and
every later look, futility followed before qualification; replica cross-checked against
`run_sequential_gate` + `plan_pooled_band_check` decision by decision, **600/600 agree**: 300
shifted to the MDE (stop passes, continued band failures, final passes and fails) and 300 at
the global null (187 futility stops, 113 final fails)). Phase R is this candidate's own outcome
evidence: these rows are **planning numbers, not inputs to the band rule** (chosen on
independent data in the amendment):

| Pool / scenario | P(PASS) | STOP_PASS / FINAL_PASS / FINAL_FAIL | stop look 1/2/3/4 | E[worlds/mix] |
|---|---|---|---|---|
| candidate's own worlds, observed effects (mass and survival) | **0.994** | 0.948 / 0.046 / 0.006 | 0.03/0.61/0.31/0.05 | 165 |
| candidate's own worlds, observed mass, survival recentred to no change | **0.966** | 0.850 / 0.117 / 0.034 | 0.00/0.41/0.44/0.15 | 189 |
| candidate's own shape at the MDE in every mix, survival no change | 0.965 | 0.843 / 0.122 / 0.035 | 0.00/0.40/0.45/0.16 | 190 |
| candidate's own shape at the all-seed means (observed survival) | 0.991 | 0.932 / 0.060 / 0.009 | 0.00/0.54/0.39/0.07 | 174 |
| candidate's own shape at the MDE in every mix (observed survival) | 0.990 | 0.930 / 0.060 / 0.010 | 0.00/0.52/0.41/0.07 | 176 |
| candidate's own shape at the global null | 0.001 | - (0.614 futile, 0.385 final fail) | - | 233 |
| all five seeds, observed effects | 0.998 | 0.963 / 0.035 / 0.002 | 0.00/0.55/0.41/0.04 | 171 |
| all five seeds' shape at the MDE in every mix | 0.998 | 0.963 / 0.035 / 0.002 | 0.00/0.53/0.43/0.04 | 173 |
| **regression what-if**: observed mass, pooled survival -0.05 in every mix | **0.041** | 0.018 / 0.023 / 0.959 | 0.00/0.00/0.02/0.98 | 274 |
| **regression what-if**: observed mass, scripted survival -0.10 | **0.049** | 0.022 / 0.028 / 0.951 | 0.00/0.00/0.02/0.98 | 273 |

**Honest expectation.** Efficacy and NI are almost certain at Phase R-sized effects (about
0.99), and the run usually qualifies at look 2 (138 worlds/mix). Under survival band v2 the
survival guardrail no longer decides the gate for a candidate without a survival regression:
expected full-gate pass probability **about 0.97-0.99** (0.994 on the candidate's own worlds
at its observed effects; 0.966 if its survival were exactly unchanged), at about 165-190 worlds
per mix (the frozen v2 band gave 0.24-0.38). A true survival regression at a protected margin
passes about 4-5% of the time. On the amendment's independent between-checkpoint pools
(FRP-v2), the same plan passes 0.986-0.995 at the MDE with no survival change. A STRICT_FAIL
now says more about the candidate than before; the look receipts record which component failed.

## Survival band v2: per-study check and band cost

- **Per-study check** (`research/survival_band_v2_20261006/simulate.py --part study`, the
  amendment's required validity check; 20,000 replicates; the frozen plan; development delta_NI
  9.946; mass effects 32.5, 43.55, 65, 97.5, 130 (0.5 / 0.67 / 1 / 1.5 / 2 x MDE) and 99.04 (the
  candidate's Phase R mean of mix means); pools `frp3_s12` and `frp3_all_seeds`): joint rate
  P(PASS and a regression exactly at a margin) for the pooled regression (-0.05 in every mix) and
  each one-mix regression (-0.10), 48 rows. **PASSES**: max joint rate **0.0502** <= 0.06
  (`frp3_s12`, theta 32.5, scripted at -0.10). Re-judged by `prepare`, `run` and the audit; its
  sha256 is frozen in the intent with the pool. If it had failed, the study would not have
  adopted survival band v2 (no retuning).
- **No-regression pass rate through the gate** (same output, `no_change` rows): 0.62 at 0.5 x
  MDE (mostly no qualification), 0.87-0.90 at 0.67 x MDE, 0.97-0.98 at the MDE and above.
- **Band cost per look** (`look1_band_cost.json`, normal approximation, Phase R pilot SDs:
  pooled 0.211, per mix 0.217 / 0.476 / 0.324 on the candidate's own worlds): probability that
  a component fails at look 1 / 2 / 3 / 4 with a true delta of 0: pooled 0.98 / 0.42 / 0.08 /
  0.01; scripted (per-mix 0.10) 0.99 / 0.54 / 0.16 / 0.04; frozen 0.55 / 0.00 / 0.00 / 0.00;
  mixed 0.92 / 0.15 / 0.01 / 0.00. A failure no longer ends the gate: a qualified run continues,
  so most candidates without a regression that are not yet clear at look 2 pass at look 3.

## Skew check

Input `research/frp3_strict_20261005/phase_r_deltas.json` (the candidate's 32 Phase R paired
deltas per mix, recomputed from the raw records and checked against the summary's per-seed means;
sha256 frozen in the intent). Thresholds 0.020 efficacy any-look (every mix), 0.06 NI any-look
(scripted only); 300k resamples; remedy on failure: stop and escalate (no final world).
Planning-only pre-run at the development delta_NI (not binding; the binding run uses the
calibrated delta_NI): efficacy any-look 0.0154 / 0.0167 / 0.0149 (frozen / scripted / mixed;
sample skewness 0.29 / -0.02 / 0.37), scripted NI 0.0499: would pass.

## Execution platform (template v3) and fallback

- **Platform:** RunPod serverless CPU workers (amendment conditions 1-4). The Mac keeps the
  orchestration (look boundaries, barrier, create-only records and receipts, ledger, audit,
  closeout), AC-power and lid guards. Remote configuration `remote_config.json` (frozen by
  sha256): gate budget **$40** (balance-delta hard stop -> `INVALID_STOP`), identity-check budget
  $8, remote wall cap 150 min, identity wall cap 60 min, live engine, H5000 pinned at
  `evaluation_profile.scored_horizon`, record pins (profile digest, v8 method), Mac episode 39.7 s,
  cloud episode 174 s (pod-calibrated live H5000 x1.5 serverless factor x1.125 for the candidate
  half), sizing objective fastest.
- **Projected sizing and speed-up** (`remote-plan`, worst case: calibration + every look, both
  sides, strict runtime still to be built): 7 workers x 32 vCPU (16 episode slots each, 8 units
  per job), remote ~57 min for the gate, ~99 min end to end including 30 min runtime seeding, the
  identity check (Mac half 6 min) and cold starts, vs **562 min (9.4 h) on 2 Mac slots: 5.67x**
  (>= the per-step **4x** rule, the user's rule since 2026-10-05; by stop look 2.49x / 3.92x /
  4.93x / 5.67x). Expected spend about $9-10 (survival band v2's continuation adds about 2-15%
  expected worlds; worst-case reservations unchanged: $33 gate, $4.3 identity).
- **Identity check** (amendment condition 3, once per intent, after `prepare`, before `run`):
  the default pre-registered sample, 3 worlds per mix of the first look's prefix (ranked by
  `sha256("apex-frp3-strict-final-v1|identity-sample|<i>")`), both arms, full H5000: 18 episodes
  on the Mac (2 slots) and 18 on RunPod; salted digests only. **PASSED** -> the gate runs on
  RunPod; **FAILED** (any byte difference or missing episode) or **ABANDONED** -> the whole gate
  runs on the Mac, 2 slots, under the same intent, plan, banks, interleaving and caps;
  **NOT_RUN** -> `run` refuses.
- **Mac fallback runtime** (disclosed): on 2 Mac slots (live H5000; champion+v8 episodes 44.8 / 26.9 /
  34.2 s from the v8 strict run, the candidate at x1.25, x1.15 overhead), calibration about 16
  min, the final stage at N_max about 37,700 s per worker = **84% of the template's final-stage
  cap** (44,970 s; a cap reached ends the run `INCOMPLETE`), about 6.2 h at the expected 162
  worlds per mix. Disclosed, not a sizing constraint (the owner's rule is N_max <= 300).
- Checkpoints: the five sha256s of `spec.remote_checkpoints()` are on the RunPod upload
  allow-list; the candidate entry (root `artifacts_root`) is the one the user approved on
  2026-10-05 for FRP-v3 Phase R; its use for this Tier-2 strict gate was approved by the user in
  chat on 2026-10-05 (relayed by the coordinator, recorded in the entry's `strict_gate_use`). The
  file is already on the private volume (registry 2026-10-05T19:46:59Z), so the gate uploads
  nothing new.

## Run, stopping, outcomes, audit

As in the template README (no deviation): calibration -> skew check -> looks with barrier and
create-only look receipts -> independent stdlib audit (`sequential_audit.py`, report schema v3)
-> closeout (`receipt.json` only on `STRICT_PASS`). Outcomes `STRICT_PASS`, `STRICT_FAIL`,
`SKEW_CHECK_FAILED`, `INCOMPLETE`, `INVALID_STOP`; never relabelled, never resumed or rerun,
never stopped early outside these rules. After an early stop per-mix means are naive and
descriptive only. Audit scope disclosed as in the v8 strict protocol (envelope fields and
finite metrics per record; record-level identity rests on the in-worker checks in the hashed
source closure and on the `arm_identity_sha256` of each envelope).

## Release and rollback (recorded before the gate; a separate owner action)

- **Release = checkpoint swap, v8 unchanged.** After `STRICT_PASS` (audit PASS) **and** a Mac
  serving qualification of the candidate (+ v8 lambda 8) on the web serving path (condition 4 of
  the RunPod amendment; a package like `apex_veto_v8_serving_20261003`, to be built and
  pre-registered separately), the served Watch hero's checkpoint becomes `eec144bf…`
  (`web/backend/session.py` `DEFAULT_CHECKPOINT`, with `web/backend/safety_veto_serving.py`'s
  served-checkpoint binding re-pointed to this gate's STRICT_PASS receipt, whose candidate block
  names the new checkpoint and the same seven v8 source sha256s). The veto variant, lambda,
  config and every other default stay as released (v8, lambda 8).
- **Rollback = the previous checkpoint**: `champion_a5_freespace_20260621.pth` (`43d4e2c5…`)
  with v8, bound to the v8 STRICT_PASS receipt `29b1f7f6…` as today (one-line revert of the
  release commit; the v8 rollback `SNAKE_SERVE_VETO_VARIANT=v7` keeps its meaning).
- No champion file, default, config, released veto or deployment changes on any gate outcome
  by itself; a STRICT_PASS does not qualify the serving path.

## Operator commands

See the hand-backs of the branches `frp3-strict` and `band-v2` (ratify the strict-on-RunPod
amendment and survival band v2 option 1, preflight, seed, remote-plan, prepare, identity check,
run, audit, decision). `prepare` takes `--n-max 275 --mde 65 --n-calibration 16
--paired-band 0.05,0.05,rci_obf,0.30,0.10 --development-delta-ni 9.945893662499998 --skew-input
research/frp3_strict_20261005/phase_r_deltas.json --remote-config
research/frp3_strict_20261005/remote_config.json`.

## Non-claims

A STRICT_PASS is a receipt about this checkpoint + v8 (lambda 8) vs champion + v8, one wrapped
hero against unwrapped opponents, under `promotion-v2-watch-rect` H5000 on the platform the gate
ran on; it does not qualify serving and does not speak to H10000 (LH-1 covers long horizons as a
non-authoritative screen). Phase R and LH-1 numbers are planning/precondition inputs only.
