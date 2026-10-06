# Governance amendment: survival band v2 (pooled paired NI, per-mix catastrophic floor), 2026-10-06

Status: **proposed amendment, ratification pending** (section "Ratification" at the end). It adds
an opt-in survival guardrail, `band_policy = "pooled_ni_continue"`, to the Tier-2 sequential
strict gate. It amends
[governance_amendment_paired_bands_2026-10-03.md](governance_amendment_paired_bands_2026-10-03.md)
(ratified 2026-10-03) by adding a policy beside `paired_ni_at_stop`. It does not change
`paired_ni_at_stop` or `block_at_stop`.

Written 2026-10-06 (UTC). At the time of writing, the FRP-v3 strict gate
(`research/frp3_strict_20261005`) had no `intent.json`, no calibration episode and no final
episode. Its final namespace `apex-frp3-strict-final-v1` was absent from the global ledger. **No
gate world of any study was played for this amendment.**

## Scope and effective date

- **Prospective only.** The policy may be used only by a strict pre-registration whose
  `intent.json` is written after this amendment is ratified. The pre-registration opts in
  through the spec's `band_policy = "pooled_ni_continue"`.
- **Ratification is enforced.** A production `prepare` binds this document by sha256. It
  refuses unless the "Ratification" section below records the decision as ratified. Dry runs
  are allowed before ratification.
- **Never retroactive.** It does not apply to, re-judge or relabel any started or completed
  run: v5, v7 and v8 strict, FRP-v2 or FRP-v3 screens, or anything else. Their verdicts stand
  under the rules they pre-registered.
- **What stays.**
  - `block_at_stop` (template v1) and `paired_ni_at_stop` (template v2) are unchanged and
    still available.
  - Every existing plan dict, spec descriptor, intent and look receipt is byte-identical
    (tested against golden hashes).
- **First intended user.** The FRP-v3 strict gate (champion change), re-pre-registered under
  this policy before `prepare`. Its world banks, namespaces, plan looks, MDE, N_max, efficacy,
  NI, futility and remote configuration are unchanged.

## Why: the ratified band decides a champion-change gate

The ratified paired band (`paired_ni_at_stop`) works like this:

- For each mix separately, it requires the paired lower bound of the survival delta to
  clear `-0.05`.
- It is judged once, at the qualifying look, and a failure ends the gate.

It was calibrated on veto-only changes (v7 vs v5, v8 vs v7). Those compare the same
checkpoint, so most worlds tie: 30-55% of paired deltas are exactly 0, and the paired SD is
0.15-0.29 per mix.

A **champion change** compares two different checkpoints, and deaths are no longer tied.
In the independent between-checkpoint data below (FRP-v2), the per-world paired survival SD is
**0.25-0.44 per mix** (0.27-0.37 in the pooled pools). At the look where the gate usually
qualifies (look 2, n = 138), a per-mix margin of 0.05 cannot be cleared reliably.

The FRP-v3 protocol (frozen 2026-10-05, before this amendment) already disclosed that its band
"decides the gate": the expected pass probability was about 24-38%, even at the candidate's
Phase R mass effects.

On the independent data the problem is worse than that disclosure:

| Rule (independent pools, plan N_max 275, MDE 65) | FRP-v2 Phase 2 | FRP-v2 Phase 1 |
|---|---|---|
| Gate P(PASS), mass effect = MDE in every mix, no survival change: **frozen rule** (`paired_ni_at_stop`, rci_obf) | **0.023** | **0.016** |
| Same, no band at all (efficacy + NI only) | 1.000 | 1.000 |

The band cannot be fixed by more worlds at the qualifying look. Under the frozen rule:

- the band-only no-regression pass rate at n = 275 (the final look) is 0.46-0.50;
- in the worst independent cell it is 0.34;
- under the normal stress model at a per-mix SD of 0.44 it is 0.17.

## Data provenance (integrity rule)

**Rule.** The guardrail was chosen and calibrated **only** on data that existed independently
of the FRP-v3 candidate's outcome evidence. Every calibration source is a completed run that
closed before FRP-v3 Phase R was merged (2026-10-05 20:14 UTC). None involves the FRP-v3
candidate's checkpoint.

**Not used for any choice or constant:**
- FRP-v3 Phase R;
- the LH-1 screen of the FRP-v3 candidate;
- any other FRP-v3 artifact.

Where FRP-v3 numbers appear in this document, they are labelled **descriptive**.

Extraction: `research/survival_band_v2_20261006/extract.py` writes
`survival_pools_20261006.json` (pinned in `outputs/oc.json` as `data_sha256`). It holds, per
source, the summary/closeout sha256, its creation time and a digest of every record file read.

| Pool | Contrast | World triples | Closed / merged | Paired survival SD per mix (frozen / scripted / mixed) | Pooled SD |
|---|---|---|---|---|---|
| `frp2_phase2` (primary) | FRP-v2 M@60000 (5 training seeds) + v8 vs champion_a5 + v8, H5000 prefix | 160 | 2026-10-05 12:12 UTC | 0.349 / 0.357 / 0.272 | 0.190 |
| `frp2_phase1` | FRP-v2 C and M at 15k/30k/60k (6 cells x 5 seeds) + v8 vs champion_a5 + v8, H5000 | 480 | 2026-10-05 09:19 UTC | 0.336 / 0.374 / 0.295 | 0.199 |
| `frp2_phase1_cell_*` (6) | one Phase 1 cell each | 80 each | (same) | 0.304-0.368 / 0.328-0.441 / 0.247-0.337 | 0.174-0.219 |
| `v8_strict` | champion + v7 -> champion + v8 (v8 strict final records) | 187 | 2026-10-03 | 0.250 / 0.291 / 0.222 | 0.141 |
| `v8_refbank` | champion + v7 -> + v8 (reference bank, SIMD) | 150 | 2026-10-04 | 0.233 / 0.264 / 0.161 | 0.129 |
| `v7_strict`, `v8_screen` | 2026-10-03 validation pools (copied, sha-pinned) | 243 / 60 | 2026-10-02 / -03 | 0.22-0.24 / 0.15-0.24 | 0.13 |

Facts that shape the design:

- **Between-mix correlation.** The paired survival deltas of one world are almost uncorrelated
  across mixes (-0.12 to 0.17). Pooling the three mixes therefore cuts the SD by about
  sqrt(3): a per-world pooled SD of 0.19-0.20 between checkpoints.
- **Mass and survival move together.** The per-world mass/survival delta correlation is
  0.95-0.99, the source of the qualifying-look selection effect.

**Disclosure.** The FRP-v3 protocol prints the candidate's Phase R paired survival SDs
(0.22 / 0.48 / 0.32), and the designer of this amendment read that protocol. To keep those
numbers from influencing the rule:

- no constant, grid or criterion below references them;
- the per-mix margin is selected by a stated criterion over the independent pools only;
- the stress grid is a uniform SD grid (0.25-0.50), not a fit to any study.

`FRP-v2 Phase 2` is legitimate between-checkpoint data. It is the closest analogue of a
champion change: a fine-tuned champion continuation vs the champion, both with v8, on fresh
worlds. It was collected and closed before FRP-v3 existed as evidence.

## Options evaluated

Survival is `survival_fraction`, the alive frames / scored frames of a hero that ends at its
first death (`hero_terminal`). That is already the **censoring-aware time-to-death** metric
(restricted mean survival time / H). The binary "survived to H" indicator has about twice the
paired SD (0.57-0.76 between checkpoints, vs 0.27-0.37) on a different effect scale. A
mass-weighted survival would mix in the efficacy endpoint, which the guardrail exists to check
independently, because mass and survival correlate 0.96 per world. Both are rejected without
a separate OC.

Rules simulated (`simulate.py --part oc`; floor 0.30 in every rule except "none"):

| Id | Rule |
|---|---|
| A_rci | frozen FRP-v3 rule: per-mix NI, M 0.05, alpha 0.05 `rci_obf`, judged once at the qualifying look, failure stops |
| A_pw | the same with the `pointwise` bound (2026-10-03 recommendation) |
| B | per-mix NI M 0.05 alpha 0.05, judged at the **final look only** (qualified runs continue to N_max) |
| C | per-mix NI **recalibrated** (M 0.075, alpha 0.10), final look only |
| D | **pooled** NI (M 0.05) + **per-mix catastrophic** NI (M 0.10), alpha 0.05, final look only |
| **E** | **D's components with `rci_obf` levels, judged at every look from the qualifying look on, continue on failure (proposed)** |
| E075 | E with a per-mix margin of 0.075 (the stricter option the owner may choose; see "Ratification") |
| F | survival co-reported, absolute floor 0.30 only |

### Band only, at N_max = 275 (20,000 resamples per cell; per-band level as the rule uses it at the last look)

P(all bands pass). The first column is "no survival change"; the others are true regressions
(the joint error that the guardrail must cap). "one mix" is the worst of the three mixes.

| Pool | Rule | No change | Pooled -0.05 (all mixes) | One mix -0.05 | One mix -0.075 | One mix -0.10 |
|---|---|---|---|---|---|---|
| frp2_phase2 | A_rci | 0.496 | 0.000 | 0.030 | 0.001 | 0.000 |
| | B | 0.537 | 0.000 | 0.037 | 0.002 | 0.000 |
| | C | 0.976 | **0.129** | 0.587 | 0.101 | 0.007 |
| | D | 0.996 | 0.048 | 0.851 | 0.387 | 0.047 |
| | **E** | **0.995** | **0.040** | 0.830 | 0.353 | **0.040** |
| | E075 | 0.930 | 0.021 | 0.387 | 0.042 | 0.002 |
| | F | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| frp2_phase1 | A_rci | 0.460 | 0.000 | 0.028 | 0.002 | 0.000 |
| | **E** | **0.990** | **0.039** | 0.769 | 0.317 | **0.037** |
| | E075 | 0.917 | 0.021 | 0.341 | 0.040 | 0.003 |
| worst cell (C@60000) | A_rci | 0.339 | 0.000 | 0.027 | 0.003 | 0.000 |
| | **E** | **0.973** | 0.037 | 0.717 | 0.265 | 0.042 |
| | E075 | 0.830 | 0.014 | 0.272 | 0.045 | 0.004 |
| v8_strict (veto change) | A_rci | 0.821 | 0.000 | 0.034 | 0.001 | 0.000 |
| | **E** | **1.000** | 0.042 | 0.972 | 0.537 | 0.036 |

**Normal stress model** (every mix at the same paired SD, independent mixes), no-change pass
rate at N_max:

| Per-mix SD | 0.30 | 0.35 | 0.40 | 0.441 | 0.50 |
|---|---|---|---|---|---|
| A_rci | 0.61 | 0.41 | 0.25 | 0.17 | 0.11 |
| B (final only, per mix) | 0.65 | 0.45 | 0.29 | 0.20 | 0.13 |
| **E** | **0.999** | **0.989** | **0.951** | **0.901** | **0.789** |
| E075 | 0.975 | 0.900 | 0.763 | 0.641 | 0.466 |

### Through the whole gate (joint world resampling; efficacy, scripted NI, futility followed, then the band rule)

Plan: the FRP-v3 strict gate's frozen looks (N_max 275, looks 69/138/207/275, MDE 65), with
development delta_NI 9.946. Both are design inputs fixed before this amendment. 20,000
resamples per cell.

- **Mass deltas** are the pool's own, centred and shifted to theta. The "rescaled" rows also
  rescale each mix's mass deltas to the plan's sizing SDs (134.6 / 311.2 / 196.4), so that
  qualification timing matches the plan. That rescale is mass only and already frozen in the
  FRP-v3 protocol.
- **"Planned"** is theta = (98.1, 68.5, 130.5), the planning mass effects in the FRP-v3
  protocol (mass only).

P(PASS) and E[worlds per mix]:

| Pool, mass | theta | No band | A_rci (frozen) | A_pw | B | C | D | **E (proposed)** | E075 |
|---|---|---|---|---|---|---|---|---|---|
| frp2_phase2 | 1 x MDE | 1.000 (164) | 0.023 | 0.224 | 0.537 (275) | 0.977 (275) | 0.995 (275) | **0.994 (178)** | 0.936 (214) |
| frp2_phase2 | planned | 1.000 (147) | 0.011 | 0.180 | 0.538 | 0.976 | 0.996 | **0.995 (174)** | 0.934 (215) |
| frp2_phase2, rescaled | 1 x MDE | 0.989 (177) | 0.051 | 0.281 | 0.540 | 0.975 | 0.987 | **0.986 (188)** | 0.933 (215) |
| frp2_phase2, rescaled | planned | 0.993 (164) | 0.030 | 0.234 | 0.537 | 0.975 | 0.990 | **0.990 (181)** | 0.933 (215) |
| frp2_phase1 | 1 x MDE | 1.000 (160) | 0.016 | 0.185 | 0.503 | 0.972 | 0.992 | **0.991 (181)** | 0.921 (219) |
| frp2_phase1, rescaled | planned | 0.993 (163) | 0.025 | 0.203 | 0.502 | 0.970 | 0.988 | **0.987 (183)** | 0.919 (218) |
| frp2_phase2 | 0.67 x MDE | 0.945 (223) | 0.208 | 0.413 | 0.534 | 0.936 | 0.944 | **0.944 (224)** | 0.912 (231) |
| frp2_phase2 | 0.5 x MDE | 0.675 (256) | 0.382 | 0.459 | 0.483 | 0.670 | 0.674 | **0.674 (256)** | 0.661 (258) |

**Joint error: P(PASS and a true regression exactly at a protected margin), maximum over all
gate cells.** The cells cover 3 pools x 6 mass effects plus 4 rescaled cells, and every
regressed mix.

| Rule | Pooled -0.05 (all mixes) | One mix -0.10 | One mix -0.075 | One mix -0.05 |
|---|---|---|---|---|
| **E (proposed)** | **0.047** (Wilson upper 0.050) | **0.047** (0.050) | 0.55 | 0.98 |
| E075 | 0.031 | 0.003 | **0.049** (0.052) | 0.58 |
| D | 0.049 | 0.048 | 0.57 | 0.98 |
| C | **0.258** | 0.009 | 0.106 | 0.73 |
| A_rci (frozen) | 0.000 | 0.000 | 0.001 | 0.016 |

### What cannot be had at N_max <= 300: a per-mix 0.05 guarantee with a usable pass rate

A drop of 0.05 confined to **one** mix is visible only in that mix's data. No rule, pooled or
otherwise, can detect it better than the single-mix test of 0.05 against 0.

At n = 275 and a between-checkpoint per-mix SD of about 0.36, that test's separation is
0.05 / (0.36 / sqrt(275)) = 2.30 standard errors. So:

- if a single-mix 0.05 drop may pass at most 10% of the time, a candidate with no regression
  passes that mix at most Phi(2.30 - 1.28) = 0.85 of the time;
- across the three mixes, that is about 0.61 at best.

Meeting both "a one-mix 0.05 drop passes <= 10%" and "no regression passes >= 80%" needs about
**400 worlds per mix** (about 500 at 5%). That is beyond the owner's N_max <= 300 cap.

The proposed rule therefore protects two cases at alpha = 0.05:

- a survival regression of 0.05 **on average over the mixes**, which includes a 0.05 drop in
  every mix;
- a **catastrophic** 0.10 drop in any single mix.

It tolerates a 0.05 loss confined to one mix, which passes about 0.72-0.85 of the time on the
between-checkpoint pools. E075 is the stricter alternative: it caps a one-mix drop of 0.075
(not 0.05) and passes a one-mix 0.05 drop 0.27-0.40 of the time. The price is a no-regression
pass rate of 0.83-0.94 at N_max, and 0.64 under the 0.44-SD stress.

## The rule (`band_policy = "pooled_ni_continue"`)

Bands are `{"metric": "survival_fraction", "mix": m}`, one per mix (exactly the three mixes
frozen, scripted, mixed). For world i of the look-k prefix (world order pre-declared;
n_k = `look_sizes[k]`):

    d_{i,m} = candidate_{i,m} - incumbent_{i,m}            (survival_fraction, same world)
    p_i     = (1/3) * (d_{i,frozen} + d_{i,scripted} + d_{i,mixed})   (pooled per-world delta)

At look k the bands pass iff **all** of:

1. **Pooled NI:** `mean(p) - t_{n_k-1}(band_nominal_p[k]) * sd(p) / sqrt(n_k) > -band_ni_margin`.
2. **Per-mix catastrophic NI**, every mix m:
   `mean(d_m) - t_{n_k-1}(band_nominal_p[k]) * sd(d_m) / sqrt(n_k) > -band_mix_margin`.
3. **Absolute floor** (if pre-registered), every mix m: `mean(candidate_m) >= band_floor`.

The comparisons are strict `>`. With zero spread, the bound is the mean.

**Levels.** `band_bound` must be `rci_obf`: `band_nominal_p[k]` is the nominal level of a
Lan-DeMets O'Brien-Fleming spending of `band_alpha` over the plan's looks. For 275 / 0.05 that
is 9.12e-5, 5.63e-3, 2.21e-2, 4.26e-2. A `pointwise` bound is refused, because the band is
judged at several looks.

**Timing and decisions.**

- **Qualification is unchanged:** at least `required_successes` mixes crossed, plus scripted
  NI established. Both are sticky.
- **Before qualification**, nothing changes: `CONTINUE` or `STOP_FUTILE`, exactly as before.
- **At the qualifying look and every later look k < last:**
  - `STOP_PASS` if the bands pass at k;
  - otherwise the new decision **`CONTINUE_BANDS`**. The gate plays the next look; efficacy
    and NI stay established and futility is no longer evaluated for stopping.
- **At the last look:** `FINAL_PASS` iff the run qualified and the bands pass at the last
  look, else `FINAL_FAIL`.
- `STOP_FAIL_BANDS` never occurs under this policy.
- Efficacy and NI are unaffected: bands can only turn a qualified run into a fail, never a
  non-qualified run into a pass.

**Recommended (and, on ratification, the ratified) values:** `band_ni_margin` 0.05, `band_mix_margin` 0.10,
`band_alpha` 0.05, `band_bound` `rci_obf`, `band_floor` 0.30.

### The guarantee

The band's components are repeated confidence bounds (RCI) at level `band_alpha`. If the true
pooled delta is -`band_ni_margin` or below, then P(the pooled bound clears at **any** look)
<= `band_alpha`, up to the t-for-z approximation. The same holds for each mix's per-mix bound
at -`band_mix_margin`. This holds at any data-dependent stopping or qualifying look, so it holds
however the correlated mass deltas select the look.

A PASS requires every component to clear at the same look. So for each protected regression,
P(PASS and that regression) <= `band_alpha`. This is an intersection-union rule: no
multiplicity share is needed across components or mixes.

**Measured.** The joint error was at most 0.047 for the pooled 0.05 regression and 0.047 for a
one-mix 0.10 regression, over every gate cell above (Wilson upper 0.050).

**This is the step that fixes the old rule's weak spot.** A pointwise bound judged at a
data-dependent look is not protected. That was the 0.0612 per-study failure of `pointwise` in
FRP-v3.

### Choice of the per-mix margin (stated criterion, independent data only)

`band_mix_margin` = the smallest value in {0.05, 0.075, 0.10, 0.125} for which the final-look
band (n = 275, level `band_nominal_p[last]`, pooled margin 0.05) passes a no-change candidate
with probability >= 0.90 in **every** independent between-checkpoint pool. The pools are
FRP-v2 Phase 2, FRP-v2 Phase 1 and each of the six Phase 1 cells: the realistic range of
between-checkpoint SDs.

The 0.90 floor leaves 0.10 of headroom over the 0.80 target. That headroom absorbs:

- the pooled component;
- the SD uncertainty of 80-world cells;
- candidates whose SDs exceed the observed ones.

| `band_mix_margin` | Worst pool (no-change pass) | Meets | Stress SD 0.441 | One mix -0.05 passes (frp2_phase2, worst mix) |
|---|---|---|---|---|
| 0.05 | C@60000 (0.339) | no | 0.17 | 0.03 |
| 0.075 | C@60000 (0.830) | no | 0.64 | 0.39 |
| **0.10** | C@60000 (0.973) | **yes (chosen)** | 0.90 | 0.83 |
| 0.125 | M@30000 (0.982) | yes | 0.94 | 0.89 |

The pooled margin stays the ratified 0.05, so the size of the tolerated average survival loss
is unchanged. The alpha stays the ratified 0.05.

### Cost

E[worlds per mix] rises from about 164-177 (efficacy and NI alone, at the MDE or the planned
effects) to about 174-188. Qualified runs whose bands do not yet clear continue one or two
looks. The worst case is unchanged at N_max = 275, so the RunPod sizing (sized for every look)
and the Mac-fallback cap disclosure do not change. Expected spend rises in proportion to
E[worlds], about +10%.

### Descriptive only: how the rule would treat FRP-v3 Phase R

`outputs/descriptive_frp3_phase_r.json` uses the same replica on the FRP-v3 pool
(`research/frp3_strict_20261005/paired_pool.json`). It is **not an input to any choice
above.**

| Pool (FRP-v3, descriptive) | Survival scenario | A_rci (frozen) | **E** | E075 |
|---|---|---|---|---|
| candidate's own worlds (`frp3_s12`), observed mass | observed survival deltas (+0.052 / +0.023 / +0.131) | 0.267 | **0.994** | 0.965 |
| same | survival centred to no change | 0.016 | **0.967** | 0.828 |
| all five Phase R seeds, observed mass | observed (+0.014 / +0.025 / +0.033) | 0.235 | **0.997** | 0.980 |

Notes:

- The frozen rule's 0.267 matches the FRP-v3 protocol's 0.27, a consistency check of the replica.
- The candidate's own scripted paired SD (0.476) exceeds every independent cell (at most
  0.441). Under E, its no-change pass at N_max is still 0.967.

## Per-study pre-registration check (required)

A study that opts in runs:

    simulate.py --part study --data <the study's paired pool> --plan-params <frozen plan doc> \
        --delta-ni <development delta_NI> --reps 20000 \
        --thetas <0.5 MDE>,<0.67 MDE>,<MDE>,<1.5 MDE>,<2 MDE>[,<planning effect>]

**Scenarios** (`research/survival_band_v2_20261006/simulate.py`):

- no change;
- the pooled regression at the margin (-`band_ni_margin` in every mix);
- each one-mix regression at -`band_mix_margin`.

**Acceptance:** every joint rate P(PASS and the regression) is at most 1.2 x `band_alpha` (0.06).

**Binding.** The output and the pool are hash-bound in the intent. `prepare` re-judges them,
and so do `validate_intent` and the audit, in their own code.

**If the check fails, the study does not adopt this policy.** No margin, alpha or floor may be
retuned on the study's own data. This check is a validity check of the guarantee on the
study's data shape, not a calibration.

## Implementation (opt-in; legacy unchanged)

**`src/evaluation/sequential_gate.py`**
- `band_policy = "pooled_ni_continue"`, with a new plan field `band_mix_margin`.
- `pooled_band_check` and `plan_pooled_band_check`.
- The `CONTINUE_BANDS` decision.
- Under the legacy policies the new field must be `None` and is left out of `as_dict()`, so
  every legacy plan dict and sha256 is unchanged.

**`research/sequential_strict_template/`, template v2 band layer, version
`sequential-strict-template/v2-pooled`**
- Usable under the v3 remote intent.
- `--paired-band M,alpha,bound,floor[,mix_margin]` (five values under this policy).
- Look inputs and receipts carry `pooled_band` rows per look, the pairs digest and the judged
  looks. `CONTINUE_BANDS` maps to `continue`.
- The per-study check output is judged by `judge_pooled_check`.
- The amendment binding (`band_amendment`: path, sha256, ratified) is re-derived from this
  document's text.

**`sequential_audit.py`**
- Recomputes the pooled and per-mix bounds and the floor at every look, from the raw record
  pairs, in its own stdlib code.
- Recomputes the OBF levels, the decision sequence including `CONTINUE_BANDS`, and the
  per-study check.
- Checks the amendment ratification.

**Tests** cover legacy golden hashes, the new rule's arithmetic, the decision sequence and the
runner/audit agreement.

## Pre-registration items (in addition to the earlier amendments)

- **Plan values:** `band_policy`, `band_ni_margin`, `band_mix_margin`, `band_alpha`,
  `band_bound` (`rci_obf`) and `band_floor`, all in the hashed plan.
- **Per-study check:** its output and pool, sha256-bound.
- **Protocol statements:** the protocol states the expected pass probability under this rule
  and the descriptive application to the study's own pilot.
- **This document:** bound by sha256 and ratified before a production `prepare`.

## Disclosed limits

- **Shift model.** The shift model borrows each pool's per-world delta shape for hypothetical
  regressions. `survival_fraction` is bounded in [0, 1], and additive shifts ignore the bound.
- **Normal approximation.** The RCI guarantee rests on the normal (t-matched) approximation.
  The bootstrap measures the joint error on real, skewed, tie-heavy deltas: at most 0.047.
- **Lineage.** All pools are descendants of one champion (champion_a5) with the v7/v8 veto. A
  candidate from a different lineage re-runs the per-study check.
- **Equal weights.** Pooling weights the three mixes equally, so a regression in one mix is
  diluted by a factor of 3 in the pooled component. That is why the per-mix catastrophic floor
  exists. Receipts report every per-mix bound, so a 0.05 one-mix loss that passes is visible
  and reported.
- **Absolute floor.** The floor (0.30) is a wiring tripwire, not a test. It does not catch any
  realistic regression.

## Evidence files

- `research/survival_band_v2_20261006/extract.py`, `survival_pools_20261006.json`: the
  calibration pools, from independent data only.
- `research/survival_band_v2_20261006/simulate.py`: parts `oc`, `descriptive`, `study` and
  `crosscheck`.
- `outputs/oc.json`: design OC, about 70 s of one core, one shared CPU slot.
- `outputs/descriptive_frp3_phase_r.json`: descriptive only.
- `outputs/crosscheck.json`: the replica against the implementation, decision by decision.

## Ratification

**Pending.** To be completed by the research loop owner (or the user). The owner ratifies one
option:

- **Option 1 (recommended):** `pooled_ni_continue` with `band_ni_margin` 0.05,
  `band_mix_margin` 0.10, `band_alpha` 0.05, `band_bound` `rci_obf`, `band_floor` 0.30.
- **Option 2 (stricter per mix):** the same with `band_mix_margin` 0.075. This changes the
  study's `--paired-band` argument, so the per-study check must be re-run.

Questions for the owner:

1. **Margins.** Do you accept that a 0.05 survival loss confined to one mix is tolerated
   unless it is catastrophic (>= 0.10)? The alternative is >= 0.075 under option 2. A per-mix
   0.05 guarantee is not attainable at N_max <= 300 with a usable pass rate (section "What
   cannot be had").
2. **Continuation cost.** Do you accept the continuation cost: about +10% expected worlds, and
   the same worst case?

Record the decision here:

- Decision: ______ (ratified / ratified with changes / rejected)
- Option: ______ (1 / 2)
- Date: ______
- Changes, if any: ______
