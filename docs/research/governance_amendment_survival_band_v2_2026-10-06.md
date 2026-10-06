# Governance amendment: survival band v2 (pooled paired NI, per-mix catastrophic floor), 2026-10-06

Status: **ratified 2026-10-06 (option 1)** (see "Ratification" at the end). It adds an
opt-in survival guardrail, `band_policy = "pooled_ni_continue"`, to the Tier-2 sequential strict
gate. It amends
[governance_amendment_paired_bands_2026-10-03.md](governance_amendment_paired_bands_2026-10-03.md)
(ratified 2026-10-03) by adding a policy beside `paired_ni_at_stop`, which it does not change.
Nor does it change `block_at_stop`.

Written 2026-10-06 (UTC). At that time the FRP-v3 strict gate (`research/frp3_strict_20261005`)
had no `intent.json`, no calibration episode and no final episode, and its final namespace
`apex-frp3-strict-final-v1` was not in the global ledger. **No gate world of any study was
played for this amendment.**

**Revision history.**

- **2026-10-06, first version** (commit `f4a3ff6`). It recommended a per-mix margin of 0.10,
  chosen with a self-set 0.90 no-regression floor.
- **2026-10-06, revised before ratification** after an independent adversarial review:
  - **Recommendation changed.** The margin is now selected at the owner's own target
    (no-regression pass rate >= 0.80). That gives **0.075**, which protects single-mix
    regressions better. The 0.10 option is kept as option 2.
  - **Provenance corrected.** The design plan's geometry (mass only) comes from the FRP-v3
    protocol. It is stated as such below.
  - **Ratification parsing made stricter.** "ratified with changes" is not accepted.
  - **Simulator bound.** The per-study check's simulator is bound by sha256.
  - **Ranges corrected.** Some quoted ranges were wrong and are fixed.

## Scope and effective date

- **Prospective only.** The policy applies only to a strict pre-registration whose `intent.json`
  is written after this amendment is ratified, and which opts in through the spec's
  `band_policy = "pooled_ni_continue"`.
- **Enforced at `prepare`.** A production `prepare` binds this document by sha256. It refuses
  unless all three hold:
  - the "Ratification" section reads exactly `- Decision: ratified`;
  - it names an option (1 or 2) and records no changes;
  - the plan's band values equal that option's values.

  Dry runs are allowed before ratification.
- **Never retroactive.** It does not apply to any started or completed run: v5, v7 and v8
  strict, the FRP-v2 and FRP-v3 screens, or anything else. It does not re-judge or relabel
  them; their verdicts stand under the rules they pre-registered.
- **What stays.**
  - `block_at_stop` (template v1) and `paired_ni_at_stop` (template v2) are unchanged and
    remain available.
  - Every existing plan dict, spec descriptor, intent and look receipt is byte-identical. This
    is tested against golden hashes, including the FRP-v3 package's first frozen plan.
- **First intended user.** The FRP-v3 strict gate (a champion change), re-pre-registered under
  this policy before `prepare`. These are all unchanged: its world banks, namespaces, plan
  looks, MDE, N_max, efficacy, NI, futility and remote configuration.

## Why: the ratified band decides a champion-change gate

The ratified paired band (`paired_ni_at_stop`) works per mix:

- each mix's paired lower bound of the survival delta must clear -0.05;
- the bound is judged once, at the qualifying look;
- a failure ends the gate.

It was calibrated on veto-only changes (v7 vs v5, v8 vs v7). Those compare the same
checkpoint, so many worlds tie: 30-55% of paired deltas are exactly 0, and the paired SD is
0.15-0.29 per mix.

A **champion change** compares two different checkpoints, and their deaths do not tie. In
the independent between-checkpoint data below (FRP-v2), the per-world paired survival SD is
**0.25-0.44 per mix** (0.27-0.37 in the pooled pools). At the look where the gate usually
qualifies (look 2, n = 138), a per-mix margin of 0.05 cannot be cleared reliably.

The FRP-v3 protocol (frozen 2026-10-05, before this amendment) already disclosed that the band
decides the gate: expected pass probability about 24-38% even at the candidate's Phase R mass
effects. On the independent data it is worse:

| Rule (independent pools, plan N_max 275, MDE 65) | FRP-v2 Phase 2 | FRP-v2 Phase 1 |
|---|---|---|
| Gate P(PASS), mass effect = MDE in every mix, no survival change: **frozen rule** (`paired_ni_at_stop`, rci_obf) | **0.023** | **0.016** |
| Same, no band at all (efficacy + NI only) | 1.000 | 1.000 |

More worlds at the qualifying look cannot fix the band. Under the frozen rule, the band-only
no-regression pass rate at n = 275 (the final look) is:

- 0.46-0.50 on the two FRP-v2 pools;
- 0.34 in the worst independent cell;
- 0.17 under the normal stress model at a per-mix SD of 0.44.

## Data provenance (integrity rule)

**Rule.** Every choice about the survival guardrail was made on **survival data that existed
independently of the FRP-v3 candidate's outcome evidence**: the policy, the components, the
margins, alpha, the bound and the margin-selection criterion. Every calibration source is a
completed run that closed before FRP-v3 Phase R was merged (2026-10-05 20:14 UTC). None
involves the FRP-v3 candidate's checkpoint.

**No FRP-v3 survival number** feeds any survival constant, grid or criterion. That covers
FRP-v3 Phase R, the LH-1 screen of the FRP-v3 candidate, and every other FRP-v3 artifact.
FRP-v3 survival numbers in this document are labelled **descriptive**.

**Design-plan geometry from FRP-v3 (mass only).** The OC simulation uses one plan: the FRP-v3
strict gate's frozen plan. Its geometry was sized in the FRP-v3 protocol from the FRP-v3
candidate's Phase R **mass** deltas, and the simulation uses these values:

- N_max 275, the looks 69 / 138 / 207 / 275, and MDE 65, all from the candidate's Phase R mass
  SDs;
- the "planned" mass effects (98.1, 68.5, 130.5), which are the candidate's Phase R mass means;
- the mass-SD rescale (134.6, 311.2, 196.4), which is the candidate's Phase R mass SDs;
- the development delta_NI 9.946, from the Phase R incumbent's scripted mass.

These are mass-only design inputs of the study the amendment serves; none is a survival
outcome. N_max = 275 is also the n at which the per-mix margin is selected below. Selecting at
a different N_max would select a margin for a different plan. Every study re-runs its own
per-study check at its own N_max.

Extraction: `research/survival_band_v2_20261006/extract.py` writes
`survival_pools_20261006.json` (sha256 `6295400a…dc06a`, recorded in `outputs/oc.json`). Per
source it records the summary or closeout sha256, its creation time and a digest of every
record file read.

| Pool | Contrast | World triples | Closed / merged | Paired survival SD per mix (frozen / scripted / mixed) | Pooled SD |
|---|---|---|---|---|---|
| `frp2_phase2` (primary) | FRP-v2 M@60000 (5 training seeds) + v8 vs champion_a5 + v8, H5000 prefix | 160 | 2026-10-05 12:12 UTC | 0.349 / 0.357 / 0.272 | 0.190 |
| `frp2_phase1` | FRP-v2 C and M at 15k/30k/60k (6 cells x 5 seeds) + v8 vs champion_a5 + v8, H5000 | 480 | 2026-10-05 09:19 UTC | 0.336 / 0.374 / 0.295 | 0.199 |
| `frp2_phase1_cell_*` (6) | one Phase 1 cell each | 80 each | (same) | 0.304-0.368 / 0.328-0.441 / 0.247-0.337 | 0.174-0.219 |
| `v8_strict` | champion + v7 -> champion + v8 (v8 strict final records) | 187 | 2026-10-03 | 0.250 / 0.291 / 0.222 | 0.141 |
| `v8_refbank` | champion + v7 -> + v8 (reference bank, SIMD) | 150 | 2026-10-04 | 0.233 / 0.264 / 0.161 | 0.129 |
| `v7_strict`, `v8_screen` | 2026-10-03 validation pools (copied, sha-pinned) | 243 / 60 | 2026-10-02 / -03 | 0.22-0.24 / 0.15-0.24 | 0.13 |

Facts that shape the design:

- **Little correlation across mixes.** The paired survival deltas of one world correlate only
  weakly across mixes (-0.16 to 0.19 over all pools and cells). Pooling the three mixes
  therefore cuts the SD by about sqrt(3), to a per-world pooled SD of 0.17-0.22 between
  checkpoints.
- **Mass and survival move together.** The per-world mass/survival delta correlation is
  0.95-0.99. That is the source of the qualifying-look selection effect.

**Disclosure.** The FRP-v3 protocol prints the candidate's Phase R paired survival SDs (0.22 /
0.48 / 0.32), and the designer of this amendment had read that protocol. No constant, grid or
criterion below references those numbers:

- the stress grid is a set of equal per-mix SDs (0.25-0.50, which includes the largest
  independent cell SD, 0.441);
- the per-mix margin is selected at the owner's own 0.80 target, over the independent pools
  only.

The revision moved the recommendation from 0.10 to 0.075. The stricter margin **lowers** the
FRP-v3 candidate's descriptive pass probability (from 0.994 to 0.965 at its observed effects).

FRP-v2 Phase 2 is legitimate between-checkpoint data and the closest analogue of a champion
change: a fine-tuned continuation of the champion vs the champion, both with v8, on fresh
worlds. It was collected and closed before FRP-v3 existed as evidence.

## Options evaluated

**The metric.** Survival is `survival_fraction`: the alive frames divided by the scored frames
of a hero that ends at its first death (`hero_terminal`). That is already the
**censoring-aware time-to-death** metric (restricted mean survival time / H). The alternatives
were rejected without a separate OC:

- **Binary "survived to H".** It has about twice the paired SD (0.57-0.76 between checkpoints,
  vs 0.27-0.37) and works on a different effect scale.
- **Mass-weighted survival.** It would mix in the efficacy endpoint, which the guardrail exists
  to check independently. Mass and survival correlate 0.96 per world.

Rules simulated by `simulate.py --part oc`. Every rule except "none" keeps the 0.30 floor.

| Id | Rule |
|---|---|
| A_rci | frozen FRP-v3 rule: per-mix NI M 0.05, alpha 0.05 `rci_obf`, judged once at the qualifying look, failure stops |
| A_pw | the same with the `pointwise` bound (the 2026-10-03 recommendation) |
| B | per-mix NI M 0.05, alpha 0.05, judged at the **final look only** (qualified runs continue to N_max) |
| C | per-mix NI **recalibrated** (M 0.075, alpha 0.10), final look only |
| D | **pooled** NI (M 0.05) + **per-mix catastrophic** NI (M 0.075), alpha 0.05, final look only |
| **E** | **D's components with `rci_obf` levels, judged at every look from the qualifying look on, continue on failure: proposed, option 1 (per-mix 0.075)** |
| E10 | E with a per-mix margin of 0.10: option 2 |
| F | survival co-reported, absolute floor 0.30 only |

### Band only, at N_max = 275

20,000 resamples per cell, each band at the level its rule uses at the last look. The table
gives P(all bands pass). "No change" is the candidate without a survival regression. The other
columns are true regressions, i.e. the error the guardrail must cap. "One mix" is the worst of
the three mixes.

| Pool | Rule | No change | Pooled -0.05 (all mixes) | One mix -0.05 | One mix -0.075 | One mix -0.10 |
|---|---|---|---|---|---|---|
| frp2_phase2 | A_rci | 0.496 | 0.000 | 0.030 | 0.001 | 0.000 |
| | B | 0.537 | 0.000 | 0.037 | 0.002 | 0.000 |
| | C | 0.976 | **0.129** | 0.587 | 0.101 | 0.007 |
| | D | 0.941 | 0.026 | 0.421 | 0.049 | 0.002 |
| | **E (option 1)** | **0.930** | **0.021** | 0.387 | **0.042** | 0.002 |
| | E10 (option 2) | 0.995 | 0.040 | 0.830 | 0.353 | 0.040 |
| | F | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| frp2_phase1 | A_rci | 0.460 | 0.000 | 0.028 | 0.002 | 0.000 |
| | **E (option 1)** | **0.917** | **0.021** | 0.341 | **0.040** | 0.003 |
| | E10 (option 2) | 0.990 | 0.039 | 0.769 | 0.317 | 0.037 |
| worst cell (C@60000) | A_rci | 0.339 | 0.000 | 0.027 | 0.003 | 0.000 |
| | **E (option 1)** | **0.830** | 0.014 | 0.272 | 0.045 | 0.004 |
| | E10 (option 2) | 0.973 | 0.037 | 0.717 | 0.265 | 0.042 |
| v8_strict (veto change) | A_rci | 0.821 | 0.000 | 0.034 | 0.001 | 0.000 |
| | **E (option 1)** | **0.995** | 0.029 | 0.559 | 0.037 | 0.001 |
| | E10 (option 2) | 1.000 | 0.042 | 0.972 | 0.537 | 0.036 |

**Normal stress model** (every mix at the same paired SD, mixes independent): no-change pass
rate at N_max.

| Per-mix SD | 0.30 | 0.35 | 0.40 | 0.441 | 0.50 |
|---|---|---|---|---|---|
| A_rci | 0.61 | 0.41 | 0.25 | 0.17 | 0.11 |
| B (final only, per mix) | 0.65 | 0.45 | 0.29 | 0.20 | 0.13 |
| **E (option 1)** | **0.975** | **0.900** | **0.763** | **0.641** | **0.466** |
| E10 (option 2) | 0.999 | 0.989 | 0.951 | 0.901 | 0.789 |

### Through the whole gate

Each replicate resamples worlds jointly and runs efficacy, scripted NI and futility (followed),
then the band rule.

**Plan.** The FRP-v3 gate's frozen looks (N_max 275, looks 69/138/207/275, MDE 65) and the
development delta_NI 9.946. These are mass-only design inputs (see "Data provenance"). 20,000
resamples per cell.

**Mass deltas.** Each pool's own mass deltas are centred and shifted to theta:

- "rescaled" rows also rescale each mix's mass deltas to the plan's sizing SDs;
- "planned" is theta = (98.1, 68.5, 130.5).

P(PASS) with no survival change, and E[worlds per mix] in brackets:

| Pool, mass | theta | No band | A_rci (frozen) | A_pw | B | C | D | **E (option 1)** | E10 (option 2) |
|---|---|---|---|---|---|---|---|---|---|
| frp2_phase2 | 1 x MDE | 1.000 (164) | 0.023 | 0.224 | 0.537 (275) | 0.977 (275) | 0.942 (275) | **0.936 (214)** | 0.994 (178) |
| frp2_phase2 | planned | 1.000 (147) | 0.011 | 0.180 | 0.538 | 0.976 | 0.940 | **0.934 (215)** | 0.995 (174) |
| frp2_phase2, rescaled | 1 x MDE | 0.989 (177) | 0.051 | 0.281 | 0.540 | 0.975 | 0.941 | **0.933 (215)** | 0.986 (188) |
| frp2_phase2, rescaled | planned | 0.993 (164) | 0.030 | 0.234 | 0.537 | 0.975 | 0.941 | **0.933 (215)** | 0.990 (181) |
| frp2_phase1 | 1 x MDE | 1.000 (160) | 0.016 | 0.185 | 0.503 | 0.972 | 0.931 | **0.921 (219)** | 0.991 (181) |
| frp2_phase1, rescaled | planned | 0.993 (163) | 0.025 | 0.203 | 0.502 | 0.970 | 0.928 | **0.919 (218)** | 0.987 (183) |
| frp2_phase2 | 0.67 x MDE | 0.945 (223) | 0.208 | 0.413 | 0.534 | 0.936 | 0.917 | **0.912 (231)** | 0.944 (224) |
| frp2_phase2 | 0.5 x MDE | 0.675 (256) | 0.382 | 0.459 | 0.483 | 0.670 | 0.663 | **0.661 (258)** | 0.674 (256) |

**Joint error.** This is P(PASS and a true regression exactly at a protected margin), maximum
over all gate cells: 3 pools x 6 mass effects plus 4 rescaled cells, every regressed mix.

| Rule | Pooled -0.05 (all mixes) | One mix -0.10 | One mix -0.075 | One mix -0.05 |
|---|---|---|---|---|
| **E (option 1)** | **0.031** (Wilson upper 0.033) | 0.003 | **0.049** (0.053) | 0.58 |
| E10 (option 2) | **0.047** (0.050) | **0.047** (0.050) | 0.55 | 0.98 |
| D | 0.035 | 0.003 | 0.051 | 0.60 |
| C | **0.258** | 0.009 | 0.102 | 0.73 |
| A_rci (frozen) | 0.000 | 0.000 | 0.001 | 0.016 |

A one-mix 0.05 drop passes through the gate at the MDE or above:

- 0.27-0.40 under option 1;
- 0.66-0.84 under option 2.

### What N_max <= 300 cannot deliver: a per-mix 0.05 guarantee with a usable pass rate

A drop of 0.05 confined to **one** mix is visible only in that mix's data. No rule, pooled or
otherwise, can detect it better than the single-mix test of 0.05 against 0.

At n = 275 and a between-checkpoint per-mix SD of about 0.36, that test's separation is
0.05 / (0.36 / sqrt(275)) = 2.30 standard errors. If a single-mix 0.05 drop may pass at most
10% of the time, then:

- a candidate with no regression passes that mix at most Phi(2.30 - 1.28) = 0.85 of the time;
- over the three mixes, that is about 0.61 at best.

Meeting both "a one-mix 0.05 drop passes <= 10%" and "no regression passes >= 80%" needs about
**390 worlds per mix** (about 500 at 5%). The owner's cap is N_max <= 300.

What the proposed rule protects at alpha = 0.05:

- a survival regression of 0.05 **on average over the mixes**, which includes a 0.05 drop in
  every mix;
- with option 1, any **single-mix drop of 0.075** or more.

It catches a 0.05 drop confined to one mix 60-73% of the time (it passes 0.27-0.40). Option 2
caps only a single-mix drop of 0.10, and passes a one-mix 0.05 drop 0.66-0.84 of the time.

## The rule (`band_policy = "pooled_ni_continue"`)

The bands are `{"metric": "survival_fraction", "mix": m}`, one per mix, for exactly the three
mixes frozen, scripted and mixed. For world i of the look-k prefix (world order pre-declared;
n_k = `look_sizes[k]`):

    d_{i,m} = candidate_{i,m} - incumbent_{i,m}            (survival_fraction, same world)
    p_i     = (1/3) * (d_{i,frozen} + d_{i,scripted} + d_{i,mixed})   (pooled per-world delta)

At look k the bands pass iff **all** of the following hold:

1. **Pooled NI:** `mean(p) - t_{n_k-1}(band_nominal_p[k]) * sd(p) / sqrt(n_k) > -band_ni_margin`.
2. **Per-mix catastrophic NI**, every mix m:
   `mean(d_m) - t_{n_k-1}(band_nominal_p[k]) * sd(d_m) / sqrt(n_k) > -band_mix_margin`.
3. **Absolute floor** (if pre-registered), every mix m: `mean(candidate_m) >= band_floor`.

The comparisons are strict `>`. With zero spread, the bound is the mean.

**Levels.** `band_bound` must be `rci_obf`: `band_nominal_p[k]` is the nominal level of a
Lan-DeMets O'Brien-Fleming spending of `band_alpha` over the plan's looks. For N_max 275 and
alpha 0.05 the levels are 9.12e-5, 5.63e-3, 2.21e-2 and 4.26e-2. A `pointwise` bound is
refused, because the band is judged at several looks.

**Timing and decisions.**

- **Qualification is unchanged:** at least `required_successes` mixes have crossed, and
  scripted NI is established. Both are sticky.
- **Before qualification** the decisions are unchanged: `CONTINUE` or `STOP_FUTILE`.
- **At the qualifying look and every later look k < last:**
  - `STOP_PASS` if the bands pass at k;
  - otherwise the new decision **`CONTINUE_BANDS`**. The gate plays the next look, efficacy
    and NI stay established, and futility is no longer evaluated for stopping.
- **At the last look:** `FINAL_PASS` iff the run qualified and the bands pass at the last look,
  else `FINAL_FAIL`.
- `STOP_FAIL_BANDS` never occurs under this policy.
- **Efficacy and NI are unaffected.** Bands can only turn a qualified run into a fail, never a
  non-qualified run into a pass. Once a run continues past its qualifying look, receipts keep
  reporting the sticky efficacy crossing; later looks do not re-judge efficacy.

### The guarantee

Each component is a repeated confidence bound (RCI) at level `band_alpha`. Suppose the true
pooled delta is `-band_ni_margin` or lower. Then the chance that the pooled bound clears at
**any** look is at most `band_alpha` (up to the t-for-z approximation, which is conservative).
The same holds for each mix's bound at `-band_mix_margin`.

This holds at any data-dependent qualifying or stopping look, so it holds however the
correlated mass deltas select the look. A PASS requires every component to clear at the same
look, so for each protected regression P(PASS and that regression) <= `band_alpha`. The rule
is intersection-union: no multiplicity share is needed across components or mixes.

**Measured joint error, option 1:**

- pooled 0.05 regression: at most 0.031;
- one-mix 0.075 regression: at most 0.049 (Wilson upper 0.053);
- FRP-v3 per-study check: at most 0.050.

This is the step that removes the old rule's weak spot. A pointwise bound judged at a
data-dependent look is not protected; that was the 0.0612 failure of `pointwise` in FRP-v3's
per-study check.

### Choice of the per-mix margin (criterion and how it was set)

**Criterion.** `band_mix_margin` is the smallest value in {0.05, 0.075, 0.10, 0.125} for which
the final-look band passes a no-change candidate with probability **>= 0.80** in **every**
independent between-checkpoint pool. The final-look band is judged at n = 275, level
`band_nominal_p[last]`, pooled margin 0.05. The pools are FRP-v2 Phase 2, FRP-v2 Phase 1 and
each of the six Phase 1 cells: the realistic range of between-checkpoint SDs. The smallest
margin that meets the owner's no-regression target is the one that best protects single-mix
regressions.

**How it was set.** The criterion was set during design, with the independent results visible.
It was not pre-registered before any simulation. The first version used a 0.90 floor, which
selected 0.10. The independent review pointed out that 0.90 was not the owner's target. At the
owner's 0.80 the same procedure selects 0.075. `simulate.py` refuses to run if the selection
differs from its constant.

| `band_mix_margin` | Worst pool (no-change pass) | Meets 0.80 | Stress SD 0.441 | One mix -0.05 passes (frp2_phase2, worst mix) |
|---|---|---|---|---|
| 0.05 | C@60000 (0.339) | no | 0.17 | 0.03 |
| **0.075** | C@60000 (0.830) | **yes (chosen: option 1)** | 0.64 | 0.39 |
| 0.10 | C@60000 (0.973) | yes (option 2) | 0.90 | 0.83 |
| 0.125 | M@30000 (0.982) | yes | 0.94 | 0.89 |

**Trade-off of option 1.** It is less robust to unusually noisy survival. With every mix at SD
0.441, its no-change pass rate at N_max is 0.64. Option 2 buys robustness (0.90) at the cost of
tolerating one-mix losses up to 0.10.

**Other values.** The pooled margin stays the ratified 0.05, so the tolerated average survival
loss is unchanged, and alpha stays the ratified 0.05.

### Cost

Qualified runs whose bands do not clear yet continue one to three looks. Expected worlds per
mix rise:

- efficacy and NI alone, at the MDE or the planned effects: about 147-177;
- option 1: about 214-219;
- option 2: about 174-188.

The worst case stays at N_max = 275, so the RunPod sizing (sized for every look) and the
Mac-fallback cap disclosure do not change. Expected spend rises with E[worlds]: about +20-45%
under option 1 and about +5-20% under option 2.

### Descriptive only: how the rule would treat FRP-v3 Phase R

`outputs/descriptive_frp3_phase_r.json` runs the same replica on the FRP-v3 pool
(`research/frp3_strict_20261005/paired_pool.json`). **It is not an input to any choice above.**

| Pool (FRP-v3, descriptive) | Survival scenario | A_rci (frozen) | **E (option 1)** | E10 (option 2) |
|---|---|---|---|---|
| candidate's own worlds (`frp3_s12`), observed mass | observed survival deltas (+0.052 / +0.023 / +0.131) | 0.267 | **0.965** | 0.994 |
| same | survival centred to no change | 0.016 | **0.828** | 0.967 |
| all five Phase R seeds, observed mass | observed (+0.014 / +0.025 / +0.033) | 0.235 | **0.980** | 0.997 |

- The frozen rule's 0.267 matches the FRP-v3 protocol's 0.27, which checks the replica.
- The candidate's own scripted paired SD (0.476) exceeds every independent cell (at most
  0.441). Under option 1 its no-change pass rate is 0.83.

## Per-study pre-registration check (required)

A study that opts in runs:

    simulate.py --part study --data <the study's paired pool> --plan-params <frozen plan doc> \
        --delta-ni <development delta_NI> --reps 20000 \
        --thetas <0.5 MDE>,<0.67 MDE>,<MDE>,<1.5 MDE>,<2 MDE>[,<planning effect>]

**Scenarios simulated:**

- no change;
- the pooled regression at its margin (-`band_ni_margin` in every mix);
- each one-mix regression at -`band_mix_margin`.

**Acceptance:** every joint rate P(PASS and that regression) is at most 1.2 x `band_alpha`
(0.06).

**Binding.** The output, the pool and the simulator (`simulate.py`) are bound by sha256 in the
intent. `prepare` re-judges them, and so do `validate_intent` and the audit, each in its own
code.

**If the check fails, the study does not adopt this policy.** No margin, alpha or floor may be
retuned on the study's own data. The check is a validity check of the guarantee on the study's
data shape, not a calibration.

## Implementation (opt-in; legacy unchanged)

- **`src/evaluation/sequential_gate.py`.**
  - `band_policy = "pooled_ni_continue"`, with a new plan field `band_mix_margin`;
  - `pooled_band_check` and `plan_pooled_band_check`;
  - the `CONTINUE_BANDS` decision.

  Under the legacy policies the new field must be `None` and is left out of `as_dict()`.
- **`research/sequential_strict_template/` (`sequential-strict-template/v2-pooled`).**
  - Usable under the v3 remote intent.
  - `--paired-band M,alpha,bound,floor,mix_margin` (five values under this policy).
  - Look receipts hold the pooled band rows per look and the judged looks.
  - `CONTINUE_BANDS` maps to the action `continue`.
  - `judge_pooled_check` judges the per-study check.
  - The amendment binding `band_amendment` (path, sha256, ratified, option, changes) is
    re-derived from this document's text. A production intent needs exactly `- Decision:
    ratified`, an option, no changes, and plan values equal to that option.
  - `remote-plan` lists a missing ratification as a blocker.
- **`sequential_audit.py`.** In its own stdlib code it recomputes:
  - the pooled and per-mix bounds and the floor at every look, from the raw record pairs;
  - the OBF levels;
  - the decision sequence, including `CONTINUE_BANDS`;
  - the per-study check, including the simulator sha256;
  - the amendment binding.
- **Tests:** legacy golden hashes, the arithmetic, the decision sequence, the runner/audit
  agreement, the ratification parsing and forged-ratification refusal, and the simulator
  binding.

## Pre-registration items (in addition to the earlier amendments)

- **Band settings in the hashed plan:** `band_policy`, `band_ni_margin`, `band_mix_margin`,
  `band_alpha`, `band_bound` (`rci_obf`) and `band_floor`.
- **Per-study check:** its output, its pool and the simulator, bound by sha256.
- **Protocol:** it states the expected pass probability under this rule and the descriptive
  application to the study's own pilot.
- **This document:** bound by sha256 and ratified (exact decision, option, no changes) before
  a production `prepare`.

## Disclosed limits

- **Shift model.** The shift model borrows each pool's per-world delta shape for hypothetical
  regressions. `survival_fraction` is bounded in [0, 1], and additive shifts ignore the bound.
- **Normal approximation.** The RCI guarantee rests on the t-matched normal approximation. The
  bootstrap measures the joint error on real, skewed, tie-heavy deltas: at most 0.049 under
  option 1.
- **One lineage.** All pools descend from one champion (champion_a5) with the v7/v8 veto. A
  candidate from another lineage re-runs the per-study check.
- **Equal weights.** Pooling weights the three mixes equally, so a regression in one mix is
  diluted by 3 in the pooled component. The per-mix catastrophic floor covers that. Receipts
  report every per-mix bound, so a tolerated one-mix loss is visible.
- **Floor.** The absolute floor (0.30) is a wiring tripwire, not a test. It catches no
  realistic regression.
- **Cross-check size.** The cross-check of the simulation replica against the implementation
  compares 1,200 replicates (frp2_phase2: four batches of 300). The FRP-v3 package's own
  cross-check adds 600, and both found 0 mismatches.

## Evidence files

- `research/survival_band_v2_20261006/extract.py` and `survival_pools_20261006.json`: the
  calibration pools, from independent data only.
- `research/survival_band_v2_20261006/simulate.py`: parts `oc`, `descriptive`, `study`,
  `crosscheck` and `design-plan`.
- `outputs/oc.json`: the design OC, about 70 s on one core, holding one shared CPU slot.
- `outputs/descriptive_frp3_phase_r.json`: descriptive only.
- `outputs/design_plan_pooled.json` and `outputs/crosscheck.json`: the replica against
  `run_sequential_gate` + `plan_pooled_band_check`, 0 of 1,200 decisions differ.

## Ratification

**Pending.** To be completed by the research loop owner (or the user). Ratify one option:

- **Option 1 (recommended):** `pooled_ni_continue` with `band_ni_margin` 0.05,
  `band_mix_margin` 0.075, `band_alpha` 0.05, `band_bound` `rci_obf`, `band_floor` 0.30.
  `--paired-band 0.05,0.05,rci_obf,0.30,0.075`.
- **Option 2 (more permissive per mix):** the same with `band_mix_margin` 0.10.
  `--paired-band 0.05,0.05,rci_obf,0.30,0.10`. A study that uses it must regenerate its plan
  and its per-study check with that value.

The code accepts only these two options. A changed rule needs its own amendment and
implementation, so "ratified with changes" is refused.

Questions for the owner:

1. **One-mix losses.** Do you accept that a survival loss confined to one mix is tolerated
   below the per-mix margin (0.075 under option 1, 0.10 under option 2)? A per-mix 0.05
   guarantee is not attainable at N_max <= 300 with a usable pass rate.
2. **Continuation cost.** Do you accept the continuation cost: higher expected worlds and the
   same worst case?

Record the decision with exactly these lines. Write `- Decision: ratified` to adopt, and leave
the Changes line `none`:

- Decision: ratified
- Option: 1
- Date: 2026-10-06
- Changes, if any: none

Ratified by the research loop owner under the user's standing decision authority (user,
2026-10-01: "i want you to make those choices going forward"; user, 2026-10-05, choosing
"Fix band design first" for the FRP-v3 strict gate). Answers: (1) yes, single-mix losses below
0.075 are tolerated, as disclosed; (2) yes, the continuation cost is accepted.
