# Governance amendment: group-sequential Tier-2 gates, 2026-10-02

Status: **proposed amendment** to Tier 2 of
[governance_tiers_2026-09-26.md](governance_tiers_2026-09-26.md). The user decided on 2026-10-02
that strict gates going forward should be group-sequential: pre-registered interim looks, alpha
spending, early efficacy and futility stops, and interleaved mixes. This document turns that
decision into rules.

It applies only to Tier-2 studies whose pre-registration is written after the amendment is
adopted. It does not apply to the v7 strict run now executing
(`research/apex_veto_v7_strict_20261002`, fixed N = 243) or to any earlier package, and it
changes nothing in `eval_stats.strict_promotion_decision`. The fixed-N gate stays available and
unchanged.

Implementation: `src/evaluation/sequential_gate.py` (opt-in), tests in
`tests/test_sequential_gate.py`, and the operating characteristics in
[research/sequential_gate_validation_20261002/README.md](../../research/sequential_gate_validation_20261002/README.md).

## Why

A fixed-N strict gate plays every world even when the answer is obvious early. A true effect
of 130 mass per mix, the size the v7 screen observed (+135 to +154, a screen estimate and not
evidence), passes a sequential gate at the first quarter of N_max in 99.8% of simulated runs. That is about 61 rather than 243 worlds per mix, saving roughly
three quarters of the final-stage compute. A null candidate stops for futility at the half or
three-quarter look in 89% of runs. At the MDE, the cost is about 2 points of power against
fixed Holm (94.3% vs 96.3% at SD 140, delta 30). Type-I error stays at or below 0.05 with
normal deltas (validation README). These figures assume the bands pass; item 5 gives the band
cost at early stops, and item 9 the skew limit.

## The design (method `strict-sequential-obf-bonferroni-v1`)

1. **Looks.** K looks at pre-declared information fractions of N_max paired worlds per mix. The
   default is 0.25, 0.5, 0.75, 1.0, with look sizes `ceil(f x N_max)`. Alpha is spent at the
   actual fractions `look_size / N_max`.
2. **Efficacy per mix.** One-sided `H0: delta <= 0`. Lan-DeMets O'Brien-Fleming-type spending
   at `0.05 / 3` per mix (Bonferroni across the three mixes). A mix is rejected at the first
   look where its paired one-sided t p-value is at or below the look's nominal level
   `1 - Phi(c_k)`, and it stays rejected.
3. **Why Bonferroni rather than Holm.** Bonferroni keeps familywise error at or below 0.05 under
   any dependence between mixes, which share world seeds, and under any stopping. A PASS
   therefore keeps the fixed gate's guarantee: with probability at least 0.95, every mix it
   counts is truly superior. A Holm-type sequential procedure is valid (Maurer and Bretz 2013,
   graphical approach), but it recomputes boundaries at recycled levels at every look,
   including past ones, which is harder to audit. It would recover about 1.5 points of power at
   the MDE. It is a possible later version (`-v2`) and is not part of this amendment.
4. **Scripted noninferiority.** Its own OBF spending at 0.05. NI is established at the first
   look where `mean - t_{n-1}(nominal_k) x se > -delta_NI` (strict `>`). NI is a conjunct of
   PASS, so it takes no multiplicity share.
5. **Behavioral bands (`band_policy = "block_at_stop"`).** Bands are judged once per run, at
   the *qualifying look*: the first look where at least 2 mixes have crossed and NI is
   established. They never delay a stop. If they fail there, the gate ends
   `STOP_FAIL_BANDS`. Looks before the qualifying look do not judge bands. At an interim
   qualifying look, every band must hold with a margin, through `band_check`. The margin is
   `band_margin_z x sd x (1/sqrt(n_k) - 1/sqrt(N_max))`, with `band_margin_z` pre-registered
   (default 1.645). At the last look the margin is 0, which is the fixed-N point rule.
   - **Why.** A band is a point threshold on a mean, for example candidate survival_fraction
     at or above the reference minus 0.02. Re-checking it at every look, as the first draft
     did, gives a band-violating candidate up to four chances, with only 61 worlds per mix at
     look 1. Simulation (validation README, `skew_probe.py --part bands`): for a candidate
     whose violation the fixed-N gate catches 95% of the time (true mean 1.645 final-N
     standard errors outside the band) and that stops at look 1, the old rule let the band
     pass 26-27% of the time. Delaying instead of failing, even with the margin, gave 11-12%.
     `block_at_stop` gave 5.0%, the same as fixed N (4.8-5.0%).
   - **Guarantee.** With band data independent of the efficacy data, a candidate that the
     fixed-N band catches with probability at least `Phi(band_margin_z)` (95% at the default)
     is caught at an interim stop with at least that probability. For smaller violations,
     the interim check is stricter than the fixed one.
   - **Cost.** A good candidate that stops early can fail the band by chance, because the
     margin is applied to only 61 worlds. With true survival 2 final-N standard errors
     inside the band, the band passed 57-58% of the time at a look-1 stop, against 98% at
     fixed N. The study's pilot survival SD sets how large this is in band units, so the
     pre-registration reports it (below).
6. **Futility (non-binding).** At interim looks, a mix that has not crossed is futile when its
   conditional power under the pre-registered MDE is below 0.10. CP uses the current B-value,
   drift `MDE x sqrt(N_max) / sd_hat`, and the final boundary. The gate stops for futility
   when the remaining non-futile mixes cannot reach 2 successes. The efficacy boundaries are
   computed ignoring futility, so following or overriding a futility stop cannot raise
   type-I error.
7. **Decisions.**
   - At an interim look: `STOP_PASS` (qualifying look, bands pass), `STOP_FAIL_BANDS`
     (qualifying look, bands fail), `STOP_FUTILE`, or `CONTINUE`.
   - At the last look: `FINAL_PASS` (qualifies and the bands pass) or `FINAL_FAIL`.
   - Outcome mapping: `STOP_PASS` and `FINAL_PASS` become `STRICT_PASS`. `STOP_FAIL_BANDS`,
     `STOP_FUTILE` and `FINAL_FAIL` become `STRICT_FAIL`. Both still need audit PASS, exactly
     as today.
8. **z/t approximation.** The boundaries are exact on the z scale (recursive integration). Each
   look's paired t statistic is compared with its own t distribution at the boundary's nominal
   level. The validation measured no detectable inflation at 61 or more worlds per look with
   normal data.
9. **Skew sensitivity, shared with the fixed-N gate.** The paired t test is not robust to
   strongly left-skewed deltas (a candidate that is occasionally much worse), and the early
   looks add a little to that. Single-mix probe at the null, 300k replicates per row,
   gamma-shaped deltas with SD 140 (`skew_probe.py --part gamma`):

   | Skewness | Efficacy any-look (target 0.0167) | Fixed-N at 0.05/3 | Look-1 crossing (nominal 1.77e-6) | NI any-look (target 0.05) | Fixed-N NI |
   |---|---|---|---|---|---|
   | -2.83 | 0.0397 | 0.0323 | 1.26e-3 | 0.0877 | 0.0720 |
   | -1.0 | 0.0232 | 0.0218 | 3.7e-5 | 0.0608 | 0.0573 |
   | 0 (normal) | 0.0165 | 0.0166 | 3.3e-6 | 0.0498 | 0.0501 |
   | +2.83 | 0.0067 | 0.0076 | 0 | 0.0291 | 0.0329 |

   Right skew is conservative in both gates. At strong left skew, the fixed-N gate is already
   about twice its nominal level, and the sequential gate adds about 0.007 (efficacy) and
   0.016 (NI). A larger first look (fractions 0.5, 0.75, 1.0) did not help: 0.0401 and
   0.0849. The check below is therefore required, and its remedy is a different statistic,
   not a different look schedule.

## What must be pre-registered (in addition to the existing Tier-2 list)

All of the following are frozen in `intent.json` before any final-stage episode runs:

- **Method version and plan.** The method version string and the full plan
  (`SequentialGatePlan.as_dict()`), with its sha256 over canonical JSON. That covers mixes,
  scripted mix, N_max, look sizes, fractions, family alpha, per-mix alpha, NI alpha, every
  boundary and nominal level, required successes, MDE, the futility CP threshold, the
  futility policy, the band policy and the band margin z.
- **Spending function.** The spending function by name (Lan-DeMets O'Brien-Fleming) and the
  multiplicity rule (Bonferroni across mixes).
- **Futility rule and policy.** The futility rule, and whether futility stops are followed,
  as the plan field `futility_policy` (`followed`, the default, or `overridable`). It is
  part of the hashed plan. Overriding is allowed only under `overridable`, and every
  override is disclosed. Under `followed`, `sequential_decision` marks any look evaluated
  after a futility stop `valid = false` with `unregistered_futility_override = true`.
- **Band policy.** `band_policy = "block_at_stop"` and `band_margin_z` (default 1.645), both
  in the hashed plan, plus every band's metric, bounds and mix scope. Also report the
  early-stop band cost: using the pilot candidate SD of each band metric, the probability
  that a candidate equal to the reference fails the band at a look-1 stop.
- **delta_NI.** It is computed by the calibration stage before the first final world, as today,
  and is frozen from then on.
- **Interleaving plan.** The world-major round-robin order from `round_robin_plan`: unit
  `u = world_index x 3 + mix_position`, and worker `u % 2` runs both arms of that world. Also
  the per-worker unit counts at each look (`worker_look_counts`) and the ordered final world
  bank.
- **Operating characteristics.** A simulation of the frozen plan in the style of
  `research/sequential_gate_validation_20261002/simulate.py`, run with the pilot SD. It
  reports type-I error under the global null, the least-favorable configuration and the NI
  null, plus power and expected worlds at the MDE.
- **Resampling operating-characteristics check (must pass before the first final world).**
  Run `research/sequential_gate_validation_20261002/skew_probe.py --part resample` on the
  study's own saved paired deltas (screen or pilot, Tier 0 data), once per mix, with the
  frozen N_max and delta_NI. The script centres the deltas at the efficacy null (mean 0) and
  at the NI null (mean -delta_NI), resamples with replacement, and reports the per-mix
  any-look crossing rates. Acceptance thresholds, fixed by this amendment:
  - per-mix efficacy any-look rate at most 1.2 x 0.05/3 = 0.020;
  - scripted NI any-look rate at most 0.06.

  The script's `passes` field applies both. Its outputs and their sha256 go into
  `intent.json`. If the check fails, the study does not proceed under this method.
  Moving the first look later does not fix strong skew (item 9). The pre-registered
  remedy is to stop and choose one of two paths. Path one is a new method version with a
  skew-robust interim and final statistic, such as a bootstrap-t or a skew-corrected t,
  validated by its own Monte Carlo before use. Path two is a governance decision on the
  fixed-N gate. That gate shares the sensitivity, so the same resampling check applies to
  it. A study may not proceed on a failed check.
- **Sizing.** N_max is sized from development-only variance. It equals the fixed-N requirement
  inflated for the looks (about 2% for four OBF looks), or is set directly by the
  simulation. The existing caps, kill criterion and 70% runtime rule apply to N_max.

Nothing in that list may change after the first final world, including N_max, the looks, the
MDE, the futility threshold and the world order.

## Running the looks

- **Barrier at every look.** Each worker runs its shard in order and pauses after exactly its
  per-look unit count. The look analysis uses exactly the first `3 x look_size` units of the
  plan, which on each worker is a prefix of its shard. No unit beyond a look's prefix may start
  before that look's decision is written.
- **Look receipts.** Each look writes a create-only `looks/look-<k>.json`. It records the
  plan sha256, the record paths and sha256 of every world used, the band verdict, and the full
  `sequential_decision` output. The look computes only the pre-registered decision. Other
  metrics are not looked at until the run closes.
- **Stopping.**
  - `STOP_PASS` ends the final stage.
  - `STOP_FUTILE` ends it unless an override was pre-registered.
  - Evaluating a look after an efficacy stop is invalid; the module flags it.
  - The serving and audit stages follow as today.
- **Never stop a fixed-N run early.** A study pre-registered as fixed N, including the v7 run
  executing now, must not have interim analyses and must not be stopped or extended on data.
  Converting a running fixed-N study to sequential is forbidden. A stop for a mechanical cause
  (crash, cap, deadline) stays `INVALID_STOP` or `INCOMPLETE`, as today.

## Audit requirements (additions)

- **Recompute the boundaries independently.** The auditor computes the spending-function
  boundaries with its own code, not with `sequential_gate.py` or `screen_stats.py`. Acceptable
  methods are an independent recursive integration, or a stdlib numerical integration checked
  against published gsDesign values. For example, four equal looks at one-sided 0.025 give
  4.3326, 2.9631, 2.3590 and 2.0141. Its boundaries must match the intent's plan within 2e-4
  on the z scale.
- **Recompute every look.** From raw records, the auditor recomputes every look up to the stop:
  per-mix paired t and nominal-level comparisons, NI lower bounds, conditional power, the
  qualifying look, the bands at that look only (mean, SD and the `band_check` margin) and the
  decision sequence. It verifies that no earlier look should already have stopped (or that the
  override was pre-registered), and that the stop look's decision matches the look receipt.
- **Check prefix integrity.** The auditor checks that every look used exactly the plan-prefix
  units, that the per-worker counts match `worker_look_counts`, and that no record outside the
  prefix existed when the look receipt was written. Use shard reports and create-only order;
  do not trust wall-clock times.
- **Cross-check the reducer.** As today, the frozen reducer is cross-checked against a minimal
  second implementation, a plain mean and t over the prefix.

## Reporting

- Report the stop look, worlds used and the decision.
- After an early efficacy stop, the naive mean delta is biased upward (truncation). Label
  per-mix means as naive and descriptive. If an effect size is quoted, give a repeated
  confidence interval at the stop look, or a stage-wise-ordering median-unbiased estimate. The
  decision never depends on these.
- A `STOP_FUTILE` result means the gate failed. It does not mean "no effect" (wording rule from
  the Tier-1 standards).

## Disclosed limits and not yet done

- No strict runner implements the look barrier, look receipts or the sequential audit yet. That
  is the next lane, as a new package that does not modify existing ones.
- **Skew.** Type-I control is shown for normal deltas only. Strong left skew inflates both
  gates (item 9). The resampling check in the pre-registration list is the safeguard, and it
  has a pass criterion. No skew-robust statistic is implemented yet.
- **Bands are not yet calibrated for selection.** The `block_at_stop` guarantee assumes band
  data independent of the mass deltas. With per-world correlation 0.5 between survival and
  mass delta, and a delta-30 candidate, the band-edge pass rate was 6.4% against 5.75% for
  fixed N. Fixed N shows the same selection effect, but less of it. Band metrics with
  stronger dependence on mass are not simulated. A study whose band metric is strongly
  tied to the efficacy metric should simulate its plan with resampled joint records before
  adopting it.
- **Bands cost power at early stops** (item 5): a good candidate can fail a band on 61
  worlds. This is a deliberate trade: the gate fails rather than letting a band violation
  through.
- The Holm-type (graphical) sequential variant is not implemented.
- Mixes run together until the overall stop. Dropping a mix that has already crossed or is
  futile would save more worlds, but it complicates NI and the bands and is not proposed.
