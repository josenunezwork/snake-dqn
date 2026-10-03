# Governance amendment: paired survival bands, 2026-10-03

Status: **proposed amendment** to item 5 ("Behavioral bands") of
[governance_amendment_sequential_gates_2026-10-02.md](governance_amendment_sequential_gates_2026-10-02.md)
and to the band rule of the Tier-2 strict gate in
[governance_tiers_2026-09-26.md](governance_tiers_2026-09-26.md).

**Scope.**
- **Who it applies to.** Only Tier-2 studies whose pre-registration is written after the
  amendment is adopted, and only when that pre-registration opts in through the plan field
  `band_policy = "paired_ni_at_stop"`.
- **Never retroactive.** It does not apply to the v8 strict gate now running
  (`apex-veto-v8-strict-20261003`, plan hashed with `band_policy = "block_at_stop"`), to the
  completed v7 strict run, or to any earlier package. That gate is judged by the rule it
  pre-registered, and its band verdict must not be recomputed, replaced or overridden under
  this rule. A paired survival summary of that run, computed after it closes, is descriptive
  only and must be labelled that way.
- **What stays.** The default `block_at_stop` and every existing plan dict and sha256 are
  unchanged (tested against golden hashes).

Implementation: `src/evaluation/sequential_gate.py` (opt-in), tests in
`tests/test_sequential_gate.py`. Evidence:
[research/paired_band_validation_20261003/README.md](../../research/paired_band_validation_20261003/README.md).

## Why

The current survival band requires, for every mix:

    mean candidate survival_fraction (final worlds) >= reference_mean - 0.02

`reference_mean` is the incumbent's mean over 16 calibration worlds.

**The reference is noisier than the band.** Per-world survival SD is 0.24-0.29, so the
16-world reference has an SD of about 0.06, three times the 0.02 band. The rule therefore
tests mostly the calibration draw, and more final worlds never fix that.

**It ignores the pairing.** The candidate is compared with the incumbent on other worlds,
although the incumbent plays every final world and the per-world arms correlate 0.56-0.74.

**Measured on resampled real records** (v7 strict final stage and v8 screen; validation
README):

| | n = 63 | 125 | 187 | 249 |
|---|---|---|---|---|
| No regression: all three bands pass (as run, `block_at_stop`) | 0.09-0.11 | 0.16-0.17 | 0.20 | 0.23 |
| True regression 0.05: that band passes | 0.22 | 0.27 | 0.29-0.30 | 0.31-0.32 |
| True regression 0.02: that band passes | 0.36-0.37 | 0.43 | 0.47 | 0.48-0.50 |

The guardrail removes most good candidates and still lets a 0.05 regression through about
30% of the time.

In the full sequential gate, take a candidate with a mass effect of 30 (the MDE) and no
survival change. Efficacy and NI qualify it 96% of the time, but the band then cuts PASS to
about 0.20 (0.957 x 0.210). The v8 strict design reported a smaller loss under its own
assumptions: full-gate power about 28%, against 55% without the band. Both analyses find that
the band, not efficacy or NI, dominates the gate's power.

## The rule (`band_policy = "paired_ni_at_stop"`)

For each band (metric survival_fraction, one per mix), at the qualifying look k only, with
the look-k prefix of n_k final worlds:

    d_i = candidate_i - incumbent_i            (same final world i)
    PASS_band  iff  mean(d) - t_{n_k - 1}(p_k) * sd(d) / sqrt(n_k)  >  -M
                and (if a floor is pre-registered) mean(candidate) >= floor

- **Margin and alpha.** M = `band_ni_margin` and alpha = `band_alpha`. The level p_k is set
  by `band_bound`:
  - `pointwise`: p_k = alpha at every look (recommended);
  - `rci_obf`: p_k is the nominal level of a Lan-DeMets OBF spending of alpha over the plan's
    looks, which gives a repeated confidence bound.
- **Strict inequality.** The comparison is strict `>`, as for scripted NI. With zero spread,
  the bound is the mean.
- **Judged once.** Bands are judged once per run, at the qualifying look, the same timing as
  `block_at_stop`. They never delay a stop: failure at the qualifying look is
  `STOP_FAIL_BANDS` at an interim look, or `FINAL_FAIL` at the last look. The decision
  function and the outcome mapping are unchanged.
- **Per mix.** Every band must pass. That is an intersection-union rule, so it takes no
  multiplicity share. A regression in any single mix is capped at the per-band level.
- **Floor (optional).** `band_floor` is a sanity tripwire, not a test. It catches a candidate
  whose absolute survival collapsed when the paired delta cannot, for example when both arms
  are broken by an environment or wiring fault. It is a pre-registered constant, frozen in
  the plan (the plan is frozen before calibration), set well below any plausible value.
  For the current veto lineage, about half the development incumbent's survival is
  suitable (development means are 0.56-0.66, so 0.30).
- **Recommended values.** M = 0.05, alpha = 0.05, `pointwise`, floor 0.30 or none.

### Why these values

- **M = 0.02 is not feasible at this survival SD** (paired delta SD about 0.23 per world). A
  candidate with no regression passes all three bands only 6-10% of the time, even at
  n = 249. The old rule's stated 0.02 tolerance was never what it enforced: it passes a 0.02
  regression about half the time and a 0.05 regression about a third of the time. M = 0.05
  is a larger number but a much stronger guarantee.
- **M = 0.05, alpha = 0.05 beats the old rule.** It passes more good candidates at every n
  and caps a 0.05 regression at the nominal 5% per band. A regression of 0.02 sits inside the
  margin and is tolerated by design; it passes 23-67% of the time, depending on n.

  | Recommended rule | n = 63 | 125 | 187 | 249 |
  |---|---|---|---|---|
  | No regression: all three pass | 0.15-0.21 | 0.51-0.58 | 0.78-0.82 | 0.91-0.93 |
  | Improves 0.05: all three pass | 0.93-0.95 | 0.999 | 1.000 | 1.000 |
  | True regression 0.05: that band passes | 0.036-0.045 | 0.040-0.046 | 0.042-0.048 | 0.042-0.049 |

- **Pointwise rather than RCI.** The RCI's guarantee holds under any stopping rule, but it
  passes very little at look 1. At alpha 0.10 (RCI-OBF), it passes 0.000 of no-regression
  candidates and 0.23-0.32 of candidates that improve by 0.05.
- **What the guarantee is.** The guarantee is on the **joint** error: P(the run qualifies and
  a band with a true regression of M passes). That bounds P(gate PASS with that regression).
  - **Why measure it.** A pointwise bound judged at a data-dependent look can be
    anti-conservative: early or rare qualification selects runs with high mass noise, and
    mass correlates about 0.98 with survival per world.
  - **Joint error, measured.** Joint resampling through the full gate covered a 0.05
    regression in each mix in turn and mass effects of 0.5, 0.67, 1, 1.5 and 2 x MDE and
    130, in 36 cells. The joint error was 0.020-0.053. The maximum, 0.053 (Wilson
    0.049-0.058), came from the small, low-variance v8 `mixed` pool. The old rule's joint
    error was 0.07-0.32.
  - **Conditional rate.** The rate given qualification is inflated below the MDE (0.09-0.12
    at 0.5 x MDE), where qualifying is rare. Those runs seldom pass at all, so the joint
    error stays below 0.05. At and above the MDE, no inflation was detected.
  - The per-study check below re-measures the joint error on each study's own data.
- **The known weak spot is look 1.** A candidate whose survival is truly unchanged passes all
  three bands only 15-21% of the time at n = 63 (the old rule as run: 9-11%). The one saved
  run with a look-1-sized mass gain (v7 strict, +119 to +130 mass per mix) also gained
  0.18-0.21 survival, with a per-world mass/survival correlation of 0.98. A flat-survival
  look-1 stop is therefore the less likely case, but it is not ruled out. A study that
  expects an early stop with flat survival may pre-register alpha = 0.10 instead. That gives
  32-40% at look 1, at about a 10% joint cap on a 0.05 regression.

## How it plugs into the sequential plan

- **Plan.** `sequential_gate_plan(..., band_policy="paired_ni_at_stop", band_ni_margin=0.05,
  band_alpha=0.05, band_bound="pointwise", band_floor=0.30)`.
  - The plan freezes `band_ni_margin`, `band_alpha`, `band_bound`, the derived per-look
    `band_nominal_p` and `band_floor`.
  - `as_dict()` includes all five, so they are in the plan sha256 in `intent.json`.
  - Under `block_at_stop`, these fields must be `None` (otherwise a `ValueError`) and are
    left out of `as_dict()`, so legacy plan dicts and hashes are byte-identical.
  - `band_margin_z` stays in the plan for schema stability and is not used by this policy.
- **Per-look verdict.** `bands_pass[k] = all(plan_paired_band_check(plan, k,
  candidate_prefix[m], incumbent_prefix[m])["passes"] for each band mix m)`. The inputs are
  the look-k prefix of both arms' survival_fraction, in the pre-declared world order.
  `sequential_decision` uses only the qualifying look's entry, as today.
- **Calibration.** Calibration still sets delta_NI. It no longer sets band bounds, because
  there is no reference mean. Its survival means are reported descriptively.
- **Look receipts.** They record, per band: n, mean delta, SD, t critical value, lower bound,
  M, p_k, the candidate mean, the floor, and the verdict.
- **Audit.**
  - The auditor recomputes each band at the qualifying look with its own code (plain mean, SD
    and a t quantile), not with `sequential_gate.py`.
  - It checks that the incumbent and candidate values come from the same world record pair,
    and that `band_nominal_p` matches the plan. For `rci_obf`, it recomputes the spending
    independently, as for the efficacy boundaries.
- **Runner and audit template.** `research/sequential_strict_template/` still implements only
  `block_at_stop`: calibration-derived bounds and candidate-only band data. A study adopting
  this policy needs a new package version that passes incumbent prefixes and replaces the
  band recompute in the audit. That work is not done in this lane.

## What must be pre-registered (in addition to the sequential amendment's list)

- **Band choices.** `band_policy`, `band_ni_margin`, `band_alpha`, `band_bound` and
  `band_floor`, all in the hashed plan, plus each band's metric and mix scope.
- **Resampling check on the study's own data.** Run
  `research/paired_band_validation_20261003/simulate.py` on the study's own saved paired
  screen or pilot records (Tier 0 data), with the frozen plan:

      simulate.py --part bands --data <pool.json> --plan-params <plan.json>
      simulate.py --part gate  --data <pool.json> --plan-params <plan.json> \
          --delta-ni <dev delta_NI> --reps 20000 \
          --thetas <0.5 MDE>,<0.67 MDE>,<MDE>,<1.5 MDE>,<2 MDE>,<screen estimate>

  `<plan.json>` is `{"plan_parameters": ..., "plan": ...}`, exactly as frozen in
  `intent.json`. The script rebuilds the plan, refuses to run unless it equals `plan`, and
  then uses its looks, alphas, futility rule and band rule. Under
  `futility_policy = "overridable"` it ignores futility stops, which is the conservative
  bound, because overriding only adds chances to qualify. The script simulates the three
  mixes `frozen`, `scripted` and `mixed` only. `<pool.json>` has the format of
  `paired_survival_20261003.json`: one entry per pool, with `world_seeds` and, per mix, the
  lists `incumbent_survival`, `candidate_survival` and `mass_delta`. delta_NI is not known
  before calibration, so use the development estimate; the check depends on it only through
  the NI conjunct. Report:
  - P(all bands pass) for no regression at each look size;
  - for a regression of exactly M in each mix in turn, at every listed mass effect: the
    joint rate P(qualify and the regressed band passes), and, for information only, the
    rate given qualification.

  **Acceptance:** the joint rate is at most 1.2 x `band_alpha` (0.06 at the recommended
  value) in every row. The `gate` output's `check.passes` field applies this. The output
  sha256 identifies the run, not a reproducible byte string: the output embeds
  `cpu_seconds`.
  If the check fails, use `band_bound = "rci_obf"`, or do
  not adopt this policy. The outputs and their sha256 go into the pre-registration, as for
  the skew check.
- **Changing the values.** The margin M is a policy choice about how much survival loss is
  tolerable. Changing it from 0.05 needs the same simulation and a stated reason.

## Disclosed limits

- The evidence comes from two pools of one network lineage (243 and 60 worlds). Hypothetical
  regressing candidates borrow the observed per-world delta shape.
- The selection check shifts mass and survival independently. That is adversarial for
  selection, but the real joint shape may differ for other candidate families. The per-study
  check covers this.
- Survival_fraction is bounded in [0, 1], and the additive shift model ignores the bounds.
  A candidate close to 1 has a compressed, skewed delta distribution. Its effect on the t
  bound was not simulated.
- The fixed-N strict gate could use the same rule at a single look with p = alpha. That is
  not proposed here.
