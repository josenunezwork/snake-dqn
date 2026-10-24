# R32-5 prospective complete-path comparison

This is the frozen implementation specification for the narrow R32-5 engineering comparison ordered by `round-32/JUDGMENT.md` §5. It does not change F4, any current gate, a serving configuration, or a future registration.

The only candidates are: equal fixed allocation at 128 paired worlds per mix; one externally frozen cost-aware fixed allocation; and the equal 128/mix design with one futility-only look after 64 complete paired worlds per mix. A 32-world look, interim efficacy, adaptive allocation, alpha recycling, GPU simulations, and use of early-finishing episodes are excluded.

`STATISTICS_PROTOCOL.json` freezes the scenario family, endpoint, estimator, error/guard accounting boundary, costs, trials, unit namespace, and selection rule before computation. `statistics.py` is intentionally limited to deterministic prospective arithmetic and simulation primitives. It has no study allocation, admission, job lifecycle, or dispatch behavior.

## Evidence status and cheapest stop

The inventory is deliberately limited to checked-in published evidence. The three B8-only timings in `docs/research/q0_mass_mac_timing_20261018.json` are useful only as an illustrative timing sensitivity. The campaign log records an operational wait, not joint paired-delta/cost calibration. Neither establishes per-mix candidate-minus-B8 paired-delta variance, complete-pair cost, Watch session cost, covariance where needed, or complete-path cost.

Accordingly the frozen inventory is inadequate for adoption. The implementation must report sensitivities only and returns `RETAIN_FIXED_DESIGN`. If a future reviewed evidence manifest is adequate, the first calculation is `favorable_bound`: it prices setup, screen pairs, confirmation, Watch, strict, serving, queue delay, and handoff. If its favorable bound cannot reach 20% at +40, stop before simulation.

## Exact futility model

The sole interim contains exactly 64 complete paired worlds in each of scripted, frozen, and mixed. With pre-frozen external mix SDs, the remaining 64 paired deltas per mix are generated from independent Student-t innovations with 5 degrees of freedom, shifted by the declared +40 remaining-effect mix profile and scaled to retain each supplied marginal SD. This is a heavy-tail sensitivity construction, never empirical calibration.

For every predictive draw, append the remaining block to the observed block. The final mass requirement passes only if the equal-weight mix mean is at least +40 and its one-sided 90% Student-t lower bound (over the 128 paired-world means) is strictly above zero. Conditional probability is the pass fraction. A probability below 0.05 means `FUTILE`; it never means efficacy or no effect. Missing blocks produce `INCOMPLETE`. Survival and Watch bars have no interim look and must be simulated/reported as final complete-contract components by the package owner.

The conditional probability is exact only for its frozen-nuisance Gaussian model; it is not claimed to be the exact probability under the later heavy-tail outer sensitivity. The outer comparisons report Wilson 95% uncertainty, including at zero or one events. Tiny Mac preflight uses 200 trials. The six homogeneous effects −20, 0, +20, +40, +60, and +80 are critical 100,000-trial CPU units; heterogeneous, heavy-tail, guard-boundary, partial-null, and correlated mass/survival/Watch sensitivities use 10,000 trials. IDs are stable under `r32-r5-statistics/20261023/v1/scenario/<index>`. The shared CPU envelope is 2.0 forecast hours / 0.16 u, 2.4 hours / 0.192 u with reserve, and 4.8 hours / 0.384 u under the owner 2× ceiling. No job is dispatched by this module.

## API and prepared commands

```bash
nice -n 15 ./venv/bin/python research/r32_rehearsal/statistics.py
nice -n 15 ./venv/bin/python research/r32_rehearsal/statistics.py --print-units
```

`optimal_cost_allocation(sd, cost)` implements only the unconstrained Neyman starting point, including the P7 floor of 16 pairs/mix. `minimum_cost_allocation_for_precision` instead asks the relevant question: the lowest-cost integer allocation matching equal-128/mix precision. With no eligible external calibration, the result table names cost-aware allocation `INELIGIBLE_RETAIN_EQUAL`; it never invents a cost model. It reports Mac execution and elapsed decision time as separate unavailable values, with queue/handoff belonging only to elapsed time. A package must separately revalidate survival A/B, product, finite-sample inference, multiplicity, and complete-path power before treating an allocation as eligible. `conservative_decision(...)` deliberately cannot emit adoption: a future Astra-judged pre-data registration remains required even if all numerical thresholds are met.
