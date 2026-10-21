# O1 excluded timing-fixture correction — before study data

The original preregistration, bank, audit and `REGISTRATION_O1.json` remain
unchanged. No study world or Colab call has run. This sibling correction changes
only timing accounting in the runner/forecaster. Hypothesis, worlds, root rule,
provenance attribution, one scientific look, KILL/GO rule and caps are unchanged.

The first excluded Mac entrypoint run completed at code `adc4aa4d`, frozen by
registration commit `df59622c`, in 14.40 s wall (13.864477 s inside runner).
Receipt: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-o1-20261021/preflight/run-01/receipt.json`.
It is preserved with its entire over-cap forecast: 16842.952127 s /9.357196 u.
That forecast conservatively used the slowest whole-world mean base-step time,
which amortized first-step compilation over only40 steps and then multiplied
that cost into every projected H5000 step.

Fix, outcome-independent: measure each world's first transition separately;
charge it in per-world setup, and estimate recurring base transitions from the
remaining measured transitions. All actual first-step time remains charged,
including on the L4 fixture. Continue to use worst fixture rates, rounded-up
worker waves, 2x caution, setup/export allowance, live session accounting, and
all registered caps. The L4 fixture still gates bank admission; Mac extrapolation
is never a measured L4 rate or serving timing claim.

Publish corrected code and a new sibling source registration
`REGISTRATION_O1_TIMING_V2.json` BEFORE the second excluded Mac fixture. Use the
same two excluded worlds (no replacement/reroll), same frames40/roots10,20/H8,
nice10/one numerical thread, fresh `preflight/run-02/` outputs. Preserve both
attempts. This is an entrypoint timing correction before study data, not a second
scientific look or an extension of the48-world study. The original registration
names the retained earlier code; only the sibling registration is used for the
prepared corrected manifest. No dispatch is authorized.

The corrected timing gate also charges already consumed supervisor allocation,
bootstrap and input-delivery time against the timing fixture's30-minute/1-unit
ceiling. It uses the broker's shortened timeout plus runtime session clock;
setup is never treated as free time before that ceiling. The outer8-unit and
ordinary6.4-unit caps are unchanged.
