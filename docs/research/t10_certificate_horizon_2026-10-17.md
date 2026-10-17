# T10 certificate audit and horizon-stratified AUC — 2026-10-17

**Development diagnostics only.** No policy training, observation/reward change,
promotion evaluation, serving change, GPU or external compute.

## Verdicts

<!-- verdicts:start -->
**Certificate verdict: lookahead does not hold the measured flag far more often.**
With the disclosed tail-by-L deadline, dense lookahead holds it on
144,914/144,958 eligible decisions (**99.970%**) versus B8's 121,158/121,261
(**99.915%**). Thus neither proposed description fits this primary flag: both
policies hold it almost always, while H5000 survival differs, **54/64 versus
38/64**. The 128-step sensitivity also barely separates them: **59.03% versus
58.84%**. Race-128 > L/2 is **85.58% versus 85.81%**, and > L is **66.54%
versus 66.94%** (dense versus B8). These marginal holding rates do not support
the hypothesis that lookahead gains survival by holding these proxies far more
often. They do not validate a safety certificate
or prove that successful play must depend on search.

For deaths with an observed eligible tail-by-L flag, the median time since its
last true decision is **1 decision for dense (6 deaths)** and **0 for B8
(18 deaths)**; another 4 dense and 8 B8 deaths never reach L >= 800. The flag
can remain true on the fatal decision. Conditional H5000 survival after ever
holding it for 100 consecutive eligible decisions is **54/60 (90.00%)** for
dense versus **38/56 (67.86%)** for B8. This is a retrospective association,
not a guaranteed survival interval after a certificate is observed.

**Horizon verdict: a strong horizon effect with input-dependent decay, not a
universal monotonic ceiling near 0.77.** D30 to D300 AUC falls from **0.9121
to 0.7261** for 60 scalars, **0.9140 to 0.7349** for 60+24, and **0.9331 to
0.7491** for arrival inputs. The five fuse+Q designs move from **0.8417–0.8572
to 0.7630–0.7693**, with different trajectories: arrival leads every fuse at
D100, while every fuse leads arrival at D300. All eight designs recover from
D200 to D300, so the decay is not monotonic. Arrival's paired D300 advantage
over 60+24 remains **+0.0142 AUC [95% CI +0.0073, +0.0209]**. This bank
supports input-dependent discrimination as the target horizon changes;
it does not establish a ceiling for other models or continuation policies.
<!-- verdicts:end -->

## Certificate definition and replay

The spec does not assign a numeric deadline to “before it would be needed.”
The disclosed operational convention is arrival at the **actual current physical
own tail within L unit-speed cell steps**, with **128 steps** as a sensitivity check.
This convention was raised with the user before the full run; no alternative
deadline was supplied at launch. The flood reuses T9’s TTL-release relaxation
`max(previous + 1, destination_TTL)` on four-neighbor cells. All owners release
their existing body at its current TTL. Sources seed time zero. Race-128 counts
hero arrivals within 128 steps strictly before all live opponents; ties lose.

Waiting, reverse moves and routes through future trail are optimistic: growth,
new trails, respawns, heading restrictions and movement feasibility during waiting
are absent. **A positive flag is an access proxy, not a guaranteed safe Snake path.**
The tail is taken from physical `seg_count`, even when logical L exceeds it.

All 64 episode identities (B8_s0 and B8_s3, 32 worlds each) are replayed under
both B8 and the recorded dense lookahead. The two arms share the same 32 world
seeds. B8 uses recorded census actions; dense uses its recorded trigger choices
and original B8 greedy continuation otherwise. No search is rerun. Every B8
predecision length, dense trigger length/greedy action, intervention count,
world/checkpoint identity, survival, alive-frame fraction and mass integral
matches its source receipt; dense death decisions and lengths also match.

Measurements include every prepared alive decision at L >= 800. The strict area
thresholds are race-128 > L/2 and race-128 > L, not >=.
Fractions pool decisions, not per-world means; the policies induce different
state exposures and trajectory lengths. These are descriptive policy comparisons.

| Policy | H5000 survivors | L >= 800 decisions | Tail by L | Tail by 128 | Race > L/2 | Race > L |
|---|---:|---:|---:|---:|---:|---:|
| b8 | 38/64 | 121,261 | 121,158 (99.92%) | 71,346 (58.84%) | 104,050 (85.81%) | 81,168 (66.94%) |
| dense | 54/64 | 144,958 | 144,914 (99.97%) | 85,563 (59.03%) | 124,059 (85.58%) | 96,452 (66.54%) |

Deadline-equality sensitivity from the same saved arrivals:

- b8: 121,158/121,261 arrive strictly before L; 0 arrive exactly at L.
- dense: 144,914/144,958 arrive strictly before L; 0 arrive exactly at L.

## Last certificate and loss duration before every death

For fatal decision D and last eligible certified decision d, duration is D−d
recorded decision transitions. Zero means the flag remains true on the fatal
decision. These durations are time since the last **observed eligible** true
state, not proven continuously false time across a later length dip below 800.
“Never” means no true flag in eligible states; “no L800” means no eligible
state at all. Both are censored/unobserved values, not zero.

| Policy | Arm | Env | Death decision | Death L | Last tail-by-L | Duration | Last tail-by-128 | Duration |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| b8 | b8_s0 | 3 | 4362 | 1284 | 4362 | 0 | 4362 | 0 |
| b8 | b8_s0 | 5 | 2041 | 724 | no L800 | — | no L800 | — |
| b8 | b8_s0 | 8 | 4553 | 1010 | 4553 | 0 | never | — |
| b8 | b8_s0 | 11 | 4198 | 1316 | 4160 | 38 | 4140 | 58 |
| b8 | b8_s0 | 12 | 4877 | 1443 | 4876 | 1 | 4876 | 1 |
| b8 | b8_s0 | 14 | 1763 | 596 | no L800 | — | no L800 | — |
| b8 | b8_s0 | 15 | 4177 | 1106 | 4165 | 12 | 3947 | 230 |
| b8 | b8_s0 | 17 | 4610 | 1304 | 4610 | 0 | 4483 | 127 |
| b8 | b8_s0 | 21 | 3080 | 974 | 3065 | 15 | 3065 | 15 |
| b8 | b8_s0 | 22 | 3694 | 1116 | 3694 | 0 | 3692 | 2 |
| b8 | b8_s0 | 30 | 1523 | 340 | no L800 | — | no L800 | — |
| b8 | b8_s3 | 1 | 2889 | 886 | 2889 | 0 | 2874 | 15 |
| b8 | b8_s3 | 6 | 4050 | 1178 | 4050 | 0 | 3071 | 979 |
| b8 | b8_s3 | 8 | 4407 | 1196 | 4386 | 21 | 4050 | 357 |
| b8 | b8_s3 | 9 | 4416 | 1020 | 4416 | 0 | 4319 | 97 |
| b8 | b8_s3 | 12 | 3492 | 972 | 3492 | 0 | 3490 | 2 |
| b8 | b8_s3 | 13 | 2208 | 624 | no L800 | — | no L800 | — |
| b8 | b8_s3 | 15 | 2636 | 907 | 2628 | 8 | 2403 | 233 |
| b8 | b8_s3 | 16 | 255 | 53 | no L800 | — | no L800 | — |
| b8 | b8_s3 | 17 | 4531 | 1089 | 4531 | 0 | 4282 | 249 |
| b8 | b8_s3 | 18 | 1623 | 559 | no L800 | — | no L800 | — |
| b8 | b8_s3 | 19 | 4939 | 1115 | 4933 | 6 | 3853 | 1086 |
| b8 | b8_s3 | 21 | 1983 | 601 | no L800 | — | no L800 | — |
| b8 | b8_s3 | 26 | 3282 | 1059 | 3281 | 1 | 2828 | 454 |
| b8 | b8_s3 | 29 | 2151 | 664 | no L800 | — | no L800 | — |
| b8 | b8_s3 | 30 | 3846 | 1035 | 3846 | 0 | 3846 | 0 |
| dense | b8_s0 | 8 | 4601 | 1014 | 4599 | 2 | never | — |
| dense | b8_s0 | 14 | 1763 | 596 | no L800 | — | no L800 | — |
| dense | b8_s0 | 21 | 4043 | 1086 | 4010 | 33 | 3535 | 508 |
| dense | b8_s0 | 30 | 1523 | 340 | no L800 | — | no L800 | — |
| dense | b8_s3 | 6 | 4776 | 1218 | 4776 | 0 | 3071 | 1705 |
| dense | b8_s3 | 8 | 4533 | 1200 | 4533 | 0 | 4050 | 483 |
| dense | b8_s3 | 9 | 4510 | 1020 | 4504 | 6 | 4326 | 184 |
| dense | b8_s3 | 16 | 255 | 53 | no L800 | — | no L800 | — |
| dense | b8_s3 | 18 | 1623 | 559 | no L800 | — | no L800 | — |
| dense | b8_s3 | 26 | 3952 | 1067 | 3952 | 0 | 3297 | 655 |

| Policy | Certificate | Observed deaths | Unobserved deaths | Min / Q1 / median / Q3 / max duration |
|---|---|---:|---:|---|
| b8 | tail_L | 18 | 8 | 0 / 0 / 0 / 7.5 / 38 |
| b8 | tail_128 | 17 | 9 | 0 / 2 / 97 / 249 / 1086 |
| dense | tail_L | 6 | 4 | 0 / 0 / 1 / 5 / 33 |
| dense | tail_128 | 5 | 5 | 184 / 483 / 508 / 655 / 1705 |

## Survival conditional on consecutive certified decisions

Each cell is H5000 survivors / episodes that **ever** held the flag at
L >= 800 for k consecutive decisions. Each episode enters once. False flags
and ineligible gaps reset the streak. This is a retrospective association
conditioned on surviving long enough to enter; it is not causal or a safety
guarantee. Decision-landmark counts are retained in the evidence JSON.

| k | B8 tail by L | Dense tail by L | B8 tail by 128 | Dense tail by 128 |
|---:|---:|---:|---:|---:|
| 1 | 38/56 (67.86%) | 54/60 (90.00%) | 38/55 (69.09%) | 54/59 (91.53%) |
| 5 | 38/56 (67.86%) | 54/60 (90.00%) | 38/55 (69.09%) | 54/59 (91.53%) |
| 10 | 38/56 (67.86%) | 54/60 (90.00%) | 38/55 (69.09%) | 54/59 (91.53%) |
| 25 | 38/56 (67.86%) | 54/60 (90.00%) | 38/55 (69.09%) | 54/59 (91.53%) |
| 50 | 38/56 (67.86%) | 54/60 (90.00%) | 38/55 (69.09%) | 54/59 (91.53%) |
| 100 | 38/56 (67.86%) | 54/60 (90.00%) | 38/54 (70.37%) | 54/59 (91.53%) |
| 200 | 38/56 (67.86%) | 54/60 (90.00%) | 38/51 (74.51%) | 54/59 (91.53%) |
| 300 | 38/55 (69.09%) | 54/60 (90.00%) | 37/50 (74.00%) | 53/58 (91.38%) |

## Horizon-stratified AUC

The immutable T5 bank contributes 19,976 states, 103,264 legal branches and
342 worlds. Death on or before H transitions is positive, including exactly
H; survival through D300 is negative at every requested H. Illegal branches
are excluded. All horizons and inputs reuse T5’s saved five world-grouped
folds, fixed GBM (200 iterations, 31 leaves, learning rate 0.1, no early
stopping) and 2,000 whole-world bootstrap draws, seed 20261017. Intervals are
95% percentile intervals of fixed out-of-fold predictions, not model-refit
uncertainty. Six action one-hot columns are appended to every input design.
Changing the horizon changes prevalence and the composition of positive outcomes;
AUC is not required to decline monotonically across these different targets.

T8 arrival features are available and included: 60+24+18.
T5b contributes 256 fuse activations plus six Q values for 5 B8 finals;
no best final is selected. No raster CNN is refit, so conclusions below
apply only to these fixed GBM input sets and this continuation-policy bank.

| Death within | Positive branches | 60 scalars AUC [95% CI] | 60+24 AUC [95% CI] | 60+24+18 arrival AUC [95% CI] |
|---:|---:|---|---|---|
| 30 | 1,655 | 0.9121 [0.8974, 0.9258] | 0.9140 [0.8998, 0.9276] | 0.9331 [0.9208, 0.9444] |
| 60 | 3,437 | 0.8177 [0.7981, 0.8362] | 0.8234 [0.8058, 0.8401] | 0.8533 [0.8356, 0.8698] |
| 100 | 5,931 | 0.7378 [0.7176, 0.7581] | 0.7409 [0.7209, 0.7619] | 0.7773 [0.7576, 0.7967] |
| 200 | 12,021 | 0.7009 [0.6816, 0.7210] | 0.7101 [0.6921, 0.7293] | 0.7295 [0.7115, 0.7474] |
| 300 | 19,094 | 0.7261 [0.7079, 0.7459] | 0.7349 [0.7161, 0.7552] | 0.7491 [0.7317, 0.7676] |

| Fuse+Q source | D30 AUC [95% CI] | D60 | D100 | D200 | D300 |
|---|---|---|---|---|---|
| B8_s0 | 0.8417 [0.8222, 0.8603] | 0.7675 [0.7504, 0.7841] | 0.7228 [0.7061, 0.7395] | 0.7280 [0.7110, 0.7454] | 0.7691 [0.7511, 0.7876] |
| B8_s1 | 0.8559 [0.8371, 0.8738] | 0.7799 [0.7622, 0.7963] | 0.7294 [0.7127, 0.7466] | 0.7302 [0.7135, 0.7466] | 0.7693 [0.7507, 0.7882] |
| B8_s2 | 0.8513 [0.8334, 0.8680] | 0.7658 [0.7469, 0.7837] | 0.7222 [0.7043, 0.7400] | 0.7286 [0.7114, 0.7462] | 0.7665 [0.7484, 0.7855] |
| B8_s3 | 0.8464 [0.8269, 0.8648] | 0.7697 [0.7507, 0.7881] | 0.7279 [0.7105, 0.7452] | 0.7261 [0.7097, 0.7432] | 0.7653 [0.7471, 0.7837] |
| B8_s4 | 0.8572 [0.8374, 0.8759] | 0.7840 [0.7662, 0.8006] | 0.7284 [0.7099, 0.7471] | 0.7258 [0.7086, 0.7429] | 0.7630 [0.7446, 0.7812] |

| Horizon | Arrival − 60 AUC [95% CI] | Arrival − 60+24 AUC [95% CI] |
|---:|---|---|
| 30 | +0.0209 [+0.0125, +0.0293] | +0.0191 [+0.0130, +0.0250] |
| 60 | +0.0356 [+0.0257, +0.0458] | +0.0299 [+0.0212, +0.0384] |
| 100 | +0.0395 [+0.0281, +0.0509] | +0.0364 [+0.0260, +0.0469] |
| 200 | +0.0287 [+0.0182, +0.0387] | +0.0194 [+0.0102, +0.0287] |
| 300 | +0.0230 [+0.0155, +0.0300] | +0.0142 [+0.0073, +0.0209] |

The paired arrival differences use the same whole-world draws. Similar
decay is descriptive evidence, not proof of a universal horizon ceiling,
chaos, observation sufficiency or impossibility for other policies/models.

## Verification and provenance

- Certificate replay: 1029.51 seconds, 128/128 episodes.
- Horizon refits: 200 retained fold fits, 8 input sets.
- Late D200/D300 shard elapsed: 1994.08 seconds (not total parallel study wall time).
- All workers used CPU only, one numerical thread each, at most two concurrent
  numerical processes. No tmux session or process not started by this task
  was controlled.
- Saved-artifact audit PASS: complete cohorts, source/output hashes, every label
  and fold identity, all AUCs and all bootstrap intervals recomputed from saved
  predictions using the established T5 routines. This is saved-artifact verification,
  not a second statistical implementation. D300 T5/T8/T5b predictions match exactly.
- Execution source: implementation `86458528`, merged/pushed launch HEAD `c3558167`.
- After certificate completion, D200/D300 ran in the free second slot under
  shard driver `a739094e`, using the unchanged fit implementation. The original
  worker was intentionally stopped only after saving D30/D60/D100; its raw
  receipt retains that partial status. The combined five-horizon receipt passes
  source/settings/branch/copy-hash checks and the complete saved-artifact audit.
- Focused tests and flake8 results are recorded in `verification.json` in the
  evidence directory. Read-only independent review found no blocking issue.

Raw artifacts are in
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t10-certificate-horizon-20261017/`.
Committed numeric receipts: [certificate](t10_certificate_horizon_2026-10-17_evidence/certificate.json),
[horizon fits](t10_certificate_horizon_2026-10-17_evidence/horizon.json),
[saved-artifact audit](t10_certificate_horizon_2026-10-17_evidence/audit.json).
Source/input/output SHA256s are embedded in those receipts. Reproduction commands
and complete measurement definitions are in
[the T10 README](../../research/t10_certificate_horizon_20261017/README.md).

Certificate episode trace SHA256: `1c75628ac540f666de497dbea7e604c0bb72d544d61fe7a776d57f17a807cca1`.

Canonical T5 bank SHA256: `e4ecbe991061c90bb2ae3cdb4b236d0025f1219d61fa806b7e96f43962d9e02a`.
