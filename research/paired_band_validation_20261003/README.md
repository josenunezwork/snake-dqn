# Paired survival bands: resampling validation, 2026-10-03

Label: **diagnostic (Tier 0 style)**. It resamples saved paired records from two completed runs:
the v7 strict final stage and the v8 Tier-1 screen. No game was run. Nothing was read from the
running v8 strict gate (`apex-veto-v8-strict-20261003`): not its records, not its look
receipts. Nothing here is evidence about any candidate. It supports
[`docs/research/governance_amendment_paired_bands_2026-10-03.md`](../../docs/research/governance_amendment_paired_bands_2026-10-03.md).

Question (written before any run): the current survival band requires candidate mean
survival_fraction >= incumbent calibration mean - 0.02, with a 16-world reference. How often
does it pass a candidate with no survival regression, and how often a candidate with a true
regression of 0.02 or 0.05? Same two questions for a paired noninferiority band at n = 63,
125, 187 and 249 worlds per mix (the looks of N_max 249). And does the paired band keep its
level when it is judged at the data-dependent qualifying look of the sequential gate?

## Reproduce

```
OMP_NUM_THREADS=1 ./venv/bin/python research/paired_band_validation_20261003/simulate.py --part extract
OMP_NUM_THREADS=1 ./venv/bin/python research/paired_band_validation_20261003/simulate.py --part bands --out <file outside repo>
OMP_NUM_THREADS=1 ./venv/bin/python research/paired_band_validation_20261003/simulate.py --part gate --out <file outside repo>
OMP_NUM_THREADS=1 ./venv/bin/python research/paired_band_validation_20261003/simulate.py --part crosscheck --out <file outside repo>
```

The defaults reproduce this README.

**A study's own pre-registration check.** The same parts can be pointed at a study's pool and
frozen plan:

- `--data` gives the pool, in the format of the data file below.
- `--plan-params` gives an `intent.json`-style file `{"plan_parameters": ..., "plan": ...}`.
  The plan is rebuilt with `sequential_gate_plan` and must equal `plan`, or the script
  exits; the plan's looks, alphas, futility rule and band rule are then used. Under
  `futility_policy = "overridable"`, futility stops are ignored (the conservative bound). Without it,
  use `--n-max`, `--mde` and `--check-rule M,alpha,bound`.
- `--delta-ni` and `--thetas` set the remaining inputs.

The check rule adds the scenarios `one_mix_at_margin@<mix>`, which regress each mix in turn by
M. In `gate` it also adds a `check` field with the amendment's acceptance test: joint
regressed-band error at most 1.2 x alpha, with the conditional rate reported only. The
script simulates exactly the three mixes `frozen`, `scripted` and `mixed`, and it exits on
any other mix set.

**How the runs were made.** The script runs on one thread with fixed seeds, using numpy and
the repo's stdlib t functions.

- **Data.** `extract` reads the saved records once and writes the committed compact data
  file `paired_survival_20261003.json` (sha256 `7fc6101c…a016`). It holds, per mix and world
  seed, both arms' survival_fraction and the mass_integral delta, plus the 16 v7 calibration
  incumbent survival values per mix. Every other part reads only that file.
- **CPU.** `bands` 5.8 s (20,000 replicates), `gate` 20.0 s (10,000), `crosscheck` 11.8 s.
- **Outputs.** They were written to the session scratchpad and are not in the repo:
  `bands.json` (`6ed332cc…a54a`), `gate2.json` (`fe976c0e…6bd3`), `crosscheck2.json`
  (`dc0c0e28…d4ff`). These sha256 values identify a run, not a reproducible result: each
  output embeds `cpu_seconds` and the data path. A re-run reproduces every number in them,
  but not the hash.
- **The `bands` run.** It was made before two scenarios (`one_mix_-0.05@scripted`,
  `@mixed`) were appended for `gate`. Appending does not change the seeds of earlier
  scenarios, so `bands.json` simply lacks those two.

## Data

| Pool | Pairs (worlds x 3 mixes, seeds shared across mixes) | Incumbent -> candidate | Paired survival-delta SD per mix | corr(mass delta, survival delta) |
|---|---|---|---|---|
| `v7_strict` | 243 x 3 | v5 veto -> v7 veto (lambda 4) | 0.227, 0.233, 0.223 | 0.981, 0.983, 0.986 |
| `v8_screen` | 60 x 3 (arms A, B; repeat arms C, D excluded) | v7 -> v8 (lambda 8) | 0.241, 0.244, 0.150 | 0.953, 0.959, 0.962 |

Incumbent survival SD per world is 0.24-0.29. The v7 calibration's 16-world means had SDs of
0.19-0.30, so a 16-world reference mean has an SD of about 0.06, three times the 0.02 band.

## Method

- **Scenario construction.** Per mix, the paired survival deltas are centred at 0 and then
  shifted to the scenario's true delta. A scenario is no regression, +0.05 in every mix,
  -0.02 or -0.05 in every mix, or -0.02 or -0.05 in one mix only. In `gate`, the one-mix
  -0.05 regression is placed in each of the three mixes in turn. The candidate's per-world
  value is the incumbent's value plus the shifted delta. Worlds are resampled with
  replacement, one index per world shared by all three mixes, because the strict mixes
  share world seeds. Looks are nested prefixes of the same draw.
- **Old rule.** The reference is the mean incumbent survival over 16 independently resampled
  worlds, mimicking the calibration stage. "As run" is `band_check` under
  `block_at_stop`: it requires mean >= reference - 0.02 + 1.645 x sd x (1/sqrt(n) -
  1/sqrt(249)), and the margin is 0 at n = 249. "Point rule" uses margin 0 at every n,
  which is the fixed-N rule at that n.
- **Paired rule.** Pass iff mean(d) - t_{n-1}(p_k) x sd(d)/sqrt(n) > -M. The grid is
  M in {0.02, 0.05, 0.075} and alpha in {0.05, 0.10}. The bound is either `pointwise`
  (p_k = alpha) or `rci_obf` (p_k from a Lan-DeMets OBF spending of alpha over the four
  looks).
- **Gate part.** Joint resampling of (mass delta, survival delta) per world through a
  vectorized replica of the whole sequential gate: `sequential_gate_plan(249, mde=30)`,
  delta_NI 5.788 (the v7 calibration value), and futility honored. Mass deltas are centred
  and shifted to theta = 15, 20, 30, 45, 60 or 130, which is 0.5, 0.67, 1, 1.5 and 2 x MDE
  plus the v7 strict look-1-sized effect. The band is judged only at the qualifying look.
  Mass and survival shifts are set independently, so a candidate can gain mass while
  losing survival. That is the adversarial case for selection. The per-world noise keeps
  its real correlation of about 0.95-0.99.
- **Cross-check.** The cross-check used 300 fresh replicates, 75 in each of four cases:
  theta 30 with a frozen -0.05 regression, theta 60, theta 30, and theta 15 with a
  scripted -0.05 regression. Three of the cases stop mostly at looks 2 to 4, and the
  theta-15 case often does not qualify. No case is dominated by look-1 stops. On these,
  the vectorized gate plus band was compared with the pure
  `run_sequential_gate` fed per-look verdicts from `plan_paired_band_check` (recommended
  rule). Result: **0 mismatches** in stop look and decision. The old rule's replica was
  checked against the module's `band_check` on 200 random prefixes: **0 mismatches**.

## Results: band only, fixed n (20,000 replicates per cell; v7 pool, v8-screen pool in parentheses)

**(a) No regression: all three bands pass**

| Rule | n=63 | 125 | 187 | 249 |
|---|---|---|---|---|
| Old, as run (block_at_stop) | 0.092 (0.111) | 0.158 (0.170) | 0.203 (0.202) | 0.231 (0.228) |
| Old, point rule (margin 0) | 0.221 (0.223) | 0.227 (0.228) | 0.230 (0.225) | 0.231 (0.228) |
| Paired, M 0.02, a 0.05 | 0.004 (0.007) | 0.012 (0.021) | 0.031 (0.054) | 0.059 (0.097) |
| **Paired, M 0.05, a 0.05 (recommended)** | 0.150 (0.213) | 0.505 (0.583) | 0.783 (0.817) | 0.914 (0.928) |
| Paired, M 0.05, a 0.10 | 0.320 (0.399) | 0.699 (0.752) | 0.893 (0.913) | 0.963 (0.970) |
| Paired, M 0.05, a 0.10, RCI-OBF | 0.000 (0.000) | 0.279 (0.361) | 0.788 (0.822) | 0.953 (0.960) |
| Paired, M 0.075, a 0.05 | 0.606 (0.672) | 0.955 (0.962) | 0.997 (0.997) | 1.000 (1.000) |

**Candidate improves survival by 0.05 in every mix: all three pass**

| Rule | n=63 | 125 | 187 | 249 |
|---|---|---|---|---|
| Old, as run (block_at_stop) | 0.354 (0.409) | 0.507 (0.544) | 0.578 (0.605) | 0.617 (0.641) |
| **Paired, M 0.05, a 0.05 (recommended)** | 0.927 (0.945) | 0.999 (0.999) | 1.000 (1.000) | 1.000 (1.000) |
| Paired, M 0.05, a 0.10, RCI-OBF | 0.233 (0.316) | 0.997 (0.997) | 1.000 (1.000) | 1.000 (1.000) |

**(b) True regression 0.02: P(the regressed band passes)**

| Rule | n=63 | 125 | 187 | 249 |
|---|---|---|---|---|
| Old, as run (block_at_stop) | 0.355 (0.370) | 0.428 (0.433) | 0.472 (0.467) | 0.495 (0.484) |
| Old, point rule (margin 0) | 0.497 (0.486) | 0.494 (0.484) | 0.496 (0.486) | 0.495 (0.484) |
| Paired, M 0.02, a 0.05 | 0.042 (0.037) | 0.044 (0.043) | 0.046 (0.044) | 0.047 (0.043) |
| **Paired, M 0.05, a 0.05 (recommended)** | 0.255 (0.232) | 0.422 (0.389) | 0.564 (0.520) | 0.673 (0.634) |
| Paired, M 0.05, a 0.10 | 0.397 (0.368) | 0.575 (0.545) | 0.703 (0.670) | 0.793 (0.767) |

**(b) True regression 0.05: P(the regressed band passes)**

| Rule | n=63 | 125 | 187 | 249 |
|---|---|---|---|---|
| Old, as run (block_at_stop) | 0.215 (0.223) | 0.273 (0.268) | 0.298 (0.289) | 0.318 (0.305) |
| Old, point rule (margin 0) | 0.338 (0.317) | 0.325 (0.309) | 0.321 (0.306) | 0.318 (0.305) |
| **Paired, M 0.05, a 0.05 (recommended)** | 0.045 (0.036) | 0.046 (0.040) | 0.048 (0.042) | 0.049 (0.042) |
| Paired, M 0.05, a 0.10 | 0.090 (0.087) | 0.096 (0.089) | 0.100 (0.092) | 0.097 (0.091) |
| Paired, M 0.05, a 0.10, RCI-OBF | 0.001 (0.000) | 0.017 (0.013) | 0.051 (0.043) | 0.079 (0.073) |
| Paired, M 0.075, a 0.05 | 0.204 (0.184) | 0.331 (0.301) | 0.441 (0.404) | 0.539 (0.499) |

`bands.json` has every rule at every cell, per mix and for all three mixes together. It also
has the one-mix -0.05 scenario. Under the recommended rule, all three bands pass that
scenario 0.014-0.043 of the time; under the old rule as run, 0.039-0.116.

## Results: band judged at the qualifying look of the full gate (10,000 replicates)

**Two error rates.** For a one-mix regression of 0.05, the error that matters for promotion is
the **joint** rate, P(run qualifies and the regressed band passes). It bounds P(gate PASS
with that regression). The **conditional** rate, given that the run qualified, is reported
too. The v8-screen pool is in parentheses throughout.

**Pass rates when survival does not regress**

| theta (mass) | P(qualify) | Main stop looks | Rule | No regression: P(bands pass \| qualified) | Improves +0.05: same |
|---|---|---|---|---|---|
| 15 (0.5 MDE) | 0.21 (0.41) | look 4 | Old, as run | 0.259 (0.238) | 0.656 (0.644) |
| | | | **Paired M 0.05 a 0.05** | 0.971 (0.968) | 1.000 (1.000) |
| 20 (0.67 MDE) | 0.51 (0.74) | looks 3-4 | Old, as run | 0.243 (0.221) | 0.651 (0.624) |
| | | | **Paired M 0.05 a 0.05** | 0.965 (0.952) | 1.000 (1.000) |
| 30 (MDE) | 0.96 (0.99) | looks 3-4 | Old, as run | 0.210 (0.195) | 0.593 (0.593) |
| | | | **Paired M 0.05 a 0.05** | 0.890 (0.840) | 1.000 (1.000) |
| 45 (1.5 MDE) | 1.00 (1.00) | looks 2-3 (mixed) | Old, as run | 0.180 (0.175) | 0.532 (0.550) |
| | | | **Paired M 0.05 a 0.05** | 0.645 (0.633) | 1.000 (1.000) |
| 60 (2 MDE) | 1.00 (1.00) | look 2 | Old, as run | 0.163 (0.162) | 0.506 (0.531) |
| | | | **Paired M 0.05 a 0.05** | 0.504 (0.568) | 1.000 (0.999) |
| 130 | 1.00 (1.00) | look 1 | Old, as run | 0.087 (0.107) | 0.353 (0.415) |
| | | | **Paired M 0.05 a 0.05** | 0.148 (0.208) | 0.929 (0.946) |
| | | | Paired M 0.05 a 0.10 | 0.316 (0.392) | 0.972 (0.978) |

**One mix regresses by 0.05: the range over the three regressed mixes**

| theta | Old as run: joint | **Paired M 0.05 a 0.05: joint** | Paired M 0.05 a 0.05: conditional | Paired M 0.05 a 0.10: joint |
|---|---|---|---|---|
| 15 | 0.066-0.078 (0.109-0.151) | **0.020-0.024 (0.032-0.042)** | 0.094-0.121 (0.077-0.102) | 0.041-0.053 (0.063-0.090) |
| 20 | 0.163-0.199 (0.202-0.248) | **0.031-0.036 (0.041-0.043)** | 0.060-0.070 (0.056-0.058) | 0.065-0.078 (0.086-0.095) |
| 30 | 0.283-0.322 (0.251-0.312) | **0.045-0.048 (0.044-0.048)** | 0.047-0.050 (0.044-0.049) | 0.096-0.101 (0.101-0.102) |
| 45 | 0.275-0.302 (0.242-0.302) | **0.043-0.047 (0.035-0.053)** | same as joint | 0.089-0.095 (0.080-0.102) |
| 60 | 0.265-0.287 (0.233-0.299) | **0.044-0.046 (0.037-0.049)** | same as joint | 0.094-0.099 (0.086-0.100) |
| 130 | 0.206-0.216 (0.193-0.247) | **0.039-0.045 (0.030-0.053)** | same as joint | 0.088-0.094 (0.074-0.107) |

**Selection.**
- **Joint error.** The recommended rule's joint error stayed within 0.020-0.053 in every cell
  (36 cells: 2 pools x 6 thetas x 3 regressed mixes). The largest, 0.053 (Wilson
  0.049-0.058), is in the 60-world v8 pool's `mixed` mix, whose paired SD is only 0.150.
  The v7 pool's maximum was 0.048.
- **Conditional rate.** It is inflated below the MDE, where qualifying is rare and selects
  runs with high mass noise and therefore high survival noise: 0.09-0.12 at theta 15 and
  0.06-0.07 at theta 20. In those runs the gate rarely passes at all, so the joint error
  stays below 0.05.
- **Above the MDE**, including theta 45 where the stop look is genuinely mixed, no
  conditional inflation was detected.
- **The old rule's joint error** is 0.07-0.32.
- **RCI-OBF** keeps its guarantee by construction. At alpha 0.10 it passes 0.000-0.001 of
  no-regression candidates that stop at look 1 (`bands` part).

## Verdict

- **Old rule.** It fails both jobs at every n. It passes a candidate with no regression in all
  three bands only 9-23% of the time. It passes a band whose true regression is 0.05 21-33%
  of the time, and one whose true regression is 0.02 36-50% of the time. In the full gate,
  its joint error for a one-mix 0.05 regression reaches 0.32. Calibration noise dominates
  it, and adding worlds does not help because the 16-world reference never improves.
- **Paired, M 0.02.** Not feasible at this survival SD (about 0.23 per world). A candidate
  with no regression passes all three bands at most 6-10% of the time, even at n = 249.
- **Paired, M 0.05, alpha 0.05, pointwise, at the qualifying look (recommended).**
  - It caps a 0.05 regression near the nominal 5%: per band at every fixed n, and as a joint
    gate error at every theta tested (maximum 0.053).
  - It passes more no-regression and improving candidates than the old rule at every n.
  - Its weak spot is look 1: a candidate with no survival change passes all three bands
    only 15-21% of the time at n = 63. The one saved run with a look-1-sized mass gain also
    gained 0.18-0.21 survival, with a per-world correlation of 0.98, so this case is the
    less likely one. A candidate that improves survival by 0.05 passes 93-95% of the time at
    n = 63.
- **Alternative.** alpha 0.10 roughly doubles look-1 passing for an unchanged candidate
  (32-40%), at about a 10% joint cap on a 0.05 regression. Each study pre-registers its
  choice.

## Limits

- Two pools, both from veto variants of one champion network, and only 60 worlds in the v8
  screen pool. The shift model borrows the observed per-world delta shape for hypothetical
  regressing candidates.
- Shifting survival and mass independently is adversarial for selection but not for the
  per-world joint shape. A study whose band metric differs, or whose screen shows a
  different mass/survival relation, re-runs `gate` on its own data (see the amendment).
- The vectorized replica approximates z by t in the conditional-power futility rule. The
  cross-check found 0 mismatches in 300 replicates; it had no case dominated by look-1
  stops.
- The `gate` part does not simulate `band_floor`, which is a tripwire set far below any
  plausible candidate mean.
