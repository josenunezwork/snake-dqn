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

The defaults reproduce this README. For a study's own pre-registration check, the flags
`--data`, `--n-max`, `--mde`, `--delta-ni`, `--thetas` and `--check-rule M,alpha,bound` point
the same parts at the study's pool and frozen plan. `--check-rule` adds the scenario
`one_mix_at_margin` and, in `gate`, a `check` field with the amendment's acceptance test:
regressed-band rate at most 1.2 x alpha. A 3,000-replicate smoke run of the recommended rule
on these pools gave a maximum of 0.052 against the threshold 0.06; that run is not a result.

The script runs on one thread with fixed seeds, using numpy and the repo's stdlib t
functions. `extract` reads the saved records once and writes the committed compact data file
`paired_survival_20261003.json` (sha256 `7fc6101c…a016`). It holds, per mix and world seed,
both arms' survival_fraction and the mass_integral delta, plus the 16 v7 calibration incumbent
survival values per mix. Every other part reads only that file. Measured CPU: `bands` 5.8 s,
`gate` 13.6 s, `crosscheck` 10.1 s. The outputs were written to the session scratchpad and are
not in the repo: `bands.json` sha256 `6ed332cc…a54a`, `gate.json` `0b6da288…791d`,
`crosscheck.json` `b558fa3b…d90e`. `bands` and `gate` were run before a black/isort
reformat of the script and before `crosscheck` was switched to the recommended rule. Neither
change touches those two parts.

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
  -0.02 or -0.05 in every mix, or -0.05 in the frozen mix only. The candidate's per-world
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
  and shifted to theta = 30, 60 or 130. The band is judged only at the qualifying look.
  Mass and survival shifts are set independently, so a candidate can gain mass while
  losing survival. That is the adversarial case for selection. The per-world noise keeps
  its real correlation of about 0.95-0.99.
- **Cross-check.** On 300 fresh replicates (theta 30 with a one-mix -0.05 regression, theta
  60, theta 30), the vectorized gate plus band was compared with the pure
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

## Results: band judged at the qualifying look of the full gate (20,000 replicates)

The figures are P(bands pass | the run qualified). The last column is P(the regressed
frozen band passes | qualified) in the one-mix -0.05 scenario, with a 95% Wilson interval
for the recommended rule. The v8-screen pool is in parentheses.

| theta (mass) | P(qualify) | Typical stop look | Rule | No regression | Improves +0.05 | One-mix -0.05: regressed band |
|---|---|---|---|---|---|---|
| 130 | 1.000 (1.000) | look 1 (n=63) | Old, as run | 0.084 (0.104) | 0.349 (0.410) | 0.211 (0.224) |
| | | | **Paired M 0.05 a 0.05** | 0.145 (0.212) | 0.925 (0.944) | **0.045 [0.043, 0.048]** (0.040) |
| | | | Paired M 0.05 a 0.10 | 0.311 (0.394) | 0.972 (0.981) | 0.096 (0.088) |
| 60 | 1.000 (1.000) | look 2 (n=125) | Old, as run | 0.157 (0.163) | 0.503 (0.537) | 0.267 (0.267) |
| | | | **Paired M 0.05 a 0.05** | 0.509 (0.565) | 1.000 (0.998) | **0.047 [0.044, 0.050]** (0.048) |
| | | | Paired M 0.05 a 0.10 | 0.699 (0.739) | 1.000 (0.999) | 0.099 (0.097) |
| 30 | 0.957 (0.988) | looks 3-4 | Old, as run | 0.214 (0.197) | 0.597 (0.600) | 0.300 (0.294) |
| | | | **Paired M 0.05 a 0.05** | 0.893 (0.849) | 1.000 (1.000) | **0.048 [0.045, 0.051]** (0.046) |
| | | | Paired M 0.05 a 0.10 | 0.956 (0.937) | 1.000 (1.000) | 0.099 (0.102) |

**Selection.** Judging the pointwise band at the data-dependent qualifying look, under the
real ~0.98 per-world mass/survival correlation, gave a regressed-band pass rate of 0.040 to
0.048. The upper Wilson bound is at most 0.051, against nominal 0.05. No inflation was
detected. The RCI-OBF bound keeps its guarantee by construction, but it passes 0.000-0.001
of no-regression candidates that stop at look 1.

## Verdict

- **Old rule.** It fails both jobs at every n. It passes a candidate with no regression in all
  three bands only 9-23% of the time. It passes a band whose true regression is 0.05 21-33%
  of the time and 0.02 36-50% of the time. Calibration noise dominates it. Adding worlds
  does not help, because the 16-world reference never improves.
- **Paired, M 0.02.** Not feasible at this survival SD (about 0.23 per world). A candidate
  with no regression passes all three bands at most 6-10% of the time, even at n = 249.
- **Paired, M 0.05, alpha 0.05, pointwise, at the qualifying look (recommended).** It caps a
  0.05 regression at the nominal 5% per band, at every n and under the gate's selection. It
  passes more no-regression and improving candidates than the old rule at every n. Its weak
  spot is look 1: a candidate with no survival change passes all three bands only 15-21% of
  the time at n = 63. That case is unrealistic for a look-1 stop, because a mass gain large
  enough to stop at look 1 comes with a survival gain (correlation 0.98). A candidate that
  improves survival by 0.05 passes 93-94% of the time at n = 63.
- **Alternative.** alpha 0.10 roughly doubles look-1 passing for an unchanged candidate
  (32-40%), at a 10% cap on a 0.05 regression. Each study pre-registers its choice.

## Limits

- Two pools, both from veto variants of one champion network, and only 60 worlds in the v8
  screen pool. The shift model borrows the observed per-world delta shape for hypothetical
  regressing candidates.
- Shifting survival and mass independently is adversarial for selection but not for the
  per-world joint shape. A study whose band metric differs, or whose screen shows a
  different mass/survival relation, re-runs `gate` on its own data (see the amendment).
- The vectorized replica approximates z by t in the conditional-power futility rule. The
  cross-check found 0 mismatches in 300 replicates.
