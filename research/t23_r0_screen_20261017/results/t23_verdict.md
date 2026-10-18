# T23 R0 analysis

**Verdict: KILL**

Only 8M decides. GO permits long-run planning; it is not a strict pass or release.

G = -80.7846; one-sided 90% Student-t limits (df=4): lower -96.059, upper -65.5103.

| Training seed | Equal-mix G | Scripted Δalive |
|---|---:|---:|
| 7101 | -66.6442 | 0.0105406 |
| 7102 | -110.713 | -0.0530062 |
| 7103 | -98.2378 | 0.0112719 |
| 7104 | -68.1785 | -0.0622375 |
| 7105 | -60.1498 | 0.0221 |

| Mix | Paired worlds | Candidate mass | B8 mass | Δmass | Candidate alive | B8 alive | Candidate no-death | B8 no-death |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scripted | 320 | 469.989 | 571.306 | -101.317 | 0.798796 | 0.813063 | 0.521875 | 0.515625 |
| frozen | 160 | 453.52 | 514.705 | -61.1853 | 0.904195 | 0.881968 | 0.7625 | 0.68125 |
| mixed | 160 | 452.659 | 532.51 | -79.8514 | 0.922949 | 0.926092 | 0.775 | 0.76875 |

## Decision evidence

- upper90(G) <= 0
- lower90_positive: fail
- mean_G_at_least_20: fail
- scripted_mass_nonnegative: fail
- at_least_four_positive_seed_effects: fail

## Descriptive checkpoints

| Checkpoint | Mean equal-mix G | Role |
|---|---:|---|
| 2M | unavailable | descriptive |
| 4M | unavailable | descriptive |
| 8M | -80.7846 | decisive |

Extension round: 0 (at most one).
Hierarchical paired-bootstrap sensitivity: [-104.88220383333334, -57.81272670833333] (descriptive).
Per-seed/mix paired variances and frozen artifact identities are retained in the JSON.
Low-boost floor: unevaluable: boost-eligible support counts were not instrumented.
