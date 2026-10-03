# Sequential strict gate: Monte Carlo validation, 2026-10-02

Label: **diagnostic (Tier 0 style; synthetic data only)**. No game was run, no record was read,
and nothing here is evidence about any candidate. It validates the opt-in group-sequential
version of the strict decision in `src/evaluation/sequential_gate.py`, proposed by
[`docs/research/governance_amendment_sequential_gates_2026-10-02.md`](../../docs/research/governance_amendment_sequential_gates_2026-10-02.md).

Question (written before any run): with paired-delta SD 140 and N_max 243 worlds per mix, does
the default sequential gate keep type-I error at or below 0.05? What power and expected world
count does it give at deltas 30, 60 and 130, compared with the fixed-N strict gate?

## How to reproduce

```
OMP_NUM_THREADS=1 ./venv/bin/python research/sequential_gate_validation_20261002/simulate.py --part mc --out <dir outside repo>
OMP_NUM_THREADS=1 ./venv/bin/python research/sequential_gate_validation_20261002/simulate.py --part crosscheck --out <dir outside repo>
```

The script pins every BLAS/OpenMP thread variable to 1 before it imports numpy. It uses numpy
plus the repo's stdlib statistics and nothing else, and it has fixed seeds. Only the
`cpu_seconds` fields change between runs. Measured CPU time: `mc` 17.0 s, `crosscheck` 43.7 s.
The outputs for this README were written to the session scratchpad (`seqgate/mc.json`, sha256
`60f6eb2c…8a2c`; `seqgate/crosscheck.json`, sha256 `329af746…1a08`). They are not in the repo.

## Design simulated

| Item | Value |
|---|---|
| Looks (worlds per mix) | 61, 122, 183, 243 (fractions 0.251, 0.502, 0.753, 1) |
| Efficacy, per mix | one-sided, Lan-DeMets O'Brien-Fleming at 0.05/3; z bounds 4.637, 3.183, 2.539, 2.174; nominal p 1.77e-6, 7.28e-4, 5.57e-3, 1.485e-2 |
| Scripted NI | own OBF spending at 0.05; z bounds 3.741, 2.534, 2.011, 1.721; nominal p 9.2e-5, 5.64e-3, 2.215e-2, 4.261e-2 |
| Statistic | paired t per mix, compared with its own t distribution at the nominal p (tail matching) |
| Futility | non-binding; a mix is futile if its conditional power under MDE 30 is below 0.10; stop when 2 or more mixes are futile |
| PASS | at least 2 of 3 mixes crossed, NI established, bands pass. Bands are not simulated and are assumed to pass. |
| Data | normal paired deltas, SD 140 per mix. Mixes are independent, or have between-mix correlation 0.5 (`corr50`), because the strict mixes share world seeds. |
| delta_NI | 3.5. This is illustrative: the v5 strict margin was 3.17, and each study takes its real margin from calibration. |

"Fixed Holm" is the current `strict_promotion_decision`: Holm at 0.05 plus NI at 0.05 on all
243 worlds. "Fixed Bonferroni" is the same with 0.05/3 per mix and no step-down.

## Exactness cross-check

The Monte Carlo uses a vectorized replica of the decision. On 1,250 fresh replicates (250 each
from null, LFC, NI-null, delta 30 and delta 60), it was compared with the pure
`run_sequential_gate` and with `eval_stats.strict_promotion_decision`. It gave the same
decision and stop look on **all 1,250** replicates (0 mismatches), and the same fixed-N Holm
verdict on **all 1,250** (0 mismatches).

## Type-I error (95% Wilson intervals)

| Scenario (true deltas frozen, scripted, mixed) | Reps | Sequential P(PASS) | Sequential false rejection of any null mix | Fixed Holm P(PASS) / FWER |
|---|---|---|---|---|
| Global null 0, 0, 0 | 200k | 0.00058 [0.0005, 0.0007] | 0.0492 [0.0483, 0.0502] with futility overridden (theory: 1 - (1 - 0.05/3)^3 = 0.0492); 0.0270 with futility honored | 0.0011 / 0.0493 |
| Global null, corr 0.5 | 200k | 0.0053 [0.0049, 0.0056] | 0.0438 [0.0429, 0.0447] overridden; 0.0355 honored | 0.0085 / 0.0435 |
| LFC 130, 0, 0 | 200k | 0.0182 [0.0176, 0.0188] | 0.0336 [0.0328, 0.0344] (theory 2 x 0.05/3 = 0.0333) | 0.0269 / 0.0493 |
| NI null 130, -3.5, 130 | 200k | 0.0500 [0.0491, 0.0510] | n/a (no null superiority mix) | 0.0499 |

Precision check of the t-matched boundaries, with 2M replicates of one mix:

- the NI null crosses at some look at rate 0.05011 [0.04981, 0.05041] (target 0.05);
- the efficacy null crosses at rate 0.016680 [0.016503, 0.016858] (target 0.016667).

The known-variance recursion gives exactly 0.016667. The z/t approximation is not
detectable at these look sizes.

**Verdict.** Type-I error is at or below 0.05 in every configuration. The global-null false
PASS rate is 0.0006, and the familywise false rejection is 0.0492 even with every futility
stop overridden. The NI-null case sits at its nominal 0.05, exactly as the fixed gate does.
That is the boundary case of the intersection-union test, not inflation. Under the global
null, the 0.0502 upper Wilson bound is MC noise around the analytic 0.0492.

## Power and expected sample size (futility honored)

| True delta (all mixes) | Reps | Sequential P(PASS) | Fixed Holm | Fixed Bonferroni | E[worlds per mix] | E[paired worlds, all mixes] vs 729 fixed | Saving | Stop look distribution (1-4) |
|---|---|---|---|---|---|---|---|---|
| 0 (null) | 200k | 0.0006 | 0.0011 | 0.0006 | 177.3 | 532 | 27.0% | 0, 0.201, 0.690, 0.109 |
| 0, corr 0.5 | 200k | 0.0053 | 0.0085 | 0.0053 | 177.6 | 533 | 26.9% | 0.000, 0.253, 0.580, 0.167 |
| 30 | 50k | 0.9434 [0.9414, 0.9454] | 0.9628 | 0.9481 | 199.1 | 597 | 18.1% | 0, 0.081, 0.569, 0.350 |
| 30, corr 0.5 | 50k | 0.9122 [0.9097, 0.9146] | 0.9384 | 0.9173 | 195.0 | 585 | 19.8% | 0.000, 0.145, 0.507, 0.347 |
| 60 | 50k | 1.0000 (50,000 of 50,000) | 1.0000 | 1.0000 | 122.9 | 369 | 49.4% | 0.008, 0.969, 0.023, 0 |
| 130 | 50k | 1.0000 (50,000 of 50,000) | 1.0000 | 1.0000 | 61.1 | 183 | 74.8% | 0.998, 0.002, 0, 0 |

Outcome splits:

- **Delta 30:** STOP_PASS 0.649, FINAL_PASS 0.295, FINAL_FAIL 0.056, STOP_FUTILE 0.0011, so
  futility costs at most 0.1 point of power.
- **Global null:** STOP_FUTILE 0.891, FINAL_FAIL 0.109.
- **LFC:** STOP_FUTILE 0.628.

Known-variance per-mix crossing probability by recursive integration: 0.882 at delta 30, 0.99999
at delta 60. The MC per-mix rate at delta 30 is 0.878, consistent with that, after the small
t-distribution loss.

## Reading

- **Expected work drops a lot when the effect is large or absent.** The v7 screen effects
  (+135 to +154) are in the delta-130 row: about 61 instead of 243 worlds per mix, a 75% saving.
  The v7 projection is 6.1 h of final-stage wall clock, which scales to about 1.6 h plus look
  overhead. That figure is a projection, not a measurement. A truly null candidate stops for
  futility at the second or third look in 89% of runs (27% saving).
- **Power cost at the MDE is small and mostly from Bonferroni.** At delta 30 the sequential gate
  passes 94.3%, against 96.3% for fixed Holm. Fixed Bonferroni already loses 1.5 points
  (94.8%). The sequential looks cost 0.5 point, and futility costs about 0.1. Most of the gap
  could come back with a graphical Holm-type sequential procedure (Maurer and Bretz 2013). That
  procedure is not implemented here.
- **The guarantee is unchanged in kind.** With familywise error at most 0.05, a PASS means that,
  with probability at least 0.95, every mix it counts is truly superior. Holm's fixed gate
  gives the same guarantee. Bonferroni is conservative under any between-mix dependence. The
  corr-0.5 runs show the familywise rate falling (0.0438), while P(2 or more false) rises to
  0.0053. That is still far below 0.05.

## Limits

- The data are normal. Real paired mass-integral deltas are skewed and heavy-tailed. At the
  first look (61 worlds), a skewed distribution could move the t-matched levels more than this
  shows. Before the first sequential Tier-2 run, repeat the type-I check with a bootstrap from
  saved screen deltas (Tier 0, saved data).
- The behavioral bands are assumed to pass. Overrun and look-barrier timing are not simulated.
- delta_NI = 3.5 and MDE 30 are illustrative. Each study simulates its own plan.
