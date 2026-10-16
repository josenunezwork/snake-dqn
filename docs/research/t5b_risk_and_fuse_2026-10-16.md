# T5b: B8 survival risk and fuse-layer separability — 2026-10-16

**Development diagnostics only. All outcome data are reused; nothing is re-scored as gate evidence. No policy training, new gameplay, serving change, or promotion evaluation was run.**

This report implements the [T5b spec](specs/t5b_risk_and_fuse_spec_2026-10-16.md), [binding strategy, sections 1 and 3](advice/2026-10-16_strategy.md), and the [prospective guard amendment](specs/guard_clause_amendment_2026-10-15.md). It preserves the original M4 **PARTIAL** outcome and the [T5 separability result](t5_separability_2026-10-16.md).

**Result:** modeled survival-guard failure risk is **38.81%**, above the advisor’s 25% flag. Tier 2 remains **FAIL** on the available champion comparisons. All fuse probes are below **0.85 AUC**; under the advisor’s rule the reading is **perception**, with GBM AUCs of **0.7630–0.7693**.

## Historical Tier 1 risk diagnostic

**Combined requested survival-clause risk: 38.81%**, under the joint plug-in normal forecast below. This exceeds the advisor’s quarter-risk flag. It is a forecast from reused historical banks; no historical result was re-scored as gate evidence and no new gameplay was run.

The original M4 Phase R outcome remains **PARTIAL**: it passed the efficacy clauses but failed its then-binding champion scripted-survival guard. That failure helped motivate the prospective 2026-10-15 guard amendment. The amendment permits no second survival-guard amendment after another failure. These diagnostics do not authorize release or select a serving seed.

### Inventory and independent units

The pool contains 1,280 unique B8 policy/mix/world rows: 640 scripted and 320 each frozen/mixed. The original round-52 32-world dev probes, census, K1 identity replay, M7 governor-off and feature-probe replays are deduplicated by exact final checkpoint SHA, mix and world seed at H5000. The 64 later K1 scripted worlds per final extend round 52 to 96 shared worlds. M4 look-0/1/2 incremental shards provide the final 32-world banks per seed and mix; H10000 records contribute only their explicit H5000 prefixes. Instrumented repeats agree on all known metrics. No five-fold multiplication of the 32 or 96 shared round-52 world identities is used for observed uncertainty.

Including references there are 1,984 canonical rows, with 944 duplicate observations merged. The manifest records every inspected input, its SHA256, and accepted observation count; zero denotes a scanned input with no eligible rows. Source identity and coverage checks fail closed on missing banks, wrong comparators, altered producer code, or changed inputs. Legacy reference payloads lack checkpoint identities; their attribution is bound to four pinned original producer modules and their named checkpoint constants. M4 references additionally carry validated explicit checkpoint SHA and v8 metadata.

The extra 64 scripted K1 worlds have no served or champion comparator. They inform absolute survival only. Old dev-probe records retain mass/deaths without alive-frame survival, so frozen/mixed paired c2 uses M4 only. Tier 2 includes census champion comparisons and M4 seed-0 champion comparisons; no unavailable champion bank is invented. T3/T4e label-collection trajectories have outcome-dependent stopping or quotas and are excluded from this complete-probe pool. Governor-on, ensembles with more than one member, non-final checkpoints, controls, and full H10000 outcomes are excluded.

### Per-clause forecast

Every mean below equally weights the five finals (a single final for a per-seed row). Observed SE uses a world-cluster sandwich estimator, retaining dependence between finals that reuse a round-52 world/reference. Forecast SE estimates sampling noise on fresh independent per-seed banks, conditional on the historical means and variances. It does **not** add the uncertainty in those historical means. Fractions are on a 0–1 scale.

| Clause | Mean | Observed clustered SE | Historical rows / unique worlds | Fresh worlds | Forecast SE | Failure by noise |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| c2 scripted Δ ≥ −0.03 | -0.016471 | 0.031440 | 320 / 192 | 480 | 0.018458 | 23.18% |
| c2 frozen Δ ≥ −0.03 | +0.011102 | 0.023227 | 160 / 160 | 320 | 0.016634 | 0.67% |
| c2 mixed Δ ≥ −0.03 | +0.038045 | 0.020010 | 160 / 160 | 320 | 0.014331 | <0.005% |
| Seed 0 scripted Δ ≥ −0.15 | -0.032650 | 0.050508 | 64 / 64 | 96 | 0.041240 | 0.22% |
| Seed 1 scripted Δ ≥ −0.15 | -0.070100 | 0.047578 | 64 / 64 | 96 | 0.038847 | 1.99% |
| Seed 2 scripted Δ ≥ −0.15 | +0.056175 | 0.056101 | 64 / 64 | 96 | 0.045806 | <0.005% |
| Seed 3 scripted Δ ≥ −0.15 | +0.056619 | 0.047441 | 64 / 64 | 96 | 0.038736 | <0.005% |
| Seed 4 scripted Δ ≥ −0.15 | -0.092397 | 0.050637 | 64 / 64 | 96 | 0.041345 | 8.18% |
| Absolute scripted episode ≥ 0.50 | +0.517188 | 0.020682 | 640 / 256 | 480 | 0.022794 | 22.54% |
| Absolute frozen episode ≥ 0.50 | +0.678125 | 0.027390 | 320 / 192 | 320 | 0.025862 | <0.005% |
| Absolute mixed episode ≥ 0.50 | +0.828125 | 0.021590 | 320 / 192 | 320 | 0.021242 | <0.005% |

The table uses the binding amendment’s 480 scripted and 320 frozen/mixed fresh worlds. The T5b spec separately requests the absolute floor’s noise at **480 worlds on each mix**: scripted SE 0.022794, failure 22.54%; frozen SE 0.021117, failure 1.65e-15%; mixed SE 0.017344, failure 4.02e-78%. A common-rate binomial sensitivity for the scripted absolute floor is 21.20% (239 or fewer survivors at 480, using the pooled survival rate). That binomial calculation assumes one success probability across finals; the primary normal forecast retains their observed variance differences.

### Combined risk and dependence

The combined forecast covers **11 requested survival clauses**, not the whole Tier 1 gate: three pooled c2 survival checks, five scripted per-seed floors, and three absolute episode floors. The unchanged efficacy mass, mi10, positive-seed and guard-mass clauses are not forecast here.

For each final, all six mix/metric correlations (paired alive-frame Δ and candidate episode survival on each mix) are estimated from its 32 complete M4 worlds. Their marginal means and variances use all available eligible observations. A joint normal draw keeps c2, per-seed-floor, absolute-floor and cross-mix dependence. Scripted has 96 fresh worlds, with frozen/mixed using the first 64; the five new seed banks are assumed independent. This uses 200,000 draws, RNG seed 20261016. The correlation estimate is noisy at 32 worlds, and future bank distributions, historical selection effects and parameter uncertainty remain unquantified.

- Joint risk with binding bank sizes: **38.81%**, Monte Carlo SE 0.11 percentage points.
- With the requested absolute-floor 480-world sensitivity on every mix: **38.66%**. The small difference is within Monte Carlo noise; frozen/mixed absolute-floor risks are negligible in both projections.
- Marginal independence approximation: **46.93%**; it ignores overlapping checks and is not the main estimate.
- Dependence-free marginal bounds: **23.18%–56.78%**.

### Tier 2 and required historical sensitivity

**Tier 2 retrospective status: FAIL.** Equally weighting the five finals, scripted episode survival minus champion_a5 + v8 is **-0.171875**, below −0.05 (observed world-cluster SE 0.082690). Coverage is 192 candidate/reference pairs across 64 distinct worlds: seed 0 has 64 pairs and seeds 1–4 have 32 each. The equal-final estimate avoids giving seed 0 twice the weight solely because it has more champion measurements. At 480 fresh worlds the conditional noise forecast SE is 0.031173 and failure probability 99.995%; incomplete historical champion-bank coverage limits this extrapolation.

| Already-seen bank / comparator | Alive-frame Δ | Episode Δ | Scope |
| --- | ---: | ---: | --- |
| M4 final look vs champion_a5 + v8 | −0.1559875 | −0.187500 | Seed 0, 32 worlds |
| Census round 52 vs champion_a5 + v8 | −0.03833125 | −0.156250 | Five finals, same 32 worlds |
| M4 final look vs served frp3-s12 + v8 | −0.0025975 | −0.050000 | Five banks, 32 worlds each |
| Census round 52 vs served frp3-s12 + v8 | −0.03034375 | −0.125000 | Five finals, same 32 worlds |

The amendment quotes the first three comparisons as −0.156/−0.188, −0.039/−0.157, and −0.003/−0.050. The table above recalculates the raw records: the census values are slightly different from those historical summaries. Neither version is re-scored as gate evidence. This expanded pool differs from the advisor’s earlier 0.531 scripted survival summary: it adds M4 and later K1 worlds and now estimates **0.5171875**.

Reproduce with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ./venv/bin/python -m research.t5b_risk_and_fuse_20261016.risk --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t5b-risk-and-fuse-20261016/risk.json`. Six focused unit tests cover conflicting repeats, nullable metrics, shared-world dependence, unequal final coverage, comparator identity, and missing source banks. No migration, backfill or mutable data service is involved; all inputs are read-only and outputs are additive diagnostic artifacts.

## Fuse-layer per-action probe

All five unchanged B8 finals are forwarded over the same **19,976 restored states in 342 worlds**, retaining exactly **103,264 legal branches** and **19,094 death labels**. Packed observations are joined from the T4e source shards to the hash-pinned T5 state IDs; every scalar byte must match. Both bank manifests are bound to the published T5 restoration receipt, every shard hash is checked, and all five final checkpoint SHA256 values are pinned. There is no simulator replay or new branch rollout in T5b.

The hook reads the **256-dimensional post-ReLU output of `fuse[2]`**, as in T1C, together with all six Q values. Features and Q values are retained as float32, the same precision used by the classifiers. The two designs are (a) fuse + Q and (b) fuse + Q + all 60 original scalars + all 24 T5 features. Both receive six one-hot action indicators, giving widths **268** and **352**. Only actual legal branches appear; illegal or missing branches do not become deaths. Death at transition 300 is positive, and survival through 300 is negative for this horizon.

The primary estimate uses T5's saved five world folds exactly, including historical training and holdout worlds, with no world split between fit and test. Each final has its own GBM and logistic probe for each design. Model settings match T5: histogram GBM learning rate 0.1, 200 iterations, 31 maximum leaves, minimum leaf size 20, L2=1, no early stopping; logistic regression uses fold-local standardization, C=1, LBFGS and a 3,000-iteration cap, with convergence warnings fatal. These supervised diagnostic fits do not update any policy weight.

AUC intervals are percentile 95% CIs from 2,000 whole-world bootstrap draws (seed 20261017). They retain every branch and sampled-world multiplicity and condition on saved out-of-fold predictions; models are not refit inside the bootstrap. The same draw sequence is used across probes. All runs use one CPU thread each, with five simultaneous arm processes, below the eight-process limit.

| Final | Inputs | GBM AUC [95% CI] | Logistic AUC [95% CI] |
| --- | --- | ---: | ---: |
| B8_s0 | fuse + Q | 0.769139 [0.751108, 0.787620] | 0.725476 [0.708878, 0.742492] |
| B8_s0 | fuse + Q + 60 + 24 | 0.767851 [0.749049, 0.788016] | 0.727646 [0.711416, 0.743418] |
| B8_s1 | fuse + Q | 0.769343 [0.750706, 0.788203] | 0.726249 [0.709263, 0.742796] |
| B8_s1 | fuse + Q + 60 + 24 | 0.768291 [0.749565, 0.788086] | 0.729601 [0.713448, 0.745433] |
| B8_s2 | fuse + Q | 0.766462 [0.748415, 0.785515] | 0.725989 [0.709484, 0.743197] |
| B8_s2 | fuse + Q + 60 + 24 | 0.767300 [0.748633, 0.787300] | 0.728465 [0.712366, 0.744384] |
| B8_s3 | fuse + Q | 0.765263 [0.747133, 0.783661] | 0.725616 [0.708853, 0.742694] |
| B8_s3 | fuse + Q + 60 + 24 | 0.766100 [0.747131, 0.786074] | 0.728200 [0.712126, 0.744446] |
| B8_s4 | fuse + Q | 0.762998 [0.744559, 0.781217] | 0.727439 [0.710900, 0.744122] |
| B8_s4 | fuse + Q + 60 + 24 | 0.763103 [0.744173, 0.783207] | 0.730243 [0.714558, 0.745709] |

T5's GBM baselines were **0.726124 [0.707854, 0.745854]** for the 60 scalars, **0.747583 [0.731339, 0.764371]** for the 24 T5 features alone, and **0.734950 [0.716062, 0.755182]** for 60 + 24. Fuse + Q yields 0.7630–0.7693 across the five finals; the expanded design yields 0.7631–0.7683. Logistic regression remains at 0.7255–0.7302. Every result has 2,000 valid world-bootstrap draws.

**Advisor reading: perception.** No final or input design reaches the stated **AUC ≥ 0.85** learning-failure criterion; even the largest upper confidence bound is below 0.80. The fuse probe therefore does not trigger the advisor's exception for reopening a learning/label arm. This is a threshold-based diagnostic of the frozen representation on the existing bank, not proof that the raw raster contains no recoverable information or that another architecture cannot improve it. No serving seed is selected from these AUCs.


The labels remain outcomes of the bank's original continuation policies. Evaluating another final on a state does not turn those labels into that final's own counterfactual outcomes. The original outside-state selection is concentrated in source final s0, and the historical holdout is included in the primary grouped cross-validation just as in T5. These constraints limit the reading to separability on this reused bank.

## Verification and reproducibility

The implementation was committed as `fc0b9d76`, followed by source-read consistency hardening in `b139d74c`. The numerical fuse runs use the latter source. Audited risk receipts were committed in `95f49e2b`. After each commit the branch fetched and merged `origin/main`, then pushed `HEAD:main` and `t5-20261016`, as specified.

- Focused tests: **14 passed** across `test_t5b_risk.py`, `test_t5b_fuse.py`, and `test_t5_fit.py`; targeted Black, isort and flake8 passed.
- Independent risk audit: **PASS**. It checked 1,221 recorded source hashes, reconstructed 2,928 accepted observations, removed 944 duplicates, and matched all 1,984 canonical rows. It independently recomputed all clause means, clustered SEs, projected SEs and marginal risks. A separate 200,000-draw joint simulation returned 38.60%, consistent with 38.81% within Monte Carlo error.
- Independent fuse audit: **PASS**. It verified all 360 source shards, all-state scalar identity, saved feature state IDs/dtypes, all ten prediction arrays against canonical labels/actions/worlds/folds, and independently recomputed all 20 AUCs and 2,000-world-bootstrap intervals. A six-row-per-final source re-forward used batch size 6 rather than the saved batch size 128; maximum absolute differences were 2.75e-6 for fuse and 5.13e-6 for Q. This sampled check is numerical agreement, not a claim of bitwise forward identity.
- Five isolated, single-thread CPU arm runs finished in **346.4–356.8 seconds per arm**, approximately six minutes elapsed in parallel. Their checkpoints and source identities were unchanged across each run.
- No tmux session was accessed. Only task-owned process sessions were operated. No checkpoint, trainer, simulator, serving file, prior study, or hash-bound governance document was changed.

Tracked evidence: [risk calculations](t5b_risk_and_fuse_2026-10-16_evidence/risk.json), [risk source inventory](t5b_risk_and_fuse_2026-10-16_evidence/risk-inputs.json), [independent risk audit](t5b_risk_and_fuse_2026-10-16_evidence/risk-independent-audit.json), [fuse results](t5b_risk_and_fuse_2026-10-16_evidence/fuse.json), [independent fuse audit](t5b_risk_and_fuse_2026-10-16_evidence/fuse-independent-audit.json), [environment](t5b_risk_and_fuse_2026-10-16_evidence/environment.json), [verification](t5b_risk_and_fuse_2026-10-16_evidence/verification.json), and [artifact identities](t5b_risk_and_fuse_2026-10-16_evidence/artifact-manifest.json).

Raw features, predictions and logs remain under `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t5b-risk-and-fuse-20261016/`. Dependencies are reused from T5's task-local `deps` directory without modifying the shared project environment.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export SNAKE_DQN_DEVICE=cpu
export PYTHONPATH=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t5-separability-20261016/deps:.

./venv/bin/python -m research.t5b_risk_and_fuse_20261016.risk \
  --out "$out/risk.json"
# Run once per k=0,1,2,3,4, using separate new output directories.
# At most five simultaneous single-thread arm processes were used here.
./venv/bin/python -m research.t5b_risk_and_fuse_20261016.fuse \
  --arms "B8_s$k" --out "$out/runs/B8_s$k" --boot 2000
./venv/bin/python -m pytest -q tests/test_t5b_risk.py \
  tests/test_t5b_fuse.py tests/test_t5_fit.py
```

