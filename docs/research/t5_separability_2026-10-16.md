# T5 input-feature separability — 2026-10-16

**Verdict: FAIL** — death-AUC gain 0.008826 (required 0.05), 24-alone AUC
0.747583 (required 0.80), and T1 17/46 = 36.9565% (required 70%).

This study implements the [T5 spec](specs/t5_separability_spec_2026-10-16.md) and
[binding input-design advice](advice/2026-10-16_inputs_design.md). It is a
CPU-only diagnostic on existing branch outcomes. No policy training, Numba port,
serving change, or promotion evaluation is part of this study.

## Features and the existing baseline

The new reference is `src/simd_env/t5_features.py`. It returns 24 float32 values,
flattened action-major: `area_log`, `boost3_area_log`, `core_area_log`,
`tail_reachable` for each of the six relative actions. Opponents remain static.
Area normalization is exactly `log2(min(area, 4096) + 1) / 12`; the floods themselves
visit the full reachable grid, without the v8 160-cell cap.

- `area_log`: flood from the hypothetical one-frame landing cell, with the own
  body advanced and boost burn applied. Movement and self collision use the
  completed post-frame body, including tails vacated by both boost substeps.
- `boost3_area_log`: three actual boosted frames from the decision state, straight
  in the resolved action direction, advancing the own body and applying each
  burn. Any collision or inability to complete all three boosted frames gives
  zero. Normal and boost actions with the same direction therefore share this
  feature. Its landing flood is tail-aware, like the other area floods.
- `core_area_log`: the same post-move tail-aware flood, excluding a candidate cell
  with at most two free four-neighbours. Free degree is computed from unpruned
  tail-aware passability at that cell's arrival distance, including own-body
  release; it is a single degree filter, not recursive erosion. An excluded seed
  has area zero.
- `tail_reachable`: whether the ordinary post-move flood visits a physical own-tail
  cell with movement TTL at most one. The BFS retains v8's conservative release
  slack of one, so physical TTL and BFS release distance are distinct.

As in the existing v5 hypothetical-move model, food growth during the
counterfactual is not simulated; current pending growth is preserved and the
flood retains the existing slack-one release rule. A one-step boost requested
below minimum length falls back to normal movement, matching v8. The distinct
three-boost feature requires three actual boosts.

**The 60-scalar baseline already includes uncapped tail-aware region sizes.**
`ego_raster_b.py` indices 23–25 are `region_frac_left/straight/right`
(`min(R/L, 4)/4`), and 26–28 are `region_log_left/straight/right`
(`log1p(R)/log1p(H*W)`). These six values are preserved byte-for-byte in all
60-scalar and 60+24 designs. The existing 30 action-specific v8 scalars occupy
indices 30–59. Consequently, the comparison measures incremental information
from the new post-move and boost lookahead features; it does not establish that
the original network lacked every uncapped region-scale signal.

## Bank restoration and labels

The source is `snake-dqn-artifacts/t4e-return-bank-20261016`. Its worlds differ
from the earlier T4b/T3 snapshot-store worlds, so this implementation uses the
spec's permitted deterministic replay path, reconstructing the exact source
world from each T4e plan. It captures observations and new features inside the
prepared decision callback, before the transition mutates the simulator.
Every restored state's packed raster bytes and all 60 float32 scalar bytes must
match its published bank observation. Source bank manifests and shard hashes
are validated before use; extraction source and input identities are checked
again at completion.

All **19,976 states in 342 worlds** passed byte identity checks. The merged
artifact retains exactly **103,264 legal action branches**:

| Bank family | States | Legal branches |
| --- | ---: | ---: |
| Training decisive | 9,462 | 49,653 |
| Training matched outside | 9,462 | 48,193 |
| Original holdout decisive | 1,052 | 5,418 |

There are 312 training worlds and 30 holdout worlds. The four extraction
partitions have disjoint state IDs and matching source hashes. Every one of the
19,976 emitted states is covered by the byte-parity receipt. Dataset SHA256:
`e4ecbe991061c90bb2ae3cdb4b236d0025f1219d61fa806b7e96f43962d9e02a`.

Only recorded legal action branches enter the death classifier. Death exactly
at transition 300 is positive; survival through transition 300 is negative for
"died within 300," without asserting any later outcome. No missing/illegal
branch is invented as a death. The state label is true when some legal boost
branch survives and every legal normal branch dies. Every source state has at
least one legal normal branch, so this qualification drops no bank states.

## Analysis protocol

The primary estimate is pooled out-of-fold AUC from five folds assigned by
unique world, using a fixed label-independent shuffled allocation (seed
20261016). All states and action branches from one world stay in the same fold.
The complete restored bank, including its historical holdout worlds, participates
in this primary grouped cross-validation. A separate secondary estimate fits
only the original training worlds and scores only the original holdout worlds.
No settings are selected using either result.

Each target is fit with three input sets: the original 60 scalars, the original
60 plus all 24 new features, and the 24 new features alone. Per-action death
models also receive six one-hot action indicators in **all three** designs, so
the actual design widths are 66, 90, and 30. State-level boost-only models have
widths 60, 84, and 24. This shared action context lets a classifier distinguish
branches of the same state. Logistic regression is an additive robustness check;
it does not explicitly add action-by-feature interaction terms.

`HistGradientBoostingClassifier` uses learning rate 0.1, 200 iterations, 31 maximum
leaf nodes, minimum leaf size 20, L2 regularization 1, and no early stopping.
Logistic regression uses fold-local standardization, C=1, LBFGS, and a 3,000
iteration cap; convergence warnings fail the run. Models are fit on one CPU
thread. These are supervised diagnostic fits, not policy training.

Intervals are percentile 95% CIs from 2,000 bootstrap samples of entire worlds
(seed 20261017), retaining all rows and multiplicities of each sampled world.
The same world draws are used for paired input-set comparisons. The intervals
condition on the saved held-out predictions; classifiers are not refit inside
bootstrap replicates. Each observed branch has equal weight for death AUC and
each state has equal weight for boost-only AUC. No family rebalancing is applied.

The published outside selection is concentrated in early B8 source finals
(8,571 states from s0, 518 from s1, 373 from s2). Grouping prevents within-world
leakage but does not remove this source-distribution limitation. These are bank
separability diagnostics, not fresh gameplay or promotion evidence.

## Results: grouped out-of-fold predictions

Death: 103,264 legal branches, 19,094 positive. Boost-only escape: 19,976 states,
1,481 positive. Both use all 342 observed worlds. Every CI below has 2,000 valid
world-bootstrap samples.

| Model | Inputs | Per-action death within 300: AUC [95% CI] | Boost-only escape: AUC [95% CI] |
| --- | --- | --- | --- |
| Histogram gradient boosting | 60 | 0.726124 [0.707854, 0.745854] | 0.726823 [0.697839, 0.754749] |
| Histogram gradient boosting | 60+24 | 0.734950 [0.716062, 0.755182] | 0.734273 [0.707795, 0.760433] |
| Histogram gradient boosting | 24 | 0.747583 [0.731339, 0.764371] | 0.730482 [0.703744, 0.757237] |
| Logistic regression | 60 | 0.729317 [0.711721, 0.746918] | 0.755473 [0.726952, 0.782515] |
| Logistic regression | 60+24 | 0.733416 [0.716905, 0.749731] | 0.758729 [0.730113, 0.785176] |
| Logistic regression | 24 | 0.726094 [0.708275, 0.744372] | 0.755333 [0.725747, 0.782380] |

The paired gradient-boosting death-AUC increment is **0.008826**,
95% CI **[0.005277, 0.012271]**.
The additive logistic-regression increment is 0.004099 [0.001051, 0.007113].

## Secondary result: original holdout

These models fit only the original 18,924 training states. They score 5,418
branches (1,901 deaths) and 1,052 states (165 boost-only escapes) in 30 original
holdout worlds. This decisive-only holdout has a different family distribution
from the pooled primary estimate.

| Model | Inputs | Per-action death within 300: AUC [95% CI] | Boost-only escape: AUC [95% CI] |
| --- | --- | --- | --- |
| Histogram gradient boosting | 60 | 0.764435 [0.722347, 0.807181] | 0.721103 [0.632792, 0.797977] |
| Histogram gradient boosting | 60+24 | 0.772046 [0.726584, 0.815438] | 0.736101 [0.653702, 0.810971] |
| Histogram gradient boosting | 24 | 0.782327 [0.743572, 0.820956] | 0.746500 [0.675264, 0.808342] |
| Logistic regression | 60 | 0.764925 [0.730390, 0.802006] | 0.747012 [0.653978, 0.825290] |
| Logistic regression | 60+24 | 0.763799 [0.728004, 0.802087] | 0.752718 [0.671865, 0.821579] |
| Logistic regression | 24 | 0.769022 [0.734780, 0.803667] | 0.753616 [0.671449, 0.823665] |

The secondary gradient-boosting death-AUC increment is 0.007611
[-0.001722, 0.018199]; 24 alone scores 0.782327. This also fails both point-estimate
separability thresholds. It does not replace the primary grouped-fold result.

## T1 late-LAD check

The recorded `best_surviving_action` has strictly greater `area_log` or
`boost3_area_log` than the recorded chosen action in **17/46 = 36.9565%** of late
LADs. The **70% bar is not met**. Area alone contributes 1/46; boost-three area
contributes 16/46; there is no overlap. The comparator action is T1's existing
best surviving action, not an action selected to maximize these new features.

All 46 rows match their canonical action, frame, hero length, and world seed.
Full cloned simulator states match, and each minimal feature-state snapshot was
saved, reloaded from disk, and recomputed with bitwise-identical features. The
T1 receipt distinguishes successful validation (`PASS`) from the failed feature
check (`FAIL`).

## Verdict

Advisor bar, verbatim:

> Proceed if the 60 + 24 AUC on per-action death beats the 60-scalar AUC by at least 0.05 and the 24-alone AUC is at least 0.80.

**FAIL.** Per-action death AUC rises from **0.726124** to
**0.734950**, an increment of **0.008826 < 0.05**;
**24-alone AUC = 0.747583 < 0.80**. The separate T1 check also
fails: **17/46 = 36.9565% < 70%**. The requested feature set does not meet the
advisor's proceed criterion on this bank. No policy training or Numba port was
started.

## Verification and reproducibility

- Focused suite: **20 passed** (`test_t5_features.py`, `test_t5_fit.py`,
  `test_t5_restore.py`), including 12 feature tests, grouped-fold and bootstrap
  checks, and the prepared-boundary regression test. Targeted Black, isort,
  flake8, and diff checks passed.
- Independent source-bank audit: every one of 360 shard hashes, 103,264 source
  branches, and all scalar bytes/labels/state identities matched the saved
  dataset. Independent numerical audit recomputed every one of 24 reported AUCs
  from saved predictions and checked worlds, folds, masks, counts, intervals,
  paired deltas, and the verdict. No unresolved blocking finding remains.
- Execution: four parallel single-thread CPU extraction processes, a single
  CPU fit process, and CPU-only T1 replay. All completed with exit code zero.
  Only task-owned processes were operated; no tmux session was accessed.
- Fit wall time: **130.23 seconds**. Python 3.12.3,
  NumPy 1.26.4, SciPy 1.14.1, scikit-learn 1.7.2, PyTorch 2.9.1.
  scikit-learn was installed in the task-local artifact `deps` directory,
  without modifying the shared project environment.
- Extraction/fit source commit: `212d5e92bc7d8c773ad63b3774b5d7f39d615e7a`.
  The NumPy feature implementation was first committed in `9ebfe504`.
  Every implementation commit was fetched/merged against `origin/main` and
  pushed to both `main` and `t5-20261016` as specified.

Tracked evidence: [fits](t5_separability_2026-10-16_evidence/fits.json),
[restoration receipt](t5_separability_2026-10-16_evidence/restoration.json),
[T1 per-row evidence](t5_separability_2026-10-16_evidence/t1_check.json),
[bank audit](t5_separability_2026-10-16_evidence/independent-feature-audit.json),
[prediction audit](t5_separability_2026-10-16_evidence/independent-fit-audit.json),
[environment](t5_separability_2026-10-16_evidence/environment.json), and
[artifact identities](t5_separability_2026-10-16_evidence/artifact-manifest.json).
Four partition receipts preserve timings, all-state parity counts, and concrete
byte-match examples for training decisive, training outside, and holdout states.

Local raw artifacts are under
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t5-separability-20261016/`:
`features.npz`, `fits/*.predictions.npz`, `fits/folds.npz`, and
`t1_feature_states.final.npz`. They remain local; the compact receipts and
per-LAD values are committed.

The reproducible entry points are:

```sh
export SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t5-separability-20261016/deps:.

# Use a fresh output directory. Run i=0,1,2,3, with at most four concurrent workers.
./venv/bin/python -m research.t5_separability_20261016.restore \
  --shard-index "$i" --shard-count 4 --out "$out/features-part-$i.npz"
./venv/bin/python -m research.t5_separability_20261016.merge \
  --parts "$out/features-part-0.npz" "$out/features-part-1.npz" \
          "$out/features-part-2.npz" "$out/features-part-3.npz" \
  --out "$out/features.npz"
./venv/bin/python -m research.t5_separability_20261016.fit \
  --data "$out/features.npz" --out "$out/fits"
./venv/bin/python -m research.t5_separability_20261016.t1_check \
  --out "$out/t1_check.json" --states-out "$out/t1_feature_states.npz"
./venv/bin/python -m pytest -q tests/test_t5_features.py \
  tests/test_t5_fit.py tests/test_t5_restore.py
```

The optional analysis dependencies are pinned in
`research/t5_separability_20261016/requirements.txt`; install them with the
project venv's pip and `--target` to a task-local dependency directory when needed.

