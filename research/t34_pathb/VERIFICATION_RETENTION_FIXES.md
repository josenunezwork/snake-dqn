# T34 retention FIX-FIRST implementation — 2026-10-20

Implementation only, pre-data. No T3 pre-check, retention efficacy, timing gameplay,
training, live service or tmux operation was run. The ratified-registration artifact
is not manufactured: the refreshed package stays an unauthorized draft.

Worktree: `/Users/josenunez/Projects/ml/snake-dqn-retention-fixes`.
Branch: `t34-retention-fixes-20261020`. Clean baseline:
`a48e89b1130fb1ebea49bae5c809a5f01f52edab`.
Authority: [operator FIX-FIRST rulings](../../docs/research/specs/2026-10-20_t34_retention_prereg_fixes.md).

## Changes

1. Paced efficacy overruns are diagnostic, with their bare fallbacks retained in
   efficacy; zero-overrun enforcement is confined to registered timing probes.
2. Near-zero mass uncertainty requires an actual failing mass category, all
   other skill categories passing and product passing; a complete pass remains GO.
3. The 24 fps probe gates update p95 <=11.688 ms; 12 fps reports p95. Both require
   zero overruns and matching ON/OFF cadence. Timing precedes T3 pre-check and efficacy.
4. Measured host admission and periodic checks replace attestation. Prospective
   thresholds: 30 s checks, load excess >3 after own two-worker allowance, another
   process >50% CPU for >60 s, known heavy runners, AC/lid/thermal. Pause on change;
   persistent nonquiet conditions for 120 s yield INCONCLUSIVE.
5. Live entry requires exact ratified JSON bytes committed in HEAD and the branch
   HEAD observed on origin. Create-only decision.json is the sole verdict; repeated
   analysis is descriptive. Frozen source and execution commits are separately recorded.
6. OC acceptance reports its observed maximum separately from the .06 limit;
   registration and profile share identical validated load-policy keys.
7. Development-only power uses T3 shapes and separate T15/T13b mixed-Watch6
   baselines. N32/mix and 16 product sessions remain fixed. Joint modeled power
   explicitly states all assumptions; low T3-like power is reported for the operator.

## Verification

Frozen executable/source-inventory revision:
`b5a2596c8b60caf52ed2ed9c3692aba4bff06bbb` (includes concurrent main integration).
Later JSON/document/test-fixture-only publication commits preserve that closure.
The final full retention selection passed **103 tests in 10.28 s**: the original
71-test selection plus 32 added cases. There were no skips or deselections.
Black reports 161 checked Python files unchanged; isort and flake8 pass.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-retention-fixes
export SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
"$PY" -m pytest -q \
  tests/test_t34_pathb_registration_retention_v2.py \
  tests/test_t34_pathb_retention_oc_v2.py \
  tests/test_t34_pathb_retention_v2.py \
  tests/test_t34_pathb_analysis_v2.py \
  tests/test_t34_pathb_timing_v2.py \
  tests/test_t34_retention_fixes_analyzer.py \
  tests/test_t34_retention_fixes_power.py \
  tests/test_t34_retention_fixes_runner.py
CHECKS=(src web/backend \
  research/t34_pathb/{analyze_v2,host_guard_v2,registration_retention_v2,retention_v2,retention_oc_v2,timing_v2}.py \
  tests/test_t34_pathb_{registration_retention_v2,retention_oc_v2,retention_v2}.py \
  tests/test_t34_retention_fixes_{analyzer,power,runner}.py)
"$PY" -m black --workers 1 --check --line-length 100 "${CHECKS[@]}"
"$PY" -m isort --profile black --check-only "${CHECKS[@]}"
"$PY" -m flake8 --jobs 1 "${CHECKS[@]}"
"$PY" -m research.t34_pathb.retention_v2 --help
"$PY" -m research.t34_pathb.registration_retention_v2 ratify --help
"$PY" -m research.t34_pathb.registration_retention_v2 validate
```

The production draft validator returns
`{"valid":true,"authorized":false,"mode":"draft"}`. The final ready registration
is not minted. A real final-source trace receipt and operator ratification are
still required before any timing/pre-check/efficacy work. See the complete
[timing-first operator sequence](README_V2.md#retention-registration-package).

## Development-only OC and power

The first real-data preparation command failed before writing a state or receipt:
it exposed the original T3 baseline key `survived`, rather than candidate key
`episode_survival`. Corrected in commit `e4574ce5`; a read-only real-data loader
smoke then read two 32-world T3 pools, 25 T15 sessions, 25 T13b sessions and 90
original file bindings. The affected 30 registration/OC/power tests passed after
the correction. Failed preparation consumed no simulation or efficacy row.

The final one-thread simulation command was:

```bash
"$PY" -m research.t34_pathb.retention_oc_v2 \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-retention-fixes-20261020/oc-02.json \
  --state /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-retention-fixes-20261020/oc-02-state.json \
  --reps 20000 --seed 3420261020 --max-seconds 60
```

Result: **PASS**, 20,000 replicates, zero failing protected cells, observed maximum
protected-regression joint rate **0.0136** versus the separately recorded **0.06**
limit. There are 120 Path B joint-power cells and four product-power cells.
Timing, boost compliance and unchanged product growth are explicit planning
assumptions. T3 scripted shapes proxy frozen/mixed variance; paired death
thinning is an optimistic effect model, not observed candidate efficacy.

| Watch baseline | 16-session product power, 20% fewer deaths | 40% fewer deaths |
|---|---:|---:|
| T15 mixed Watch6 | 0.46160 | 0.90120 |
| T13b mixed Watch6 | 0.54845 | 0.94940 |

At the declared T3-like +92 mass / +.25 scripted no-death scenario, joint GO power
is **0.26630 / 0.31715** at 20% death reduction for T15 / T13b, and
**0.52050 / 0.54825** at 40%. `t3_like_power_review_required.required` is **true**.
Per ruling 7, **N remains 32/mix**; only the operator decides whether to change
funding or design prospectively. This is no retention verdict or release claim.

The refreshed draft was built create-only outside the repository, then copied
to the explicitly authorized pre-data draft path:

```bash
"$PY" -m research.t34_pathb.registration_retention_v2 build \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-retention-fixes-20261020/draft-02.json \
  --checkpoints research/t34_pathb/RETENTION_CHECKPOINTS_V2.json
"$PY" -m research.t34_pathb.registration_retention_v2 validate
```

SHA-256 bindings:

| Artifact | SHA-256 |
|---|---|
| RETENTION_OC_V2.json | `ebe295826ef088397af73e0d55de46681760c9a8deaed8161a4dd94a1eb541ce` |
| REGISTRATION_RETENTION_V2.json | `5afe9e274b6887aeec34b2b7f7718a32d8ae8079f96f205862fa7eb2779e8d5a` |
| RETENTION_PROFILE_V2.json | `90b3054d8480cccb904a7f957d007511ce45d53c32c4a88226ac222f38d05d8f` |
| PREREGISTRATION_RETENTION_V2.md | `1a0889ea3084bd6396f43410d843d957cfa54e97bed7eb56384b74c63589da19` |

## Scope and remaining operational checks

No scientific run, live timing qualification, pre-check or efficacy row was
produced. No service/tmux action, checkpoint write, serving-pin change, training,
cloud allocation or release action occurred. The committed terminal/product/
timing banks, their collision audit and checkpoint inventories are unchanged.
The pre-data draft preregistration/profile/OC/registration were updated only as
expressly directed by the FIX-FIRST specification; original versions remain in Git.

Independent source review identified and root integration fixed admission-bypass,
ancestor-exclusion, terminal-decision, cap-debit, timing-handoff and bounded-query
issues. New tests use fake host samples/adapters and never open real worlds.
Actual host admission, the ~.36 h timing gate, the 64-trajectory T3 pre-check and
fixed-N efficacy remain future, separately authorized operator work.

