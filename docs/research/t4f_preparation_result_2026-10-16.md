# T4f RETURN_C preparation result — 2026-10-16

A, B, and C are complete. The five 8M RETURN_C runs are unstarted. This receipt
reports implementation, collection, and calibration, not a trained-policy result.
Artifact paths are under `/Users/josenunez/Projects/ml/snake-dqn-artifacts/`.

## Implementation and verification

The binding Planner addendum is implemented: full-target argmax reference with
detached reference Q, uniform epsilon-insensitive Huber (epsilon2, delta100),
1% L<700 /2% overall greedy-boost tripwires with four length buckets, .5M/1M
checkpoint evaluation, and binding 1M stop plus shared arm KILL receipt.
Branch sampling retains128 candidates plus256 anchors and768 agent TD rows.
`level` remains the default, preserving legacy sampler calls and lazy grouping.

The focused non-slow suite passed **68 tests**, with three slow tests deselected:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ./venv/bin/python -m pytest -q -m 'not slow' \
  tests/test_t4f_contrast.py tests/test_t4f_early_warning.py \
  tests/test_t4f_integration.py tests/test_t4f_macro.py tests/test_t4d_return_bank.py \
  tests/test_t4e_calibration_telemetry.py tests/test_m3c_trainer.py \
  tests/test_perf_learner.py tests/test_t3_label_bank_trainer.py
```

Evidence: `t4f-validation-20261016/final-suite.log`. Three longer trainer tests also
passed in the earlier focused run; failures elsewhere in that earlier run were
fixed and covered by the final suite. That earlier aggregate run is not claimed green.
Fast/reference active-contrast parity, MPS loss behavior, full-support reference
selection, exact threshold boundaries, evaluator failure, shared stop, and macro
lifecycle are covered. Six modes ×eight updates reproduce baseline tensors, RNG,
and stable telemetry exactly (`default-trace-484f2e4-comparison.json`). Owned-file
flake8 and both lane shell syntax checks passed. No full-repository suite was run.

The full-bank evaluator smoke passed with identical B8_s0 initial/current inputs:
976 eligible decisive states,76 explicit exclusions,40,559 outside states, zero
outside-gap change. Synthetic telemetry makes this an evaluator test only, not
actual .5M or1M training evidence (`early-warning-smoke/full_bank_500k.json`).

## Collection

Task-owned `tmux t4f-collect` completed with exit0 and exited naturally. Twelve
single-thread CPU workers used **1,378.0596 seconds (22.97 minutes)**.

| Bank | Shards | States | Rows | Boost replaced | Non-boost unchanged |
|---|---:|---:|---:|---:|---:|
| Train |330|18,924|97,846|47,269|50,577|
| Holdout |30|1,052|5,418|2,622|2,796|
| Total |360|19,976|103,264|49,891|53,373|

Independent and root validators both passed every shard hash, source linkage,
world identity, and non-boost field byte comparison. Receipts are
`t4f-validation-20261016/macro-validation.json` and `root-macro-validation.json`.
The completed bank is `t4f-return-bank-20261016/banks/{train,holdout}`.

Mean full bootstrapped boost-minus-normal target gap, pairing within each state
then equally averaging states (Q units):

| Family | Source | Macro |
|---|---:|---:|
| Training decisive |+31.9855|-26.5725|
| Training outside |-0.1588|-61.3863|
| Held-out decisive |+30.6650|-25.2620|

`k=3` is the simulator boost cost period. The macro repeats the same relative
action for all three frames before B8 continuation through D300. Repeated turns
change geometry as well as burn cost; the large target changes do not isolate
boost's metabolic cost. Full target distributions, survival fractions, and source
checkpoint hashes are in `source-*-targets-final.json` and `macro-*-targets.json`.

The frozen outside evaluator set is `t4f-heldout-outside-v2-20261016`:40,559 unique
multi-legal observations,100 represented worlds of120 reserved holdout worlds,
no training observation overlap. The earlier non-v2 export is invalid and excluded.
The final seed audit passes for fresh rounds70/71 against all1,200 collection
plans and completed banks; documentary exclusions and an unfiltered audit are
preserved. Refresh the audit before future evaluation launch.

## Calibration and prepared execution

**W=0.00053**. Measured pooled initial ratios bracket the target: .107188 atW=.0001
and .906949 atW=.001. Log-log interpolation rounded to two significant figures
selected W; a separate confirmation pair measured pooled ratio **.595511**.
The [metadata clarification](t4f_calibration_clarification_2026-10-16.json) distinguishes
the frozen bracket interpolation from an unused diagnostic interpolation recomputed
over the confirmation-inclusive grid; the selected weight was never changed.

| Initialization | Initial ratio | Active fraction initial | Active fraction300 updates |
|---|---:|---:|---:|
| B8_s0 |.718737|69.2331%|76.0966%|
| B8_s2 |.493413|69.1676%|76.4530%|

Active fractions use sampled non-reference rows as denominator. All six MPS
smokes completed300 updates, each under its200k transition ceiling, with no halt
and actor exits[0,0]. Ratios drift after initialization; window statistics and maxQ
are retained in [the calibration receipt](t4f_calibration_2026-10-16.json).

[The preregistration](specs/t4f_prereg_2026-10-16.md) binds the amended recipe,
tripwires, early warnings, fresh rounds70/71, unchanged primary GO/KILL/EXTEND,
and final diagnostic readouts. `research/t4f_return_20261016/launch_inputs.json`
binds80 input/evidence files plus exact bank shard sets and hashes.

Unstarted artifact scripts:
- `redesign-t4f-mac-20261016/lane0.sh`: seeds10/12/14.
- `redesign-t4f-mac-20261016/lane1.sh`: seeds11/13.

Both `--check` calls passed. All five printed commands were independently parsed
for MPS,8M,weight,loss,tripwires,checkpoint frequency,early-warning config,and shared
stop wiring. Receipts: `t4f-validation-20261016/lane{0,1}-check.txt`,
`launch-binding-check.json`, and `launch-commands-validation.json`.
No 8M training was started; the launch directory contains only the two scripts.
No unrelated tmux session or process was manipulated.
