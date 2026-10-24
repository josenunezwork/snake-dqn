# F0 nice-15 interim: mechanical recovery and launcher repair

Binding authority: `ALLOW_NICE15_INTERIM`,
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/JUDGMENT_FACTORY_CONCURRENCY_INTERIM_20261023.md`,
SHA256 `ea5ae541efc434fe3e5888eeab7a51a821220ce242f935795eacd0542af1accd`.
This is a separate interim policy. The full companion supervisor remains
PREPARED_NOT_QUALIFIED / activation HOLD.

The failed independent review of operational pin
`4b46454d509c4b9f40166437d6aa952013fa45e2` is retained unchanged in
`CODEX_V_FACTORY_INTERIM_20261023.json`. Its earlier nice-only review did not
establish compliance with this ruling. The mechanical repair requires a new
independent focused verdict, `CODEX_V_FACTORY_INTERIM_02_20261023.json`; that
verdict and the campaign-log entry identify the exact reviewed operational pin.
Only a PASS of that pin permits the existing operator continuation.

Recovery publishes `bl-labels-f0-slowdown-host-check-nice-interim/v2` with the
explicit interim policy, ruling SHA, and actual operational checkout HEAD.
`F0_COMMIT` still identifies frozen science and cannot supply the operational
identity. Recovery accepts only matching schema/policy/ruling/pin observations
for the required trip. Legacy-unbound and foreign receipts cannot authorize
recovery; prior immutable receipts stay in place. The receipt truthfully records
`no_disallowed_competing_compute` and `serial_test_lane_stopped`; it never emits
`no_sol_codex`, monitor health or companion CPU-budget claims. The common test
lock is held through host-health, process-priority and top-consumer observations
and create-only publication. Existing AC/lid/thermal/disk guards must pass.

The startup/runtime process-presence exception is unchanged: relevant foreign
processes at observed nice >=15 may coexist, and nice <15 or unknown priority
fails closed. Descendants are checked individually. Qualifying foreign tasks
are not adopted as owned workers and receive no accounting discount. Factory
workers still enforce nice 10. Health, timing, slowdown, deadline and all other
registered stops remain binding.

`tools/ops/agent_nice.sh` provides separate light-client and reviewed serial-test
routes. Both force CPU-only execution, all six numerical thread variables to
one and effective nice >=15 before imports; lower inherited priority is kept.
The numerical test route uses the host-wide
`ops/factory-companions/test-slot.lock`, a maximum elapsed deadline of 120 s
including owned process cleanup, reviewed exact commands, and Torch intra/inter-op
limits of one. Timeout or leftover descendants is incomplete, never PASS. There
is no automatic repeat or splitting into continuous compute. See launcher usage
for the hash-bound review manifest interface.

For conservative protected-window enforcement, the serial test route also holds
the existing exclusive Mac lease through cleanup. It refuses tests while any
factory/timing/calibration lease is active, including first-hour baseline,
fresh-window rearming and W12 throughput sanity. Advice/editing can continue via
the light route. Timing arms refuse/preempt numerical tests. This restriction
does not activate or depend on the unqualified full supervisor. No xdist,
process pools, detached children, GPU/MPS, whole-suite runs, training, evaluation
or substantial builds are admitted as short serial checks.

Science remains in the clean detached tree at
`61900af28932e47e22e332840fc4406559adb062`. The operational D256 continuation,
immutable exports, slowdown guard/state, W12 runtime and frozen science loader
retain their qualified bytes. Namespace, allocation, verified-unit skips,
baselines, attempts, cumulative wall and worker charges, caps and registered
stops are unchanged. No factory or long job is launched by this repair/review.

After independent PASS and publication, use the exact operational pin recorded
in `CODEX_V_FACTORY_INTERIM_02_20261023.json`. Check that `-03` is unused (otherwise
use fresh `-04`); the operator command is:

```bash
/Users/josenunez/Projects/ml/snake-dqn-f0/tools/ops/dispatch.sh --mac \
  --repo /Users/josenunez/Projects/ml/snake-dqn-f0 \
  bl-f0-d256-guard-w12-03 REVIEWED_OPERATIONAL_PIN -- \
  /bin/bash research/bl_labels/RUN_D256_RESUME.sh --host-check \
  --calibration-receipt /Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/w12-admission-01/receipt.json
```

This invokes the existing lawful fresh-attempt union-ledger continuation; it
neither resets the baseline/accounting nor recomputes verified labels. The final
operator message supplies the resolved pin and dispatch name. Release/serving
authority remains unchanged.
