# R30-1 timing calibration probe: offline handoff, 2026-10-25

Disposition: **RETURN FOR ASTRA ACCEPT; G5 HOLD. No live dispatch or Colab call.**
Implementation source pin `7b027d4a0f63e62ecde2242a44a62e25047c8c03` is published on origin/main.
Worktree `/Users/josenunez/Projects/ml/snake-dqn-r30post`, branch
`sol-r30-post-f4-20261025`; clean baseline was `185d83959bdc4be68f56a8fe9d036060fd12924d`.
This document records preparation and offline checks, not a calibration result.

The exact [prospective amendment](../../research/bl_round30/TIMING_CALIBRATION_PROBE_AMENDMENT_20261025.md)
is SHA256 `447609b306bde15de4014640e1248a4349f8cb2feb811e455511a66886e943ff`.
It is bound alongside the unchanged original ruling
`9b07b66663d8a7ac168f80dbfc35889b3551df8797f347335c10d83a8d81801f`.

| Ruling item | Implemented behavior |
|---|---|
| 1: cap and fresh packet | Base 15 / effective 30 minutes; worker maximum 1,500 seconds, also bounded by actual supervisor remaining runtime minus 300 seconds for export/cleanup. Forecast refuses supervisor time >=1,800 seconds. New packet/output/state paths. Full workload and 600-second certificate cap retained. |
| 2: calibration is the probe | Exact family `research/bl_round30/timing_colab.sh`; canary mode probe, instrumentation and package env enabled, CUDA/L4, required export timing.json, 15-minute estimate and 30-minute short-canary exemption. No preliminary canary or reused timing-family proof. |
| 3: genuine decision and redaction | One B8 masked six-action CUDA argmax on fixed admitted row zero; action/Q discarded. Its observer/forward overhead is timed separately and excluded from the main forecast. Strict timing metadata allowlist, existing operational observer metadata and suppressed numerical logs retained; synthetic sentinel/observer/terminal tests pass. |
| 4: enforced no retry | Boolean manifest opt-in refuses accounting-allocation retry with a nonzero exit before a retry manifest or second start. Triggering-failure test proves one start; absent/false legacy behavior is covered. Qualified subtree and registry unchanged. |
| 5: source binding and proof | 224 source hashes re-pinned to published source, tested dispatcher SHA bound, fresh qualification-tree/registry checks, 398 regression tests and 49 final targeted tests pass, offline dispatch dry-run passes, independent internal source/packet review has no findings. |

The full scientific schedule remains all admitted training rows, 2,000 fit batches
(seed 3403) plus the full audit plan (seed 3404), one continuous Adam optimizer for
updates 0–10/10–20/20–30, full closures at zero/10/30 and the original certificate.
Timeout or an incomplete phase means HOLD, with no reduced workload or retry.
No calibration numerical data were produced here.

Preserve the original reservation and full 40-minute liability until verified
stop/debit reconciliation. The unused 10-minute difference is nontransferable;
combined liability remains 472 pending reconciliation. Main ceiling remains
106 minutes and estimate ceiling 53 minutes. Frozen technical guards use L4
1.54 units/hour and 20% reserve (new manifest cap 0.77, reserve 0.154 units),
without rationing credits.

The qualified `research/colab_cli` tree is
`c148431277f4e33c3f589251df892ea158407e25`; registry SHA256 remains
`579e9638c23692352a5f0c0467a9abbe093cc820cc45cd52ec4f77900dbdd5bb`.
Dispatcher SHA256 is `4148c663a3451b90ffab76bd368cbb121a6fa64f493ad4cc2ee80ea30f77544e`.
No diff from baseline exists for the qualified subtree or scientific consumers
job.py/objective.py/certificate.py/colab.sh.

Offline verification: **398 passed in 579.13s**; after string wrapping and formatting,
**49 targeted passed in 10.19s**. Black --workers 1, isort, flake8 --jobs 1,
Bash syntax, real Bash canary-disabled refusal (exit 2 before Python), git diff
--check and unchanged-consumer diff checks pass. Tests were serial (-n 0),
launched with nice -n 15 and native thread pools set to one, using ./venv/bin/python.
The repository's macOS OpenMP isolation re-executed pytest; no xdist workers.
The complete commands, environment and logs are bound in VERIFICATION.json.

The regression command, with SNAKE_DQN_DEVICE=cpu and OMP_NUM_THREADS,
OMP_THREAD_LIMIT, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, NUMBA_NUM_THREADS and
VECLIB_MAXIMUM_THREADS each set to 1, was:

```bash
nice -n 15 ./venv/bin/python -m pytest -q -n 0 \
  tests/test_bl_round30_timing.py \
  tests/test_bl_round30_timing_pack.py \
  tests/test_bl_round30_timing_forecast.py \
  tests/test_ops_dispatch.py \
  tests/test_colab_cli_canary_observer.py \
  tests/test_colab_cli_canary.py \
  tests/test_colab_cli_supervisor.py \
  tests/test_bl_round30_admission.py \
  tests/test_bl_round30_batch_plan.py \
  tests/test_bl_round30_corpus.py \
  tests/test_bl_round30_factory_check.py \
  tests/test_bl_round30_job.py \
  tests/test_bl_round30_metrics.py \
  tests/test_bl_round30_objective.py \
  tests/test_bl_round30_pack.py \
  tests/test_bl_round30_protocol.py
```

The final targeted command used the same environment and prefix with only
`tests/test_bl_round30_timing.py tests/test_bl_round30_timing_pack.py tests/test_bl_round30_timing_forecast.py`.
Independent review verified all 224 source hashes, local input SHA/size bindings
(opaque bundle hash only), both rulings, old packet preservation, fresh unused
paths, qualified-tree identity and all reviewed export triples. Its disposition
is internal NO_FINDINGS_OFFLINE, not Astra ACCEPT or G5 acceptance.

Packet root: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/bl-r30-postf4-20261025-v1/timing-calibration-probe-01`. Every listed payload has SHA and completion sidecars.

| File | SHA256 |
|---|---|
| [G5_TIMING_CALIBRATION_PROBE_PACKET.json](/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/bl-r30-postf4-20261025-v1/timing-calibration-probe-01/G5_TIMING_CALIBRATION_PROBE_PACKET.json) | `a84ed706fdf75cd9e8f1693b8d1fc6cafd572e802b3f4bb4a6ab0c3293d87a28` |
| [job.json](/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/bl-r30-postf4-20261025-v1/timing-calibration-probe-01/job.json) | `fba128659a1d63a91f1a35d81ae67c1b2f5df30cdb1df6bfc220e2ab76048b91` |
| [inputs/timing-config.json](/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/bl-r30-postf4-20261025-v1/timing-calibration-probe-01/inputs/timing-config.json) | `4867b68c9973f91782728fc525e0a8bdd10b00301a99c3df6d499a2c8b1c3427` |
| [DISPATCH_DRY_RUN.json](/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/bl-r30-postf4-20261025-v1/timing-calibration-probe-01/DISPATCH_DRY_RUN.json) | `90950276594aaea39b26379fbb6dc9b672d78fdaf73b69553182e16173e9c13e` |
| [SOURCE_BINDINGS.json](/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/bl-r30-postf4-20261025-v1/timing-calibration-probe-01/SOURCE_BINDINGS.json) | `f62925f94033590ca0383c0cb57aa75498d63f417349d02c793ec9c7d4846821` |
| [VERIFICATION.json](/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/bl-r30-postf4-20261025-v1/timing-calibration-probe-01/VERIFICATION.json) | `fe3eb49a5b24bf0a70a74f5f8230ab5f6c4b7a2c69ec0e5137ef940fb072ab73` |
| [INDEPENDENT_REVIEW.json](/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/bl-r30-postf4-20261025-v1/timing-calibration-probe-01/INDEPENDENT_REVIEW.json) | `31a26df66f29b2736a9b5dd650087fb917c04689bbf3e9f1af9d87a87f4cdf96` |
| [accounting-reservation.json](/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/bl-r30-postf4-20261025-v1/timing-calibration-probe-01/accounting-reservation.json) | `cb07e346fa1588caf27d261589319ced6829b61466962bcefd6c092b2d804b7c` |
| [OPERATOR_COMMANDS.txt](/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/bl-r30-postf4-20261025-v1/timing-calibration-probe-01/OPERATOR_COMMANDS.txt) | `e264baab79a5304e305c06593e7ef97f4d858da1b537bae949c84125fd6135b0` |

Old packet remains `db548467bb267aa952107419f82aec97ac01d945bc11e5f9f92094f130af24d1`;
old manifest remains `9f946c6f2d55ba1bbe27f9f12b6e666e60bc0077ac492fe6c679e9f947ec5009`;
old reservation remains `621fc26028b93fc5be0fe933f642ebaac8ef79ab48277129e03091e041aad6c6`.

The exact offline command below was executed with --dry-run only:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-r30post
PATH="$PWD/venv/bin:$PATH" tools/ops/dispatch.sh r30-timing-calibration-probe-01 \
  /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/bl-r30-postf4-20261025-v1/timing-calibration-probe-01/job.json \
  --broker "$PWD" --cwd "$PWD" \
  --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/bl-r30-postf4-20261025-v1/timing-calibration-probe-01/supervisor-state --dry-run
```

```text
PASS_DRY_RUN r30-timing-calibration-probe-01 allocations=0 live_accounting=NOT_CHECKED
```

Operator command after Astra ACCEPT; **not executed**:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-r30post
PATH="$PWD/venv/bin:$PATH" tools/ops/dispatch.sh r30-timing-calibration-probe-01 \
  /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/bl-r30-postf4-20261025-v1/timing-calibration-probe-01/job.json \
  --broker "$PWD" --cwd "$PWD" \
  --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/bl-r30-postf4-20261025-v1/timing-calibration-probe-01/supervisor-state
```

No current live account check, CUDA probe result, terminal receipt or verified
runtime disappearance exists from this preparation. Keep account cap three,
lease/watchdog, transfer <=150 seconds, staging <=300 seconds and verified
release requirements. Main, G4 rebinding and serving/release remain separately
gated. No scientific units, D4096 or protected study were opened. No tmux
sessions, released checkpoints, serving pins, original hash-bound packets,
qualified transport source or qualification registry were modified.
