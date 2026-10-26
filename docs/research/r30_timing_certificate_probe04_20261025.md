# R30 probe-04 certificate amendment: prepared, dispatch blocked

The cap-only implementation and one fresh packet are prepared. **The required
`dispatch --dry-run` PASS was not achieved.** G5/main and R30-2 remain HOLD.
No Colab calls, live dispatch, scientific consumption, serving change or release
occurred. Independent packet ACCEPT remains required.

Worktree `/Users/josenunez/Projects/ml/snake-dqn-r30post`, branch
`sol-r30-post-f4-20261025`. Clean initial HEAD
`3a830e711c1851c1258912408d520ce1faf248ce`. Runtime source was committed, rebased
onto `124dd22c` (only an unrelated campaign-log addition), and pushed to main at
`0644e3a94e137fa6fb488f15f66290c6eec2726b`.

Astra's full ruling and sidecar were read and hash-checked. The exact immutable
copy is [TIMING_PROBE04_CERTIFICATE_AMENDMENT_20261025.md](../../research/bl_round30/TIMING_PROBE04_CERTIFICATE_AMENDMENT_20261025.md),
SHA256 `d21347475aba7b4e3be8e4e615b97d6a9f72d487d024a701aa81bf269d1bc837`.

The timing worker, config, report and forecast bind this authority and use the
1,200-second certificate cap, 2,100-second worker cutoff, 2,400-second session and
300-second cleanup/export reserve. The earlier remaining-session cutoff remains
binding. The manifest and reservation freeze certificate base/effective 600/1,200
seconds and session base/effective 20/40 minutes. The distinct, nontransferable
reservation preserves 70 minutes of prior calibration liability, reclaims none,
and raises the combined envelope from 502 to 542 minutes. R30-1/R30-2 allowances
remain 144/288 minutes; main estimate/timeout ceilings remain 53/106 minutes.

The shared solver exposes a certificate cap defaulting to 600 seconds. Only
prospective real R30-1 explicitly passes 1,200; synthetic R30-1 and both R30-2
certificate paths retain 600. Solver bounds, Decimal50 certificate implementation,
precision, objective and endpoint selection remain unchanged.

The full admitted input hashes, 2,000-batch plan, continuous 30 updates, closures
0/10/30, predetermined update-30 certificate, CPU threads=1/CUDA features, telemetry,
redaction, immutable publication, 64 MiB transport/512 KiB complete-output caps,
150/300-second transfer/staging bars, probe route, and no_automatic_retry remain
intact. Previous packets, reservations and failed attempts were preserved.

## Verification

Tests used nice 15, CPU, serial pytest (`-n 0`), and numerical pools set to 1:
`OMP_NUM_THREADS`, `OMP_THREAD_LIMIT`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`NUMBA_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`. No parallel test workers were used.

```bash
nice -n 15 ./venv/bin/python -m pytest -q -n 0 \
  tests/test_bl_round30_timing.py tests/test_bl_round30_timing_pack.py \
  tests/test_bl_round30_timing_forecast.py tests/test_bl_round30_timing_telemetry.py \
  tests/test_bl_round30_objective.py
# 81 passed in 11.51s

nice -n 15 ./venv/bin/python -m pytest -q -n 0 tests/test_ops_dispatch.py \
  -k 'timing_calibration or dry_run'
# 11 passed, 77 deselected in 6.83s
```

The timing tests execute the actual synthetic CPU CLI and exercise full-plan
scheduling, cap forwarding, production-versus-synthetic scope, redaction,
immutable marker-last publication, interruption/failure telemetry, and incomplete
forecast/ledger refusal. The dispatch tests exercise actual wrapper dry-run and
no-automatic-retry behavior using local fakes. The existing test for absence of a
40-minute qualified route confirms the blocker below.

Black (`--workers 1 --check --line-length 100`), isort (`--profile black
--check-only`), flake8 (`--jobs 1`) passed for all eight changed Python files;
`git diff --check` passed. Independent read-only review found no additional
implementation correctness defect. No live qualification or packet ACCEPT is
claimed. Qualified transport, dispatcher, qualification checker, telemetry,
low-level certificate, protocol, job entrypoint and shell launcher are unchanged.

## Admission blocker and real dry-run

`research/colab_cli/canary.py:213` rejects every family probe above 30 minutes.
`supervisor.validate` always calls that gate. The wrapper's short-canary
40-minute resume exemption does not override it. Modifying transport changes its
exact qualified tree, requiring new acceptance; no Colab calls are authorized by
this implementation request. Dropping probe mode, changing entrypoint, mutating
the qualification registry or suppressing validation would violate the requested
route or qualification contract. None was done.

The exact operator dry-run below was executed with `--dry-run`. Actual output:

```text
REFUSE: Command '[.../research/colab_cli/launch.sh, ..., validate, .../timing-calibration-probe-04/job.json]' returned non-zero exit status 1.
returncode=1
runtime_output_exists=False supervisor_state_exists=False
```

`DISPATCH_DRY_RUN.json` preserves the complete unabridged stderr and command.
Direct supervisor validation independently records the underlying reason:

```text
canary whole-session cap must be <=30 minutes
```

Source delivery and upload-size inspection passed for 14 files without bypassing
or claiming supervisor admission. `UPLOAD_SIZE_CHECK.json` explicitly records
admission REFUSED. Source packaging inspected only allowlisted source and opaque
input hashes; no scientific unit, D4096 or `t34-round0-study-01/` was opened.
No tmux session, released checkpoint, serving pin or old hash-bound evidence was
modified.

The remaining prerequisite is a prospective resolution of the qualified
30-minute probe limit versus this ruling's mandatory 40-minute session. This
packet is blocked and must not be dispatched until that conflict is resolved,
the real dry-run passes, and independent packet review ACCEPTS.

## Packet and operator handoff

Packet root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/bl-r30-postf4-20261025-v1/timing-calibration-probe-04`.
Job ID `r30-1-timing-calibration-probe-04-0644e3a9`.

| Payload | SHA256 |
|---|---|
| G5_TIMING_CERTIFICATE_PACKET.json | `b00ab0b9530ff49a9c4ba71273d219e6bc60e0cce9a543ad0417695d56bd2696` |
| job.json | `1847e79190fce311f3c00e8521fbd9f8f5f19408dc5ea955fe1274131e1f0123` |
| inputs/timing-config.json | `2e7eb31d826fdd8bb04f2bec36e9df488433b81f2f9743927fdcc5da9db926d8` |
| accounting-reservation.json | `aaa24092e1433747dfcb1eb9d0e29d16589d91dc6cee37e498f0e4f288c089f8` |
| VERIFICATION.json | `57ad670a5f0780b25f53960b8121a739fa77e19fdeef49cc7d5c7b13a3a81bd5` |
| DISPATCH_DRY_RUN.json | `e7358786b7efec951e0c445570317346c250c16c72b5ceeac2c5fe772364e775` |
| SUPERVISOR_VALIDATION.json | `a26314d84112726dbc9f42c5aefcb0d6a5a4c36788e675968fec260710ddc26e` |

Exact dispatch command for the operator after those prerequisites; **not executed**:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-r30post
PATH="$PWD/venv/bin:$PATH" SNAKE_DQN_PYTHON="$PWD/venv/bin/python" \
  tools/ops/dispatch.sh r30-timing-calibration-probe-04 \
  /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/bl-r30-postf4-20261025-v1/timing-calibration-probe-04/job.json \
  --broker /Users/josenunez/Projects/ml/snake-dqn-r30post \
  --cwd /Users/josenunez/Projects/ml/snake-dqn-r30post \
  --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/bl-r30-postf4-20261025-v1/timing-calibration-probe-04/supervisor-state
```

Append `--dry-run` for the offline check. No automatic retry or additional
calibration packet is authorized. A second timeout or completed forecast above
53 minutes closes this R30-1 route as operationally unresolved/not admitted,
never a scientific KILL or universal L4 infeasibility claim. F5 preparation may
continue under the ruling; this implementation starts no F5 work.
