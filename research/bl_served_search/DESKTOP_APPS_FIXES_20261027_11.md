# Desktop transition repair and attempt-11 handoff

Repairs the three Argon findings against `26b6e889`. This note supersedes the
attempt-10 implementation description and operator commands, without changing
sealed enrollment, registrations, past receipts, or timing policy. No host-only
observation or panel was executed. Independent code review and short integration
evidence do not constitute Argon acceptance or timing qualification.

## State machine and replay

The registry explicitly distinguishes unbound, bound, changed, and revoked
identities. An unbound verified PID/birth may be admitted; an unchanged binding
may be reused. Failure of a bound identity or an observed command/UID transition
revokes it for the rest of the registry attempt. Revocation is terminal even for
transitions between two enrolled applications or transitions that were never
admitted. A different birth denotes a new process identity. Every classification
and admission failure uses the common invalidation path. First revocation reasons
and historical admissions remain available; repeated replacement checks retain
`executable_replaced` and `identity_binding_changed: true`.

The collector captures the previous classification exactly once against the
binding state before applying current-row transitions. It admits only new or
unchanged current rows, then classifies current evidence. Workload precedence is
conservative across both identities; otherwise an identity change is unknown,
including system executables. Reducer replay applies the same precedence and
uses historical bindings to verify earlier admitted rows.

Tests cover all 36 ordered pairs, including unchanged controls, among invalid,
desktop app, workload, UID mismatch, system service, and another enrolled app.
They exercise the actual collector and reducer replay, host rejection, repeated
admission, and restoration after revocation. `/usr/bin/true` has separate
regression coverage: it is outside the existing system roots, which stay
unchanged. The real-Mac integration test is retained and explicitly enabled.

## Verification

Final validation: focused **234 passed** in 2.19 seconds; full served **475
passed** in 15.72 seconds, with the unmocked real-Mac test enabled. Black, isort,
Flake8, `make lint`, and `git diff --check` passed. Independent read-only production
review found no remaining actionable findings.

Smoke returned `SMOKE_OK_NOT_TIMING`, with state/DTO/actions parity and no
mismatches. Final smoke artifacts are in
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/served-tick-20261026/smoke-desktop-11-final/`.
The real `--check` returned 0, `PREFLIGHT_OK_NOT_TIMING`, `admitted: true`.
Runtime checks use CPU, all five math thread caps at 1, serial pytest, and
`nice -n 15`. The launcher sets these caps and niceness itself.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-served
export SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -n 0 tests/test_served_background_report.py tests/test_served_desktop_apps.py tests/test_served_host_load.py tests/test_served_probe_host.py tests/test_served_tick_report.py tests/test_served_tick_probe.py
nice -n 15 env SERVED_DESKTOP_INTEGRATION=1 SERVED_DESKTOP_INTEGRATION_ARTIFACT_ROOT=/Users/josenunez/Projects/ml/snake-dqn-artifacts/served-tick-20261026 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -s -n 0 tests/test_served*.py
nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m black --check --workers 1 --line-length 100 research/bl_served_search tests/test_served*.py
nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m isort --check-only --profile black research/bl_served_search tests/test_served*.py
nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m flake8 research/bl_served_search tests/test_served*.py
nice -n 15 make PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python lint
git diff --check
/Users/josenunez/Projects/ml/snake-dqn-served/research/bl_served_search/run_tick_probe.sh --desktop-app-registry /Users/josenunez/Projects/ml/snake-dqn-served/research/bl_served_search/DESKTOP_APP_REGISTRY_20261027_02.json --smoke --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/served-tick-20261026/smoke-desktop-11-final
/Users/josenunez/Projects/ml/snake-dqn-served/research/bl_served_search/run_tick_probe.sh --desktop-app-registry /Users/josenunez/Projects/ml/snake-dqn-served/research/bl_served_search/DESKTOP_APP_REGISTRY_20261027_02.json --check
```

Full-suite retained unmocked evidence:

- Directory: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/served-tick-20261026/served-desktop-integration-2fnsbw0t/`.
- `host.json` SHA256: `c67b586c9539084beacbb04157995e105e3b095f9e3ef69f721864b2e4d11513`.
- Three actual CPU intervals; admitted Claude, Codex desktop, Gemini, Google
  Chrome, and Microsoft Outlook. Zero point violations, guard rejections, and
  reducer errors. Immutable raw export, reducer results, receipt, and sidecars
  remain outside Git.
- The alignment trace is synthetic; this is collector/reducer integration
  evidence, not a timing observation or gameplay qualification.

## Prepared operator commands

Run only after independent acceptance, in a naturally suitable host window.
The successful observation and subsequent panel must bind the same final source
and registry. All output directories are create-only; preserve invalid attempts.

```bash
/Users/josenunez/Projects/ml/snake-dqn-served/research/bl_served_search/run_tick_probe.sh --desktop-app-registry /Users/josenunez/Projects/ml/snake-dqn-served/research/bl_served_search/DESKTOP_APP_REGISTRY_20261027_02.json --check
/Users/josenunez/Projects/ml/snake-dqn-served/research/bl_served_search/run_tick_probe.sh --desktop-app-registry /Users/josenunez/Projects/ml/snake-dqn-served/research/bl_served_search/DESKTOP_APP_REGISTRY_20261027_02.json --host-only-seconds 120 --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/served-tick-20261026/host-desktop-11
/Users/josenunez/Projects/ml/snake-dqn-served/research/bl_served_search/run_tick_probe.sh --desktop-app-registry /Users/josenunez/Projects/ml/snake-dqn-served/research/bl_served_search/DESKTOP_APP_REGISTRY_20261027_02.json --desktop-observation-receipt /Users/josenunez/Projects/ml/snake-dqn-artifacts/served-tick-20261026/host-desktop-11/observation.json --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/served-tick-20261026/panel-desktop-11
```

No serving code, pins, checkpoints, tmux sessions, running jobs,
`research/colab_cli`, or `exp/mac-lease.json*` were modified.
