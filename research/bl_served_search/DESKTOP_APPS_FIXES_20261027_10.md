# Desktop collector/reducer repair and attempt-10 handoff

Repairs Argon's rejection of `3c286e44`. This is implementation evidence;
no host-only observation or measured panel was run. Wait for the live A100
supervisor to complete its normal export/cleanup before the operator observation
and panel. The short collector test and point admission do not qualify timing
or establish future host quietness. Independent code review found no remaining
actionable issue in the repaired diff; this does not stand in for Argon acceptance.

The existing sealed base and `_02` extension are unchanged. There is no enrollment
change and no new registry. No serving code, pins, checkpoints, tmux sessions,
running jobs, `research/colab_cli`, or `exp/mac-lease.json*` were modified.

## Repairs

- Reducers read desktop identity from the collector's nested `classification`.
- The off-tick interval collector admits current identities before classifying
  their CPU deltas. New/exec/UID changes retain churn and incomplete coverage.
- Verified Claude/Codex GUI mains use the same exact-executable exception live
  and on replay. CLI processes and unverified GUI copies remain workloads.
- Interval transitions export both identities and conservative precedence.
  Revocations remove active bindings and preserve historical admissions so old
  samples still replay; changed identities reach the explicit contamination guard.
- Desktop host-only CLI accepts 120–450 seconds without clamping. Observation
  binding requires at least 120 seconds. Invalid host/monitor/reducer outcomes
  produce INVALID observation receipts without reducing an invalid host export.

## Verification

All runtime checks used CPU, five math thread caps of 1, and `nice -n 15`.
The launcher inherited niceness 20, satisfying its >=15 requirement.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-served
export SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q tests/test_served_background_report.py tests/test_served_desktop_apps.py tests/test_served_host_load.py tests/test_served_probe_host.py tests/test_served_tick_report.py tests/test_served_tick_probe.py
nice -n 15 env SERVED_DESKTOP_INTEGRATION=1 SERVED_DESKTOP_INTEGRATION_ARTIFACT_ROOT=/Users/josenunez/Projects/ml/snake-dqn-artifacts/served-tick-20261026 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -s tests/test_served*.py
nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m black --check --line-length 100 research/bl_served_search tests/test_served*.py
nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m isort --check-only --profile black research/bl_served_search tests/test_served*.py
nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m flake8 research/bl_served_search tests/test_served*.py
git diff --check
/Users/josenunez/Projects/ml/snake-dqn-served/research/bl_served_search/run_tick_probe.sh --desktop-app-registry /Users/josenunez/Projects/ml/snake-dqn-served/research/bl_served_search/DESKTOP_APP_REGISTRY_20261027_02.json --smoke --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/served-tick-20261026/smoke-desktop-10
/Users/josenunez/Projects/ml/snake-dqn-served/research/bl_served_search/run_tick_probe.sh --desktop-app-registry /Users/josenunez/Projects/ml/snake-dqn-served/research/bl_served_search/DESKTOP_APP_REGISTRY_20261027_02.json --check
```

Results: focused **197 passed**; full served **438 passed** in 16.38 seconds;
Black/isort/Flake8 and diff checks passed. Smoke was `SMOKE_OK_NOT_TIMING`, two
measured ticks per arm, state/DTO/actions parity passed with no mismatch. Real
`--check` returned 0, `PREFLIGHT_OK_NOT_TIMING`, `admitted: true`.

The unmocked integration test uses real signature/digest verification, point
inventory, and three real cumulative CPU intervals. It seals `host.json`, reads
it back, and requires `_samples`, `desktop_overall`, `validate_desktop_host`, and
`summarize_background` to replay without classification/schema failures. A
synthetic trace only aligns the captured intervals; it supplies no gameplay or
timing claim. Live contamination cannot hide replay failures. Real capture is
explicitly enabled with `SERVED_DESKTOP_INTEGRATION=1` on the enrolled Mac.

Full-suite retained evidence:

- Directory: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/served-tick-20261026/served-desktop-integration-ao9crs8h/`
- `host.json` SHA256: `abd2be7f0e417a303131a4f1fae3a095ca7d0e7afe307fd729dba53700db40a8`
- Three intervals; admitted Claude, Codex desktop, Gemini, Chrome and Outlook.
- No point violations, interval guard rejections, or reducer errors.
- `host.json`, `reducer-results.json`, and `receipt.json` retain immutable hash
  and completion sidecars. Earlier standalone capture is retained in sibling
  `served-desktop-integration-6so_cidv/`.

## Operator commands (prepared only)

The launcher sets CPU, all five caps at 1, caffeinate, and `nice -n 15`.
Use a naturally supervisor-free window and fresh output directories. The
observation must complete successfully under exactly the same source/enrollment
as the subsequent panel. Do not reuse an invalid receipt or partially written
directory. Independent acceptance and the successful observation remain
prerequisites for panel execution.

```bash
/Users/josenunez/Projects/ml/snake-dqn-served/research/bl_served_search/run_tick_probe.sh --desktop-app-registry /Users/josenunez/Projects/ml/snake-dqn-served/research/bl_served_search/DESKTOP_APP_REGISTRY_20261027_02.json --check
/Users/josenunez/Projects/ml/snake-dqn-served/research/bl_served_search/run_tick_probe.sh --desktop-app-registry /Users/josenunez/Projects/ml/snake-dqn-served/research/bl_served_search/DESKTOP_APP_REGISTRY_20261027_02.json --host-only-seconds 120 --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/served-tick-20261026/host-desktop-10
/Users/josenunez/Projects/ml/snake-dqn-served/research/bl_served_search/run_tick_probe.sh --desktop-app-registry /Users/josenunez/Projects/ml/snake-dqn-served/research/bl_served_search/DESKTOP_APP_REGISTRY_20261027_02.json --desktop-observation-receipt /Users/josenunez/Projects/ml/snake-dqn-artifacts/served-tick-20261026/host-desktop-10/observation.json --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/served-tick-20261026/panel-desktop-10
```
