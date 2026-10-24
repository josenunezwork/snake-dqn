# O1 offline implementation verification — 2026-10-23

Status: **offline verification passed; native acceptance NOT RUN**. This is a
prepared implementation record, not transport qualification or adoption.

The final serial run passed **86 tests in 100.90 seconds**. Its exact command
and full stdout/JUnit evidence are create-only artifacts under:

`/Users/josenunez/Projects/ml/snake-dqn-artifacts/a100-retention-20261023/verification-03/`

```sh
bash /Users/josenunez/Projects/ml/snake-dqn-artifacts/a100-retention-20261023/verification-03/run.sh
```

The script uses the shared project interpreter, `nice -n 15`, CPU/thread limits
of one, and `-p no:xdist`. It selects the four new test modules, the corrected
dispatch warmup fixture, all source-delivery tests, normal/H100/fallback broker
regressions, and a distinct executed test for every required fault. Independent
Codex-V fixtures cover all eight changed cache pins, real sparse archive
installation, Git source identity, generation and inventory refusals, sealed T0
epoch/ISO values, and absolute-deadline reservations.

The earlier broad transport/ops run selected 714 tests and finished **710 passed,
4 failed in 1653.65 seconds**. Its log/JUnit remain immutable in
`verification-01/`. The warmup fixture needed to permit the wrapper's read-only
account snapshot and normalize its argument format for the fake CLI; its guarded
session/upload assertions remain intact and it passes in the final run.

Three baseline failures remain outside this owned change:

- `test_ops_calendar_lint.py::test_repository_launchers_contain_no_calendar_windows`:
  `research/compute/factory_interim_launcher.py:1` contains a literal date.
- Both `test_t34_transport_qualification.py::test_saved_manifest_is_rechecked_by_shared_broker_before_allocation`
  variants construct manifests without explicit device/export/canary admission
  fields while expecting validation to pass.

The involved product guards, launcher and fixtures are unchanged from both
implementation baseline `d063bfe1` and the then-current `origin/main`.
Independent review confirmed this static baseline comparison. Those files were
not edited and the product guards were not weakened. This does not claim a
separately executed baseline suite or a completely green repository-wide gate.
The intermediate `verification-02/` run and its two corrected fixture failures
are also preserved.

Black (100 columns), isort (Black profile), and flake8 passed for all 14 changed
Python files. `bash -n` passed for bootstrap, native probe, acceptance launcher
and the existing dispatch wrapper; `git diff --check` passed.

The real `retention_acceptance_probe.sh --cpu-smoke` entrypoint ran with frozen
input plus an empty ledger, one worker and the manifest's output/input/root
bindings. Three measured units took **0.000918292, 0.000061042, 0.000038333 seconds**.
The real runtime scanner accepted both immutable exports. Input digest:
`6508de77a1a08d1333b57d8d3014015c267c73cf1f1ca608b70553eca38331b1`;
ledger digest:
`5c4b7bd72b37dbdeff1a450867e5e03284c8eef631d2566cd440a63ebefa70ac`.
Full evidence is in `verification-03/entrypoint-smoke/smoke-report.json`.
These are CPU entrypoint timings, not native CUDA estimates or measured savings.
The registered O1 forecast remains 2 A100 hours.

The existing external broker still matches the unchanged qualification registry:

```sh
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python tools/ops/qualification.py check \
  --repo /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-33/qualified-broker-01.noindex \
  --record /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/colab-cli-qualified.json
```

It resolves transport tree `d0d92bdb572b770284a1fea6152417ffa5f06bea`.
The candidate's final published commit and SHA-bound offline fault review belong
in the freshly prepared acceptance package. Native compile/timing/fault/release
evidence and independent adoption review remain pending; no acceptance dispatch,
registry update, live-session operation, serving change or scientific-owner edit
was performed.
