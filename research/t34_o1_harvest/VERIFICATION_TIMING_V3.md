# O1 timing V3 recovery verification — 2026-10-21 Pacific

Outcome: excluded timing fixture repaired and Mac preflight COMPLETE; replacement
L4 manifest locally validated and NOT dispatched. No study world was played in
this recovery. This supplies no KILL/GO, L4 admission, parity or serving result.
See `TIMING_FIXTURE_V3.md` for the diagnosis and prospective fixture correction.
Earlier verification and manifest describe the preserved previous attempt.

## Published identities and preserved scope

- Code: `0eeff0ab` (Fix O1 excluded timing roots and align Mac preflight).
- Source pin after main integration: `eba4195c319cef8b8b71494fd1c2d7b4688a88c0`.
- V3 registration published before the replacement Mac run:
  `c02e5c8054b852caf593720e9b7b5167f4fff377`.
- `REGISTRATION_O1_TIMING_V3.json` SHA-256:
  `a82691cb73fd90b7287c621b685d336faa59070b01224afc5e415b780f02bb12`.

Compared original bank, preregistration, both audits and V1/V2 registrations to
baseline `d1876abda451772cdec1210e50bab70ca882c3d3`: byte-identical. Every existing
non-source registration field equals V2; V3 adds only the excluded timing spec.
Study collection, roots/reservoir, ranking, attribution, decisions and caps are
unchanged. Changes are confined to the O1 package and its documentation. No edits
to Round 0, Path B, ExIt-B8 or colab_cli; the sealed Round 0 artifact directory
was never opened. Released checkpoints/pins and existing tmux jobs were untouched.

Commit-format deviation: the already-published automatic integration merge
`eba4195c` lacks the requested co-author trailer. The implementation, registration
and subsequent recovery commits carry it. Published history was preserved.

## Targeted verification

Run with nice -n 10 and numerical thread env values 1, CPU:

```bash
nice -n 10 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q \
  research/t34_o1_harvest \
  tests/test_colab_cli_supervisor.py::test_existing_output_directory_is_refused_as_nonfresh \
  tests/test_t34_transport_qualification.py::test_record_matches_actual_accepted_source
```

**40 passed in 3.29 seconds**, including exact fixture selection at worker counts 1
and 11, early terminal death with no events, missing/late/event-root rejection,
stop before study collection, verified preflight hashes and forecast
recomputation, and runtime export discovery with required sidecars. Initial run
before the forecast-mutation regression: 39 passed in 3.42 seconds. No broad suite.

Passed: `python -m black --check --line-length 100 research/t34_o1_harvest`,
`python -m isort --check-only --profile black research/t34_o1_harvest`,
`python -m flake8 research/t34_o1_harvest`,
`bash -n research/t34_o1_harvest/launch.sh`, `git diff --check`.
Read-only independent review found no blocker and checked the failed archive,
study invariants, shared timing validator, sizing and sidecar tests.

## Exact L4-world Mac entrypoint preflight

Executed from `/Users/josenunez/Projects/ml/snake-dqn-o1`:

```bash
/usr/bin/time -p /usr/bin/caffeinate -is nice -n 10 env \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
  T34_O1_PREFLIGHT=1 T34_O1_WORKERS=1 \
  T34_O1_PYTHON=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
  T34_O1_REGISTRATION=/Users/josenunez/Projects/ml/snake-dqn-o1/research/t34_o1_harvest/REGISTRATION_O1_TIMING_V3.json \
  T34_O1_CHECKPOINT=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth \
  T34_O1_OUTPUT_ROOT=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-o1-20261021/preflight/run-03 \
  bash research/t34_o1_harvest/launch.sh \
  > /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-o1-20261021/preflight/run-03.log 2>&1
```

COMPLETE, exit 0, **71.64s wall**, runner **71.2421727089677s** (local cap 180s).
macOS 27.2 arm64, actual niceness 15 (inherited 5 plus nice -n 10), one thread for each
numerical library, CPU only. AC attached, lid open, no reported thermal/performance
warning. Host load was about 20 with ongoing work; these are contended operational
measurements, never quiet-host serving latency or Mac/x86 identity proof.
Existing sessions observed and untouched: `colab-limit-test`, `git-sync`,
`phase0-selection`, `t34-claima-d0r`.

| Excluded world | Seed | Base transitions | Fixed roots | Branches | Branch steps |
| --- | ---: | ---: | --- | ---: | ---: |
| timing/frozen/0000 | 943080529 | 4479 | 10,20 | 6 | 3000 |
| timing/mixed/0000 | 2671445542 | 2216 | 10,20 | 9 | 4500 |
| Total | | 6695 | 4 roots | 15 | 7500 |

Same world rows, rosters, H5000 maximum base trajectory and H500 branches as the
replacement L4 fixture. Both terminal base lengths reproduced the failed L4
payloads; all four source-preservation checks and accounting checks passed;
events 0. Full runtime observer parity remains unrun (`parity_pass:false` and
`timing_admitted:false` correctly remain in the local preflight receipt).

Receipt: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-o1-20261021/preflight/run-03/receipt.json`.
SHA-256: `ba1d064c8b77843fa10b73afcb5a242306d99fcb4dbb36f2d8d595c2c15524da`.
Payload SHA-256 values:

- `world-000.json`: `c5d1989b66b368ab63497e3c923a1808b95a4667c1cefbf465f92eff3f6d80ba`.
- `world-001.json`: `f4eb9de803e8d4e49be08a180cb3d30677ddab58550dbb1dbc0a419af71821ae`.

Real export CLI also passed: `python -m research.t34_o1_harvest.export --run-dir
<run-03> --out <preflight/exports-v3/o1-preflight.tar.gz>`. Bundle 347614 bytes,
SHA-256 `869eac041255123462d6ae0d84790f35df07fdc5785568c0817adad1779dba37`,
with matching `.sha256` and `.complete` sidecars. All previous outputs preserved.

## Forecast and prepared manifest

Worst measured recurring base step 0.00707878184072942s; branch row-step
0.00437825308192987s; setup/world 1.223751708283089s. Existing five study worker
waves plus parity wave at 11 workers, upper-bound branch work, 2x caution:

- Remaining work 1560.9804890136256s.
- Including fixed 1200s setup/export: **2760.9804890136256s = 46.01634148356043min**.
- At 2 u/h: **1.5338780494520143u**; rounded manifest estimate **47min**.
- This is extrapolation from Mac, not measured L4 throughput. Early timing roots
  do not prove late-state cost equivalence. Live excluded L4 timing and three
  parity worlds still gate study admission inside the remaining ordinary cap.
- Historical failed attempt receipt elapsed 485.825224375003s; pricing its full
  interval at the 2 u/h ceiling gives 0.2699029024305572u. Combined elapsed-ceiling
  forecast is 3246.8057133886286s / 54.11342855647714min / 1.8037809518825714u.
  This is conservative planning arithmetic, not measured provider billing.

`MANIFEST_L4_CENSUS_02.json` SHA-256:
`f65b904f4111a0b40a948444518143ba1a93d434d201265122327376e4cf8e02`.
Source pin `eba4195c319cef8b8b71494fd1c2d7b4688a88c0`; three hash-bound local
uploads: V3 registration, public-input catalog SHA
`736f7b4e7730f760b8b81db00772d14abcf84c74b261a8dc7630732fdd383a10`,
and preflight receipt. The original `MANIFEST_L4.json` remains unchanged.

Fresh output directory (not created):
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-o1-20261021/colab-census-02`.
One L4, min 12 vCPUs, 11 one-thread CPU workers, 240 min timeout, 8 u cap at 2 u/h,
1.6 u reserve, 192 min ordinary allowance; timing ≤30 min / 1 u including charged setup;
lease 600 s; export ≤64 MiB. Registered caps are unchanged.

The actual `make_manifest` CLI succeeded, checking qualified executable
inventories and pinned source hashes, then running local supervisor `validate`
and source `prepare`. These operations call no Colab executable, download no
inputs, and allocate nothing. Reproduce with a fresh `--out` filename:

```bash
nice -n 10 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
  -m research.t34_o1_harvest.make_manifest \
  --registration /Users/josenunez/Projects/ml/snake-dqn-o1/research/t34_o1_harvest/REGISTRATION_O1_TIMING_V3.json \
  --catalog /Users/josenunez/Projects/ml/snake-dqn-o1/research/t34_exit_b8/PUBLIC_INPUTS.json \
  --preflight-receipt /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-o1-20261021/preflight/run-03/receipt.json \
  --commit eba4195c319cef8b8b71494fd1c2d7b4688a88c0 \
  --output-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-o1-20261021/colab-census-02 \
  --out /absolute/fresh/manifest.json
```

No Colab calls or dispatch. Account assignments/rates have not been refreshed;
before any future dispatch the operator must supply the current inventory and
positive reviewed rates under `budget.other_assignments`, and charge the prior
failed attempt against the study envelope. The manifest retains the registered
per-run transport caps; it does not mechanically deduct earlier consumption. No automatic allocation/job retry,
study analysis, strict gate, serving change or release was performed.
