# Amendment 03 validation and delivery

Source/contract freeze: `f92b7d718f1415aeebb7856d9d3f1649543ace7a`, published to
origin/main before the exported preflight. Independent review PASS is recorded in
REVIEW_AMENDMENT_03.md. DELIVERY_AMENDMENT_03.json binds the artifact paths and hashes.

All Mac execution used nice -n 10, CPU, numerical/Numba threads=1, and at most two
single-thread workers. No Colab operations or live-job/tmux changes were performed.
The protected T34 directories, checkpoint bytes, pins and sealed artifacts were untouched.

Targeted verification (project interpreter `/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python`):

- `-m pytest -q research/bl_density`: **45 passed in 1.51s**.
- `-m pytest -q research/bl_density tests/test_colab_cli_supervisor.py tests/test_colab_cli_source.py tests/test_colab_cli_transfers.py`:
  initial combined run **133 passed, 1 failed in 401.14s**. Failure was a forecast test
  expected arithmetic value (28.655833 vs correct 28.725), corrected before source freeze.
  All **95** selected supervisor/source/transfer cases passed. The subsequent density-only
  final run above passed every affected package test; unchanged transport tests not repeated.
- Black check at line-length100, isort check profile black, flake8 over research/bl_density,
  `bash -n research/bl_density/colab.sh` and `git diff --check`: PASS.
- Both manifest builders call the real unchanged supervisor validator; final saved manifest
  payloads revalidated, sidecars verified and fresh output directories confirmed.
- Z-9 `qualify` used the real pinned transport source builder, verified qualification inventory
  and exact sparse fallback archive. Both receipts: 1,163,885 compressed archive bytes,
  65.408409s normal upload forecast, 93.572073s fallback-upload forecast plus120s fetch
  =213.572073s <=300s. These are engineering forecasts, not live transfer measurements.
- Real pinned archive installed using `source.install_fallback`; its file inventory/hashes
  verified. Actual `colab.sh` ran with manifest environment, real public checkpoint catalog
  materialization and real export sidecars:12/12 episodes,13 payloads,12.357s runner wall.
  Git-less sparse document checks use the already hash-bound uploaded source; expected
  `git show` failures are present in its log and the entrypoint exited0.
- `-m research.bl_density.verify --reference <sparse>/output/density --b8 <B8> --workers 1`
  and `--workers 2`: all12 episode semantic SHA-256s IDENTICAL; timing excluded. Three
  real H300 continuation units executed. This qualifies only the tiny CPU paths tested,
  not cross-platform identity, production throughput or the L4 canary.
- Stage CPU sums and cell total-CPU denominator sums reconcile. Forecast uses all
  continuation decisions: max observed .00368706 CPU-s/decision; worst case1,638,000
  decisions. Whole-session2x/3x/5x slowdown forecasts52.166/70.749/107.915min plus the
  registered reserve. L4 canary must establish operational suitability before production.

Both manifests are VALIDATED, NOT DISPATCHED. Production remains runtime-blocked without
real recovered L4 proof and both sidecars. DISPATCH.txt records exact operator commands
and full limitations. Its normal-upload figure65.409s is a conservative upward display;
the authoritative exact value is65.408409s in both immutable upload receipts.

The amendment records two invalid unexported pre-freeze one-decision engineering smokes;
those probes are excluded from every endpoint, reference, forecast and qualification.
C-P1's length-specific8/9/11-row construction and explicit8349 engineering-floor rounding
are prospective in amendment03. All prior metric contract/amendment bytes remain unchanged.
