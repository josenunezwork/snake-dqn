# Local verification — no dispatch

Prepared source pin: `70dfc1ec0ec8fa1749c04c8648b3d105f4fb3ac9`.

The targeted package/shared supervisor run passed **171 tests in 408.60s**:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q research/bl_served_search/test_native_manifests.py research/bl_served_search/test_native_ruler.py research/bl_served_search/test_native_l4.py research/bl_served_search/test_fixture_bundle.py research/bl_served_search/test_placement_stop.py research/bl_served_search/test_exact_stop.py tests/test_colab_cli_canary.py tests/test_colab_cli_schema_failures.py tests/test_colab_cli_canary_observer.py tests/test_colab_cli_source.py tests/test_colab_cli_supervisor.py
```

After adding the explicit sparse identity-file closure, this targeted rerun
passed **7 tests in 1.11s** (overlaps the first run):

```sh
SNAKE_DQN_DEVICE=cpu /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q research/bl_served_search/test_native_manifests.py
```

Independent final changed-boundary review: **30 passed in 1.35s**, no remaining
findings. It checked the actual manifest exports, matching five-input
inventories, all 25 native source-identity paths, immutable scanner lifecycle,
caps, source compatibility, transport hashes and the exact DISPATCH sequence.
The AST test confirms unchanged historical computational functions; mutated
hashes and missing identity files fail. No measured evidence was rewritten.

Black100 --check, isort --profile black --check-only, and flake8 pass for all
10 changed Python files: fixture_bundle, native_l4, native_manifests,
native_provenance, native_ruler, placement_stop, test_fixture_bundle,
test_native_l4, test_native_manifests and test_placement_stop.
`bash -n research/bl_served_search/run_native_l4.sh` passed. CLI help smoke
passed for native_manifests, native_l4 and fixture_bundle. `git diff --check`
passed.

The real `python -m research.colab_cli.supervisor ... validate canary.json`
entrypoint passed locally. Main passed shared schema, source/inputs/envelope
checks and reached the exact final missing-real-canary receipt refusal.
`native_manifests bind` will create the full admitted main manifest only after
real canary success/release. No mock or synthetic receipt qualified admission.

`native_manifests prepare` exercised pinned source preparation and real
supervisor transfer code through the local ContentsClient emulator for both
manifests. Normal/fallback upload forecasts: 89.034223s/119.689952s;
fallback including fetch bound: 239.689952s. Both sets of actual local staged
bytes verified; five-input inventories match. The supplemental source-plan
check verifies every native identity digest against its selected Git blob.
`local-validation.json` records that proof and the Mac receipt digest.

The requested output receipt is absent pending real computation. There were
zero Colab calls, no native timing/gameplay/selection-bank run, no tmux or
lease changes, and no modifications to released checkpoints, serving pins,
protected T34 packages or historical hash-bound receipts. The prohibited
`t34-round0-study-01` directory was never opened. Earlier offline family01 was
preserved and marked superseded; only family02 is handed off.
