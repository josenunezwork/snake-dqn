# A1-PROF prepared handoff — 2026-10-23

Prepared and published; **no Colab allocation or dispatch performed**.
Source pin: `557ad8a66677acd14bf78b0acc087020fa42a7c4` on origin/main.
Operational checkout: `/Users/josenunez/Projects/ml/snake-dqn-a1-prof-run`, detached at that pin.
Implementation/review branch: `t34-a1-prof-20261023` in sibling `snake-dqn-a1-prof`.
The separate operational checkout keeps commands usable after this review-only commit.

Protocol: [A1_PROF_PROTOCOL.md](A1_PROF_PROTOCOL.md).
Independent light review: [CODEX_V_A1_PROF_20261023.json](CODEX_V_A1_PROF_20261023.json).
Artifact root: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a1-prof-a100-20261023`.

## Prepared artifacts and qualification

- Canary manifest: `a1-prof-canary.json`, SHA256 `1be0fea64616d124b15f61e1192407ddc3533d82999089cca4382351a5f90aa9`.
- Main template: `a1-prof-main-template.json`, SHA256 `4b438eada5decfb03ee601996edf0730849a4e581306edc2d4119d8c560654ad`.
  A template is not a dispatchable main manifest. The binding command below creates `a1-prof-main.json` only after a real same-pin canary PASS and verified release.
- Create-only `DISPATCH.txt`, SHA256 `bd1815bfc0df07f1a47729d483b6e9609ac294ee8b79c0caa24fd8d1a2f1877a`.
- Mac preflight wrapper: `mac-preflight/mac-preflight.json`, SHA256 `cd088e468bac5fb076a2886d01c249ea7304ce9f127bf053f30ec589eb2a9800`.
- Runtime receipt: `mac-preflight/runtime/output/a1-prof.json`, SHA256 `f3a4651d2d28010933f8785cf7ee3ad896c24ffb891444a29be8611c3a127895`.
- Smoke union analysis: `mac-preflight-analysis.json`, SHA256 `351101b9d6cad163daed94b5aa6a410bbeb247a64ff9c09f02b6cd7c85f2db9a`.

`a1_prof_pack --validate` passed through the qualified supervisor, including full published source closure, current transport registry, family-canary declarations, immutable empty ledgers, exact compact public checkpoint catalog and 64 MiB export bound. Inputs comprise seven uploaded catalog/ledger files; public checkpoint bytes are fetched and SHA-verified by the existing qualified input path. Runtime inputs are supervisor-relative; no Mac filesystem lookup or Drive is needed on the VM. Qualified transport and core engine files are unchanged.

## Measured Mac preflight

The real `colab.sh` → `a1_prof.sh` entrypoint completed in **19.955589165969286 s**, CPU E1, three rosters, staged copies of all six pinned checkpoint inputs. Every cell ran one warmup, three uninstrumented ticks, three timed ticks and one profiler tick; progress, payload sidecars and final receipt were verified. Baseline means: scripted 0.206876889 s/tick, frozen 0.379766820, mixed 0.326082528. These are portability/workload measurements, not A100 speed forecasts or gate evidence. The Mac allocator warning says allocations made before the profiler cannot have matching deallocation events; net stage memory is therefore descriptive. Device peak memory and the CUDA canary remain required.

Command actually run from the implementation checkout at the source pin:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-a1-prof
./venv/bin/python -m research.gpu_sim_spike.a1_prof_preflight \
  --commit 557ad8a66677acd14bf78b0acc087020fa42a7c4 \
  --root /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a1-prof-a100-20261023/mac-preflight
```

For a repeat, use the detached operational checkout and a **fresh** output root; the original preflight is immutable:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-a1-prof-run
./venv/bin/python -m research.gpu_sim_spike.a1_prof_preflight \
  --commit 557ad8a66677acd14bf78b0acc087020fa42a7c4 \
  --root /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a1-prof-a100-20261023/mac-preflight-repeat-01
```

## Exact operator sequence

These commands were prepared, **not executed**. Run the first dispatch, then wait for the real canary PASS with verified exports and backend disappearance before proceeding to the remaining commands. The main binding command independently enforces these conditions. Dispatch refreshes account-wide assignment inventory and calibrated rates under its allocation lock; an unknown endpoint or unpriced A100 substitution refuses.

```bash
REPO=/Users/josenunez/Projects/ml/snake-dqn-a1-prof-run
ART=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a1-prof-a100-20261023
PIN=557ad8a66677acd14bf78b0acc087020fa42a7c4
cd "$REPO"
"$REPO/tools/ops/dispatch.sh" a1-prof-a100-canary "$ART/a1-prof-canary.json" \
  --broker "$REPO" --cwd "$REPO" --state-dir "$ART/state-canary" \
  --dispatch-txt "$ART/DISPATCH.txt"
```

After verified canary success and release:

```bash
"$REPO/venv/bin/python" -m research.gpu_sim_spike.a1_prof_pack \
  --commit "$PIN" --artifact-dir "$ART" \
  --canary-receipt "$ART/canary-output/receipt.json" --validate
"$REPO/tools/ops/dispatch.sh" a1-prof-a100-main "$ART/a1-prof-main.json" \
  --broker "$REPO" --cwd "$REPO" --state-dir "$ART/state-main" \
  --dispatch-txt "$ART/DISPATCH.txt"
```

After main verified export and release, take the Mac-only descriptive look:

```bash
"$REPO/venv/bin/python" -m research.gpu_sim_spike.a1_prof_analyze \
  --commit "$PIN" --exports "$ART/main-output/exports" \
  --output "$ART/mac-profile-analysis.json"
```

Expected total: **40 min, 5.13–6.67 u** (main 30 min, canary 10 min, including setup/export). Planning rate 7.7–10 u/h; admission ceiling 10 u/h is not a claim of current price. Effective doubled caps: main 90 min / 20 u + 4 u reserve, workload 2,400 s; canary 30 min / 5 u + 1 u reserve, workload 900 s. Combined reservation 30 u. First applicable cap wins; no infinite idle/retry loop. The original 0.512 s E1024 reference remains a descriptive column. Pure feature and v8 action work overlap structurally, so report both broad bundle-subtracted remainder and conservative remainder retaining the opponent v8 bundle.

## Verification evidence

All test commands used CPU and one numerical-library thread per process. Changed Python passed Black (100 columns), isort (Black profile), flake8; both Bash entrypoints passed `bash -n`; `git diff --check` passed.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu ./venv/bin/python -m pytest -q \
  research/gpu_sim_spike/test_a1_prof.py research/gpu_sim_spike/test_a1_prof_units.py \
  research/gpu_sim_spike/test_a1_prof_pack.py tests/test_colab_cli_canary.py \
  tests/test_colab_cli_schema_failures.py tests/test_ops_dispatch.py
```

Result: **178 passed, 1 CUDA-only skip in 43.97 s**, before two final narrow regression additions. Latest new-package rerun of the first three modules: **18 passed, 1 CUDA-only skip in 0.14 s**. Independent source/export selection: **20 passed in 21.32 s**. A broad supervisor integration selection was deliberately interrupted after ten passing tests to retain light scope; it is not represented as a full-suite pass. Exact reviewer commands and limitations are in its receipt.

Preparation/analysis commands executed from the operational checkout:

```bash
./venv/bin/python -m research.gpu_sim_spike.a1_prof_pack \
  --commit 557ad8a66677acd14bf78b0acc087020fa42a7c4 \
  --artifact-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a1-prof-a100-20261023 --validate
./venv/bin/python -m research.gpu_sim_spike.a1_prof_analyze --smoke \
  --commit 557ad8a66677acd14bf78b0acc087020fa42a7c4 \
  --exports /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a1-prof-a100-20261023/mac-preflight/runtime/output \
  --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a1-prof-a100-20261023/mac-preflight-analysis.json
```

Earlier development-only uncommitted CPU probes at `/tmp/a1-prof-preflight.DYUkap` and `/tmp/a1-prof-preflight.a6hJFH` are preserved and excluded from admission. Only the published-pin preflight above is bound into the manifests.

No Colab, RunPod, long local job or tmux session was launched/stopped by this implementation. Existing `git-sync`, `sol-a100prof`, `watchd` sessions were observed only. Existing primary `exp/mac-lease.json.live` and `.lock` were preserved. No core simulator/policy/features, released checkpoint, serving pin, legacy queue, qualified transport or historical hash-bound evidence was changed. Only family routing gained the new opt-in entrypoint. CUDA profiling, native selfchecks, production-shape memory and actual provider debit remain unmeasured until the operator runs the new canary/main.
